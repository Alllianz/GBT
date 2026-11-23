use crate::backtest::equity_point::EquityPoint;
use crate::backtest::portfolio::Portfolio;

/// Snapshot resultante de la evaluación de equity al cierre de una vela
#[derive(Debug, Clone, PartialEq)]
pub struct CloseEquitySnapshot {
    pub equity: f64,
    pub peak_equity: f64,
    pub drawdown_nominal: f64,
    pub drawdown_pct: f64,
    pub is_bankrupt: bool,
    pub point: EquityPoint,
}

/// Evaluador de Equity, Drawdown y Bancarrota al Cierre de Vela
///
/// Implementa la política institucional de Allianz:
/// "Elimina los equity intra, ni drawdown ni profit, solo cuando cierre la vela."
///
/// Toda la valoración de equity, tracking de picos (`peak_equity`), medición de drawdown
/// y control de insolvencia se efectúan única y exclusivamente al CIERRE de cada vela (Close),
/// descartando completamente el ruido y las distorsiones de las mechas intra-vela (High/Low).
pub struct CloseEquityEvaluator;

impl CloseEquityEvaluator {
    /// Evalúa la equity, el peak, el drawdown y la condición de bancarrota exclusivamente
    /// al CIERRE de la vela (t <= T). Sin ninguna estimación intra-vela (mechas High/Low).
    ///
    /// - `equity`: `cash_balance + PnL no realizado valorado al Close` (si hay posición activa; si no, `cash_balance`).
    /// - `peak_equity`: actualizado con `current_equity.max(current_peak)`.
    /// - `drawdown_nominal`: `(updated_peak - current_equity).max(0.0)`.
    /// - `drawdown_pct`: `(drawdown_nominal / updated_peak) * 100.0`.
    /// - `is_bankrupt`: verificado únicamente contra el equity al Close (evita falsas quiebras por mechas, C-01).
    #[inline]
    pub fn evaluate_candle_close(
        portfolio: &Portfolio,
        close_price: f64,
        timestamp: i64,
        eval_idx: usize,
        current_peak: f64,
        initial_capital: f64,
    ) -> CloseEquitySnapshot {
        let current_eq = portfolio.current_equity(close_price);
        let updated_peak = if current_eq > current_peak {
            current_eq
        } else {
            current_peak
        };

        let dd_nominal = (updated_peak - current_eq).max(0.0);
        let dd_pct = if updated_peak > 0.0 {
            (dd_nominal / updated_peak) * 100.0
        } else {
            0.0
        };

        // Umbral institucional de insolvencia operativa (0.1% del capital inicial o mínimo $5.00 nocional de Binance Futures)
        let insolvency_threshold = (initial_capital * 0.001).max(5.0);
        let is_bankrupt = current_eq <= insolvency_threshold || portfolio.is_bankrupt(close_price);

        let point = EquityPoint::new(
            timestamp,
            eval_idx,
            portfolio.cash_balance,
            current_eq,
            dd_nominal,
            dd_pct,
        );

        CloseEquitySnapshot {
            equity: current_eq,
            peak_equity: updated_peak,
            drawdown_nominal: dd_nominal,
            drawdown_pct: dd_pct,
            is_bankrupt,
            point,
        }
    }

    /// Actualiza de forma coherente el último punto de la curva de equity tras el cierre forzado
    /// de posiciones al fin del horizonte (C-22), garantizando consistencia matemática con el peak previo.
    pub fn update_final_horizon_point(
        last_point: &mut EquityPoint,
        peak_before_final_bar: f64,
        final_balance: f64,
    ) -> f64 {
        let updated_peak = peak_before_final_bar.max(final_balance);
        let dd_nominal = (updated_peak - final_balance).max(0.0);
        let dd_pct = if updated_peak > 0.0 {
            (dd_nominal / updated_peak) * 100.0
        } else {
            0.0
        };

        last_point.balance = final_balance;
        last_point.equity = final_balance;
        last_point.drawdown_nominal = dd_nominal;
        last_point.drawdown_pct = dd_pct;

        updated_peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Position, PositionSide};

    #[test]
    fn test_evaluate_candle_close_without_position() {
        let portfolio = Portfolio::new(10_000.0);
        let snapshot = CloseEquityEvaluator::evaluate_candle_close(
            &portfolio,
            100.0,
            1000,
            0,
            10_000.0,
            10_000.0,
        );

        assert_eq!(snapshot.equity, 10_000.0);
        assert_eq!(snapshot.peak_equity, 10_000.0);
        assert_eq!(snapshot.drawdown_nominal, 0.0);
        assert_eq!(snapshot.drawdown_pct, 0.0);
        assert!(!snapshot.is_bankrupt);
        assert_eq!(snapshot.point.equity, 10_000.0);
    }

    #[test]
    fn test_evaluate_candle_close_with_unrealized_pnl() {
        let mut portfolio = Portfolio::new(10_000.0);
        portfolio.active_position = Some(Position::new(
            PositionSide::Long,
            100.0,
            10.0,
            1000,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        ));

        // Close a 105.0 -> PnL no realizado = (105 - 100) * 10 = +$50 -> Equity = 10,050
        let snapshot = CloseEquityEvaluator::evaluate_candle_close(
            &portfolio,
            105.0,
            2000,
            1,
            10_000.0,
            10_000.0,
        );

        assert_eq!(snapshot.equity, 10_050.0);
        assert_eq!(snapshot.peak_equity, 10_050.0);
        assert_eq!(snapshot.drawdown_nominal, 0.0);
        assert_eq!(snapshot.drawdown_pct, 0.0);

        // Siguiente vela: Close a 95.0 -> PnL no realizado = (95 - 100) * 10 = -$50 -> Equity = 9,950
        let snapshot_loss = CloseEquityEvaluator::evaluate_candle_close(
            &portfolio,
            95.0,
            3000,
            2,
            10_050.0,
            10_000.0,
        );

        assert_eq!(snapshot_loss.equity, 9_950.0);
        assert_eq!(snapshot_loss.peak_equity, 10_050.0);
        assert_eq!(snapshot_loss.drawdown_nominal, 100.0);
        assert!((snapshot_loss.drawdown_pct - (100.0 / 10_050.0 * 100.0)).abs() < 1e-6);
    }

    #[test]
    fn test_insolvency_evaluated_only_at_close() {
        let mut portfolio = Portfolio::new(10_000.0);
        // Posición apalancada con gran caída de precio al close
        portfolio.active_position = Some(Position::new(
            PositionSide::Long,
            100.0,
            100.0,
            1000,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        ));

        // Precio al close cae a 0.0 -> PnL no realizado = -10,000 -> Equity = 0
        let snapshot = CloseEquityEvaluator::evaluate_candle_close(
            &portfolio,
            0.0,
            2000,
            1,
            10_000.0,
            10_000.0,
        );

        assert!(snapshot.is_bankrupt);
    }

    #[test]
    fn test_update_final_horizon_point_coherence() {
        let mut point = EquityPoint::new(1000, 5, 10_000.0, 10_200.0, 0.0, 0.0);
        // Supongamos que tras la liquidación forzada con comisión el saldo final es 10_190
        // y el peak previo era 10_000
        let updated_peak = CloseEquityEvaluator::update_final_horizon_point(
            &mut point,
            10_000.0,
            10_190.0,
        );

        assert_eq!(updated_peak, 10_190.0);
        assert_eq!(point.balance, 10_190.0);
        assert_eq!(point.equity, 10_190.0);
        assert_eq!(point.drawdown_nominal, 0.0);
        assert_eq!(point.drawdown_pct, 0.0);
    }
}

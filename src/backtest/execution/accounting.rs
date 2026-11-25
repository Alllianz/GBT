use crate::backtest::config::BacktestConfig;
use crate::backtest::portfolio::Portfolio;
use crate::types::{CompletedTrade, Position, PositionSide};

pub struct TradeAccounting;

impl TradeAccounting {
    /// Cierra contablemente una posición, computa el PnL bruto, deduce comisiones,
    /// actualiza el saldo del portafolio y genera el `CompletedTrade` definitivo.
    pub fn close_position(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        pos: Position,
        target_price: f64,
        exit_timestamp: i64,
        candle_idx: usize,
        reason: String,
    ) -> CompletedTrade {
        // Directriz institucional de Allianz: GBT no utiliza slippage.
        // El precio de ejecución de salida es exactamente el precio objetivo (Open o nivel de SL/TP/liquidación).
        let exit_price = target_price;

        let gross_pnl = match pos.side {
            PositionSide::Long => (exit_price - pos.entry_price) * pos.quantity,
            PositionSide::Short => (pos.entry_price - exit_price) * pos.quantity,
        };

        let exit_notional = exit_price * pos.quantity;
        let exit_exec_type = if let Some(ref ms) = config.microstructure {
            ms.execution_type_for_exit_reason(&reason)
        } else {
            crate::microstructure::OrderExecutionType::Taker
        };
        let exit_commission_rate = config.get_commission_rate(exit_exec_type);
        let exit_commission = exit_notional * exit_commission_rate;
        let total_commissions = pos.entry_commission + exit_commission;

        // C-34: Se eliminó la creación y descarte inútil (_exit_order, _exit_fill) sin tracking.

        // BUG-38: Deducir comisiones completas (entrada + salida) y funding devengado del PnL neto
        let total_trading_costs = total_commissions + pos.accumulated_funding;
        let uncapped_pnl = gross_pnl - total_trading_costs;
        let is_liquidation = crate::microstructure::ExitReason::from_reason_str_lenient(&reason).is_liquidation();

        let net_pnl = if let Some(isolated_margin) = pos.isolated_margin {
            if is_liquidation || uncapped_pnl < -isolated_margin {
                -isolated_margin
            } else {
                uncapped_pnl
            }
        } else {
            uncapped_pnl
        };

        // C-03: Actualización del balance según el modo de margen
        let balance_impact = if let Some(isolated_margin) = pos.isolated_margin {
            // Modo Isolated: La comisión de entrada salió del margen aislado asignado (no se descontó
            // de cash_balance libre al abrir). Por tanto, la variación neta de la cuenta es exactamente net_pnl:
            // - Si hubo liquidación (is_liquidation o uncapped_pnl < -isolated_margin), la pérdida total es exactamente -isolated_margin.
            // - En salida normal, las comisiones de salida y entrada salen del resultado de la posición.
            // En ambos casos, net_pnl y la variación de cash_balance cuadran al centavo.
            net_pnl.max(-isolated_margin.min(portfolio.cash_balance))
        } else {
            // Modo Cross: La comisión de entrada ya se descontó de portfolio.cash_balance en process_signal_at_open.
            // Al cerrar la posición solo se impacta el balance con gross_pnl - exit_commission.
            gross_pnl - exit_commission
        };

        portfolio.cash_balance += balance_impact;
        if portfolio.cash_balance < 0.0 {
            // Protección de insolvencia
            portfolio.cash_balance = 0.0;
        }
        portfolio.total_commissions_paid += exit_commission;

        let entry_notional = pos.notional_value();
        let return_pct = if entry_notional > 0.0 {
            (net_pnl / entry_notional) * 100.0
        } else {
            0.0
        };

        let duration_bars = candle_idx.saturating_sub(pos.entry_candle_idx);
        let duration_millis = exit_timestamp.saturating_sub(pos.entry_timestamp);

        let (mfe, mae) = match pos.side {
            PositionSide::Long => (
                (pos.max_favorable_price - pos.entry_price) / pos.entry_price * 100.0,
                (pos.entry_price - pos.max_adverse_price) / pos.entry_price * 100.0,
            ),
            PositionSide::Short => (
                (pos.entry_price - pos.max_favorable_price) / pos.entry_price * 100.0,
                (pos.max_adverse_price - pos.entry_price) / pos.entry_price * 100.0,
            ),
        };

        CompletedTrade {
            side: pos.side,
            entry_timestamp: pos.entry_timestamp,
            exit_timestamp,
            entry_price: pos.entry_price,
            exit_price,
            quantity: pos.quantity,
            gross_pnl,
            commissions: total_commissions,
            funding_paid: pos.accumulated_funding,
            net_pnl,
            return_pct,
            duration_bars,
            duration_millis,
            max_favorable_excursion: mfe,
            max_adverse_excursion: mae,
            exit_reason: reason,
            entry_candle_idx: pos.entry_candle_idx,
            exit_candle_idx: candle_idx,
            regime: pos.regime,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c03_isolated_liquidation_accounting_loss_equals_margin() {
        // Test numérico requerido por Allianz:
        // Cuenta de 1000, margen aislado de 100, comisión de entrada de 0.5 y liquidación;
        // el saldo final debe ser exactamente 900, y net_pnl debe ser exactamente -100.
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            commission_rate: 0.0005,
            ..Default::default()
        };

        let pos = Position::new(
            PositionSide::Long,
            100.0,
            10.0, // notional = 1000.0
            1000,
            0,
            None,
            None,
            Some(100.0), // Margen aislado asignado
            90.50,
            0.50, // Comisión de entrada
        );

        // En Isolated, la comisión no se dedujo de cash_balance libre al abrir
        assert_eq!(portfolio.cash_balance, 1000.0);

        // Se produce liquidación con caída a 80.0 (pérdida bruta -200)
        let trade = TradeAccounting::close_position(
            &mut portfolio,
            &config,
            pos,
            80.0,
            2000,
            1,
            "Liquidated (Margin Call)".to_string(),
        );

        assert_eq!(trade.net_pnl, -100.0, "net_pnl debe ser exactamente -100.0");
        assert_eq!(portfolio.cash_balance, 900.0, "Saldo final debe ser exactamente 900.0");
    }

    #[test]
    fn test_c03_isolated_normal_exit_accounting() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            commission_rate: 0.001,
            ..Default::default()
        };

        let pos = Position::new(
            PositionSide::Long,
            100.0,
            1.0, // notional = 100.0
            1000,
            0,
            None,
            None,
            Some(100.0),
            0.0,
            0.10, // Comisión de entrada
        );

        // Salida con beneficio a 110.0 (+10.0 bruto, exit_comm = 0.11)
        let trade = TradeAccounting::close_position(
            &mut portfolio,
            &config,
            pos,
            110.0,
            2000,
            1,
            "Take Profit".to_string(),
        );

        // net_pnl = gross_pnl - total_commissions
        let expected_net_pnl = trade.gross_pnl - trade.commissions;
        assert_eq!(trade.gross_pnl, 10.0);
        assert!((trade.net_pnl - expected_net_pnl).abs() < 1e-6);
        // Cuadre al centavo de la variación de cash_balance y net_pnl
        assert!(((portfolio.cash_balance - 1000.0) - trade.net_pnl).abs() < 1e-6);
    }
}

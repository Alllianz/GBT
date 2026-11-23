use super::engine::BacktestResult;

/// Calculador institucional del Ratio de Robustez Walk-Forward (C-24).
///
/// Resuelve el sesgo temporal donde la disparidad de duración entre la ventana
/// In-Sample (usualmente 70-80% del tiempo) y la ventana Out-of-Sample (20-30%)
/// distorsionaba el ratio OOS / IS.
///
/// Normaliza los rendimientos acumulados por la duración temporal (o por número
/// de barras evaluadas) para calcular la tasa de retorno relativa por unidad
/// de tiempo antes de computar el ratio de retención de rendimiento.
pub struct RobustnessCalculator;

impl RobustnessCalculator {
    /// Calcula el Ratio de Robustez normalizado temporalmente entre IS y OOS.
    ///
    /// Fórmula:
    ///   `robustness = (oos_rate / is_rate).max(0.0)`
    /// donde `rate = net_profit_pct / duration`.
    pub fn calculate(is_result: &BacktestResult, oos_result: &BacktestResult) -> f64 {
        let is_ret = is_result.metrics.net_profit_pct;
        let oos_ret = oos_result.metrics.net_profit_pct;

        // Si IS no generó beneficio neto positivo mensurable, la robustez no está definida o es 0
        if !is_ret.is_finite() || is_ret <= 1e-6 {
            return 0.0;
        }

        // 1. Intentar normalización por duración temporal exacta (milisegundos) de la curva de equity
        let is_duration_ms = Self::extract_total_duration_ms(&is_result.equity_curve);
        let oos_duration_ms = Self::extract_total_duration_ms(&oos_result.equity_curve);

        if let (Some(is_dt), Some(oos_dt)) = (is_duration_ms, oos_duration_ms) {
            if is_dt > 0.0 && oos_dt > 0.0 {
                let is_rate = is_ret / is_dt;
                let oos_rate = oos_ret / oos_dt;
                if is_rate > 1e-15 {
                    let ratio = (oos_rate / is_rate).max(0.0);
                    return if ratio.is_finite() { ratio } else { 0.0 };
                }
            }
        }

        // 2. Normalización por número de barras evaluadas en la curva de equity
        let is_bars = is_result.equity_curve.len() as f64;
        let oos_bars = oos_result.equity_curve.len() as f64;

        if is_bars > 0.0 && oos_bars > 0.0 {
            let is_rate = is_ret / is_bars;
            let oos_rate = oos_ret / oos_bars;
            if is_rate > 1e-15 {
                let ratio = (oos_rate / is_rate).max(0.0);
                return if ratio.is_finite() { ratio } else { 0.0 };
            }
        }

        // 3. Fallback directo si no hay curva de equity (ej. resultados sintéticos mínimos)
        let ratio = (oos_ret / is_ret).max(0.0);
        if ratio.is_finite() {
            ratio
        } else {
            0.0
        }
    }

    fn extract_total_duration_ms(curve: &[super::equity_point::EquityPoint]) -> Option<f64> {
        let n = curve.len();
        if n >= 2 {
            let first_ts = curve.first().map(|p| p.timestamp)?;
            let last_ts = curve.last().map(|p| p.timestamp)?;
            let dt = (last_ts - first_ts).abs() as f64;
            let n_intervals = (n - 1) as f64;
            if dt > 0.0 && n_intervals > 0.0 {
                let bar_dt = dt / n_intervals;
                Some(dt + bar_dt)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::metrics::BacktestMetrics;
    use crate::backtest::EquityPoint;

    fn mock_backtest_result(
        net_profit_pct: f64,
        start_ts: i64,
        num_bars: usize,
        bar_interval_ms: i64,
    ) -> BacktestResult {
        let mut equity_curve = Vec::with_capacity(num_bars);
        for i in 0..num_bars {
            equity_curve.push(EquityPoint::new(
                start_ts + (i as i64 * bar_interval_ms),
                i,
                1000.0,
                1000.0 * (1.0 + net_profit_pct / 100.0),
                0.0,
                0.0,
            ));
        }

        let metrics = BacktestMetrics {
            initial_capital: 1000.0,
            final_capital: 1000.0 * (1.0 + net_profit_pct / 100.0),
            net_profit: 1000.0 * (net_profit_pct / 100.0),
            net_profit_pct,
            total_trades: 10,
            winning_trades: 5,
            losing_trades: 5,
            win_rate_pct: 50.0,
            gross_profit: 100.0,
            gross_loss: 50.0,
            profit_factor: 2.0,
            avg_trade_pnl: 10.0,
            avg_win_pnl: 20.0,
            avg_loss_pnl: 10.0,
            payoff_ratio: 2.0,
            max_drawdown_nominal: 50.0,
            max_drawdown_pct: 5.0,
            recovery_factor: 2.0,
            total_commissions: 5.0,
            total_funding: 0.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            curve_r2: 0.95,
            capital_to_maxdd_ratio: 20.0,
            composite_score: 1.5,
            avg_duration_bars: 5.0,
            max_consecutive_losses: 2,
        };

        BacktestResult {
            metrics,
            trades: Vec::new(),
            equity_curve,
        }
    }

    #[test]
    fn test_c24_robustness_temporal_normalization_equal_rate() {
        // Escenario institucional del reporte:
        // IS = 700 días (+70% de retorno)
        // OOS = 300 días (+30% de retorno)
        // Misma tasa diaria de retorno: +0.1% / día.
        // Sin normalizar: 30 / 70 = 0.428 (sesgo temporal grave).
        // Con normalización temporal: ratio debe ser exactamente 1.0!
        let day_ms = 86_400_000_i64;
        let is_res = mock_backtest_result(70.0, 1_000_000, 700, day_ms);
        let oos_res = mock_backtest_result(30.0, 1_000_000 + 700 * day_ms, 300, day_ms);

        let ratio = RobustnessCalculator::calculate(&is_res, &oos_res);
        assert!(
            (ratio - 1.0).abs() < 1e-4,
            "Robustness ratio esperado ~1.0 pero se obtuvo {}",
            ratio
        );
    }

    #[test]
    fn test_c24_robustness_outperformance_and_loss() {
        let day_ms = 86_400_000_i64;
        // IS: +50% en 500 días
        let is_res = mock_backtest_result(50.0, 1_000_000, 500, day_ms);
        // OOS A: +25% en 100 días (tasa superior: 0.25%/día vs 0.10%/día => 2.5x)
        let oos_superior = mock_backtest_result(25.0, 1_000_000 + 500 * day_ms, 100, day_ms);
        let ratio_sup = RobustnessCalculator::calculate(&is_res, &oos_superior);
        assert!((ratio_sup - 2.5).abs() < 1e-4);

        // OOS B: Pérdida en OOS (-10%) => ratio debe acotarse en 0.0
        let oos_perdedor = mock_backtest_result(-10.0, 1_000_000 + 500 * day_ms, 100, day_ms);
        let ratio_loss = RobustnessCalculator::calculate(&is_res, &oos_perdedor);
        assert_eq!(ratio_loss, 0.0);
    }
}

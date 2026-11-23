use crate::types::CompletedTrade;
use super::composite_score::CompositeScoreCalculator;
use super::equity_point::EquityPoint;
use super::metrics::BacktestMetrics;
use super::sharpe::SharpeSortinoCalculator;

pub struct MetricsCalculator;

impl MetricsCalculator {
    pub fn calculate(
        initial_capital: f64,
        trades: &[CompletedTrade],
        equity_curve: &[EquityPoint],
    ) -> BacktestMetrics {
        let final_capital = equity_curve
            .last()
            .map(|p| p.equity)
            .unwrap_or(initial_capital);

        let net_profit = final_capital - initial_capital;
        let net_profit_pct = if initial_capital > 0.0 {
            (net_profit / initial_capital) * 100.0
        } else {
            0.0
        };

        let total_trades = trades.len();
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;
        let mut net_winning_pnl = 0.0;
        let mut net_losing_pnl = 0.0;
        let mut total_commissions = 0.0;
        let mut total_funding = 0.0;
        let mut total_duration_bars = 0;

        let mut current_consecutive_losses = 0;
        let mut max_consecutive_losses = 0;

        for t in trades {
            total_commissions += t.commissions;
            total_funding += t.funding_paid;
            total_duration_bars += t.duration_bars;

            // C-25: Acumulación de beneficio y pérdida verdaderamente brutos a partir de `gross_pnl` (pre-costes)
            if t.gross_pnl > 0.0 {
                gross_profit += t.gross_pnl;
            } else if t.gross_pnl < 0.0 {
                gross_loss += t.gross_pnl.abs();
            }

            // Conteo y acumulación de resultados netos (post-comisiones y post-funding)
            if t.net_pnl > 0.0 {
                winning_trades += 1;
                net_winning_pnl += t.net_pnl;
                current_consecutive_losses = 0;
            } else if t.net_pnl < 0.0 {
                losing_trades += 1;
                net_losing_pnl += t.net_pnl.abs();
                current_consecutive_losses += 1;
                if current_consecutive_losses > max_consecutive_losses {
                    max_consecutive_losses = current_consecutive_losses;
                }
            } else {
                current_consecutive_losses = 0;
            }
        }

        let win_rate_pct = if total_trades > 0 {
            (winning_trades as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        // C-25: Profit Factor bruto estándar institucional (`gross_profit / gross_loss`)
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            99.99
        } else {
            0.0
        };

        let avg_trade_pnl = if total_trades > 0 {
            net_profit / total_trades as f64
        } else {
            0.0
        };

        let avg_win_pnl = if winning_trades > 0 {
            net_winning_pnl / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss_pnl = if losing_trades > 0 {
            net_losing_pnl / losing_trades as f64
        } else {
            0.0
        };

        let payoff_ratio = if avg_loss_pnl > 0.0 {
            avg_win_pnl / avg_loss_pnl
        } else if avg_win_pnl > 0.0 {
            99.99
        } else {
            0.0
        };

        let avg_duration_bars = if total_trades > 0 {
            total_duration_bars as f64 / total_trades as f64
        } else {
            0.0
        };

        // Drawdowns sobre la curva de equity
        let mut max_dd_nominal = 0.0;
        let mut max_dd_pct = 0.0;

        for p in equity_curve {
            if p.drawdown_nominal > max_dd_nominal {
                max_dd_nominal = p.drawdown_nominal;
            }
            if p.drawdown_pct > max_dd_pct {
                max_dd_pct = p.drawdown_pct;
            }
        }

        let recovery_factor = if max_dd_nominal > 0.0 {
            net_profit / max_dd_nominal
        } else if net_profit > 0.0 {
            99.99
        } else {
            0.0
        };

        let (sharpe_ratio, sortino_ratio) = Self::calculate_sharpe_sortino(equity_curve);
        let curve_r2 = Self::calculate_r2(equity_curve);
        let (capital_to_maxdd_ratio, composite_score) = CompositeScoreCalculator::calculate(
            curve_r2,
            initial_capital,
            final_capital,
            max_dd_nominal,
            total_trades,
        );

        BacktestMetrics {
            initial_capital,
            final_capital,
            net_profit,
            net_profit_pct,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate_pct,
            gross_profit,
            gross_loss,
            profit_factor,
            avg_trade_pnl,
            avg_win_pnl,
            avg_loss_pnl,
            payoff_ratio,
            max_drawdown_nominal: max_dd_nominal,
            max_drawdown_pct: max_dd_pct,
            recovery_factor,
            total_commissions,
            total_funding,
            sharpe_ratio,
            sortino_ratio,
            curve_r2,
            capital_to_maxdd_ratio,
            composite_score,
            avg_duration_bars,
            max_consecutive_losses,
        }
    }

    /// Calcula la tupla `(sharpe_ratio, sortino_ratio)` delegando en `SharpeSortinoCalculator`.
    /// Implementa las garantías C-12 (ajuste dimensional Israelsen en pérdidas) y C-39 (curvas cortas/varianza nula).
    pub fn calculate_sharpe_sortino(curve: &[EquityPoint]) -> (f64, f64) {
        SharpeSortinoCalculator::calculate(curve)
    }

    pub fn calculate_r2(curve: &[EquityPoint]) -> f64 {
        let n = curve.len();
        if n < 3 {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for (i, p) in curve.iter().enumerate() {
            sum_x += i as f64;
            sum_y += p.equity;
        }

        let mean_x = sum_x / (n as f64);
        let mean_y = sum_y / (n as f64);

        let mut s_xx = 0.0;
        let mut s_yy = 0.0;
        let mut s_xy = 0.0;

        for (i, p) in curve.iter().enumerate() {
            let dx = (i as f64) - mean_x;
            let dy = p.equity - mean_y;
            s_xx += dx * dx;
            s_yy += dy * dy;
            s_xy += dx * dy;
        }

        if !s_xx.is_finite() || !s_yy.is_finite() || !s_xy.is_finite() || s_xx <= 1e-12 || s_yy <= 1e-12 {
            return 0.0;
        }

        let slope = s_xy / s_xx;
        if slope <= 0.0 {
            return 0.0; // Penalización si la pendiente general no es positiva
        }

        let r2 = (s_xy * s_xy) / (s_xx * s_yy);
        if !r2.is_finite() {
            return 0.0;
        }
        r2.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PositionSide;

    fn mock_trade(net_pnl: f64) -> CompletedTrade {
        CompletedTrade {
            side: PositionSide::Long,
            entry_timestamp: 1000,
            exit_timestamp: 2000,
            entry_price: 100.0,
            exit_price: 100.0 + net_pnl,
            quantity: 1.0,
            gross_pnl: net_pnl,
            commissions: 0.0,
            funding_paid: 0.0,
            net_pnl,
            return_pct: net_pnl,
            duration_bars: 1,
            duration_millis: 1000,
            max_favorable_excursion: 0.0,
            max_adverse_excursion: 0.0,
            exit_reason: "Test".to_string(),
            entry_candle_idx: 0,
            exit_candle_idx: 1,
        }
    }

    #[test]
    fn test_bug44_breakeven_resets_consecutive_losses() {
        let trades = vec![
            mock_trade(-10.0), // pérdida 1
            mock_trade(-10.0), // pérdida 2
            mock_trade(0.0),   // breakeven -> debe reiniciar racha a 0
            mock_trade(-10.0), // pérdida 1
        ];

        let metrics = MetricsCalculator::calculate(1000.0, &trades, &[]);
        assert_eq!(metrics.max_consecutive_losses, 2);
    }

    #[test]
    fn test_bug10_dynamic_sharpe_annualization_seconds_and_ms() {
        // Curva en segundos (ej: barras horarias = 3600s, mínimo 4 puntos para 3 retornos con significancia)
        let base_s = 1_700_000_000_i64;
        let curve_seconds = vec![
            EquityPoint::new(base_s, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(base_s + 3600, 1, 1010.0, 1010.0, 0.0, 0.0),
            EquityPoint::new(base_s + 7200, 2, 1020.0, 1020.0, 0.0, 0.0),
            EquityPoint::new(base_s + 10800, 3, 1030.0, 1030.0, 0.0, 0.0),
        ];

        // Curva en milisegundos (ej: barras horarias = 3_600_000ms)
        let base_ms = 1_700_000_000_000_i64;
        let curve_millis = vec![
            EquityPoint::new(base_ms, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(base_ms + 3_600_000, 1, 1010.0, 1010.0, 0.0, 0.0),
            EquityPoint::new(base_ms + 7_200_000, 2, 1020.0, 1020.0, 0.0, 0.0),
            EquityPoint::new(base_ms + 10_800_000, 3, 1030.0, 1030.0, 0.0, 0.0),
        ];

        let (sharpe_s, _) = MetricsCalculator::calculate_sharpe_sortino(&curve_seconds);
        let (sharpe_ms, _) = MetricsCalculator::calculate_sharpe_sortino(&curve_millis);

        // Ambas series de 1 hora deben coincidir en su factor de anualización y Sharpe
        assert!((sharpe_s - sharpe_ms).abs() < 1e-4, "Sharpe en segundos ({}) y ms ({}) deben ser consistentes", sharpe_s, sharpe_ms);
        assert!(sharpe_s > 0.0);
    }

    #[test]
    fn test_bug_f06_israelsen_monotonicity_in_losses() {
        let ts = 1_700_000_000_i64;
        // Estrategia A: Pérdida suave con baja volatilidad (baja desviación)
        // 1000 -> 990 -> 980 -> 970 (retornos ~ -1%)
        let curve_a = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 990.0, 990.0, 10.0, 1.0),
            EquityPoint::new(ts + 7200, 2, 980.0, 980.0, 20.0, 2.0),
            EquityPoint::new(ts + 10800, 3, 970.0, 970.0, 30.0, 3.0),
        ];

        // Estrategia B: Misma pérdida final pero volatilidad caótica destructiva
        // 1000 -> 900 -> 1050 -> 970
        let curve_b = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 900.0, 900.0, 100.0, 10.0),
            EquityPoint::new(ts + 7200, 2, 1050.0, 1050.0, 0.0, 0.0),
            EquityPoint::new(ts + 10800, 3, 970.0, 970.0, 80.0, 7.6),
        ];

        let (sharpe_a, sortino_a) = MetricsCalculator::calculate_sharpe_sortino(&curve_a);
        let (sharpe_b, sortino_b) = MetricsCalculator::calculate_sharpe_sortino(&curve_b);

        // Ambas tienen retornos negativos (< 0.0)
        assert!(sharpe_a < 0.0 && sharpe_b < 0.0);
        assert!(sortino_a < 0.0 && sortino_b < 0.0);

        // Bajo formulación de Israelsen, Estrategia A DEBE tener un ratio MENOS negativo (mejor score)
        // que Estrategia B, penalizando estrictamente la volatilidad caótica.
        assert!(
            sharpe_a > sharpe_b,
            "Sharpe A ({}) debe ser superior (menos penalizado) que Sharpe B ({})",
            sharpe_a,
            sharpe_b
        );
        assert!(
            sortino_a > sortino_b,
            "Sortino A ({}) debe ser superior (menos penalizado) que Sortino B ({})",
            sortino_a,
            sortino_b
        );
    }

    #[test]
    fn test_c39_short_curve_in_metrics_calculator() {
        let ts = 1_700_000_000_i64;
        let short_curve = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 1050.0, 1050.0, 0.0, 0.0),
        ];

        let metrics = MetricsCalculator::calculate(1000.0, &[], &short_curve);
        assert_eq!(metrics.sharpe_ratio, 0.0, "Curva corta no debe reportar 99.99");
        assert_eq!(metrics.sortino_ratio, 0.0, "Curva corta no debe reportar 99.99");
    }

    #[test]
    fn test_c38_initial_capital_zero_in_metrics_calculator() {
        let metrics = MetricsCalculator::calculate(0.0, &[], &[]);
        assert_eq!(metrics.composite_score, 0.0);
        assert_eq!(metrics.capital_to_maxdd_ratio, 0.0);
        assert_eq!(metrics.net_profit_pct, 0.0);
    }

    #[test]
    fn test_c37_crypto_365_annualization_in_metrics() {
        let base_s = 1_700_000_000_i64;
        // 4 barras diarias
        let curve_daily = vec![
            EquityPoint::new(base_s, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(base_s + 86400, 1, 1010.0, 1010.0, 0.0, 0.0),
            EquityPoint::new(base_s + 172800, 2, 1025.0, 1025.0, 0.0, 0.0),
            EquityPoint::new(base_s + 259200, 3, 1040.0, 1040.0, 0.0, 0.0),
        ];

        let metrics = MetricsCalculator::calculate(1000.0, &[], &curve_daily);
        assert!(metrics.sharpe_ratio > 0.0);
        assert!(metrics.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_c18_total_funding_aggregated_in_metrics() {
        let mut t1 = mock_trade(50.0);
        t1.commissions = 5.0;
        t1.funding_paid = 2.50;

        let mut t2 = mock_trade(-20.0);
        t2.commissions = 4.0;
        t2.funding_paid = 1.75;

        let trades = vec![t1, t2];
        let curve = vec![EquityPoint::new(1000, 0, 1000.0, 1000.0, 0.0, 0.0)];

        let metrics = MetricsCalculator::calculate(1000.0, &trades, &curve);
        assert!((metrics.total_commissions - 9.0).abs() < 1e-6);
        assert!((metrics.total_funding - 4.25).abs() < 1e-6);
    }

    #[test]
    fn test_c25_gross_vs_net_profit_metrics() {
        // Trade 1: Bruto positivo (+50.0), pero tras comisiones (60.0), neto negativo (-10.0)
        let mut t1 = mock_trade(-10.0);
        t1.gross_pnl = 50.0;
        t1.commissions = 60.0;
        t1.funding_paid = 0.0;
        t1.net_pnl = -10.0;

        // Trade 2: Bruto negativo (-30.0), comisiones 5.0, neto (-35.0)
        let mut t2 = mock_trade(-35.0);
        t2.gross_pnl = -30.0;
        t2.commissions = 5.0;
        t2.funding_paid = 0.0;
        t2.net_pnl = -35.0;

        // Trade 3: Bruto positivo (+150.0), comisiones 10.0, neto (+140.0)
        let mut t3 = mock_trade(140.0);
        t3.gross_pnl = 150.0;
        t3.commissions = 10.0;
        t3.funding_paid = 0.0;
        t3.net_pnl = 140.0;

        let trades = vec![t1, t2, t3];
        let curve = vec![EquityPoint::new(1000, 0, 1000.0, 1095.0, 0.0, 0.0)];

        let metrics = MetricsCalculator::calculate(1000.0, &trades, &curve);

        // C-25: gross_profit debe sumar 50.0 + 150.0 = 200.0 (pre-costes)
        assert!((metrics.gross_profit - 200.0).abs() < 1e-6, "gross_profit debe ser 200.0, obtenido {}", metrics.gross_profit);
        // gross_loss debe sumar 30.0 (pre-costes)
        assert!((metrics.gross_loss - 30.0).abs() < 1e-6, "gross_loss debe ser 30.0, obtenido {}", metrics.gross_loss);
        // profit_factor debe ser 200.0 / 30.0 = 6.6667
        assert!((metrics.profit_factor - (200.0 / 30.0)).abs() < 1e-4);

        // Métricas netas: 1 ganador (t3: +140), 2 perdedores (t1: -10, t2: -35)
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 2);
        assert!((metrics.avg_win_pnl - 140.0).abs() < 1e-6);
        assert!((metrics.avg_loss_pnl - 22.5).abs() < 1e-6); // (10 + 35) / 2 = 22.5
    }
}

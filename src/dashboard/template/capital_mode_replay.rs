use crate::backtest::{BacktestResult, EquityPoint};
use crate::types::CompletedTrade;

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayedSeries {
    pub equity: Vec<f64>,
    pub drawdown_pct: Vec<f64>,
    pub final_capital: f64,
}

pub struct CapitalModeReplayer;

impl CapitalModeReplayer {
    pub fn resolve_initial_capital(result: &BacktestResult, fallback: f64) -> f64 {
        if result.metrics.initial_capital.is_finite() && result.metrics.initial_capital > 0.0 {
            return result.metrics.initial_capital;
        }
        if let Some(first_pos) = result
            .equity_curve
            .iter()
            .map(|p| p.equity)
            .find(|&eq| eq.is_finite() && eq > 0.0)
        {
            return first_pos;
        }
        if fallback.is_finite() && fallback > 0.0 {
            fallback
        } else {
            10_000.0
        }
    }

    pub fn normalize_nominal_segment(
        curve: &[EquityPoint],
        seg_initial: f64,
        base_initial: f64,
        anchor_equity: f64,
        initial_peak: f64,
    ) -> (ReplayedSeries, f64) {
        let n = curve.len();
        if n == 0 {
            let final_cap = anchor_equity.max(0.0);
            return (
                ReplayedSeries {
                    equity: Vec::new(),
                    drawdown_pct: Vec::new(),
                    final_capital: final_cap,
                },
                initial_peak.max(final_cap),
            );
        }

        let mut eq_vec = Vec::with_capacity(n);
        let mut dd_vec = Vec::with_capacity(n);

        if !anchor_equity.is_finite() || anchor_equity <= 0.0 {
            eq_vec.resize(n, 0.0);
            dd_vec.resize(n, 100.0);
            return (
                ReplayedSeries {
                    equity: eq_vec,
                    drawdown_pct: dd_vec,
                    final_capital: 0.0,
                },
                initial_peak.max(0.0),
            );
        }

        let scale = if seg_initial.is_finite() && seg_initial > 0.0 && base_initial.is_finite() && base_initial > 0.0 {
            base_initial / seg_initial
        } else {
            1.0
        };

        let mut peak = initial_peak.max(anchor_equity);
        let mut last_eq = anchor_equity;

        for p in curve {
            let eq = if p.equity.is_finite() {
                (anchor_equity + (p.equity - seg_initial) * scale).max(0.0)
            } else {
                last_eq
            };
            peak = peak.max(eq);
            let dd = if peak > 0.0 {
                ((peak - eq).max(0.0) / peak) * 100.0
            } else {
                0.0
            };
            eq_vec.push(eq);
            dd_vec.push(dd);
            last_eq = eq;
        }

        (
            ReplayedSeries {
                equity: eq_vec,
                drawdown_pct: dd_vec,
                final_capital: last_eq,
            },
            peak,
        )
    }

    pub fn replay_percentual_segment(
        result: &BacktestResult,
        seg_initial: f64,
        start_capital: f64,
        initial_peak: f64,
    ) -> (ReplayedSeries, f64) {
        let curve = &result.equity_curve;
        let n = curve.len();
        if n == 0 {
            let final_cap = start_capital.max(0.0);
            return (
                ReplayedSeries {
                    equity: Vec::new(),
                    drawdown_pct: Vec::new(),
                    final_capital: final_cap,
                },
                initial_peak.max(final_cap),
            );
        }

        let mut eq_vec = Vec::with_capacity(n);
        let mut dd_vec = Vec::with_capacity(n);

        if !start_capital.is_finite() || start_capital <= 0.0 {
            eq_vec.resize(n, 0.0);
            dd_vec.resize(n, 100.0);
            return (
                ReplayedSeries {
                    equity: eq_vec,
                    drawdown_pct: dd_vec,
                    final_capital: 0.0,
                },
                initial_peak.max(0.0),
            );
        }

        if result.trades.is_empty() {
            let mut peak = initial_peak.max(start_capital);
            let mut prev_comp = start_capital;
            let mut prev_raw = if seg_initial.is_finite() && seg_initial > 0.0 {
                seg_initial
            } else {
                curve[0].equity
            };

            for p in curve {
                let curr = if prev_raw > 0.0 && p.equity.is_finite() && p.equity >= 0.0 {
                    (prev_comp * (p.equity / prev_raw)).max(0.0)
                } else {
                    prev_comp
                };
                peak = peak.max(curr);
                let dd = if peak > 0.0 {
                    ((peak - curr).max(0.0) / peak) * 100.0
                } else {
                    0.0
                };
                eq_vec.push(curr);
                dd_vec.push(dd);
                prev_comp = curr;
                if p.equity.is_finite() && p.equity > 0.0 {
                    prev_raw = p.equity;
                }
            }

            return (
                ReplayedSeries {
                    equity: eq_vec,
                    drawdown_pct: dd_vec,
                    final_capital: prev_comp,
                },
                peak,
            );
        }

        let trades = &result.trades;
        let num_trades = trades.len();
        let mut entries_at = vec![Vec::new(); n];
        let mut open_exits_at = vec![Vec::new(); n];
        let mut close_exits_at = vec![Vec::new(); n];
        let mut trade_scales = vec![0.0_f64; num_trades];
        let mut realized = vec![false; num_trades];

        for (k, t) in trades.iter().enumerate() {
            let entry_pos = curve
                .partition_point(|p| p.candle_idx < t.entry_candle_idx)
                .min(n - 1);
            let exit_pos = curve
                .partition_point(|p| p.candle_idx < t.exit_candle_idx)
                .min(n - 1);

            entries_at[entry_pos].push(k);
            if exit_pos > entry_pos && Self::is_open_exit(&t.exit_reason) {
                open_exits_at[exit_pos].push(k);
            } else {
                close_exits_at[exit_pos].push(k);
            }
        }

        let mut comp_balance = start_capital;
        let mut closed_nom_balance = if seg_initial.is_finite() && seg_initial > 0.0 {
            seg_initial
        } else {
            curve[0].equity.max(1.0)
        };
        let mut open_trades: Vec<usize> = Vec::new();
        let mut peak = initial_peak.max(start_capital);
        let mut last_eq = start_capital;

        for pos in 0..n {
            for &k in &open_exits_at[pos] {
                if !realized[k] {
                    let pnl = sanitize_pnl(trades[k].net_pnl);
                    closed_nom_balance += pnl;
                    comp_balance = (comp_balance + pnl * trade_scales[k]).max(0.0);
                    realized[k] = true;
                }
            }
            if !open_exits_at[pos].is_empty() {
                open_trades.retain(|&k| !realized[k]);
            }

            for &k in &entries_at[pos] {
                let scale = if seg_initial > 0.0 && comp_balance > 0.0 {
                    comp_balance / seg_initial
                } else {
                    0.0
                };
                trade_scales[k] = scale;
                open_trades.push(k);
            }

            for &k in &close_exits_at[pos] {
                if !realized[k] {
                    let pnl = sanitize_pnl(trades[k].net_pnl);
                    closed_nom_balance += pnl;
                    comp_balance = (comp_balance + pnl * trade_scales[k]).max(0.0);
                    realized[k] = true;
                }
            }
            if !close_exits_at[pos].is_empty() {
                open_trades.retain(|&k| !realized[k]);
            }

            let eq = if comp_balance <= 0.0 {
                comp_balance = 0.0;
                0.0
            } else if open_trades.is_empty() {
                comp_balance
            } else {
                let mtm_open_nom = if curve[pos].equity.is_finite() {
                    curve[pos].equity - closed_nom_balance
                } else {
                    0.0
                };
                let open_scale = Self::weighted_open_scale(trades, &open_trades, &trade_scales);
                (comp_balance + mtm_open_nom * open_scale).max(0.0)
            };

            peak = peak.max(eq);
            let dd = if peak > 0.0 {
                ((peak - eq).max(0.0) / peak) * 100.0
            } else {
                0.0
            };

            eq_vec.push(eq);
            dd_vec.push(dd);
            last_eq = eq;
        }

        (
            ReplayedSeries {
                equity: eq_vec,
                drawdown_pct: dd_vec,
                final_capital: comp_balance.max(0.0).min(last_eq.max(comp_balance)),
            },
            peak,
        )
    }

    fn is_open_exit(reason: &str) -> bool {
        matches!(
            reason,
            "Signal Exit" | "Signal Reversal" | "Hedge Signal Exit"
        )
    }

    fn weighted_open_scale(
        trades: &[CompletedTrade],
        open_trades: &[usize],
        trade_scales: &[f64],
    ) -> f64 {
        let mut weight_sum = 0.0;
        let mut weighted_scale = 0.0;
        for &k in open_trades {
            let t = &trades[k];
            let notional = (t.entry_price * t.quantity).abs();
            let w = if notional.is_finite() && notional > 0.0 {
                notional
            } else {
                1.0
            };
            weight_sum += w;
            weighted_scale += w * trade_scales[k];
        }
        if weight_sum > 0.0 {
            weighted_scale / weight_sum
        } else {
            1.0
        }
    }
}

fn sanitize_pnl(pnl: f64) -> f64 {
    if pnl.is_finite() {
        pnl
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::metrics_calculator::MetricsCalculator;
    use crate::types::{PositionSide, TradeRegime};

    fn make_trade(
        entry_idx: usize,
        exit_idx: usize,
        net_pnl: f64,
        exit_reason: &str,
    ) -> CompletedTrade {
        CompletedTrade {
            side: PositionSide::Long,
            entry_timestamp: (entry_idx as i64 + 1) * 3600_000,
            exit_timestamp: (exit_idx as i64 + 1) * 3600_000,
            entry_price: 100.0,
            exit_price: 100.0 + net_pnl,
            quantity: 10.0,
            gross_pnl: net_pnl,
            commissions: 0.0,
            funding_paid: 0.0,
            net_pnl,
            return_pct: net_pnl,
            duration_bars: exit_idx.saturating_sub(entry_idx),
            duration_millis: (exit_idx.saturating_sub(entry_idx) as i64) * 3600_000,
            max_favorable_excursion: 0.0,
            max_adverse_excursion: 0.0,
            exit_reason: exit_reason.to_string(),
            entry_candle_idx: entry_idx,
            exit_candle_idx: exit_idx,
            regime: TradeRegime::PrimaryTrend,
        }
    }

    #[test]
    fn test_nominal_normalization_removes_oos_base_jump() {
        let oos_curve = vec![
            EquityPoint::new(1000, 0, 100_000.0, 100_000.0, 0.0, 0.0),
            EquityPoint::new(2000, 1, 120_000.0, 120_000.0, 0.0, 0.0),
        ];
        let (replayed, _) = CapitalModeReplayer::normalize_nominal_segment(
            &oos_curve,
            100_000.0,
            10_000.0,
            100_000.0,
            100_000.0,
        );
        assert!((replayed.equity[0] - 100_000.0).abs() < 1e-6);
        assert!((replayed.equity[1] - 102_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_percentual_replay_compounds_causally_without_lookahead() {
        let curve = vec![
            EquityPoint::new(1000, 0, 10_000.0, 11_000.0, 0.0, 0.0),
            EquityPoint::new(2000, 1, 15_000.0, 15_000.0, 0.0, 0.0),
            EquityPoint::new(3000, 2, 15_000.0, 16_000.0, 0.0, 0.0),
            EquityPoint::new(4000, 3, 20_000.0, 20_000.0, 0.0, 0.0),
        ];
        let trades = vec![
            make_trade(0, 1, 5_000.0, "Take Profit"),
            make_trade(2, 3, 5_000.0, "Take Profit"),
        ];
        let res = BacktestResult {
            metrics: MetricsCalculator::calculate(10_000.0, &trades, &curve),
            trades,
            equity_curve: curve,
        };

        let (replayed, _) =
            CapitalModeReplayer::replay_percentual_segment(&res, 10_000.0, 10_000.0, 10_000.0);

        assert!((replayed.equity[0] - 11_000.0).abs() < 1e-6);
        assert!((replayed.equity[1] - 15_000.0).abs() < 1e-6);
        assert!((replayed.equity[2] - 16_500.0).abs() < 1e-6);
        assert!((replayed.equity[3] - 22_500.0).abs() < 1e-6);
    }

    #[test]
    fn test_percentual_replay_open_exit_before_same_bar_entry() {
        let curve = vec![
            EquityPoint::new(1000, 0, 10_000.0, 10_000.0, 0.0, 0.0),
            EquityPoint::new(2000, 1, 20_000.0, 20_000.0, 0.0, 0.0),
            EquityPoint::new(3000, 2, 25_000.0, 25_000.0, 0.0, 0.0),
        ];
        let trades = vec![
            make_trade(0, 1, 10_000.0, "Signal Reversal"),
            make_trade(1, 2, 5_000.0, "End of Data Horizon"),
        ];
        let res = BacktestResult {
            metrics: MetricsCalculator::calculate(10_000.0, &trades, &curve),
            trades,
            equity_curve: curve,
        };

        let (replayed, _) =
            CapitalModeReplayer::replay_percentual_segment(&res, 10_000.0, 10_000.0, 10_000.0);

        assert!((replayed.equity[1] - 20_000.0).abs() < 1e-6);
        assert!((replayed.equity[2] - 30_000.0).abs() < 1e-6);
    }
}

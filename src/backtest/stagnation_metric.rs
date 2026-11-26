use super::equity_point::EquityPoint;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StagnationStats {
    pub max_underwater_bars: usize,
    pub max_stagnation_ratio: f64,
    pub total_underwater_ratio: f64,
    pub episode_count: usize,
    pub avg_recovery_bars: f64,
}

pub fn compute_stagnation_stats(equity_curve: &[EquityPoint]) -> StagnationStats {
    if equity_curve.is_empty() {
        return StagnationStats::default();
    }

    let mut peak = if equity_curve[0].equity.is_finite() {
        equity_curve[0].equity
    } else {
        0.0
    };
    let mut current_underwater = 0usize;
    let mut max_underwater = 0usize;
    let mut total_underwater = 0usize;
    let mut current_dd_episode = 0usize;
    let mut episode_count = 0usize;
    let mut episode_bars_sum = 0usize;

    for (idx, pt) in equity_curve.iter().enumerate() {
        if idx > 0 {
            let eq = if pt.equity.is_finite() {
                pt.equity
            } else {
                peak
            };
            if eq > peak {
                peak = eq;
                current_underwater = 0;
            } else {
                current_underwater += 1;
                if current_underwater > max_underwater {
                    max_underwater = current_underwater;
                }
            }
        }

        let dd_pct = if pt.drawdown_pct.is_finite() {
            pt.drawdown_pct
        } else {
            0.0
        };
        if dd_pct >= 1.0 {
            total_underwater += 1;
            current_dd_episode += 1;
        } else {
            if current_dd_episode >= 6 {
                episode_count += 1;
                episode_bars_sum += current_dd_episode;
            }
            current_dd_episode = 0;
        }
    }

    if current_dd_episode >= 6 {
        episode_count += 1;
        episode_bars_sum += current_dd_episode;
    }

    let denom = equity_curve.len().max(1) as f64;
    let avg_recovery_bars = if episode_count > 0 {
        episode_bars_sum as f64 / episode_count as f64
    } else {
        0.0
    };

    StagnationStats {
        max_underwater_bars: max_underwater,
        max_stagnation_ratio: max_underwater as f64 / denom,
        total_underwater_ratio: total_underwater as f64 / denom,
        episode_count,
        avg_recovery_bars,
    }
}

pub fn max_stagnation_ratio(equity_curve: &[EquityPoint]) -> (usize, f64) {
    if equity_curve.is_empty() {
        return (0, 0.0);
    }

    let mut peak = if equity_curve[0].equity.is_finite() {
        equity_curve[0].equity
    } else {
        0.0
    };
    let mut current_underwater = 0usize;
    let mut max_underwater = 0usize;

    for pt in equity_curve.iter().skip(1) {
        let eq = if pt.equity.is_finite() {
            pt.equity
        } else {
            peak
        };
        if eq > peak {
            peak = eq;
            current_underwater = 0;
        } else {
            current_underwater += 1;
            if current_underwater > max_underwater {
                max_underwater = current_underwater;
            }
        }
    }

    let ratio = max_underwater as f64 / (equity_curve.len().max(1) as f64);
    (max_underwater, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(idx: usize, eq: f64) -> EquityPoint {
        EquityPoint::new((idx as i64 + 1) * 1000, idx, eq, eq, 0.0, 0.0)
    }

    fn pt_dd(idx: usize, eq: f64, dd_pct: f64) -> EquityPoint {
        EquityPoint::new((idx as i64 + 1) * 1000, idx, eq, eq, 0.0, dd_pct)
    }

    #[test]
    fn test_max_stagnation_ratio_empty_and_monotonic() {
        assert_eq!(max_stagnation_ratio(&[]), (0, 0.0));
        let mono = vec![pt(0, 100.0), pt(1, 110.0), pt(2, 120.0)];
        assert_eq!(max_stagnation_ratio(&mono), (0, 0.0));
    }

    #[test]
    fn test_max_stagnation_ratio_underwater_window() {
        let curve = vec![
            pt(0, 100.0),
            pt(1, 110.0),
            pt(2, 105.0),
            pt(3, 108.0),
            pt(4, 109.0),
            pt(5, 115.0),
        ];
        let (bars, ratio) = max_stagnation_ratio(&curve);
        assert_eq!(bars, 3);
        assert!((ratio - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_stagnation_stats_episodes_and_ratios() {
        assert_eq!(compute_stagnation_stats(&[]), StagnationStats::default());

        let curve = vec![
            pt_dd(0, 100.0, 0.0),
            pt_dd(1, 110.0, 0.0),
            pt_dd(2, 109.5, 0.45),
            pt_dd(3, 110.0, 0.0),
            pt_dd(4, 112.0, 0.0),
            pt_dd(5, 110.0, 1.79),
            pt_dd(6, 109.0, 2.68),
            pt_dd(7, 108.0, 3.57),
            pt_dd(8, 107.0, 4.46),
            pt_dd(9, 106.0, 5.36),
            pt_dd(10, 110.5, 1.34),
            pt_dd(11, 115.0, 0.0),
        ];
        let stats = compute_stagnation_stats(&curve);
        assert_eq!(stats.max_underwater_bars, 6);
        assert!((stats.max_stagnation_ratio - (6.0 / 12.0)).abs() < 1e-12);
        assert!((stats.total_underwater_ratio - (6.0 / 12.0)).abs() < 1e-12);
        assert_eq!(stats.episode_count, 1);
        assert!((stats.avg_recovery_bars - 6.0).abs() < 1e-12);
    }
}

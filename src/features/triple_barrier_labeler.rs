use rayon::prelude::*;
use crate::types::Kline;
use super::technical_indicators::TechnicalIndicators;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetLabel {
    Short = 0,
    Long = 1,
    Neutral = 2,
}

pub struct TripleBarrierLabeler;

impl TripleBarrierLabeler {
    pub fn label_classification(
        klines: &[Kline],
        horizon: usize,
        multiplier: f64,
        commission_rate: f64,
    ) -> Vec<TargetLabel> {
        let n = klines.len();
        if n <= horizon || horizon == 0 || !multiplier.is_finite() || !commission_rate.is_finite() {
            return vec![TargetLabel::Neutral; n];
        }

        let roundtrip_fee = commission_rate * 2.0;
        let atrs = TechnicalIndicators::atr(klines, 14);
        let valid_n = n.saturating_sub(horizon);

        let mut labels: Vec<TargetLabel> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let entry_price = klines[i + 1].open;
                if !entry_price.is_finite() || entry_price <= 0.0 {
                    return TargetLabel::Neutral;
                }
                let raw_atr = atrs[i];
                if !raw_atr.is_finite() {
                    return TargetLabel::Neutral;
                }
                let atr = raw_atr.max(entry_price * 0.002);
                if !atr.is_finite() || atr <= 0.0 {
                    return TargetLabel::Neutral;
                }
                let upper_barrier = entry_price + multiplier * atr;
                let lower_barrier = entry_price - multiplier * atr;
                if !upper_barrier.is_finite() || !lower_barrier.is_finite() {
                    return TargetLabel::Neutral;
                }

                for step in 0..horizon {
                    let candle = &klines[i + 1 + step];
                    if !candle.high.is_finite()
                        || !candle.low.is_finite()
                        || candle.high <= 0.0
                        || candle.low <= 0.0
                    {
                        return TargetLabel::Neutral;
                    }
                    let hit_upper = candle.high >= upper_barrier;
                    let hit_lower = candle.low <= lower_barrier;

                    if hit_upper && hit_lower {
                        return TargetLabel::Neutral;
                    } else if hit_upper {
                        let gross_return = (upper_barrier - entry_price) / entry_price;
                        return if gross_return - roundtrip_fee > 0.0 {
                            TargetLabel::Long
                        } else {
                            TargetLabel::Neutral
                        };
                    } else if hit_lower {
                        let gross_return = (entry_price - lower_barrier) / entry_price;
                        return if gross_return - roundtrip_fee > 0.0 {
                            TargetLabel::Short
                        } else {
                            TargetLabel::Neutral
                        };
                    }
                }

                let p_exit = klines[i + horizon].close;
                if !p_exit.is_finite() || p_exit <= 0.0 {
                    return TargetLabel::Neutral;
                }
                let long_net = ((p_exit - entry_price) / entry_price) - roundtrip_fee;
                let short_net = ((entry_price - p_exit) / entry_price) - roundtrip_fee;

                if long_net > 0.0 && long_net > short_net {
                    TargetLabel::Long
                } else if short_net > 0.0 && short_net > long_net {
                    TargetLabel::Short
                } else {
                    TargetLabel::Neutral
                }
            })
            .collect();

        labels.resize(n, TargetLabel::Neutral);
        labels
    }

    pub fn label_classification_barriers(
        klines: &[Kline],
        horizon: usize,
        pt: f64,
        sl: f64,
        commission_rate: f64,
    ) -> Vec<TargetLabel> {
        let n = klines.len();
        if n <= horizon
            || horizon == 0
            || !pt.is_finite()
            || pt <= 0.0
            || !sl.is_finite()
            || sl <= 0.0
            || !commission_rate.is_finite()
        {
            return vec![TargetLabel::Neutral; n];
        }

        let roundtrip_fee = commission_rate * 2.0;
        let valid_n = n.saturating_sub(horizon);
        let mut labels: Vec<TargetLabel> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let entry_price = klines[i + 1].open;
                if !entry_price.is_finite() || entry_price <= 0.0 {
                    return TargetLabel::Neutral;
                }

                let long_tp = entry_price * (1.0 + pt);
                let long_sl = entry_price * (1.0 - sl);
                let short_tp = entry_price * (1.0 - pt);
                let short_sl = entry_price * (1.0 + sl);

                let mut long_active = true;
                let mut short_active = true;

                for step in 0..horizon {
                    let candle = &klines[i + 1 + step];
                    if !candle.high.is_finite()
                        || !candle.low.is_finite()
                        || candle.high <= 0.0
                        || candle.low <= 0.0
                    {
                        return TargetLabel::Neutral;
                    }

                    let hit_long_tp = long_active && candle.high >= long_tp;
                    let hit_long_sl = long_active && candle.low <= long_sl;

                    let hit_short_tp = short_active && candle.low <= short_tp;
                    let hit_short_sl = short_active && candle.high >= short_sl;

                    let long_won = hit_long_tp && !hit_long_sl && (pt - roundtrip_fee > 0.0);
                    let short_won = hit_short_tp && !hit_short_sl && (pt - roundtrip_fee > 0.0);

                    if long_won && short_won {
                        return TargetLabel::Neutral;
                    } else if long_won {
                        return TargetLabel::Long;
                    } else if short_won {
                        return TargetLabel::Short;
                    }

                    if hit_long_tp || hit_long_sl {
                        long_active = false;
                    }
                    if hit_short_tp || hit_short_sl {
                        short_active = false;
                    }

                    if !long_active && !short_active {
                        return TargetLabel::Neutral;
                    }
                }

                let p_exit = klines[i + horizon].close;
                if !p_exit.is_finite() || p_exit <= 0.0 {
                    return TargetLabel::Neutral;
                }
                let long_net = ((p_exit - entry_price) / entry_price) - roundtrip_fee;
                let short_net = ((entry_price - p_exit) / entry_price) - roundtrip_fee;

                if long_active && !short_active && long_net > 0.0 {
                    TargetLabel::Long
                } else if short_active && !long_active && short_net > 0.0 {
                    TargetLabel::Short
                } else if long_active && short_active {
                    if long_net > 0.0 && long_net > short_net {
                        TargetLabel::Long
                    } else if short_net > 0.0 && short_net > long_net {
                        TargetLabel::Short
                    } else {
                        TargetLabel::Neutral
                    }
                } else {
                    TargetLabel::Neutral
                }
            })
            .collect();

        labels.resize(n, TargetLabel::Neutral);
        labels
    }

    pub fn label_regression(klines: &[Kline], horizon: usize, commission_rate: f64) -> Vec<f64> {
        let n = klines.len();
        if n <= horizon || horizon == 0 || !commission_rate.is_finite() {
            return vec![0.0; n];
        }

        let roundtrip_fee = commission_rate * 2.0;

        let valid_n = n.saturating_sub(horizon);
        let mut returns: Vec<f64> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let p_entry = klines[i + 1].open;
                let p_exit = klines[i + horizon].close;
                if p_entry.is_finite() && p_entry > 0.0 && p_exit.is_finite() && p_exit > 0.0 {
                    let gross_return = (p_exit - p_entry) / p_entry;
                    if !gross_return.is_finite() {
                        return 0.0;
                    }
                    if gross_return > roundtrip_fee {
                        gross_return - roundtrip_fee
                    } else if gross_return < -roundtrip_fee {
                        gross_return + roundtrip_fee
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();

        returns.resize(n, 0.0);
        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kline(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Kline {
        Kline {
            timestamp: ts,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_barrier_long_take_profit() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 104.5, 99.5, 104.0),
            make_kline(220, 104.0, 105.0, 103.0, 104.0),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_eq!(labels[0], TargetLabel::Long);
    }

    #[test]
    fn test_barrier_short_take_profit() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 100.5, 95.5, 95.8),
            make_kline(220, 95.8, 96.0, 94.0, 95.0),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_eq!(labels[0], TargetLabel::Short);
    }

    #[test]
    fn test_barrier_short_not_triggered_by_long_sl_distance() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 100.2, 97.5, 98.0),
            make_kline(220, 98.0, 101.0, 98.0, 100.5),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_eq!(labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_barrier_short_stopped_out_before_drop() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 102.5, 99.5, 102.0),
            make_kline(220, 102.0, 102.0, 94.0, 94.5),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_ne!(labels[0], TargetLabel::Short);
        assert_eq!(labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_tp_hit_with_pt_le_fee_deactivates_active_flag() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 100.2, 99.9, 100.1),
            make_kline(220, 100.1, 105.0, 100.0, 104.0),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.001, 0.05, 0.0005);
        assert_eq!(labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_non_finite_ohlc_returns_neutral_and_zero() {
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, f64::NAN, 105.0, 99.0, 104.0),
            make_kline(220, 104.0, 106.0, 103.0, f64::INFINITY),
        ];

        let cls = TripleBarrierLabeler::label_classification(&klines, 2, 1.5, 0.0005);
        assert_eq!(cls[0], TargetLabel::Neutral);

        let bar = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.02, 0.01, 0.0005);
        assert_eq!(bar[0], TargetLabel::Neutral);

        let reg = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        assert_eq!(reg[0], 0.0);
    }

    #[test]
    fn test_bug_f01_fee_deduction_prevents_false_positive_gains() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.9, 100.02),
            make_kline(220, 100.02, 100.06, 99.98, 100.05),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        assert_eq!(reg_targets[0], 0.0);

        let class_labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.02, 0.02, 0.0005);
        assert_eq!(class_labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_positive() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.5, 99.9, 100.1),
            make_kline(220, 100.1, 100.5, 100.0, 100.3),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        let expected = (100.3 - 100.0) / 100.0 - 0.0010;
        assert!((reg_targets[0] - expected).abs() < 1e-9);
        assert!(reg_targets[0] > 0.0);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_negative() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.5, 99.9),
            make_kline(220, 99.9, 100.0, 99.4, 99.6),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        let expected = (99.6 - 100.0) / 100.0 + 0.0010;
        assert!((reg_targets[0] - expected).abs() < 1e-9);
        assert!(reg_targets[0] < 0.0);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_neutral() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.9, 100.0),
            make_kline(220, 100.0, 100.05, 99.95, 99.95),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        assert_eq!(reg_targets[0], 0.0);
    }
}

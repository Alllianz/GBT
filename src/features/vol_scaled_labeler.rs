use rayon::prelude::*;
use crate::types::Kline;

pub struct VolScaledLabeler;

impl VolScaledLabeler {
    pub fn label_vol_scaled_regression(
        klines: &[Kline],
        horizon: usize,
        commission_rate: f64,
        atr_window: usize,
    ) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || horizon == 0 || atr_window == 0 {
            return vec![0.0; n];
        }

        let valid_end = n.saturating_sub(horizon);
        if valid_end <= atr_window {
            return vec![0.0; n];
        }

        let mut tr_prefix = Vec::with_capacity(n + 1);
        tr_prefix.push(0.0);
        for i in 0..n {
            let h = klines[i].high;
            let l = klines[i].low;
            let tr = if i == 0 {
                (h - l).max(0.0)
            } else {
                let prev_c = klines[i - 1].close;
                (h - l)
                    .max((h - prev_c).abs())
                    .max((l - prev_c).abs())
            };
            let safe_tr = if tr.is_finite() { tr.max(0.0) } else { 0.0 };
            tr_prefix.push(tr_prefix[i] + safe_tr);
        }

        let deadband = 2.0 * commission_rate.max(0.0);
        let sqrt_h = (horizon as f64).sqrt();
        let inv_atr_window = 1.0 / (atr_window as f64);

        let mut labels = vec![0.0; n];
        let computed: Vec<f64> = (atr_window..valid_end)
            .into_par_iter()
            .map(|i| {
                let close_i = klines[i].close;
                if !close_i.is_finite() || close_i <= 0.0 {
                    return 0.0;
                }
                let atr_i = (tr_prefix[i + 1] - tr_prefix[i + 1 - atr_window]) * inv_atr_window;
                let atr_pct_i = atr_i / close_i;
                if !atr_pct_i.is_finite() || atr_pct_i <= 1e-12 {
                    return 0.0;
                }

                let entry = klines[i + 1].open;
                let exit = klines[i + horizon].close;
                if !entry.is_finite() || entry <= 0.0 || !exit.is_finite() {
                    return 0.0;
                }

                let raw_ret = (exit - entry) / entry;
                if !raw_ret.is_finite() {
                    return 0.0;
                }

                let net_ret = if raw_ret.abs() <= deadband {
                    0.0
                } else {
                    raw_ret - raw_ret.signum() * deadband
                };

                let denom = atr_pct_i * sqrt_h;
                if !denom.is_finite() || denom <= 1e-12 {
                    return 0.0;
                }

                (net_ret / denom).clamp(-3.0, 3.0)
            })
            .collect();

        labels[atr_window..valid_end].copy_from_slice(&computed);
        labels
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
    fn test_vol_scaled_deadband_and_clamping() {
        let mut klines = Vec::new();
        for i in 0..25 {
            klines.push(make_kline(i as i64 * 3600, 100.0, 101.0, 99.0, 100.0));
        }
        klines[15].open = 100.0;
        klines[16].close = 100.05;
        klines[18].open = 100.0;
        klines[19].close = 120.0;
        klines[20].open = 100.0;
        klines[21].close = 80.0;

        let labels = VolScaledLabeler::label_vol_scaled_regression(&klines, 2, 0.0005, 14);
        assert_eq!(labels.len(), klines.len());
        assert!(labels[..14].iter().all(|&v| v == 0.0));
        assert!(labels[23..].iter().all(|&v| v == 0.0));
        assert_eq!(labels[14], 0.0);
        assert_eq!(labels[17], 3.0);
        assert_eq!(labels[19], -3.0);
    }

    #[test]
    fn test_vol_scaled_causality_atr_t_le_t() {
        let mut klines_a = Vec::new();
        for i in 0..40 {
            klines_a.push(make_kline(i as i64 * 3600, 100.0, 101.0, 99.0, 100.0));
        }
        klines_a[15].open = 100.0;
        klines_a[18].close = 102.0;

        let mut klines_b = klines_a.clone();
        for k in &mut klines_b[19..] {
            k.high = 500.0;
            k.low = 10.0;
            k.close = 300.0;
        }

        let labels_a = VolScaledLabeler::label_vol_scaled_regression(&klines_a, 4, 0.0005, 14);
        let labels_b = VolScaledLabeler::label_vol_scaled_regression(&klines_b, 4, 0.0005, 14);
        assert!((labels_a[14] - labels_b[14]).abs() < 1e-12);

        let expected_atr_pct = 2.0 / 100.0;
        let expected_net = 0.02 - 0.001;
        let expected = expected_net / (expected_atr_pct * (4.0f64).sqrt());
        assert!((labels_a[14] - expected).abs() < 1e-12);
    }
}

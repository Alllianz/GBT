use crate::types::Kline;

pub struct HmmFeatureExtractor;

impl HmmFeatureExtractor {
    pub fn extract_causal_step(history: &[Kline], idx: usize, vol_window: usize) -> Option<[f64; 2]> {
        if vol_window < 2 || idx < vol_window || idx >= history.len() {
            return None;
        }

        let start = idx - vol_window;
        let window_slice = &history[start..=idx];
        for k in window_slice {
            if !k.close.is_finite() || k.close <= 0.0 {
                return None;
            }
        }

        let n = vol_window as f64;
        let mut sum = 0.0;
        let mut log_rets = Vec::with_capacity(vol_window);
        for w in window_slice.windows(2) {
            let r = (w[1].close / w[0].close).ln();
            if !r.is_finite() {
                return None;
            }
            sum += r;
            log_rets.push(r);
        }

        let latest_return = *log_rets.last()?;
        let mean = sum / n;
        let mut var_sum = 0.0;
        for r in &log_rets {
            let diff = *r - mean;
            var_sum += diff * diff;
        }
        let realized_vol = (var_sum / (n - 1.0)).max(0.0).sqrt();

        if !latest_return.is_finite() || !realized_vol.is_finite() {
            return None;
        }

        Some([latest_return, realized_vol])
    }

    pub fn extract_causal_window(history: &[Kline], end_idx: usize, vol_window: usize) -> Vec<[f64; 2]> {
        if history.is_empty() || vol_window < 2 {
            return Vec::new();
        }
        let bound = end_idx.min(history.len() - 1);
        if bound < vol_window {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(bound - vol_window + 1);
        for idx in vol_window..=bound {
            if let Some(feat) = Self::extract_causal_step(history, idx, vol_window) {
                out.push(feat);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_klines(closes: &[f64]) -> Vec<Kline> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Kline {
                timestamp: (i as i64) * 3600_000,
                open: c,
                high: c * 1.01,
                low: c * 0.99,
                close: c,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_extract_causal_step_strict_no_lookahead() {
        let base_closes = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 102.0, 105.0];
        let mut perturbed_closes = base_closes.clone();
        perturbed_closes[6] = 200.0;
        perturbed_closes[7] = 50.0;

        let k_base = make_klines(&base_closes);
        let k_pert = make_klines(&perturbed_closes);

        let f_base = HmmFeatureExtractor::extract_causal_step(&k_base, 5, 3).unwrap();
        let f_pert = HmmFeatureExtractor::extract_causal_step(&k_pert, 5, 3).unwrap();
        assert!((f_base[0] - f_pert[0]).abs() < 1e-12);
        assert!((f_base[1] - f_pert[1]).abs() < 1e-12);

        let w_base = HmmFeatureExtractor::extract_causal_window(&k_base, 5, 3);
        let w_pert = HmmFeatureExtractor::extract_causal_window(&k_pert, 5, 3);
        assert_eq!(w_base.len(), 3);
        assert_eq!(w_base, w_pert);
    }

    #[test]
    fn test_extract_causal_step_bounds() {
        let klines = make_klines(&[100.0, 102.0, 101.0, 103.0]);
        assert!(HmmFeatureExtractor::extract_causal_step(&klines, 0, 2).is_none());
        assert!(HmmFeatureExtractor::extract_causal_step(&klines, 1, 2).is_none());
        assert!(HmmFeatureExtractor::extract_causal_step(&klines, 2, 1).is_none());
        assert!(HmmFeatureExtractor::extract_causal_step(&klines, 4, 2).is_none());
        assert!(HmmFeatureExtractor::extract_causal_step(&klines, 2, 2).is_some());
    }
}

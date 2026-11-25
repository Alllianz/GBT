use crate::features::TechnicalIndicators;
use crate::types::Kline;

pub struct TreeBoostMrIndicators;

impl TreeBoostMrIndicators {
    pub fn bb_zscore(history: &[Kline], current_idx: usize, window: usize) -> Option<f64> {
        if window < 2 || current_idx + 1 < window || current_idx >= history.len() {
            return None;
        }
        let start = current_idx + 1 - window;
        let slice = &history[start..=current_idx];
        let mut sum = 0.0;
        for k in slice {
            if !k.close.is_finite() || k.close <= 0.0 {
                return None;
            }
            sum += k.close;
        }
        let n = window as f64;
        let sma = sum / n;
        let var = slice
            .iter()
            .map(|k| {
                let d = k.close - sma;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.max(0.0).sqrt();
        if !std.is_finite() || std <= 1e-12 {
            return None;
        }
        let z = (history[current_idx].close - sma) / std;
        if z.is_finite() {
            Some(z)
        } else {
            None
        }
    }

    pub fn rsi_14(history: &[Kline], current_idx: usize) -> Option<f64> {
        if current_idx < 14 || current_idx >= history.len() {
            return None;
        }
        let start = current_idx.saturating_sub(84);
        let slice = &history[start..=current_idx];
        let mut closes = Vec::with_capacity(slice.len());
        for k in slice {
            if !k.close.is_finite() || k.close <= 0.0 {
                return None;
            }
            closes.push(k.close);
        }
        TechnicalIndicators::rsi(&closes, 14)
            .last()
            .copied()
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 100.0))
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
                timestamp: (i as i64 + 1) * 3600,
                open: c,
                high: c + 0.5,
                low: (c - 0.5).max(0.1),
                close: c,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_bb_zscore_and_rsi_causality() {
        let mut closes: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 0.5).collect();
        closes[29] = 140.0;
        let klines_a = make_klines(&closes);

        let mut closes_future_mutated = closes.clone();
        closes_future_mutated.extend_from_slice(&[50.0, 200.0, 10.0]);
        let klines_b = make_klines(&closes_future_mutated);

        let z_a = TreeBoostMrIndicators::bb_zscore(&klines_a, 29, 20).unwrap();
        let z_b = TreeBoostMrIndicators::bb_zscore(&klines_b, 29, 20).unwrap();
        assert!((z_a - z_b).abs() < 1e-12);
        assert!(z_a > 2.0);

        let rsi_a = TreeBoostMrIndicators::rsi_14(&klines_a, 29).unwrap();
        let rsi_b = TreeBoostMrIndicators::rsi_14(&klines_b, 29).unwrap();
        assert!((rsi_a - rsi_b).abs() < 1e-12);
        assert!((0.0..=100.0).contains(&rsi_a));
        assert!(rsi_a > 70.0);

        assert!(TreeBoostMrIndicators::bb_zscore(&klines_a, 10, 20).is_none());
        assert!(TreeBoostMrIndicators::rsi_14(&klines_a, 10).is_none());
    }
}

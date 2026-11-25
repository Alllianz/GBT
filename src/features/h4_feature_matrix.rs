use rayon::prelude::*;
use crate::types::Kline;
use super::technical_indicators::TechnicalIndicators;

pub struct H4FeatureMatrix;

impl H4FeatureMatrix {
    pub const WARMUP_BARS: usize = 185;

    pub const FEATURE_NAMES: [&'static str; 13] = [
        "ret_6",
        "ret_18",
        "ret_30",
        "ema60_ratio",
        "ema180_ratio",
        "ema_macro_spread",
        "rsi_84_norm",
        "atr_84_pct",
        "macro_range_pct",
        "macro_body_pct",
        "vol_ratio_120",
        "ret_6_vol_norm",
        "vol_regime_ratio",
    ];

    pub fn num_features() -> usize {
        Self::FEATURE_NAMES.len()
    }

    pub fn build(klines: &[Kline]) -> (Vec<f64>, usize) {
        let n = klines.len();
        if n < Self::WARMUP_BARS {
            return (Vec::new(), 0);
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let (ema60, (ema180, (rsi84, (atr14, (atr84, ema_vol120))))) = rayon::join(
            || TechnicalIndicators::ema(&closes, 60),
            || rayon::join(
                || TechnicalIndicators::ema(&closes, 180),
                || rayon::join(
                    || TechnicalIndicators::rsi(&closes, 84),
                    || rayon::join(
                        || TechnicalIndicators::atr(klines, 14),
                        || rayon::join(
                            || TechnicalIndicators::atr(klines, 84),
                            || TechnicalIndicators::ema(&volumes, 120),
                        ),
                    ),
                ),
            ),
        );

        let matrix: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let c = klines[i].close;

                let ret_6 = causal_return(klines, i, 6);
                let ret_18 = causal_return(klines, i, 18);
                let ret_30 = causal_return(klines, i, 30);

                let e60 = ema60[i];
                let e180 = ema180[i];

                let ema60_ratio = safe_ratio_diff(c, e60);
                let ema180_ratio = safe_ratio_diff(c, e180);
                let ema_macro_spread = safe_ratio_diff(e60, e180);

                let rsi_raw = rsi84[i];
                let rsi_84_norm = if rsi_raw.is_finite() {
                    ((rsi_raw - 50.0) / 50.0).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                let atr_raw = atr84[i];
                let atr_84_pct = if c > 0.0 && c.is_finite() && atr_raw.is_finite() {
                    atr_raw / c
                } else {
                    0.0
                };

                let atr_14_raw = atr14[i];
                let atr_14_pct = if c > 0.0 && c.is_finite() && atr_14_raw.is_finite() {
                    atr_14_raw / c
                } else {
                    0.0
                };

                let (macro_range_pct, macro_body_pct, vol_ratio_120) = if i >= 5 {
                    let window = &klines[i - 5..=i];
                    let max_h = window.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max);
                    let min_l = window.iter().map(|k| k.low).fold(f64::INFINITY, f64::min);
                    let range_pct = if c > 0.0 && c.is_finite() && max_h.is_finite() && min_l.is_finite() {
                        (max_h - min_l) / c
                    } else {
                        0.0
                    };

                    let macro_open = window[0].open;
                    let body_pct = if c > 0.0 && c.is_finite() && macro_open.is_finite() {
                        (c - macro_open) / c
                    } else {
                        0.0
                    };

                    let avg_vol_6 = window.iter().map(|k| k.volume).sum::<f64>() / 6.0;
                    let v_ema = ema_vol120[i];
                    let v_ratio = if v_ema > 0.0 && v_ema.is_finite() && avg_vol_6.is_finite() {
                        (avg_vol_6 / v_ema).clamp(0.0, 10.0)
                    } else {
                        1.0
                    };

                    (range_pct, body_pct, v_ratio)
                } else {
                    (0.0, 0.0, 1.0)
                };

                let ret_6_vol_norm = if atr_84_pct > 1e-8 && atr_84_pct.is_finite() && ret_6.is_finite() {
                    (ret_6 / atr_84_pct).clamp(-10.0, 10.0)
                } else {
                    0.0
                };
                let vol_regime_ratio =
                    if atr_84_pct > 1e-8 && atr_84_pct.is_finite() && atr_14_pct.is_finite() {
                        (atr_14_pct / atr_84_pct).clamp(0.0, 10.0)
                    } else {
                        1.0
                    };

                [
                    ret_6,
                    ret_18,
                    ret_30,
                    ema60_ratio,
                    ema180_ratio,
                    ema_macro_spread,
                    rsi_84_norm,
                    atr_84_pct,
                    macro_range_pct,
                    macro_body_pct,
                    vol_ratio_120,
                    ret_6_vol_norm,
                    vol_regime_ratio,
                ]
            })
            .collect();

        (matrix, n)
    }

    pub fn extract_single_vector(history: &[Kline], idx: usize) -> Option<Vec<f64>> {
        if idx + 1 < Self::WARMUP_BARS || idx >= history.len() {
            return None;
        }

        let slice = &history[0..=idx];
        let (matrix, n) = Self::build(slice);
        if n == 0 {
            return None;
        }

        let f_dim = Self::num_features();
        let start = (n - 1) * f_dim;
        let end = start + f_dim;
        Some(matrix[start..end].to_vec())
    }
}

fn causal_return(klines: &[Kline], i: usize, lookback: usize) -> f64 {
    if i < lookback {
        return 0.0;
    }
    safe_ratio_diff(klines[i].close, klines[i - lookback].close)
}

fn safe_ratio_diff(num: f64, den: f64) -> f64 {
    if den > 0.0 && den.is_finite() && num.is_finite() {
        (num - den) / den
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_klines(n: usize) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let change = ((i as f64 * 0.13).sin() * 2.0) + ((i as f64 * 0.04).cos() * 0.8);
            let open = price;
            let close = (price + change).max(1.0);
            let high = open.max(close) + 0.75;
            let low = (open.min(close) - 0.75).max(0.5);
            let volume = 1000.0 + (i as f64 * 15.0) + ((i as f64 * 0.2).sin() * 200.0);
            klines.push(Kline {
                timestamp: (1600000000 + i * 14400) as i64,
                open,
                high,
                low,
                close,
                volume,
            });
            price = close;
        }
        klines
    }

    #[test]
    fn test_warmup_and_dimensions() {
        assert_eq!(H4FeatureMatrix::WARMUP_BARS, 185);
        assert_eq!(H4FeatureMatrix::num_features(), 13);
        assert_eq!(H4FeatureMatrix::FEATURE_NAMES.len(), 13);
        assert_eq!(
            H4FeatureMatrix::FEATURE_NAMES,
            [
                "ret_6",
                "ret_18",
                "ret_30",
                "ema60_ratio",
                "ema180_ratio",
                "ema_macro_spread",
                "rsi_84_norm",
                "atr_84_pct",
                "macro_range_pct",
                "macro_body_pct",
                "vol_ratio_120",
                "ret_6_vol_norm",
                "vol_regime_ratio",
            ]
        );

        let klines_short = generate_test_klines(184);
        let (mat, rows) = H4FeatureMatrix::build(&klines_short);
        assert!(mat.is_empty());
        assert_eq!(rows, 0);

        let klines = generate_test_klines(220);
        assert!(H4FeatureMatrix::extract_single_vector(&klines, 183).is_none());
        assert!(H4FeatureMatrix::extract_single_vector(&klines, 184).is_some());
        assert!(H4FeatureMatrix::extract_single_vector(&klines, 220).is_none());
    }

    #[test]
    fn test_causality_t_le_t_and_batch_consistency() {
        let klines_a = generate_test_klines(260);
        let mut klines_b = klines_a.clone();
        for k in &mut klines_b[201..] {
            k.open *= 5.0;
            k.high *= 8.0;
            k.low *= 0.2;
            k.close *= 6.0;
            k.volume *= 50.0;
        }

        let (mat_a, rows_a) = H4FeatureMatrix::build(&klines_a);
        let (mat_b, rows_b) = H4FeatureMatrix::build(&klines_b);
        assert_eq!(rows_a, 260);
        assert_eq!(rows_b, 260);
        let f_dim = H4FeatureMatrix::num_features();
        assert_eq!(mat_a.len(), 260 * f_dim);

        for idx in 0..=200 {
            for f in 0..f_dim {
                let va = mat_a[idx * f_dim + f];
                let vb = mat_b[idx * f_dim + f];
                assert!(
                    (va - vb).abs() < 1e-12,
                    "Future mutation leaked at idx {} feature {}: {} vs {}",
                    idx,
                    H4FeatureMatrix::FEATURE_NAMES[f],
                    va,
                    vb
                );
            }
        }

        for &idx in &[184, 195, 200, 259] {
            let single = H4FeatureMatrix::extract_single_vector(&klines_a, idx)
                .expect("expected valid feature vector");
            assert_eq!(single.len(), f_dim);
            let batch_row = &mat_a[idx * f_dim..(idx + 1) * f_dim];
            for f in 0..f_dim {
                assert!(
                    (single[f] - batch_row[f]).abs() < 1e-12,
                    "Single vs batch mismatch at idx {} feature {}",
                    idx,
                    f
                );
            }
        }

        let single_a_200 = H4FeatureMatrix::extract_single_vector(&klines_a, 200).unwrap();
        let single_b_200 = H4FeatureMatrix::extract_single_vector(&klines_b, 200).unwrap();
        assert_eq!(single_a_200, single_b_200);
    }

    #[test]
    fn test_all_features_finite_and_bounded() {
        let klines = generate_test_klines(250);
        let (matrix, rows) = H4FeatureMatrix::build(&klines);
        assert_eq!(rows, 250);
        let f_dim = H4FeatureMatrix::num_features();

        for idx in 0..rows {
            let row = &matrix[idx * f_dim..(idx + 1) * f_dim];
            for (f, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite value at idx {} feature {}",
                    idx,
                    H4FeatureMatrix::FEATURE_NAMES[f]
                );
            }
            assert!((-1.0..=1.0).contains(&row[6]));
            assert!(row[7] >= 0.0);
            assert!((0.0..=10.0).contains(&row[10]));
            assert!((-10.0..=10.0).contains(&row[11]));
            assert!((0.0..=10.0).contains(&row[12]));
        }

        let zero_klines = vec![
            Kline {
                timestamp: 0,
                open: 0.0,
                high: 0.0,
                low: 0.0,
                close: 0.0,
                volume: 0.0,
            };
            200
        ];
        let (z_mat, z_rows) = H4FeatureMatrix::build(&zero_klines);
        assert_eq!(z_rows, 200);
        for (i, &val) in z_mat.iter().enumerate() {
            assert!(val.is_finite(), "Non-finite zero-kline feature at {}", i);
        }
    }

    #[test]
    fn test_macro_feature_formulas_exact() {
        let mut klines = vec![
            Kline {
                timestamp: 0,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 100.0,
                volume: 1000.0,
            };
            200
        ];
        for (i, k) in klines.iter_mut().enumerate() {
            k.timestamp = (1600000000 + i * 14400) as i64;
        }
        klines[185].open = 98.0;
        klines[186].high = 120.0;
        klines[187].low = 90.0;
        klines[190].open = 102.0;
        klines[190].close = 110.0;
        for i in 185..=190 {
            klines[i].volume = 1_000_000.0;
        }
        klines[196].close = 121.0;

        let vec190 = H4FeatureMatrix::extract_single_vector(&klines, 190).unwrap();
        let expected_ret6 = (110.0 - 100.0) / 100.0;
        let expected_ret18 = (110.0 - 100.0) / 100.0;
        let expected_ret30 = (110.0 - 100.0) / 100.0;
        let expected_macro_range = (120.0 - 90.0) / 110.0;
        let expected_macro_body = (110.0 - 98.0) / 110.0;

        assert!((vec190[0] - expected_ret6).abs() < 1e-12);
        assert!((vec190[1] - expected_ret18).abs() < 1e-12);
        assert!((vec190[2] - expected_ret30).abs() < 1e-12);
        assert!((vec190[6] - 1.0).abs() < 1e-12);
        assert!((vec190[8] - expected_macro_range).abs() < 1e-12);
        assert!((vec190[9] - expected_macro_body).abs() < 1e-12);
        assert!((vec190[10] - 10.0).abs() < 1e-12);

        let vec196 = H4FeatureMatrix::extract_single_vector(&klines, 196).unwrap();
        let expected_ret6_196 = (121.0 - 110.0) / 110.0;
        let expected_ret18_196 = (121.0 - 100.0) / 100.0;
        let expected_ret30_196 = (121.0 - 100.0) / 100.0;
        assert!((vec196[0] - expected_ret6_196).abs() < 1e-12);
        assert!((vec196[1] - expected_ret18_196).abs() < 1e-12);
        assert!((vec196[2] - expected_ret30_196).abs() < 1e-12);
    }
}

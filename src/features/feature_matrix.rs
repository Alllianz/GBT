use rayon::prelude::*;
use crate::types::Kline;
use super::technical_indicators::TechnicalIndicators;
use super::volatility::{GarmanKlassVolatility, ParkinsonVolatility, YangZhangVolatility};

pub struct FeatureMatrix;

impl FeatureMatrix {
    pub const FEATURE_NAMES: [&'static str; 11] = [
        "return_1",
        "return_3",
        "return_5",
        "ema10_ratio",
        "ema30_ratio",
        "ema_cross_spread",
        "rsi_14_norm",
        "atr_14_pct",
        "candle_range_pct",
        "candle_body_pct",
        "volume_ratio_20",
    ];

    /// Catálogo extendido con estimadores institucionales de microestructura y volatilidad
    pub const EXTENDED_FEATURE_NAMES: [&'static str; 14] = [
        "return_1",
        "return_3",
        "return_5",
        "ema10_ratio",
        "ema30_ratio",
        "ema_cross_spread",
        "rsi_14_norm",
        "atr_14_pct",
        "candle_range_pct",
        "candle_body_pct",
        "volume_ratio_20",
        "volatility_parkinson_14",
        "volatility_garman_klass_14",
        "volatility_yang_zhang_14",
    ];

    pub fn num_features() -> usize {
        Self::FEATURE_NAMES.len()
    }

    pub fn num_extended_features() -> usize {
        Self::EXTENDED_FEATURE_NAMES.len()
    }

    /// Construye una matriz 2D plana con las features causales de cada vela en paralelo con Rayon (20 hilos)
    /// Retorna (vector aplanado [N x F], número de filas N).
    pub fn build(klines: &[Kline]) -> (Vec<f64>, usize) {
        let n = klines.len();
        if n < 35 {
            return (Vec::new(), 0);
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Cálculo concurrente de los 5 indicadores técnicos con Rayon
        let (ema10, (ema30, (rsi14, (atr14, vol_sma20)))) = rayon::join(
            || TechnicalIndicators::ema(&closes, 10),
            || rayon::join(
                || TechnicalIndicators::ema(&closes, 30),
                || rayon::join(
                    || TechnicalIndicators::rsi(&closes, 14),
                    || rayon::join(
                        || TechnicalIndicators::atr(klines, 14),
                        || TechnicalIndicators::ema(&volumes, 20),
                    ),
                ),
            ),
        );

        // Ensamblado paralelo del vector de características de todas las velas
        let matrix: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let c = klines[i].close;
                let o = klines[i].open;
                let h = klines[i].high;
                let l = klines[i].low;
                let v = klines[i].volume;

                let r1 = if i >= 1 && klines[i - 1].close > 0.0 {
                    (c - klines[i - 1].close) / klines[i - 1].close
                } else {
                    0.0
                };

                let r3 = if i >= 3 && klines[i - 3].close > 0.0 {
                    (c - klines[i - 3].close) / klines[i - 3].close
                } else {
                    0.0
                };

                let r5 = if i >= 5 && klines[i - 5].close > 0.0 {
                    (c - klines[i - 5].close) / klines[i - 5].close
                } else {
                    0.0
                };

                let e10 = ema10[i];
                let e30 = ema30[i];

                let ema10_ratio = if e10 > 0.0 { (c - e10) / e10 } else { 0.0 };
                let ema30_ratio = if e30 > 0.0 { (c - e30) / e30 } else { 0.0 };
                let ema_spread = if e30 > 0.0 { (e10 - e30) / e30 } else { 0.0 };

                let rsi_norm = (rsi14[i] / 100.0).clamp(0.0, 1.0);
                let atr_pct = if c > 0.0 { atr14[i] / c } else { 0.0 };

                let range_pct = if c > 0.0 { (h - l) / c } else { 0.0 };
                let body_pct = if c > 0.0 { (c - o) / c } else { 0.0 };

                let v_sma = vol_sma20[i];
                let vol_ratio = if v_sma > 0.0 { (v / v_sma).min(10.0) } else { 1.0 };

                [
                    r1,
                    r3,
                    r5,
                    ema10_ratio,
                    ema30_ratio,
                    ema_spread,
                    rsi_norm,
                    atr_pct,
                    range_pct,
                    body_pct,
                    vol_ratio,
                ]
            })
            .collect();

        (matrix, n)
    }

    /// Construye la matriz enriquecida incorporando estimadores institucionales de volatilidad
    /// (Parkinson, Garman-Klass y Yang-Zhang) calculados concurrentemente de forma causal.
    pub fn build_extended(klines: &[Kline]) -> (Vec<f64>, usize) {
        let n = klines.len();
        if n < 35 {
            return (Vec::new(), 0);
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Cómputo concurrente de indicadores técnicos y estimadores de volatilidad
        let (
            (ema10, (ema30, (rsi14, (atr14, vol_sma20)))),
            (parkinson, (garman_klass, yang_zhang)),
        ) = rayon::join(
            || {
                rayon::join(
                    || TechnicalIndicators::ema(&closes, 10),
                    || rayon::join(
                        || TechnicalIndicators::ema(&closes, 30),
                        || rayon::join(
                            || TechnicalIndicators::rsi(&closes, 14),
                            || rayon::join(
                                || TechnicalIndicators::atr(klines, 14),
                                || TechnicalIndicators::ema(&volumes, 20),
                            ),
                        ),
                    ),
                )
            },
            || {
                rayon::join(
                    || ParkinsonVolatility::calculate(klines, 14),
                    || rayon::join(
                        || GarmanKlassVolatility::calculate(klines, 14),
                        || YangZhangVolatility::calculate(klines, 14),
                    ),
                )
            },
        );

        let matrix: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let c = klines[i].close;
                let o = klines[i].open;
                let h = klines[i].high;
                let l = klines[i].low;
                let v = klines[i].volume;

                let r1 = if i >= 1 && klines[i - 1].close > 0.0 {
                    (c - klines[i - 1].close) / klines[i - 1].close
                } else {
                    0.0
                };

                let r3 = if i >= 3 && klines[i - 3].close > 0.0 {
                    (c - klines[i - 3].close) / klines[i - 3].close
                } else {
                    0.0
                };

                let r5 = if i >= 5 && klines[i - 5].close > 0.0 {
                    (c - klines[i - 5].close) / klines[i - 5].close
                } else {
                    0.0
                };

                let e10 = ema10[i];
                let e30 = ema30[i];

                let ema10_ratio = if e10 > 0.0 { (c - e10) / e10 } else { 0.0 };
                let ema30_ratio = if e30 > 0.0 { (c - e30) / e30 } else { 0.0 };
                let ema_spread = if e30 > 0.0 { (e10 - e30) / e30 } else { 0.0 };

                let rsi_norm = (rsi14[i] / 100.0).clamp(0.0, 1.0);
                let atr_pct = if c > 0.0 { atr14[i] / c } else { 0.0 };

                let range_pct = if c > 0.0 { (h - l) / c } else { 0.0 };
                let body_pct = if c > 0.0 { (c - o) / c } else { 0.0 };

                let v_sma = vol_sma20[i];
                let vol_ratio = if v_sma > 0.0 { (v / v_sma).min(10.0) } else { 1.0 };

                let vol_p = parkinson[i];
                let vol_gk = garman_klass[i];
                let vol_yz = yang_zhang[i];

                [
                    r1,
                    r3,
                    r5,
                    ema10_ratio,
                    ema30_ratio,
                    ema_spread,
                    rsi_norm,
                    atr_pct,
                    range_pct,
                    body_pct,
                    vol_ratio,
                    vol_p,
                    vol_gk,
                    vol_yz,
                ]
            })
            .collect();

        (matrix, n)
    }

    /// Extrae las features de una sola vela en el instante `idx` causalmente (t <= T).
    pub fn extract_single_vector(klines: &[Kline], idx: usize) -> Option<Vec<f64>> {
        if idx < 34 || idx >= klines.len() {
            return None;
        }

        let slice = &klines[0..=idx];
        let (matrix, n) = Self::build(slice);
        if n == 0 {
            return None;
        }

        let f_dim = Self::num_features();
        let start = (n - 1) * f_dim;
        let end = start + f_dim;
        Some(matrix[start..end].to_vec())
    }

    /// Extrae el vector individual de 14 features extendidas causalmente (t <= T).
    pub fn extract_single_extended_vector(klines: &[Kline], idx: usize) -> Option<Vec<f64>> {
        if idx < 34 || idx >= klines.len() {
            return None;
        }

        let slice = &klines[0..=idx];
        let (matrix, n) = Self::build_extended(slice);
        if n == 0 {
            return None;
        }

        let f_dim = Self::num_extended_features();
        let start = (n - 1) * f_dim;
        let end = start + f_dim;
        Some(matrix[start..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_mock_klines(n: usize) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let change = ((i as f64 * 0.1).sin() * 1.5) + ((i as f64 * 0.05).cos() * 0.5);
            let open = price;
            let close = price + change;
            let high = open.max(close) + 0.5;
            let low = open.min(close) - 0.5;
            let volume = 1000.0 + (i as f64 * 10.0);
            klines.push(Kline {
                timestamp: (1600000000 + i * 60) as i64,
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
    fn test_extract_single_vector_matches_batch_matrix() {
        let klines = generate_mock_klines(100);
        let (full_matrix, total_rows) = FeatureMatrix::build(&klines);
        assert_eq!(total_rows, 100);
        let f_dim = FeatureMatrix::num_features();

        // Validar identidad numérica exacta para múltiples índices temporales
        for &idx in &[35, 50, 75, 99] {
            let single_vec = FeatureMatrix::extract_single_vector(&klines, idx)
                .expect("Debe extraer vector individual causal");
            let batch_slice = &full_matrix[idx * f_dim..(idx + 1) * f_dim];

            assert_eq!(single_vec.len(), f_dim);
            for f in 0..f_dim {
                let diff = (single_vec[f] - batch_slice[f]).abs();
                assert!(
                    diff < 1e-12,
                    "Discrepancia en feature {} ('{}') en idx {}: single={}, batch={}, diff={}",
                    f,
                    FeatureMatrix::FEATURE_NAMES[f],
                    idx,
                    single_vec[f],
                    batch_slice[f],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_extended_feature_matrix_consistency_and_dimensions() {
        let klines = generate_mock_klines(80);
        let (matrix_ext, rows) = FeatureMatrix::build_extended(&klines);
        assert_eq!(rows, 80);
        let f_dim_ext = FeatureMatrix::num_extended_features();
        assert_eq!(f_dim_ext, 14);
        assert_eq!(matrix_ext.len(), 80 * 14);

        for &idx in &[40, 60, 79] {
            let single_ext = FeatureMatrix::extract_single_extended_vector(&klines, idx)
                .expect("Debe extraer vector extendido");
            assert_eq!(single_ext.len(), 14);

            let batch_slice = &matrix_ext[idx * f_dim_ext..(idx + 1) * f_dim_ext];
            for f in 0..f_dim_ext {
                let diff = (single_ext[f] - batch_slice[f]).abs();
                assert!(
                    diff < 1e-12,
                    "Discrepancia extendida en feature {} ('{}') en idx {}: single={}, batch={}",
                    f,
                    FeatureMatrix::EXTENDED_FEATURE_NAMES[f],
                    idx,
                    single_ext[f],
                    batch_slice[f]
                );
            }
        }
    }
}

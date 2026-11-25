//! Generador Adaptativo de Rejillas de Umbrales Causal (Adaptive Threshold Grid)
//!
//! REGLA FINANCIERA: Analiza la distribución empírica de las predicciones recibidas
//! (Out-of-Fold o In-Sample) para construir una rejilla de umbrales adaptada a la escala,
//! varianza y asimetría real del estimador, garantizando que los candidatos exploren
//! regiones activas de la densidad de probabilidad sin quedar fuera de rango por contracción
//! de regularización o comisiones.

pub struct AdaptiveThresholdGrid;

impl AdaptiveThresholdGrid {
    /// Rejilla de respaldo por defecto cuando las predicciones carecen de varianza estadística
    pub const DEFAULT_FALLBACK_GRID: [f64; 7] = [
        0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050,
    ];

    pub const TARGET_QUANTILES: [f64; 9] = [0.15, 0.25, 0.35, 0.45, 0.60, 0.70, 0.80, 0.88, 0.94];

    pub const SIGMA_MULTIPLIERS: [f64; 6] = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00];

    /// Construye una rejilla simétrica de candidatos positivos tau > 0 adaptada a la escala de predicciones
    pub fn compute_symmetric_grid(predictions: &[f64]) -> Vec<f64> {
        let mut abs_vals: Vec<f64> = predictions
            .iter()
            .copied()
            .filter(|&p| p.is_finite() && p.abs() > 1e-7)
            .map(|p| p.abs())
            .collect();

        if abs_vals.len() < 10 {
            return Self::DEFAULT_FALLBACK_GRID.to_vec();
        }

        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = abs_vals.iter().sum::<f64>() / abs_vals.len() as f64;
        let variance = abs_vals
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / abs_vals.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-7 {
            return Self::DEFAULT_FALLBACK_GRID.to_vec();
        }

        let mut candidates = Vec::with_capacity(Self::TARGET_QUANTILES.len() + Self::SIGMA_MULTIPLIERS.len());

        // 1. Agregar cuantiles empíricos
        for &q in &Self::TARGET_QUANTILES {
            let val = Self::quantile_sorted(&abs_vals, q);
            if val > 1e-6 {
                candidates.push(val);
            }
        }

        // 2. Agregar múltiplos de la desviación estándar
        for &m in &Self::SIGMA_MULTIPLIERS {
            let val = m * std_dev;
            if val > 1e-6 {
                candidates.push(val);
            }
        }

        Self::sanitize_and_dedup_positive(&mut candidates)
    }

    /// Construye dos rejillas independientes (tau_long > 0, tau_short < 0) adaptadas a cada cola distributiva
    pub fn compute_asymmetric_grids(predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pos_vals: Vec<f64> = predictions
            .iter()
            .copied()
            .filter(|&p| p.is_finite() && p > 1e-7)
            .collect();

        let mut neg_mags: Vec<f64> = predictions
            .iter()
            .copied()
            .filter(|&p| p.is_finite() && p < -1e-7)
            .map(|p| p.abs())
            .collect();

        // Si alguna de las dos colas carece de observaciones suficientes, usamos la simétrica como base
        let long_candidates = if pos_vals.len() >= 10 {
            pos_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mean = pos_vals.iter().sum::<f64>() / pos_vals.len() as f64;
            let var = pos_vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / pos_vals.len() as f64;
            let std_dev = var.sqrt();

            let mut c = Vec::new();
            for &q in &Self::TARGET_QUANTILES {
                let v = Self::quantile_sorted(&pos_vals, q);
                if v > 1e-6 {
                    c.push(v);
                }
            }
            if std_dev > 1e-7 {
                for &m in &Self::SIGMA_MULTIPLIERS {
                    let v = m * std_dev;
                    if v > 1e-6 {
                        c.push(v);
                    }
                }
            }
            Self::sanitize_and_dedup_positive(&mut c)
        } else {
            Self::compute_symmetric_grid(predictions)
        };

        let short_mags = if neg_mags.len() >= 10 {
            neg_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mean = neg_mags.iter().sum::<f64>() / neg_mags.len() as f64;
            let var = neg_mags.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / neg_mags.len() as f64;
            let std_dev = var.sqrt();

            let mut c = Vec::new();
            for &q in &Self::TARGET_QUANTILES {
                let v = Self::quantile_sorted(&neg_mags, q);
                if v > 1e-6 {
                    c.push(v);
                }
            }
            if std_dev > 1e-7 {
                for &m in &Self::SIGMA_MULTIPLIERS {
                    let v = m * std_dev;
                    if v > 1e-6 {
                        c.push(v);
                    }
                }
            }
            Self::sanitize_and_dedup_positive(&mut c)
        } else {
            long_candidates.clone()
        };

        // Convertir short magnitudes a valores estrictamente negativos
        let short_candidates: Vec<f64> = short_mags.into_iter().map(|v| -v).collect();

        (long_candidates, short_candidates)
    }

    /// Calcula el cuantil continuo lineal sobre un slice ya ordenado ascendentemente
    fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }

        let q_clamped = q.clamp(0.0, 1.0);
        let index_f = q_clamped * (sorted.len() - 1) as f64;
        let lower = index_f.floor() as usize;
        let upper = index_f.ceil() as usize;
        let weight = index_f - lower as f64;

        if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    /// Ordena ascendentemente, elimina duplicados numéricos cercanos y asegura umbrales viables
    fn sanitize_and_dedup_positive(candidates: &mut Vec<f64>) -> Vec<f64> {
        candidates.retain(|v| v.is_finite() && *v > 1e-6);
        if candidates.is_empty() {
            return Self::DEFAULT_FALLBACK_GRID.to_vec();
        }

        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut deduped: Vec<f64> = Vec::with_capacity(candidates.len());
        for &val in candidates.iter() {
            if let Some(&last) = deduped.last() {
                // Descartar si está a menos de 5% de distancia relativa o 1e-5 absoluta
                let denom = if last > 1e-6 { last } else { 1e-6 };
                let rel_diff = (val - last).abs() / denom;
                let abs_diff = (val - last).abs();
                if rel_diff >= 0.05 && abs_diff >= 1e-5 {
                    deduped.push(val);
                }
            } else {
                deduped.push(val);
            }
        }

        if deduped.len() < 3 {
            // Si la deduplicación dejó muy pocos puntos, fusionar con escala de fallback proporcional
            let base = deduped.first().copied().unwrap_or(0.0005);
            return vec![base * 0.5, base, base * 1.5, base * 2.0, base * 3.0];
        }

        deduped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_predictions_returns_default_fallback() {
        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(&[]);
        assert_eq!(grid, AdaptiveThresholdGrid::DEFAULT_FALLBACK_GRID.to_vec());
    }

    #[test]
    fn test_constant_predictions_returns_fallback() {
        let preds = vec![0.0005; 100];
        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(&preds);
        assert!(!grid.is_empty());
        for &val in &grid {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_contracted_distribution_adapts_to_small_scale() {
        // Simular predicciones fuertemente contraídas por regularización L1/L2
        // rango: [-0.0006, +0.0006] con std_dev aprox 0.0003
        let mut preds = Vec::new();
        for i in 0..200 {
            let val = ((i as f64 - 100.0) / 100.0) * 0.0006;
            preds.push(val);
        }

        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(&preds);
        assert!(!grid.is_empty());

        // Verificar que los umbrales se adaptan al rango real (menores a 0.0010)
        let min_val = *grid.first().unwrap();
        let max_val = *grid.last().unwrap();
        assert!(min_val < 0.0004, "El umbral mínimo debe ser sensible ({})", min_val);
        assert!(max_val <= 0.0010, "El umbral máximo no debe escapar la distribución ({})", max_val);
    }

    #[test]
    fn test_asymmetric_distribution_adapts_both_sides() {
        // Asimetría: alcista débil (+0.0003) vs bajista fuerte (-0.0012)
        let mut preds = Vec::new();
        for _ in 0..100 {
            preds.push(0.0003);
            preds.push(-0.0012);
        }

        let (long_grid, short_grid) = AdaptiveThresholdGrid::compute_asymmetric_grids(&preds);
        assert!(!long_grid.is_empty());
        assert!(!short_grid.is_empty());

        for &l in &long_grid {
            assert!(l > 0.0);
        }
        for &s in &short_grid {
            assert!(s < 0.0);
        }
    }
}

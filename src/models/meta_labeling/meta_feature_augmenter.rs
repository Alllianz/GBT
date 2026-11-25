use super::meta_target_builder::MetaSample;

/// Aumentador de Características Causal para Meta-Modelos (Marcos López de Prado)
///
/// Enriquece el vector de características base $X_t \in \mathbb{R}^D$ con metadatos
/// causales derivados de la señal del modelo primario $\hat{y}_{\text{primary}, t}$:
/// 1. `raw_prediction`: Retorno esperado continuo predicho por el modelo primario $\hat{y}_{\text{primary}, t}$.
/// 2. `conviction_magnitude`: Magnitud de convicción primaria $|\hat{y}_{\text{primary}, t}|$.
/// 3. `threshold_excess`: Exceso sobre el umbral activado ($|\hat{y}_{\text{primary}, t}| - \tau_{\text{active}}$).
/// 4. `proposed_direction`: Dirección propuesta por el modelo primario ($+1.0$ Long, $-1.0$ Short, $0.0$ Neutral).
pub struct MetaFeatureAugmenter;

impl MetaFeatureAugmenter {
    /// Número de características sintéticas añadidas por el aumentador
    pub const NUM_AUGMENTED_FEATURES: usize = 4;

    /// Aumenta un vector individual de características correspondiente a un instante temporal $t$.
    pub fn augment_row(
        base_features: &[f64],
        primary_pred: f64,
        long_th: f64,
        short_th: f64,
    ) -> Vec<f64> {
        let mut augmented = Vec::with_capacity(base_features.len() + Self::NUM_AUGMENTED_FEATURES);
        augmented.extend_from_slice(base_features);

        let conviction = primary_pred.abs();
        let (excess, direction) = if primary_pred > long_th {
            ((primary_pred - long_th).max(0.0), 1.0)
        } else if primary_pred < short_th {
            ((short_th - primary_pred).max(0.0), -1.0)
        } else {
            (0.0, 0.0)
        };

        augmented.push(primary_pred);
        augmented.push(conviction);
        augmented.push(excess);
        augmented.push(direction);

        augmented
    }

    /// Construye una matriz plana `(N_active x (D + 4))` para el conjunto de muestras activas del meta-modelo.
    ///
    /// REGLA FINANCIERA: Solo extrae filas para las cuales el modelo primario generó una señal ($d_t \neq 0$).
    pub fn build_augmented_matrix_for_samples(
        base_matrix_flat: &[f64],
        base_dim: usize,
        samples: &[MetaSample],
        long_th: f64,
        short_th: f64,
    ) -> (Vec<f64>, usize) {
        let num_samples = samples.len();
        let new_dim = base_dim + Self::NUM_AUGMENTED_FEATURES;
        let mut out_matrix = Vec::with_capacity(num_samples * new_dim);

        for sample in samples {
            let row_start = sample.index * base_dim;
            let row_end = row_start + base_dim;

            if row_end <= base_matrix_flat.len() {
                let base_row = &base_matrix_flat[row_start..row_end];
                let augmented_row = Self::augment_row(base_row, sample.primary_prediction, long_th, short_th);
                out_matrix.extend_from_slice(&augmented_row);
            }
        }

        (out_matrix, new_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_feature_augmenter_row() {
        let base_feats = vec![1.0, 2.0, 3.0];
        let long_th = 0.002;
        let short_th = -0.002;

        // Long activado con 0.005
        let aug_long = MetaFeatureAugmenter::augment_row(&base_feats, 0.005, long_th, short_th);
        assert_eq!(aug_long.len(), 3 + MetaFeatureAugmenter::NUM_AUGMENTED_FEATURES);
        assert_eq!(aug_long[0], 1.0);
        assert_eq!(aug_long[1], 2.0);
        assert_eq!(aug_long[2], 3.0);
        assert_eq!(aug_long[3], 0.005); // primary_pred
        assert_eq!(aug_long[4], 0.005); // conviction
        assert!((aug_long[5] - 0.003).abs() < 1e-9); // excess = 0.005 - 0.002
        assert_eq!(aug_long[6], 1.0); // direction

        // Short activado con -0.007
        let aug_short = MetaFeatureAugmenter::augment_row(&base_feats, -0.007, long_th, short_th);
        assert_eq!(aug_short[3], -0.007);
        assert_eq!(aug_short[4], 0.007);
        assert!((aug_short[5] - 0.005).abs() < 1e-9); // excess = 0.007 - 0.002
        assert_eq!(aug_short[6], -1.0);

        // Neutral con 0.001
        let aug_neutral = MetaFeatureAugmenter::augment_row(&base_feats, 0.001, long_th, short_th);
        assert_eq!(aug_neutral[3], 0.001);
        assert_eq!(aug_neutral[4], 0.001);
        assert_eq!(aug_neutral[5], 0.0);
        assert_eq!(aug_neutral[6], 0.0);
    }
}

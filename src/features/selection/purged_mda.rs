use crate::features::PurgedKFold;
use crate::models::decision_tree::CartTree;
use crate::models::regularization::TreeRegularizationConfig;
use super::selection_config::FeatureSelectionConfig;
use super::xorshift::Xorshift64;

/// Resultado de Importancia Causal por Característica vía Purged MDA
#[derive(Debug, Clone, PartialEq)]
pub struct MdaScore {
    pub feature_index: usize,
    pub mean_mda: f64,
    pub std_mda: f64,
    pub t_stat: f64,
    pub is_informative: bool,
}

/// Evaluador de Importancia de Características por Disminución Media de Precisión
/// con Purga y Embargo (Purged Mean Decrease Accuracy / MSE - Marcos López de Prado)
///
/// Metodología:
/// 1. Para cada fold $k \in \{1, \dots, K\}$ de `PurgedKFold`:
///    a) Entrena un árbol CART regularizado en las muestras de entrenamiento purgadas y embargadas.
///    b) Evalúa la pérdida base fuera de muestra ($L_{\text{base}}^{(k)}$) en el fold de validación purgado.
///    c) Para cada característica $j$, permuta determinísticamente sus valores en el fold de prueba
///       utilizando PRNG Xorshift64 y mide la nueva pérdida permutada ($L_{\text{perm}, j}^{(k)}$).
///    d) Calcula el impacto causal: $\Delta L_j^{(k)} = L_{\text{perm}, j}^{(k)} - L_{\text{base}}^{(k)}$.
/// 2. Agrega entre todos los folds:
///    $\text{MDA}_j = \frac{1}{K} \sum_{k=1}^K \Delta L_j^{(k)}$.
///    Si $\text{MDA}_j \le 0$, la permutación de la variable no degrada el modelo (o incluso mejora el error),
///    lo que demuestra que la variable aporta ruido espurio y debe ser podada.
#[derive(Debug, Clone)]
pub struct PurgedMdaEvaluator {
    pub config: FeatureSelectionConfig,
}

impl PurgedMdaEvaluator {
    pub fn new(config: FeatureSelectionConfig) -> Self {
        Self { config }
    }

    /// Evalúa la importancia causal MDA de cada columna de la matriz de características
    pub fn evaluate(
        &self,
        features_flat: &[f64],
        targets: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Vec<MdaScore>, String> {
        if n_rows < 50 {
            return Err(format!(
                "Muestras insuficientes ({}) para evaluación Purged MDA (mínimo 50)",
                n_rows
            ));
        }
        if n_cols == 0 {
            return Err("Número de características debe ser mayor a 0".to_string());
        }
        if features_flat.len() < n_rows * n_cols {
            return Err("Dimensión de matriz de características inconsistente".to_string());
        }
        if targets.len() < n_rows {
            return Err("Longitud del vector de targets inconsistente".to_string());
        }

        let kf = PurgedKFold::new(
            self.config.n_splits,
            self.config.horizon,
            self.config.embargo_bars,
        );
        let folds = kf.split(n_rows)?;
        let num_folds = folds.len();

        let mut fold_mda_matrix = vec![vec![0.0; n_cols]; num_folds];

        let mut reg_config = TreeRegularizationConfig::default();
        reg_config.l2_lambda = 1.0;
        reg_config.min_child_weight = 3.0;

        for (fold_idx, fold) in folds.iter().enumerate() {
            if fold.train_indices.len() < 20 || fold.test_indices.is_empty() {
                continue;
            }

            let seed_k = self
                .config
                .random_seed
                .wrapping_add((fold_idx as u64).wrapping_mul(100_003));

            // 1. Entrenar árbol CART en muestras de train purgadas y embargadas
            let mut tree = CartTree::new_regularized(4, 4, reg_config, seed_k);
            tree.fit_indices(
                features_flat,
                targets,
                &fold.train_indices,
                n_cols,
                None,
            );

            // 2. Pérdida base MSE en fold de prueba
            let mut base_loss_sum = 0.0;
            for &test_idx in &fold.test_indices {
                let row = &features_flat[(test_idx * n_cols)..((test_idx + 1) * n_cols)];
                let pred = tree.predict(row);
                let diff = targets[test_idx] - pred;
                base_loss_sum += diff * diff;
            }
            let base_loss = base_loss_sum / (fold.test_indices.len() as f64);

            // 3. Permutar cada característica j y evaluar degradación
            for j in 0..n_cols {
                let mut perm_values: Vec<f64> = fold
                    .test_indices
                    .iter()
                    .map(|&idx| features_flat[idx * n_cols + j])
                    .collect();

                let perm_seed = seed_k.wrapping_add((j as u64 + 1).wrapping_mul(65_537));
                let mut rng = Xorshift64::new(perm_seed);
                rng.shuffle(&mut perm_values);

                let mut perm_loss_sum = 0.0;
                for (p, &test_idx) in fold.test_indices.iter().enumerate() {
                    let mut row_buf = features_flat[(test_idx * n_cols)..((test_idx + 1) * n_cols)].to_vec();
                    row_buf[j] = perm_values[p];
                    let pred = tree.predict(&row_buf);
                    let diff = targets[test_idx] - pred;
                    perm_loss_sum += diff * diff;
                }
                let perm_loss = perm_loss_sum / (fold.test_indices.len() as f64);

                let mda_fold_j = perm_loss - base_loss;
                fold_mda_matrix[fold_idx][j] = mda_fold_j;
            }
        }

        // 4. Estadísticas agregadas entre folds
        let k_f64 = num_folds as f64;
        let mut scores = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            let mda_vals: Vec<f64> = (0..num_folds).map(|k| fold_mda_matrix[k][j]).collect();
            let mean_mda = mda_vals.iter().sum::<f64>() / k_f64;

            let var_mda = if num_folds > 1 {
                mda_vals
                    .iter()
                    .map(|&v| {
                        let diff = v - mean_mda;
                        diff * diff
                    })
                    .sum::<f64>()
                    / (k_f64 - 1.0)
            } else {
                0.0
            };

            let std_mda = var_mda.sqrt();
            let se_mda = if num_folds > 1 {
                std_mda / (k_f64.sqrt())
            } else {
                1.0
            };

            let t_stat = if se_mda > 1e-12 {
                mean_mda / se_mda
            } else {
                0.0
            };

            let is_informative = mean_mda > self.config.min_mda_threshold;

            scores.push(MdaScore {
                feature_index: j,
                mean_mda,
                std_mda,
                t_stat,
                is_informative,
            });
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purged_mda_identifies_informative_vs_noise_feature() {
        let n_samples = 120;
        let n_cols = 2;
        let mut features = Vec::with_capacity(n_samples * n_cols);
        let mut targets = Vec::with_capacity(n_samples);

        let mut rng = Xorshift64::new(999);

        for i in 0..n_samples {
            let signal = (i as f64 * 0.1).sin();
            // Col 0: variable predictiva
            features.push(signal);
            // Col 1: variable de puro ruido
            let noise = ((rng.next_u64() % 1000) as f64) / 1000.0 - 0.5;
            features.push(noise);

            // Target depende principalmente de signal
            targets.push(signal * 2.0 + noise * 0.05);
        }

        let config = FeatureSelectionConfig::default()
            .with_n_splits(3)
            .with_horizon(2)
            .with_embargo_bars(4)
            .with_min_mda_threshold(0.0);

        let evaluator = PurgedMdaEvaluator::new(config);
        let scores = evaluator
            .evaluate(&features, &targets, n_samples, n_cols)
            .expect("Evaluación MDA debe ser exitosa");

        assert_eq!(scores.len(), 2);
        // La característica 0 (informativa) debe tener mayor MDA que la característica 1 (ruido)
        assert!(
            scores[0].mean_mda > scores[1].mean_mda,
            "Col 0 (señal) MDA={:.6} debe superar Col 1 (ruido) MDA={:.6}",
            scores[0].mean_mda,
            scores[1].mean_mda
        );
        assert!(scores[0].is_informative);
    }
}

use crate::features::PurgedKFold;
use crate::models::decision_tree::CartTree;
use crate::models::regularization::TreeRegularizationConfig;
use super::selection_config::FeatureSelectionConfig;
use super::xorshift::Xorshift64;

#[derive(Debug, Clone, PartialEq)]
pub struct MdaScore {
    pub feature_index: usize,
    pub mean_mda: f64,
    pub std_mda: f64,
    pub t_stat: f64,
    pub is_informative: bool,
}

#[derive(Debug, Clone)]
pub struct PurgedMdaEvaluator {
    pub config: FeatureSelectionConfig,
}

impl PurgedMdaEvaluator {
    pub fn new(config: FeatureSelectionConfig) -> Self {
        Self { config }
    }

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

        let mut fold_mda_rows: Vec<Vec<f64>> = Vec::with_capacity(num_folds);
        let mut evaluated_folds = 0usize;

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

            let mut tree = CartTree::new_regularized(4, 4, reg_config, seed_k);
            tree.fit_indices(
                features_flat,
                targets,
                &fold.train_indices,
                n_cols,
                None,
            );

            let mut base_loss_sum = 0.0;
            for &test_idx in &fold.test_indices {
                let row = &features_flat[(test_idx * n_cols)..((test_idx + 1) * n_cols)];
                let pred = tree.predict(row);
                let diff = targets[test_idx] - pred;
                base_loss_sum += diff * diff;
            }
            let base_loss = base_loss_sum / (fold.test_indices.len() as f64);

            let mut fold_row = vec![0.0; n_cols];
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
                    let mut row_buf =
                        features_flat[(test_idx * n_cols)..((test_idx + 1) * n_cols)].to_vec();
                    row_buf[j] = perm_values[p];
                    let pred = tree.predict(&row_buf);
                    let diff = targets[test_idx] - pred;
                    perm_loss_sum += diff * diff;
                }
                let perm_loss = perm_loss_sum / (fold.test_indices.len() as f64);

                fold_row[j] = perm_loss - base_loss;
            }

            fold_mda_rows.push(fold_row);
            evaluated_folds += 1;
        }

        if evaluated_folds == 0 {
            return Ok((0..n_cols)
                .map(|j| MdaScore {
                    feature_index: j,
                    mean_mda: 0.0,
                    std_mda: 0.0,
                    t_stat: 0.0,
                    is_informative: false,
                })
                .collect());
        }

        let k_f64 = evaluated_folds as f64;
        let mut scores = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            let mda_vals: Vec<f64> = fold_mda_rows.iter().map(|row| row[j]).collect();
            let mean_mda = mda_vals.iter().sum::<f64>() / k_f64;

            let var_mda = if evaluated_folds > 1 {
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
            let se_mda = if evaluated_folds > 1 {
                std_mda / k_f64.sqrt()
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
            features.push(signal);
            let noise = ((rng.next_u64() % 1000) as f64) / 1000.0 - 0.5;
            features.push(noise);

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
        assert!(
            scores[0].mean_mda > scores[1].mean_mda,
            "Col 0 (señal) MDA={:.6} debe superar Col 1 (ruido) MDA={:.6}",
            scores[0].mean_mda,
            scores[1].mean_mda
        );
        assert!(scores[0].is_informative);
    }

    #[test]
    fn test_purged_mda_skips_insufficient_folds_without_diluting_mean() {
        let n_samples = 50;
        let n_cols = 2;
        let mut features = Vec::with_capacity(n_samples * n_cols);
        let mut targets = Vec::with_capacity(n_samples);
        let mut rng = Xorshift64::new(42);

        for i in 0..n_samples {
            let signal = (i as f64 * 0.2).sin();
            features.push(signal);
            let noise = ((rng.next_u64() % 1000) as f64) / 1000.0 - 0.5;
            features.push(noise);
            targets.push(signal * 3.0);
        }

        let config = FeatureSelectionConfig::default()
            .with_n_splits(4)
            .with_horizon(10)
            .with_embargo_bars(10)
            .with_min_mda_threshold(0.0);

        let evaluator = PurgedMdaEvaluator::new(config);
        let scores = evaluator
            .evaluate(&features, &targets, n_samples, n_cols)
            .expect("Evaluación MDA debe manejar folds omitidos");
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.mean_mda.is_finite() && s.std_mda.is_finite() && s.t_stat.is_finite()));
    }
}

use rayon::prelude::*;
use crate::features::{FeatureMatrix, TripleBarrierLabeler};
use crate::models::decision_tree::CartTree;
use crate::models::model_config::ModelConfig;
use crate::models::regularization::{
    ColSubsampler, EarlyStoppingController, RowSubsampler, TreeRegularizationConfig,
};
use crate::types::Kline;

/// Implementación Nativa de Gradient Boosted Decision Trees (GBDT) Regularizada
///
/// Incorpora:
/// 1. Regularización analítica L1 (soft-thresholding) y L2 (Ridge) en cada hoja.
/// 2. Pre-poda analítica con ganancia de Taylor de segundo orden y peso mínimo de nodo hijo.
/// 3. Submuestreo estocástico de instancias por árbol (`RowSubsampler`).
/// 4. Submuestreo de características por árbol (`ColSubsampler`).
/// 5. Control causal de parada temprana (`EarlyStoppingController`) sin Look-Ahead Bias.
pub struct NativeGbdtModel {
    base_score: f64,
    trees: Vec<CartTree>,
    config: ModelConfig,
    reg_config: TreeRegularizationConfig,
    is_trained: bool,
    best_iteration: Option<usize>,
}

impl NativeGbdtModel {
    pub fn new(config: ModelConfig) -> Self {
        let reg_config = TreeRegularizationConfig::from_model_config(&config);
        Self {
            base_score: 0.0,
            trees: Vec::new(),
            config,
            reg_config,
            is_trained: false,
            best_iteration: None,
        }
    }

    pub fn with_regularization(mut self, reg_config: TreeRegularizationConfig) -> Self {
        self.reg_config = reg_config;
        self
    }

    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        const WARMUP: usize = 60;
        let n = train_klines.len();
        if n <= WARMUP {
            return Err("Se requieren más de 60 velas para entrenar Native GBDT".to_string());
        }

        let min_val_len = WARMUP + self.config.horizon + 10;
        if n >= 160 {
            let train_ratio = 0.80;
            let purge_bars = self.config.horizon + 1;
            let train_end = ((n as f64) * train_ratio).round() as usize;
            let val_start = train_end + purge_bars;

            if train_end >= WARMUP + self.config.horizon + 15
                && val_start < n
                && (n - val_start) >= min_val_len
            {
                let inner_train = &train_klines[0..train_end];
                let inner_val = &train_klines[val_start..n];
                return self.train_with_validation(inner_train, inner_val, 15);
            }
        }

        self.train_direct(train_klines)
    }

    pub fn train_direct(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        const WARMUP: usize = 60;
        if train_klines.len() <= WARMUP {
            return Err("Muestras insuficientes tras descontar el warmup".to_string());
        }
        let (matrix_flat, rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();

        let valid_end = rows.saturating_sub(self.config.horizon);
        if valid_end <= WARMUP || (valid_end - WARMUP) < 15 {
            return Err("Muestras insuficientes tras descontar warmup y horizonte".to_string());
        }

        let y = TripleBarrierLabeler::label_regression(
            train_klines,
            self.config.horizon,
            self.config.commission_rate,
        );
        let x_slice = &matrix_flat[(WARMUP * f_dim)..(valid_end * f_dim)];
        let y_slice = &y[WARMUP..valid_end];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    /// Entrenamiento causal directo a partir de una matriz plana y etiquetas pre-computadas
    pub fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        if y.is_empty() || num_features == 0 {
            return Err("Matriz de entrenamiento o etiquetas vacías".to_string());
        }
        let valid_rows = y.len();
        if x.len() != valid_rows * num_features {
            return Err(format!(
                "Dimensión de matriz inconsistente: esperados {} elementos, recibidos {}",
                valid_rows * num_features,
                x.len()
            ));
        }
        if valid_rows < 15 {
            return Err(format!(
                "Muestras insuficientes ({}) para entrenar Native GBDT (mínimo 15)",
                valid_rows
            ));
        }

        let (finite_sum, finite_count) = y
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
        if finite_count == 0 {
            return Err("Todas las etiquetas y son no finitas".to_string());
        }
        self.base_score = finite_sum / finite_count as f64;

        let mut current_preds = vec![self.base_score; valid_rows];
        self.trees.clear();

        let lr = self.config.learning_rate as f64;
        let num_trees = self.config.num_trees.min(200);

        for m in 0..num_trees {
            let residuals: Vec<f64> = y
                .iter()
                .zip(&current_preds)
                .map(|(&target, &pred)| if target.is_finite() { target - pred } else { 0.0 })
                .collect();

            let seed_m = 42u64.wrapping_add((m as u64).wrapping_mul(1_000_003));

            // 3. Submuestreo estocástico de filas
            let active_rows = if self.reg_config.subsample_ratio < 1.0 {
                RowSubsampler::sample_uniform(valid_rows, self.reg_config.subsample_ratio, seed_m)
            } else {
                (0..valid_rows).collect()
            };

            // 4. Submuestreo de características a nivel de árbol
            let tree_features = if self.reg_config.colsample_bytree < 1.0 {
                Some(ColSubsampler::sample_by_tree(
                    num_features,
                    self.reg_config.colsample_bytree,
                    seed_m.wrapping_add(1),
                ))
            } else {
                None
            };

            // 5. Ajustar árbol CART regularizado
            let mut tree = CartTree::new_regularized(
                self.config.max_depth,
                4,
                self.reg_config,
                seed_m.wrapping_add(2),
            );

            tree.fit_indices(
                x,
                &residuals,
                &active_rows,
                num_features,
                tree_features.as_deref(),
            );

            // 6. Actualizar predicciones con shrinkage: F_m(x) = F_{m-1}(x) + lr * h_m(x)
            current_preds
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, pred)| {
                    let row_feats = &x[(r * num_features)..((r + 1) * num_features)];
                    *pred += lr * tree.predict(row_feats);
                });

            self.trees.push(tree);
        }

        self.best_iteration = Some(self.trees.len().saturating_sub(1));
        self.is_trained = true;
        Ok(())
    }

    pub fn train_with_validation(
        &mut self,
        train_klines: &[Kline],
        val_klines: &[Kline],
        patience: usize,
    ) -> Result<(), String> {
        const WARMUP: usize = 60;
        if train_klines.len() <= WARMUP {
            return Err("Muestras de entrenamiento insuficientes tras warmup".to_string());
        }
        let (train_matrix, train_rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();
        let train_valid_end = train_rows.saturating_sub(self.config.horizon);

        if train_valid_end <= WARMUP || (train_valid_end - WARMUP) < 15 {
            return Err("Muestras de entrenamiento insuficientes".to_string());
        }

        let y_train = TripleBarrierLabeler::label_regression(
            train_klines,
            self.config.horizon,
            self.config.commission_rate,
        );
        let x_train_slice = &train_matrix[(WARMUP * f_dim)..(train_valid_end * f_dim)];
        let y_train_slice = &y_train[WARMUP..train_valid_end];
        let train_valid = y_train_slice.len();

        if val_klines.len() <= WARMUP {
            return self.train_direct(train_klines);
        }
        let (val_matrix, val_rows) = FeatureMatrix::build(val_klines);
        let val_valid_end = val_rows.saturating_sub(self.config.horizon);
        if val_valid_end <= WARMUP || (val_valid_end - WARMUP) < 10 {
            return self.train_direct(train_klines);
        }

        let y_val = TripleBarrierLabeler::label_regression(
            val_klines,
            self.config.horizon,
            self.config.commission_rate,
        );
        let x_val_slice = &val_matrix[(WARMUP * f_dim)..(val_valid_end * f_dim)];
        let y_val_slice = &y_val[WARMUP..val_valid_end];
        let val_valid = y_val_slice.len();

        let (finite_sum, finite_count) = y_train_slice
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold((0.0_f64, 0usize), |(s, c), v| (s + v, c + 1));
        if finite_count == 0 {
            return Err("Todas las etiquetas de entrenamiento son no finitas".to_string());
        }
        self.base_score = finite_sum / finite_count as f64;

        let mut train_preds = vec![self.base_score; train_valid];
        let mut val_preds = vec![self.base_score; val_valid];
        self.trees.clear();

        let lr = self.config.learning_rate as f64;
        let num_trees = self.config.num_trees.min(250);

        let mut early_stopping = EarlyStoppingController::new(patience, 1e-6, true);

        for m in 0..num_trees {
            let residuals: Vec<f64> = y_train_slice
                .iter()
                .zip(&train_preds)
                .map(|(&target, &pred)| if target.is_finite() { target - pred } else { 0.0 })
                .collect();

            let seed_m = 42u64.wrapping_add((m as u64).wrapping_mul(1_000_003));

            let active_rows = if self.reg_config.subsample_ratio < 1.0 {
                RowSubsampler::sample_uniform(train_valid, self.reg_config.subsample_ratio, seed_m)
            } else {
                (0..train_valid).collect()
            };

            let tree_features = if self.reg_config.colsample_bytree < 1.0 {
                Some(ColSubsampler::sample_by_tree(
                    f_dim,
                    self.reg_config.colsample_bytree,
                    seed_m.wrapping_add(1),
                ))
            } else {
                None
            };

            let mut tree = CartTree::new_regularized(
                self.config.max_depth,
                4,
                self.reg_config,
                seed_m.wrapping_add(2),
            );

            tree.fit_indices(
                x_train_slice,
                &residuals,
                &active_rows,
                f_dim,
                tree_features.as_deref(),
            );

            // Actualizar train predictions
            train_preds
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, pred)| {
                    let row_feats = &x_train_slice[(r * f_dim)..((r + 1) * f_dim)];
                    *pred += lr * tree.predict(row_feats);
                });

            let mut val_mse_sum = 0.0;
            let mut val_mse_count = 0usize;
            for r in 0..val_valid {
                let row_feats = &x_val_slice[(r * f_dim)..((r + 1) * f_dim)];
                val_preds[r] += lr * tree.predict(row_feats);
                if y_val_slice[r].is_finite() {
                    let diff = y_val_slice[r] - val_preds[r];
                    val_mse_sum += diff * diff;
                    val_mse_count += 1;
                }
            }
            let val_mse = if val_mse_count > 0 {
                val_mse_sum / val_mse_count as f64
            } else {
                f64::INFINITY
            };

            self.trees.push(tree);

            let should_stop = early_stopping.update(m, val_mse);
            if should_stop {
                // Truncar ensamble al mejor m*
                let best_m = early_stopping.best_iteration();
                self.trees.truncate(best_m + 1);
                self.best_iteration = Some(best_m);
                break;
            }
        }

        if self.best_iteration.is_none() {
            self.best_iteration = Some(early_stopping.best_iteration());
            self.trees.truncate(early_stopping.best_iteration() + 1);
        }

        self.is_trained = true;
        Ok(())
    }

    /// Inferencia vectorizada en paralelo evaluando todos los árboles con Rayon
    pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("Native GBDT no está entrenado".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(klines);
        let f_dim = FeatureMatrix::num_features();

        if rows == 0 || matrix_flat.is_empty() {
            return Ok(Vec::new());
        }

        self.predict_from_matrix(&matrix_flat, f_dim)
    }

    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("Native GBDT no está entrenado".to_string());
        }
        if x.is_empty() {
            return Ok(Vec::new());
        }
        if num_features == 0 || x.len() % num_features != 0 {
            return Err(format!(
                "Dimensión inconsistente en predict_from_matrix: {} no es múltiplo de {}",
                x.len(),
                num_features
            ));
        }

        let rows = x.len() / num_features;
        let lr = self.config.learning_rate as f64;
        let base = self.base_score;

        let preds: Vec<f64> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let row_feats = &x[(r * num_features)..((r + 1) * num_features)];
                let mut pred = base;
                for tree in &self.trees {
                    pred += lr * tree.predict(row_feats);
                }
                if pred.is_finite() { pred } else { 0.0 }
            })
            .collect();

        Ok(preds)
    }

    pub fn predict_single(&self, features: &[f64]) -> Result<f64, String> {
        if !self.is_trained {
            return Err("Native GBDT no está entrenado".to_string());
        }

        let lr = self.config.learning_rate as f64;
        let mut pred = self.base_score;

        for tree in &self.trees {
            pred += lr * tree.predict(features);
        }

        Ok(if pred.is_finite() { pred } else { 0.0 })
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn regularization_config(&self) -> &TreeRegularizationConfig {
        &self.reg_config
    }

    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }

    pub fn num_effective_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.config.set_thresholds(long_th, short_th);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.1).sin() * 2.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p + 0.1,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_native_gbdt_warmup_nonfinite_y_and_matrix_validation() {
        let mut model = NativeGbdtModel::new(ModelConfig::default());
        assert!(model.train(&sample_klines(60)).is_err());
        assert!(model.train_direct(&sample_klines(60)).is_err());
        assert!(model.train(&sample_klines(90)).is_ok());

        let f_dim = FeatureMatrix::num_features();
        assert!(model.predict_from_matrix(&vec![0.0; f_dim + 1], f_dim).is_err());

        let mut nan_model = NativeGbdtModel::new(ModelConfig::default());
        let rows = 20;
        let x = vec![0.1; rows * f_dim];
        let mut y = vec![0.02; rows];
        y[0] = f64::NAN;
        y[1] = f64::INFINITY;
        assert!(nan_model.train_from_matrix(&x, &y, f_dim).is_ok());
        let pred = nan_model.predict_single(&vec![0.1; f_dim]).unwrap();
        assert!(pred.is_finite());
    }
}

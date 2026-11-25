use rayon::prelude::*;
use crate::features::{FeatureMatrix, TripleBarrierLabeler};
use crate::models::model_config::ModelConfig;
use crate::types::Kline;
use super::cart_tree::CartTree;

pub struct DecisionTreeModel {
    tree: CartTree,
    config: ModelConfig,
    is_trained: bool,
}

impl DecisionTreeModel {
    pub fn new(config: ModelConfig) -> Self {
        let tree = CartTree::new(config.max_depth, 5, 1e-6);
        Self {
            tree,
            config,
            is_trained: false,
        }
    }

    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        const WARMUP: usize = 60;
        let n = train_klines.len();
        if n <= WARMUP {
            return Err("Se requieren más de 60 velas para entrenar el Decision Tree".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();

        let valid_end = rows.saturating_sub(self.config.horizon);
        if valid_end <= WARMUP || (valid_end - WARMUP) < 15 {
            return Err("Muestras insuficientes tras descontar warmup y horizonte".to_string());
        }

        let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon, self.config.commission_rate);
        let x_slice = &matrix_flat[(WARMUP * f_dim)..(valid_end * f_dim)];
        let y_slice = &y[WARMUP..valid_end];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    /// Entrena el árbol CART directamente sobre una matriz plana y etiquetas continuas
    pub fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        let valid_rows = y.len();
        if valid_rows < 15 {
            return Err("Muestras insuficientes para entrenar Decision Tree (mínimo 15)".to_string());
        }
        if x.len() != valid_rows * num_features {
            return Err(format!(
                "Dimensión inconsistente en Decision Tree: esperados {} elementos, recibidos {}",
                valid_rows * num_features,
                x.len()
            ));
        }

        self.tree.fit(x, y, num_features, None);
        self.is_trained = true;
        Ok(())
    }

    /// Inferencia vectorizada en paralelo evaluando el árbol con los 20 hilos
    pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("Modelo Decision Tree no entrenado".to_string());
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
            return Err("Modelo Decision Tree no entrenado".to_string());
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
        let preds: Vec<f64> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let row_feats = &x[(r * num_features)..((r + 1) * num_features)];
                let p = self.tree.predict(row_feats);
                if p.is_finite() { p } else { 0.0 }
            })
            .collect();

        Ok(preds)
    }

    pub fn predict_single(&self, features: &[f64]) -> Result<f64, String> {
        if !self.is_trained {
            return Err("Modelo Decision Tree no entrenado".to_string());
        }
        let p = self.tree.predict(features);
        Ok(if p.is_finite() { p } else { 0.0 })
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
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
    fn test_cart_model_warmup_and_predict_matrix_validation() {
        let mut model = DecisionTreeModel::new(ModelConfig::default());
        assert!(model.train(&sample_klines(60)).is_err());
        assert!(model.train(&sample_klines(90)).is_ok());
        let f_dim = FeatureMatrix::num_features();
        assert!(model.predict_from_matrix(&vec![0.0; f_dim + 1], f_dim).is_err());
    }
}

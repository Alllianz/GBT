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
        let n = train_klines.len();
        if n < 50 {
            return Err("Se requieren al menos 50 velas para entrenar el Decision Tree".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();

        let valid_rows = rows.saturating_sub(self.config.horizon);
        if valid_rows < 30 {
            return Err("Muestras insuficientes tras descontar el horizonte".to_string());
        }

        let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon, self.config.commission_rate);
        let x_slice = &matrix_flat[0..(valid_rows * f_dim)];
        let y_slice = &y[0..valid_rows];

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

    /// Inferencia vectorizada sobre matriz plana pre-computada [N x num_features]
    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("Modelo Decision Tree no entrenado".to_string());
        }
        if x.is_empty() || num_features == 0 {
            return Ok(Vec::new());
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

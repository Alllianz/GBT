use serde::{Deserialize, Serialize};
use crate::models::model_config::ModelConfig;

/// Configuración de Regularización Algorítmica y Estocástica para Árboles
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TreeRegularizationConfig {
    /// Penalización L1 sobre los pesos de las hojas (soft-thresholding)
    pub l1_alpha: f64,
    /// Penalización L2 sobre los pesos de las hojas (shrinkage cuadrático)
    pub l2_lambda: f64,
    /// Ganancia mínima requerida para autorizar una división (pre-poda analítica gamma)
    pub min_split_gain: f64,
    /// Suma mínima de hessianos requerida en cada nodo hijo
    pub min_child_weight: f64,
    /// Tasa de submuestreo de filas por árbol (stochastic bagging / subsampling)
    pub subsample_ratio: f64,
    /// Tasa de submuestreo de características por árbol
    pub colsample_bytree: f64,
    /// Tasa de submuestreo de características por nodo
    pub colsample_bynode: f64,
}

impl Default for TreeRegularizationConfig {
    fn default() -> Self {
        Self {
            l1_alpha: 0.001,
            l2_lambda: 1.0,
            min_split_gain: 0.0,
            min_child_weight: 1.0,
            subsample_ratio: 1.0,
            colsample_bytree: 1.0,
            colsample_bynode: 1.0,
        }
    }
}

impl TreeRegularizationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn institutional() -> Self {
        Self {
            l1_alpha: 0.005,
            l2_lambda: 2.0,
            min_split_gain: 1e-5,
            min_child_weight: 5.0,
            subsample_ratio: 0.8,
            colsample_bytree: 0.8,
            colsample_bynode: 0.8,
        }
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        Self {
            l1_alpha: config.l1_regularization as f64,
            l2_lambda: (config.l2_regularization as f64).max(1.0),
            min_split_gain: config.min_split_gain as f64,
            min_child_weight: 1.0,
            subsample_ratio: 0.85,
            colsample_bytree: 0.85,
            colsample_bynode: 1.0,
        }
    }

    pub fn with_l1(mut self, alpha: f64) -> Self {
        self.l1_alpha = alpha.max(0.0);
        self
    }

    pub fn with_l2(mut self, lambda: f64) -> Self {
        self.l2_lambda = lambda.max(0.0);
        self
    }

    pub fn with_min_split_gain(mut self, gain: f64) -> Self {
        self.min_split_gain = gain.max(0.0);
        self
    }

    pub fn with_min_child_weight(mut self, weight: f64) -> Self {
        self.min_child_weight = weight.max(0.0);
        self
    }

    pub fn with_subsample(mut self, ratio: f64) -> Self {
        self.subsample_ratio = ratio.clamp(0.05, 1.0);
        self
    }

    pub fn with_colsample_bytree(mut self, ratio: f64) -> Self {
        self.colsample_bytree = ratio.clamp(0.05, 1.0);
        self
    }

    pub fn with_colsample_bynode(mut self, ratio: f64) -> Self {
        self.colsample_bynode = ratio.clamp(0.05, 1.0);
        self
    }

    pub fn is_regularized(&self) -> bool {
        self.l1_alpha > 0.0
            || self.l2_lambda > 0.0
            || self.min_split_gain > 0.0
            || self.min_child_weight > 1.0
            || self.subsample_ratio < 1.0
            || self.colsample_bytree < 1.0
            || self.colsample_bynode < 1.0
    }
}

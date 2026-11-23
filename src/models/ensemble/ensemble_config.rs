use serde::{Deserialize, Serialize};

/// Configuración Institucional para el Ensamble Multi-Modelo Stacking / Blending
///
/// Define los hiperparámetros para:
/// 1. Validación cruzada purgada sobre cada modelo base para generar predicciones Out-of-Fold (OOF).
/// 2. Penalización Ridge L2 del meta-combinador lineal analítico.
/// 3. Restricción de proyección simplex / pesos no negativos normalizados.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Parámetro de regularización Ridge L2 ($\lambda$) en la resolución analítica $(P^T P + \lambda I)^{-1} P^T y$
    pub ridge_lambda: f64,

    /// Si es `true`, proyecta analíticamente los pesos sobre el simplex unitario ($w_m \ge 0, \sum w_m = 1$)
    /// evitando inversiones espurias de señales o apalancamiento sintético no causal
    pub non_negative_weights: bool,

    /// Número de particiones temporales continuas para Purged K-Fold CV
    pub n_splits: usize,

    /// Horizonte temporal de retornos en número de velas
    pub horizon: usize,

    /// Número de barras de embargo post-validación para eliminar memoria autoregresiva
    pub embargo_bars: usize,

    /// Tasa de comisión por operación (sin slippage por orden de Allianz)
    pub commission_rate: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1.0,
            non_negative_weights: true,
            n_splits: 5,
            horizon: 12,
            embargo_bars: 10,
            commission_rate: 0.0004,
        }
    }
}

impl EnsembleConfig {
    pub fn new(ridge_lambda: f64, non_negative_weights: bool) -> Self {
        Self {
            ridge_lambda,
            non_negative_weights,
            ..Default::default()
        }
    }

    pub fn with_cv_params(mut self, n_splits: usize, horizon: usize, embargo_bars: usize) -> Self {
        self.n_splits = n_splits.max(2);
        self.horizon = horizon.max(1);
        self.embargo_bars = embargo_bars;
        self
    }

    pub fn with_ridge_lambda(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda.max(1e-6);
        self
    }

    pub fn with_non_negative_weights(mut self, non_negative: bool) -> Self {
        self.non_negative_weights = non_negative;
        self
    }

    pub fn with_commission_rate(mut self, commission_rate: f64) -> Self {
        self.commission_rate = commission_rate.max(0.0);
        self
    }
}

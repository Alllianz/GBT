use serde::{Deserialize, Serialize};

/// Configuración Institucional para Meta-Labeling en 2 Etapas (Marcos López de Prado)
///
/// Define los hiperparámetros para:
/// 1. Validación cruzada purgada del modelo primario (Etapa 1).
/// 2. Extracción de muestras de meta-etiquetado sobre señales no neutrales.
/// 3. Entrenamiento y umbrales del modelo gatekeeper secundario (Etapa 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLabelConfig {
    /// Umbral mínimo de probabilidad del meta-modelo para autorizar la operación primaria
    /// (ej. 0.55; si P(win | x) < confidence_threshold, la señal primaria se descarta como falso positivo)
    pub confidence_threshold: f64,

    /// Número de particiones temporales continuas para Purged K-Fold CV en Etapa 1
    pub n_splits: usize,

    /// Horizonte temporal de retornos en número de velas
    pub horizon: usize,

    /// Número de barras de embargo post-validación para prevenir fuga autoregresiva
    pub embargo_bars: usize,

    /// Penalización L2 (Ridge) en hojas del estimador de probabilidad secundario
    pub l2_regularization: f64,

    /// Si es `true`, modula causalmente el `signal_strength` en función de la probabilidad meta estimada
    pub include_primary_strength: bool,

    /// Tasa de comisión por operación (ej. 0.0004 para Binance VIP0; sin slippage por orden de Allianz)
    pub commission_rate: f64,

    /// Número de árboles en el ensamble regularizado del Gatekeeper secundario
    pub num_meta_trees: usize,

    /// Profundidad máxima de los árboles del Gatekeeper secundario
    pub meta_max_depth: usize,

    /// Tasa de aprendizaje (shrinkage) del Gatekeeper secundario
    pub meta_learning_rate: f64,

    /// Mínimo número de muestras requeridas en cada hoja del Gatekeeper secundario
    pub min_samples_leaf: usize,
}

impl Default for MetaLabelConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.55,
            n_splits: 5,
            horizon: 12,
            embargo_bars: 10,
            l2_regularization: 1.0,
            include_primary_strength: true,
            commission_rate: 0.0004,
            num_meta_trees: 50,
            meta_max_depth: 3,
            meta_learning_rate: 0.05,
            min_samples_leaf: 5,
        }
    }
}

impl MetaLabelConfig {
    pub fn new(confidence_threshold: f64, n_splits: usize, horizon: usize) -> Self {
        Self {
            confidence_threshold,
            n_splits,
            horizon,
            ..Default::default()
        }
    }

    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.50, 0.99);
        self
    }

    pub fn with_splits_and_embargo(mut self, n_splits: usize, embargo_bars: usize) -> Self {
        self.n_splits = n_splits.max(2);
        self.embargo_bars = embargo_bars;
        self
    }

    pub fn with_meta_boosting(
        mut self,
        num_trees: usize,
        max_depth: usize,
        learning_rate: f64,
        l2_reg: f64,
    ) -> Self {
        self.num_meta_trees = num_trees.max(1);
        self.meta_max_depth = max_depth.max(1);
        self.meta_learning_rate = learning_rate.clamp(0.001, 1.0);
        self.l2_regularization = l2_reg.max(0.0);
        self
    }

    pub fn with_commission_rate(mut self, commission_rate: f64) -> Self {
        self.commission_rate = commission_rate.max(0.0);
        self
    }
}

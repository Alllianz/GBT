/// Configuración de Hiperparámetros para Selección y Poda Causal de Características
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSelectionConfig {
    /// Umbral máximo de correlación absoluta (|ρ|) entre variables antes de podar colinealidad (ej. 0.85)
    pub max_correlation_threshold: f64,
    /// Umbral mínimo de importancia MDA (Mean Decrease Accuracy/MSE) para retener una característica (ej. 0.0)
    pub min_mda_threshold: f64,
    /// Número de particiones temporales continuas para Purged K-Fold Cross Validation
    pub n_splits: usize,
    /// Horizonte de maduración de la etiqueta para purga causal anterior (t + horizon)
    pub horizon: usize,
    /// Número de barras de enfriamiento para el embargo temporal posterior
    pub embargo_bars: usize,
    /// Semilla pseudoaleatoria determinista para el PRNG Xorshift64
    pub random_seed: u64,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            max_correlation_threshold: 0.85,
            min_mda_threshold: 0.0,
            n_splits: 5,
            horizon: 5,
            embargo_bars: 10,
            random_seed: 42,
        }
    }
}

impl FeatureSelectionConfig {
    pub fn new(
        max_correlation_threshold: f64,
        min_mda_threshold: f64,
        n_splits: usize,
        horizon: usize,
        embargo_bars: usize,
        random_seed: u64,
    ) -> Self {
        Self {
            max_correlation_threshold: max_correlation_threshold.clamp(0.05, 0.99),
            min_mda_threshold,
            n_splits: n_splits.max(2),
            horizon: horizon.max(1),
            embargo_bars,
            random_seed,
        }
    }

    pub fn with_max_correlation_threshold(mut self, threshold: f64) -> Self {
        self.max_correlation_threshold = threshold.clamp(0.05, 0.99);
        self
    }

    pub fn with_min_mda_threshold(mut self, threshold: f64) -> Self {
        self.min_mda_threshold = threshold;
        self
    }

    pub fn with_n_splits(mut self, n_splits: usize) -> Self {
        self.n_splits = n_splits.max(2);
        self
    }

    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon.max(1);
        self
    }

    pub fn with_embargo_bars(mut self, embargo_bars: usize) -> Self {
        self.embargo_bars = embargo_bars;
        self
    }

    pub fn with_random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_selection_config_defaults() {
        let config = FeatureSelectionConfig::default();
        assert_eq!(config.max_correlation_threshold, 0.85);
        assert_eq!(config.min_mda_threshold, 0.0);
        assert_eq!(config.n_splits, 5);
        assert_eq!(config.horizon, 5);
        assert_eq!(config.embargo_bars, 10);
        assert_eq!(config.random_seed, 42);
    }

    #[test]
    fn test_feature_selection_config_builder() {
        let config = FeatureSelectionConfig::default()
            .with_max_correlation_threshold(0.75)
            .with_min_mda_threshold(0.001)
            .with_n_splits(4)
            .with_horizon(3)
            .with_embargo_bars(8)
            .with_random_seed(12345);

        assert_eq!(config.max_correlation_threshold, 0.75);
        assert_eq!(config.min_mda_threshold, 0.001);
        assert_eq!(config.n_splits, 4);
        assert_eq!(config.horizon, 3);
        assert_eq!(config.embargo_bars, 8);
        assert_eq!(config.random_seed, 12345);
    }
}

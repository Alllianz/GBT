use crate::types::Timeframe;

/// Configuración institucional para Validación Cruzada Purgada con Embargo Temporal (Purged K-Fold CV)
///
/// Implementa los principios de causalidad temporal estricta de Marcos López de Prado (2018):
/// 1. `n_splits`: Número de particiones temporales contiguas.
/// 2. `horizon`: Ventana de purga anterior (evita fuga de retornos futuros en las etiquetas).
/// 3. `embargo_bars`: Ventana de enfriamiento posterior (evita fuga por memoria autorregresiva).
#[derive(Debug, Clone)]
pub struct PurgedCvConfig {
    /// Número de splits / particiones (por defecto 5)
    pub n_splits: usize,
    /// Horizonte de purga en barras hacia adelante (debe ser >= horizonte de la etiqueta)
    pub horizon: usize,
    /// Período de embargo en barras tras la partición de prueba
    pub embargo_bars: usize,
    pub commission_rate: f64,
    pub timeframe: Option<Timeframe>,
    pub use_vol_scaled_labels: bool,
}

impl Default for PurgedCvConfig {
    fn default() -> Self {
        Self {
            n_splits: 5,
            horizon: 10,
            embargo_bars: 30,
            commission_rate: 0.0005,
            timeframe: None,
            use_vol_scaled_labels: false,
        }
    }
}

impl PurgedCvConfig {
    pub fn new(n_splits: usize, horizon: usize, embargo_bars: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            horizon: horizon.max(1),
            embargo_bars,
            commission_rate: 0.0005,
            timeframe: None,
            use_vol_scaled_labels: false,
        }
    }

    pub fn with_commission_rate(mut self, commission_rate: f64) -> Self {
        self.commission_rate = commission_rate;
        self
    }

    pub fn with_timeframe(mut self, timeframe: Option<Timeframe>) -> Self {
        self.timeframe = timeframe;
        self
    }

    pub fn with_vol_scaled_labels(mut self, use_vol_scaled_labels: bool) -> Self {
        self.use_vol_scaled_labels = use_vol_scaled_labels;
        self
    }
}

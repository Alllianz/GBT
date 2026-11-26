use crate::types::{HedgeSignal, Kline, Signal};

/// Trait institucional para cualquier estrategia o modelo de decisión.
/// REGLA FINANCIERA FUNDAMENTAL:
/// Las implementaciones tienen estrictamente prohibido acceder a velas en índices > current_idx.
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;

    /// Genera una señal basada exclusivamente en la información disponible hasta current_idx
    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal;

    fn evaluate_hedge(&mut self, history: &[Kline], current_idx: usize) -> HedgeSignal {
        HedgeSignal::from_net_signal(self.evaluate(history, current_idx))
    }

    fn sync_open_legs(&mut self, _long_open: bool, _short_open: bool) {}

    /// Limpia cualquier caché interno de predicciones o estado entre ejecuciones de backtest (ej. IS vs OOS)
    fn clear_cache(&mut self) {}
}

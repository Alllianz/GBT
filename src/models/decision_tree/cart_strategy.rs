use crate::backtest::Strategy;
use crate::types::{Kline, Signal, SignalDirection};
use super::cart_model::DecisionTreeModel;

pub struct DecisionTreeStrategy {
    name: String,
    model: DecisionTreeModel,
    cached_predictions: Option<Vec<f64>>,
}

impl DecisionTreeStrategy {
    pub fn new(model: DecisionTreeModel) -> Self {
        Self {
            name: "DecisionTree_CART_Strategy".to_string(),
            model,
            cached_predictions: None,
        }
    }

    pub fn set_cached_predictions(&mut self, preds: Vec<f64>) {
        self.cached_predictions = Some(preds);
    }

    pub fn clear_cache(&mut self) {
        self.cached_predictions = None;
    }
}

impl Strategy for DecisionTreeStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn clear_cache(&mut self) {
        self.cached_predictions = None;
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        let ts = history[current_idx].timestamp;

        if !self.model.is_trained() || current_idx < 35 {
            return Signal::flat(ts);
        }

        let predicted_return = if let Some(ref preds) = self.cached_predictions {
            if current_idx < preds.len() {
                preds[current_idx]
            } else {
                match crate::features::FeatureMatrix::extract_single_vector(history, current_idx) {
                    Some(f) => self.model.predict_single(&f).unwrap_or(0.0),
                    None => return Signal::flat(ts),
                }
            }
        } else {
            match crate::features::FeatureMatrix::extract_single_vector(history, current_idx) {
                Some(f) => self.model.predict_single(&f).unwrap_or(0.0),
                None => return Signal::flat(ts),
            }
        };

        let long_th = self.model.config().long_threshold;
        let short_th = self.model.config().short_threshold;

        if predicted_return > long_th {
            let divisor = (long_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { predicted_return / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Long, strength, ts)
        } else if predicted_return < short_th {
            let divisor = (short_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { predicted_return.abs() / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            Signal::flat(ts)
        }
    }
}

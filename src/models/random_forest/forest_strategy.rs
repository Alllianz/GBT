use crate::backtest::Strategy;
use crate::types::{Kline, Signal, SignalDirection};
use super::forest_model::RandomForestModel;

pub struct RandomForestStrategy {
    name: String,
    model: RandomForestModel,
    cached_predictions: Option<Vec<f64>>,
    last_direction: SignalDirection,
}

impl RandomForestStrategy {
    pub fn new(model: RandomForestModel) -> Self {
        Self {
            name: "RandomForest_Strategy".to_string(),
            model,
            cached_predictions: None,
            last_direction: SignalDirection::Flat,
        }
    }

    pub fn set_cached_predictions(&mut self, preds: Vec<f64>) {
        self.cached_predictions = Some(preds);
        self.last_direction = SignalDirection::Flat;
    }

    pub fn clear_cache(&mut self) {
        self.cached_predictions = None;
        self.last_direction = SignalDirection::Flat;
    }
}

impl Strategy for RandomForestStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn clear_cache(&mut self) {
        self.cached_predictions = None;
        self.last_direction = SignalDirection::Flat;
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        let ts = history[current_idx].timestamp;

        if !self.model.is_trained() || current_idx < 35 {
            self.last_direction = SignalDirection::Flat;
            return Signal::flat(ts);
        }

        let predicted_return = if let Some(ref preds) = self.cached_predictions {
            if current_idx < preds.len() {
                preds[current_idx]
            } else {
                match crate::features::FeatureMatrix::extract_single_vector(history, current_idx) {
                    Some(f) => self.model.predict_single(&f).unwrap_or(0.0),
                    None => {
                        self.last_direction = SignalDirection::Flat;
                        return Signal::flat(ts);
                    }
                }
            }
        } else {
            match crate::features::FeatureMatrix::extract_single_vector(history, current_idx) {
                Some(f) => self.model.predict_single(&f).unwrap_or(0.0),
                None => {
                    self.last_direction = SignalDirection::Flat;
                    return Signal::flat(ts);
                }
            }
        };

        let long_th = self.model.config().long_threshold;
        let short_th = self.model.config().short_threshold;

        if predicted_return > long_th {
            self.last_direction = SignalDirection::Long;
            let divisor = (long_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { predicted_return / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Long, strength, ts)
        } else if predicted_return < short_th {
            self.last_direction = SignalDirection::Short;
            let divisor = (short_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { predicted_return.abs() / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Short, strength, ts)
        } else if self.last_direction == SignalDirection::Long && predicted_return > 0.0 {
            let strength = (predicted_return / long_th.max(1e-9)).clamp(0.01, 1.0);
            Signal::new(SignalDirection::Long, strength, ts)
        } else if self.last_direction == SignalDirection::Short && predicted_return < 0.0 {
            let strength = (predicted_return.abs() / short_th.abs().max(1e-9)).clamp(0.01, 1.0);
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            self.last_direction = SignalDirection::Flat;
            Signal::flat(ts)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_config::ModelConfig;

    fn sample_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.1;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p,
                    volume: 100.0,
                }
            })
            .collect()
    }

    fn sample_trained_model() -> RandomForestModel {
        let mut model = RandomForestModel::new(ModelConfig::default());
        let n = 20;
        let f_dim = crate::features::FeatureMatrix::num_features();
        let x: Vec<f64> = (0..n * f_dim).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        model.train_from_matrix(&x, &y, f_dim).unwrap();
        model
    }

    #[test]
    fn test_random_forest_strategy_warmup_and_untrained() {
        let model = RandomForestModel::new(ModelConfig::default());
        let mut strat = RandomForestStrategy::new(model);
        let klines = sample_klines(40);
        let sig = strat.evaluate(&klines, 10);
        assert_eq!(sig.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);
    }

    #[test]
    fn test_random_forest_strategy_long_hysteresis() {
        let model = sample_trained_model();
        let mut strat = RandomForestStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        preds[37] = 0.001;
        preds[38] = -0.0001;
        preds[39] = 0.001;
        strat.set_cached_predictions(preds);

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Long);
        assert_eq!(strat.last_direction, SignalDirection::Long);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Long);
        let expected_strength = (0.001 / 0.002f64).clamp(0.01, 1.0);
        assert!((sig37.strength - expected_strength).abs() < 1e-6);
        assert_eq!(strat.last_direction, SignalDirection::Long);

        let sig38 = strat.evaluate(&klines, 38);
        assert_eq!(sig38.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);

        let sig39 = strat.evaluate(&klines, 39);
        assert_eq!(sig39.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);
    }

    #[test]
    fn test_random_forest_strategy_short_hysteresis() {
        let model = sample_trained_model();
        let mut strat = RandomForestStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = -0.005;
        preds[37] = -0.001;
        preds[38] = 0.0001;
        preds[39] = -0.001;
        strat.set_cached_predictions(preds);

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Short);
        assert_eq!(strat.last_direction, SignalDirection::Short);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Short);
        let expected_strength = (0.001 / 0.002f64).clamp(0.01, 1.0);
        assert!((sig37.strength - expected_strength).abs() < 1e-6);
        assert_eq!(strat.last_direction, SignalDirection::Short);

        let sig38 = strat.evaluate(&klines, 38);
        assert_eq!(sig38.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);

        let sig39 = strat.evaluate(&klines, 39);
        assert_eq!(sig39.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);
    }

    #[test]
    fn test_random_forest_strategy_cache_reset_clears_direction() {
        let model = sample_trained_model();
        let mut strat = RandomForestStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        strat.set_cached_predictions(preds.clone());

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Long);
        assert_eq!(strat.last_direction, SignalDirection::Long);

        strat.clear_cache();
        assert_eq!(strat.last_direction, SignalDirection::Flat);

        preds[37] = 0.001;
        strat.set_cached_predictions(preds);
        assert_eq!(strat.last_direction, SignalDirection::Flat);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction, SignalDirection::Flat);
    }
}

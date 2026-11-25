use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub num_trees: usize,
    pub max_depth: usize,
    pub learning_rate: f32,
    pub l1_regularization: f32,
    pub l2_regularization: f32,
    pub min_split_gain: f32,
    pub horizon: usize,
    pub long_threshold: f64,
    pub short_threshold: f64,
    pub commission_rate: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_trees: 100,
            max_depth: 4,
            learning_rate: 0.05,
            l1_regularization: 0.001,
            l2_regularization: 0.01,
            min_split_gain: 0.0,
            horizon: 5,
            long_threshold: 0.002,
            short_threshold: -0.002,
            commission_rate: 0.0005,
        }
    }
}

impl ModelConfig {
    pub fn fast() -> Self {
        Self {
            num_trees: 50,
            max_depth: 3,
            learning_rate: 0.08,
            ..Default::default()
        }
    }

    pub fn institutional() -> Self {
        Self {
            num_trees: 150,
            max_depth: 5,
            learning_rate: 0.03,
            l1_regularization: 0.005,
            l2_regularization: 0.02,
            ..Default::default()
        }
    }

    pub fn for_timeframe(tf: crate::types::Timeframe) -> Self {
        let l2_regularization = match tf {
            crate::types::Timeframe::D1 => 0.18,
            crate::types::Timeframe::H4 => 0.16,
            crate::types::Timeframe::H1 => 0.16,
            _ => 0.10,
        };
        let num_trees = match tf {
            crate::types::Timeframe::D1 => 30,
            crate::types::Timeframe::H4 => 35,
            crate::types::Timeframe::H1 => 35,
            _ => 40,
        };
        let (max_depth, learning_rate) = (2, 0.02);
        let horizon = match tf {
            crate::types::Timeframe::D1 => 5,
            crate::types::Timeframe::H4 => 12,
            crate::types::Timeframe::H1 => 12,
            _ => 16,
        };
        Self {
            num_trees,
            max_depth,
            learning_rate,
            l1_regularization: 0.01,
            l2_regularization,
            horizon,
            ..Self::institutional()
        }
    }

    pub fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.long_threshold = long_th;
        self.short_threshold = short_th;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Timeframe;

    #[test]
    fn test_for_timeframe_presets() {
        let base = ModelConfig::institutional();

        let d1 = ModelConfig::for_timeframe(Timeframe::D1);
        assert_eq!(d1.max_depth, 2);
        assert_eq!(d1.num_trees, 30);
        assert!((d1.learning_rate - 0.02).abs() < 1e-6);
        assert!((d1.l1_regularization - 0.01).abs() < 1e-6);
        assert!((d1.l2_regularization - 0.18).abs() < 1e-6);
        assert_eq!(d1.horizon, 5);
        assert!((d1.commission_rate - base.commission_rate).abs() < 1e-12);
        assert!((d1.long_threshold - base.long_threshold).abs() < 1e-12);
        assert!((d1.short_threshold - base.short_threshold).abs() < 1e-12);

        let h4 = ModelConfig::for_timeframe(Timeframe::H4);
        assert_eq!(h4.max_depth, 2);
        assert_eq!(h4.num_trees, 35);
        assert!((h4.learning_rate - 0.02).abs() < 1e-6);
        assert!((h4.l1_regularization - 0.01).abs() < 1e-6);
        assert!((h4.l2_regularization - 0.16).abs() < 1e-6);
        assert_eq!(h4.horizon, 12);

        let h1 = ModelConfig::for_timeframe(Timeframe::H1);
        assert_eq!(h1.max_depth, 2);
        assert_eq!(h1.num_trees, 35);
        assert!((h1.learning_rate - 0.02).abs() < 1e-6);
        assert!((h1.l2_regularization - 0.16).abs() < 1e-6);
        assert_eq!(h1.horizon, 12);

        let m15 = ModelConfig::for_timeframe(Timeframe::M15);
        assert!((m15.l2_regularization - 0.10).abs() < 1e-6);
        assert_eq!(m15.horizon, 16);
        assert_eq!(m15.max_depth, 2);
        assert_eq!(m15.num_trees, 40);
    }
}

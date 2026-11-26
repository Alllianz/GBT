use crate::models::ModelConfig;
use crate::types::Timeframe;

#[derive(Clone)]
pub struct ModelGroupSpec {
    pub label: String,
    pub model_cfg: ModelConfig,
    pub secondary_model_cfg: Option<ModelConfig>,
    pub primary_weight: f64,
}

impl ModelGroupSpec {
    pub fn parse_timeframe(tf_str: &str) -> Result<Timeframe, String> {
        let tf: Timeframe = tf_str.parse()?;
        match tf {
            Timeframe::H4 | Timeframe::H1 => Ok(tf),
            _ => Err(format!(
                "Unsupported timeframe '{}' for evolve-gbt (expected 4H or 1H)",
                tf_str
            )),
        }
    }

    fn build_cfg(
        tf: Timeframe,
        horizon: usize,
        num_trees: usize,
        max_depth: usize,
        learning_rate: f32,
        l2_regularization: f32,
    ) -> ModelConfig {
        let mut model_cfg = ModelConfig::for_timeframe(tf);
        model_cfg.horizon = horizon;
        model_cfg.num_trees = num_trees;
        model_cfg.max_depth = max_depth;
        model_cfg.learning_rate = learning_rate;
        model_cfg.l2_regularization = l2_regularization;
        model_cfg
    }

    pub fn build_model_groups(tf: Timeframe) -> Vec<Self> {
        vec![Self {
            label: "M0_BASE".to_string(),
            model_cfg: Self::build_cfg(tf, 12, 35, 2, 0.020, 0.16),
            secondary_model_cfg: None,
            primary_weight: 1.0,
        }]
    }
}

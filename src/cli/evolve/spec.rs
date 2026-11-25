use crate::models::ModelConfig;
use crate::types::Timeframe;

#[derive(Clone)]
pub struct ModelGroupSpec {
    pub label: String,
    pub model_cfg: ModelConfig,
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

    pub fn build_model_groups(tf: Timeframe) -> Vec<Self> {
        let mut base = ModelConfig::for_timeframe(tf);
        base.horizon = 12;
        base.num_trees = 35;
        base.max_depth = 2;
        base.learning_rate = 0.02;
        base.l2_regularization = 0.16;

        vec![Self {
            label: "M0_BASE".to_string(),
            model_cfg: base,
        }]
    }
}

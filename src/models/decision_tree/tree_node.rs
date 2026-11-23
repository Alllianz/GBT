use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        gain: f64,
    },
}

impl TreeNode {
    pub fn predict(&self, features: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                let val = features.get(*feature_idx).copied().unwrap_or(0.0);
                if val <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

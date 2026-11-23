use serde::{Deserialize, Serialize};
use crate::models::regularization::{
    ColSubsampler, LeafPenaltyCalculator, TreeRegularizationConfig,
};
use super::regularized_split_finder::RegularizedSplitFinder;
use super::split_finder::SplitFinder;
use super::tree_node::TreeNode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartTree {
    pub root: Option<TreeNode>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    #[serde(default)]
    pub feature_subsample: Option<usize>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub regularization: TreeRegularizationConfig,
}

impl CartTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize, min_gain: f64) -> Self {
        let mut reg = TreeRegularizationConfig::default();
        reg.min_split_gain = min_gain;
        Self {
            root: None,
            max_depth,
            min_samples_leaf,
            min_gain,
            feature_subsample: None,
            seed: 0,
            regularization: reg,
        }
    }

    pub fn new_with_subsample(
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        feature_subsample: Option<usize>,
        seed: u64,
    ) -> Self {
        let mut reg = TreeRegularizationConfig::default();
        reg.min_split_gain = min_gain;
        Self {
            root: None,
            max_depth,
            min_samples_leaf,
            min_gain,
            feature_subsample,
            seed,
            regularization: reg,
        }
    }

    pub fn new_regularized(
        max_depth: usize,
        min_samples_leaf: usize,
        regularization: TreeRegularizationConfig,
        seed: u64,
    ) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_leaf,
            min_gain: regularization.min_split_gain,
            feature_subsample: None,
            seed,
            regularization,
        }
    }

    pub fn with_feature_subsample(mut self, feature_subsample: Option<usize>, seed: u64) -> Self {
        self.feature_subsample = feature_subsample;
        self.seed = seed;
        self
    }

    pub fn with_regularization(mut self, regularization: TreeRegularizationConfig) -> Self {
        self.min_gain = regularization.min_split_gain;
        self.regularization = regularization;
        self
    }

    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[f64],
        num_features: usize,
        allowed_features: Option<&[usize]>,
    ) {
        let n_samples = y.len();
        if n_samples == 0 {
            return;
        }

        let all_indices: Vec<usize> = (0..n_samples).collect();
        let root = Self::build_tree(
            x,
            y,
            &all_indices,
            num_features,
            allowed_features,
            0,
            self.max_depth,
            self.min_samples_leaf,
            &self.regularization,
            self.feature_subsample,
            self.seed,
        );
        self.root = Some(root);
    }

    pub fn fit_indices(
        &mut self,
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        num_features: usize,
        allowed_features: Option<&[usize]>,
    ) {
        if indices.is_empty() {
            return;
        }

        let root = Self::build_tree(
            x,
            y,
            indices,
            num_features,
            allowed_features,
            0,
            self.max_depth,
            self.min_samples_leaf,
            &self.regularization,
            self.feature_subsample,
            self.seed,
        );
        self.root = Some(root);
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        num_features: usize,
        allowed_features: Option<&[usize]>,
        current_depth: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        reg: &TreeRegularizationConfig,
        feature_subsample: Option<usize>,
        node_seed: u64,
    ) -> TreeNode {
        let n_samples = indices.len();
        let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();

        // Asignación de peso en la hoja mediante contracción analítica L1/L2
        let leaf_value = LeafPenaltyCalculator::compute_leaf_weight_sum(
            sum_y,
            n_samples,
            reg.l1_alpha,
            reg.l2_lambda,
        );

        if current_depth >= max_depth || n_samples < min_samples_leaf * 2 {
            return TreeNode::Leaf {
                value: leaf_value,
                samples: n_samples,
            };
        }

        let mut rng_state = if node_seed == 0 {
            0x853c49e6748fea9b
        } else {
            node_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        };

        // 1. Submuestreo dinámico de variables por nodo (Random Forest clásico o colsample_bynode)
        let node_features: Option<Vec<usize>> = if let Some(m_try) = feature_subsample {
            if m_try < num_features {
                let mut pool: Vec<usize> = match allowed_features {
                    Some(f) => f.to_vec(),
                    None => (0..num_features).collect(),
                };
                let effective_m_try = m_try.max(1).min(pool.len());
                if effective_m_try > 0 && effective_m_try < pool.len() {
                    for i in 0..effective_m_try {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        if rng_state == 0 {
                            rng_state = 0x853c49e6748fea9b;
                        }
                        let j = i + ((rng_state as usize) % (pool.len() - i));
                        pool.swap(i, j);
                    }
                    pool.truncate(effective_m_try);
                }
                Some(pool)
            } else {
                None
            }
        } else if reg.colsample_bynode < 1.0 {
            let base_features: Vec<usize> = match allowed_features {
                Some(f) => f.to_vec(),
                None => (0..num_features).collect(),
            };
            Some(ColSubsampler::sample_by_node(
                &base_features,
                reg.colsample_bynode,
                rng_state,
            ))
        } else {
            None
        };

        let split_features = node_features.as_deref().or(allowed_features);

        // 2. Selección de la mejor división candidata (regularizada de Taylor o reducción de varianza)
        let split_candidate = if reg.is_regularized() {
            RegularizedSplitFinder::find_best_split(
                x,
                y,
                indices,
                num_features,
                split_features,
                min_samples_leaf,
                reg,
            )
        } else {
            SplitFinder::find_best_split(
                x,
                y,
                indices,
                num_features,
                split_features,
                min_samples_leaf,
            )
        };

        if let Some(split) = split_candidate {
            if split.gain < reg.min_split_gain {
                return TreeNode::Leaf {
                    value: leaf_value,
                    samples: n_samples,
                };
            }

            // Derivar seeds deterministas e independientes para los subárboles izquierdo y derecho
            let left_seed = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let right_seed = rng_state.wrapping_mul(6364136223846793005).wrapping_add(2);

            let left = Self::build_tree(
                x,
                y,
                &split.left_indices,
                num_features,
                allowed_features,
                current_depth + 1,
                max_depth,
                min_samples_leaf,
                reg,
                feature_subsample,
                left_seed,
            );

            let right = Self::build_tree(
                x,
                y,
                &split.right_indices,
                num_features,
                allowed_features,
                current_depth + 1,
                max_depth,
                min_samples_leaf,
                reg,
                feature_subsample,
                right_seed,
            );

            TreeNode::Split {
                feature_idx: split.feature_idx,
                threshold: split.threshold,
                left: Box::new(left),
                right: Box::new(right),
                gain: split.gain,
            }
        } else {
            TreeNode::Leaf {
                value: leaf_value,
                samples: n_samples,
            }
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        match &self.root {
            Some(node) => node.predict(features),
            None => 0.0,
        }
    }
}

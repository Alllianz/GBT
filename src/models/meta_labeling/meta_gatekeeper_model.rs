use rayon::prelude::*;
use crate::models::decision_tree::{CartTree, SplitCandidate, TreeNode};
use crate::models::regularization::{
    LeafPenaltyCalculator, RegularizedGainCalculator, TreeRegularizationConfig,
};
use super::meta_label_config::MetaLabelConfig;

pub struct MetaGatekeeperModel {
    config: MetaLabelConfig,
    base_log_odds: f64,
    trees: Vec<CartTree>,
    is_fitted: bool,
    num_features: usize,
}

impl MetaGatekeeperModel {
    pub fn new(config: MetaLabelConfig) -> Self {
        Self {
            config,
            base_log_odds: 0.0,
            trees: Vec::new(),
            is_fitted: false,
            num_features: 0,
        }
    }

    #[inline]
    fn sigmoid(f: f64) -> f64 {
        if !f.is_finite() {
            return 0.5;
        }
        let clamped = f.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-clamped).exp())
    }

    pub fn fit(
        &mut self,
        x_flat: &[f64],
        y_binary: &[f64],
        num_features: usize,
    ) -> Result<(), String> {
        let n_samples = y_binary.len();
        if n_samples < 5 {
            return Err(format!(
                "Muestras insuficientes ({}) para entrenar Meta Gatekeeper (mínimo 5)",
                n_samples
            ));
        }
        if num_features == 0 {
            return Err("Número de características debe ser mayor que 0".to_string());
        }
        if x_flat.len() != n_samples * num_features {
            return Err(format!(
                "Inconsistencia de dimensiones en Gatekeeper: x_flat={}, n={}, dim={}",
                x_flat.len(),
                n_samples,
                num_features
            ));
        }
        if y_binary.iter().any(|v| !v.is_finite()) {
            return Err("Etiquetas binarias contienen valores no finitos en Gatekeeper".to_string());
        }

        self.num_features = num_features;

        let sum_y: f64 = y_binary.iter().sum();
        let p0 = (sum_y / n_samples as f64).clamp(0.01, 0.99);
        self.base_log_odds = (p0 / (1.0 - p0)).ln();

        let mut current_f = vec![self.base_log_odds; n_samples];
        self.trees.clear();

        let mut reg_config = TreeRegularizationConfig::default();
        reg_config.l2_lambda = self.config.l2_regularization;
        reg_config.min_child_weight = 1.0;
        reg_config.min_split_gain = 1e-5;

        let lr = self.config.meta_learning_rate;
        let num_trees = self.config.num_meta_trees;

        for m in 0..num_trees {
            let mut g = Vec::with_capacity(n_samples);
            let mut h = Vec::with_capacity(n_samples);
            for (&y, &f) in y_binary.iter().zip(&current_f) {
                let p = Self::sigmoid(f);
                g.push(p - y);
                h.push(p * (1.0 - p));
            }

            let tree = Self::fit_newton_tree(
                x_flat,
                &g,
                &h,
                num_features,
                self.config.meta_max_depth,
                self.config.min_samples_leaf.max(1),
                &reg_config,
                m as u64 + 42,
            );

            for i in 0..n_samples {
                let row = &x_flat[(i * num_features)..((i + 1) * num_features)];
                current_f[i] += lr * tree.predict(row);
            }

            self.trees.push(tree);
        }

        self.is_fitted = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_newton_tree(
        x_flat: &[f64],
        g: &[f64],
        h: &[f64],
        num_features: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        reg: &TreeRegularizationConfig,
        seed: u64,
    ) -> CartTree {
        let all_indices: Vec<usize> = (0..g.len()).collect();
        let root = Self::build_newton_node(
            x_flat,
            g,
            h,
            &all_indices,
            num_features,
            0,
            max_depth,
            min_samples_leaf,
            reg,
        );
        CartTree {
            root: Some(root),
            max_depth,
            min_samples_leaf,
            min_gain: reg.min_split_gain,
            feature_subsample: None,
            seed,
            regularization: *reg,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_newton_node(
        x_flat: &[f64],
        g: &[f64],
        h: &[f64],
        indices: &[usize],
        num_features: usize,
        current_depth: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        reg: &TreeRegularizationConfig,
    ) -> TreeNode {
        let n_samples = indices.len();
        let sum_g: f64 = indices.iter().map(|&i| g[i]).sum();
        let sum_h: f64 = indices.iter().map(|&i| h[i]).sum();
        let leaf_value = LeafPenaltyCalculator::compute_leaf_weight_gh(
            sum_g,
            sum_h,
            reg.l1_alpha,
            reg.l2_lambda,
        );

        if current_depth >= max_depth || n_samples < min_samples_leaf * 2 {
            return TreeNode::Leaf {
                value: leaf_value,
                samples: n_samples,
            };
        }

        if let Some(split) = Self::find_best_newton_split(
            x_flat,
            g,
            h,
            indices,
            num_features,
            min_samples_leaf,
            reg,
            sum_g,
            sum_h,
        ) {
            let left = Self::build_newton_node(
                x_flat,
                g,
                h,
                &split.left_indices,
                num_features,
                current_depth + 1,
                max_depth,
                min_samples_leaf,
                reg,
            );
            let right = Self::build_newton_node(
                x_flat,
                g,
                h,
                &split.right_indices,
                num_features,
                current_depth + 1,
                max_depth,
                min_samples_leaf,
                reg,
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

    #[allow(clippy::too_many_arguments)]
    fn find_best_newton_split(
        x_flat: &[f64],
        g: &[f64],
        h: &[f64],
        indices: &[usize],
        num_features: usize,
        min_samples_leaf: usize,
        reg: &TreeRegularizationConfig,
        _total_g: f64,
        _total_h: f64,
    ) -> Option<SplitCandidate> {
        let min_leaf = min_samples_leaf.max(1);
        let n_samples = indices.len();
        if n_samples < min_leaf.saturating_mul(2) {
            return None;
        }

        (0..num_features)
            .into_par_iter()
            .filter_map(|feat_idx| {
                let mut feature_vals: Vec<(f64, usize)> = indices
                    .iter()
                    .copied()
                    .filter_map(|i| {
                        let flat_idx = i.checked_mul(num_features)?.checked_add(feat_idx)?;
                        let val = x_flat.get(flat_idx).copied()?;
                        if val.is_finite() {
                            Some((val, i))
                        } else {
                            None
                        }
                    })
                    .collect();
                let valid_n = feature_vals.len();
                if valid_n < min_leaf.saturating_mul(2) {
                    return None;
                }
                feature_vals.sort_by(|a, b| a.0.total_cmp(&b.0));

                let feat_total_g: f64 = feature_vals.iter().map(|p| g[p.1]).sum();
                let feat_total_h: f64 = feature_vals.iter().map(|p| h[p.1]).sum();

                let mut g_left = 0.0;
                let mut h_left = 0.0;
                for p in &feature_vals[0..min_leaf] {
                    g_left += g[p.1];
                    h_left += h[p.1];
                }

                let mut feat_best_gain = 0.0;
                let mut best_split_i = None;

                for split_i in min_leaf..=(valid_n - min_leaf) {
                    let current_val = feature_vals[split_i - 1].0;
                    let next_val = feature_vals[split_i].0;

                    if (current_val - next_val).abs() >= 1e-9 {
                        let g_right = feat_total_g - g_left;
                        let h_right = feat_total_h - h_left;

                        if let Some(gain) = RegularizedGainCalculator::evaluate_split_gain(
                            g_left,
                            h_left,
                            g_right,
                            h_right,
                            feat_total_g,
                            feat_total_h,
                            reg.l1_alpha,
                            reg.l2_lambda,
                            reg.min_split_gain,
                            reg.min_child_weight,
                        ) {
                            if gain > feat_best_gain {
                                feat_best_gain = gain;
                                best_split_i = Some(split_i);
                            }
                        }
                    }

                    if split_i < valid_n {
                        let idx = feature_vals[split_i].1;
                        g_left += g[idx];
                        h_left += h[idx];
                    }
                }

                if let Some(best_i) = best_split_i {
                    let a = feature_vals[best_i - 1].0;
                    let b = feature_vals[best_i].0;
                    let mid = a + (b - a) * 0.5;
                    let threshold = if mid.is_finite() { mid } else { a * 0.5 + b * 0.5 };
                    let left_indices: Vec<usize> = feature_vals[0..best_i].iter().map(|p| p.1).collect();
                    let right_indices: Vec<usize> = feature_vals[best_i..valid_n].iter().map(|p| p.1).collect();

                    Some(SplitCandidate {
                        feature_idx: feat_idx,
                        threshold,
                        gain: feat_best_gain,
                        left_indices,
                        right_indices,
                    })
                } else {
                    None
                }
            })
            .reduce_with(|a, b| if a.gain >= b.gain { a } else { b })
    }

    pub fn predict_probability(&self, augmented_features: &[f64]) -> f64 {
        if !self.is_fitted || self.trees.is_empty() {
            return 0.5;
        }

        let mut f = self.base_log_odds;
        let lr = self.config.meta_learning_rate;

        for tree in &self.trees {
            f += lr * tree.predict(augmented_features);
        }

        Self::sigmoid(f)
    }

    pub fn predict_probability_batch(
        &self,
        x_flat: &[f64],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f64> {
        if !self.is_fitted || self.trees.is_empty() || num_features == 0 {
            return vec![0.5; num_samples];
        }

        let mut probs = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let prob = i
                .checked_mul(num_features)
                .and_then(|start| start.checked_add(num_features).map(|end| (start, end)))
                .and_then(|(start, end)| x_flat.get(start..end))
                .map_or(0.5, |row| self.predict_probability(row));
            probs.push(prob);
        }

        probs
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_gatekeeper_learning() {
        let config = MetaLabelConfig::default().with_meta_boosting(20, 2, 0.1, 0.5);
        let mut model = MetaGatekeeperModel::new(config);

        let n = 40;
        let dim = 2;
        let mut x = Vec::new();
        let mut y = Vec::new();

        for i in 0..n {
            let val = if i % 2 == 0 { 2.0 } else { -2.0 };
            x.push(val);
            x.push(val * 0.5);
            y.push(if val > 0.0 { 1.0 } else { 0.0 });
        }

        let res = model.fit(&x, &y, dim);
        assert!(res.is_ok());
        assert!(model.is_fitted());

        let p_positive = model.predict_probability(&[2.0, 1.0]);
        let p_negative = model.predict_probability(&[-2.0, -1.0]);

        assert!(p_positive > 0.6);
        assert!(p_negative < 0.4);
    }

    #[test]
    fn test_sigmoid_non_finite_and_batch_bounds() {
        assert_eq!(MetaGatekeeperModel::sigmoid(f64::NAN), 0.5);
        assert_eq!(MetaGatekeeperModel::sigmoid(f64::INFINITY), 0.5);
        assert_eq!(MetaGatekeeperModel::sigmoid(f64::NEG_INFINITY), 0.5);

        let config = MetaLabelConfig::default().with_meta_boosting(5, 2, 0.2, 0.5);
        let mut model = MetaGatekeeperModel::new(config);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            let val = if i % 2 == 0 { 1.5 } else { -1.5 };
            x.push(val);
            x.push(0.0);
            y.push(if val > 0.0 { 1.0 } else { 0.0 });
        }
        model.fit(&x, &y, 2).unwrap();

        let short_batch = model.predict_probability_batch(&[1.5, 0.0], 3, 2);
        assert_eq!(short_batch.len(), 3);
        assert_eq!(short_batch[1], 0.5);
        assert_eq!(short_batch[2], 0.5);

        let zero_dim = model.predict_probability_batch(&[1.5, 0.0], 2, 0);
        assert_eq!(zero_dim, vec![0.5, 0.5]);
    }
}

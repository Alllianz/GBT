use rayon::prelude::*;
use crate::models::regularization::{RegularizedGainCalculator, TreeRegularizationConfig};
use super::split_finder::SplitCandidate;

/// Buscador de Divisiones con Regularización de Segundo Orden (Taylor Expansion)
///
/// Evalúa divisiones candidatas en paralelo a través de todas las variables autorizadas
/// utilizando la ganancia canónica de segundo orden con penalización L1 (soft-thresholding),
/// contracción L2 (Ridge), hessiano mínimo (`min_child_weight`) y ganancia mínima (`gamma`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RegularizedSplitFinder;

impl RegularizedSplitFinder {
    pub fn find_best_split(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        num_features: usize,
        allowed_features: Option<&[usize]>,
        min_samples_leaf: usize,
        reg: &TreeRegularizationConfig,
    ) -> Option<SplitCandidate> {
        let min_samples_leaf = min_samples_leaf.max(1);
        if indices.len() < min_samples_leaf.saturating_mul(2) {
            return None;
        }

        let features_to_evaluate: Vec<usize> = match allowed_features {
            Some(f) => f.to_vec(),
            None => (0..num_features).collect(),
        };

        features_to_evaluate
            .into_par_iter()
            .filter_map(|feat_idx| {
                let mut feature_vals: Vec<(f64, usize)> = indices
                    .iter()
                    .copied()
                    .filter_map(|i| {
                        let x_val = x[i * num_features + feat_idx];
                        let y_val = y[i];
                        if x_val.is_finite() && y_val.is_finite() {
                            Some((x_val, i))
                        } else {
                            None
                        }
                    })
                    .collect();

                let valid_n = feature_vals.len();
                if valid_n < min_samples_leaf.saturating_mul(2) {
                    return None;
                }

                feature_vals.sort_by(|a, b| a.0.total_cmp(&b.0));

                let total_sum_y: f64 = feature_vals.iter().map(|p| y[p.1]).sum();

                let mut sum_y_left = 0.0;
                for p in &feature_vals[0..min_samples_leaf] {
                    sum_y_left += y[p.1];
                }

                let mut feat_best_gain = 0.0;
                let mut best_split_i = None;

                for split_i in min_samples_leaf..=(valid_n - min_samples_leaf) {
                    let current_val = feature_vals[split_i - 1].0;
                    let next_val = feature_vals[split_i].0;

                    if (current_val - next_val).abs() >= 1e-9 {
                        let sum_y_right = total_sum_y - sum_y_left;
                        let n_left = split_i;
                        let n_right = valid_n - split_i;

                        let gain_opt = RegularizedGainCalculator::evaluate_residual_split_gain(
                            sum_y_left,
                            n_left,
                            sum_y_right,
                            n_right,
                            total_sum_y,
                            valid_n,
                            reg.l1_alpha,
                            reg.l2_lambda,
                            reg.min_split_gain,
                            reg.min_child_weight,
                        );

                        if let Some(gain) = gain_opt {
                            if gain > feat_best_gain {
                                feat_best_gain = gain;
                                best_split_i = Some(split_i);
                            }
                        }
                    }

                    if split_i < valid_n {
                        sum_y_left += y[feature_vals[split_i].1];
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regularized_split_finder_handles_zero_leaf_and_non_finite() {
        let x = vec![f64::NAN, 1.0, 2.0, f64::INFINITY, 10.0, 11.0];
        let y = vec![0.0, -1.0, -1.0, 0.0, 1.0, 1.0];
        let indices: Vec<usize> = (0..x.len()).collect();
        let reg = TreeRegularizationConfig::new().with_l1(0.0).with_l2(1.0);

        let cand = RegularizedSplitFinder::find_best_split(&x, &y, &indices, 1, None, 0, &reg)
            .expect("should clamp min_samples_leaf and filter non-finite");
        assert!(cand.threshold.is_finite());
        assert!((cand.threshold - 6.0).abs() < 1e-12);
        assert_eq!(cand.left_indices.len(), 2);
        assert_eq!(cand.right_indices.len(), 2);
    }
}

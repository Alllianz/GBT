use rayon::prelude::*;

pub struct SplitCandidate {
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
}

pub struct SplitFinder;

impl SplitFinder {
    pub fn find_best_split(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        num_features: usize,
        allowed_features: Option<&[usize]>,
        min_samples_leaf: usize,
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

                let mean_y = feature_vals.iter().map(|p| y[p.1]).sum::<f64>() / valid_n as f64;
                let mut total_sum_d = 0.0;
                let mut total_sum_sq_d = 0.0;
                for p in &feature_vals {
                    let d = y[p.1] - mean_y;
                    total_sum_d += d;
                    total_sum_sq_d += d * d;
                }
                let total_variance = (total_sum_sq_d - (total_sum_d * total_sum_d) / valid_n as f64).max(0.0);

                let mut sum_d_left = 0.0;
                let mut sum_sq_d_left = 0.0;
                for p in &feature_vals[0..min_samples_leaf] {
                    let d = y[p.1] - mean_y;
                    sum_d_left += d;
                    sum_sq_d_left += d * d;
                }

                let mut feat_best_gain = 0.0;
                let mut best_split_i = None;

                for split_i in min_samples_leaf..=(valid_n - min_samples_leaf) {
                    let current_val = feature_vals[split_i - 1].0;
                    let next_val = feature_vals[split_i].0;

                    if (current_val - next_val).abs() >= 1e-9 {
                        let n_left = split_i as f64;
                        let n_right = (valid_n - split_i) as f64;

                        let sum_d_right = total_sum_d - sum_d_left;
                        let sum_sq_d_right = total_sum_sq_d - sum_sq_d_left;

                        let var_left = (sum_sq_d_left - (sum_d_left * sum_d_left) / n_left).max(0.0);
                        let var_right = (sum_sq_d_right - (sum_d_right * sum_d_right) / n_right).max(0.0);

                        let gain = total_variance - (var_left + var_right);

                        if gain > feat_best_gain {
                            feat_best_gain = gain;
                            best_split_i = Some(split_i);
                        }
                    }

                    if split_i < valid_n {
                        let d_next = y[feature_vals[split_i].1] - mean_y;
                        sum_d_left += d_next;
                        sum_sq_d_left += d_next * d_next;
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
    fn test_split_finder_centered_variance_and_non_finite_safety() {
        let base = 1.0e9;
        let x = vec![f64::NAN, 1.0, 2.0, f64::NEG_INFINITY, 10.0, 11.0];
        let y = vec![base, base - 1.0, base - 1.0, base, base + 1.0, base + 1.0];
        let indices: Vec<usize> = (0..x.len()).collect();

        let cand = SplitFinder::find_best_split(&x, &y, &indices, 1, None, 0)
            .expect("centered variance should detect split despite large offset and non-finite");
        assert!(cand.threshold.is_finite());
        assert!((cand.threshold - 6.0).abs() < 1e-12);
        assert!((cand.gain - 4.0).abs() < 1e-6);
        assert_eq!(cand.left_indices.len(), 2);
        assert_eq!(cand.right_indices.len(), 2);
    }
}

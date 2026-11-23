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
        let n_samples = indices.len();
        if n_samples < min_samples_leaf * 2 {
            return None;
        }

        let total_sum_y: f64 = indices.iter().map(|&i| y[i]).sum();

        let features_to_evaluate: Vec<usize> = match allowed_features {
            Some(f) => f.to_vec(),
            None => (0..num_features).collect(),
        };

        features_to_evaluate
            .into_par_iter()
            .filter_map(|feat_idx| {
                let mut feature_vals: Vec<(f64, usize)> = indices
                    .iter()
                    .map(|&i| (x[i * num_features + feat_idx], i))
                    .collect();
                feature_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Inicializar acumulador del hijo izquierdo con las primeras min_samples_leaf muestras
                let mut sum_y_left = 0.0;
                for p in &feature_vals[0..min_samples_leaf] {
                    sum_y_left += y[p.1];
                }

                let mut feat_best_gain = 0.0;
                let mut best_split_i = None;

                for split_i in min_samples_leaf..=(n_samples - min_samples_leaf) {
                    let current_val = feature_vals[split_i - 1].0;
                    let next_val = feature_vals[split_i].0;

                    if (current_val - next_val).abs() >= 1e-9 {
                        let sum_y_right = total_sum_y - sum_y_left;
                        let n_left = split_i;
                        let n_right = n_samples - split_i;

                        let gain_opt = RegularizedGainCalculator::evaluate_residual_split_gain(
                            sum_y_left,
                            n_left,
                            sum_y_right,
                            n_right,
                            total_sum_y,
                            n_samples,
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

                    if split_i < n_samples {
                        sum_y_left += y[feature_vals[split_i].1];
                    }
                }

                if let Some(best_i) = best_split_i {
                    let threshold = (feature_vals[best_i - 1].0 + feature_vals[best_i].0) * 0.5;
                    let left_indices: Vec<usize> = feature_vals[0..best_i].iter().map(|p| p.1).collect();
                    let right_indices: Vec<usize> = feature_vals[best_i..n_samples].iter().map(|p| p.1).collect();

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

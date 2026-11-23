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
        let n_samples = indices.len();
        if n_samples < min_samples_leaf * 2 {
            return None;
        }

        // Suma y suma de cuadrados base del nodo para cálculo incremental O(N) sin asignaciones
        let mut total_sum_y = 0.0;
        let mut total_sum_sq_y = 0.0;
        for &i in indices {
            let val = y[i];
            total_sum_y += val;
            total_sum_sq_y += val * val;
        }
        let total_variance = (total_sum_sq_y - (total_sum_y * total_sum_y) / n_samples as f64).max(0.0);

        let features_to_evaluate: Vec<usize> = match allowed_features {
            Some(f) => f.to_vec(),
            None => (0..num_features).collect(),
        };

        // Búsqueda del mejor corte paralelizada a través de las variables con Rayon (20 hilos)
        features_to_evaluate
            .into_par_iter()
            .filter_map(|feat_idx| {
                // Extraer y ordenar valores de la variable
                let mut feature_vals: Vec<(f64, usize)> = indices
                    .iter()
                    .map(|&i| (x[i * num_features + feat_idx], i))
                    .collect();
                feature_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Inicializar sumas acumulativas del hijo izquierdo con las primeras min_samples_leaf muestras
                let mut sum_y_left = 0.0;
                let mut sum_sq_y_left = 0.0;
                for p in &feature_vals[0..min_samples_leaf] {
                    let y_val = y[p.1];
                    sum_y_left += y_val;
                    sum_sq_y_left += y_val * y_val;
                }

                let mut feat_best_gain = 0.0;
                let mut best_split_i = None;

                // Rango inclusivo: evalúa hasta n_samples - min_samples_leaf para permitir que el hijo derecho tenga exactamente min_samples_leaf
                for split_i in min_samples_leaf..=(n_samples - min_samples_leaf) {
                    let current_val = feature_vals[split_i - 1].0;
                    let next_val = feature_vals[split_i].0;

                    if (current_val - next_val).abs() >= 1e-9 {
                        let n_left = split_i as f64;
                        let n_right = (n_samples - split_i) as f64;

                        let sum_y_right = total_sum_y - sum_y_left;
                        let sum_sq_y_right = total_sum_sq_y - sum_sq_y_left;

                        let var_left = (sum_sq_y_left - (sum_y_left * sum_y_left) / n_left).max(0.0);
                        let var_right = (sum_sq_y_right - (sum_y_right * sum_y_right) / n_right).max(0.0);

                        let gain = total_variance - (var_left + var_right);

                        if gain > feat_best_gain {
                            feat_best_gain = gain;
                            best_split_i = Some(split_i);
                        }
                    }

                    // Avanzar las sumas acumulativas para el siguiente corte candidato
                    if split_i < n_samples {
                        let y_next = y[feature_vals[split_i].1];
                        sum_y_left += y_next;
                        sum_sq_y_left += y_next * y_next;
                    }
                }

                // Asignar los vectores de índices únicamente una vez para el mejor corte encontrado
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

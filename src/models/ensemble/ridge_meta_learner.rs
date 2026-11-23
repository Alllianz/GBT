/// Meta-Aprendiz Lineal Regularizado Ridge (Tikhonov) con Proyección Simplex
///
/// Resuelve analíticamente el problema de regresión regularizada de segundo orden sobre
/// la matriz de predicciones Out-of-Fold (OOF) de $M$ modelos base:
///
/// $$w^* = \arg\min_w \frac{1}{2} \|P_{\text{OOF}} w - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2$$
///
/// La solución cerrada del sistema normal es:
/// $$(P_{\text{OOF}}^T P_{\text{OOF}} + \lambda I) w = P_{\text{OOF}}^T y$$
///
/// Resuelto analíticamente mediante descomposición de Cholesky ($A = L L^T$).
/// Si `non_negative` es `true`, los pesos se proyectan euclídeamente sobre el simplex unitario:
/// $$\Delta = \{w \in \mathbb{R}^M : w_m \ge 0, \sum_{m=1}^M w_m = 1\}$$
/// garantizando que ningún modelo sea invertido de manera espuria o sobre-apalancado fuera de muestra.
#[derive(Debug, Clone)]
pub struct RidgeMetaLearner {
    weights: Vec<f64>,
    lambda: f64,
    non_negative: bool,
    num_models: usize,
    is_fitted: bool,
}

impl RidgeMetaLearner {
    pub fn new(lambda: f64, non_negative: bool) -> Self {
        Self {
            weights: Vec::new(),
            lambda: lambda.max(1e-6),
            non_negative,
            num_models: 0,
            is_fitted: false,
        }
    }

    /// Ajusta analíticamente los pesos del ensamble sobre la matriz OOF y las etiquetas reales.
    ///
    /// - `oof_predictions_flat`: Matriz plana `(N x M)` con las predicciones fuera de muestra de cada modelo base.
    /// - `targets`: Vector de retornos reales $y \in \mathbb{R}^N$.
    /// - `num_models`: Número de estimadores base $M$.
    pub fn fit(
        &mut self,
        oof_predictions_flat: &[f64],
        targets: &[f64],
        num_models: usize,
    ) -> Result<(), String> {
        let n_samples = targets.len();
        if n_samples == 0 || num_models == 0 {
            return Err("Muestras o modelos insuficientes para ajustar RidgeMetaLearner".to_string());
        }
        if oof_predictions_flat.len() != n_samples * num_models {
            return Err(format!(
                "Dimensiones incompatibles en RidgeMetaLearner: flat_len={}, n={}, m={}",
                oof_predictions_flat.len(),
                n_samples,
                num_models
            ));
        }

        self.num_models = num_models;

        // 1. Calcular matriz normal A = P^T P + \lambda I (dimensión M x M)
        let mut a = vec![0.0; num_models * num_models];
        for j in 0..num_models {
            for k in 0..num_models {
                let mut sum = 0.0;
                for i in 0..n_samples {
                    let p_ij = oof_predictions_flat[i * num_models + j];
                    let p_ik = oof_predictions_flat[i * num_models + k];
                    sum += p_ij * p_ik;
                }
                if j == k {
                    sum += self.lambda;
                }
                a[j * num_models + k] = sum;
            }
        }

        // 2. Calcular vector de correlación cruzada b = P^T y (dimensión M)
        let mut b = vec![0.0; num_models];
        for j in 0..num_models {
            let mut sum = 0.0;
            for i in 0..n_samples {
                let p_ij = oof_predictions_flat[i * num_models + j];
                sum += p_ij * targets[i];
            }
            b[j] = sum;
        }

        // 3. Resolver A w = b mediante Descomposición de Cholesky
        let raw_weights = Self::solve_cholesky(&a, &b, num_models)?;

        // 4. Proyección simplex si non_negative está activo
        if self.non_negative {
            self.weights = Self::project_to_simplex(&raw_weights);
        } else {
            self.weights = raw_weights;
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Resuelve el sistema lineal simétrico definido positivo A w = b mediante Cholesky (A = L L^T)
    fn solve_cholesky(a: &[f64], b: &[f64], m: usize) -> Result<Vec<f64>, String> {
        let mut l = vec![0.0; m * m];

        // Factorización Cholesky: A = L L^T
        for j in 0..m {
            let mut sum_diag = 0.0;
            for k in 0..j {
                let val = l[j * m + k];
                sum_diag += val * val;
            }

            let diag_val = a[j * m + j] - sum_diag;
            let l_jj = if diag_val > 1e-12 {
                diag_val.sqrt()
            } else {
                // Regularización estocástica adicional ante cuasi-singularidad
                1e-6
            };
            l[j * m + j] = l_jj;

            for i in (j + 1)..m {
                let mut sum_off = 0.0;
                for k in 0..j {
                    sum_off += l[i * m + k] * l[j * m + k];
                }
                l[i * m + j] = (a[i * m + j] - sum_off) / l_jj;
            }
        }

        // Sustitución hacia adelante: L z = b
        let mut z = vec![0.0; m];
        for i in 0..m {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i * m + k] * z[k];
            }
            z[i] = (b[i] - sum) / l[i * m + i];
        }

        // Sustitución hacia atrás: L^T w = z
        let mut w = vec![0.0; m];
        for i in (0..m).rev() {
            let mut sum = 0.0;
            for k in (i + 1)..m {
                sum += l[k * m + i] * w[k];
            }
            w[i] = (z[i] - sum) / l[i * m + i];
        }

        Ok(w)
    }

    /// Proyección Euclídea exacta sobre el Simplex Unitario $\sum w_m = 1, w_m \ge 0$
    ///
    /// Algoritmo exacto de Wang & Carreira-Perpiñán (2013) / Held et al. O(M log M).
    pub fn project_to_simplex(v: &[f64]) -> Vec<f64> {
        let m = v.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            return vec![1.0];
        }

        // Ordenar v en orden descendente
        let mut u = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Encontrar rho
        let mut cum_sum = 0.0;
        let mut rho = 0;
        for (j, &val) in u.iter().enumerate() {
            cum_sum += val;
            let temp = val + (1.0 - cum_sum) / (j as f64 + 1.0);
            if temp > 0.0 {
                rho = j + 1;
            }
        }

        // Calcular theta
        let sum_rho: f64 = u.iter().take(rho).sum();
        let theta = (1.0 - sum_rho) / (rho as f64);

        // Proyectar: w_i = max(v_i + theta, 0.0)
        let mut projected: Vec<f64> = v.iter().map(|&x| (x + theta).max(0.0)).collect();

        // Normalización final por estabilidad de redondeo en punto flotante
        let sum_proj: f64 = projected.iter().sum();
        if sum_proj > 1e-12 {
            for val in &mut projected {
                *val /= sum_proj;
            }
        } else {
            // Fallback a pesos uniformes si la suma degeneró a cero
            let uniform = 1.0 / m as f64;
            projected = vec![uniform; m];
        }

        projected
    }

    /// Combina las predicciones puntuales de los $M$ modelos base ponderadas por $w^*$
    pub fn predict(&self, model_predictions: &[f64]) -> f64 {
        if !self.is_fitted || self.weights.is_empty() {
            if model_predictions.is_empty() {
                return 0.0;
            }
            return model_predictions.iter().sum::<f64>() / model_predictions.len() as f64;
        }

        let m = self.weights.len().min(model_predictions.len());
        let mut blended = 0.0;
        for i in 0..m {
            blended += self.weights[i] * model_predictions[i];
        }

        blended
    }

    /// Inferencia vectorizada en batch
    pub fn predict_batch(
        &self,
        predictions_matrix_flat: &[f64],
        num_samples: usize,
    ) -> Vec<f64> {
        let mut output = Vec::with_capacity(num_samples);
        let m = self.num_models;

        for i in 0..num_samples {
            let row = &predictions_matrix_flat[(i * m)..((i + 1) * m)];
            output.push(self.predict(row));
        }

        output
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_projection() {
        // Vector que ya suma 1 y es no-negativo
        let v1 = vec![0.6, 0.4];
        let p1 = RidgeMetaLearner::project_to_simplex(&v1);
        assert!((p1[0] - 0.6).abs() < 1e-6);
        assert!((p1[1] - 0.4).abs() < 1e-6);

        // Vector con componente negativa
        let v2 = vec![1.2, -0.4];
        let p2 = RidgeMetaLearner::project_to_simplex(&v2);
        assert!((p2[0] - 1.0).abs() < 1e-6);
        assert!((p2[1] - 0.0).abs() < 1e-6);

        // Vector simétrico desbalanceado
        let v3 = vec![2.0, 1.0, 0.0];
        let p3 = RidgeMetaLearner::project_to_simplex(&v3);
        let sum_p3: f64 = p3.iter().sum();
        assert!((sum_p3 - 1.0).abs() < 1e-6);
        assert!(p3.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_ridge_meta_learner_fit_analytic() {
        let n = 50;
        let num_models = 2;
        let mut oof_preds = Vec::with_capacity(n * num_models);
        let mut targets = Vec::with_capacity(n);

        // Modelo 0 es muy bueno (correlación perfecta), Modelo 1 es ruido
        for i in 0..n {
            let true_val = (i as f64 * 0.001) - 0.025;
            let pred_good = true_val;
            let pred_noise = 0.05 * if i % 2 == 0 { 1.0 } else { -1.0 };

            oof_preds.push(pred_good);
            oof_preds.push(pred_noise);
            targets.push(true_val);
        }

        let mut learner = RidgeMetaLearner::new(0.01, true);
        let res = learner.fit(&oof_preds, &targets, num_models);

        assert!(res.is_ok(), "Ajuste analítico debe completarse");
        assert!(learner.is_fitted());
        let w = learner.weights();
        assert_eq!(w.len(), 2);
        assert!(
            w[0] > w[1],
            "El modelo con alta correlación debe recibir mayor peso (w[0]={}, w[1]={})",
            w[0],
            w[1]
        );
        let sum_w: f64 = w.iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-6);
    }
}

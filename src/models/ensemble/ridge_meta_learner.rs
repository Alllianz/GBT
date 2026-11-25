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
        let safe_lambda = if lambda.is_finite() {
            lambda.max(1e-6)
        } else {
            1.0
        };
        Self {
            weights: Vec::new(),
            lambda: safe_lambda,
            non_negative,
            num_models: 0,
            is_fitted: false,
        }
    }

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
        if oof_predictions_flat.iter().any(|v| !v.is_finite()) {
            return Err("Predicciones OOF contienen NaN o Inf en RidgeMetaLearner".to_string());
        }
        if targets.iter().any(|v| !v.is_finite()) {
            return Err("Targets contienen NaN o Inf en RidgeMetaLearner".to_string());
        }

        self.num_models = num_models;

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

        let mut b = vec![0.0; num_models];
        for j in 0..num_models {
            let mut sum = 0.0;
            for i in 0..n_samples {
                let p_ij = oof_predictions_flat[i * num_models + j];
                sum += p_ij * targets[i];
            }
            b[j] = sum;
        }

        if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
            return Err("Sistema normal contiene valores no finitos en RidgeMetaLearner".to_string());
        }

        let raw_weights = Self::solve_cholesky(&a, &b, num_models)?;

        if self.non_negative {
            self.weights = Self::project_to_simplex(&raw_weights);
        } else {
            self.weights = raw_weights;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn try_cholesky(a: &[f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0; m * m];

        for j in 0..m {
            let mut sum_diag = 0.0;
            for k in 0..j {
                let val = l[j * m + k];
                sum_diag += val * val;
            }

            let diag_val = a[j * m + j] - sum_diag;
            if !diag_val.is_finite() || diag_val <= 1e-12 {
                return None;
            }
            let l_jj = diag_val.sqrt();
            l[j * m + j] = l_jj;

            for i in (j + 1)..m {
                let mut sum_off = 0.0;
                for k in 0..j {
                    sum_off += l[i * m + k] * l[j * m + k];
                }
                let val = (a[i * m + j] - sum_off) / l_jj;
                if !val.is_finite() {
                    return None;
                }
                l[i * m + j] = val;
            }
        }

        let mut z = vec![0.0; m];
        for i in 0..m {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i * m + k] * z[k];
            }
            let zi = (b[i] - sum) / l[i * m + i];
            if !zi.is_finite() {
                return None;
            }
            z[i] = zi;
        }

        let mut w = vec![0.0; m];
        for i in (0..m).rev() {
            let mut sum = 0.0;
            for k in (i + 1)..m {
                sum += l[k * m + i] * w[k];
            }
            let wi = (z[i] - sum) / l[i * m + i];
            if !wi.is_finite() {
                return None;
            }
            w[i] = wi;
        }

        Some(w)
    }

    fn solve_cholesky(a: &[f64], b: &[f64], m: usize) -> Result<Vec<f64>, String> {
        if let Some(w) = Self::try_cholesky(a, b, m) {
            return Ok(w);
        }

        let mean_diag = (0..m).map(|i| a[i * m + i].abs()).sum::<f64>() / (m as f64).max(1.0);
        let base_jitter = (mean_diag * 1e-8).max(1e-10);

        for exp in 0..12 {
            let jitter = base_jitter * 10.0_f64.powi(exp);
            let mut a_reg = a.to_vec();
            for j in 0..m {
                a_reg[j * m + j] += jitter;
            }
            if let Some(w) = Self::try_cholesky(&a_reg, b, m) {
                return Ok(w);
            }
        }

        Err("Matriz normal no definida positiva tras jitter adaptativo en Cholesky".to_string())
    }

    pub fn project_to_simplex(v: &[f64]) -> Vec<f64> {
        let m = v.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            return vec![1.0];
        }
        if v.iter().any(|x| !x.is_finite()) {
            return vec![1.0 / m as f64; m];
        }

        let mut u = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum_sum = 0.0;
        let mut rho = 0;
        for (j, &val) in u.iter().enumerate() {
            cum_sum += val;
            let temp = val + (1.0 - cum_sum) / (j as f64 + 1.0);
            if temp > 0.0 {
                rho = j + 1;
            }
        }

        if rho == 0 {
            return vec![1.0 / m as f64; m];
        }

        let sum_rho: f64 = u.iter().take(rho).sum();
        let theta = (1.0 - sum_rho) / (rho as f64);

        let mut projected: Vec<f64> = v.iter().map(|&x| (x + theta).max(0.0)).collect();

        let sum_proj: f64 = projected.iter().sum();
        if sum_proj.is_finite() && sum_proj > 1e-12 {
            for val in &mut projected {
                *val /= sum_proj;
            }
        } else {
            let uniform = 1.0 / m as f64;
            projected = vec![uniform; m];
        }

        projected
    }

    pub fn predict(&self, model_predictions: &[f64]) -> f64 {
        if !self.is_fitted || self.weights.is_empty() {
            if model_predictions.is_empty() {
                return 0.0;
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for &p in model_predictions {
                if p.is_finite() {
                    sum += p;
                    count += 1;
                }
            }
            if count == 0 {
                return 0.0;
            }
            let avg = sum / count as f64;
            return if avg.is_finite() { avg } else { 0.0 };
        }

        let m = self.weights.len().min(model_predictions.len());
        let mut blended = 0.0;
        for i in 0..m {
            let p = model_predictions[i];
            let w = self.weights[i];
            if p.is_finite() && w.is_finite() {
                blended += w * p;
            }
        }

        if blended.is_finite() {
            blended
        } else {
            0.0
        }
    }

    pub fn predict_batch(
        &self,
        predictions_matrix_flat: &[f64],
        num_samples: usize,
    ) -> Vec<f64> {
        let m = self.num_models;
        if !self.is_fitted || m == 0 {
            return vec![0.0; num_samples];
        }

        let mut output = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let val = i
                .checked_mul(m)
                .and_then(|start| start.checked_add(m).map(|end| (start, end)))
                .and_then(|(start, end)| predictions_matrix_flat.get(start..end))
                .map_or(0.0, |row| self.predict(row));
            output.push(val);
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

    #[test]
    fn test_ridge_rejects_nan_and_inf() {
        let mut learner = RidgeMetaLearner::new(0.01, true);
        assert!(learner.fit(&[f64::NAN, 0.1], &[0.1], 2).is_err());
        assert!(learner.fit(&[0.1, f64::INFINITY], &[0.1], 2).is_err());
        assert!(learner.fit(&[0.1, 0.2], &[f64::NEG_INFINITY], 2).is_err());
    }

    #[test]
    fn test_ridge_adaptive_jitter_and_simplex_guard() {
        let p_uniform = RidgeMetaLearner::project_to_simplex(&[f64::NAN, 1.0, 2.0]);
        assert_eq!(p_uniform.len(), 3);
        assert!((p_uniform.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        let a = vec![1.0, 2.0, 2.0, 1.0];
        let b = vec![1.0, 1.0];
        let w = RidgeMetaLearner::solve_cholesky(&a, &b, 2).expect("adaptive jitter solves non-PD");
        assert!(w.iter().all(|v| v.is_finite()));
    }
}

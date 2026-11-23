use crate::models::decision_tree::CartTree;
use crate::models::regularization::TreeRegularizationConfig;
use super::meta_label_config::MetaLabelConfig;

/// Modelo Gatekeeper Secundario basado en GBDT Logístico Regularizado
///
/// Implementa un estimador de probabilidad de éxito $\hat{p}_{\text{meta}}(x) = P(y_{\text{meta}} = 1 \mid x)$
/// utilizando Gradient Boosted Trees con función de pérdida logística (Binary Cross-Entropy):
/// - Predicción log-odds acumulada: $F(x) = F_0 + \sum_{m=1}^M \eta \cdot h_m(x)$
/// - Probabilidad sigmoide: $\hat{p}_{\text{meta}}(x) = \frac{1}{1 + e^{-F(x)}}$
///
/// Si $\hat{p}_{\text{meta}} < \text{confidence\_threshold}$, la señal primaria es vetada.
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

    /// Entrena el Gatekeeper secundario sobre el dataset de meta-muestras activas.
    ///
    /// `x_flat`: Matriz plana de características aumentadas `(N x num_features)`.
    /// `y_binary`: Vector de etiquetas binarias $y_i \in \{0.0, 1.0\}$.
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
        if x_flat.len() != n_samples * num_features {
            return Err(format!(
                "Inconsistencia de dimensiones en Gatekeeper: x_flat={}, n={}, dim={}",
                x_flat.len(),
                n_samples,
                num_features
            ));
        }

        self.num_features = num_features;

        // 1. Probabilidad base inicial $p_0$ y log-odds $F_0 = \ln(p_0 / (1 - p_0))$
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
            // 2. Calcular probabilidades actuales y gradientes negativos de la pérdida logística
            // Pérdida: L(y, F) = -[y ln(p) + (1-y) ln(1-p)]
            // Gradiente: g_i = p_i - y_i  =>  Residual negativo r_i = -g_i = y_i - p_i
            let residuals: Vec<f64> = y_binary
                .iter()
                .zip(&current_f)
                .map(|(&y, &f)| {
                    let p = 1.0 / (1.0 + (-f).exp());
                    y - p
                })
                .collect();

            // 3. Ajustar un árbol CART sobre los residuales
            let mut tree = CartTree::new_regularized(
                self.config.meta_max_depth,
                self.config.min_samples_leaf,
                reg_config.clone(),
                m as u64 + 42,
            );

            tree.fit(x_flat, &residuals, num_features, None);

            // 4. Actualizar log-odds: F_{m}(x) = F_{m-1}(x) + lr * tree(x)
            for i in 0..n_samples {
                let row = &x_flat[(i * num_features)..((i + 1) * num_features)];
                current_f[i] += lr * tree.predict(row);
            }

            self.trees.push(tree);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Predice la probabilidad condicional de éxito $\hat{p}_{\text{meta}} \in [0, 1]$
    /// para un vector individual de características aumentadas.
    pub fn predict_probability(&self, augmented_features: &[f64]) -> f64 {
        if !self.is_fitted || self.trees.is_empty() {
            return 0.5;
        }

        let mut f = self.base_log_odds;
        let lr = self.config.meta_learning_rate;

        for tree in &self.trees {
            f += lr * tree.predict(augmented_features);
        }

        1.0 / (1.0 + (-f).exp())
    }

    /// Inferencia vectorizada sobre una matriz de características aumentadas
    pub fn predict_probability_batch(
        &self,
        x_flat: &[f64],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f64> {
        if !self.is_fitted || self.trees.is_empty() {
            return vec![0.5; num_samples];
        }

        let mut probs = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let row = &x_flat[(i * num_features)..((i + 1) * num_features)];
            probs.push(self.predict_probability(row));
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

        // Crear dataset sintético separable: si x[0] > 0 -> win (1.0), sino -> loss (0.0)
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
        assert!(res.is_ok(), "Fit debe completarse sin error");
        assert!(model.is_fitted());

        let p_positive = model.predict_probability(&[2.0, 1.0]);
        let p_negative = model.predict_probability(&[-2.0, -1.0]);

        assert!(
            p_positive > 0.6,
            "Probabilidad para señal ganadora debe ser alta ({})",
            p_positive
        );
        assert!(
            p_negative < 0.4,
            "Probabilidad para señal perdedora debe ser baja ({})",
            p_negative
        );
    }
}

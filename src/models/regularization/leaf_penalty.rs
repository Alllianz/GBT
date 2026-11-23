/// Calculador de Penalización y Contracción de Segundo Orden para Hojas
///
/// Implementa la contracción analítica óptima de Taylor con regularización L1 (soft-thresholding)
/// y regularización L2 (shrinkage Ridge cuadrático):
///
/// w^* = sign(sum r_i) * max(0, |sum r_i| - alpha) / (|I| + lambda)
#[derive(Debug, Clone, Copy, Default)]
pub struct LeafPenaltyCalculator;

impl LeafPenaltyCalculator {
    /// Calcula el peso óptimo de una hoja regularizada dados la suma de residuales y el conteo de muestras
    #[inline]
    pub fn compute_leaf_weight_sum(
        sum_residuals: f64,
        count: usize,
        l1_alpha: f64,
        l2_lambda: f64,
    ) -> f64 {
        let h = count as f64;
        Self::compute_leaf_weight_gh(-sum_residuals, h, l1_alpha, l2_lambda)
    }

    /// Calcula el peso óptimo a partir de un slice directo de residuales
    pub fn compute_leaf_weight_residuals(
        residuals: &[f64],
        l1_alpha: f64,
        l2_lambda: f64,
    ) -> f64 {
        if residuals.is_empty() {
            return 0.0;
        }
        let sum_residuals: f64 = residuals.iter().sum();
        Self::compute_leaf_weight_sum(sum_residuals, residuals.len(), l1_alpha, l2_lambda)
    }

    /// Calcula el peso óptimo general de segundo orden a partir del gradiente acumulado G y hessiano H:
    /// Para MSE: g_i = -r_i, por lo que G = -sum(r_i).
    /// El peso óptimo del árbol es:
    /// w^* = -sign(G) * max(0, |G| - alpha) / (H + lambda)
    /// Lo cual equivale exactamente a:
    /// w^* = sign(sum r_i) * max(0, |sum r_i| - alpha) / (H + lambda)
    #[inline]
    pub fn compute_leaf_weight_gh(
        sum_g: f64,
        sum_h: f64,
        l1_alpha: f64,
        l2_lambda: f64,
    ) -> f64 {
        let denominator = sum_h + l2_lambda;
        if denominator <= 1e-12 {
            return 0.0;
        }

        let abs_g = sum_g.abs();
        if abs_g <= l1_alpha {
            return 0.0;
        }

        let thresholded_g = abs_g - l1_alpha;
        let sign_g = if sum_g >= 0.0 { 1.0 } else { -1.0 };

        // El paso de boosting minimiza la pérdida en dirección opuesta al gradiente (-g)
        -sign_g * (thresholded_g / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_weight_without_regularization() {
        let residuals = vec![1.0, 2.0, 3.0];
        let w = LeafPenaltyCalculator::compute_leaf_weight_residuals(&residuals, 0.0, 0.0);
        // Media exacta = 2.0
        assert!((w - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_leaf_weight_with_l1_thresholding() {
        let sum_r = 0.5;
        let count = 10;
        // Si alpha >= 0.5, el peso colapsa a 0.0
        let w_zero = LeafPenaltyCalculator::compute_leaf_weight_sum(sum_r, count, 0.6, 0.0);
        assert_eq!(w_zero, 0.0);

        // Si alpha < 0.5, se contrae
        let w_shrink = LeafPenaltyCalculator::compute_leaf_weight_sum(sum_r, count, 0.1, 0.0);
        // (0.5 - 0.1) / 10 = 0.04
        assert!((w_shrink - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_leaf_weight_with_l2_shrinkage() {
        let sum_r = 10.0;
        let count = 10;
        let w = LeafPenaltyCalculator::compute_leaf_weight_sum(sum_r, count, 0.0, 10.0);
        // 10 / (10 + 10) = 0.5
        assert!((w - 0.5).abs() < 1e-10);
    }
}

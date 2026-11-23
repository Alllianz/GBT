/// Calculador de Ganancia de División Regularizada de Segundo Orden
///
/// Implementa la formulación canónica de Taylor de XGBoost / Forust con contracción
/// analítica L1 y L2:
///
/// Gain_reg = 0.5 * [ G_{L,alpha}^2 / (H_L + lambda) + G_{R,alpha}^2 / (H_R + lambda) - G_{alpha}^2 / (H + lambda) ] - gamma
///
/// donde G_{alpha}^2 = (max(0, |G| - alpha))^2.
/// Si H_L < min_child_weight o H_R < min_child_weight, el corte queda invalidado.
/// Si Gain_reg <= 0.0, el corte es rechazado (pre-poda analítica).
#[derive(Debug, Clone, Copy, Default)]
pub struct RegularizedGainCalculator;

impl RegularizedGainCalculator {
    /// Calcula el término cuadrático regularizado G_{alpha}^2 / (H + lambda)
    #[inline]
    pub fn compute_score(sum_g: f64, sum_h: f64, l1_alpha: f64, l2_lambda: f64) -> f64 {
        let denom = sum_h + l2_lambda;
        if denom <= 1e-12 {
            return 0.0;
        }

        let abs_g = sum_g.abs();
        if abs_g <= l1_alpha {
            return 0.0;
        }

        let thresholded = abs_g - l1_alpha;
        (thresholded * thresholded) / denom
    }

    /// Calcula el término de score a partir de la suma de residuales r_i (donde g_i = -r_i, por ende |G| = |sum_r|)
    #[inline]
    pub fn compute_residual_score(
        sum_residuals: f64,
        count: usize,
        l1_alpha: f64,
        l2_lambda: f64,
    ) -> f64 {
        Self::compute_score(sum_residuals, count as f64, l1_alpha, l2_lambda)
    }

    /// Evalúa la ganancia regularizada canónica entre dos particiones hijo izquierda y derecha
    ///
    /// Retorna `Some(gain)` si y solo si:
    /// 1. Ambos hijos satisfacen `min_child_weight`.
    /// 2. La ganancia neta tras deducir `gamma` (`min_split_gain`) es estrictamente positiva (> 0.0).
    pub fn evaluate_split_gain(
        g_left: f64,
        h_left: f64,
        g_right: f64,
        h_right: f64,
        g_total: f64,
        h_total: f64,
        l1_alpha: f64,
        l2_lambda: f64,
        min_split_gain: f64,
        min_child_weight: f64,
    ) -> Option<f64> {
        if h_left < min_child_weight || h_right < min_child_weight {
            return None;
        }

        let score_left = Self::compute_score(g_left, h_left, l1_alpha, l2_lambda);
        let score_right = Self::compute_score(g_right, h_right, l1_alpha, l2_lambda);
        let score_total = Self::compute_score(g_total, h_total, l1_alpha, l2_lambda);

        let gain = 0.5 * (score_left + score_right - score_total) - min_split_gain;

        if gain > 0.0 {
            Some(gain)
        } else {
            None
        }
    }

    /// Evalúa la ganancia regularizada basada en residuales
    #[inline]
    pub fn evaluate_residual_split_gain(
        sum_r_left: f64,
        n_left: usize,
        sum_r_right: f64,
        n_right: usize,
        sum_r_total: f64,
        n_total: usize,
        l1_alpha: f64,
        l2_lambda: f64,
        min_split_gain: f64,
        min_child_weight: f64,
    ) -> Option<f64> {
        Self::evaluate_split_gain(
            sum_r_left,
            n_left as f64,
            sum_r_right,
            n_right as f64,
            sum_r_total,
            n_total as f64,
            l1_alpha,
            l2_lambda,
            min_split_gain,
            min_child_weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_gain_min_child_weight_rejection() {
        let gain = RegularizedGainCalculator::evaluate_split_gain(
            10.0, 1.0, // h_left = 1.0 < min_child_weight 5.0
            10.0, 10.0,
            20.0, 11.0,
            0.0, 1.0,
            0.0,
            5.0,
        );
        assert!(gain.is_none());
    }

    #[test]
    fn test_split_gain_gamma_prepruning() {
        let score_l = RegularizedGainCalculator::compute_score(5.0, 10.0, 0.0, 1.0);
        let score_r = RegularizedGainCalculator::compute_score(5.0, 10.0, 0.0, 1.0);
        let score_t = RegularizedGainCalculator::compute_score(10.0, 20.0, 0.0, 1.0);
        let raw_gain = 0.5 * (score_l + score_r - score_t);

        // Si gamma supera raw_gain, el corte debe ser rechazado
        let rejected = RegularizedGainCalculator::evaluate_split_gain(
            5.0, 10.0,
            5.0, 10.0,
            10.0, 20.0,
            0.0, 1.0,
            raw_gain + 1.0, // gamma alto
            1.0,
        );
        assert!(rejected.is_none());
    }
}

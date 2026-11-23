//! Módulo de cálculo del Score Compuesto de Optimización
//!
//! BUG-F07: Recalibración armónica del Score Compuesto:
//! 1. Normalización equilibrada entre la linealidad R² de balance [0.0, 1.0]
//!    y el ratio Capital / Max Drawdown [0.0, 100.0] -> normalizado a [0.0, 1.0].
//!    Ambos términos aportan un peso armónico y representativo de 50% cada uno.
//! 2. Puntuación estrictamente no positiva si PnL <= 0:
//!    Toda estrategia con pérdida neta final (PnL < 0) recibe una penalización monótona
//!    estrictamente negativa (< 0.0), evitando falsos positivos donde caídas lineales suaves
//!    generaban scores positivos.

pub struct CompositeScoreCalculator;

impl CompositeScoreCalculator {
    /// Calcula la métrica compuesta y el ratio Capital / MaxDD:
    /// - `curve_r2`: Coeficiente de determinación R² de la recta de regresión de la curva de equity (0.0 a 1.0).
    /// - `initial_capital`: Capital inicial de la prueba. Si `<= 0.0` o no finito, retorna `(0.0, 0.0)`.
    /// - `final_capital`: Capital total acumulado al finalizar el periodo simulado.
    /// - `max_drawdown_nominal`: Máxima caída nominal en USD experimentada desde un pico.
    /// - `total_trades`: Número total de operaciones cerradas ejecutadas.
    ///
    /// Retorna una tupla: (capital_to_maxdd_ratio, composite_score)
    pub fn calculate(
        curve_r2: f64,
        initial_capital: f64,
        final_capital: f64,
        max_drawdown_nominal: f64,
        total_trades: usize,
    ) -> (f64, f64) {
        // C-38: Salvaguarda institucional ante initial_capital <= 0.0 o no finito.
        // Si el capital inicial es inválido o no positivo, se retorna el valor neutro documentado (0.0, 0.0)
        // para prevenir divisiones inválidas o corrupción en penalty_score.
        if !initial_capital.is_finite() || initial_capital <= 0.0 {
            return (0.0, 0.0);
        }

        // Sanitización ante valores no finitos en capital final o drawdown
        if !final_capital.is_finite() || !max_drawdown_nominal.is_finite() {
            return (0.0, 0.0);
        }

        // Linealidad acotada en [0.0, 1.0] para evitar distorsiones por pendientes negativas
        let linearity = if curve_r2.is_finite() {
            curve_r2.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Si no hubo operaciones ejecutadas, el score es neutro 0.0
        if total_trades == 0 {
            return (0.0, 0.0);
        }

        // Prevención de división por cero o explosión matemática si nunca hubo drawdown:
        // Se asume un piso mínimo de drawdown proporcional (0.5% del capital inicial o mínimo $1.0)
        let safe_max_dd = if max_drawdown_nominal <= 0.0 {
            (initial_capital * 0.005).max(1.0)
        } else {
            max_drawdown_nominal
        };

        let net_pnl = final_capital - initial_capital;

        if net_pnl > 0.0 {
            // Rendimiento estrictamente positivo:
            // raw_ratio: Capital acumulado respecto al Max Drawdown sufrido
            let raw_ratio = final_capital / safe_max_dd;
            let capital_to_maxdd = raw_ratio.min(100.0);

            // Normalización armónica: ratio escalado en [0.0, 1.0] para equilibrar el 50% con R²
            let normalized_cap_dd = (capital_to_maxdd / 100.0).clamp(0.0, 1.0);
            let composite_score = (0.5 * linearity) + (0.5 * normalized_cap_dd);
            (capital_to_maxdd, composite_score)
        } else if net_pnl == 0.0 {
            // Breakeven sin ganancia ni pérdida
            (0.0, 0.0)
        } else {
            // BUG-F07: Si la cuenta termina con pérdidas netas (net_pnl < 0), la puntuación DEBE ser
            // estrictamente negativa (< 0.0) sin excepción.
            // (1.0 - linearity) penaliza la irregularidad de la curva de pérdida (0.0 a 1.0)
            // loss_impact.abs() penaliza la magnitud de la pérdida y el drawdown sufrido
            let loss_impact = (net_pnl - safe_max_dd) / initial_capital;
            let bounded_loss = loss_impact.clamp(-100.0, 0.0);
            let penalty_score = -((1.0 - linearity) + loss_impact.abs()).max(1e-6);
            (bounded_loss, penalty_score)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_trades_returns_zero() {
        let (ratio, score) = CompositeScoreCalculator::calculate(0.9, 10_000.0, 10_500.0, 200.0, 0);
        assert_eq!(ratio, 0.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_profitable_strategy() {
        let (ratio, score) = CompositeScoreCalculator::calculate(0.8, 10_000.0, 11_000.0, 500.0, 10);
        assert!(ratio > 0.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_losing_gradient_preserves_ranking() {
        // BUG-18: Estrategia defensiva con pérdida mínima debe tener mejor score que pérdida severa
        let (ratio_defensive, score_defensive) = CompositeScoreCalculator::calculate(
            0.5, 10_000.0, 9_990.0, 50.0, 5
        );
        let (ratio_severe_loss, score_severe_loss) = CompositeScoreCalculator::calculate(
            0.5, 10_000.0, 5_000.0, 5_000.0, 5
        );

        assert!(score_defensive > score_severe_loss);
        assert!(ratio_defensive > ratio_severe_loss);
    }

    #[test]
    fn test_higher_drawdown_penalized_in_loss() {
        // BUG-01: Para un mismo nivel de pérdida neta, un mayor drawdown debe penalizarse estrictamente.
        // Estrategia A: Pérdida de -$1000 con Drawdown de $1000
        let (ratio_a, score_a) = CompositeScoreCalculator::calculate(
            0.5, 10_000.0, 9_000.0, 1_000.0, 10
        );
        // Estrategia B: Pérdida idéntica de -$1000 pero con Drawdown catastrófico de $5000
        let (ratio_b, score_b) = CompositeScoreCalculator::calculate(
            0.5, 10_000.0, 9_000.0, 5_000.0, 10
        );

        // Estrategia A debe tener un score estrictamente superior (menos castigado) que Estrategia B
        assert!(score_a > score_b, "Estrategia con menor Drawdown debe tener mejor score: {} vs {}", score_a, score_b);
        assert!(ratio_a > ratio_b, "Ratio A debe ser superior a Ratio B: {} vs {}", ratio_a, ratio_b);
    }

    #[test]
    fn test_bug_f07_strictly_non_positive_when_losing() {
        // Estrategia que pierde dinero suavemente con R^2 casi perfecto (0.99)
        let (_, score) = CompositeScoreCalculator::calculate(0.99, 10_000.0, 9_500.0, 500.0, 10);
        assert!(score < 0.0, "Score en pérdidas debe ser estrictamente negativo incluso con R^2 alto: {}", score);

        // Breakeven
        let (_, score_be) = CompositeScoreCalculator::calculate(0.99, 10_000.0, 10_000.0, 500.0, 10);
        assert_eq!(score_be, 0.0, "Score en breakeven debe ser 0.0");
    }

    #[test]
    fn test_bug_f07_harmonic_weighting_r2() {
        // La linealidad R² debe mover el puntaje en un 50% efectivo
        let (_, score_high_r2) = CompositeScoreCalculator::calculate(1.0, 10_000.0, 11_000.0, 500.0, 10);
        let (_, score_low_r2) = CompositeScoreCalculator::calculate(0.0, 10_000.0, 11_000.0, 500.0, 10);
        assert!((score_high_r2 - score_low_r2 - 0.5).abs() < 1e-4, "R² debe aportar un peso representativo de 0.5");
    }

    #[test]
    fn test_c38_initial_capital_zero_returns_neutral_zero() {
        // Capital inicial 0.0 debe devolver (0.0, 0.0) de forma segura y sin divisiones por cero
        let (ratio, score) = CompositeScoreCalculator::calculate(0.8, 0.0, 100.0, 10.0, 5);
        assert_eq!(ratio, 0.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_c38_initial_capital_negative_returns_neutral_zero() {
        // Capital inicial negativo debe devolver (0.0, 0.0)
        let (ratio, score) = CompositeScoreCalculator::calculate(0.8, -1000.0, -900.0, 50.0, 5);
        assert_eq!(ratio, 0.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_c38_initial_capital_nan_inf_returns_neutral_zero() {
        // Valores no finitos en capital inicial
        let (ratio_nan, score_nan) = CompositeScoreCalculator::calculate(0.8, f64::NAN, 1000.0, 50.0, 5);
        assert_eq!(ratio_nan, 0.0);
        assert_eq!(score_nan, 0.0);

        let (ratio_inf, score_inf) = CompositeScoreCalculator::calculate(0.8, f64::INFINITY, 1000.0, 50.0, 5);
        assert_eq!(ratio_inf, 0.0);
        assert_eq!(score_inf, 0.0);
    }
}

/// Controlador Causal de Parada Temprana (Early Stopping Controller)
///
/// Monitoriza una métrica de validación estrictamente causal evaluada paso a paso
/// en cada iteración de boosting (m = 0, 1, ..., M-1).
///
/// Si la métrica no mejora tras un número consecutivo de iteraciones equivalente a `patience`,
/// el controlador detiene el entrenamiento y señala el índice óptimo m* para truncar el ensamble,
/// previniendo el overfitting tardío.
#[derive(Debug, Clone)]
pub struct EarlyStoppingController {
    pub patience: usize,
    pub min_delta: f64,
    pub minimize: bool,
    best_score: f64,
    best_iteration: usize,
    stagnant_iterations: usize,
    is_stopped: bool,
}

impl EarlyStoppingController {
    /// Construye un nuevo controlador
    /// - `patience`: número de rondas de tolerancia sin mejora
    /// - `min_delta`: umbral mínimo absoluto de mejora para considerarse progreso real
    /// - `minimize`: `true` si la métrica es una función de pérdida (ej. MSE, log-loss),
    ///               `false` si es una métrica de fitness (ej. Sharpe ratio, información)
    pub fn new(patience: usize, min_delta: f64, minimize: bool) -> Self {
        let initial_score = if minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        Self {
            patience: patience.max(1),
            min_delta: min_delta.max(0.0),
            minimize,
            best_score: initial_score,
            best_iteration: 0,
            stagnant_iterations: 0,
            is_stopped: false,
        }
    }

    /// Actualiza el estado con la métrica observada en la iteración actual.
    ///
    /// Retorna `true` si se ha alcanzado la paciencia máxima y el entrenamiento debe abortar.
    pub fn update(&mut self, iteration: usize, current_score: f64) -> bool {
        if !current_score.is_finite() {
            self.stagnant_iterations += 1;
            if self.stagnant_iterations >= self.patience {
                self.is_stopped = true;
                return true;
            }
            return false;
        }

        let is_improvement = if self.minimize {
            current_score < (self.best_score - self.min_delta)
        } else {
            current_score > (self.best_score + self.min_delta)
        };

        if is_improvement || self.best_score.is_infinite() {
            self.best_score = current_score;
            self.best_iteration = iteration;
            self.stagnant_iterations = 0;
        } else {
            self.stagnant_iterations += 1;
            if self.stagnant_iterations >= self.patience {
                self.is_stopped = true;
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    #[inline]
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    #[inline]
    pub fn is_stopped(&self) -> bool {
        self.is_stopped
    }

    #[inline]
    pub fn stagnant_steps(&self) -> usize {
        self.stagnant_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_minimization_triggers() {
        let mut controller = EarlyStoppingController::new(3, 0.01, true);

        // Iteración 0: 1.0 (nuevo mejor)
        assert!(!controller.update(0, 1.0));
        assert_eq!(controller.best_iteration(), 0);

        // Iteración 1: 0.8 (nuevo mejor)
        assert!(!controller.update(1, 0.8));
        assert_eq!(controller.best_iteration(), 1);

        // Iteración 2: 0.805 (no mejora, stagnant 1)
        assert!(!controller.update(2, 0.805));
        // Iteración 3: 0.810 (no mejora, stagnant 2)
        assert!(!controller.update(3, 0.810));
        // Iteración 4: 0.820 (no mejora, stagnant 3 >= patience -> stop!)
        let should_stop = controller.update(4, 0.820);
        assert!(should_stop);
        assert!(controller.is_stopped());
        // El mejor m* sigue siendo la iteración 1
        assert_eq!(controller.best_iteration(), 1);
        assert_eq!(controller.best_score(), 0.8);
    }
}

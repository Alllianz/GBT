/// Regulador de exposición por Drawdown (Drawdown Governor).
/// Desescala la exposición conforme el drawdown actual se aproxima al drawdown máximo tolerable,
/// preservando el capital y evitando la ruina mediante reducción progresiva no lineal del tamaño de lote.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawdownGovernor {
    /// Drawdown máximo tolerable (ej: 0.15 para 15% de DD)
    pub max_tolerable_drawdown: f64,
    /// Exponente de penalización progresiva (gamma: 1.0 lineal, > 1.0 conservador)
    pub gamma: f64,
    /// Escala mínima residual permitida (D_min, ej: 0.0 para cesar operaciones o 0.10)
    pub min_scale: f64,
}

impl DrawdownGovernor {
    pub fn new(max_tolerable_drawdown: f64, gamma: f64, min_scale: f64) -> Self {
        let safe_max_dd = if max_tolerable_drawdown.is_finite() && max_tolerable_drawdown > 0.0 {
            max_tolerable_drawdown.clamp(0.01, 0.99)
        } else {
            0.15
        };
        let safe_gamma = if gamma.is_finite() && gamma > 0.0 {
            gamma
        } else {
            1.5
        };
        let safe_min = if min_scale.is_finite() && min_scale >= 0.0 {
            min_scale.clamp(0.0, 1.0)
        } else {
            0.0
        };

        Self {
            max_tolerable_drawdown: safe_max_dd,
            gamma: safe_gamma,
            min_scale: safe_min,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.15, 1.5, 0.0)
    }

    /// Calcula el factor multiplicador de reducción en [min_scale, 1.0].
    ///
    /// Fórmula:
    ///   D(dd) = max( min_scale, 1.0 - (DrawdownActual / MaxDrawdownTolerado)^gamma )
    pub fn calculate_scale(&self, current_drawdown: f64) -> f64 {
        if !current_drawdown.is_finite() {
            return self.min_scale;
        }

        // Si no hay drawdown o es despreciable, exposición plena
        if current_drawdown <= 0.0 {
            return 1.0;
        }

        // Si el drawdown supera o iguala la tolerancia máxima, se aplica la escala mínima
        if current_drawdown >= self.max_tolerable_drawdown {
            return self.min_scale;
        }

        let ratio = (current_drawdown / self.max_tolerable_drawdown).clamp(0.0, 1.0);
        let factor = 1.0 - ratio.powf(self.gamma);

        if !factor.is_finite() {
            return self.min_scale;
        }

        factor.clamp(self.min_scale, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_drawdown_returns_full_scale() {
        let governor = DrawdownGovernor::new(0.15, 1.5, 0.0);
        assert_eq!(governor.calculate_scale(0.0), 1.0);
        assert_eq!(governor.calculate_scale(-0.05), 1.0);
    }

    #[test]
    fn test_max_drawdown_returns_min_scale() {
        let governor = DrawdownGovernor::new(0.15, 1.5, 0.0);
        assert_eq!(governor.calculate_scale(0.15), 0.0);
        assert_eq!(governor.calculate_scale(0.20), 0.0);
    }

    #[test]
    fn test_intermediate_drawdown_scaling() {
        // Linear gamma = 1.0
        let governor = DrawdownGovernor::new(0.20, 1.0, 0.0);
        // DD = 0.10 (50% de la tolerancia) -> factor = 1.0 - 0.50 = 0.50
        let scale = governor.calculate_scale(0.10);
        assert!((scale - 0.50).abs() < 1e-4);
    }

    #[test]
    fn test_nan_drawdown_returns_min_scale() {
        let governor = DrawdownGovernor::with_defaults();
        assert_eq!(governor.calculate_scale(f64::NAN), 0.0);
    }
}

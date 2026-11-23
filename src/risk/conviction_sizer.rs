/// Modulador de tamaño de posición por Convicción del Modelo de ML.
/// Escala el nocional de la orden proporcionalmente a la distancia entre la certeza
/// predicha por el modelo de ML (|y_hat|) y el umbral de activación (tau).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvictionSizer {
    /// Umbral mínimo de certeza o activación (tau, ej: 0.50)
    pub activation_threshold: f64,
    /// Factor de sensibilidad de escala (kappa, ej: 0.50)
    pub kappa: f64,
    /// Multiplicador mínimo permitido (s_min, ej: 0.50)
    pub min_multiplier: f64,
    /// Multiplicador máximo permitido (s_max, ej: 1.50)
    pub max_multiplier: f64,
}

impl ConvictionSizer {
    pub fn new(
        activation_threshold: f64,
        kappa: f64,
        min_multiplier: f64,
        max_multiplier: f64,
    ) -> Self {
        let safe_tau = if activation_threshold.is_finite() && activation_threshold > 0.0 {
            activation_threshold.clamp(0.01, 1.0)
        } else {
            0.50
        };
        let safe_kappa = if kappa.is_finite() && kappa >= 0.0 {
            kappa
        } else {
            0.50
        };
        let safe_min = if min_multiplier.is_finite() && min_multiplier >= 0.0 {
            min_multiplier
        } else {
            0.50
        };
        let safe_max = if max_multiplier.is_finite() && max_multiplier >= safe_min {
            max_multiplier
        } else {
            1.50_f64.max(safe_min)
        };

        Self {
            activation_threshold: safe_tau,
            kappa: safe_kappa,
            min_multiplier: safe_min,
            max_multiplier: safe_max,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.50, 0.50, 0.50, 1.50)
    }

    /// Calcula el multiplicador de escala a partir de la intensidad predicha de la señal (|y_hat|).
    ///
    /// Fórmula:
    ///   s_t = clamp( 1.0 + kappa * ((|y_hat| - tau) / tau), s_min, s_max )
    pub fn calculate_multiplier(&self, signal_strength: f64) -> f64 {
        if !signal_strength.is_finite() {
            return 1.0;
        }

        let tau = self.activation_threshold;
        let y_abs = signal_strength.abs();

        let normalized_diff = (y_abs - tau) / tau;
        let raw_mult = 1.0 + self.kappa * normalized_diff;

        if !raw_mult.is_finite() {
            return 1.0;
        }

        raw_mult.clamp(self.min_multiplier, self.max_multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conviction_at_threshold_is_neutral() {
        let sizer = ConvictionSizer::new(0.50, 0.50, 0.50, 1.50);
        let m = sizer.calculate_multiplier(0.50);
        assert!((m - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_conviction_above_threshold_scales_up() {
        // tau = 0.50, y = 1.00 -> diff = (1.0 - 0.5) / 0.5 = 1.0
        // mult = 1.0 + 0.50 * 1.0 = 1.50
        let sizer = ConvictionSizer::new(0.50, 0.50, 0.50, 1.50);
        let m = sizer.calculate_multiplier(1.00);
        assert!((m - 1.50).abs() < 1e-4);
    }

    #[test]
    fn test_conviction_below_threshold_scales_down() {
        // tau = 0.50, y = 0.25 -> diff = (0.25 - 0.50) / 0.50 = -0.50
        // mult = 1.0 + 0.50 * (-0.50) = 0.75
        let sizer = ConvictionSizer::new(0.50, 0.50, 0.50, 1.50);
        let m = sizer.calculate_multiplier(0.25);
        assert!((m - 0.75).abs() < 1e-4);
    }

    #[test]
    fn test_conviction_nan_input_returns_neutral() {
        let sizer = ConvictionSizer::with_defaults();
        assert_eq!(sizer.calculate_multiplier(f64::NAN), 1.0);
    }
}

/// Dimensionador por objetivo constante de volatilidad anualizada (Volatility Targeting).
/// Normaliza la exposición al mercado para mantener un nivel de riesgo uniforme a lo largo del tiempo,
/// desapalancando en regímenes de alta turbulencia y apalancando en regímenes de calma.
#[derive(Debug, Clone, PartialEq)]
pub struct VolatilityTargetingSizer {
    /// Volatilidad objetivo anualizada deseada (ej: 0.15 para 15%)
    pub target_annual_vol: f64,
    /// Límite inferior de ponderación de posición
    pub min_weight: f64,
    /// Límite superior de apalancamiento / ponderación
    pub max_weight: f64,
    /// Factor de escala para anualizar la volatilidad del timeframe base cripto 24/7 (ej: sqrt(365 * 24) = sqrt(8760))
    pub annualization_factor: f64,
}

impl VolatilityTargetingSizer {
    pub fn new(
        target_annual_vol: f64,
        min_weight: f64,
        max_weight: f64,
        annualization_factor: f64,
    ) -> Self {
        let safe_target = if target_annual_vol.is_finite() && target_annual_vol > 0.0 {
            target_annual_vol
        } else {
            0.15
        };
        let safe_min = if min_weight.is_finite() && min_weight >= 0.0 {
            min_weight
        } else {
            0.05
        };
        let safe_max = if max_weight.is_finite() && max_weight >= safe_min {
            max_weight
        } else {
            2.0_f64.max(safe_min)
        };
        let safe_ann = if annualization_factor.is_finite() && annualization_factor > 0.0 {
            annualization_factor
        } else {
            (365.0 * 24.0_f64).sqrt()
        };

        Self {
            target_annual_vol: safe_target,
            min_weight: safe_min,
            max_weight: safe_max,
            annualization_factor: safe_ann,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.15, 0.05, 2.0, (365.0 * 24.0_f64).sqrt())
    }

    /// Construye un dimensionador de volatilidad tipado para un timeframe específico (C-21).
    pub fn for_timeframe(
        target_annual_vol: f64,
        min_weight: f64,
        max_weight: f64,
        timeframe: crate::types::Timeframe,
    ) -> Self {
        Self::new(
            target_annual_vol,
            min_weight,
            max_weight,
            timeframe.annualization_factor(),
        )
    }

    /// Construye un dimensionador de volatilidad tipado para un número explícito de períodos por año (C-21).
    pub fn for_periods_per_year(
        target_annual_vol: f64,
        min_weight: f64,
        max_weight: f64,
        periods_per_year: f64,
    ) -> Self {
        let ann_factor = if periods_per_year > 0.0 && periods_per_year.is_finite() {
            periods_per_year.sqrt()
        } else {
            (365.0 * 24.0_f64).sqrt()
        };
        Self::new(target_annual_vol, min_weight, max_weight, ann_factor)
    }

    /// Calcula la ponderación objetivo a partir de la volatilidad instantánea estimada de la barra actual.
    ///
    /// Fórmula:
    ///   Weight_t = clamp( sigma_target / (sigma_bar * annualization_factor), min_weight, max_weight )
    pub fn calculate_weight(&self, current_bar_volatility: f64) -> f64 {
        if !current_bar_volatility.is_finite() || current_bar_volatility <= 0.0 {
            return self.min_weight;
        }

        let annual_vol = current_bar_volatility * self.annualization_factor;
        if !annual_vol.is_finite() || annual_vol <= 1e-8 {
            return self.max_weight;
        }

        let raw_weight = self.target_annual_vol / annual_vol;
        if !raw_weight.is_finite() {
            return self.min_weight;
        }

        raw_weight.clamp(self.min_weight, self.max_weight)
    }

    /// Estima la volatilidad realizada causal (desviación estándar muestral de log-retornos)
    /// a partir de una serie de precios de cierre conocidos (t <= T).
    pub fn calculate_realized_volatility(prices: &[f64]) -> Option<f64> {
        if prices.len() < 2 {
            return None;
        }

        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            let p_prev = prices[i - 1];
            let p_curr = prices[i];
            if p_prev > 0.0 && p_curr > 0.0 && p_prev.is_finite() && p_curr.is_finite() {
                returns.push((p_curr / p_prev).ln());
            }
        }

        if returns.len() < 2 {
            return None;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let sum_sq_diff: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum();
        let sample_var = sum_sq_diff / (n - 1.0);

        if sample_var.is_finite() && sample_var >= 0.0 {
            Some(sample_var.sqrt())
        } else {
            None
        }
    }

    /// Estima la volatilidad relativa instantánea usando el Average True Range normalizado: ATR / Precio.
    pub fn calculate_atr_volatility(atr: f64, current_price: f64) -> Option<f64> {
        if atr.is_finite() && atr > 0.0 && current_price.is_finite() && current_price > 0.0 {
            Some(atr / current_price)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_targeting_scaling() {
        let ann_factor = (252.0 * 24.0_f64).sqrt(); // ~77.7688
        let sizer = VolatilityTargetingSizer::new(0.15, 0.10, 2.0, ann_factor);

        // Volatilidad por vela que anualizada da exactamente 0.15 (15%)
        let exact_bar_vol = 0.15 / ann_factor;
        let weight = sizer.calculate_weight(exact_bar_vol);
        assert!((weight - 1.0).abs() < 1e-4);

        // Volatilidad el doble de alta (30% anualizada) -> peso debe reducirse a 0.50
        let high_vol = exact_bar_vol * 2.0;
        let weight_high = sizer.calculate_weight(high_vol);
        assert!((weight_high - 0.50).abs() < 1e-4);

        // Volatilidad muy baja (5% anualizada) -> peso debe subir a 3.0, acotado por max_weight (2.0)
        let low_vol = exact_bar_vol / 3.0;
        let weight_low = sizer.calculate_weight(low_vol);
        assert_eq!(weight_low, 2.0);
    }

    #[test]
    fn test_invalid_volatility_inputs() {
        let sizer = VolatilityTargetingSizer::with_defaults();
        assert_eq!(sizer.calculate_weight(-0.5), sizer.min_weight);
        assert_eq!(sizer.calculate_weight(0.0), sizer.min_weight);
        assert_eq!(sizer.calculate_weight(f64::NAN), sizer.min_weight);
        assert_eq!(sizer.calculate_weight(f64::INFINITY), sizer.min_weight);
    }

    #[test]
    fn test_calculate_realized_volatility() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 102.5];
        let vol = VolatilityTargetingSizer::calculate_realized_volatility(&prices);
        assert!(vol.is_some());
        assert!(vol.unwrap() > 0.0);

        // Menos de 2 precios debe ser None
        assert!(VolatilityTargetingSizer::calculate_realized_volatility(&[100.0]).is_none());
    }

    #[test]
    fn test_calculate_atr_volatility() {
        let vol = VolatilityTargetingSizer::calculate_atr_volatility(2.0, 100.0);
        assert_eq!(vol, Some(0.02));

        assert!(VolatilityTargetingSizer::calculate_atr_volatility(0.0, 100.0).is_none());
        assert!(VolatilityTargetingSizer::calculate_atr_volatility(-1.0, 100.0).is_none());
        assert!(VolatilityTargetingSizer::calculate_atr_volatility(2.0, 0.0).is_none());
    }

    #[test]
    fn test_c21_volatility_targeting_timeframes() {
        use crate::types::Timeframe;

        // 15m => 35040 períodos
        let sizer_15m = VolatilityTargetingSizer::for_timeframe(0.15, 0.05, 2.0, Timeframe::M15);
        assert_eq!(sizer_15m.annualization_factor, (35040.0_f64).sqrt());

        // 1H => 8760 períodos
        let sizer_1h = VolatilityTargetingSizer::for_timeframe(0.15, 0.05, 2.0, Timeframe::H1);
        assert_eq!(sizer_1h.annualization_factor, (8760.0_f64).sqrt());

        // 4H => 2190 períodos
        let sizer_4h = VolatilityTargetingSizer::for_timeframe(0.15, 0.05, 2.0, Timeframe::H4);
        assert_eq!(sizer_4h.annualization_factor, (2190.0_f64).sqrt());

        // 1D => 365 períodos
        let sizer_1d = VolatilityTargetingSizer::for_timeframe(0.15, 0.05, 2.0, Timeframe::D1);
        assert_eq!(sizer_1d.annualization_factor, (365.0_f64).sqrt());
    }
}

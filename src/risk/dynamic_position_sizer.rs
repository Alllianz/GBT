use super::conviction_sizer::ConvictionSizer;
use super::drawdown_governor::DrawdownGovernor;
use super::kelly_criterion::FractionalKellySizer;
use super::risk_config::RiskConfig;
use super::risk_context::RiskEvaluationContext;
use super::volatility_targeting::VolatilityTargetingSizer;

/// Orquestador central del subsistema institucional de dimensionamiento dinámico de posición.
/// Integra armónicamente:
/// 1. Criterio de Kelly Fraccional (Fractional Kelly)
/// 2. Control de Volatilidad Objetivo (Volatility Targeting)
/// 3. Modulación por Convicción ML (Conviction Sizing)
/// 4. Gobernador de Preservación de Capital (Drawdown Governor)
#[derive(Debug, Clone)]
pub struct DynamicPositionSizer {
    pub config: RiskConfig,
    pub kelly_sizer: FractionalKellySizer,
    pub vol_sizer: VolatilityTargetingSizer,
    pub conviction_sizer: ConvictionSizer,
    pub drawdown_governor: DrawdownGovernor,
}

impl DynamicPositionSizer {
    pub fn new(config: RiskConfig) -> Self {
        let kelly_sizer = FractionalKellySizer::new(
            config.kelly_fraction,
            config.kelly_window,
            config.kelly_min_fraction,
            config.kelly_max_fraction,
            config.kelly_min_trades,
            config.kelly_default_fraction,
        );

        let vol_sizer = VolatilityTargetingSizer::new(
            config.target_annual_vol,
            config.vol_min_weight,
            config.vol_max_weight,
            config.annualization_factor,
        );

        let conviction_sizer = ConvictionSizer::new(
            config.conviction_threshold,
            config.conviction_kappa,
            config.conviction_min_scale,
            config.conviction_max_scale,
        );

        let drawdown_governor = DrawdownGovernor::new(
            config.max_tolerable_drawdown,
            config.drawdown_gamma,
            config.drawdown_min_scale,
        );

        Self {
            config,
            kelly_sizer,
            vol_sizer,
            conviction_sizer,
            drawdown_governor,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(RiskConfig::default())
    }

    /// Calcula el valor nocional objetivo final en USD para una nueva orden.
    /// Garantiza rigurosamente causalidad temporal (t <= T) y robustez anti-NaN.
    pub fn calculate_notional(&self, ctx: &RiskEvaluationContext) -> f64 {
        if !ctx.equity.is_finite()
            || ctx.equity <= 0.0
            || !ctx.leverage.is_finite()
            || ctx.leverage <= 0.0
            || !ctx.exec_price.is_finite()
            || ctx.exec_price <= 0.0
        {
            return 0.0;
        }

        let effective_leverage = ctx.leverage.min(self.config.max_leverage);
        let max_notional_capacity = ctx.equity * effective_leverage;

        // C-21: Resolver dimensionador de volatilidad efectivo según anualización explícita del contexto o configuración
        let effective_ann = ctx
            .annualization_factor
            .or_else(|| ctx.timeframe.map(|tf| tf.annualization_factor()))
            .unwrap_or(self.config.annualization_factor);

        let effective_vol_sizer = if (effective_ann - self.vol_sizer.annualization_factor).abs() > 1e-9 {
            VolatilityTargetingSizer::new(
                self.vol_sizer.target_annual_vol,
                self.vol_sizer.min_weight,
                self.vol_sizer.max_weight,
                effective_ann,
            )
        } else {
            self.vol_sizer.clone()
        };

        // 1. Determinar el peso base de asignación
        let base_weight = match (self.config.use_kelly, self.config.use_volatility_targeting) {
            (true, true) => {
                let f_kelly = self.kelly_sizer.calculate_fraction(ctx.closed_trades);
                let w_vol = if let Some(vol) = ctx.current_volatility {
                    effective_vol_sizer.calculate_weight(vol)
                } else {
                    1.0
                };
                // Modulación de Kelly por régimen de volatilidad relativa:
                // Si la volatilidad es alta (w_vol < 1.0), el tamaño Kelly se contrae.
                // Si la volatilidad es baja (w_vol > 1.0), se permite mayor asignación hasta effective_leverage.
                (f_kelly * w_vol).clamp(self.config.kelly_min_fraction, effective_leverage)
            }
            (true, false) => self.kelly_sizer.calculate_fraction(ctx.closed_trades),
            (false, true) => {
                if let Some(vol) = ctx.current_volatility {
                    effective_vol_sizer.calculate_weight(vol)
                } else {
                    self.config.kelly_default_fraction
                }
            }
            (false, false) => self.config.kelly_default_fraction,
        };

        // 2. Modulador de Convicción ML
        let conviction_factor = if self.config.use_conviction {
            self.conviction_sizer.calculate_multiplier(ctx.signal_strength)
        } else {
            1.0
        };

        // 3. Freno de Drawdown Governor
        let dd_governor_factor = if self.config.use_drawdown_governor {
            self.drawdown_governor.calculate_scale(ctx.current_drawdown)
        } else {
            1.0
        };

        // 4. Cálculo del nocional integrado
        let mut notional = ctx.equity * base_weight * conviction_factor * dd_governor_factor;

        // 5. Aplicar límites institucionales duros
        notional = notional.min(max_notional_capacity);

        if let Some(max_usd) = self.config.max_notional_usd {
            if max_usd.is_finite() && max_usd > 0.0 {
                notional = notional.min(max_usd);
            }
        }

        if notional.is_finite() && notional > 0.0 {
            notional
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompletedTrade;

    #[test]
    fn test_dynamic_position_sizer_integration() {
        let config = RiskConfig::default();
        let sizer = DynamicPositionSizer::new(config);

        let trades: Vec<CompletedTrade> = Vec::new();
        let ctx = RiskEvaluationContext::new(
            10_000.0,
            2.0,
            100.0,
            &trades,
            None,
            0.0,  // 0% DD
            0.50, // Convicción neutral
            None,
        );

        let notional = sizer.calculate_notional(&ctx);
        // Default fraction = 0.05, equity = 10,000 -> 500 USD
        assert!((notional - 500.0).abs() < 1e-4);
    }

    #[test]
    fn test_dynamic_sizer_drawdown_brake() {
        let config = RiskConfig {
            max_tolerable_drawdown: 0.15,
            drawdown_min_scale: 0.0,
            ..Default::default()
        };
        let sizer = DynamicPositionSizer::new(config);

        let trades: Vec<CompletedTrade> = Vec::new();
        // Con drawdown de 15% (máximo tolerable), el freno debe anular el nocional (0.0 USD)
        let ctx = RiskEvaluationContext::new(
            8_500.0,
            2.0,
            100.0,
            &trades,
            None,
            0.15,
            0.50,
            None,
        );

        let notional = sizer.calculate_notional(&ctx);
        assert_eq!(notional, 0.0);
    }

    #[test]
    fn test_dynamic_sizer_conviction_scaling() {
        let config = RiskConfig {
            use_volatility_targeting: false,
            use_drawdown_governor: false,
            ..Default::default()
        };
        let sizer = DynamicPositionSizer::new(config);

        let trades: Vec<CompletedTrade> = Vec::new();
        // Convicción máxima: 1.0 -> factor 1.50
        let ctx = RiskEvaluationContext::new(
            10_000.0,
            2.0,
            100.0,
            &trades,
            None,
            0.0,
            1.00,
            None,
        );

        let notional = sizer.calculate_notional(&ctx);
        // Base 500.0 * 1.50 = 750.0 USD
        assert!((notional - 750.0).abs() < 1e-4);
    }
}

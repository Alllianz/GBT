use crate::risk::{
    DynamicPositionSizer, FractionalKellySizer, RiskEvaluationContext, VolatilityTargetingSizer,
};
use crate::types::PositionSizing;

pub struct PositionSizer;

impl PositionSizer {
    /// Calcula el valor nocional objetivo en USD a partir de la política de tamaño de posición,
    /// el equity actual, el capital inicial, la distancia de stop loss y el contexto de riesgo causal.
    ///
    /// BUG-F10: Sanitización robusta contra NaNs, infinitos y montos no positivos.
    pub fn calculate_position_size(
        sizing: &PositionSizing,
        initial_capital: f64,
        equity: f64,
        leverage: f64,
        exec_price: f64,
        dynamic_sl: Option<f64>,
        fixed_sl_pct: Option<f64>,
        risk_context: Option<&RiskEvaluationContext>,
    ) -> f64 {
        if !equity.is_finite()
            || equity <= 0.0
            || !leverage.is_finite()
            || leverage <= 0.0
            || !exec_price.is_finite()
            || exec_price <= 0.0
        {
            return 0.0;
        }

        let safe_initial = if initial_capital.is_finite() && initial_capital > 0.0 {
            initial_capital
        } else {
            equity
        };
        let max_notional = equity * leverage;

        let target = match sizing {
            PositionSizing::PercentOfInitialCapital(pct) => {
                // C-35: Clamp inferior reducido a 0.0001 (0.01%) permitiendo micro-dimensionamiento institucional.
                // Valores <= 0.0 o no finitos se rechazan con factor 0.0.
                let safe_pct = if pct.is_finite() && *pct > 0.0 {
                    pct.clamp(0.0001, 1.0)
                } else {
                    0.0
                };
                safe_initial * safe_pct * leverage
            }
            PositionSizing::PercentOfEquity(pct) => {
                // C-35: Clamp inferior reducido a 0.0001 (0.01%) permitiendo micro-dimensionamiento institucional.
                // Valores <= 0.0 o no finitos se rechazan con factor 0.0.
                let safe_pct = if pct.is_finite() && *pct > 0.0 {
                    pct.clamp(0.0001, 1.0)
                } else {
                    0.0
                };
                equity * safe_pct * leverage
            }
            PositionSizing::FixedAmount(amount) => {
                let safe_amount = if amount.is_finite() && *amount > 0.0 {
                    *amount
                } else {
                    0.0
                };
                safe_amount * leverage
            }
            PositionSizing::RiskBasedPercent(risk_pct) => {
                let safe_risk = if risk_pct.is_finite() {
                    risk_pct.clamp(0.001, 0.10)
                } else {
                    0.01
                };
                let mut sl_dist = if let Some(sl) = dynamic_sl {
                    if sl.is_finite() && sl > 0.0 {
                        ((exec_price - sl).abs() / exec_price).max(0.002)
                    } else {
                        fixed_sl_pct.unwrap_or(0.02).max(0.002)
                    }
                } else {
                    fixed_sl_pct.unwrap_or(0.02).max(0.002)
                };
                if !sl_dist.is_finite() || sl_dist <= 0.0001 {
                    sl_dist = 0.02;
                }
                let risk_amount = equity * safe_risk;
                risk_amount / sl_dist
            }
            PositionSizing::DynamicRisk(cfg) => {
                let sizer = DynamicPositionSizer::new(cfg.clone());
                if let Some(ctx) = risk_context {
                    // C-06: Sincronización causal: si el equity del contexto difiere del equity
                    // actualizado post-cierre, emplear el equity real del portfolio.
                    if (ctx.equity - equity).abs() > 1e-6 {
                        let mut synced_ctx = ctx.clone();
                        synced_ctx.equity = equity;
                        sizer.calculate_notional(&synced_ctx)
                    } else {
                        sizer.calculate_notional(ctx)
                    }
                } else {
                    let fallback_ctx = RiskEvaluationContext::baseline(equity, leverage, exec_price);
                    sizer.calculate_notional(&fallback_ctx)
                }
            }
            PositionSizing::VolatilityTargeting {
                target_annual_vol,
                max_leverage: max_lev,
            } => {
                let safe_max_lev = if max_lev.is_finite() && *max_lev > 0.0 {
                    max_lev.min(leverage)
                } else {
                    leverage
                };
                // C-21: Derivar el factor de anualización tipado desde el contexto de riesgo o base cripto 24/7/365
                let ann_factor = risk_context
                    .and_then(|ctx| ctx.annualization_factor)
                    .or_else(|| {
                        risk_context
                            .and_then(|ctx| ctx.timeframe)
                            .map(|tf| tf.annualization_factor())
                    })
                    .unwrap_or_else(|| (365.0 * 24.0_f64).sqrt());

                let sizer = VolatilityTargetingSizer::new(
                    *target_annual_vol,
                    0.0,
                    safe_max_lev,
                    ann_factor,
                );
                let weight = if let Some(ctx) = risk_context {
                    if let Some(vol) = ctx.current_volatility {
                        sizer.calculate_weight(vol)
                    } else {
                        1.0_f64.min(safe_max_lev)
                    }
                } else {
                    1.0_f64.min(safe_max_lev)
                };
                equity * weight
            }
            PositionSizing::FractionalKelly { fraction, window } => {
                let sizer = FractionalKellySizer::new(*fraction, *window, 0.01, 1.0, 20, 0.05);
                let frac = if let Some(ctx) = risk_context {
                    sizer.calculate_fraction(ctx.closed_trades)
                } else {
                    0.05
                };
                equity * frac * leverage
            }
        };

        if target.is_finite() && target > 0.0 {
            target.min(max_notional)
        } else {
            0.0
        }
    }

    /// Wrapper de compatibilidad histórica para `calculate_position_size` sin contexto de riesgo.
    pub fn calculate_notional_target(
        sizing: &PositionSizing,
        initial_capital: f64,
        equity: f64,
        leverage: f64,
        exec_price: f64,
        dynamic_sl: Option<f64>,
        fixed_sl_pct: Option<f64>,
    ) -> f64 {
        Self::calculate_position_size(
            sizing,
            initial_capital,
            equity,
            leverage,
            exec_price,
            dynamic_sl,
            fixed_sl_pct,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::RiskConfig;
    use crate::types::{CompletedTrade, PositionSide};

    #[test]
    fn test_bug_f10_fixed_amount_sanitization() {
        // Monto negativo debe retornar 0.0
        let target_neg = PositionSizer::calculate_notional_target(
            &PositionSizing::FixedAmount(-1000.0),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target_neg, 0.0);

        // Monto NaN debe retornar 0.0
        let target_nan = PositionSizer::calculate_notional_target(
            &PositionSizing::FixedAmount(f64::NAN),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target_nan, 0.0);

        // Monto infinito debe ser saneado o acotado
        let target_inf = PositionSizer::calculate_notional_target(
            &PositionSizing::FixedAmount(f64::INFINITY),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target_inf, 0.0);

        // Monto válido positivo debe ser calculado y acotado por equity * leverage
        let target_valid = PositionSizer::calculate_notional_target(
            &PositionSizing::FixedAmount(2_000.0),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target_valid, 4_000.0);

        // Monto superior a max_notional debe quedar acotado
        let target_excess = PositionSizer::calculate_notional_target(
            &PositionSizing::FixedAmount(50_000.0),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target_excess, 20_000.0); // 10_000 * 2.0
    }

    #[test]
    fn test_nan_inputs_return_zero() {
        let target = PositionSizer::calculate_notional_target(
            &PositionSizing::PercentOfEquity(0.5),
            10_000.0,
            f64::NAN,
            2.0,
            100.0,
            None,
            None,
        );
        assert_eq!(target, 0.0);
    }

    #[test]
    fn test_dynamic_risk_sizing_with_context() {
        let cfg = RiskConfig {
            kelly_default_fraction: 0.10,
            use_volatility_targeting: false,
            use_conviction: false,
            use_drawdown_governor: false,
            ..Default::default()
        };
        let sizing = PositionSizing::DynamicRisk(cfg);

        let trades: Vec<CompletedTrade> = Vec::new();
        let ctx = RiskEvaluationContext::new(
            10_000.0,
            2.0,
            100.0,
            &trades,
            None,
            0.0,
            0.50,
            None,
        );

        let target = PositionSizer::calculate_position_size(
            &sizing,
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
            Some(&ctx),
        );

        // Equity 10_000 * 0.10 fraction = 1000.0 USD
        assert!((target - 1000.0).abs() < 1e-4);
    }

    #[test]
    fn test_volatility_targeting_sizing() {
        let sizing = PositionSizing::VolatilityTargeting {
            target_annual_vol: 0.15,
            max_leverage: 2.0,
        };

        // C-21: Base cripto 24/7 continua para 1H: 365 días * 24h = 8760 períodos/año
        let ann_factor = (365.0 * 24.0_f64).sqrt();
        let exact_vol = 0.15 / ann_factor;

        let trades: Vec<CompletedTrade> = Vec::new();
        let ctx = RiskEvaluationContext::new(
            10_000.0,
            2.0,
            100.0,
            &trades,
            Some(exact_vol),
            0.0,
            0.50,
            None,
        ).with_timeframe(crate::types::Timeframe::H1);

        let target = PositionSizer::calculate_position_size(
            &sizing,
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
            Some(&ctx),
        );

        // Weight = 1.0 -> target = 10,000 USD
        assert!((target - 10_000.0).abs() < 1e-4);
    }

    #[test]
    fn test_fractional_kelly_sizing() {
        let sizing = PositionSizing::FractionalKelly {
            fraction: 0.25,
            window: 30,
        };

        fn dummy_trade(net_pnl: f64) -> CompletedTrade {
            CompletedTrade {
                side: PositionSide::Long,
                entry_timestamp: 1000,
                exit_timestamp: 2000,
                entry_price: 100.0,
                exit_price: 100.0 + net_pnl,
                quantity: 1.0,
                gross_pnl: net_pnl,
                commissions: 0.0,
                funding_paid: 0.0,
                net_pnl,
                return_pct: net_pnl / 100.0,
                duration_bars: 1,
                duration_millis: 1000,
                max_favorable_excursion: 0.0,
                max_adverse_excursion: 0.0,
                exit_reason: "Test".to_string(),
                entry_candle_idx: 0,
                exit_candle_idx: 1,
                regime: crate::types::TradeRegime::PrimaryTrend,
            }
        }

        // 20 trades con edge positivo: 12 ganadores de $20, 8 perdedores de $10
        // full_kelly = 0.40, lambda = 0.25 -> f* = 0.10
        let mut trades = Vec::new();
        for _ in 0..12 {
            trades.push(dummy_trade(20.0));
        }
        for _ in 0..8 {
            trades.push(dummy_trade(-10.0));
        }

        let ctx = RiskEvaluationContext::new(
            10_000.0,
            2.0,
            100.0,
            &trades,
            None,
            0.0,
            0.50,
            None,
        );

        let target = PositionSizer::calculate_position_size(
            &sizing,
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
            Some(&ctx),
        );

        // Equity 10_000 * 0.10 * leverage 2.0 = 2000.0 USD
        assert!((target - 2000.0).abs() < 1e-4);
    }

    #[test]
    fn test_c35_micro_sizing_percent_of_equity_and_capital() {
        // Micro-dimensionamiento con PercentOfEquity: 0.002 (0.2%)
        // Con equity $10,000 y leverage 2.0: target = 10,000 * 0.002 * 2.0 = $40.0
        // (Bajo el antiguo clamp de 1%, se forzaba erróneamente a $200.0)
        let target_micro_eq = PositionSizer::calculate_notional_target(
            &PositionSizing::PercentOfEquity(0.002),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert!((target_micro_eq - 40.0).abs() < 1e-4, "Target esperado 40.0, obtenido {}", target_micro_eq);

        // Umbral institucional mínimo: 0.0001 (0.01%)
        // target = 10,000 * 0.0001 * 2.0 = $2.0
        let target_min_eq = PositionSizer::calculate_notional_target(
            &PositionSizing::PercentOfEquity(0.0001),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert!((target_min_eq - 2.0).abs() < 1e-4);

        // Rechazo estricto de porcentajes <= 0.0 o no finitos
        assert_eq!(
            PositionSizer::calculate_notional_target(
                &PositionSizing::PercentOfEquity(0.0),
                10_000.0,
                10_000.0,
                2.0,
                100.0,
                None,
                None,
            ),
            0.0
        );
        assert_eq!(
            PositionSizer::calculate_notional_target(
                &PositionSizing::PercentOfEquity(-0.01),
                10_000.0,
                10_000.0,
                2.0,
                100.0,
                None,
                None,
            ),
            0.0
        );
        assert_eq!(
            PositionSizer::calculate_notional_target(
                &PositionSizing::PercentOfEquity(f64::NAN),
                10_000.0,
                10_000.0,
                2.0,
                100.0,
                None,
                None,
            ),
            0.0
        );

        // Micro-dimensionamiento con PercentOfInitialCapital: 0.0005 (0.05%)
        // target = 10,000 * 0.0005 * 2.0 = $10.0
        let target_micro_init = PositionSizer::calculate_notional_target(
            &PositionSizing::PercentOfInitialCapital(0.0005),
            10_000.0,
            10_000.0,
            2.0,
            100.0,
            None,
            None,
        );
        assert!((target_micro_init - 10.0).abs() < 1e-4);

        // Rechazo de PercentOfInitialCapital <= 0.0
        assert_eq!(
            PositionSizer::calculate_notional_target(
                &PositionSizing::PercentOfInitialCapital(0.0),
                10_000.0,
                10_000.0,
                2.0,
                100.0,
                None,
                None,
            ),
            0.0
        );
    }
}

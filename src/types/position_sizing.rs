use serde::{Deserialize, Serialize};
use crate::risk::RiskConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PositionSizing {
    /// Porcentaje nominal FIJO del capital inicial (ej: 0.10 para 10% de $10,000 = $1,000 margen * Leverage)
    PercentOfInitialCapital(f64),
    /// Porcentaje dinámico del equity acumulado (ej: 0.25 = 25% compuesto)
    PercentOfEquity(f64),
    /// Tamaño nominal fijo en USD (ej: 1000.0)
    FixedAmount(f64),
    /// Porcentaje de riesgo sobre el equity respecto al Stop Loss (ej: 0.01 = 1% de riesgo)
    RiskBasedPercent(f64),
    /// Gestión de riesgo cuantitativa institucional integral
    DynamicRisk(RiskConfig),
    /// Dimensionamiento dinámico por Volatility Targeting independiente
    VolatilityTargeting {
        target_annual_vol: f64,
        max_leverage: f64,
    },
    /// Dimensionamiento dinámico por Criterio de Kelly Fraccional independiente
    FractionalKelly {
        fraction: f64,
        window: usize,
    },
    DrawdownGovernedInitialCapital {
        pct: f64,
        max_dd_pct: f64,
        gamma: f64,
        min_scale: f64,
    },
}

pub type PositionSizingMethod = PositionSizing;

impl Default for PositionSizing {
    fn default() -> Self {
        PositionSizing::PercentOfEquity(0.25)
    }
}

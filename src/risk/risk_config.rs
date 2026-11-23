use serde::{Deserialize, Serialize};

/// Configuración integral para el subsistema de dimensionamiento y gestión de riesgo dinámico.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskConfig {
    // --- Fractional Kelly ---
    /// Fracción de Kelly defensiva (lambda: típicamente 0.25 para Quarter Kelly o 0.5 para Half Kelly)
    pub kelly_fraction: f64,
    /// Ventana de trades cerrados retrospectivos para estimar win-rate y payoff ratio
    pub kelly_window: usize,
    /// Mínimo de trades cerrados en la ventana requeridos para activar el cálculo de Kelly
    pub kelly_min_trades: usize,
    /// Asignación mínima fraccional garantizada por Kelly
    pub kelly_min_fraction: f64,
    /// Asignación máxima fraccional permitida por Kelly
    pub kelly_max_fraction: f64,
    /// Asignación fraccional por defecto cuando no hay suficientes trades previos
    pub kelly_default_fraction: f64,

    // --- Volatility Targeting ---
    /// Volatilidad objetivo anualizada deseada (ej: 0.15 para 15%)
    pub target_annual_vol: f64,
    /// Ponderación de apalancamiento mínima permitida por el modelo de volatilidad
    pub vol_min_weight: f64,
    /// Ponderación de apalancamiento máxima permitida por el modelo de volatilidad
    pub vol_max_weight: f64,
    /// Factor de anualización temporal cripto 24/7 (ej: sqrt(365 * 24) = sqrt(8760) para velas horarias)
    pub annualization_factor: f64,

    // --- Conviction Sizing ---
    /// Umbral mínimo de activación de señal ML (tau, ej: 0.50)
    pub conviction_threshold: f64,
    /// Sensibilidad de escala por convicción (kappa, ej: 0.50)
    pub conviction_kappa: f64,
    /// Multiplicador mínimo de convicción
    pub conviction_min_scale: f64,
    /// Multiplicador máximo de convicción
    pub conviction_max_scale: f64,

    // --- Drawdown Governor ---
    /// Drawdown máximo tolerable antes de desescalar o cortar exposición (ej: 0.15 para 15%)
    pub max_tolerable_drawdown: f64,
    /// Exponente de curvatura de desescalada (gamma: 1.0 lineal, > 1.0 penalización progresiva)
    pub drawdown_gamma: f64,
    /// Escala mínima de asignación remanente cuando se alcanza el drawdown máximo
    pub drawdown_min_scale: f64,

    // --- Global Portfolio Constraints ---
    /// Apalancamiento institucional máximo permitido
    pub max_leverage: f64,
    /// Límite nocional máximo en USD absoluto (opcional)
    pub max_notional_usd: Option<f64>,

    // --- Feature Toggles ---
    pub use_kelly: bool,
    pub use_volatility_targeting: bool,
    pub use_conviction: bool,
    pub use_drawdown_governor: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            // Fractional Kelly (Quarter-Kelly conservador)
            kelly_fraction: 0.25,
            kelly_window: 30,
            kelly_min_trades: 20,
            kelly_min_fraction: 0.02,
            kelly_max_fraction: 0.25,
            kelly_default_fraction: 0.05,

            // Volatility Targeting (15% anualizada estándar, base cripto 24/7/365 en 1H)
            target_annual_vol: 0.15,
            vol_min_weight: 0.05,
            vol_max_weight: 2.0,
            annualization_factor: (365.0 * 24.0_f64).sqrt(),

            // Conviction Sizing
            conviction_threshold: 0.50,
            conviction_kappa: 0.50,
            conviction_min_scale: 0.50,
            conviction_max_scale: 1.50,

            // Drawdown Governor (15% Max DD con curvatura 1.5)
            max_tolerable_drawdown: 0.15,
            drawdown_gamma: 1.5,
            drawdown_min_scale: 0.0,

            // Global Caps
            max_leverage: 3.0,
            max_notional_usd: None,

            // Feature Toggles
            use_kelly: true,
            use_volatility_targeting: true,
            use_conviction: true,
            use_drawdown_governor: true,
        }
    }
}

impl RiskConfig {
    /// Construye una configuración calibrada para un timeframe específico según la base continua 24/7 de cripto (C-21).
    pub fn for_timeframe(timeframe: crate::types::Timeframe) -> Self {
        let mut cfg = Self::default();
        cfg.annualization_factor = timeframe.annualization_factor();
        cfg
    }

    /// Asigna de forma explícita y tipada el factor de anualización derivado del timeframe (C-21).
    pub fn with_timeframe(mut self, timeframe: crate::types::Timeframe) -> Self {
        self.annualization_factor = timeframe.annualization_factor();
        self
    }

    /// Asigna de forma explícita los períodos por año para la anualización.
    pub fn with_periods_per_year(mut self, periods: f64) -> Self {
        if periods > 0.0 && periods.is_finite() {
            self.annualization_factor = periods.sqrt();
        }
        self
    }

    /// Asigna directamente el factor de escala de anualización (`sqrt(periods_per_year)`).
    pub fn with_annualization_factor(mut self, factor: f64) -> Self {
        if factor > 0.0 && factor.is_finite() {
            self.annualization_factor = factor;
        }
        self
    }
}

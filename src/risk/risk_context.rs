use crate::types::{CompletedTrade, Timeframe};

/// Contexto temporal causal que contiene el estado histórico observable en el instante t <= T.
/// Garantiza cero look-ahead bias al encapsular únicamente magnitudes calculadas en o antes de t.
#[derive(Debug, Clone)]
pub struct RiskEvaluationContext<'a> {
    /// Equity total en el momento de la evaluación
    pub equity: f64,
    /// Apalancamiento configurado de la cuenta
    pub leverage: f64,
    /// Precio de ejecución de entrada (Open exacto, sin slippage por orden de Allianz)
    pub exec_price: f64,
    /// Historial de trades cerrados estrictamente antes de la apertura de esta posición (t <= T)
    pub closed_trades: &'a [CompletedTrade],
    /// Volatilidad causal estimada (ej: desviación de retornos o ATR relativo)
    pub current_volatility: Option<f64>,
    /// Drawdown actual fraccional respecto al pico histórico causal (ej: 0.05 = 5% DD)
    pub current_drawdown: f64,
    /// Intensidad normalizada de la señal predicha por el modelo de ML en [0, 1]
    pub signal_strength: f64,
    /// Precio de Stop Loss dinámico emitido con la señal (si existe)
    pub dynamic_sl: Option<f64>,
    /// Factor de escala para anualizar la volatilidad (`sqrt(periods_per_year)`) (C-21)
    pub annualization_factor: Option<f64>,
    /// Timeframe tipado explícito del backtest (C-21)
    pub timeframe: Option<Timeframe>,
    /// Pico de equity histórico causal previo o actualizado tras cierre en Open (C-06)
    pub peak_equity: f64,
}

impl<'a> RiskEvaluationContext<'a> {
    pub fn new(
        equity: f64,
        leverage: f64,
        exec_price: f64,
        closed_trades: &'a [CompletedTrade],
        current_volatility: Option<f64>,
        current_drawdown: f64,
        signal_strength: f64,
        dynamic_sl: Option<f64>,
    ) -> Self {
        let safe_dd = current_drawdown.max(0.0);
        let peak = if safe_dd > 0.0 && safe_dd < 1.0 && equity > 0.0 {
            equity / (1.0 - safe_dd)
        } else {
            equity
        };
        Self {
            equity,
            leverage,
            exec_price,
            closed_trades,
            current_volatility,
            current_drawdown: safe_dd,
            signal_strength: signal_strength.clamp(0.0, 1.0),
            dynamic_sl,
            annualization_factor: None,
            timeframe: None,
            peak_equity: peak,
        }
    }

    /// Asigna de forma explícita el pico histórico causal de equity.
    pub fn with_peak_equity(mut self, peak: f64) -> Self {
        if peak > 0.0 && peak.is_finite() {
            self.peak_equity = peak;
            if self.equity.is_finite() && self.equity > 0.0 {
                self.current_drawdown = ((peak - self.equity).max(0.0) / peak).max(0.0);
            }
        }
        self
    }

    /// Asigna de forma explícita y tipada el timeframe y su factor de anualización correspondiente (C-21).
    pub fn with_timeframe(mut self, timeframe: Timeframe) -> Self {
        self.annualization_factor = Some(timeframe.annualization_factor());
        self.timeframe = Some(timeframe);
        self
    }

    /// Asigna directamente el factor de anualización de volatilidad.
    pub fn with_annualization_factor(mut self, factor: f64) -> Self {
        if factor > 0.0 && factor.is_finite() {
            self.annualization_factor = Some(factor);
        }
        self
    }

    /// Asigna el factor de anualización a partir del número de períodos por año.
    pub fn with_periods_per_year(mut self, periods: f64) -> Self {
        if periods > 0.0 && periods.is_finite() {
            self.annualization_factor = Some(periods.sqrt());
        }
        self
    }

    /// Construye un contexto base para pruebas o fallbacks sin historial previo
    pub fn baseline(equity: f64, leverage: f64, exec_price: f64) -> Self {
        Self {
            equity,
            leverage,
            exec_price,
            closed_trades: &[],
            current_volatility: None,
            current_drawdown: 0.0,
            signal_strength: 0.5,
            dynamic_sl: None,
            annualization_factor: None,
            timeframe: None,
            peak_equity: equity,
        }
    }
}

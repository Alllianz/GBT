use serde::{Deserialize, Serialize};
use super::position::PositionSide;
use super::trade_regime::TradeRegime;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletedTrade {
    pub side: PositionSide,
    pub entry_timestamp: i64,
    pub exit_timestamp: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub gross_pnl: f64,
    pub commissions: f64,
    /// Costes acumulados de tasas de financiación (Funding Rate) durante la vida del trade.
    /// Signo estandarizado: positivo = importe pagado (reduce el beneficio neto).
    #[serde(default)]
    pub funding_paid: f64,
    pub net_pnl: f64,
    pub return_pct: f64,
    pub duration_bars: usize,
    pub duration_millis: i64,
    pub max_favorable_excursion: f64,
    pub max_adverse_excursion: f64,
    pub exit_reason: String,
    #[serde(default)]
    pub entry_candle_idx: usize,
    #[serde(default)]
    pub exit_candle_idx: usize,
    #[serde(default)]
    pub regime: TradeRegime,
}

impl CompletedTrade {
    #[inline]
    pub fn is_winner(&self) -> bool {
        self.net_pnl > 0.0
    }

    #[inline]
    pub fn is_loser(&self) -> bool {
        self.net_pnl < 0.0
    }
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub side: PositionSide,
    pub entry_price: f64,
    pub quantity: f64,
    pub entry_timestamp: i64,
    pub entry_candle_idx: usize,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub isolated_margin: Option<f64>,
    pub liquidation_price: f64,
    pub max_favorable_price: f64,
    pub max_adverse_price: f64,
    pub entry_commission: f64,
    pub accumulated_funding: f64,
    pub last_funding_timestamp: i64,
}

impl Position {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: PositionSide,
        entry_price: f64,
        quantity: f64,
        entry_timestamp: i64,
        entry_candle_idx: usize,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
        isolated_margin: Option<f64>,
        liquidation_price: f64,
        entry_commission: f64,
    ) -> Self {
        Self {
            side,
            entry_price,
            quantity,
            entry_timestamp,
            entry_candle_idx,
            stop_loss,
            take_profit,
            isolated_margin,
            liquidation_price,
            max_favorable_price: entry_price,
            max_adverse_price: entry_price,
            entry_commission,
            accumulated_funding: 0.0,
            last_funding_timestamp: entry_timestamp,
        }
    }

    #[inline]
    pub fn notional_value(&self) -> f64 {
        self.entry_price * self.quantity
    }

    /// Retorna el margen efectivo neto de la posición tras descontar la comisión de apertura (C-03)
    /// y el funding acumulado devengado (C-11). En modo Cross, retorna `None`.
    #[inline]
    pub fn effective_margin(&self) -> Option<f64> {
        self.isolated_margin.map(|m| (m - self.entry_commission - self.accumulated_funding).max(0.0))
    }

    #[inline]
    pub fn apply_funding(&mut self, funding_payment: f64, timestamp: i64) {
        self.accumulated_funding += funding_payment;
        self.last_funding_timestamp = timestamp;
    }

    pub fn update_extremes(&mut self, high: f64, low: f64) {
        match self.side {
            PositionSide::Long => {
                if high > self.max_favorable_price {
                    self.max_favorable_price = high;
                }
                if low < self.max_adverse_price {
                    self.max_adverse_price = low;
                }
            }
            PositionSide::Short => {
                if low < self.max_favorable_price {
                    self.max_favorable_price = low;
                }
                if high > self.max_adverse_price {
                    self.max_adverse_price = high;
                }
            }
        }
    }

    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let raw = match self.side {
            PositionSide::Long => (current_price - self.entry_price) * self.quantity,
            PositionSide::Short => (self.entry_price - current_price) * self.quantity,
        };
        if let Some(margin) = self.isolated_margin {
            if margin > 0.0 {
                (raw - (self.entry_commission + self.accumulated_funding)).max(-margin)
            } else {
                raw
            }
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unrealized_pnl_isolated_margin_with_costs() {
        let mut pos = Position::new(
            PositionSide::Long,
            100.0,
            2.0,
            1000,
            0,
            None,
            None,
            Some(50.0),
            75.0,
            0.5,
        );
        pos.apply_funding(0.3, 1001);

        let pnl_gain = pos.unrealized_pnl(110.0);
        assert!((pnl_gain - 19.2).abs() < 1e-9);

        let pnl_capped = pos.unrealized_pnl(50.0);
        assert_eq!(pnl_capped, -50.0);
    }

    #[test]
    fn test_unrealized_pnl_cross_margin() {
        let pos = Position::new(
            PositionSide::Long,
            100.0,
            2.0,
            1000,
            0,
            None,
            None,
            None,
            50.0,
            0.5,
        );
        let pnl = pos.unrealized_pnl(110.0);
        assert!((pnl - 20.0).abs() < 1e-9);
    }
}

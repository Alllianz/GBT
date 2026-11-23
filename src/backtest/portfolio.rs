use crate::types::Position;

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub initial_capital: f64,
    pub cash_balance: f64,
    pub active_position: Option<Position>,
    pub total_commissions_paid: f64,
    pub total_funding_paid: f64,
}

impl Portfolio {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            cash_balance: initial_capital,
            active_position: None,
            total_commissions_paid: 0.0,
            total_funding_paid: 0.0,
        }
    }

    pub fn current_equity(&self, current_price: f64) -> f64 {
        match &self.active_position {
            Some(pos) => self.cash_balance + pos.unrealized_pnl(current_price),
            None => self.cash_balance,
        }
    }

    pub fn has_position(&self) -> bool {
        self.active_position.is_some()
    }

    pub fn is_bankrupt(&self, current_price: f64) -> bool {
        self.current_equity(current_price) <= 0.0
    }
}

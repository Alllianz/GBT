use crate::types::Position;

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub initial_capital: f64,
    pub cash_balance: f64,
    pub active_position: Option<Position>,
    pub long_position: Option<Position>,
    pub short_position: Option<Position>,
    pub total_commissions_paid: f64,
    pub total_funding_paid: f64,
}

impl Portfolio {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            cash_balance: initial_capital,
            active_position: None,
            long_position: None,
            short_position: None,
            total_commissions_paid: 0.0,
            total_funding_paid: 0.0,
        }
    }

    pub fn current_equity(&self, current_price: f64) -> f64 {
        let unrealized: f64 = [&self.active_position, &self.long_position, &self.short_position]
            .into_iter()
            .flatten()
            .map(|pos| pos.unrealized_pnl(current_price))
            .sum();
        self.cash_balance + unrealized
    }

    pub fn has_position(&self) -> bool {
        self.active_position.is_some()
            || self.long_position.is_some()
            || self.short_position.is_some()
    }

    pub fn has_long(&self) -> bool {
        self.long_position.is_some()
    }

    pub fn has_short(&self) -> bool {
        self.short_position.is_some()
    }

    pub fn total_isolated_margin(&self) -> f64 {
        [&self.active_position, &self.long_position, &self.short_position]
            .into_iter()
            .flatten()
            .filter_map(|pos| pos.isolated_margin)
            .sum()
    }

    pub fn is_bankrupt(&self, current_price: f64) -> bool {
        self.current_equity(current_price) <= 0.0
    }
}

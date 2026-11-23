use serde::{Deserialize, Serialize};
use super::position::PositionSide;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// Determina el lado de la orden para la apertura de una posición
    #[inline]
    pub fn from_position_entry(side: PositionSide) -> Self {
        match side {
            PositionSide::Long => OrderSide::Buy,
            PositionSide::Short => OrderSide::Sell,
        }
    }

    /// Determina el lado de la orden para el cierre de una posición
    #[inline]
    pub fn from_position_exit(side: PositionSide) -> Self {
        match side {
            PositionSide::Long => OrderSide::Sell,
            PositionSide::Short => OrderSide::Buy,
        }
    }
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderRequest {
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub timestamp: i64,
    pub reason: String,
}

impl OrderRequest {
    pub fn market(side: OrderSide, quantity: f64, price: Option<f64>, timestamp: i64, reason: impl Into<String>) -> Self {
        Self {
            side,
            order_type: OrderType::Market,
            quantity,
            price,
            timestamp,
            reason: reason.into(),
        }
    }

    #[inline]
    pub fn notional_value(&self) -> f64 {
        self.quantity * self.price.unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub side: OrderSide,
    pub quantity: f64,
    pub executed_price: f64,
    pub commission: f64,
    pub timestamp: i64,
}

impl ExecutionResult {
    pub fn new(
        side: OrderSide,
        quantity: f64,
        executed_price: f64,
        commission: f64,
        timestamp: i64,
    ) -> Self {
        Self {
            side,
            quantity,
            executed_price,
            commission,
            timestamp,
        }
    }

    #[inline]
    pub fn notional_value(&self) -> f64 {
        self.quantity * self.executed_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_side_mapping() {
        assert_eq!(OrderSide::from_position_entry(PositionSide::Long), OrderSide::Buy);
        assert_eq!(OrderSide::from_position_entry(PositionSide::Short), OrderSide::Sell);
        assert_eq!(OrderSide::from_position_exit(PositionSide::Long), OrderSide::Sell);
        assert_eq!(OrderSide::from_position_exit(PositionSide::Short), OrderSide::Buy);
    }

    #[test]
    fn test_order_and_execution_notional() {
        let req = OrderRequest::market(OrderSide::Buy, 2.5, Some(100.0), 1000, "Entry");
        assert_eq!(req.notional_value(), 250.0);

        let exec = ExecutionResult::new(OrderSide::Buy, 2.5, 100.0, 0.25, 1000);
        assert_eq!(exec.notional_value(), 250.0);
    }
}

use serde::{Deserialize, Serialize};

/// Punto de la curva de equity registrado estrictamente al CIERRE de cada vela (Close).
///
/// Directriz institucional de Allianz: Cero equity, drawdown o profit intra-vela.
/// Todas las métricas de balance, equity y drawdown reflejan exclusivamente el estado al Close.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: i64,
    pub candle_idx: usize,
    pub balance: f64,
    pub equity: f64,
    pub drawdown_nominal: f64,
    pub drawdown_pct: f64,
}

impl EquityPoint {
    pub fn new(
        timestamp: i64,
        candle_idx: usize,
        balance: f64,
        equity: f64,
        drawdown_nominal: f64,
        drawdown_pct: f64,
    ) -> Self {
        Self {
            timestamp,
            candle_idx,
            balance,
            equity,
            drawdown_nominal,
            drawdown_pct,
        }
    }
}

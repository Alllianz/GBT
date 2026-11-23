use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub net_profit: f64,
    pub net_profit_pct: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate_pct: f64,
    /// Beneficio bruto total acumulado de operaciones con gross_pnl > 0 (pre-comisiones y pre-funding).
    pub gross_profit: f64,
    /// Pérdida bruta total acumulada en valor absoluto de operaciones con gross_pnl < 0 (pre-comisiones y pre-funding).
    pub gross_loss: f64,
    /// Profit Factor bruto estándar institucional (`gross_profit / gross_loss`).
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub avg_win_pnl: f64,
    pub avg_loss_pnl: f64,
    pub payoff_ratio: f64,
    pub max_drawdown_nominal: f64,
    pub max_drawdown_pct: f64,
    pub recovery_factor: f64,
    pub total_commissions: f64,
    /// Importe total neto de tasas de financiación pagadas (+ pagado, - cobrado).
    #[serde(default)]
    pub total_funding: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub curve_r2: f64,
    pub capital_to_maxdd_ratio: f64,
    pub composite_score: f64,
    pub avg_duration_bars: f64,
    pub max_consecutive_losses: usize,
}

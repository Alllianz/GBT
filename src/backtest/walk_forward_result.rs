use super::engine::BacktestResult;
use crate::types::FitnessMetric;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardMetrics {
    pub is_net_profit_pct: f64,
    pub oos_net_profit_pct: f64,
    pub is_sharpe: f64,
    pub oos_sharpe: f64,
    pub is_max_dd_pct: f64,
    pub oos_max_dd_pct: f64,
    pub is_profit_factor: f64,
    pub oos_profit_factor: f64,
    pub is_win_rate_pct: f64,
    pub oos_win_rate_pct: f64,
    pub is_curve_r2: f64,
    pub oos_curve_r2: f64,
    /// Ratio de consistencia / sobreajuste: retorno relativo OOS / IS
    pub robustness_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    pub model_name: String,
    pub timeframe: String,
    pub is_result: BacktestResult,
    pub oos_result: BacktestResult,
    pub metrics: WalkForwardMetrics,
    pub fitness_metric: FitnessMetric,
}

impl WalkForwardResult {
    pub fn new(
        model_name: impl Into<String>,
        timeframe: impl Into<String>,
        is_result: BacktestResult,
        oos_result: BacktestResult,
        fitness_metric: FitnessMetric,
    ) -> Self {
        // C-24: Ratio de Robustez corregido con normalización temporal delegando en RobustnessCalculator
        let robustness_ratio = super::robustness::RobustnessCalculator::calculate(&is_result, &oos_result);

        let metrics = WalkForwardMetrics {
            is_net_profit_pct: is_result.metrics.net_profit_pct,
            oos_net_profit_pct: oos_result.metrics.net_profit_pct,
            is_sharpe: is_result.metrics.sharpe_ratio,
            oos_sharpe: oos_result.metrics.sharpe_ratio,
            is_max_dd_pct: is_result.metrics.max_drawdown_pct,
            oos_max_dd_pct: oos_result.metrics.max_drawdown_pct,
            is_profit_factor: is_result.metrics.profit_factor,
            oos_profit_factor: oos_result.metrics.profit_factor,
            is_win_rate_pct: is_result.metrics.win_rate_pct,
            oos_win_rate_pct: oos_result.metrics.win_rate_pct,
            is_curve_r2: is_result.metrics.curve_r2,
            oos_curve_r2: oos_result.metrics.curve_r2,
            robustness_ratio,
        };

        Self {
            model_name: model_name.into(),
            timeframe: timeframe.into(),
            is_result,
            oos_result,
            metrics,
            fitness_metric,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::metrics::BacktestMetrics;
    use crate::backtest::EquityPoint;

    fn mock_result(net_profit_pct: f64, n_bars: usize) -> BacktestResult {
        let mut equity_curve = Vec::with_capacity(n_bars);
        for i in 0..n_bars {
            equity_curve.push(EquityPoint::new(
                1_000_000 + i as i64 * 86_400_000,
                i,
                1000.0,
                1000.0 * (1.0 + net_profit_pct / 100.0),
                0.0,
                0.0,
            ));
        }

        let metrics = BacktestMetrics {
            initial_capital: 1000.0,
            final_capital: 1000.0 * (1.0 + net_profit_pct / 100.0),
            net_profit: 1000.0 * (net_profit_pct / 100.0),
            net_profit_pct,
            total_trades: 10,
            winning_trades: 5,
            losing_trades: 5,
            win_rate_pct: 50.0,
            gross_profit: 100.0,
            gross_loss: 50.0,
            profit_factor: 2.0,
            avg_trade_pnl: 10.0,
            avg_win_pnl: 20.0,
            avg_loss_pnl: 10.0,
            payoff_ratio: 2.0,
            max_drawdown_nominal: 50.0,
            max_drawdown_pct: 5.0,
            recovery_factor: 2.0,
            total_commissions: 5.0,
            total_funding: 0.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            curve_r2: 0.95,
            capital_to_maxdd_ratio: 20.0,
            composite_score: 1.5,
            avg_duration_bars: 5.0,
            max_consecutive_losses: 2,
        };

        BacktestResult {
            metrics,
            trades: Vec::new(),
            equity_curve,
        }
    }

    #[test]
    fn test_c24_walk_forward_result_robustness_integration() {
        let is_res = mock_result(70.0, 700);
        let oos_res = mock_result(30.0, 300);

        let wf = WalkForwardResult::new("TestModel", "1d", is_res, oos_res, FitnessMetric::SharpeRatio);
        assert!(
            (wf.metrics.robustness_ratio - 1.0).abs() < 1e-4,
            "Robustness ratio en WalkForwardResult debe ser 1.0 pero fue {}",
            wf.metrics.robustness_ratio
        );
    }
}

use crate::backtest::{
    max_stagnation_ratio, BacktestResult, MetricsCalculator, RobustnessCalculator, WalkForwardResult,
};
use crate::models::{ModelConfig, TreeBoostHedgeConfig};
use crate::types::{CompletedTrade, FitnessMetric, Timeframe, TradeRegime};
use super::grid::EvolveCandidate;

pub struct EvaluatedCandidate {
    pub id: usize,
    pub group_idx: usize,
    pub label: String,
    pub group_label: String,
    pub is_baseline: bool,
    pub model_cfg: ModelConfig,
    pub hedge_cfg: TreeBoostHedgeConfig,
    pub opt_long: f64,
    pub opt_short: f64,
    pub best_fit: f64,
    pub is_res: BacktestResult,
    pub is_stag: f64,
    pub is_liq: usize,
    pub is_trend_net_pnl: f64,
    pub is_mr_net_pnl: f64,
    pub is_trend_trades: usize,
    pub is_mr_trades: usize,
    pub is_dual_regime_score: f64,
    pub oos_res: Option<BacktestResult>,
    pub oos_stag: f64,
    pub oos_liq: usize,
    pub oos_trend_net_pnl: f64,
    pub oos_mr_net_pnl: f64,
    pub oos_trend_trades: usize,
    pub oos_mr_trades: usize,
    pub oos_dual_regime_score: f64,
    pub robustness_ratio: f64,
    pub passes_gate: bool,
    pub inst_score: f64,
}

impl EvaluatedCandidate {
    pub fn regime_breakdown(trades: &[CompletedTrade]) -> (f64, f64, usize, usize) {
        let mut trend_net_pnl = 0.0;
        let mut mr_net_pnl = 0.0;
        let mut trend_trades = 0usize;
        let mut mr_trades = 0usize;
        for t in trades {
            let pnl = if t.net_pnl.is_finite() { t.net_pnl } else { 0.0 };
            match t.regime {
                TradeRegime::PrimaryTrend => {
                    trend_net_pnl += pnl;
                    trend_trades += 1;
                }
                TradeRegime::MeanReversionHedge => {
                    mr_net_pnl += pnl;
                    mr_trades += 1;
                }
            }
        }
        (trend_net_pnl, mr_net_pnl, trend_trades, mr_trades)
    }

    pub fn dual_regime_score(
        trend_net_pnl: f64,
        mr_net_pnl: f64,
        trend_trades: usize,
        mr_trades: usize,
    ) -> f64 {
        if trend_trades == 0 || mr_trades == 0 {
            return 0.0;
        }
        if !trend_net_pnl.is_finite()
            || !mr_net_pnl.is_finite()
            || trend_net_pnl <= 0.0
            || mr_net_pnl <= 0.0
        {
            return 0.0;
        }
        let min_pnl = trend_net_pnl.min(mr_net_pnl);
        let max_pnl = trend_net_pnl.max(mr_net_pnl);
        if max_pnl <= 0.0 {
            return 0.0;
        }
        (min_pnl / max_pnl).clamp(0.0, 1.0).sqrt().clamp(0.0, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn score(
        cand: &EvolveCandidate,
        group_label: &str,
        model_cfg: &ModelConfig,
        opt_long: f64,
        opt_short: f64,
        best_fit: f64,
        is_res: BacktestResult,
        oos_res: Option<BacktestResult>,
    ) -> Self {
        let (_, is_stag) = max_stagnation_ratio(&is_res.equity_curve);
        let is_liq = is_res
            .trades
            .iter()
            .filter(|t| t.exit_reason.contains("Liquidated"))
            .count();
        let (is_trend_net_pnl, is_mr_net_pnl, is_trend_trades, is_mr_trades) =
            Self::regime_breakdown(&is_res.trades);
        let is_dual_regime_score = Self::dual_regime_score(
            is_trend_net_pnl,
            is_mr_net_pnl,
            is_trend_trades,
            is_mr_trades,
        );

        let oos_res = if is_res.metrics.final_capital > 0.0 {
            oos_res
        } else {
            None
        };
        let (
            oos_stag,
            oos_liq,
            robustness_ratio,
            oos_trend_net_pnl,
            oos_mr_net_pnl,
            oos_trend_trades,
            oos_mr_trades,
            oos_dual_regime_score,
        ) = match oos_res.as_ref() {
            Some(oos) => {
                let (_, stag) = max_stagnation_ratio(&oos.equity_curve);
                let liq = oos
                    .trades
                    .iter()
                    .filter(|t| t.exit_reason.contains("Liquidated"))
                    .count();
                let rob = RobustnessCalculator::calculate(&is_res, oos);
                let (t_pnl, m_pnl, t_cnt, m_cnt) = Self::regime_breakdown(&oos.trades);
                let dual = Self::dual_regime_score(t_pnl, m_pnl, t_cnt, m_cnt);
                (stag, liq, rob, t_pnl, m_pnl, t_cnt, m_cnt, dual)
            }
            None => (0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0),
        };

        let is_m = &is_res.metrics;

        let passes_gate = is_m.total_trades >= 120
            && is_m.net_profit_pct > 0.0
            && is_m.final_capital > 0.0;

        let is_comp = is_m.composite_score.clamp(-10.0, 1.0);
        let is_r2 = is_m.curve_r2.clamp(0.0, 1.0);
        let is_cap_dd = is_m.capital_to_maxdd_ratio.max(0.0);
        let norm_cap_dd = (is_cap_dd / (is_cap_dd + 10.0)).clamp(0.0, 1.0);
        let stag = is_stag.clamp(0.0, 1.0);
        let oof_score = if best_fit.is_finite() {
            best_fit.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let liq_penalty = is_liq as f64 * 0.25;
        let regime_penalty = if is_trend_trades == 0 || is_mr_trades == 0 {
            0.35
        } else if is_trend_net_pnl <= 0.0 || is_mr_net_pnl <= 0.0 {
            0.25
        } else {
            0.0
        };
        let base_score = (0.25 * is_comp.max(0.0))
            + (0.20 * is_r2)
            + (0.15 * norm_cap_dd)
            + (0.15 * oof_score)
            + (0.25 * is_dual_regime_score)
            - (0.30 * stag)
            - liq_penalty
            - regime_penalty;

        let inst_score = if passes_gate {
            base_score
        } else {
            -1000.0 + is_comp
        };

        Self {
            id: cand.id,
            group_idx: cand.group_idx,
            label: cand.label.clone(),
            group_label: group_label.to_string(),
            is_baseline: cand.is_baseline,
            model_cfg: model_cfg.clone(),
            hedge_cfg: cand.hedge_cfg,
            opt_long,
            opt_short,
            best_fit,
            is_res,
            is_stag,
            is_liq,
            is_trend_net_pnl,
            is_mr_net_pnl,
            is_trend_trades,
            is_mr_trades,
            is_dual_regime_score,
            oos_res,
            oos_stag,
            oos_liq,
            oos_trend_net_pnl,
            oos_mr_net_pnl,
            oos_trend_trades,
            oos_mr_trades,
            oos_dual_regime_score,
            robustness_ratio,
            passes_gate,
            inst_score,
        }
    }

    pub fn to_walk_forward_result(
        &self,
        tf: Timeframe,
        fitness_metric: FitnessMetric,
    ) -> WalkForwardResult {
        let model_name = format!("TreeBoost_Evolve_{}", self.label);
        let oos_result = self.oos_res.clone().unwrap_or_else(|| BacktestResult {
            metrics: MetricsCalculator::calculate(0.0, &[], &[]),
            trades: Vec::new(),
            equity_curve: Vec::new(),
        });
        WalkForwardResult::new(
            model_name,
            tf.as_db_str(),
            self.is_res.clone(),
            oos_result,
            fitness_metric,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::EquityPoint;
    use crate::models::TreeBoostHedgeConfig;
    use crate::types::PositionSide;

    fn sample_trade(net_pnl: f64, regime: TradeRegime) -> CompletedTrade {
        CompletedTrade {
            side: PositionSide::Long,
            entry_timestamp: 1000,
            exit_timestamp: 2000,
            entry_price: 100.0,
            exit_price: 100.0 + net_pnl,
            quantity: 1.0,
            gross_pnl: net_pnl,
            commissions: 0.0,
            funding_paid: 0.0,
            net_pnl,
            return_pct: net_pnl,
            duration_bars: 1,
            duration_millis: 1000,
            max_favorable_excursion: 0.0,
            max_adverse_excursion: 0.0,
            exit_reason: "Test".to_string(),
            entry_candle_idx: 0,
            exit_candle_idx: 1,
            regime,
        }
    }

    fn sample_backtest(trades: Vec<CompletedTrade>) -> BacktestResult {
        let total_pnl: f64 = trades.iter().map(|t| t.net_pnl).sum();
        let n = trades.len().max(4);
        let curve: Vec<EquityPoint> = (0..n)
            .map(|i| {
                let frac = (i + 1) as f64 / n as f64;
                let eq = 10_000.0 + total_pnl * frac;
                EquityPoint::new((i as i64 + 1) * 3600, i, eq, eq, 10.0, 0.1)
            })
            .collect();
        let metrics = MetricsCalculator::calculate(10_000.0, &trades, &curve);
        BacktestResult {
            metrics,
            trades,
            equity_curve: curve,
        }
    }

    #[test]
    fn test_dual_regime_score_bounds_and_penalties() {
        assert_eq!(EvaluatedCandidate::dual_regime_score(100.0, 100.0, 0, 10), 0.0);
        assert_eq!(EvaluatedCandidate::dual_regime_score(100.0, 100.0, 10, 0), 0.0);
        assert_eq!(EvaluatedCandidate::dual_regime_score(-10.0, 100.0, 10, 10), 0.0);
        assert_eq!(EvaluatedCandidate::dual_regime_score(100.0, 0.0, 10, 10), 0.0);
        assert_eq!(EvaluatedCandidate::dual_regime_score(f64::NAN, 50.0, 10, 10), 0.0);

        let equal = EvaluatedCandidate::dual_regime_score(200.0, 200.0, 60, 60);
        assert!((equal - 1.0).abs() < 1e-12);

        let imbalanced = EvaluatedCandidate::dual_regime_score(400.0, 100.0, 60, 60);
        assert!((imbalanced - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_inst_score_prefers_dual_regime_over_single_regime() {
        let cand = EvolveCandidate {
            id: 1,
            group_idx: 0,
            label: "test".to_string(),
            is_baseline: false,
            hedge_cfg: TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H4), true),
        };
        let model_cfg = ModelConfig::default();

        let mut single_trades = Vec::new();
        for _ in 0..120 {
            single_trades.push(sample_trade(10.0, TradeRegime::PrimaryTrend));
        }
        let single_ev = EvaluatedCandidate::score(
            &cand,
            "g",
            &model_cfg,
            0.002,
            -0.002,
            0.8,
            sample_backtest(single_trades),
            None,
        );

        let mut dual_trades = Vec::new();
        for _ in 0..60 {
            dual_trades.push(sample_trade(10.0, TradeRegime::PrimaryTrend));
            dual_trades.push(sample_trade(10.0, TradeRegime::MeanReversionHedge));
        }
        let dual_ev = EvaluatedCandidate::score(
            &cand,
            "g",
            &model_cfg,
            0.002,
            -0.002,
            0.8,
            sample_backtest(dual_trades),
            None,
        );

        assert!(single_ev.passes_gate);
        assert!(dual_ev.passes_gate);
        assert_eq!(single_ev.is_dual_regime_score, 0.0);
        assert!((dual_ev.is_dual_regime_score - 1.0).abs() < 1e-12);
        assert!(dual_ev.inst_score > single_ev.inst_score);
    }
}

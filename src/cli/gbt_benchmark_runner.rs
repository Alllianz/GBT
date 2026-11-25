use rusqlite::Connection;
use crate::backtest::{max_stagnation_ratio, BacktestConfig, BacktestEngine, WalkForwardResult};
use crate::features::TrainTestSplit;
use crate::models::{ModelConfig, TreeBoostModel, TreeBoostStrategy};
use crate::storage::KlineRepository;
use crate::types::{CompletedTrade, FitnessMetric, PositionSide, PositionSizing, Timeframe};
use super::backtest_presenter::BacktestPresenter;
use super::threshold_calibration_helper::ThresholdCalibrationHelper;

pub struct GbtBenchmarkRunner;

impl GbtBenchmarkRunner {
    pub fn run(
        conn: &Connection,
        tf_str: &str,
        hedge: bool,
        save_dashboard: bool,
    ) -> Result<(), String> {
        let tf = Self::parse_benchmark_timeframe(tf_str)?;
        let klines = KlineRepository::load_all(conn, "BTCUSDT", tf)
            .map_err(|e| format!("DB error loading BTCUSDT [{}]: {}", tf, e))?;
        if klines.len() < 100 {
            return Err(format!(
                "Insufficient klines for BTCUSDT [{}]: {} (min 100)",
                tf,
                klines.len()
            ));
        }

        let active_m_cfg = ModelConfig::for_timeframe(tf);
        let purge_bars = active_m_cfg.horizon + 2;
        let split = TrainTestSplit::split_chronological(&klines, 0.70, purge_bars)?;

        let mut bt_cfg = BacktestConfig::default();
        bt_cfg.timeframe = Some(tf);
        bt_cfg.leverage = 20.0;
        bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
        bt_cfg.fitness_metric = FitnessMetric::CompositeAllianz;
        bt_cfg.hedge_mode = hedge;
        bt_cfg.allow_short = true;

        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train_oof(
            &split.train_klines,
            &active_m_cfg,
            &bt_cfg,
            || TreeBoostModel::new(active_m_cfg.clone()).with_timeframe(Some(tf)),
        )?;

        let mut strategy = TreeBoostStrategy::new(model).with_hedge_enabled(hedge);
        let engine_is = BacktestEngine::new(bt_cfg.clone());
        let is_result = engine_is.run(&mut strategy, &split.train_klines)?;

        let oos_result = if is_result.metrics.final_capital > 0.0 {
            strategy.clear_cache();
            let mut oos_bt_cfg = bt_cfg.clone();
            oos_bt_cfg.initial_capital = is_result.metrics.final_capital;
            oos_bt_cfg.leverage = 20.0;
            oos_bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
            let engine_oos = BacktestEngine::new(oos_bt_cfg);
            engine_oos.run_with_warmup(
                &mut strategy,
                &split.test_warmup_klines,
                split.warmup_bars,
            )?
        } else {
            println!(
                "  ⚠️ ALERTA: Capital IS agotado o insolvente (${:.2}). No se inyecta capital zombi a OOS.",
                is_result.metrics.final_capital
            );
            crate::backtest::BacktestResult {
                metrics: crate::backtest::MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            }
        };

        let wf = WalkForwardResult::new(
            "TreeBoost",
            tf.as_db_str(),
            is_result,
            oos_result,
            bt_cfg.fitness_metric,
        );

        Self::print_metrics_line("IS ", tf, hedge, &wf.is_result);
        Self::print_metrics_line("OOS", tf, hedge, &wf.oos_result);
        Self::print_gap_line(tf, hedge, &wf, opt_long, opt_short, best_fit);

        if save_dashboard {
            let model_slug = if hedge { "treeboost_hedge" } else { "treeboost" };
            BacktestPresenter::present_walk_forward(
                "TreeBoost",
                model_slug,
                tf,
                &wf,
                opt_long,
                opt_short,
                best_fit,
                bt_cfg.fitness_metric,
            )
            .map_err(|e| format!("Error presenting walk-forward dashboard: {}", e))?;
        }

        Ok(())
    }

    fn parse_benchmark_timeframe(tf_str: &str) -> Result<Timeframe, String> {
        let tf: Timeframe = tf_str.parse()?;
        match tf {
            Timeframe::D1 | Timeframe::H4 | Timeframe::H1 => Ok(tf),
            _ => Err(format!(
                "Unsupported timeframe '{}' for bench-gbt (expected 1D, 4H, or 1H)",
                tf_str
            )),
        }
    }

    fn count_hedge_overlaps(trades: &[CompletedTrade]) -> usize {
        let longs: Vec<&CompletedTrade> = trades
            .iter()
            .filter(|t| t.side == PositionSide::Long)
            .collect();
        let shorts: Vec<&CompletedTrade> = trades
            .iter()
            .filter(|t| t.side == PositionSide::Short)
            .collect();

        let mut overlaps = 0usize;
        for l in &longs {
            for s in &shorts {
                if l.entry_timestamp < s.exit_timestamp && s.entry_timestamp < l.exit_timestamp {
                    overlaps += 1;
                }
            }
        }
        overlaps
    }

    fn safe_ratio(oos: f64, is: f64) -> f64 {
        if is.abs() > 1e-9 && oos.is_finite() && is.is_finite() {
            oos / is
        } else {
            0.0
        }
    }

    fn print_metrics_line(
        phase: &str,
        tf: Timeframe,
        hedge: bool,
        res: &crate::backtest::BacktestResult,
    ) {
        let m = &res.metrics;
        let overlaps = Self::count_hedge_overlaps(&res.trades);
        let (_, stag_ratio) = max_stagnation_ratio(&res.equity_curve);
        let liq_count = res
            .trades
            .iter()
            .filter(|t| t.exit_reason.contains("Liquidated"))
            .count();
        println!(
            "{} [{}|hedge={}] | PnL%: {:+.2}% | PF: {:.3} | Composite: {:.4} | R2: {:.4} | Sharpe: {:.3} | MaxDD%: {:.2}% | CapDD: {:.2} | Stag: {:.1}% | Trades: {} | WR%: {:.2}% | Comm: {:.2} | Funding: {:.2} | Liq: {} | HedgeOverlaps: {}",
            phase,
            tf.as_db_str(),
            hedge,
            m.net_profit_pct,
            m.profit_factor,
            m.composite_score,
            m.curve_r2,
            m.sharpe_ratio,
            m.max_drawdown_pct,
            m.capital_to_maxdd_ratio,
            stag_ratio * 100.0,
            m.total_trades,
            m.win_rate_pct,
            m.total_commissions,
            m.total_funding,
            liq_count,
            overlaps,
        );
    }

    fn print_gap_line(
        tf: Timeframe,
        hedge: bool,
        wf: &WalkForwardResult,
        opt_long: f64,
        opt_short: f64,
        best_fit: f64,
    ) {
        let is_m = &wf.is_result.metrics;
        let oos_m = &wf.oos_result.metrics;
        println!(
            "GAP [{}|hedge={}] | Robustness(OOS/IS): {:.3}x | PnL_Ratio: {:.3}x | PF_Ratio: {:.3}x | Composite_Ratio: {:.3}x | R2_Ratio: {:.3}x | Sharpe_Ratio: {:.3}x | MaxDD_Ratio: {:.3}x | Th: +{:.4}/{:.4} (OOF: {:.4})",
            tf.as_db_str(),
            hedge,
            wf.metrics.robustness_ratio,
            Self::safe_ratio(oos_m.net_profit_pct, is_m.net_profit_pct),
            Self::safe_ratio(oos_m.profit_factor, is_m.profit_factor),
            Self::safe_ratio(oos_m.composite_score, is_m.composite_score),
            Self::safe_ratio(oos_m.curve_r2, is_m.curve_r2),
            Self::safe_ratio(oos_m.sharpe_ratio, is_m.sharpe_ratio),
            Self::safe_ratio(oos_m.max_drawdown_pct, is_m.max_drawdown_pct),
            opt_long,
            opt_short,
            best_fit,
        );
    }
}

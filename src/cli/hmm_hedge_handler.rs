use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine, BacktestReport, WalkForwardResult};
use crate::dashboard::{DashboardGenerator, EquityCurveExporter};
use crate::features::TrainTestSplit;
use crate::models::{HmmDualHedgeStrategy, HmmHedgeStrategyConfig};
use crate::storage::KlineRepository;
use crate::types::{PositionSizing, Timeframe};
use super::fitness_prompt::FitnessPrompt;
use super::input_reader::InputReader;

pub struct HmmHedgeHandler;

impl HmmHedgeHandler {
    pub fn handle(
        conn: &Connection,
        backtest_config: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n---------------------------------------------------------------");
        println!("      🛡️  HMM DUAL HEDGE: RÉGIMEN CAUSAL 3-ESTADOS (L+S)       ");
        println!("---------------------------------------------------------------");
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let tf_str = InputReader::read_with_default("Timeframe para evaluación", "1h".to_string());
        let tf: Timeframe = tf_str.parse()?;

        let mut bt_cfg = backtest_config.clone();
        bt_cfg.timeframe = Some(tf);
        bt_cfg.hedge_mode = true;
        bt_cfg.allow_short = true;
        bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.25);
        bt_cfg.fitness_metric = FitnessPrompt::prompt_selection(bt_cfg.fitness_metric);

        let hmm_cfg = HmmHedgeStrategyConfig::for_timeframe(tf);
        let klines = KlineRepository::load_all(conn, &symbol, tf)?;
        let min_klines = (hmm_cfg.vol_window + 50).max(100);
        if klines.len() < min_klines {
            println!(
                "Se requieren al menos {} velas para evaluar HMM Dual Hedge en [{}]. Velas en DB: {}",
                min_klines,
                tf,
                klines.len()
            );
            return Ok(());
        }

        let train_ratio = InputReader::read_with_default("Proporción In-Sample (IS) (0.70 = 70%)", 0.70);
        let purge_bars = hmm_cfg.vol_window + 2;
        let warmup_bars = hmm_cfg.vol_window.max(50);

        let split = TrainTestSplit::split_chronological_with_warmup(
            &klines,
            train_ratio,
            purge_bars,
            warmup_bars,
        )?;
        println!("\n📊 {}", split.train_range_desc);
        println!("📊 {}", split.test_range_desc);

        println!("\n🧠 [1/3] Ajustando Gaussian HMM causal en In-Sample (IS)...");
        let mut strategy = HmmDualHedgeStrategy::new(hmm_cfg);
        strategy.fit_on_klines(&split.train_klines)?;

        println!("📈 [2/3] Evaluando HMM Dual Hedge en In-Sample (IS) (${:.2})...", bt_cfg.initial_capital);
        let engine_is = BacktestEngine::new(bt_cfg.clone());
        let is_result = engine_is.run(&mut strategy, &split.train_klines)?;

        println!("🚀 [3/3] Evaluando HMM Dual Hedge en Out-of-Sample (OOS) (${:.2})...", bt_cfg.initial_capital);
        strategy.clear_cache();
        let mut oos_cfg = bt_cfg.clone();
        oos_cfg.initial_capital = bt_cfg.initial_capital;
        let engine_oos = BacktestEngine::new(oos_cfg);
        let oos_result = engine_oos.run_with_warmup(
            &mut strategy,
            &split.test_warmup_klines,
            split.warmup_bars,
        )?;

        let model_name = format!("HMM_Dual_Hedge_{}", tf.as_db_str());
        let wf = WalkForwardResult::new(
            model_name,
            tf.as_db_str(),
            is_result,
            oos_result,
            bt_cfg.fitness_metric,
        );

        let is_csv = EquityCurveExporter::save_csv(&wf.is_result.equity_curve, "hmm_dual_hedge", tf.as_db_str(), "IS")?;
        let oos_csv = EquityCurveExporter::save_csv(&wf.oos_result.equity_curve, "hmm_dual_hedge", tf.as_db_str(), "OOS")?;
        let _oos_json = EquityCurveExporter::save_json(&wf.oos_result.equity_curve, "hmm_dual_hedge", tf.as_db_str(), "OOS")?;
        let dashboard_path = DashboardGenerator::generate_and_save(&wf)?;

        println!("\n==================================================");
        println!("    RESUMEN WALK-FORWARD HMM DUAL HEDGE IS vs OOS ");
        println!("==================================================");
        println!("Capital Inicial IS / OOS:   ${:.2} (Idéntico)", bt_cfg.initial_capital);
        println!("Retorno In-Sample (IS):     {:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.is_net_profit_pct, wf.metrics.is_sharpe, wf.metrics.is_max_dd_pct);
        println!("Retorno Out-of-Sample (OOS):{:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.oos_net_profit_pct, wf.metrics.oos_sharpe, wf.metrics.oos_max_dd_pct);
        println!("Profit Factor (OOS):        {:.2}", wf.metrics.oos_profit_factor);
        println!("Win Rate (OOS):             {:.1}%", wf.metrics.oos_win_rate_pct);
        println!("Consistencia R² (OOS):      {:.4}", wf.metrics.oos_curve_r2);
        println!("Ratio de Robustez (OOS/IS): {:.2}x", wf.metrics.robustness_ratio);
        println!("{}", BacktestReport::format_fitness_comparison(&wf.oos_result, bt_cfg.fitness_metric));
        println!("--------------------------------------------------");
        println!("💾 Equity Curve IS guardada en:  {}", is_csv.display());
        println!("💾 Equity Curve OOS guardada en: {}", oos_csv.display());
        println!("🌐 DASHBOARD HTML GENERADO EN:   {}", dashboard_path.display());
        println!("==================================================");

        let report_is = BacktestReport::format_summary(&wf.is_result);
        let report_oos = BacktestReport::format_summary(&wf.oos_result);
        println!("\nReporte Detallado In-Sample (IS):\n{}", report_is);
        println!("\nReporte Detallado Out-of-Sample (OOS):\n{}", report_oos);

        Ok(())
    }
}

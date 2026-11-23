use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine, BacktestReport, WalkForwardResult};
use crate::dashboard::{DashboardGenerator, EquityCurveExporter};
use crate::features::TrainTestSplit;
use crate::models::{ModelConfig, RandomForestModel, RandomForestStrategy};
use crate::storage::KlineRepository;
use crate::types::Timeframe;
use super::input_reader::InputReader;
use super::threshold_calibration_helper::ThresholdCalibrationHelper;

pub struct RandomForestHandler;

impl RandomForestHandler {
    pub fn handle(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n---------------------------------------------------------------");
        println!("         🌲 MODELO RANDOM FOREST: ENSEMBLE POR BAGGING        ");
        println!("---------------------------------------------------------------");
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let tf_str = InputReader::read_with_default("Timeframe para evaluación", "1h".to_string());
        let tf: Timeframe = tf_str.parse()?;

        let mut bt_cfg = bt_cfg.clone();
        bt_cfg.timeframe = Some(tf);
        bt_cfg.fitness_metric = super::fitness_prompt::FitnessPrompt::prompt_selection(bt_cfg.fitness_metric);

        let klines = KlineRepository::load_all(conn, &symbol, tf)?;
        if klines.len() < 100 {
            println!("Se requieren al menos 100 velas en la DB para [{}] (disponibles: {}).", tf, klines.len());
            return Ok(());
        }

        let mut active_m_cfg = ModelConfig::for_timeframe(tf);
        let default_comm = ModelConfig::default().commission_rate;
        if (m_cfg.commission_rate - default_comm).abs() > 1e-12 {
            active_m_cfg.commission_rate = m_cfg.commission_rate;
        } else if (bt_cfg.commission_rate - default_comm).abs() > 1e-12 {
            active_m_cfg.commission_rate = bt_cfg.commission_rate;
        }

        let train_ratio = InputReader::read_with_default("Proporción In-Sample (IS) (0.70 = 70%)", 0.70);
        let purge_bars = active_m_cfg.horizon + 2;

        let split = TrainTestSplit::split_chronological(&klines, train_ratio, purge_bars)?;
        println!("\n📊 {}", split.train_range_desc);
        println!("📊 {}", split.test_range_desc);

        // Calibración de umbrales institucional (OOF Purged CV predeterminado o Fallback IS)
        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train(
            &split.train_klines,
            &active_m_cfg,
            &bt_cfg,
            "Random Forest",
            || RandomForestModel::new(active_m_cfg.clone()),
        )?;

        println!("\n📈 [2/3] Evaluando Backtest In-Sample (IS)...");
        let mut strategy_is = RandomForestStrategy::new(model);
        let engine_is = BacktestEngine::new(bt_cfg.clone());
        let is_result = engine_is.run(&mut strategy_is, &split.train_klines)?;

        println!("\n🚀 [3/3] Evaluando Backtest Out-of-Sample (OOS)...");
        let oos_capital = if is_result.metrics.final_capital > 0.0 {
            is_result.metrics.final_capital
        } else {
            println!("  ⚠️ ALERTA: Capital IS agotado o insolvente (${:.2}). No se inyecta capital zombi a OOS.", 
                is_result.metrics.final_capital);
            0.0
        };

        let oos_result = if oos_capital > 0.0 {
            println!("  -> OOS inicia con el capital: ${:.2}", oos_capital);
            let mut oos_cfg = bt_cfg.clone();
            oos_cfg.initial_capital = oos_capital;
            let engine_oos = BacktestEngine::new(oos_cfg);
            let mut strategy_oos = strategy_is;
            strategy_oos.clear_cache();
            engine_oos.run_with_warmup(&mut strategy_oos, &split.test_warmup_klines, split.warmup_bars)?
        } else {
            crate::backtest::BacktestResult {
                metrics: crate::backtest::MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            }
        };

        let wf = WalkForwardResult::new("RandomForest", tf.as_db_str(), is_result, oos_result, bt_cfg.fitness_metric);

        let is_csv = EquityCurveExporter::save_csv(&wf.is_result.equity_curve, "rf", tf.as_db_str(), "IS")?;
        let oos_csv = EquityCurveExporter::save_csv(&wf.oos_result.equity_curve, "rf", tf.as_db_str(), "OOS")?;
        let dashboard_path = DashboardGenerator::generate_and_save(&wf)?;

        println!("\n==================================================");
        println!("      RESUMEN WALK-FORWARD RANDOM FOREST IS/OOS   ");
        println!("==================================================");
        println!("Calibración Umbrales:       Long > +{:.4}, Short < {:.4} (Score: {:.4})", opt_long, opt_short, best_fit);
        println!("Retorno IS:                 {:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.is_net_profit_pct, wf.metrics.is_sharpe, wf.metrics.is_max_dd_pct);
        println!("Retorno OOS:                {:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.oos_net_profit_pct, wf.metrics.oos_sharpe, wf.metrics.oos_max_dd_pct);
        println!("Profit Factor (OOS):        {:.2}", wf.metrics.oos_profit_factor);
        println!("Win Rate (OOS):             {:.1}%", wf.metrics.oos_win_rate_pct);
        println!("Ratio de Robustez (OOS/IS): {:.2}x", wf.metrics.robustness_ratio);
        println!("--------------------------------------------------");
        println!("💾 Equity Curve IS guardada en:  {}", is_csv.display());
        println!("💾 Equity Curve OOS guardada en: {}", oos_csv.display());
        println!("🌐 DASHBOARD HTML GENERADO EN:   {}", dashboard_path.display());
        println!("==================================================");

        let report_oos = BacktestReport::format_summary(&wf.oos_result);
        println!("\nReporte Detallado Out-of-Sample (OOS):\n{}", report_oos);
        println!("{}", BacktestReport::format_fitness_comparison(&wf.oos_result, bt_cfg.fitness_metric));

        Ok(())
    }
}

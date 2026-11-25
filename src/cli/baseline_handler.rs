use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine, BacktestReport, WalkForwardResult};
use crate::dashboard::{DashboardGenerator, EquityCurveExporter};
use crate::features::TrainTestSplit;
use crate::storage::KlineRepository;
use crate::strategies::EmaCrossoverStrategy;
use crate::types::Timeframe;
use super::input_reader::InputReader;

pub struct BaselineHandler;

impl BaselineHandler {
    pub fn handle(
        conn: &Connection,
        backtest_config: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n---------------------------------------------------------------");
        println!("     📈 ESTRATEGIA DE REFERENCIA HEURÍSTICA: EMA CROSSOVER     ");
        println!("---------------------------------------------------------------");
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let tf_str = InputReader::read_with_default("Timeframe para evaluación", "1h".to_string());
        let tf: Timeframe = tf_str.parse()?;

        let fast_p = InputReader::read_with_default("Período EMA Rápida", 9);
        let slow_p = InputReader::read_with_default("Período EMA Lenta", 21);

        let mut bt_cfg = backtest_config.clone();
        bt_cfg.timeframe = Some(tf);
        bt_cfg.fitness_metric = super::fitness_prompt::FitnessPrompt::prompt_selection(bt_cfg.fitness_metric);

        let klines = KlineRepository::load_all(conn, &symbol, tf)?;
        if klines.len() < slow_p + 50 {
            println!("Se requieren al menos {} velas para evaluar. Velas en DB: {}", slow_p + 50, klines.len());
            return Ok(());
        }

        let train_ratio = InputReader::read_with_default("Proporción In-Sample (IS) (0.70 = 70%)", 0.70);
        let purge_bars = slow_p + 2;

        let split = TrainTestSplit::split_chronological(&klines, train_ratio, purge_bars)?;
        println!("\n📊 {}", split.train_range_desc);
        println!("📊 {}", split.test_range_desc);

        println!("\n📈 [1/2] Evaluando Baseline en In-Sample (IS)...");
        let mut strategy_is = EmaCrossoverStrategy::new(fast_p, slow_p);
        let engine_is = BacktestEngine::new(bt_cfg.clone());
        let is_result = engine_is.run(&mut strategy_is, &split.train_klines)?;

        println!("\n🚀 [2/2] Evaluando Baseline en Out-of-Sample (OOS)...");
        let oos_capital = if is_result.metrics.final_capital > 0.0 {
            is_result.metrics.final_capital
        } else {
            println!(
                "  ⚠️ ALERTA: Capital IS agotado o insolvente (${:.2}). No se inyecta capital zombi a OOS.",
                is_result.metrics.final_capital
            );
            0.0
        };

        let oos_result = if oos_capital > 0.0 {
            println!("  -> OOS inicia con el capital: ${:.2}", oos_capital);
            let mut oos_cfg = bt_cfg.clone();
            oos_cfg.initial_capital = oos_capital;
            let engine_oos = BacktestEngine::new(oos_cfg);
            let mut strategy_oos = EmaCrossoverStrategy::new(fast_p, slow_p);
            engine_oos.run_with_warmup(&mut strategy_oos, &split.test_warmup_klines, split.warmup_bars)?
        } else {
            crate::backtest::BacktestResult {
                metrics: crate::backtest::MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            }
        };

        let model_name = format!("Baseline_EMA_{}_{}", fast_p, slow_p);
        let wf = WalkForwardResult::new(model_name, tf.as_db_str(), is_result, oos_result, bt_cfg.fitness_metric);

        // Guardar curvas de balance
        let is_csv = EquityCurveExporter::save_csv(&wf.is_result.equity_curve, "baseline_ema", tf.as_db_str(), "IS")?;
        let oos_csv = EquityCurveExporter::save_csv(&wf.oos_result.equity_curve, "baseline_ema", tf.as_db_str(), "OOS")?;

        // Generar Dashboard HTML
        let dashboard_path = DashboardGenerator::generate_and_save(&wf)?;

        println!("\n==================================================");
        println!("      RESUMEN WALK-FORWARD BASELINE IS vs OOS     ");
        println!("==================================================");
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

        let report_oos = BacktestReport::format_summary(&wf.oos_result);
        println!("\nReporte Detallado Out-of-Sample (OOS):\n{}", report_oos);

        Ok(())
    }
}

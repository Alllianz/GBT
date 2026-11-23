use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine, FullBacktestRunner, WalkForwardResult};
use crate::models::{ForustModel, ForustStrategy, ModelConfig};
use super::backtest_dataset_helper::BacktestDatasetHelper;
use super::backtest_presenter::BacktestPresenter;
use super::input_reader::InputReader;
use super::menu_printer::MenuPrinter;
use super::threshold_calibration_helper::ThresholdCalibrationHelper;

pub struct ForustHandler;

impl ForustHandler {
    pub fn handle(
        conn: &Connection,
        model_config: &mut ModelConfig,
        backtest_config: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            MenuPrinter::print_forust_menu();
            let choice = InputReader::read_line();

            match choice.as_str() {
                "1" => {
                    Self::run_walk_forward_backtest(conn, model_config, backtest_config)?;
                }
                "2" => {
                    Self::run_full_backtest(conn, model_config, backtest_config)?;
                }
                "3" => {
                    Self::tune_hyperparameters(model_config);
                }
                "0" => break,
                _ => println!("Opción no reconocida."),
            }
        }
        Ok(())
    }

    fn run_walk_forward_backtest(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bundle = match BacktestDatasetHelper::prepare_for_walk_forward(conn, m_cfg, bt_cfg)? {
            Some(b) => b,
            None => return Ok(()),
        };

        // Calibración de umbrales institucional (OOF Purged CV predeterminado o Fallback IS)
        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train(
            &bundle.split.train_klines,
            m_cfg,
            &bundle.bt_cfg,
            "Forust GBDT",
            || ForustModel::new(m_cfg.clone()),
        )?;

        println!("\n📈 [2/3] Evaluando Backtest In-Sample (IS)...");
        let mut strategy_is = ForustStrategy::new(model);
        let engine_is = BacktestEngine::new(bundle.bt_cfg.clone());
        let is_result = engine_is.run(&mut strategy_is, &bundle.split.train_klines)?;

        println!("\n🚀 [3/3] Evaluando Backtest Out-of-Sample (OOS)...");
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
            let mut oos_cfg = bundle.bt_cfg.clone();
            oos_cfg.initial_capital = oos_capital;
            let engine_oos = BacktestEngine::new(oos_cfg);
            let mut strategy_oos = strategy_is;
            strategy_oos.clear_cache();
            engine_oos.run_with_warmup(&mut strategy_oos, &bundle.split.test_warmup_klines, bundle.split.warmup_bars)?
        } else {
            crate::backtest::BacktestResult {
                metrics: crate::backtest::MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            }
        };

        let wf = WalkForwardResult::new(
            "Forust-ML",
            bundle.timeframe.as_db_str(),
            is_result,
            oos_result,
            bundle.bt_cfg.fitness_metric,
        );

        BacktestPresenter::present_walk_forward(
            "Forust GBDT",
            "forust",
            bundle.timeframe,
            &wf,
            opt_long,
            opt_short,
            best_fit,
            bundle.bt_cfg.fitness_metric,
        )?;

        Ok(())
    }

    fn run_full_backtest(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bundle = match BacktestDatasetHelper::prepare_for_full_backtest(conn, m_cfg, bt_cfg)? {
            Some(b) => b,
            None => return Ok(()),
        };

        // Calibración de umbrales institucional (OOF Purged CV predeterminado o Fallback IS)
        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train(
            &bundle.split.train_klines,
            m_cfg,
            &bundle.bt_cfg,
            "Forust GBDT",
            || ForustModel::new(m_cfg.clone()),
        )?;

        println!("\n🚀 [2/2] Ejecutando Backtest Completo Purged (IS -> Purga/Cuarentena -> OOS)...");
        let mut strategy = ForustStrategy::new(model);

        let full_result = FullBacktestRunner::run_purged_full_backtest(
            &mut strategy,
            &bundle.split,
            &bundle.klines,
            &bundle.bt_cfg,
        )?;

        BacktestPresenter::present_full_backtest(
            "Forust GBDT",
            "forust",
            bundle.timeframe,
            &full_result,
            opt_long,
            opt_short,
            best_fit,
            bundle.bt_cfg.fitness_metric,
        )?;

        Ok(())
    }

    fn tune_hyperparameters(cfg: &mut ModelConfig) {
        println!("\n--- CONFIGURACIÓN ACTUAL DE FORUST-ML ---");
        println!("  1. Árboles (iterations):   {}", cfg.num_trees);
        println!("  2. Profundidad Máx:        {}", cfg.max_depth);
        println!("  3. Learning Rate:          {}", cfg.learning_rate);
        println!("  4. Regularización L1:      {}", cfg.l1_regularization);
        println!("  5. Regularización L2:      {}", cfg.l2_regularization);
        println!("  6. Horizonte (velas):      {}", cfg.horizon);
        println!("  7. Umbral Long:            {}", cfg.long_threshold);
        println!("  8. Umbral Short:           {}", cfg.short_threshold);

        let update = InputReader::read_with_default("¿Deseas modificar parámetros? (s/n)", "n".to_string());
        if update.to_lowercase() == "s" {
            cfg.num_trees = InputReader::read_with_default("Número de árboles", cfg.num_trees);
            cfg.max_depth = InputReader::read_with_default("Profundidad máxima", cfg.max_depth);
            cfg.learning_rate = InputReader::read_with_default("Learning rate", cfg.learning_rate);
            cfg.horizon = InputReader::read_with_default("Horizonte de predicción (velas)", cfg.horizon);
            cfg.long_threshold = InputReader::read_with_default("Umbral señal Long", cfg.long_threshold);
            cfg.short_threshold = InputReader::read_with_default("Umbral señal Short", cfg.short_threshold);
            println!("✅ Parámetros actualizados.");
        }
    }
}

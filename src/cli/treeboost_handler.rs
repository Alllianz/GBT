use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine, FullBacktestRunner, WalkForwardResult};
use crate::features::TrainTestSplit;
use crate::models::{ModelConfig, TreeBoostModel, TreeBoostStrategy};
use crate::types::{PositionSizing, Timeframe};
use super::backtest_dataset_helper::{BacktestDatasetBundle, BacktestDatasetHelper};
use super::backtest_presenter::BacktestPresenter;
use super::input_reader::InputReader;
use super::menu_printer::MenuPrinter;
use super::threshold_calibration_helper::ThresholdCalibrationHelper;

pub struct TreeBoostHandler;

impl TreeBoostHandler {
    pub fn handle(
        conn: &Connection,
        model_config: &mut ModelConfig,
        backtest_config: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            MenuPrinter::print_treeboost_menu();
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

    fn resolve_model_config(tf: Timeframe, m_cfg: &ModelConfig, bt_cfg: &BacktestConfig) -> ModelConfig {
        let mut active = ModelConfig::for_timeframe(tf);
        let def = ModelConfig::default();
        if m_cfg.horizon != def.horizon {
            active.horizon = m_cfg.horizon;
        }
        if (m_cfg.long_threshold - def.long_threshold).abs() > 1e-12 {
            active.long_threshold = m_cfg.long_threshold;
        }
        if (m_cfg.short_threshold - def.short_threshold).abs() > 1e-12 {
            active.short_threshold = m_cfg.short_threshold;
        }
        if (m_cfg.commission_rate - def.commission_rate).abs() > 1e-12 {
            active.commission_rate = m_cfg.commission_rate;
        } else if (bt_cfg.commission_rate - def.commission_rate).abs() > 1e-12 {
            active.commission_rate = bt_cfg.commission_rate;
        }
        active
    }

    fn ensure_causal_purge(
        bundle: &mut BacktestDatasetBundle,
        active_m_cfg: &ModelConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let required_purge = active_m_cfg.horizon + 2;
        if bundle.split.purge_bars < required_purge {
            let train_ratio = bundle.split.train_end as f64 / bundle.klines.len() as f64;
            bundle.split = TrainTestSplit::split_chronological(&bundle.klines, train_ratio, required_purge)?;
        }
        Ok(())
    }

    fn run_walk_forward_backtest(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut bundle = match BacktestDatasetHelper::prepare_for_walk_forward(conn, m_cfg, bt_cfg)? {
            Some(b) => b,
            None => return Ok(()),
        };
        bundle.bt_cfg.leverage = 20.0;
        bundle.bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
        let active_m_cfg = Self::resolve_model_config(bundle.timeframe, m_cfg, &bundle.bt_cfg);
        Self::ensure_causal_purge(&mut bundle, &active_m_cfg)?;

        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train(
            &bundle.split.train_klines,
            &active_m_cfg,
            &bundle.bt_cfg,
            "TreeBoost",
            || TreeBoostModel::new(active_m_cfg.clone()).with_timeframe(Some(bundle.timeframe)),
        )?;

        println!("\n📈 [2/3] Evaluando Backtest In-Sample (IS)...");
        let mut strategy_is = TreeBoostStrategy::new(model);
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
            let mut oos_bt_cfg = bundle.bt_cfg.clone();
            oos_bt_cfg.initial_capital = oos_capital;
            oos_bt_cfg.leverage = 20.0;
            oos_bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
            let engine_oos = BacktestEngine::new(oos_bt_cfg);
            let mut strategy_oos = strategy_is;
            strategy_oos.clear_cache();
            engine_oos.run_with_warmup(
                &mut strategy_oos,
                &bundle.split.test_warmup_klines,
                bundle.split.warmup_bars,
            )?
        } else {
            crate::backtest::BacktestResult {
                metrics: crate::backtest::MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            }
        };

        let wf = WalkForwardResult::new(
            "TreeBoost",
            bundle.timeframe.as_db_str(),
            is_result,
            oos_result,
            bundle.bt_cfg.fitness_metric,
        );

        BacktestPresenter::present_walk_forward(
            "TreeBoost",
            "treeboost",
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
        let mut bundle = match BacktestDatasetHelper::prepare_for_full_backtest(conn, m_cfg, bt_cfg)? {
            Some(b) => b,
            None => return Ok(()),
        };
        bundle.bt_cfg.leverage = 20.0;
        bundle.bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
        let active_m_cfg = Self::resolve_model_config(bundle.timeframe, m_cfg, &bundle.bt_cfg);
        Self::ensure_causal_purge(&mut bundle, &active_m_cfg)?;

        // Calibración de umbrales institucional (OOF Purged CV predeterminado o Fallback IS)
        let (model, opt_long, opt_short, best_fit) = ThresholdCalibrationHelper::calibrate_and_train(
            &bundle.split.train_klines,
            &active_m_cfg,
            &bundle.bt_cfg,
            "TreeBoost",
            || TreeBoostModel::new(active_m_cfg.clone()).with_timeframe(Some(bundle.timeframe)),
        )?;

        println!("\n🚀 [2/2] Ejecutando Backtest Completo Purged (IS -> Purga/Cuarentena -> OOS)...");
        let mut strategy = TreeBoostStrategy::new(model);

        let full_result = FullBacktestRunner::run_purged_full_backtest(
            &mut strategy,
            &bundle.split,
            &bundle.klines,
            &bundle.bt_cfg,
        )?;

        BacktestPresenter::present_full_backtest(
            "TreeBoost",
            "treeboost",
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
        println!("\n--- CONFIGURACIÓN ACTUAL DE TREEBOOST ---");
        println!("  1. Horizonte (velas):      {}", cfg.horizon);
        println!("  2. Umbral Long:            {}", cfg.long_threshold);
        println!("  3. Umbral Short:           {}", cfg.short_threshold);

        let update = InputReader::read_with_default("¿Deseas modificar parámetros? (s/n)", "n".to_string());
        if update.to_lowercase() == "s" {
            cfg.horizon = InputReader::read_with_default("Horizonte de predicción (velas)", cfg.horizon);
            cfg.long_threshold = InputReader::read_with_default("Umbral señal Long", cfg.long_threshold);
            cfg.short_threshold = InputReader::read_with_default("Umbral señal Short", cfg.short_threshold);
            println!("✅ Parámetros actualizados.");
        }
    }
}

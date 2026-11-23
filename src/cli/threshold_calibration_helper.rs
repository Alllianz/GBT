use crate::backtest::BacktestConfig;
use crate::models::validation::{OofCalibrationPipeline, PurgedCvConfig, TrainableModel};
use crate::models::{ModelConfig, ThresholdOptimizer};
use crate::types::Kline;
use super::input_reader::InputReader;

/// Asistente CLI para Calibración de Umbrales Causal y Selección de Algoritmo
pub struct ThresholdCalibrationHelper;

impl ThresholdCalibrationHelper {
    /// Ejecuta la calibración de umbrales según la preferencia del usuario,
    /// teniendo como estándar institucional por defecto la Validación Cruzada Purgada Out-of-Fold (OOF).
    pub fn calibrate_and_train<M, F>(
        train_klines: &[Kline],
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
        model_name: &str,
        model_factory: F,
    ) -> Result<(M, f64, f64, f64), String>
    where
        M: TrainableModel,
        F: Fn() -> M,
    {
        println!("\n---------------------------------------------------------------");
        println!("  🎯 MÉTODO DE CALIBRACIÓN DE UMBRALES (LONG / SHORT)");
        println!("  1. Purged K-Fold Out-of-Fold (OOF) [Estándar Institucional - Cero Meta-Overfitting]");
        println!("  2. Optimización In-Sample Clásica [Legacy]");
        println!("---------------------------------------------------------------");
        let choice = InputReader::read_with_default("Seleccione método (1 o 2)", 1);

        if choice == 1 {
            println!("\n🔬 [OOF] Ejecutando Purged K-Fold Cross-Validation (5 folds, horizonte={}, embargo=30)...", m_cfg.horizon);
            let cv_config = PurgedCvConfig::new(5, m_cfg.horizon, 30);
            let (model, calib_res) = OofCalibrationPipeline::calibrate_and_train(
                train_klines,
                cv_config,
                bt_cfg.fitness_metric,
                bt_cfg,
                &model_factory,
            )?;

            println!("✅ Calibración OOF completada sin Look-Ahead Bias.");
            println!(
                "  -> Umbrales OOF óptimos: Long > +{:.4}, Short < {:.4} (Score OOF [{}]: {:.4})",
                calib_res.opt_long, calib_res.opt_short, bt_cfg.fitness_metric, calib_res.oof_score
            );
            Ok((model, calib_res.opt_long, calib_res.opt_short, calib_res.oof_score))
        } else {
            println!("\n⚠️ [Legacy] Entrenando modelo {} directamente en In-Sample (IS)...", model_name);
            let mut model = model_factory();
            model.train(train_klines)?;

            println!("🎯 [Legacy] Calibrando umbrales óptimos bajo métrica fitness [{}] en In-Sample...", bt_cfg.fitness_metric);
            let is_preds = model.predict_batch(train_klines)?;
            let (opt_long, opt_short, best_fit) = ThresholdOptimizer::optimize_in_sample(
                &is_preds,
                train_klines,
                bt_cfg.fitness_metric,
                bt_cfg,
            );
            model.set_thresholds(opt_long, opt_short);
            println!(
                "  -> Umbrales IS calibrados: Long > +{:.4}, Short < {:.4} (Score IS: {:.4})",
                opt_long, opt_short, best_fit
            );
            Ok((model, opt_long, opt_short, best_fit))
        }
    }
}

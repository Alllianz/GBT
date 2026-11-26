use crate::backtest::BacktestConfig;
use crate::features::H4FeatureMatrix;
use crate::microstructure::OrderExecutionType;
use crate::models::validation::{
    OofCalibrationPipeline, PurgedCvConfig, PurgedCvRunner, TrainableModel,
};
use crate::models::{ModelConfig, OofCalibrator, ThresholdOptimizer};
use crate::types::{Kline, Timeframe};
use super::input_reader::InputReader;

pub struct ThresholdCalibrationHelper;

impl ThresholdCalibrationHelper {
    pub fn build_purged_cv_config(m_cfg: &ModelConfig, bt_cfg: &BacktestConfig) -> PurgedCvConfig {
        let embargo_bars = match bt_cfg.timeframe {
            Some(Timeframe::D1) => 15,
            Some(Timeframe::H4) => 90,
            Some(Timeframe::H1) => 48,
            _ => 30,
        };
        PurgedCvConfig::new(5, m_cfg.horizon, embargo_bars)
            .with_timeframe(bt_cfg.timeframe)
            .with_commission_rate(bt_cfg.get_commission_rate(OrderExecutionType::Taker))
    }

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
        let cv_config = Self::build_purged_cv_config(m_cfg, bt_cfg);
        Self::calibrate_and_train_with_cv(train_klines, cv_config, bt_cfg, model_name, model_factory)
    }

    pub fn calibrate_and_train_oof<M, F>(
        train_klines: &[Kline],
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
        model_factory: F,
    ) -> Result<(M, f64, f64, f64), String>
    where
        M: TrainableModel,
        F: Fn() -> M,
    {
        let cv_config = Self::build_purged_cv_config(m_cfg, bt_cfg);
        Self::calibrate_and_train_oof_with_cv(train_klines, cv_config, bt_cfg, model_factory)
    }

    pub fn calibrate_and_train_ensemble_oof<M, F1, F2>(
        train_klines: &[Kline],
        primary_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
        primary_weight: f64,
        primary_factory: F1,
        secondary_factory: F2,
    ) -> Result<(M, M, f64, f64, f64), String>
    where
        M: TrainableModel,
        F1: Fn() -> M,
        F2: Fn() -> M,
    {
        if !primary_weight.is_finite() || !(0.0..=1.0).contains(&primary_weight) {
            return Err(format!("Invalid ensemble primary_weight: {}", primary_weight));
        }

        let cv_config = Self::build_purged_cv_config(primary_cfg, bt_cfg);
        let min_klines = match cv_config.timeframe {
            Some(Timeframe::H4) => H4FeatureMatrix::WARMUP_BARS,
            _ => 50,
        };
        if train_klines.len() < min_klines {
            return Err(format!(
                "Se requieren al menos {} velas para ejecutar la calibración OOF ensemble",
                min_klines
            ));
        }

        println!(
            "\n🔬 [OOF-ENS] Ejecutando Purged K-Fold Cross-Validation dual ({} folds, horizonte={}, embargo={}, w={:.2})...",
            cv_config.n_splits, cv_config.horizon, cv_config.embargo_bars, primary_weight
        );

        let cv_runner = PurgedCvRunner::new(cv_config);
        let cv_a = cv_runner.run_cv(train_klines, &primary_factory)?;
        let cv_b = cv_runner.run_cv(train_klines, &secondary_factory)?;

        if cv_a.predictions.len() != cv_b.predictions.len() {
            return Err(format!(
                "Longitudes OOF incompatibles en ensemble: {} vs {}",
                cv_a.predictions.len(),
                cv_b.predictions.len()
            ));
        }

        let oof_blend: Vec<f64> = cv_a
            .predictions
            .iter()
            .zip(cv_b.predictions.iter())
            .map(|(&a, &b)| primary_weight * a + (1.0 - primary_weight) * b)
            .collect();
        let warmup_bars = cv_a.warmup_bars.max(cv_b.warmup_bars);

        let mut calib_bt_cfg = bt_cfg.clone();
        calib_bt_cfg.hedge_mode = false;
        let (opt_long, opt_short, oof_score) = OofCalibrator::calibrate_asymmetric_with_warmup(
            &oof_blend,
            train_klines,
            calib_bt_cfg.fitness_metric,
            &calib_bt_cfg,
            warmup_bars,
        );

        let mut model_a = primary_factory();
        model_a.train(train_klines)?;
        let mut model_b = secondary_factory();
        model_b.train(train_klines)?;
        model_a.set_thresholds(opt_long, opt_short);

        println!("✅ Calibración OOF Ensemble completada sin Look-Ahead Bias.");
        println!(
            "  -> Umbrales OOF óptimos: Long > +{:.4}, Short < {:.4} (Score OOF [{}]: {:.4})",
            opt_long, opt_short, bt_cfg.fitness_metric, oof_score
        );

        Ok((model_a, model_b, opt_long, opt_short, oof_score))
    }

    pub fn calibrate_and_train_oof_with_cv<M, F>(
        train_klines: &[Kline],
        cv_config: PurgedCvConfig,
        bt_cfg: &BacktestConfig,
        model_factory: F,
    ) -> Result<(M, f64, f64, f64), String>
    where
        M: TrainableModel,
        F: Fn() -> M,
    {
        println!(
            "\n🔬 [OOF] Ejecutando Purged K-Fold Cross-Validation ({} folds, horizonte={}, embargo={})...",
            cv_config.n_splits, cv_config.horizon, cv_config.embargo_bars
        );
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
    }

    pub fn calibrate_and_train_with_cv<M, F>(
        train_klines: &[Kline],
        cv_config: PurgedCvConfig,
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
            Self::calibrate_and_train_oof_with_cv(train_klines, cv_config, bt_cfg, model_factory)
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

use crate::backtest::{BacktestConfig, FullBacktestResult, FullBacktestRunner, Strategy};
use crate::features::{SplitDataset, TrainTestSplit};
use crate::models::validation::{
    OofCalibrationPipeline, OofCalibrationResult, PurgedCvConfig, TrainableModel,
};
use crate::types::Kline;

/// Resultado consolidado del Pipeline Completo Purged CV + OOF + Full Backtest
#[derive(Debug, Clone)]
pub struct PurgedPipelineResult {
    /// Resultado del backtest completo (IS + Purga + OOS)
    pub full_backtest: FullBacktestResult,
    /// Métricas y umbrales óptimos calibrados en Out-of-Fold
    pub oof_calibration: OofCalibrationResult,
    /// Partición temporal causal del dataset
    pub split: SplitDataset,
}

/// Orquestador del Pipeline Purged Causal e Institucional
///
/// Integra de extremo a extremo:
/// 1. Partición cronológica causal con purga y cuarentena (In-Sample vs Out-of-Sample).
/// 2. Validación cruzada purgada sobre el set In-Sample (`PurgedCvRunner`).
/// 3. Calibración de umbrales óptimos libres de meta-overfitting (`OofCalibrator`).
/// 4. Entrenamiento de la instancia definitiva del modelo sobre todo el In-Sample.
/// 5. Backtest institucional continuo con liquidación IS, período de purga y evaluación OOS.
pub struct PurgedPipelineRunner;

impl PurgedPipelineRunner {
    /// Ejecuta el pipeline completo institucional
    pub fn run<M, Strat, F, S>(
        klines: &[Kline],
        train_ratio: f64,
        purge_bars: usize,
        cv_config: PurgedCvConfig,
        bt_config: &BacktestConfig,
        model_factory: F,
        strategy_builder: S,
    ) -> Result<PurgedPipelineResult, String>
    where
        M: TrainableModel,
        Strat: Strategy,
        F: Fn() -> M,
        S: Fn(M) -> Strat,
    {
        if klines.is_empty() {
            return Err("El conjunto de velas está vacío".to_string());
        }

        // 1. Partición cronológica causal In-Sample vs Out-of-Sample
        let split = TrainTestSplit::split_chronological(klines, train_ratio, purge_bars)?;

        // 2 & 3 & 4. Calibración OOF y entrenamiento del modelo definitivo
        let (final_model, oof_calibration) = OofCalibrationPipeline::calibrate_and_train(
            &split.train_klines,
            cv_config,
            bt_config.fitness_metric,
            bt_config,
            &model_factory,
        )?;

        // 5. Construir estrategia y ejecutar Full Backtest causal
        let mut strategy = strategy_builder(final_model);
        let full_backtest = FullBacktestRunner::run_purged_full_backtest(
            &mut strategy,
            &split,
            klines,
            bt_config,
        )?;

        Ok(PurgedPipelineResult {
            full_backtest,
            oof_calibration,
            split,
        })
    }
}

use crate::backtest::BacktestConfig;
use crate::models::OofCalibrator;
use crate::types::{FitnessMetric, Kline};
use super::purged_cv_config::PurgedCvConfig;
use super::purged_cv_runner::PurgedCvRunner;
use super::trainable_model::TrainableModel;

/// Resultado estructurado de la calibración Out-of-Fold (OOF) y entrenamiento final
#[derive(Debug, Clone)]
pub struct OofCalibrationResult {
    /// Umbral óptimo para posiciones Long
    pub opt_long: f64,
    /// Umbral óptimo para posiciones Short
    pub opt_short: f64,
    /// Score obtenido por la métrica de fitness sobre las predicciones Out-of-Fold
    pub oof_score: f64,
    /// Vector de predicciones Out-of-Fold
    pub oof_predictions: Vec<f64>,
    /// Cantidad de particiones (folds) evaluadas
    pub num_folds: usize,
    /// Cantidad de muestras evaluadas
    pub total_samples: usize,
}

/// Pipeline Institucional de Calibración Out-of-Fold (OOF) y Entrenamiento Definitivo
///
/// REGLA FINANCIERA: Elimina el meta-overfitting en la selección de hiperparámetros de umbral
/// optimizando tau_long y tau_short exclusivamente sobre predicciones Out-of-Fold generadas
/// mediante Purged K-Fold Cross-Validation.
pub struct OofCalibrationPipeline;

impl OofCalibrationPipeline {
    /// Ejecuta el pipeline completo:
    /// 1. Generación de predicciones Out-of-Fold (OOF) mediante `PurgedCvRunner`.
    /// 2. Optimización causal de umbrales asimétricos con `OofCalibrator::calibrate_asymmetric`.
    /// 3. Entrenamiento del modelo final definitivo sobre todo el conjunto In-Sample (`train_klines`).
    /// 4. Asignación de los umbrales óptimos calibrados al modelo final.
    pub fn calibrate_and_train<M, F>(
        train_klines: &[Kline],
        cv_config: PurgedCvConfig,
        fitness_metric: FitnessMetric,
        bt_config: &BacktestConfig,
        model_factory: F,
    ) -> Result<(M, OofCalibrationResult), String>
    where
        M: TrainableModel,
        F: Fn() -> M,
    {
        if train_klines.len() < 50 {
            return Err("Se requieren al menos 50 velas para ejecutar la calibración OOF".to_string());
        }

        // 1. Ejecutar Purged Cross-Validation para generar predicciones OOF sin Look-Ahead
        let cv_runner = PurgedCvRunner::new(cv_config);
        let cv_result = cv_runner.run_cv(train_klines, &model_factory)?;

        // 2. Calibrar umbrales óptimos sobre las predicciones Out-of-Fold
        let (opt_long, opt_short, oof_score) = OofCalibrator::calibrate_asymmetric(
            &cv_result.predictions,
            train_klines,
            fitness_metric,
            bt_config,
        );

        // 3. Entrenar la instancia definitiva del modelo sobre el 100% de train_klines
        let mut final_model = model_factory();
        final_model.train(train_klines)?;

        // 4. Asignar los umbrales óptimos calibrados
        final_model.set_thresholds(opt_long, opt_short);

        let calibration_result = OofCalibrationResult {
            opt_long,
            opt_short,
            oof_score,
            oof_predictions: cv_result.predictions,
            num_folds: cv_result.num_folds,
            total_samples: cv_result.total_samples,
        };

        Ok((final_model, calibration_result))
    }
}

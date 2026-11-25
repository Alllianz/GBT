use crate::features::{FeatureMatrix, H4FeatureMatrix, PurgedKFold, TripleBarrierLabeler, VolScaledLabeler};
use crate::types::{Kline, Timeframe};
use super::oof_prediction_result::OofPredictionResult;
use super::purged_cv_config::PurgedCvConfig;
use super::trainable_model::TrainableModel;

pub struct PurgedCvRunner {
    pub config: PurgedCvConfig,
}

impl PurgedCvRunner {
    pub fn new(config: PurgedCvConfig) -> Self {
        Self { config }
    }

    pub fn run_cv<M, F>(
        &self,
        klines: &[Kline],
        model_factory: F,
    ) -> Result<OofPredictionResult, String>
    where
        M: TrainableModel,
        F: Fn() -> M,
    {
        let total_samples = klines.len();
        let mut probe_model = Some(model_factory());
        let is_h4 = matches!(self.config.timeframe, Some(Timeframe::H4))
            || probe_model
                .as_ref()
                .is_some_and(|m| matches!(m.timeframe(), Some(Timeframe::H4)));
        let use_vol_scaled = self.config.use_vol_scaled_labels
            || probe_model
                .as_ref()
                .is_some_and(|m| m.uses_vol_scaled_target());
        let min_samples = if is_h4 { H4FeatureMatrix::WARMUP_BARS } else { 50 };
        if total_samples < min_samples {
            return Err(format!(
                "Muestras insuficientes ({}) para ejecutar Purged CV (mínimo {})",
                total_samples, min_samples
            ));
        }

        let (matrix_flat, total_rows, f_dim, warmup_bars) = if is_h4 {
            let (m, r) = H4FeatureMatrix::build(klines);
            (m, r, H4FeatureMatrix::num_features(), H4FeatureMatrix::WARMUP_BARS)
        } else {
            let (m, r) = FeatureMatrix::build(klines);
            (m, r, FeatureMatrix::num_features(), 60)
        };

        if total_rows != total_samples || matrix_flat.is_empty() {
            return Err(format!(
                "Error extrayendo matriz de características causales: esperadas {} filas, obtenidas {}",
                total_samples, total_rows
            ));
        }

        let y_all = if use_vol_scaled {
            VolScaledLabeler::label_vol_scaled_regression(
                klines,
                self.config.horizon,
                self.config.commission_rate,
                14,
            )
        } else {
            TripleBarrierLabeler::label_regression(
                klines,
                self.config.horizon,
                self.config.commission_rate,
            )
        };

        let kf = PurgedKFold::new(
            self.config.n_splits,
            self.config.horizon,
            self.config.embargo_bars,
        );
        let folds = kf.split(total_samples)?;

        let mut oof_predictions = vec![0.0; total_samples];
        let mut predicted_mask = vec![false; total_samples];

        let max_labelable_idx = total_samples.saturating_sub(self.config.horizon);

        for fold in &folds {
            let valid_train_indices: Vec<usize> = fold
                .train_indices
                .iter()
                .copied()
                .filter(|&idx| idx >= warmup_bars && idx < max_labelable_idx)
                .collect();

            if valid_train_indices.len() < 15 {
                return Err(format!(
                    "Fold {} quedó con muestras de entrenamiento insuficientes ({}) tras purga y embargo",
                    fold.fold_index,
                    valid_train_indices.len()
                ));
            }

            let num_train = valid_train_indices.len();
            let mut x_train = Vec::with_capacity(num_train * f_dim);
            let mut y_train = Vec::with_capacity(num_train);

            for &idx in &valid_train_indices {
                let row_slice = &matrix_flat[(idx * f_dim)..((idx + 1) * f_dim)];
                x_train.extend_from_slice(row_slice);
                y_train.push(y_all[idx]);
            }

            let mut fold_model = probe_model.take().unwrap_or_else(&model_factory);
            fold_model.train_from_matrix(&x_train, &y_train, f_dim)?;

            let valid_test_indices: Vec<usize> = fold
                .test_indices
                .iter()
                .copied()
                .filter(|&idx| idx >= warmup_bars)
                .collect();

            if valid_test_indices.is_empty() {
                continue;
            }

            let num_test = valid_test_indices.len();
            let mut x_test = Vec::with_capacity(num_test * f_dim);

            for &idx in &valid_test_indices {
                let row_slice = &matrix_flat[(idx * f_dim)..((idx + 1) * f_dim)];
                x_test.extend_from_slice(row_slice);
            }

            let test_preds = fold_model.predict_from_matrix(&x_test, f_dim)?;
            if test_preds.len() != num_test {
                return Err(format!(
                    "Fold {}: el modelo devolvió {} predicciones pero se esperaban {}",
                    fold.fold_index,
                    test_preds.len(),
                    num_test
                ));
            }

            for (i, &idx) in valid_test_indices.iter().enumerate() {
                oof_predictions[idx] = test_preds[i];
                predicted_mask[idx] = true;
            }
        }

        let valid_predictions_count = predicted_mask.iter().filter(|&&p| p).count();

        Ok(OofPredictionResult {
            predictions: oof_predictions,
            total_samples,
            num_folds: folds.len(),
            valid_predictions_count,
            warmup_bars,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel {
        trained: bool,
    }

    impl TrainableModel for MockModel {
        fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
            if train_klines.is_empty() {
                return Err("Train set vacío".to_string());
            }
            self.trained = true;
            Ok(())
        }

        fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
            if !self.trained {
                return Err("MockModel no entrenado".to_string());
            }
            Ok(klines.iter().map(|k| k.timestamp as f64 * 0.001).collect())
        }

        fn set_thresholds(&mut self, _long_th: f64, _short_th: f64) {}

        fn train_from_matrix(&mut self, x: &[f64], y: &[f64], _num_features: usize) -> Result<(), String> {
            if x.is_empty() || y.is_empty() {
                return Err("Entrenamiento con matriz vacía".to_string());
            }
            self.trained = true;
            Ok(())
        }

        fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
            if !self.trained {
                return Err("MockModel no entrenado".to_string());
            }
            let rows = x.len() / num_features;
            let preds: Vec<f64> = (0..rows)
                .map(|r| {
                    let f0 = x[r * num_features];
                    f0 * 1.5 + 0.0001
                })
                .collect();
            Ok(preds)
        }
    }

    #[test]
    fn test_purged_cv_runner_completes_all_folds_and_covers_dataset() {
        let n = 200;
        let klines: Vec<Kline> = (0..n)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0 + (i as f64) * 0.1,
                high: 105.0 + (i as f64) * 0.1,
                low: 95.0 + (i as f64) * 0.1,
                close: 102.0 + (i as f64) * 0.1,
                volume: 500.0,
            })
            .collect();

        let config = PurgedCvConfig::new(4, 5, 10);
        let runner = PurgedCvRunner::new(config);

        let result = runner
            .run_cv(&klines, || MockModel { trained: false })
            .expect("Purged CV debe ejecutarse exitosamente");

        assert_eq!(result.total_samples, n);
        assert_eq!(result.num_folds, 4);
        assert_eq!(result.warmup_bars, 60);
        assert_eq!(result.valid_predictions_count, n - 60);
        assert_eq!(result.predictions.len(), n);
        assert!(result.predictions[..60].iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_purged_cv_preserves_temporal_warmup_and_zero_loss_of_bars() {
        let n = 240;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.05).sin() * 10.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 1.0,
                    low: p - 1.0,
                    close: p + 0.2,
                    volume: 1000.0 + (i as f64) * 2.0,
                }
            })
            .collect();

        let config = PurgedCvConfig::new(4, 5, 10);
        let runner = PurgedCvRunner::new(config);

        let result = runner
            .run_cv(&klines, || MockModel { trained: false })
            .expect("Purged CV debe ejecutarse exitosamente");

        assert_eq!(result.valid_predictions_count, n - 60);

        let mid_fold_predictions = &result.predictions[60..120];
        let non_zero_count = mid_fold_predictions.iter().filter(|&&p| p.abs() > 1e-6).count();
        assert_eq!(
            non_zero_count,
            mid_fold_predictions.len(),
            "Todas las barras del fold intermedio deben tener predicciones activas (sin pérdida por falso warmup)"
        );
    }

    #[test]
    fn test_purged_cv_h4_uses_185_warmup() {
        let n = 450;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.05).sin() * 10.0;
                Kline {
                    timestamp: (i as i64 + 1) * 14400,
                    open: p,
                    high: p + 1.0,
                    low: p - 1.0,
                    close: p + 0.2,
                    volume: 1000.0 + (i as f64) * 2.0,
                }
            })
            .collect();

        let config = PurgedCvConfig::new(4, 30, 30).with_timeframe(Some(Timeframe::H4));
        let runner = PurgedCvRunner::new(config);

        let result = runner
            .run_cv(&klines, || MockModel { trained: false })
            .expect("Purged CV H4 debe ejecutarse exitosamente");

        assert_eq!(result.valid_predictions_count, n - H4FeatureMatrix::WARMUP_BARS);
        assert!(result.predictions[..H4FeatureMatrix::WARMUP_BARS].iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_purged_cv_treeboost_h4_vol_scaled() {
        let n = 450;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.07).sin() * 6.0;
                Kline {
                    timestamp: (i as i64 + 1) * 14400,
                    open: p,
                    high: p + 1.5,
                    low: p - 1.5,
                    close: p + 0.25,
                    volume: 1000.0 + (i as f64) * 3.0,
                }
            })
            .collect();

        let m_cfg = crate::models::ModelConfig::for_timeframe(Timeframe::H4);
        let cv_cfg = PurgedCvConfig::new(3, m_cfg.horizon, 30);
        let runner = PurgedCvRunner::new(cv_cfg);

        let result = runner
            .run_cv(&klines, || {
                crate::models::TreeBoostModel::new(m_cfg.clone()).with_timeframe(Some(Timeframe::H4))
            })
            .expect("Purged CV con TreeBoost H4 debe ejecutarse exitosamente");

        assert_eq!(result.warmup_bars, H4FeatureMatrix::WARMUP_BARS);
        assert_eq!(result.valid_predictions_count, n - H4FeatureMatrix::WARMUP_BARS);
    }
}

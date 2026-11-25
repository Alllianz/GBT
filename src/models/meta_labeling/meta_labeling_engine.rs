use crate::backtest::{BacktestConfig, Strategy};
use crate::features::FeatureMatrix;
use crate::models::threshold_optimizer::OofCalibrator;
use crate::models::validation::{PurgedCvConfig, PurgedCvRunner, TrainableModel};
use crate::types::{FitnessMetric, Kline, Signal, SignalDirection};
use super::meta_feature_augmenter::MetaFeatureAugmenter;
use super::meta_gatekeeper_model::MetaGatekeeperModel;
use super::meta_label_config::MetaLabelConfig;
use super::meta_target_builder::MetaTargetBuilder;

/// Motor Institucional de Meta-Labeling en 2 Etapas (Marcos López de Prado)
///
/// Arquitectura Algorítmica:
/// - **Etapa 1 (Modelo Primario):**
///   Entrena un modelo base de alta sensibilidad. Genera predicciones Out-of-Fold con `PurgedCvRunner`
///   (purga y embargo temporal estricto) y calibra los umbrales de activación $(\tau_{\text{long}}, \tau_{\text{short}})$.
///
/// - **Etapa 2 (Gatekeeper Secundario):**
///   Sobre los instantes temporales donde el modelo primario emitió señal fuera de muestra ($d_t \neq 0$),
///   etiqueta el resultado real ($y_{\text{meta}} = 1$ si fue ganador tras comisiones, $0$ si fue perdedor).
///   Entrena un GBDT logístico regularizado sobre las características aumentadas con metadatos de la señal primaria.
///
/// - **Inferencia Causal ($t \le T$):**
///   Al llegar una nueva vela, si el modelo primario emite señal, el Gatekeeper estima la probabilidad
///   $\hat{p}_{\text{meta}, t}$. Si $\hat{p}_{\text{meta}, t} < \text{confidence\_threshold}$, la señal
///   es descartada inmediatamente como falso positivo. Si supera el umbral, se autoriza y modula
///   su tamaño de posición causalmente.
pub struct MetaLabelingEngine {
    config: MetaLabelConfig,
    primary_model: Box<dyn TrainableModel>,
    gatekeeper: Option<MetaGatekeeperModel>,
    long_threshold: f64,
    short_threshold: f64,
    is_trained: bool,
    cached_primary_predictions: Option<Vec<f64>>,
    cached_features: Option<(Vec<f64>, usize)>,
}

impl MetaLabelingEngine {
    pub fn new(config: MetaLabelConfig, primary_model: Box<dyn TrainableModel>) -> Self {
        Self {
            config,
            primary_model,
            gatekeeper: None,
            long_threshold: 0.002,
            short_threshold: -0.002,
            is_trained: false,
            cached_primary_predictions: None,
            cached_features: None,
        }
    }

    /// Ejecuta el entrenamiento causal completo en 2 etapas mediado por predicciones OOF
    pub fn train_2stage<F>(
        &mut self,
        train_klines: &[Kline],
        primary_factory: F,
        backtest_config: &BacktestConfig,
    ) -> Result<(), String>
    where
        F: Fn() -> Box<dyn TrainableModel>,
    {
        let n_samples = train_klines.len();
        if n_samples < 50 {
            return Err(format!(
                "Muestras insuficientes ({}) para Meta-Labeling en 2 etapas (mínimo 50)",
                n_samples
            ));
        }

        // =========================================================================
        // ETAPA 1: Predicciones Out-of-Fold del Modelo Primario y Calibración
        // =========================================================================
        let cv_config = PurgedCvConfig::new(
            self.config.n_splits,
            self.config.horizon,
            self.config.embargo_bars,
        )
        .with_commission_rate(self.config.commission_rate)
        .with_timeframe(backtest_config.timeframe);
        let cv_runner = PurgedCvRunner::new(cv_config);

        let oof_result = cv_runner.run_cv(train_klines, &primary_factory)?;
        let oof_predictions = oof_result.predictions;

        // Calibrar umbrales óptimos libres de meta-overfitting sobre las predicciones OOF
        let (opt_long_th, opt_short_th, _score) = OofCalibrator::calibrate_symmetric(
            &oof_predictions,
            train_klines,
            FitnessMetric::SharpeRatio,
            backtest_config,
        );
        self.long_threshold = opt_long_th;
        self.short_threshold = opt_short_th;

        // Re-entrenar el modelo primario en el 100% de los datos In-Sample
        self.primary_model.train(train_klines)?;
        self.primary_model.set_thresholds(opt_long_th, opt_short_th);

        // =========================================================================
        // ETAPA 2: Meta-Etiquetado y Entrenamiento del Gatekeeper Secundario
        // =========================================================================
        let meta_samples = MetaTargetBuilder::build_from_klines(
            &oof_predictions,
            train_klines,
            self.config.horizon,
            self.config.commission_rate,
            self.long_threshold,
            self.short_threshold,
        );

        if meta_samples.len() >= 10 {
            let (base_matrix_flat, _base_rows) = FeatureMatrix::build(train_klines);
            let base_dim = FeatureMatrix::num_features();

            let (aug_matrix, aug_dim) = MetaFeatureAugmenter::build_augmented_matrix_for_samples(
                &base_matrix_flat,
                base_dim,
                &meta_samples,
                self.long_threshold,
                self.short_threshold,
            );

            let y_binary: Vec<f64> = meta_samples.iter().map(|s| s.meta_label).collect();

            let mut gatekeeper = MetaGatekeeperModel::new(self.config.clone());
            gatekeeper.fit(&aug_matrix, &y_binary, aug_dim)?;
            self.gatekeeper = Some(gatekeeper);
        } else {
            // Si el modelo primario generó menos de 10 operaciones en OOF, no se dispone de suficiente
            // masa estadística para ajustar el Gatekeeper de forma causal; se opera con el modelo primario
            self.gatekeeper = None;
        }

        self.is_trained = true;
        self.clear_cache();
        Ok(())
    }

    pub fn evaluate_signal(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            return Signal::flat(0);
        }
        let ts = history[current_idx].timestamp;

        if !self.is_trained || current_idx < 35 {
            return Signal::flat(ts);
        }

        // Obtener o generar causalmente las predicciones del modelo primario
        let primary_pred = if let Some(ref preds) = self.cached_primary_predictions {
            if current_idx < preds.len() {
                preds[current_idx]
            } else {
                match self.primary_model.predict_batch(history) {
                    Ok(preds) => preds.get(current_idx).copied().unwrap_or(0.0),
                    Err(_) => return Signal::flat(ts),
                }
            }
        } else {
            match self.primary_model.predict_batch(history) {
                Ok(preds) => preds.get(current_idx).copied().unwrap_or(0.0),
                Err(_) => return Signal::flat(ts),
            }
        };

        let direction = if primary_pred > self.long_threshold {
            SignalDirection::Long
        } else if primary_pred < self.short_threshold {
            SignalDirection::Short
        } else {
            SignalDirection::Flat
        };

        if direction == SignalDirection::Flat {
            return Signal::flat(ts);
        }

        // Si disponemos de Gatekeeper secundario, consultamos la probabilidad de éxito de la operación
        if let Some(ref gatekeeper) = self.gatekeeper {
            // Obtener o construir matriz de características base
            let (matrix_flat, num_rows) = if let Some(ref cache) = self.cached_features {
                if current_idx < cache.1 {
                    (cache.0.clone(), cache.1)
                } else {
                    FeatureMatrix::build(history)
                }
            } else {
                FeatureMatrix::build(history)
            };

            let base_dim = FeatureMatrix::num_features();
            if current_idx >= num_rows {
                return Signal::flat(ts);
            }

            let row_start = current_idx * base_dim;
            let row_end = row_start + base_dim;
            if row_end > matrix_flat.len() {
                return Signal::flat(ts);
            }

            let base_row = &matrix_flat[row_start..row_end];
            let aug_row = MetaFeatureAugmenter::augment_row(
                base_row,
                primary_pred,
                self.long_threshold,
                self.short_threshold,
            );

            let win_probability = gatekeeper.predict_probability(&aug_row);

            // FILTRO DE VETO DE LÓPEZ DE PRADO:
            // Si la probabilidad no supera el umbral de convicción, se clasifica como Falso Positivo y se descarta
            if win_probability < self.config.confidence_threshold {
                return Signal::flat(ts);
            }

            let active_th = if direction == SignalDirection::Long {
                self.long_threshold
            } else {
                self.short_threshold.abs()
            };

            let denom = active_th * 3.0;
            let raw_conviction = if denom.is_finite() && denom > 1e-9 && primary_pred.is_finite() {
                primary_pred.abs() / denom
            } else {
                0.5
            };
            let primary_conviction = if raw_conviction.is_finite() {
                raw_conviction.clamp(0.1, 1.0)
            } else {
                0.5
            };

            let strength = if self.config.include_primary_strength {
                let conf_th = self.config.confidence_threshold;
                let prob_factor = if conf_th.is_finite() && conf_th > 1e-9 && win_probability.is_finite() {
                    win_probability / conf_th
                } else {
                    1.0
                };
                let raw_strength = primary_conviction * prob_factor;
                if raw_strength.is_finite() {
                    raw_strength.clamp(0.1, 1.0)
                } else {
                    0.5
                }
            } else if win_probability.is_finite() {
                win_probability.clamp(0.1, 1.0)
            } else {
                0.5
            };

            Signal::new(direction, strength, ts)
        } else {
            let active_th = if direction == SignalDirection::Long {
                self.long_threshold
            } else {
                self.short_threshold.abs()
            };
            let denom = active_th * 3.0;
            let raw_strength = if denom.is_finite() && denom > 1e-9 && primary_pred.is_finite() {
                primary_pred.abs() / denom
            } else {
                0.5
            };
            let strength = if raw_strength.is_finite() {
                raw_strength.clamp(0.1, 1.0)
            } else {
                0.5
            };
            Signal::new(direction, strength, ts)
        }
    }

    pub fn thresholds(&self) -> (f64, f64) {
        (self.long_threshold, self.short_threshold)
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn has_gatekeeper(&self) -> bool {
        self.gatekeeper.is_some()
    }

    pub fn config(&self) -> &MetaLabelConfig {
        &self.config
    }
}

impl Strategy for MetaLabelingEngine {
    fn name(&self) -> &str {
        "MetaLabeling_2Stage_Engine"
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        self.evaluate_signal(history, current_idx)
    }

    fn clear_cache(&mut self) {
        self.cached_primary_predictions = None;
        self.cached_features = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decision_tree::DecisionTreeModel;
    use crate::models::model_config::ModelConfig;

    #[test]
    fn test_meta_labeling_engine_lifecycle() {
        let n = 200;
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = if i % 2 == 0 { 1.5 } else { -1.2 };
            price += change;
            klines.push(Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
            });
        }

        let model_cfg = ModelConfig {
            num_trees: 5,
            max_depth: 2,
            horizon: 5,
            ..Default::default()
        };
        let primary = Box::new(DecisionTreeModel::new(model_cfg.clone()));
        let meta_cfg = MetaLabelConfig::default()
            .with_splits_and_embargo(3, 2)
            .with_confidence_threshold(0.55);

        let mut engine = MetaLabelingEngine::new(meta_cfg, primary);
        let backtest_cfg = BacktestConfig::default();

        let train_res = engine.train_2stage(&klines, move || {
            Box::new(DecisionTreeModel::new(model_cfg.clone()))
        }, &backtest_cfg);

        assert!(train_res.is_ok(), "Entrenamiento en 2 etapas debe completarse");
        assert!(engine.is_trained());

        let sig = engine.evaluate(&klines, 50);
        assert!(sig.strength >= 0.0 && sig.strength <= 1.0);

        let oob_sig = engine.evaluate(&klines, klines.len() + 10);
        assert_eq!(oob_sig.direction, SignalDirection::Flat);
    }

    #[test]
    fn test_meta_labeling_zero_threshold_division_guard() {
        let model_cfg = ModelConfig {
            num_trees: 3,
            max_depth: 2,
            horizon: 5,
            ..Default::default()
        };
        let primary = Box::new(DecisionTreeModel::new(model_cfg));
        let meta_cfg = MetaLabelConfig::default().with_confidence_threshold(0.5);
        let mut engine = MetaLabelingEngine::new(meta_cfg, primary);
        engine.long_threshold = 0.0;
        engine.short_threshold = 0.0;
        engine.is_trained = true;

        let klines: Vec<Kline> = (0..60)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 1000.0,
            })
            .collect();
        engine.cached_primary_predictions = Some(vec![0.01; 60]);
        let sig = engine.evaluate(&klines, 40);
        assert!(sig.strength.is_finite());
        assert!(sig.strength >= 0.0 && sig.strength <= 1.0);
    }
}

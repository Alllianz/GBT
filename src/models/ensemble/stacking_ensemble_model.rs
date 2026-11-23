use rayon::prelude::*;
use crate::backtest::{BacktestConfig, Strategy};
use crate::features::TripleBarrierLabeler;
use crate::models::threshold_optimizer::OofCalibrator;
use crate::models::validation::{PurgedCvConfig, PurgedCvRunner, TrainableModel};
use crate::types::{FitnessMetric, Kline, Signal, SignalDirection};
use super::ensemble_config::EnsembleConfig;
use super::ridge_meta_learner::RidgeMetaLearner;

/// Ensamble Multi-Modelo Stacking / Blending sobre Predicciones Out-of-Fold
///
/// Arquitectura Institucional:
/// 1. Validación Cruzada Purgada ($K$-Fold con purga $H$ y embargo $E$) sobre cada modelo base
///    para generar matrices de predicción 100% Out-of-Fold ($P_{\text{OOF}} \in \mathbb{R}^{N \times M}$).
/// 2. Meta-Aprendizaje Ridge Analítico: Estima el vector óptimo de ponderación $w^*$ resolviendo
///    $(P_{\text{OOF}}^T P_{\text{OOF}} + \lambda I) w = P_{\text{OOF}}^T y$ mediante Cholesky.
/// 3. Proyección Simplex: Garantiza pesos no negativos acotados ($\sum w_m = 1, w_m \ge 0$).
/// 4. Calibración de Umbrales OOF: Optimiza $(\tau_{\text{long}}, \tau_{\text{short}})$ del ensamble
///    sobre el vector combinado $P_{\text{blended, OOF}}$.
/// 5. Re-entrenamiento In-Sample al 100%: Ajusta los estimadores base sobre la serie completa.
///
/// Implementa nativamente `Strategy` para BacktestEngine y `TrainableModel` para validación jerárquica.
pub struct StackingEnsembleModel {
    name: String,
    config: EnsembleConfig,
    meta_learner: Option<RidgeMetaLearner>,
    base_models: Vec<Box<dyn TrainableModel>>,
    long_threshold: f64,
    short_threshold: f64,
    is_trained: bool,
    cached_predictions: Option<Vec<f64>>,
}

impl StackingEnsembleModel {
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            name: "StackingEnsembleModel".to_string(),
            config,
            meta_learner: None,
            base_models: Vec::new(),
            long_threshold: 0.002,
            short_threshold: -0.002,
            is_trained: false,
            cached_predictions: None,
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn add_base_model(&mut self, model: Box<dyn TrainableModel>) {
        self.base_models.push(model);
    }

    /// Entrena el ensamble completo mediante Stacking sobre predicciones Out-of-Fold.
    ///
    /// `model_factories`: Colección de funciones constructoras que instancian modelos base limpios.
    pub fn fit_stacking<F>(
        &mut self,
        train_klines: &[Kline],
        model_factories: Vec<F>,
        backtest_config: &BacktestConfig,
    ) -> Result<(), String>
    where
        F: Fn() -> Box<dyn TrainableModel> + Send + Sync,
    {
        let n_samples = train_klines.len();
        let num_models = model_factories.len();

        if n_samples < 50 {
            return Err(format!(
                "Muestras insuficientes ({}) para entrenar Stacking Ensemble (mínimo 50)",
                n_samples
            ));
        }
        if num_models == 0 {
            return Err("Se requiere al menos 1 modelo base para el ensamble".to_string());
        }

        let cv_config = PurgedCvConfig::new(
            self.config.n_splits,
            self.config.horizon,
            self.config.embargo_bars,
        );
        let cv_runner = PurgedCvRunner::new(cv_config);

        // 1. Generar predicciones OOF para cada modelo base
        let mut model_oof_list = Vec::with_capacity(num_models);
        self.base_models.clear();

        for (m_idx, factory) in model_factories.iter().enumerate() {
            let oof_result = cv_runner.run_cv(train_klines, factory).map_err(|e| {
                format!("Error en CV del modelo base {}: {}", m_idx, e)
            })?;

            if oof_result.predictions.len() != n_samples {
                return Err(format!(
                    "Discrepancia en OOF del modelo base {}: {} vs {}",
                    m_idx,
                    oof_result.predictions.len(),
                    n_samples
                ));
            }

            model_oof_list.push(oof_result.predictions);

            // Re-entrenar el estimador base final en el 100% de la serie In-Sample
            let mut full_model = factory();
            full_model.train(train_klines)?;
            self.base_models.push(full_model);
        }

        // 2. Construir la matriz plana P_OOF (N x M)
        let mut p_oof_flat = vec![0.0; n_samples * num_models];
        for i in 0..n_samples {
            for m in 0..num_models {
                p_oof_flat[i * num_models + m] = model_oof_list[m][i];
            }
        }

        // 3. Obtener el vector objetivo causal de retornos
        let targets = TripleBarrierLabeler::label_regression(
            train_klines,
            self.config.horizon,
            self.config.commission_rate,
        );

        // 4. Ajustar analíticamente el RidgeMetaLearner con proyección simplex
        let mut meta = RidgeMetaLearner::new(
            self.config.ridge_lambda,
            self.config.non_negative_weights,
        );
        meta.fit(&p_oof_flat, &targets, num_models)?;

        // 5. Blended OOF predictions del ensamble: P_blended = P_OOF * w*
        let blended_oof = meta.predict_batch(&p_oof_flat, n_samples);

        // 6. Calibrar umbrales óptimos libres de sesgo sobre las predicciones ensambladas OOF
        let (opt_long, opt_short, _score) = OofCalibrator::calibrate_symmetric(
            &blended_oof,
            train_klines,
            FitnessMetric::SharpeRatio,
            backtest_config,
        );

        self.long_threshold = opt_long;
        self.short_threshold = opt_short;

        // Actualizar umbrales en todos los modelos base
        for model in &mut self.base_models {
            model.set_thresholds(opt_long, opt_short);
        }

        self.meta_learner = Some(meta);
        self.is_trained = true;
        self.clear_cache();

        Ok(())
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.meta_learner.as_ref().map(|m| m.weights())
    }

    pub fn thresholds(&self) -> (f64, f64) {
        (self.long_threshold, self.short_threshold)
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn num_base_models(&self) -> usize {
        self.base_models.len()
    }

    pub fn set_cached_predictions(&mut self, preds: Vec<f64>) {
        self.cached_predictions = Some(preds);
    }
}

impl TrainableModel for StackingEnsembleModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        if self.base_models.is_empty() {
            return Err("No hay modelos base configurados en StackingEnsembleModel".to_string());
        }

        for model in &mut self.base_models {
            model.train(train_klines)?;
        }

        self.is_trained = true;
        self.clear_cache();
        Ok(())
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if !self.is_trained || self.base_models.is_empty() {
            return Err("StackingEnsembleModel no está entrenado o no tiene modelos base".to_string());
        }

        let n = klines.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n < 35 {
            return Ok(vec![0.0; n]);
        }

        let num_models = self.base_models.len();

        let mut all_model_preds = Vec::with_capacity(num_models);
        for (idx, model) in self.base_models.iter().enumerate() {
            let preds = model.predict_batch(klines).map_err(|e| {
                format!("Error en predict_batch del modelo base {}: {}", idx, e)
            })?;
            all_model_preds.push(preds);
        }

        let weights = match &self.meta_learner {
            Some(meta) if meta.is_fitted() => meta.weights().to_vec(),
            _ => vec![1.0 / num_models as f64; num_models],
        };

        let blended: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for m in 0..num_models {
                    let pred = if i < all_model_preds[m].len() {
                        all_model_preds[m][i]
                    } else {
                        0.0
                    };
                    sum += weights[m] * pred;
                }
                if sum.is_finite() { sum } else { 0.0 }
            })
            .collect();

        Ok(blended)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.long_threshold = long_th;
        self.short_threshold = short_th;
        for model in &mut self.base_models {
            model.set_thresholds(long_th, short_th);
        }
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        if self.base_models.is_empty() {
            return Err("No hay modelos base configurados en StackingEnsembleModel".to_string());
        }

        for model in &mut self.base_models {
            model.train_from_matrix(x, y, num_features)?;
        }

        self.is_trained = true;
        self.clear_cache();
        Ok(())
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        if !self.is_trained || self.base_models.is_empty() {
            return Err("StackingEnsembleModel no está entrenado o no tiene modelos base".to_string());
        }

        if x.is_empty() || num_features == 0 {
            return Ok(Vec::new());
        }

        let n = x.len() / num_features;
        let num_models = self.base_models.len();

        let mut all_model_preds = Vec::with_capacity(num_models);
        for (idx, model) in self.base_models.iter().enumerate() {
            let preds = model.predict_from_matrix(x, num_features).map_err(|e| {
                format!("Error en predict_from_matrix del modelo base {}: {}", idx, e)
            })?;
            all_model_preds.push(preds);
        }

        let weights = match &self.meta_learner {
            Some(meta) if meta.is_fitted() => meta.weights().to_vec(),
            _ => vec![1.0 / num_models as f64; num_models],
        };

        let blended: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for m in 0..num_models {
                    sum += weights[m] * all_model_preds[m][i];
                }
                if sum.is_finite() { sum } else { 0.0 }
            })
            .collect();

        Ok(blended)
    }
}

impl Strategy for StackingEnsembleModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        let ts = history[current_idx].timestamp;

        if !self.is_trained || current_idx < 35 {
            return Signal::flat(ts);
        }

        let blended_pred = if let Some(ref preds) = self.cached_predictions {
            if current_idx < preds.len() {
                preds[current_idx]
            } else {
                match self.predict_batch(history) {
                    Ok(preds) => preds.get(current_idx).copied().unwrap_or(0.0),
                    Err(_) => return Signal::flat(ts),
                }
            }
        } else {
            match self.predict_batch(history) {
                Ok(preds) => preds.get(current_idx).copied().unwrap_or(0.0),
                Err(_) => return Signal::flat(ts),
            }
        };

        if blended_pred > self.long_threshold {
            let divisor = (self.long_threshold * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { blended_pred / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Long, strength, ts)
        } else if blended_pred < self.short_threshold {
            let divisor = (self.short_threshold * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { blended_pred.abs() / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            Signal::flat(ts)
        }
    }

    fn clear_cache(&mut self) {
        self.cached_predictions = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decision_tree::DecisionTreeModel;
    use crate::models::model_config::ModelConfig;

    #[test]
    fn test_stacking_ensemble_lifecycle() {
        let n = 120;
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = if i % 2 == 0 { 1.2 } else { -0.8 };
            price += change;
            klines.push(Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 800.0,
            });
        }

        let cfg1 = ModelConfig {
            num_trees: 5,
            max_depth: 2,
            horizon: 5,
            ..Default::default()
        };
        let cfg2 = ModelConfig {
            num_trees: 3,
            max_depth: 3,
            horizon: 5,
            ..Default::default()
        };

        let ens_cfg = EnsembleConfig::default().with_cv_params(3, 5, 2);
        let mut ensemble = StackingEnsembleModel::new(ens_cfg);
        let backtest_cfg = BacktestConfig::default();

        let factories: Vec<Box<dyn Fn() -> Box<dyn TrainableModel> + Send + Sync>> = vec![
            Box::new(move || Box::new(DecisionTreeModel::new(cfg1.clone()))),
            Box::new(move || Box::new(DecisionTreeModel::new(cfg2.clone()))),
        ];

        let fit_res = ensemble.fit_stacking(&klines, factories, &backtest_cfg);
        assert!(fit_res.is_ok(), "Fit Stacking debe completarse sin error");
        assert!(ensemble.is_trained());
        assert_eq!(ensemble.num_base_models(), 2);

        let weights = ensemble.weights().expect("Debe contener pesos calculados");
        assert_eq!(weights.len(), 2);
        let sum_w: f64 = weights.iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-6);

        // Inferencia por batch
        let preds = ensemble.predict_batch(&klines).expect("predict_batch exitoso");
        assert_eq!(preds.len(), n);

        // Inferencia mediante Strategy
        let sig = ensemble.evaluate(&klines, 40);
        assert!(sig.strength >= 0.0 && sig.strength <= 1.0);
    }

    #[test]
    fn test_predict_batch_guard_n_less_than_35() {
        let n = 120;
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let change = if i % 2 == 0 { 1.2 } else { -0.8 };
            price += change;
            klines.push(Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 800.0,
            });
        }

        let cfg1 = ModelConfig {
            num_trees: 5,
            max_depth: 2,
            horizon: 5,
            ..Default::default()
        };
        let cfg2 = ModelConfig {
            num_trees: 3,
            max_depth: 3,
            horizon: 5,
            ..Default::default()
        };

        let ens_cfg = EnsembleConfig::default().with_cv_params(3, 5, 2);
        let mut ensemble = StackingEnsembleModel::new(ens_cfg);
        let backtest_cfg = BacktestConfig::default();

        let factories: Vec<Box<dyn Fn() -> Box<dyn TrainableModel> + Send + Sync>> = vec![
            Box::new(move || Box::new(DecisionTreeModel::new(cfg1.clone()))),
            Box::new(move || Box::new(DecisionTreeModel::new(cfg2.clone()))),
        ];

        let fit_res = ensemble.fit_stacking(&klines, factories, &backtest_cfg);
        assert!(fit_res.is_ok());

        let short_klines = &klines[0..10];
        let preds_short = ensemble.predict_batch(short_klines).expect("predict_batch n<35");
        assert_eq!(preds_short.len(), 10);
        for p in preds_short {
            assert_eq!(p, 0.0);
        }

        let empty_klines: &[Kline] = &[];
        let preds_empty = ensemble.predict_batch(empty_klines).expect("predict_batch n==0");
        assert!(preds_empty.is_empty());
    }
}

use polars::prelude::*;
use treeboost::learner::TreeConfig;
use treeboost::model::{
    AutoBuilder, AutoConfig, AutoModel, BoostingMode, FeatureEngineeringMode, ModeSelection,
    TuningLevel, UniversalConfig,
};
use crate::features::{FeatureMatrix, H4FeatureMatrix, VolScaledLabeler};
use crate::types::{Kline, Timeframe};
use super::model_config::ModelConfig;

pub struct TreeBoostModel {
    model: Option<AutoModel>,
    config: ModelConfig,
    timeframe: Option<Timeframe>,
}

impl TreeBoostModel {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            model: None,
            config,
            timeframe: None,
        }
    }

    pub fn with_timeframe(mut self, timeframe: Option<Timeframe>) -> Self {
        self.timeframe = timeframe;
        self
    }

    pub fn timeframe(&self) -> Option<Timeframe> {
        self.timeframe
    }

    fn is_h4(&self) -> bool {
        matches!(self.timeframe, Some(Timeframe::H4))
    }

    fn feature_names(&self) -> &'static [&'static str] {
        if self.is_h4() {
            &H4FeatureMatrix::FEATURE_NAMES
        } else {
            &FeatureMatrix::FEATURE_NAMES
        }
    }

    fn num_features(&self) -> usize {
        if self.is_h4() {
            H4FeatureMatrix::num_features()
        } else {
            FeatureMatrix::num_features()
        }
    }

    fn build_features(&self, klines: &[Kline]) -> (Vec<f64>, usize, usize, usize) {
        if self.is_h4() {
            let (matrix_flat, rows) = H4FeatureMatrix::build(klines);
            (
                matrix_flat,
                rows,
                H4FeatureMatrix::num_features(),
                H4FeatureMatrix::WARMUP_BARS,
            )
        } else {
            let (matrix_flat, rows) = FeatureMatrix::build(klines);
            (matrix_flat, rows, FeatureMatrix::num_features(), 60)
        }
    }

    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        let min_klines = if self.is_h4() {
            H4FeatureMatrix::WARMUP_BARS
        } else {
            60
        };
        if train_klines.len() < min_klines {
            return Err(format!(
                "Se requieren al menos {} velas para entrenar TreeBoost",
                min_klines
            ));
        }

        let (matrix_flat, rows, f_dim, warmup) = self.build_features(train_klines);

        let valid_rows = rows.saturating_sub(self.config.horizon);
        if valid_rows < warmup + 15 {
            return Err("Muestras insuficientes tras descontar warmup y horizonte".to_string());
        }

        let y = VolScaledLabeler::label_vol_scaled_regression(
            train_klines,
            self.config.horizon,
            self.config.commission_rate,
            14,
        );
        let x_slice = &matrix_flat[(warmup * f_dim)..(valid_rows * f_dim)];
        let y_slice = &y[warmup..valid_rows];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    fn build_dataframe_with_names(
        x: &[f64],
        y: &[f64],
        num_features: usize,
        feature_names: &[&str],
    ) -> Result<DataFrame, String> {
        let rows = y.len();
        let mut columns: Vec<Column> = Vec::with_capacity(num_features + 1);

        for (col_idx, &feature_name) in feature_names.iter().take(num_features).enumerate() {
            let mut col_values = Vec::with_capacity(rows);
            for r in 0..rows {
                let val = x[r * num_features + col_idx];
                col_values.push(val);
            }
            let s = Series::new(feature_name.into(), col_values);
            columns.push(s.into());
        }

        let target_values = y.to_vec();
        let target_series = Series::new("target".into(), target_values);
        columns.push(target_series.into());

        DataFrame::new(rows, columns)
            .map_err(|e| format!("Error creando DataFrame de Polars: {:?}", e))
    }

    #[cfg(test)]
    fn build_dataframe(x: &[f64], y: &[f64], num_features: usize) -> Result<DataFrame, String> {
        Self::build_dataframe_with_names(x, y, num_features, &FeatureMatrix::FEATURE_NAMES)
    }

    pub fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        let valid_rows = y.len();
        if valid_rows < 15 {
            return Err("Muestras insuficientes para entrenar TreeBoost (mínimo 15)".to_string());
        }
        if x.len() != valid_rows * num_features {
            return Err(format!(
                "Dimensión inconsistente en matriz de TreeBoost: esperados {} elementos, recibidos {}",
                valid_rows * num_features,
                x.len()
            ));
        }

        let feature_names = self.feature_names();
        let mut u_cfg = self.build_universal_config(valid_rows);
        let early_stopping_rounds = u_cfg.early_stopping_rounds;

        let mut auto_cfg = AutoConfig::default();
        auto_cfg.tuning_level = TuningLevel::None;
        auto_cfg.feature_engineering = FeatureEngineeringMode::None;
        auto_cfg.mode_selection = ModeSelection::Fixed(BoostingMode::PureTree);
        auto_cfg.val_ratio = 0.0;

        let purge = self.config.horizon;
        let train_rows = (valid_rows as f64 * 0.80) as usize;
        let val_start = train_rows.saturating_add(purge);
        let can_presplit = early_stopping_rounds > 0
            && train_rows >= 15
            && valid_rows.saturating_sub(val_start) >= 5;

        let build_result = if !can_presplit {
            u_cfg.early_stopping_rounds = 0;
            u_cfg.validation_ratio = 0.0;
            let train_df = Self::build_dataframe_with_names(x, y, num_features, feature_names)?;
            auto_cfg.custom_config = Some(u_cfg);
            AutoBuilder::with_config(auto_cfg)
                .fit(&train_df, "target")
                .map_err(|e| format!("Error entrenando TreeBoost AutoModel: {:?}", e))?
        } else {
            let train_df = Self::build_dataframe_with_names(
                &x[0..(train_rows * num_features)],
                &y[0..train_rows],
                num_features,
                feature_names,
            )?;
            let val_df = Self::build_dataframe_with_names(
                &x[(val_start * num_features)..(valid_rows * num_features)],
                &y[val_start..valid_rows],
                num_features,
                feature_names,
            )?;
            let u_cfg_split = self.build_universal_config(train_rows);
            auto_cfg.custom_config = Some(u_cfg_split);
            AutoBuilder::with_config(auto_cfg)
                .with_presplit_validation(val_df)
                .fit(&train_df, "target")
                .map_err(|e| format!("Error entrenando TreeBoost AutoModel: {:?}", e))?
        };

        let trained_auto_model = AutoModel::from_build_result(build_result)
            .map_err(|e| format!("Error construyendo AutoModel desde BuildResult: {:?}", e))?;

        self.model = Some(trained_auto_model);
        Ok(())
    }

    fn build_universal_config(&self, train_rows: usize) -> UniversalConfig {
        let max_depth = self.config.max_depth.clamp(2, 4);
        let max_leaves = (1usize << max_depth).clamp(4, 16);
        let mut tree_config = TreeConfig::default();
        tree_config.max_depth = max_depth;
        tree_config.max_leaves = max_leaves;
        tree_config.lambda = (self.config.l2_regularization * 300.0).clamp(10.0, 80.0);
        tree_config.min_samples_leaf = (train_rows / 25).clamp(40, 160);
        tree_config.post_pruning_gamma = 0.03;
        tree_config.min_gain = self.config.min_split_gain.max(0.008);
        tree_config.entropy_weight = 0.0;

        let mut u_cfg = UniversalConfig::default();
        u_cfg.mode = BoostingMode::PureTree;
        u_cfg.num_rounds = self.config.num_trees.clamp(25, 100);
        u_cfg.learning_rate = self.config.learning_rate.clamp(0.015, 0.05);
        u_cfg.subsample = 0.50;
        u_cfg.colsample = 0.50;
        u_cfg.validation_ratio = 0.0;
        u_cfg.early_stopping_rounds = 6;
        u_cfg.tree_config = tree_config;
        u_cfg
    }

    pub fn predict_single(&self, feature_vec: &[f64]) -> Result<f64, String> {
        let model = self.model.as_ref().ok_or_else(|| "Modelo TreeBoost no entrenado".to_string())?;
        let f_dim = self.num_features();

        if feature_vec.len() != f_dim {
            return Err(format!(
                "Dimensión incorrecta. Esperada: {}, recibida: {}",
                f_dim,
                feature_vec.len()
            ));
        }

        let mut columns: Vec<Column> = Vec::with_capacity(f_dim);
        for (col_idx, &feature_name) in self.feature_names().iter().enumerate() {
            let val = vec![feature_vec[col_idx]];
            let s = Series::new(feature_name.into(), val);
            columns.push(s.into());
        }

        let df = DataFrame::new(1, columns)
            .map_err(|e| format!("Error creando DataFrame de inferencia: {:?}", e))?;

        let preds = model
            .predict(&df)
            .map_err(|e| format!("Error en inferencia de TreeBoost: {:?}", e))?;

        preds
            .first()
            .map(|&v| {
                let f = v as f64;
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            })
            .ok_or_else(|| "TreeBoost no devolvió ninguna predicción".to_string())
    }

    pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if self.model.is_none() {
            return Err("Modelo TreeBoost no entrenado".to_string());
        }
        let (matrix_flat, rows, f_dim, _) = self.build_features(klines);

        if rows == 0 || matrix_flat.is_empty() {
            return Ok(Vec::new());
        }

        self.predict_from_matrix(&matrix_flat, f_dim)
    }

    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        let model = self.model.as_ref().ok_or_else(|| "Modelo TreeBoost no entrenado".to_string())?;
        if x.is_empty() {
            return Ok(Vec::new());
        }
        if num_features == 0 || x.len() % num_features != 0 {
            return Err(format!(
                "Dimensión inconsistente en predict_from_matrix: {} no es múltiplo de {}",
                x.len(),
                num_features
            ));
        }

        let rows = x.len() / num_features;
        let mut columns: Vec<Column> = Vec::with_capacity(num_features);

        for (col_idx, &feature_name) in self.feature_names().iter().take(num_features).enumerate() {
            let mut col_values = Vec::with_capacity(rows);
            for r in 0..rows {
                let val = x[r * num_features + col_idx];
                col_values.push(val);
            }
            let s = Series::new(feature_name.into(), col_values);
            columns.push(s.into());
        }

        let df = DataFrame::new(rows, columns)
            .map_err(|e| format!("Error creando DataFrame de inferencia batch: {:?}", e))?;

        let preds = model
            .predict(&df)
            .map_err(|e| format!("Error en inferencia batch de TreeBoost: {:?}", e))?;

        Ok(preds
            .into_iter()
            .map(|v| {
                let f = v as f64;
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            })
            .collect())
    }

    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.config.set_thresholds(long_th, short_th);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treeboost_model_initial_state() {
        let config = ModelConfig::default();
        let model = TreeBoostModel::new(config);
        assert!(!model.is_trained());
    }

    #[test]
    fn test_treeboost_model_insufficient_samples() {
        let config = ModelConfig::default();
        let mut model = TreeBoostModel::new(config);
        let x = vec![0.0; 10 * FeatureMatrix::num_features()];
        let y = vec![0.0; 10];
        let res = model.train_from_matrix(&x, &y, FeatureMatrix::num_features());
        assert!(res.is_err());
    }

    #[test]
    fn test_treeboost_model_dimension_mismatch() {
        let config = ModelConfig::default();
        let mut model = TreeBoostModel::new(config);
        let x = vec![0.0; 20];
        let y = vec![0.0; 20];
        let res = model.train_from_matrix(&x, &y, FeatureMatrix::num_features());
        assert!(res.is_err());
    }

    #[test]
    fn test_treeboost_build_dataframe() {
        let f_dim = FeatureMatrix::num_features();
        let rows = 5;
        let x = vec![1.0; rows * f_dim];
        let y = vec![0.5; rows];
        let df = TreeBoostModel::build_dataframe(&x, &y, f_dim).unwrap();
        assert_eq!(df.height(), rows);
        assert_eq!(df.width(), f_dim + 1);
        assert!(df.column("target").is_ok());
    }

    #[test]
    fn test_treeboost_train_and_predict() {
        let f_dim = FeatureMatrix::num_features();
        let rows = 45;
        let mut x = Vec::with_capacity(rows * f_dim);
        let mut y = Vec::with_capacity(rows);
        for r in 0..rows {
            let val = (r as f64) * 0.01;
            for c in 0..f_dim {
                x.push(val + (c as f64) * 0.001);
            }
            y.push(val);
        }
        let config = ModelConfig {
            num_trees: 2,
            max_depth: 2,
            learning_rate: 0.1,
            l1_regularization: 0.0,
            l2_regularization: 0.01,
            min_split_gain: 0.0,
            horizon: 3,
            ..Default::default()
        };
        let mut model = TreeBoostModel::new(config);
        let res = model.train_from_matrix(&x, &y, f_dim);
        assert!(res.is_ok());
        assert!(model.is_trained());

        let single_sample = &x[0..f_dim];
        let pred = model.predict_single(single_sample).unwrap();
        assert!(pred.is_finite());

        let batch_preds = model.predict_from_matrix(&x[0..(5 * f_dim)], f_dim).unwrap();
        assert_eq!(batch_preds.len(), 5);
        for p in batch_preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_treeboost_universal_config_tightening() {
        let config = ModelConfig {
            num_trees: 150,
            max_depth: 6,
            learning_rate: 0.1,
            l1_regularization: 0.25,
            l2_regularization: 0.01,
            min_split_gain: 0.0,
            ..Default::default()
        };
        let model = TreeBoostModel::new(config);
        let u_small = model.build_universal_config(40);
        assert_eq!(u_small.num_rounds, 100);
        assert!((u_small.learning_rate - 0.05).abs() < 1e-6);
        assert!((u_small.subsample - 0.50).abs() < 1e-6);
        assert!((u_small.colsample - 0.50).abs() < 1e-6);
        assert_eq!(u_small.early_stopping_rounds, 6);
        assert_eq!(u_small.tree_config.max_depth, 4);
        assert_eq!(u_small.tree_config.max_leaves, 16);
        assert!((u_small.tree_config.lambda - 10.0).abs() < 1e-6);
        assert_eq!(u_small.tree_config.min_samples_leaf, 40);
        assert!((u_small.tree_config.post_pruning_gamma - 0.03).abs() < 1e-6);
        assert!((u_small.tree_config.min_gain - 0.008).abs() < 1e-6);
        assert!((u_small.tree_config.entropy_weight - 0.0).abs() < 1e-6);

        let low_cfg = ModelConfig {
            num_trees: 5,
            max_depth: 1,
            learning_rate: 0.001,
            l2_regularization: 0.10,
            min_split_gain: 0.025,
            ..Default::default()
        };
        let low_model = TreeBoostModel::new(low_cfg);
        let u_large = low_model.build_universal_config(1600);
        assert_eq!(u_large.num_rounds, 25);
        assert!((u_large.learning_rate - 0.015).abs() < 1e-6);
        assert_eq!(u_large.tree_config.max_depth, 2);
        assert_eq!(u_large.tree_config.max_leaves, 4);
        assert!((u_large.tree_config.lambda - 30.0).abs() < 1e-6);
        assert_eq!(u_large.tree_config.min_samples_leaf, 64);
        assert!((u_large.tree_config.min_gain - 0.025).abs() < 1e-6);

        let max_reg_cfg = ModelConfig {
            l2_regularization: 0.50,
            ..Default::default()
        };
        let max_reg_model = TreeBoostModel::new(max_reg_cfg);
        let u_capped = max_reg_model.build_universal_config(4000);
        assert_eq!(u_capped.tree_config.min_samples_leaf, 160);
        assert!((u_capped.tree_config.lambda - 80.0).abs() < 1e-6);
    }

    #[test]
    fn test_treeboost_h4_train_and_predict() {
        let mut cfg = ModelConfig::for_timeframe(Timeframe::H4);
        cfg.horizon = 18;
        let mut model = TreeBoostModel::new(cfg).with_timeframe(Some(Timeframe::H4));
        let klines: Vec<Kline> = (0..260)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.15).sin() * 3.0;
                Kline {
                    timestamp: (i as i64 + 1) * 14400,
                    open: p,
                    high: p + 1.0,
                    low: p - 1.0,
                    close: p + 0.2,
                    volume: 1000.0 + (i as f64) * 5.0,
                }
            })
            .collect();
        model.train(&klines).expect("H4 TreeBoost training should succeed");
        let preds = model.predict_batch(&klines).expect("H4 predict_batch should succeed");
        assert_eq!(preds.len(), klines.len());
        let single_vec = H4FeatureMatrix::extract_single_vector(&klines, 200).unwrap();
        let single_pred = model.predict_single(&single_vec).expect("H4 predict_single should succeed");
        assert!(single_pred.is_finite());
        assert!(model.predict_single(&[0.0; 5]).is_err());
    }

    #[test]
    fn test_treeboost_train_skips_warmup_rows() {
        let mut model = TreeBoostModel::new(ModelConfig::default());
        let short_klines: Vec<Kline> = (0..60)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.1;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p,
                    volume: 1000.0,
                }
            })
            .collect();
        assert!(model.train(&short_klines).is_err());

        let borderline_klines: Vec<Kline> = (0..75)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.1;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p,
                    volume: 1000.0,
                }
            })
            .collect();
        assert!(model.train(&borderline_klines).is_err());

        let enough_klines: Vec<Kline> = (0..90)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.1).sin() * 2.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p + 0.1,
                    volume: 1000.0,
                }
            })
            .collect();
        let res = model.train(&enough_klines);
        assert!(res.is_ok());
        assert!(model.is_trained());

        let f_dim = FeatureMatrix::num_features();
        assert!(model.predict_from_matrix(&vec![0.0; f_dim + 1], f_dim).is_err());
    }
}

use polars::prelude::*;
use treeboost::learner::TreeConfig;
use treeboost::model::{
    AutoBuilder, AutoConfig, AutoModel, BoostingMode, FeatureEngineeringMode, ModeSelection,
    TuningLevel, UniversalConfig,
};
use crate::features::{FeatureMatrix, TripleBarrierLabeler};
use crate::types::Kline;
use super::model_config::ModelConfig;

pub struct TreeBoostModel {
    model: Option<AutoModel>,
    config: ModelConfig,
}

impl TreeBoostModel {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            model: None,
            config,
        }
    }

    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        if train_klines.len() < 50 {
            return Err("Se requieren al menos 50 velas para entrenar TreeBoost".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();

        let valid_rows = rows.saturating_sub(self.config.horizon);
        let warmup = 35;
        if valid_rows < warmup + 15 {
            return Err("Muestras insuficientes tras descontar warmup y horizonte".to_string());
        }

        let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon, self.config.commission_rate);
        let x_slice = &matrix_flat[(warmup * f_dim)..(valid_rows * f_dim)];
        let y_slice = &y[warmup..valid_rows];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    fn build_dataframe(x: &[f64], y: &[f64], num_features: usize) -> Result<DataFrame, String> {
        let rows = y.len();
        let mut columns: Vec<Column> = Vec::with_capacity(num_features + 1);

        for (col_idx, &feature_name) in FeatureMatrix::FEATURE_NAMES.iter().take(num_features).enumerate() {
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

        let (train_rows, val_start) = if valid_rows >= 40 {
            let train_rows = (valid_rows as f64 * 0.80) as usize;
            let purge = self.config.horizon.min(valid_rows.saturating_sub(train_rows + 5));
            let val_start = train_rows + purge;
            (train_rows, val_start)
        } else {
            let train_rows = (valid_rows * 4 / 5).max(10);
            let val_start = train_rows;
            (train_rows, val_start)
        };

        let train_df = Self::build_dataframe(&x[0..(train_rows * num_features)], &y[0..train_rows], num_features)?;
        let val_df = Self::build_dataframe(&x[(val_start * num_features)..(valid_rows * num_features)], &y[val_start..valid_rows], num_features)?;

        let mut u_cfg = self.build_universal_config(train_rows);
        u_cfg.tree_config.max_depth = if train_rows < 1000 {
            self.config.max_depth.clamp(3, 4)
        } else {
            self.config.max_depth.clamp(2, 3)
        };

        let mut auto_cfg = AutoConfig::default();
        auto_cfg.tuning_level = TuningLevel::None;
        auto_cfg.feature_engineering = FeatureEngineeringMode::None;
        auto_cfg.mode_selection = ModeSelection::Fixed(BoostingMode::PureTree);
        auto_cfg.val_ratio = 0.0;
        auto_cfg.custom_config = Some(u_cfg);

        let build_result = AutoBuilder::with_config(auto_cfg)
            .with_presplit_validation(val_df)
            .fit(&train_df, "target")
            .map_err(|e| format!("Error entrenando TreeBoost AutoModel: {:?}", e))?;

        let trained_auto_model = AutoModel::from_build_result(build_result)
            .map_err(|e| format!("Error construyendo AutoModel desde BuildResult: {:?}", e))?;

        self.model = Some(trained_auto_model);
        Ok(())
    }

    fn build_universal_config(&self, train_rows: usize) -> UniversalConfig {
        let max_depth = self.config.max_depth.clamp(2, 3);
        let max_leaves = (1usize << max_depth).clamp(4, 8);
        let mut tree_config = TreeConfig::default();
        tree_config.max_depth = max_depth;
        tree_config.max_leaves = max_leaves;
        tree_config.lambda = if train_rows >= 60 {
            (train_rows as f32 * 0.002).clamp(1.0, 8.0)
        } else {
            1.0
        };
        tree_config.min_samples_leaf = if train_rows >= 60 {
            (train_rows / 70).clamp(6, 20)
        } else {
            5
        };
        tree_config.post_pruning_gamma = 0.0;
        tree_config.min_gain = (self.config.min_split_gain as f32).max(0.0);
        tree_config.entropy_weight = 0.0;

        let mut u_cfg = UniversalConfig::default();
        u_cfg.mode = BoostingMode::PureTree;
        u_cfg.num_rounds = self.config.num_trees.clamp(35, 55);
        u_cfg.learning_rate = (self.config.learning_rate as f32).clamp(0.02, 0.04);
        u_cfg.subsample = 0.65;
        u_cfg.colsample = 0.60;
        u_cfg.validation_ratio = 0.0;
        u_cfg.early_stopping_rounds = 0;
        u_cfg.tree_config = tree_config;
        u_cfg
    }

    pub fn predict_single(&self, feature_vec: &[f64]) -> Result<f64, String> {
        let model = self.model.as_ref().ok_or_else(|| "Modelo TreeBoost no entrenado".to_string())?;
        let f_dim = FeatureMatrix::num_features();

        if feature_vec.len() != f_dim {
            return Err(format!(
                "Dimensión incorrecta. Esperada: {}, recibida: {}",
                f_dim,
                feature_vec.len()
            ));
        }

        let mut columns: Vec<Column> = Vec::with_capacity(f_dim);
        for (col_idx, &feature_name) in FeatureMatrix::FEATURE_NAMES.iter().enumerate() {
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
        let (matrix_flat, rows) = FeatureMatrix::build(klines);
        let f_dim = FeatureMatrix::num_features();

        if rows == 0 || matrix_flat.is_empty() {
            return Ok(Vec::new());
        }

        self.predict_from_matrix(&matrix_flat, f_dim)
    }

    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        let model = self.model.as_ref().ok_or_else(|| "Modelo TreeBoost no entrenado".to_string())?;
        if x.is_empty() || num_features == 0 {
            return Ok(Vec::new());
        }

        let rows = x.len() / num_features;
        let mut columns: Vec<Column> = Vec::with_capacity(num_features);

        for (col_idx, &feature_name) in FeatureMatrix::FEATURE_NAMES.iter().take(num_features).enumerate() {
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
            ..Default::default()
        };
        let model = TreeBoostModel::new(config);
        let u_small = model.build_universal_config(40);
        assert_eq!(u_small.num_rounds, 55);
        assert!((u_small.learning_rate - 0.04).abs() < 1e-6);
        assert!((u_small.subsample - 0.65).abs() < 1e-6);
        assert!((u_small.colsample - 0.60).abs() < 1e-6);
        assert_eq!(u_small.early_stopping_rounds, 0);
        assert_eq!(u_small.tree_config.max_depth, 3);
        assert_eq!(u_small.tree_config.max_leaves, 8);
        assert!((u_small.tree_config.lambda - 1.0).abs() < 1e-6);
        assert_eq!(u_small.tree_config.min_samples_leaf, 5);
        assert!((u_small.tree_config.post_pruning_gamma - 0.0).abs() < 1e-6);
        assert!((u_small.tree_config.entropy_weight - 0.0).abs() < 1e-6);

        let low_cfg = ModelConfig {
            num_trees: 5,
            max_depth: 1,
            learning_rate: 0.001,
            l2_regularization: 0.05,
            ..Default::default()
        };
        let low_model = TreeBoostModel::new(low_cfg);
        let u_large = low_model.build_universal_config(1000);
        assert_eq!(u_large.num_rounds, 35);
        assert!((u_large.learning_rate - 0.02).abs() < 1e-6);
        assert_eq!(u_large.tree_config.max_depth, 2);
        assert_eq!(u_large.tree_config.max_leaves, 4);
        assert!((u_large.tree_config.lambda - 2.0).abs() < 1e-6);
        assert_eq!(u_large.tree_config.min_samples_leaf, 14);

        let u_capped = low_model.build_universal_config(4000);
        assert_eq!(u_capped.tree_config.min_samples_leaf, 20);
        assert!((u_capped.tree_config.lambda - 8.0).abs() < 1e-6);
        let u_min_leaf = low_model.build_universal_config(60);
        assert_eq!(u_min_leaf.tree_config.min_samples_leaf, 6);
        assert!((u_min_leaf.tree_config.lambda - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_treeboost_train_skips_warmup_rows() {
        let mut model = TreeBoostModel::new(ModelConfig::default());
        let short_klines: Vec<Kline> = (0..52)
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
    }
}

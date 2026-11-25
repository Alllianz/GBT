use rayon::prelude::*;
use crate::features::{FeatureMatrix, H4FeatureMatrix, TripleBarrierLabeler};
use crate::models::decision_tree::CartTree;
use crate::models::model_config::ModelConfig;
use crate::models::random_forest::TemporalBlockBootstrap;
use crate::models::regularization::TreeRegularizationConfig;
use crate::types::{Kline, Timeframe};

pub struct RandomForestModel {
    trees: Vec<CartTree>,
    config: ModelConfig,
    num_trees: usize,
    is_trained: bool,
    timeframe: Option<Timeframe>,
}

impl RandomForestModel {
    pub fn new(config: ModelConfig) -> Self {
        let num_trees = config.num_trees.max(10);
        Self {
            trees: Vec::with_capacity(num_trees),
            config,
            num_trees,
            is_trained: false,
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

    fn build_features(&self, klines: &[Kline]) -> (Vec<f64>, usize, usize, usize) {
        if matches!(self.timeframe, Some(Timeframe::H4)) {
            let (matrix_flat, rows) = H4FeatureMatrix::build(klines);
            (
                matrix_flat,
                rows,
                H4FeatureMatrix::num_features(),
                H4FeatureMatrix::WARMUP_BARS,
            )
        } else {
            let (matrix_flat, rows) = FeatureMatrix::build(klines);
            (matrix_flat, rows, FeatureMatrix::num_features(), 35)
        }
    }

    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        let n = train_klines.len();
        if n < 50 {
            return Err("Se requieren al menos 50 velas para entrenar Random Forest".to_string());
        }

        let (matrix_flat, rows, f_dim, warmup) = self.build_features(train_klines);
        let valid_end = rows.saturating_sub(self.config.horizon);
        if valid_end <= warmup || (valid_end - warmup) < 15 {
            return Err("Muestras insuficientes tras descontar warmup y horizonte".to_string());
        }

        let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon, self.config.commission_rate);
        let x_slice = &matrix_flat[(warmup * f_dim)..(valid_end * f_dim)];
        let y_slice = &y[warmup..valid_end];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    pub fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        let valid_rows = y.len();
        if valid_rows < 15 {
            return Err("Muestras insuficientes para entrenar Random Forest (mínimo 15)".to_string());
        }
        if x.len() != valid_rows * num_features {
            return Err(format!(
                "Dimensión inconsistente en Random Forest: esperados {} elementos, recibidos {}",
                valid_rows * num_features,
                x.len()
            ));
        }

        self.trees.clear();

        let m_try = ((num_features as f64).sqrt().ceil() as usize).max(2).min(num_features);

        let max_depth = self.config.max_depth.clamp(2, 3);
        let min_samples_leaf = if valid_rows >= 60 {
            (valid_rows / 40).clamp(12, 45)
        } else {
            5
        };
        let l2_reg = (valid_rows as f64 * 0.006).clamp(5.0, 25.0);
        let reg_config = TreeRegularizationConfig::from_model_config(&self.config)
            .with_l1((self.config.l1_regularization as f64).max(0.005))
            .with_l2(l2_reg)
            .with_min_child_weight(20.0)
            .with_min_split_gain((self.config.min_split_gain as f64).max(1e-7));
        let block_size = (self.config.horizon * 2).max(36);
        let sample_size = (valid_rows * 60) / 100;

        self.trees = (0..self.num_trees)
            .into_par_iter()
            .map(|tree_idx| {
                let tree_seed = 42u64.wrapping_add((tree_idx as u64).wrapping_mul(1_000_003));
                let mut seed = tree_seed;
                let rng = move || {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    seed
                };

                let bootstrap_indices = TemporalBlockBootstrap::sample_indices(
                    valid_rows,
                    block_size,
                    sample_size,
                    rng,
                );

                let mut tree = CartTree::new_regularized(max_depth, min_samples_leaf, reg_config, tree_seed)
                    .with_feature_subsample(Some(m_try), tree_seed);
                tree.fit_indices(x, y, &bootstrap_indices, num_features, None);
                tree
            })
            .collect();

        self.is_trained = true;
        Ok(())
    }

    pub fn predict_single(&self, features: &[f64]) -> Result<f64, String> {
        if !self.is_trained || self.trees.is_empty() {
            return Err("Random Forest no está entrenado".to_string());
        }

        let sum: f64 = self.trees.iter().map(|t| t.predict(features)).sum();
        let val = sum / self.trees.len() as f64;
        Ok(if val.is_finite() { val } else { 0.0 })
    }

    pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if !self.is_trained || self.trees.is_empty() {
            return Err("Random Forest no está entrenado".to_string());
        }

        let (matrix_flat, rows, f_dim, _) = self.build_features(klines);

        if rows == 0 || matrix_flat.is_empty() {
            return Ok(Vec::new());
        }

        self.predict_from_matrix(&matrix_flat, f_dim)
    }

    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        if !self.is_trained || self.trees.is_empty() {
            return Err("Random Forest no está entrenado".to_string());
        }
        if x.is_empty() || num_features == 0 {
            return Ok(Vec::new());
        }

        let rows = x.len() / num_features;
        let n_trees = self.trees.len() as f64;

        let preds: Vec<f64> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let row_feats = &x[(r * num_features)..((r + 1) * num_features)];
                let sum: f64 = self.trees.iter().map(|t| t.predict(row_feats)).sum();
                let val = sum / n_trees;
                if val.is_finite() { val } else { 0.0 }
            })
            .collect();

        Ok(preds)
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
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

    fn sample_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.25 + ((i % 7) as f64 - 3.0) * 0.1;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.6,
                    low: p - 0.6,
                    close: p + 0.1,
                    volume: 100.0 + (i % 10) as f64 * 5.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_train_skips_warmup_and_checks_min_samples() {
        let mut cfg = ModelConfig::default();
        cfg.num_trees = 10;
        cfg.horizon = 5;

        let mut model_too_short = RandomForestModel::new(cfg.clone());
        let short_klines = sample_klines(50);
        assert!(model_too_short.train(&short_klines).is_err());

        let mut model_ok = RandomForestModel::new(cfg);
        let klines = sample_klines(65);
        assert!(model_ok.train(&klines).is_ok());
        assert!(model_ok.is_trained());
        assert_eq!(model_ok.trees.len(), 10);
    }

    #[test]
    fn test_train_from_matrix_caps_depth_adaptive_leaf_and_regularization() {
        let mut cfg = ModelConfig::default();
        cfg.num_trees = 10;
        cfg.max_depth = 10;
        cfg.l1_regularization = 0.001;
        cfg.l2_regularization = 0.01;
        cfg.horizon = 5;

        let f_dim = 6;
        let small_rows = 50;
        let x_small: Vec<f64> = (0..small_rows * f_dim).map(|i| (i as f64) * 0.01).collect();
        let y_small: Vec<f64> = (0..small_rows).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();

        let mut small_model = RandomForestModel::new(cfg.clone());
        small_model.train_from_matrix(&x_small, &y_small, f_dim).unwrap();
        assert_eq!(small_model.trees.len(), 10);
        for tree in &small_model.trees {
            assert_eq!(tree.max_depth, 3);
            assert_eq!(tree.min_samples_leaf, 5);
            assert!((tree.regularization.l1_alpha - 0.005).abs() < 1e-12);
            assert!((tree.regularization.l2_lambda - 5.0).abs() < 1e-12);
            assert!((tree.regularization.min_child_weight - 20.0).abs() < 1e-12);
        }

        cfg.max_depth = 1;
        cfg.l1_regularization = 0.02;
        cfg.l2_regularization = 0.1;
        let large_rows = 800;
        let x_large: Vec<f64> = (0..large_rows * f_dim).map(|i| ((i % 97) as f64) * 0.05).collect();
        let y_large: Vec<f64> = (0..large_rows).map(|i| if i % 3 == 0 { 0.015 } else { -0.008 }).collect();

        let mut large_model = RandomForestModel::new(cfg);
        large_model.train_from_matrix(&x_large, &y_large, f_dim).unwrap();
        for tree in &large_model.trees {
            assert_eq!(tree.max_depth, 2);
            assert_eq!(tree.min_samples_leaf, 20);
            assert!((tree.regularization.l1_alpha - 0.02f32 as f64).abs() < 1e-12);
            assert!((tree.regularization.l2_lambda - 5.0).abs() < 1e-12);
            assert!((tree.regularization.min_child_weight - 20.0).abs() < 1e-12);
        }

        let preds = large_model.predict_from_matrix(&x_large[0..(10 * f_dim)], f_dim).unwrap();
        assert_eq!(preds.len(), 10);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_h4_timeframe_uses_h4_warmup_and_features() {
        let cfg = ModelConfig::for_timeframe(Timeframe::H4);
        let mut model_short = RandomForestModel::new(cfg.clone()).with_timeframe(Some(Timeframe::H4));
        assert_eq!(model_short.timeframe(), Some(Timeframe::H4));
        assert!(model_short.train(&sample_klines(210)).is_err());

        let mut model_ok = RandomForestModel::new(cfg).with_timeframe(Some(Timeframe::H4));
        let klines = sample_klines(240);
        assert!(model_ok.train(&klines).is_ok());
        let preds = model_ok.predict_batch(&klines).unwrap();
        assert_eq!(preds.len(), 240);
        assert!(preds.iter().all(|p| p.is_finite()));
    }
}

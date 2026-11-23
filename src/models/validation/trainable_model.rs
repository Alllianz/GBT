use crate::models::decision_tree::DecisionTreeModel;
use crate::models::forust_model::ForustModel;
use crate::models::native_gbdt::NativeGbdtModel;
use crate::models::random_forest::RandomForestModel;
use crate::models::treeboost_model::TreeBoostModel;
use crate::types::Kline;

/// Trait institucional para modelos de Machine Learning entrenables y evaluables de manera causal.
pub trait TrainableModel: Send + Sync {
    /// Entrena el modelo causalmente sobre el conjunto de velas provisto
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String>;

    /// Inferencia vectorizada en batch sobre una serie de velas
    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String>;

    /// Asigna los umbrales de decisión (Long, Short) óptimos calibrados
    fn set_thresholds(&mut self, long_th: f64, short_th: f64);

    /// Entrena el modelo causalmente directo sobre una matriz plana de features y etiquetas
    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String>;

    /// Inferencia vectorizada sobre una matriz plana de features pre-computada [N x num_features]
    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String>;
}

impl TrainableModel for NativeGbdtModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        self.train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        self.predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        self.train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        self.predict_from_matrix(x, num_features)
    }
}

impl TrainableModel for ForustModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        self.train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        self.predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        self.train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        self.predict_from_matrix(x, num_features)
    }
}

impl TrainableModel for TreeBoostModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        self.train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        self.predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        self.train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        self.predict_from_matrix(x, num_features)
    }
}

impl TrainableModel for DecisionTreeModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        self.train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        self.predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        self.train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        self.predict_from_matrix(x, num_features)
    }
}

impl TrainableModel for RandomForestModel {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        self.train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        self.predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        self.set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        self.train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        self.predict_from_matrix(x, num_features)
    }
}

impl TrainableModel for Box<dyn TrainableModel> {
    fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        (**self).train(train_klines)
    }

    fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        (**self).predict_batch(klines)
    }

    fn set_thresholds(&mut self, long_th: f64, short_th: f64) {
        (**self).set_thresholds(long_th, short_th);
    }

    fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        (**self).train_from_matrix(x, y, num_features)
    }

    fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        (**self).predict_from_matrix(x, num_features)
    }
}

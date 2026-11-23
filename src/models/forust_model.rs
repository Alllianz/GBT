use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};
use crate::features::{FeatureMatrix, TripleBarrierLabeler};
use crate::types::Kline;
use super::model_config::ModelConfig;

pub struct ForustModel {
    booster: GradientBooster,
    config: ModelConfig,
    is_trained: bool,
}

impl ForustModel {
    pub fn new(config: ModelConfig) -> Self {
        let mut booster = GradientBooster::default();
        booster.objective_type = ObjectiveType::SquaredLoss;
        booster.iterations = config.num_trees;
        booster.max_depth = config.max_depth;
        booster.learning_rate = config.learning_rate;
        booster.l1 = config.l1_regularization;
        booster.l2 = config.l2_regularization;

        Self {
            booster,
            config,
            is_trained: false,
        }
    }

    /// Entrena el modelo causalmente sobre el conjunto In-Sample de velas
    pub fn train(&mut self, train_klines: &[Kline]) -> Result<(), String> {
        let n = train_klines.len();
        if n < 50 {
            return Err("Se requieren al menos 50 velas para entrenar el modelo Forust".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(train_klines);
        let f_dim = FeatureMatrix::num_features();

        if rows == 0 || matrix_flat.is_empty() {
            return Err("Error construyendo matriz de variables causales".to_string());
        }

        // Generar etiquetas causales (retorno a H velas deduciendo comisiones netas roundtrip)
        let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon, self.config.commission_rate);

        // Omitir las últimas `horizon` filas donde la etiqueta no está completa
        let valid_rows = rows.saturating_sub(self.config.horizon);
        if valid_rows < 30 {
            return Err("Muestras válidas insuficientes tras descontar el horizonte de etiqueta".to_string());
        }

        let x_slice = &matrix_flat[0..(valid_rows * f_dim)];
        let y_slice = &y[0..valid_rows];

        self.train_from_matrix(x_slice, y_slice, f_dim)
    }

    /// Entrena el modelo directamente sobre una matriz plana y etiquetas pre-computadas causales
    pub fn train_from_matrix(&mut self, x: &[f64], y: &[f64], num_features: usize) -> Result<(), String> {
        let valid_rows = y.len();
        if valid_rows < 15 {
            return Err("Muestras insuficientes para entrenar Forust (mínimo 15)".to_string());
        }
        if x.len() != valid_rows * num_features {
            return Err(format!(
                "Dimensión inconsistente en matriz de Forust: esperados {} elementos, recibidos {}",
                valid_rows * num_features,
                x.len()
            ));
        }

        let weights = vec![1.0; valid_rows];
        let matrix = Matrix::new(x, valid_rows, num_features);

        self.booster
            .fit(&matrix, y, &weights, None)
            .map_err(|e| format!("Error en fit de Forust: {:?}", e))?;

        self.is_trained = true;
        Ok(())
    }

    /// Realiza inferencia causal sobre un vector de features
    pub fn predict_single(&self, feature_vec: &[f64]) -> Result<f64, String> {
        if !self.is_trained {
            return Err("El modelo Forust no ha sido entrenado".to_string());
        }

        let f_dim = FeatureMatrix::num_features();
        if feature_vec.len() != f_dim {
            return Err(format!(
                "Dimensión de vector esperada: {}, recibida: {}",
                f_dim,
                feature_vec.len()
            ));
        }

        let matrix = Matrix::new(feature_vec, 1, f_dim);
        let preds = self.booster.predict(&matrix, false);

        preds
            .first()
            .copied()
            .map(|val| if val.is_finite() { val } else { 0.0 })
            .ok_or_else(|| "Error al obtener predicción de Forust".to_string())
    }

    /// Inferencia vectorizada en paralelo a través de los 20 hilos para toda la serie temporal
    pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("El modelo Forust no ha sido entrenado".to_string());
        }

        let (matrix_flat, rows) = FeatureMatrix::build(klines);
        let f_dim = FeatureMatrix::num_features();

        if rows == 0 || matrix_flat.is_empty() {
            return Ok(Vec::new());
        }

        self.predict_from_matrix(&matrix_flat, f_dim)
    }

    /// Inferencia vectorizada a partir de una matriz plana pre-computada [N x num_features]
    pub fn predict_from_matrix(&self, x: &[f64], num_features: usize) -> Result<Vec<f64>, String> {
        if !self.is_trained {
            return Err("El modelo Forust no ha sido entrenado".to_string());
        }
        if x.is_empty() || num_features == 0 {
            return Ok(Vec::new());
        }

        let rows = x.len() / num_features;
        let matrix = Matrix::new(x, rows, num_features);
        // parallel = true: Forust ejecuta la inferencia con Rayon en paralelo
        let preds = self.booster.predict(&matrix, true);
        Ok(preds
            .into_iter()
            .map(|v| if v.is_finite() { v } else { 0.0 })
            .collect())
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

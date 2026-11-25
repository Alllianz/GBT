/// Resultado consolidado de predicciones Out-of-Fold (OOF) generadas por Purged Cross-Validation
///
/// Todas las predicciones en `predictions` fueron generadas de forma estrictamente Out-of-Fold,
/// garantizando que ningún valor predicho en el índice $t$ utilizó información del propio punto $t$,
/// ni de sus etiquetas futuras (purga), ni de su estela temporal autorregresiva (embargo).
#[derive(Debug, Clone)]
pub struct OofPredictionResult {
    /// Vector de predicciones OOF de longitud N coincidente con la serie evaluada
    pub predictions: Vec<f64>,
    /// Cantidad total de observaciones en la serie original
    pub total_samples: usize,
    /// Cantidad de particiones (folds) evaluadas
    pub num_folds: usize,
    /// Cantidad de predicciones efectivamente asignadas fuera de muestra
    pub valid_predictions_count: usize,
    pub warmup_bars: usize,
}

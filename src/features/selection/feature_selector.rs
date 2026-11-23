use super::collinearity_pruner::CollinearityPruner;
use super::correlation_matrix::RankCorrelationCalculator;
use super::purged_mda::{MdaScore, PurgedMdaEvaluator};
use super::selection_config::FeatureSelectionConfig;

/// Orquestador Institucional de Selección y Poda Causal de Características (CausalFeatureSelector)
///
/// Combina secuencialmente:
/// 1. Cómputo de la matriz de dependencia no lineal con correlación de rangos de Spearman.
/// 2. Evaluación causal out-of-sample mediante Purged MDA (Mean Decrease Accuracy / MSE).
/// 3. Poda de multicolinealidad reteniendo el representante de mayor relevancia causal por clúster.
/// 4. Poda estricta de ruido blanco y variables no informativas (MDA <= 0).
/// 5. Proyección ortogonal vectorizada para entrenamiento e inferencia individual causal (t <= T).
#[derive(Debug, Clone)]
pub struct CausalFeatureSelector {
    pub config: FeatureSelectionConfig,
    pub pruner: CollinearityPruner,
    pub evaluator: PurgedMdaEvaluator,
    pub selected_indices: Vec<usize>,
    pub mda_scores: Vec<MdaScore>,
    pub is_fitted: bool,
}

impl CausalFeatureSelector {
    pub fn new(config: FeatureSelectionConfig) -> Self {
        let max_corr = config.max_correlation_threshold;
        Self {
            pruner: CollinearityPruner::new(max_corr),
            evaluator: PurgedMdaEvaluator::new(config.clone()),
            config,
            selected_indices: Vec::new(),
            mda_scores: Vec::new(),
            is_fitted: false,
        }
    }

    /// Ajusta el selector causalmente identificando las características ortogonales e informativas
    pub fn fit(
        &mut self,
        features_flat: &[f64],
        targets: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<&[usize], String> {
        if n_cols == 0 {
            return Err("Número de columnas debe ser mayor a 0".to_string());
        }

        // 1. Matriz de correlación de Spearman entre columnas
        let corr_matrix = RankCorrelationCalculator::compute_spearman_matrix(
            features_flat,
            n_rows,
            n_cols,
        );

        // 2. Evaluación causal MDA con purga y embargo
        let mda_scores = self
            .evaluator
            .evaluate(features_flat, targets, n_rows, n_cols)?;

        let mda_weights: Vec<f64> = mda_scores.iter().map(|s| s.mean_mda).collect();

        // 3. Poda de colinealidad priorizando mayor relevancia causal
        let non_collinear = self.pruner.prune_features(&corr_matrix, Some(&mda_weights));

        // 4. Filtrar por significancia informativa (MDA > min_threshold)
        let mut final_selected: Vec<usize> = non_collinear
            .into_iter()
            .filter(|&idx| mda_scores[idx].is_informative)
            .collect();

        // Si todas las variables quedan podadas por ruido, preservar al menos la de mayor MDA
        if final_selected.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;
            for (idx, score) in mda_scores.iter().enumerate() {
                if score.mean_mda > best_score {
                    best_score = score.mean_mda;
                    best_idx = idx;
                }
            }
            final_selected.push(best_idx);
        }

        final_selected.sort_unstable();

        self.selected_indices = final_selected;
        self.mda_scores = mda_scores;
        self.is_fitted = true;

        Ok(&self.selected_indices)
    }

    /// Proyecta una matriz aplanada conservando únicamente las columnas seleccionadas
    pub fn transform(
        &self,
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(Vec<f64>, usize), String> {
        if !self.is_fitted {
            return Err("CausalFeatureSelector no ha sido ajustado (llamar a fit previamente)".to_string());
        }

        let pruned = CollinearityPruner::prune_matrix(
            features_flat,
            n_rows,
            n_cols,
            &self.selected_indices,
        );

        Ok((pruned, self.selected_indices.len()))
    }

    /// Ajusta y transforma en una sola operación causal
    pub fn fit_transform(
        &mut self,
        features_flat: &[f64],
        targets: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(Vec<f64>, usize), String> {
        self.fit(features_flat, targets, n_rows, n_cols)?;
        self.transform(features_flat, n_rows, n_cols)
    }

    /// Proyecta un vector de inferencia causal individual (t <= T)
    pub fn transform_single(&self, row_features: &[f64]) -> Result<Vec<f64>, String> {
        if !self.is_fitted {
            return Err("CausalFeatureSelector no ha sido ajustado".to_string());
        }

        Ok(CollinearityPruner::prune_vector(
            row_features,
            &self.selected_indices,
        ))
    }

    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    pub fn mda_scores(&self) -> &[MdaScore] {
        &self.mda_scores
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    pub fn num_selected(&self) -> usize {
        self.selected_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_feature_selector_pipeline() {
        let n_samples = 150;
        let n_cols = 4;
        let mut features = Vec::with_capacity(n_samples * n_cols);
        let mut targets = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let signal = (i as f64 * 0.1).sin();
            let collinear_signal = signal + 0.001 * (i as f64 * 0.05).cos();
            let noise1 = ((i * 17) % 100) as f64 / 100.0 - 0.5;
            let noise2 = ((i * 31) % 100) as f64 / 100.0 - 0.5;

            // Col 0: Señal primaria
            features.push(signal);
            // Col 1: Colineal a Col 0 (corr ~ 0.99)
            features.push(collinear_signal);
            // Col 2: Ruido 1
            features.push(noise1);
            // Col 3: Ruido 2
            features.push(noise2);

            targets.push(signal * 3.0);
        }

        let config = FeatureSelectionConfig::default()
            .with_max_correlation_threshold(0.85)
            .with_min_mda_threshold(0.0)
            .with_n_splits(3)
            .with_horizon(2)
            .with_embargo_bars(4);

        let mut selector = CausalFeatureSelector::new(config);
        let (transformed, selected_dim) = selector
            .fit_transform(&features, &targets, n_samples, n_cols)
            .expect("Pipeline de selección debe ejecutarse correctamente");

        assert!(selector.is_fitted());
        // Se debe haber podado la variable colineal duplicada
        assert!(selected_dim < n_cols);
        assert_eq!(transformed.len(), n_samples * selected_dim);

        // Proyección individual
        let sample_row = vec![1.0, 1.0, 0.2, -0.2];
        let transformed_row = selector
            .transform_single(&sample_row)
            .expect("Transformación de fila individual");
        assert_eq!(transformed_row.len(), selected_dim);
    }
}

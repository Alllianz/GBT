/// Podador de Colinealidad y Multicolinealidad (CollinearityPruner)
///
/// Agrupa y poda características cuya correlación mutua absoluta supera el umbral
/// permitido ($|\rho| > \tau$), preservando exclusivamente el representante más informativo
/// de cada clúster ortogonal para evitar la dispersión de ganancia en árboles de decisión (GBDT).
#[derive(Debug, Clone)]
pub struct CollinearityPruner {
    pub max_correlation: f64,
}

impl CollinearityPruner {
    pub fn new(max_correlation: f64) -> Self {
        Self {
            max_correlation: max_correlation.clamp(0.05, 0.99),
        }
    }

    /// Selecciona los índices de características retenidas ortogonales.
    /// Si se provee `scores` (ej. importancias MDA o varianzas), se priorizan
    /// las variables con mayor puntuación informativa dentro de cada par colineal.
    pub fn prune_features(&self, corr_matrix: &[Vec<f64>], scores: Option<&[f64]>) -> Vec<usize> {
        let n_cols = corr_matrix.len();
        if n_cols == 0 {
            return Vec::new();
        }

        // Determinar orden de prioridad de los candidatos
        let mut candidates: Vec<usize> = (0..n_cols).collect();
        if let Some(s) = scores {
            if s.len() == n_cols {
                candidates.sort_by(|&a, &b| {
                    s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let mut selected: Vec<usize> = Vec::new();

        for &feat_idx in &candidates {
            let mut is_collinear = false;
            for &sel_idx in &selected {
                let corr = corr_matrix[feat_idx][sel_idx].abs();
                if corr > self.max_correlation {
                    is_collinear = true;
                    break;
                }
            }

            if !is_collinear {
                selected.push(feat_idx);
            }
        }

        // Ordenar los índices resultantes en orden ascendente para indexación consistente
        selected.sort_unstable();
        selected
    }

    /// Identifica los clústeres de variables colineales usando componentes conectados
    pub fn find_clusters(&self, corr_matrix: &[Vec<f64>]) -> Vec<Vec<usize>> {
        let n = corr_matrix.len();
        if n == 0 {
            return Vec::new();
        }

        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut queue = vec![i];
            visited[i] = true;

            while let Some(curr) = queue.pop() {
                cluster.push(curr);
                for j in 0..n {
                    if !visited[j] && corr_matrix[curr][j].abs() > self.max_correlation {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }

            cluster.sort_unstable();
            clusters.push(cluster);
        }

        clusters
    }

    /// Filtra y proyecta una matriz aplanada [n_rows x n_cols] conservando solo las columnas seleccionadas
    pub fn prune_matrix(
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
        selected: &[usize],
    ) -> Vec<f64> {
        let n_selected = selected.len();
        if n_rows == 0 || n_selected == 0 {
            return Vec::new();
        }

        let mut pruned = Vec::with_capacity(n_rows * n_selected);
        for r in 0..n_rows {
            let row_offset = r * n_cols;
            for &col in selected {
                pruned.push(features_flat[row_offset + col]);
            }
        }

        pruned
    }

    /// Filtra y proyecta un único vector de inferencia causal
    pub fn prune_vector(row_features: &[f64], selected: &[usize]) -> Vec<f64> {
        selected
            .iter()
            .map(|&idx| {
                if idx < row_features.len() {
                    row_features[idx]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_collinear_features_without_scores() {
        // Matriz de 3 variables: col 0 y col 1 correlacionadas a 0.95, col 2 ortogonal a 0.10
        let corr_matrix = vec![
            vec![1.00, 0.95, 0.10],
            vec![0.95, 1.00, 0.12],
            vec![0.10, 0.12, 1.00],
        ];

        let pruner = CollinearityPruner::new(0.85);
        let selected = pruner.prune_features(&corr_matrix, None);

        // Debe conservar la primera de las dos colineales (col 0) y la ortogonal (col 2)
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn test_prune_collinear_features_with_scores_preserves_best() {
        let corr_matrix = vec![
            vec![1.00, 0.95, 0.10],
            vec![0.95, 1.00, 0.12],
            vec![0.10, 0.12, 1.00],
        ];

        // La col 1 tiene mayor score causal que la col 0
        let scores = vec![0.02, 0.08, 0.05];

        let pruner = CollinearityPruner::new(0.85);
        let selected = pruner.prune_features(&corr_matrix, Some(&scores));

        // Debe seleccionar col 1 (por mayor score) y col 2
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn test_find_clusters() {
        let corr_matrix = vec![
            vec![1.00, 0.90, 0.05],
            vec![0.90, 1.00, 0.05],
            vec![0.05, 0.05, 1.00],
        ];

        let pruner = CollinearityPruner::new(0.80);
        let clusters = pruner.find_clusters(&corr_matrix);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0], vec![0, 1]);
        assert_eq!(clusters[1], vec![2]);
    }

    #[test]
    fn test_prune_matrix_slicing() {
        let n_rows = 2;
        let n_cols = 4;
        let flat = vec![
            1.0, 2.0, 3.0, 4.0, // Fila 0
            5.0, 6.0, 7.0, 8.0, // Fila 1
        ];

        let selected = vec![0, 2];
        let pruned = CollinearityPruner::prune_matrix(&flat, n_rows, n_cols, &selected);

        assert_eq!(pruned, vec![1.0, 3.0, 5.0, 7.0]);
    }
}

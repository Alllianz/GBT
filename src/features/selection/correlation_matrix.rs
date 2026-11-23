/// Calculador de Correlaciones Lineales y de Rangos (RankCorrelationCalculator)
///
/// Implementa:
/// 1. Correlación de rangos de Spearman ($\rho_S$): Robusta ante colas pesadas, distribuciones no gaussianas
///    y dependencias monótonas no lineales en series financieras mediante asignación de rangos fraccionales.
/// 2. Correlación lineal de Pearson ($r$): Mide la covarianza normalizada estándar.
/// 3. Cómputo vectorizado de matrices completas de correlación entre columnas.
#[derive(Debug, Clone, Copy, Default)]
pub struct RankCorrelationCalculator;

impl RankCorrelationCalculator {
    /// Calcula la correlación lineal de Pearson entre dos vectores
    pub fn compute_pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 || n != y.len() {
            return 0.0;
        }

        let mean_x: f64 = x.iter().sum::<f64>() / (n as f64);
        let mean_y: f64 = y.iter().sum::<f64>() / (n as f64);

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x <= 1e-15 || var_y <= 1e-15 {
            return 0.0;
        }

        let denom = (var_x * var_y).sqrt();
        if denom <= 1e-15 {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    }

    /// Asigna rangos fraccionales (average ranks) para manejar empates de forma estadísticamente exacta
    pub fn compute_fractional_ranks(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }

        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; n];
        let mut i = 0;

        while i < n {
            let mut j = i;
            while j + 1 < n && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }

            // El rango 1-based promedio para el grupo con empates desde i hasta j
            let avg_rank = ((i + 1) + (j + 1)) as f64 * 0.5;
            for k in i..=j {
                ranks[indexed[k].0] = avg_rank;
            }

            i = j + 1;
        }

        ranks
    }

    /// Calcula la correlación de rangos de Spearman entre dos vectores
    pub fn compute_spearman(x: &[f64], y: &[f64]) -> f64 {
        if x.len() < 2 || x.len() != y.len() {
            return 0.0;
        }

        let ranks_x = Self::compute_fractional_ranks(x);
        let ranks_y = Self::compute_fractional_ranks(y);

        Self::compute_pearson(&ranks_x, &ranks_y)
    }

    /// Extrae una columna j de una matriz aplanada [n_rows x n_cols]
    pub fn extract_column(features_flat: &[f64], n_rows: usize, n_cols: usize, col_idx: usize) -> Vec<f64> {
        if col_idx >= n_cols || features_flat.len() < n_rows * n_cols {
            return Vec::new();
        }

        let mut col = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            col.push(features_flat[r * n_cols + col_idx]);
        }
        col
    }

    /// Calcula la matriz completa de correlación de Spearman [n_cols x n_cols]
    pub fn compute_spearman_matrix(
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<Vec<f64>> {
        if n_cols == 0 || n_rows < 2 {
            return Vec::new();
        }

        // 1. Extraer y precomputar rangos para cada columna
        let mut ranks_per_col = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let col_vals = Self::extract_column(features_flat, n_rows, n_cols, c);
            ranks_per_col.push(Self::compute_fractional_ranks(&col_vals));
        }

        // 2. Cómputo de la matriz simétrica de Spearman
        let mut matrix = vec![vec![0.0; n_cols]; n_cols];
        for i in 0..n_cols {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n_cols {
                let corr = Self::compute_pearson(&ranks_per_col[i], &ranks_per_col[j]);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        matrix
    }

    /// Calcula la matriz completa de correlación de Pearson [n_cols x n_cols]
    pub fn compute_pearson_matrix(
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<Vec<f64>> {
        if n_cols == 0 || n_rows < 2 {
            return Vec::new();
        }

        let mut cols = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            cols.push(Self::extract_column(features_flat, n_rows, n_cols, c));
        }

        let mut matrix = vec![vec![0.0; n_cols]; n_cols];
        for i in 0..n_cols {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n_cols {
                let corr = Self::compute_pearson(&cols[i], &cols[j]);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_positive_monotonic_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let pearson = RankCorrelationCalculator::compute_pearson(&x, &y);
        let spearman = RankCorrelationCalculator::compute_spearman(&x, &y);

        assert!((pearson - 1.0).abs() < 1e-12);
        assert!((spearman - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_nonlinear_monotonic_spearman() {
        // Relación estrictamente monótona no lineal: y = x^3
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 8.0, 27.0, 64.0, 125.0];

        let spearman = RankCorrelationCalculator::compute_spearman(&x, &y);
        assert!((spearman - 1.0).abs() < 1e-12, "Spearman debe ser exactamente 1.0 para relación monótona");

        let pearson = RankCorrelationCalculator::compute_pearson(&x, &y);
        assert!(pearson < 0.99, "Pearson es subóptimo ante no linealidades");
    }

    #[test]
    fn test_fractional_ranks_handling_ties() {
        let values = vec![10.0, 20.0, 20.0, 30.0];
        let ranks = RankCorrelationCalculator::compute_fractional_ranks(&values);
        // Índices ordenados: 10.0 (rango 1), 20.0 y 20.0 (rangos 2 y 3 -> promedio 2.5), 30.0 (rango 4)
        assert_eq!(ranks[0], 1.0);
        assert_eq!(ranks[1], 2.5);
        assert_eq!(ranks[2], 2.5);
        assert_eq!(ranks[3], 4.0);
    }

    #[test]
    fn test_spearman_matrix_symmetry_and_diagonal() {
        let n_rows = 10;
        let n_cols = 3;
        let mut flat = Vec::with_capacity(n_rows * n_cols);
        for r in 0..n_rows {
            flat.push(r as f64);
            flat.push((r as f64) * 2.0 + 1.0); // Col 1 perfectamente correlacionada con col 0
            flat.push(-(r as f64));            // Col 2 perfectamente invertida
        }

        let matrix = RankCorrelationCalculator::compute_spearman_matrix(&flat, n_rows, n_cols);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0][0], 1.0);
        assert_eq!(matrix[1][1], 1.0);
        assert_eq!(matrix[2][2], 1.0);

        assert!((matrix[0][1] - 1.0).abs() < 1e-12);
        assert!((matrix[1][0] - 1.0).abs() < 1e-12);
        assert!((matrix[0][2] - (-1.0)).abs() < 1e-12);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RankCorrelationCalculator;

impl RankCorrelationCalculator {
    pub fn compute_pearson(x: &[f64], y: &[f64]) -> f64 {
        if x.len() < 2 || x.len() != y.len() {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut valid_n = 0usize;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if xi.is_finite() && yi.is_finite() {
                sum_x += xi;
                sum_y += yi;
                valid_n += 1;
            }
        }

        if valid_n < 2 {
            return 0.0;
        }

        let n_f64 = valid_n as f64;
        let mean_x = sum_x / n_f64;
        let mean_y = sum_y / n_f64;
        if !mean_x.is_finite() || !mean_y.is_finite() {
            return 0.0;
        }

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if xi.is_finite() && yi.is_finite() {
                let dx = xi - mean_x;
                let dy = yi - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
        }

        if !var_x.is_finite()
            || !var_y.is_finite()
            || !cov.is_finite()
            || var_x <= 1e-15
            || var_y <= 1e-15
        {
            return 0.0;
        }

        let denom = (var_x * var_y).sqrt();
        if !denom.is_finite() || denom <= 1e-15 {
            0.0
        } else {
            let r = cov / denom;
            if r.is_finite() {
                r.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        }
    }

    pub fn compute_fractional_ranks(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }

        let mut indexed: Vec<(usize, f64)> = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .collect();
        let mut ranks = vec![0.0; n];
        if indexed.is_empty() {
            return ranks;
        }

        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let m = indexed.len();
        let mut i = 0;

        while i < m {
            let mut j = i;
            while j + 1 < m && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }

            let avg_rank = ((i + 1) + (j + 1)) as f64 * 0.5;
            for k in i..=j {
                ranks[indexed[k].0] = avg_rank;
            }

            i = j + 1;
        }

        ranks
    }

    pub fn compute_spearman(x: &[f64], y: &[f64]) -> f64 {
        if x.len() < 2 || x.len() != y.len() {
            return 0.0;
        }

        let mut clean_x = Vec::with_capacity(x.len());
        let mut clean_y = Vec::with_capacity(y.len());
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if xi.is_finite() && yi.is_finite() {
                clean_x.push(xi);
                clean_y.push(yi);
            }
        }

        if clean_x.len() < 2 {
            return 0.0;
        }

        let ranks_x = Self::compute_fractional_ranks(&clean_x);
        let ranks_y = Self::compute_fractional_ranks(&clean_y);

        Self::compute_pearson(&ranks_x, &ranks_y)
    }

    pub fn extract_column(
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
        col_idx: usize,
    ) -> Vec<f64> {
        if col_idx >= n_cols || features_flat.len() < n_rows * n_cols {
            return Vec::new();
        }

        let mut col = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            col.push(features_flat[r * n_cols + col_idx]);
        }
        col
    }

    pub fn compute_spearman_matrix(
        features_flat: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<Vec<f64>> {
        if n_cols == 0 || n_rows < 2 {
            return Vec::new();
        }

        let mut cols = Vec::with_capacity(n_cols);
        let mut all_finite = Vec::with_capacity(n_cols);
        let mut ranks_per_col = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let col_vals = Self::extract_column(features_flat, n_rows, n_cols, c);
            all_finite.push(col_vals.iter().all(|v| v.is_finite()));
            ranks_per_col.push(Self::compute_fractional_ranks(&col_vals));
            cols.push(col_vals);
        }

        let mut matrix = vec![vec![0.0; n_cols]; n_cols];
        for i in 0..n_cols {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n_cols {
                let corr = if all_finite[i] && all_finite[j] {
                    Self::compute_pearson(&ranks_per_col[i], &ranks_per_col[j])
                } else {
                    Self::compute_spearman(&cols[i], &cols[j])
                };
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }

        matrix
    }

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
            flat.push((r as f64) * 2.0 + 1.0);
            flat.push(-(r as f64));
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

    #[test]
    fn test_non_finite_values_filtered_without_nan_output() {
        let x = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, f64::INFINITY];

        let p = RankCorrelationCalculator::compute_pearson(&x, &y);
        assert!(p.is_finite());
        assert!((p - 1.0).abs() < 1e-12);

        let ranks = RankCorrelationCalculator::compute_fractional_ranks(&x);
        assert!(ranks.iter().all(|v| v.is_finite()));
        assert_eq!(ranks[2], 0.0);

        let flat = vec![
            1.0, 2.0,
            2.0, 4.0,
            f64::NAN, 6.0,
            4.0, 8.0,
        ];
        let mat = RankCorrelationCalculator::compute_spearman_matrix(&flat, 4, 2);
        assert!(mat[0][1].is_finite());
        assert!((mat[0][1] - 1.0).abs() < 1e-12);
    }
}

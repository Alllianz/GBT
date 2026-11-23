/// Submuestreador Estocástico de Columnas (Stochastic Column Subsampler)
///
/// Implementa la selección estocástica de características a dos niveles jerárquicos:
/// 1. Nivel Árbol (`colsample_bytree`): Se selecciona un subconjunto de features antes de construir el árbol.
/// 2. Nivel Nodo (`colsample_bynode`): Se selecciona un subconjunto adicional en cada split individual.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColSubsampler;

impl ColSubsampler {
    #[inline]
    fn next_u64(state: &mut u64) -> u64 {
        if *state == 0 {
            *state = 0x2545f4914f6cdd1d;
        }
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Submuestreo de características a nivel de árbol (`colsample_bytree`)
    ///
    /// Retorna una lista ordenada con los índices de variables autorizadas para el árbol completo.
    pub fn sample_by_tree(total_features: usize, ratio: f64, seed: u64) -> Vec<usize> {
        if total_features == 0 {
            return Vec::new();
        }

        let ratio = ratio.clamp(0.01, 1.0);
        if (ratio - 1.0).abs() < 1e-6 {
            return (0..total_features).collect();
        }

        let target_count = ((total_features as f64 * ratio).ceil() as usize)
            .max(1)
            .min(total_features);

        let mut pool: Vec<usize> = (0..total_features).collect();
        let mut rng = if seed == 0 { 0xa0761d6478bd642f } else { seed };

        for i in 0..target_count {
            let rand_val = Self::next_u64(&mut rng);
            let j = i + ((rand_val as usize) % (total_features - i));
            pool.swap(i, j);
        }

        let mut selected = pool[0..target_count].to_vec();
        selected.sort_unstable();
        selected
    }

    /// Submuestreo de características a nivel de nodo (`colsample_bynode`)
    ///
    /// Dado un conjunto de variables activas, selecciona una fracción para evaluar en la división actual.
    pub fn sample_by_node(available_features: &[usize], ratio: f64, seed: u64) -> Vec<usize> {
        let n = available_features.len();
        if n == 0 {
            return Vec::new();
        }

        let ratio = ratio.clamp(0.01, 1.0);
        if (ratio - 1.0).abs() < 1e-6 {
            return available_features.to_vec();
        }

        let target_count = ((n as f64 * ratio).ceil() as usize).max(1).min(n);

        let mut pool = available_features.to_vec();
        let mut rng = if seed == 0 { 0xe7037ed1a0b428db } else { seed };

        for i in 0..target_count {
            let rand_val = Self::next_u64(&mut rng);
            let j = i + ((rand_val as usize) % (n - i));
            pool.swap(i, j);
        }

        let mut selected = pool[0..target_count].to_vec();
        selected.sort_unstable();
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colsample_by_tree_ratio() {
        let total = 20;
        let selected = ColSubsampler::sample_by_tree(total, 0.50, 42);
        assert_eq!(selected.len(), 10);
        for &feat in &selected {
            assert!(feat < total);
        }
    }

    #[test]
    fn test_colsample_by_node_subset() {
        let available = vec![2, 5, 8, 12, 15, 19];
        let selected = ColSubsampler::sample_by_node(&available, 0.50, 99);
        assert_eq!(selected.len(), 3);
        for &f in &selected {
            assert!(available.contains(&f));
        }
    }
}

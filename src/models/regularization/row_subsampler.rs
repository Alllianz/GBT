/// Submuestreador Estocástico de Filas (Stochastic Row Subsampler)
///
/// Implementa submuestreo de instancias sin reemplazo y submuestreo por bloques temporales
/// continuos, garantizando reproducibilidad determinista mediante un generador PRNG Xorshift64
/// de alto desempeño sin dependencias externas.
///
/// REGLA FINANCIERA: Los índices resultantes siempre se devuelven en orden cronológico estricto
/// (t_0 < t_1 < ... < t_k) para preservar la causalidad temporal (t <= T).
#[derive(Debug, Clone, Copy, Default)]
pub struct RowSubsampler;

impl RowSubsampler {
    /// PRNG determinista Xorshift64
    #[inline]
    fn next_u64(state: &mut u64) -> u64 {
        if *state == 0 {
            *state = 0x853c49e6748fea9b;
        }
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Submuestreo uniforme sin reemplazo de filas
    ///
    /// Retorna un vector con los índices seleccionados ordenados cronológicamente.
    pub fn sample_uniform(n_samples: usize, subsample_ratio: f64, seed: u64) -> Vec<usize> {
        if n_samples == 0 {
            return Vec::new();
        }

        let ratio = subsample_ratio.clamp(0.01, 1.0);
        if (ratio - 1.0).abs() < 1e-6 {
            return (0..n_samples).collect();
        }

        let target_count = ((n_samples as f64 * ratio).round() as usize).max(1).min(n_samples);
        let mut pool: Vec<usize> = (0..n_samples).collect();
        let mut rng = if seed == 0 { 0x517cc1b727220a95 } else { seed };

        // Permutación parcial de Fisher-Yates sobre los primeros target_count elementos
        for i in 0..target_count {
            let rand_val = Self::next_u64(&mut rng);
            let j = i + ((rand_val as usize) % (n_samples - i));
            pool.swap(i, j);
        }

        let mut selected = pool[0..target_count].to_vec();
        // Ordenamiento temporal estricto para preservar la causalidad
        selected.sort_unstable();
        selected
    }

    /// Submuestreo por bloques temporales contiguos (Block Subsampling)
    ///
    /// Preserva la estructura de memoria y autocorrelación temporal serial en series financieras.
    pub fn sample_temporal_blocks(
        n_samples: usize,
        subsample_ratio: f64,
        block_size: usize,
        seed: u64,
    ) -> Vec<usize> {
        if n_samples == 0 {
            return Vec::new();
        }

        let ratio = subsample_ratio.clamp(0.01, 1.0);
        if (ratio - 1.0).abs() < 1e-6 || block_size >= n_samples {
            return (0..n_samples).collect();
        }

        let effective_block = block_size.max(1).min(n_samples);
        let num_blocks = n_samples / effective_block;
        if num_blocks <= 1 {
            return (0..n_samples).collect();
        }

        let target_blocks = ((num_blocks as f64 * ratio).round() as usize).max(1).min(num_blocks);
        let mut block_indices: Vec<usize> = (0..num_blocks).collect();
        let mut rng = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };

        // Permutación de Fisher-Yates de bloques
        for i in 0..target_blocks {
            let rand_val = Self::next_u64(&mut rng);
            let j = i + ((rand_val as usize) % (num_blocks - i));
            block_indices.swap(i, j);
        }

        let mut selected_blocks = block_indices[0..target_blocks].to_vec();
        selected_blocks.sort_unstable();

        let mut result = Vec::with_capacity(target_blocks * effective_block);
        for b in selected_blocks {
            let start = b * effective_block;
            let end = (start + effective_block).min(n_samples);
            for idx in start..end {
                result.push(idx);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_uniform_size_and_monotonicity() {
        let n = 100;
        let ratio = 0.70;
        let selected = RowSubsampler::sample_uniform(n, ratio, 42);

        assert_eq!(selected.len(), 70);
        // Verificar que los índices son estrictamente crecientes (t_0 < t_1 < ...)
        for i in 1..selected.len() {
            assert!(selected[i] > selected[i - 1]);
        }
    }

    #[test]
    fn test_sample_blocks_preserves_blocks() {
        let n = 100;
        let block_size = 10;
        let selected = RowSubsampler::sample_temporal_blocks(n, 0.50, block_size, 123);

        assert_eq!(selected.len(), 50);
        for i in 1..selected.len() {
            assert!(selected[i] > selected[i - 1]);
        }
    }
}

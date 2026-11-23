//! Muestreo temporal por bloques continuos (Block Bootstrapping) para Random Forest en Series de Tiempo Financieras
//!
//! En el modelado financiero con etiquetas de eventos solapados (horizonte H >= 1),
//! el muestreo bootstrap IID tradicional con reemplazo descompone la dependencia serial,
//! genera muestras artificialmente idénticas y produce sobreajuste severo en In-Sample.
//!
//! TemporalBlockBootstrap implementa Moving Block Bootstrapping:
//! 1. Selecciona bloques continuos contiguos de longitud L >= H.
//! 2. Cada bloque preserva intacta la estructura de autocorrelación causal interna (t <= T).
//! 3. Genera una muestra de tamaño N ensamblada a partir de bloques seleccionados aleatoriamente con reemplazo.

pub struct TemporalBlockBootstrap;

impl TemporalBlockBootstrap {
    /// Genera un vector de índices para entrenamiento de un árbol mediante Block Bootstrapping temporal.
    ///
    /// - `total_rows`: Cantidad total de filas causales disponibles (valid_rows).
    /// - `block_size`: Tamaño del bloque temporal continuo L (recomendado >= horizon).
    /// - `sample_size`: Cantidad total de índices requeridos (típicamente `total_rows`).
    /// - `rng`: Generador pseudoaleatorio determinista de la semilla del árbol.
    pub fn sample_indices<F>(
        total_rows: usize,
        block_size: usize,
        sample_size: usize,
        mut rng: F,
    ) -> Vec<usize>
    where
        F: FnMut() -> u64,
    {
        if total_rows == 0 || sample_size == 0 {
            return Vec::new();
        }

        // El tamaño del bloque L se acota en [1, total_rows]
        let l = block_size.max(1).min(total_rows);
        let max_start = total_rows.saturating_sub(l);
        let mut indices = Vec::with_capacity(sample_size);

        while indices.len() < sample_size {
            let start_idx = if max_start > 0 {
                (rng() as usize) % (max_start + 1)
            } else {
                0
            };

            let remaining = sample_size - indices.len();
            let count_to_add = l.min(remaining);
            for offset in 0..count_to_add {
                indices.push(start_idx + offset);
            }
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_indices_length_and_bounds() {
        let mut seed = 123456789u64;
        let rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        let total_rows = 100;
        let block_size = 10;
        let sample_size = 100;

        let indices = TemporalBlockBootstrap::sample_indices(total_rows, block_size, sample_size, rng);
        assert_eq!(indices.len(), sample_size);
        for &idx in &indices {
            assert!(idx < total_rows, "Índice {} fuera de rango total {}", idx, total_rows);
        }
    }

    #[test]
    fn test_contiguous_blocks_preserved() {
        let mut seed = 987654321u64;
        let rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        let total_rows = 200;
        let block_size = 8;
        let sample_size = 80;

        let indices = TemporalBlockBootstrap::sample_indices(total_rows, block_size, sample_size, rng);
        assert_eq!(indices.len(), sample_size);

        // Cada bloque de 8 elementos debe ser estrictamente consecutivo: idx[k+1] == idx[k] + 1
        for block_start in (0..sample_size).step_by(block_size) {
            let block_end = (block_start + block_size).min(sample_size);
            for i in block_start..(block_end - 1) {
                assert_eq!(
                    indices[i + 1],
                    indices[i] + 1,
                    "Continuidad temporal violada dentro del bloque en posición {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_edge_case_small_rows() {
        let mut seed = 42u64;
        let rng = move || {
            seed = seed.wrapping_add(1);
            seed
        };

        // Si total_rows <= block_size
        let indices = TemporalBlockBootstrap::sample_indices(5, 10, 15, rng);
        assert_eq!(indices.len(), 15);
        for &idx in &indices {
            assert!(idx < 5);
        }

        // Si total_rows == 0
        let empty = TemporalBlockBootstrap::sample_indices(0, 5, 10, rng);
        assert!(empty.is_empty());
    }
}

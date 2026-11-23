/// Generador de Números Pseudoaleatorios Determinista Xorshift64
///
/// Diseñado para permutaciones reproducibles en la evaluación Purged MDA sin dependencias externas
/// ni colisiones de hilos.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c49e6748fea9b
            } else {
                seed
            },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Permutación determinista in-place de Fisher-Yates
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n <= 1 {
            return;
        }

        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_reproducibility() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_shuffle_permutes_elements() {
        let mut rng = Xorshift64::new(12345);
        let mut arr: Vec<usize> = (0..50).collect();
        let original = arr.clone();

        rng.shuffle(&mut arr);
        assert_eq!(arr.len(), 50);
        // Debe contener los mismos elementos
        let mut sorted = arr.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
        // Y el orden debe haber cambiado
        assert_ne!(arr, original);
    }
}

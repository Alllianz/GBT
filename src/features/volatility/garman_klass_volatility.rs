use crate::types::Kline;

/// Estimador Institucional de Volatilidad de Garman-Klass (1980)
///
/// Incorpora información completa de OHLC (Open, High, Low, Close) para estimar la varianza
/// de un movimiento browniano geométrico con deriva cero. Ofrece una eficiencia teórica ~7.4 veces
/// superior a la volatilidad estándar Close-to-Close.
///
/// Fórmula matemática causal (t <= T):
/// $$\sigma_{GK, t} = \sqrt{\frac{1}{N} \sum_{i=t-N+1}^t \left[ \frac{1}{2}\left(\ln \frac{H_i}{L_i}\right)^2 - (2 \ln 2 - 1)\left(\ln \frac{C_i}{O_i}\right)^2 \right]}$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GarmanKlassVolatility {
    pub window: usize,
}

impl GarmanKlassVolatility {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
        }
    }

    /// Calcula la serie temporal completa de volatilidad Garman-Klass de forma causal
    pub fn compute(&self, klines: &[Kline]) -> Vec<f64> {
        Self::calculate(klines, self.window)
    }

    /// Calcula la volatilidad Garman-Klass en el instante específico `idx` (t <= T)
    pub fn compute_at(&self, klines: &[Kline], idx: usize) -> f64 {
        Self::calculate_single(klines, idx, self.window)
    }

    /// Implementación estática causal de Garman-Klass
    pub fn calculate(klines: &[Kline], window: usize) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || window == 0 {
            return Vec::new();
        }

        let mut out = vec![0.0; n];
        if n < window {
            return out;
        }

        let inv_n = 1.0 / (window as f64);
        let coeff_co = 2.0 * 2.0_f64.ln() - 1.0;

        // Precomputar términos para cada vela i de forma causal
        let terms: Vec<f64> = klines
            .iter()
            .map(|k| {
                if k.high > 0.0 && k.low > 0.0 && k.open > 0.0 && k.close > 0.0 && k.high >= k.low {
                    let hl = (k.high / k.low).ln();
                    let co = (k.close / k.open).ln();
                    0.5 * hl * hl - coeff_co * co * co
                } else {
                    0.0
                }
            })
            .collect();

        // Ventana móvil causal sin look-ahead bias
        for t in (window - 1)..n {
            let start = t + 1 - window;
            let sum_terms: f64 = terms[start..=t].iter().sum();
            let variance = (sum_terms * inv_n).max(0.0);
            out[t] = variance.sqrt();
        }

        out
    }

    /// Calcula la volatilidad para una sola vela en `idx` de forma causal (t <= T)
    pub fn calculate_single(klines: &[Kline], idx: usize, window: usize) -> f64 {
        if window == 0 || idx + 1 < window || idx >= klines.len() {
            return 0.0;
        }

        let start = idx + 1 - window;
        let inv_n = 1.0 / (window as f64);
        let coeff_co = 2.0 * 2.0_f64.ln() - 1.0;

        let mut sum_terms = 0.0;
        for i in start..=idx {
            let k = &klines[i];
            if k.high > 0.0 && k.low > 0.0 && k.open > 0.0 && k.close > 0.0 && k.high >= k.low {
                let hl = (k.high / k.low).ln();
                let co = (k.close / k.open).ln();
                sum_terms += 0.5 * hl * hl - coeff_co * co * co;
            }
        }

        (sum_terms * inv_n).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64) * 0.5;
                Kline {
                    timestamp: (i as i64) * 60,
                    open: base,
                    high: base * 1.025,
                    low: base * 0.975,
                    close: base * 1.01,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_garman_klass_matches_batch_and_single() {
        let klines = create_test_klines(50);
        let window = 14;
        let series = GarmanKlassVolatility::calculate(&klines, window);

        assert_eq!(series.len(), 50);
        for i in 0..(window - 1) {
            assert_eq!(series[i], 0.0);
        }

        for i in (window - 1)..50 {
            let single = GarmanKlassVolatility::calculate_single(&klines, i, window);
            assert!(
                (series[i] - single).abs() < 1e-12,
                "Discrepancia en index {}: series={}, single={}",
                i,
                series[i],
                single
            );
            assert!(series[i] > 0.0, "Volatilidad debe ser estrictamente positiva");
        }
    }

    #[test]
    fn test_garman_klass_zero_range_produces_zero() {
        let flat_klines: Vec<Kline> = (0..20)
            .map(|i| Kline {
                timestamp: (i as i64) * 60,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 500.0,
            })
            .collect();

        let series = GarmanKlassVolatility::calculate(&flat_klines, 10);
        for val in series {
            assert_eq!(val, 0.0);
        }
    }
}

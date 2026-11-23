use crate::types::Kline;

/// Estimador Institucional de Volatilidad de Parkinson (1980)
///
/// Modela la volatilidad basándose en el rango High/Low intradiario, logrando una eficiencia
/// estadística aproximadamente 5 veces superior al estimador Close-to-Close clásico al eliminar
/// el ruido de muestreo de precios puntuales de cierre.
///
/// Fórmula matemática causal (t <= T):
/// $$\sigma_{P, t} = \sqrt{\frac{1}{4 N \ln 2} \sum_{i=t-N+1}^t \left(\ln \frac{H_i}{L_i}\right)^2}$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParkinsonVolatility {
    pub window: usize,
}

impl ParkinsonVolatility {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
        }
    }

    /// Calcula la serie temporal completa de volatilidad de Parkinson de forma causal
    pub fn compute(&self, klines: &[Kline]) -> Vec<f64> {
        Self::calculate(klines, self.window)
    }

    /// Calcula la volatilidad de Parkinson en el instante específico `idx` (t <= T)
    pub fn compute_at(&self, klines: &[Kline], idx: usize) -> f64 {
        Self::calculate_single(klines, idx, self.window)
    }

    /// Implementación estática causal de Parkinson
    pub fn calculate(klines: &[Kline], window: usize) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || window == 0 {
            return Vec::new();
        }

        let mut out = vec![0.0; n];
        if n < window {
            return out;
        }

        let inv_scale = 1.0 / (4.0 * (window as f64) * 2.0_f64.ln());

        // Precomputar términos (ln(High / Low))^2 para cada vela de forma causal
        let log_hl_sq: Vec<f64> = klines
            .iter()
            .map(|k| {
                if k.high > 0.0 && k.low > 0.0 && k.high >= k.low {
                    let hl = (k.high / k.low).ln();
                    hl * hl
                } else {
                    0.0
                }
            })
            .collect();

        // Ventana móvil causal sin look-ahead bias
        for t in (window - 1)..n {
            let start = t + 1 - window;
            let sum_sq: f64 = log_hl_sq[start..=t].iter().sum();
            let variance = sum_sq * inv_scale;
            out[t] = variance.max(0.0).sqrt();
        }

        out
    }

    /// Calcula la volatilidad para una sola vela en `idx` de forma causal (t <= T)
    pub fn calculate_single(klines: &[Kline], idx: usize, window: usize) -> f64 {
        if window == 0 || idx + 1 < window || idx >= klines.len() {
            return 0.0;
        }

        let start = idx + 1 - window;
        let inv_scale = 1.0 / (4.0 * (window as f64) * 2.0_f64.ln());

        let mut sum_sq = 0.0;
        for i in start..=idx {
            let k = &klines[i];
            if k.high > 0.0 && k.low > 0.0 && k.high >= k.low {
                let hl = (k.high / k.low).ln();
                sum_sq += hl * hl;
            }
        }

        (sum_sq * inv_scale).max(0.0).sqrt()
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
                    high: base * 1.02,
                    low: base * 0.98,
                    close: base * 1.01,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_parkinson_volatility_matches_batch_and_single() {
        let klines = create_test_klines(50);
        let window = 14;
        let series = ParkinsonVolatility::calculate(&klines, window);

        assert_eq!(series.len(), 50);
        for i in 0..(window - 1) {
            assert_eq!(series[i], 0.0);
        }

        for i in (window - 1)..50 {
            let single = ParkinsonVolatility::calculate_single(&klines, i, window);
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
    fn test_parkinson_zero_range_produces_zero_volatility() {
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

        let series = ParkinsonVolatility::calculate(&flat_klines, 10);
        for val in series {
            assert_eq!(val, 0.0);
        }
    }
}

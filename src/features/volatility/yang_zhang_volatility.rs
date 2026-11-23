use crate::types::Kline;

/// Estimador Institucional de Volatilidad de Yang-Zhang (2000)
///
/// Estimador de mínima varianza invariante ante derivas de precios (drift-independent)
/// y saltos inter-barra / overnight. Combina la varianza de apertura-cierre, la varianza
/// overnight y el estimador de Rogers-Satchell de manera estadísticamente óptima:
///
/// $$\sigma_{YZ, t}^2 = \sigma_{\text{overnight}, t}^2 + k \cdot \sigma_{\text{open-close}, t}^2 + (1 - k) \cdot \sigma_{RS, t}^2$$
///
/// donde:
/// $$k = \frac{0.34}{1.34 + \frac{N+1}{N-1}}$$
/// $$\sigma_{RS, t}^2 = \frac{1}{N} \sum_{i=t-N+1}^t \left[ \ln\frac{H_i}{O_i} \ln\frac{H_i}{C_i} + \ln\frac{L_i}{O_i} \ln\frac{L_i}{C_i} \right]$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YangZhangVolatility {
    pub window: usize,
}

impl YangZhangVolatility {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Calcula la serie temporal completa de volatilidad Yang-Zhang de forma causal (t <= T)
    pub fn compute(&self, klines: &[Kline]) -> Vec<f64> {
        Self::calculate(klines, self.window)
    }

    /// Calcula la volatilidad Yang-Zhang en el instante específico `idx` (t <= T)
    pub fn compute_at(&self, klines: &[Kline], idx: usize) -> f64 {
        Self::calculate_single(klines, idx, self.window)
    }

    /// Implementación estática causal de Yang-Zhang
    pub fn calculate(klines: &[Kline], window: usize) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || window < 2 {
            return Vec::new();
        }

        let mut out = vec![0.0; n];
        // Se requieren al menos `window` velas previas más una vela base anterior para C_{i-1}
        if n <= window {
            return out;
        }

        let n_f64 = window as f64;
        let k = 0.34 / (1.34 + (n_f64 + 1.0) / (n_f64 - 1.0));
        let inv_n_minus_1 = 1.0 / (n_f64 - 1.0);
        let inv_n = 1.0 / n_f64;

        // Precomputar retornos inter-barra (overnight u_i), intra-barra (open-close d_i) y Rogers-Satchell rs_i
        let mut u = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut rs = vec![0.0; n];

        for i in 1..n {
            let prev_c = klines[i - 1].close;
            let o = klines[i].open;
            let h = klines[i].high;
            let l = klines[i].low;
            let c = klines[i].close;

            if prev_c > 0.0 && o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && h >= l {
                u[i] = (o / prev_c).ln();
                d[i] = (c / o).ln();

                let ho = (h / o).ln();
                let hc = (h / c).ln();
                let lo = (l / o).ln();
                let lc = (l / c).ln();

                let rs_val = ho * hc + lo * lc;
                rs[i] = rs_val.max(0.0);
            }
        }

        // Ventana móvil causal [t - window + 1 ..= t] donde t >= window
        for t in window..n {
            let start = t + 1 - window;

            // Media de overnight u_i
            let u_slice = &u[start..=t];
            let mean_u: f64 = u_slice.iter().sum::<f64>() * inv_n;
            let var_overnight: f64 = u_slice
                .iter()
                .map(|&val| {
                    let diff = val - mean_u;
                    diff * diff
                })
                .sum::<f64>()
                * inv_n_minus_1;

            // Media de open-to-close d_i
            let d_slice = &d[start..=t];
            let mean_d: f64 = d_slice.iter().sum::<f64>() * inv_n;
            let var_open_close: f64 = d_slice
                .iter()
                .map(|&val| {
                    let diff = val - mean_d;
                    diff * diff
                })
                .sum::<f64>()
                * inv_n_minus_1;

            // Rogers-Satchell
            let rs_slice = &rs[start..=t];
            let var_rs: f64 = rs_slice.iter().sum::<f64>() * inv_n;

            let yz_var = var_overnight + k * var_open_close + (1.0 - k) * var_rs;
            out[t] = yz_var.max(0.0).sqrt();
        }

        out
    }

    /// Calcula la volatilidad para una sola vela en `idx` de forma causal (t <= T)
    pub fn calculate_single(klines: &[Kline], idx: usize, window: usize) -> f64 {
        if window < 2 || idx < window || idx >= klines.len() {
            return 0.0;
        }

        let start = idx + 1 - window;
        let n_f64 = window as f64;
        let k = 0.34 / (1.34 + (n_f64 + 1.0) / (n_f64 - 1.0));
        let inv_n_minus_1 = 1.0 / (n_f64 - 1.0);
        let inv_n = 1.0 / n_f64;

        let mut sum_u = 0.0;
        let mut sum_d = 0.0;
        let mut sum_rs = 0.0;

        let mut u_vals = Vec::with_capacity(window);
        let mut d_vals = Vec::with_capacity(window);

        for i in start..=idx {
            let prev_c = klines[i - 1].close;
            let o = klines[i].open;
            let h = klines[i].high;
            let l = klines[i].low;
            let c = klines[i].close;

            if prev_c > 0.0 && o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && h >= l {
                let u_val = (o / prev_c).ln();
                let d_val = (c / o).ln();

                let ho = (h / o).ln();
                let hc = (h / c).ln();
                let lo = (l / o).ln();
                let lc = (l / c).ln();

                let rs_val = (ho * hc + lo * lc).max(0.0);

                sum_u += u_val;
                sum_d += d_val;
                sum_rs += rs_val;

                u_vals.push(u_val);
                d_vals.push(d_val);
            } else {
                u_vals.push(0.0);
                d_vals.push(0.0);
            }
        }

        let mean_u = sum_u * inv_n;
        let var_overnight = u_vals
            .iter()
            .map(|&val| {
                let diff = val - mean_u;
                diff * diff
            })
            .sum::<f64>()
            * inv_n_minus_1;

        let mean_d = sum_d * inv_n;
        let var_open_close = d_vals
            .iter()
            .map(|&val| {
                let diff = val - mean_d;
                diff * diff
            })
            .sum::<f64>()
            * inv_n_minus_1;

        let var_rs = sum_rs * inv_n;

        let yz_var = var_overnight + k * var_open_close + (1.0 - k) * var_rs;
        yz_var.max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64) * 0.5;
                // Introducir pequeño salto overnight de apertura
                let jump = if i % 2 == 0 { 0.2 } else { -0.2 };
                Kline {
                    timestamp: (i as i64) * 60,
                    open: base + jump,
                    high: (base + 1.5).max(base + jump + 0.5),
                    low: (base - 1.5).min(base + jump - 0.5),
                    close: base + 0.3,
                    volume: 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_yang_zhang_matches_batch_and_single() {
        let klines = create_test_klines(50);
        let window = 14;
        let series = YangZhangVolatility::calculate(&klines, window);

        assert_eq!(series.len(), 50);
        for i in 0..window {
            assert_eq!(series[i], 0.0);
        }

        for i in window..50 {
            let single = YangZhangVolatility::calculate_single(&klines, i, window);
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
    fn test_yang_zhang_zero_volatility() {
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

        let series = YangZhangVolatility::calculate(&flat_klines, 10);
        for val in series {
            assert_eq!(val, 0.0);
        }
    }
}

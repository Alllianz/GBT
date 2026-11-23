use crate::types::Kline;

/// Métodos causales para cálculo de indicadores técnicos sin mirar al futuro (t <= T).
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Media Móvil Exponencial Causal (EMA)
    pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 || period == 0 {
            return Vec::new();
        }

        let mut out = vec![0.0; n];
        if n < period {
            return out;
        }

        let k = 2.0 / (period as f64 + 1.0);
        let sma: f64 = values[0..period].iter().sum::<f64>() / (period as f64);
        out[period - 1] = sma;

        let mut current_ema = sma;
        for i in period..n {
            current_ema = values[i] * k + current_ema * (1.0 - k);
            out[i] = current_ema;
        }

        out
    }

    /// Relative Strength Index Causal (RSI)
    pub fn rsi(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        if n <= period || period == 0 {
            return vec![50.0; n];
        }

        let mut out = vec![50.0; n];
        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            let diff = values[i] - values[i - 1];
            if diff > 0.0 {
                gains.push(diff);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(diff.abs());
            }
        }

        let mut avg_gain = gains[0..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[0..period].iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 && avg_gain == 0.0 {
            out[period] = 50.0;
        } else if avg_loss == 0.0 {
            out[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            out[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        for i in period..(n - 1) {
            avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;

            if avg_loss == 0.0 && avg_gain == 0.0 {
                out[i + 1] = 50.0;
            } else if avg_loss == 0.0 {
                out[i + 1] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                out[i + 1] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        out
    }

    /// Average True Range Causal (ATR)
    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || period == 0 {
            return Vec::new();
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(klines[0].high - klines[0].low);

        for i in 1..n {
            let h = klines[i].high;
            let l = klines[i].low;
            let prev_c = klines[i - 1].close;

            let tr_val = (h - l).max((h - prev_c).abs()).max((l - prev_c).abs());
            tr.push(tr_val);
        }

        Self::ema(&tr, period)
    }

    /// Rate of Change (ROC / Momentum porcentual)
    pub fn roc(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut out = vec![0.0; n];
        for i in period..n {
            // BUG-36: Validar que tanto values[i] como prev sean finitos y prev > 0.0 para prevenir propagación de NaN/Inf
            let prev = values[i - period];
            let curr = values[i];
            if prev > 0.0 && prev.is_finite() && curr.is_finite() {
                out[i] = (curr - prev) / prev;
            }
        }
        out
    }
}

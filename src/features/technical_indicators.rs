use crate::types::Kline;

pub struct TechnicalIndicators;

impl TechnicalIndicators {
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
        let mut sum = 0.0;
        let mut count = 0usize;
        for &v in &values[0..period] {
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        let sma = if count > 0 { sum / (count as f64) } else { 0.0 };
        let mut current_ema = if sma.is_finite() { sma } else { 0.0 };
        out[period - 1] = current_ema;

        for i in period..n {
            if values[i].is_finite() {
                let next_ema = values[i] * k + current_ema * (1.0 - k);
                if next_ema.is_finite() {
                    current_ema = next_ema;
                }
            }
            out[i] = current_ema;
        }

        out
    }

    pub fn rsi(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        if n <= period || period == 0 {
            return vec![50.0; n];
        }

        let mut out = vec![50.0; n];
        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            if values[i].is_finite() && values[i - 1].is_finite() {
                let diff = values[i] - values[i - 1];
                if diff.is_finite() {
                    if diff > 0.0 {
                        gains.push(diff);
                        losses.push(0.0);
                    } else {
                        gains.push(0.0);
                        losses.push(diff.abs());
                    }
                } else {
                    gains.push(0.0);
                    losses.push(0.0);
                }
            } else {
                gains.push(0.0);
                losses.push(0.0);
            }
        }

        let mut avg_gain = gains[0..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[0..period].iter().sum::<f64>() / period as f64;
        if !avg_gain.is_finite() {
            avg_gain = 0.0;
        }
        if !avg_loss.is_finite() {
            avg_loss = 0.0;
        }

        if avg_loss == 0.0 && avg_gain == 0.0 {
            out[period] = 50.0;
        } else if avg_loss == 0.0 {
            out[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            let val = 100.0 - (100.0 / (1.0 + rs));
            out[period] = if val.is_finite() { val } else { 50.0 };
        }

        for i in period..(n - 1) {
            let next_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            let next_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
            if next_gain.is_finite() {
                avg_gain = next_gain;
            }
            if next_loss.is_finite() {
                avg_loss = next_loss;
            }

            if avg_loss == 0.0 && avg_gain == 0.0 {
                out[i + 1] = 50.0;
            } else if avg_loss == 0.0 {
                out[i + 1] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                let val = 100.0 - (100.0 / (1.0 + rs));
                out[i + 1] = if val.is_finite() { val } else { 50.0 };
            }
        }

        out
    }

    pub fn atr(klines: &[Kline], period: usize) -> Vec<f64> {
        let n = klines.len();
        if n == 0 || period == 0 {
            return Vec::new();
        }

        let mut tr = Vec::with_capacity(n);
        let tr0 = if klines[0].high.is_finite() && klines[0].low.is_finite() {
            let d = (klines[0].high - klines[0].low).abs();
            if d.is_finite() { d } else { 0.0 }
        } else {
            0.0
        };
        tr.push(tr0);

        for i in 1..n {
            let h = klines[i].high;
            let l = klines[i].low;
            let prev_c = klines[i - 1].close;

            let tr_val = if h.is_finite() && l.is_finite() {
                let hl = (h - l).abs();
                let raw = if prev_c.is_finite() {
                    hl.max((h - prev_c).abs()).max((l - prev_c).abs())
                } else {
                    hl
                };
                if raw.is_finite() { raw } else { 0.0 }
            } else {
                0.0
            };
            tr.push(tr_val);
        }

        Self::ema(&tr, period)
    }

    pub fn roc(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut out = vec![0.0; n];
        for i in period..n {
            let prev = values[i - period];
            let curr = values[i];
            if prev > 0.0 && prev.is_finite() && curr.is_finite() {
                out[i] = (curr - prev) / prev;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicators_sanitize_non_finite_values() {
        let mut vals = vec![100.0; 20];
        vals[2] = f64::NAN;
        vals[10] = f64::INFINITY;
        vals[11] = f64::NEG_INFINITY;

        let ema = TechnicalIndicators::ema(&vals, 5);
        assert_eq!(ema.len(), 20);
        assert!(ema.iter().all(|v| v.is_finite()));

        let rsi = TechnicalIndicators::rsi(&vals, 5);
        assert_eq!(rsi.len(), 20);
        assert!(rsi.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 100.0));

        let mut klines: Vec<Kline> = (0..20)
            .map(|i| Kline {
                timestamp: i as i64 * 60,
                open: 100.0,
                high: 102.0,
                low: 98.0,
                close: 101.0,
                volume: 1000.0,
            })
            .collect();
        klines[0].high = f64::NAN;
        klines[5].close = f64::INFINITY;
        klines[8].low = f64::NEG_INFINITY;

        let atr = TechnicalIndicators::atr(&klines, 5);
        assert_eq!(atr.len(), 20);
        assert!(atr.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(atr[19] > 0.0);
    }
}

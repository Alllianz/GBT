use crate::backtest::Strategy;
use crate::features::TechnicalIndicators;
use crate::types::{Kline, Signal, SignalDirection};

/// Estrategia de cruce de medias móviles exponenciales causales (EMA Fast vs Slow).
/// Cumple estrictamente t <= T sin ningún tipo de Look-Ahead Bias.
pub struct EmaCrossoverStrategy {
    name: String,
    fast_period: usize,
    slow_period: usize,
    cached_fast: Option<Vec<f64>>,
    cached_slow: Option<Vec<f64>>,
}

impl EmaCrossoverStrategy {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            name: format!("EMA_Cross_{}_{}", fast_period, slow_period),
            fast_period,
            slow_period,
            cached_fast: None,
            cached_slow: None,
        }
    }

    pub fn clear_cache(&mut self) {
        self.cached_fast = None;
        self.cached_slow = None;
    }
}

impl Strategy for EmaCrossoverStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn clear_cache(&mut self) {
        self.cached_fast = None;
        self.cached_slow = None;
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            return Signal::flat(0);
        }
        let current_ts = history[current_idx].timestamp;

        if current_idx < self.slow_period + 1 {
            return Signal::flat(current_ts);
        }

        let mut fast = self.cached_fast.take().unwrap_or_default();
        let mut slow = self.cached_slow.take().unwrap_or_default();

        if fast.is_empty() || slow.is_empty() {
            let closes: Vec<f64> = history.iter().map(|k| k.close).collect();
            fast = TechnicalIndicators::ema(&closes, self.fast_period);
            slow = TechnicalIndicators::ema(&closes, self.slow_period);
        } else {
            let k_fast = 2.0 / (self.fast_period as f64 + 1.0);
            let k_slow = 2.0 / (self.slow_period as f64 + 1.0);
            while fast.len() <= current_idx && fast.len() < history.len() {
                let idx = fast.len();
                let close = history[idx].close;
                let prev_fast = *fast.last().unwrap();
                fast.push(close * k_fast + prev_fast * (1.0 - k_fast));
            }
            while slow.len() <= current_idx && slow.len() < history.len() {
                let idx = slow.len();
                let close = history[idx].close;
                let prev_slow = *slow.last().unwrap();
                slow.push(close * k_slow + prev_slow * (1.0 - k_slow));
            }
        }

        let can_eval = current_idx < fast.len() && current_idx < slow.len() && current_idx >= 1;
        let sig = if !can_eval {
            Signal::flat(current_ts)
        } else {
            let f_c = fast[current_idx];
            let s_c = slow[current_idx];
            let f_p = fast[current_idx - 1];
            let s_p = slow[current_idx - 1];

            if f_p <= s_p && f_c > s_c {
                Signal::new(SignalDirection::Long, 1.0, current_ts)
            } else if f_p >= s_p && f_c < s_c {
                Signal::new(SignalDirection::Short, 1.0, current_ts)
            } else if f_c > s_c {
                Signal::new(SignalDirection::Long, 0.5, current_ts)
            } else {
                Signal::new(SignalDirection::Short, 0.5, current_ts)
            }
        };

        self.cached_fast = Some(fast);
        self.cached_slow = Some(slow);
        sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_cross_oob_index_returns_flat() {
        let mut strat = EmaCrossoverStrategy::new(5, 10);
        let sig = strat.evaluate(&[], 0);
        assert_eq!(sig.direction, SignalDirection::Flat);
    }
}

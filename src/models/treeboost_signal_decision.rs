use crate::features::TechnicalIndicators;
use crate::types::{Kline, Signal, SignalDirection};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeBoostSignalDecision {
    last_direction: SignalDirection,
    bars_held: usize,
    trail_extreme: Option<f64>,
}

impl Default for TreeBoostSignalDecision {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeBoostSignalDecision {
    pub fn new() -> Self {
        Self {
            last_direction: SignalDirection::Flat,
            bars_held: 0,
            trail_extreme: None,
        }
    }

    pub fn reset(&mut self) {
        self.last_direction = SignalDirection::Flat;
        self.bars_held = 0;
        self.trail_extreme = None;
    }

    pub fn last_direction(&self) -> SignalDirection {
        self.last_direction
    }

    pub fn bars_held(&self) -> usize {
        self.bars_held
    }

    pub fn causal_atr_14(history: &[Kline], current_idx: usize) -> Option<f64> {
        if current_idx >= history.len() {
            return None;
        }
        let start = current_idx.saturating_sub(84);
        let slice = &history[start..=current_idx];
        let period = 14usize.min(slice.len());
        if period >= 2 {
            TechnicalIndicators::atr(slice, period)
                .last()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
        } else if slice.len() == 1 {
            let tr = slice[0].high - slice[0].low;
            if tr.is_finite() && tr > 0.0 {
                Some(tr)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn causal_return_zscore(history: &[Kline], current_idx: usize, window: usize) -> Option<f64> {
        if window < 2 || current_idx < window || current_idx >= history.len() {
            return None;
        }
        let start = current_idx + 1 - window;
        let mut returns = Vec::with_capacity(window);
        for i in start..=current_idx {
            let prev_c = history[i - 1].close;
            let curr_c = history[i].close;
            if !prev_c.is_finite() || prev_c <= 0.0 || !curr_c.is_finite() || curr_c <= 0.0 {
                return None;
            }
            returns.push((curr_c - prev_c) / prev_c);
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns
            .iter()
            .map(|&r| {
                let d = r - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        let std = var.max(0.0).sqrt();
        if !std.is_finite() || std <= 1e-12 {
            return None;
        }
        let last_r = *returns.last()?;
        let z = (last_r - mean) / std;
        if z.is_finite() {
            Some(z)
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        history: &[Kline],
        current_idx: usize,
        predicted_return: f64,
        long_th: f64,
        short_th: f64,
        primary_sl_atr_mult: f64,
        min_hold_bars: usize,
    ) -> Signal {
        self.step_with_params(
            history,
            current_idx,
            predicted_return,
            long_th,
            short_th,
            primary_sl_atr_mult,
            None,
            None,
            min_hold_bars,
            0.25,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_params(
        &mut self,
        history: &[Kline],
        current_idx: usize,
        predicted_return: f64,
        long_th: f64,
        short_th: f64,
        primary_sl_atr_mult: f64,
        primary_tp_atr_mult: Option<f64>,
        trailing_atr_mult: Option<f64>,
        min_hold_bars: usize,
        hysteresis_frac: f64,
    ) -> Signal {
        if current_idx >= history.len() {
            self.reset();
            return Signal::flat(0);
        }

        let ts = history[current_idx].timestamp;
        let close = history[current_idx].close;
        let atr = Self::causal_atr_14(history, current_idx);

        if let (Some(t_mult), Some(a)) = (
            trailing_atr_mult.filter(|&m| m.is_finite() && m > 0.0),
            atr,
        ) {
            if self.bars_held >= 1 && self.last_direction != SignalDirection::Flat {
                let high = history[current_idx].high;
                let low = history[current_idx].low;
                match self.last_direction {
                    SignalDirection::Long => {
                        let peak = self.trail_extreme.map_or(close, |p| {
                            if high.is_finite() {
                                p.max(high).max(close)
                            } else {
                                p.max(close)
                            }
                        });
                        self.trail_extreme = Some(peak);
                        if close.is_finite() && close <= peak - t_mult * a {
                            self.reset();
                            return Signal::flat(ts);
                        }
                    }
                    SignalDirection::Short => {
                        let trough = self.trail_extreme.map_or(close, |p| {
                            if low.is_finite() {
                                p.min(low).min(close)
                            } else {
                                p.min(close)
                            }
                        });
                        self.trail_extreme = Some(trough);
                        if close.is_finite() && close >= trough + t_mult * a {
                            self.reset();
                            return Signal::flat(ts);
                        }
                    }
                    SignalDirection::Flat => {}
                }
            }
        }

        let pred = if predicted_return.is_finite() {
            predicted_return
        } else {
            0.0
        };

        let h_frac = if hysteresis_frac.is_finite() {
            hysteresis_frac.clamp(0.0, 1.0)
        } else {
            0.25
        };
        let hold_long_th = h_frac * long_th;
        let hold_short_th = h_frac * short_th;

        let raw_dir = if pred > long_th {
            SignalDirection::Long
        } else if pred < short_th {
            SignalDirection::Short
        } else if self.last_direction == SignalDirection::Long && pred > hold_long_th {
            SignalDirection::Long
        } else if self.last_direction == SignalDirection::Short && pred < hold_short_th {
            SignalDirection::Short
        } else {
            SignalDirection::Flat
        };

        let min_hold = min_hold_bars.max(1);
        let effective_dir = if self.last_direction != SignalDirection::Flat
            && raw_dir != self.last_direction
            && self.bars_held < min_hold
        {
            self.bars_held = self.bars_held.saturating_add(1);
            self.last_direction
        } else if raw_dir == SignalDirection::Flat {
            self.last_direction = SignalDirection::Flat;
            self.bars_held = 0;
            SignalDirection::Flat
        } else if raw_dir == self.last_direction {
            self.bars_held = self.bars_held.saturating_add(1);
            raw_dir
        } else {
            self.last_direction = raw_dir;
            self.bars_held = 1;
            raw_dir
        };

        match effective_dir {
            SignalDirection::Flat => {
                self.trail_extreme = None;
            }
            SignalDirection::Long | SignalDirection::Short => {
                if self.bars_held == 1 {
                    self.trail_extreme = if close.is_finite() { Some(close) } else { None };
                }
            }
        }

        let strength = match effective_dir {
            SignalDirection::Long => {
                if pred > long_th {
                    let divisor = (long_th * 3.0).abs();
                    let raw = if divisor > 1e-9 { pred / divisor } else { 0.5 };
                    if raw.is_finite() {
                        raw.clamp(0.25, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    let denom = long_th.abs().max(1e-9);
                    let raw = pred / denom;
                    if raw.is_finite() {
                        raw.clamp(0.25, 1.0)
                    } else {
                        0.25
                    }
                }
            }
            SignalDirection::Short => {
                if pred < short_th {
                    let divisor = (short_th * 3.0).abs();
                    let raw = if divisor > 1e-9 {
                        pred.abs() / divisor
                    } else {
                        0.5
                    };
                    if raw.is_finite() {
                        raw.clamp(0.25, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    let denom = short_th.abs().max(1e-9);
                    let raw = pred.abs() / denom;
                    if raw.is_finite() {
                        raw.clamp(0.25, 1.0)
                    } else {
                        0.25
                    }
                }
            }
            SignalDirection::Flat => return Signal::flat(ts),
        };

        let sl = if primary_sl_atr_mult > 0.0 && close.is_finite() && close > 0.0 {
            atr.and_then(|a| {
                let level = match effective_dir {
                    SignalDirection::Long => close - primary_sl_atr_mult * a,
                    SignalDirection::Short => close + primary_sl_atr_mult * a,
                    SignalDirection::Flat => return None,
                };
                if level.is_finite() && level > 0.0 {
                    Some(level)
                } else {
                    None
                }
            })
        } else {
            None
        };

        let tp = match primary_tp_atr_mult {
            Some(tp_mult)
                if tp_mult.is_finite() && tp_mult > 0.0 && close.is_finite() && close > 0.0 =>
            {
                atr.and_then(|a| {
                    let level = match effective_dir {
                        SignalDirection::Long => close + tp_mult * a,
                        SignalDirection::Short => close - tp_mult * a,
                        SignalDirection::Flat => return None,
                    };
                    if level.is_finite() && level > 0.0 {
                        Some(level)
                    } else {
                        None
                    }
                })
            }
            _ => None,
        };

        Signal::with_sl_tp(effective_dir, strength, ts, sl, tp)
    }
}

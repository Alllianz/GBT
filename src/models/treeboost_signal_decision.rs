use crate::features::TechnicalIndicators;
use crate::types::{HedgeLegAction, HedgeSignal, Kline, Signal, SignalDirection};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeBoostSignalDecision {
    last_direction: SignalDirection,
    bars_held: usize,
    trail_extreme: Option<f64>,
    entry_price: Option<f64>,
    trailing_armed: bool,
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
            entry_price: None,
            trailing_armed: false,
        }
    }

    pub fn reset(&mut self) {
        self.last_direction = SignalDirection::Flat;
        self.bars_held = 0;
        self.trail_extreme = None;
        self.entry_price = None;
        self.trailing_armed = false;
    }

    pub fn last_direction(&self) -> SignalDirection {
        self.last_direction
    }

    pub fn current_direction(&self) -> SignalDirection {
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
        self.step_with_profit_lock(
            history,
            current_idx,
            predicted_return,
            long_th,
            short_th,
            primary_sl_atr_mult,
            primary_tp_atr_mult,
            trailing_atr_mult,
            min_hold_bars,
            hysteresis_frac,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_profit_lock(
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
        trail_activate_atr_mult: Option<f64>,
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
                let act_mult = trail_activate_atr_mult.filter(|&m| m.is_finite() && m >= 0.0);
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
                        if let Some(act) = act_mult {
                            if !self.trailing_armed {
                                if let Some(entry) = self.entry_price.filter(|&e| e.is_finite()) {
                                    if peak - entry >= act * a {
                                        self.trailing_armed = true;
                                    }
                                }
                            }
                        }
                        let armed = act_mult.is_none() || self.trailing_armed;
                        if armed && close.is_finite() && close <= peak - t_mult * a {
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
                        if let Some(act) = act_mult {
                            if !self.trailing_armed {
                                if let Some(entry) = self.entry_price.filter(|&e| e.is_finite()) {
                                    if entry - trough >= act * a {
                                        self.trailing_armed = true;
                                    }
                                }
                            }
                        }
                        let armed = act_mult.is_none() || self.trailing_armed;
                        if armed && close.is_finite() && close >= trough + t_mult * a {
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
                self.entry_price = None;
                self.trailing_armed = false;
            }
            SignalDirection::Long | SignalDirection::Short => {
                if self.bars_held == 1 {
                    let valid_close = if close.is_finite() { Some(close) } else { None };
                    self.trail_extreme = valid_close;
                    self.entry_price = valid_close;
                    self.trailing_armed = false;
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

pub(crate) fn clamp_signal_max_sl(
    mut signal: Signal,
    close: f64,
    max_sl_pct: Option<f64>,
) -> Signal {
    if let (Some(pct), Some(sl_price)) = (
        max_sl_pct.filter(|&p| p.is_finite() && p > 0.0),
        signal.stop_loss_price,
    ) {
        if close.is_finite() && close > 0.0 {
            let max_dist = close * pct;
            let clamped = match signal.direction {
                SignalDirection::Long => {
                    let atr_sl = (close - sl_price).max(0.0);
                    Some(close - atr_sl.min(max_dist))
                }
                SignalDirection::Short => {
                    let atr_sl = (sl_price - close).max(0.0);
                    Some(close + atr_sl.min(max_dist))
                }
                SignalDirection::Flat => None,
            };
            if let Some(level) = clamped.filter(|&v| v.is_finite() && v > 0.0) {
                signal.stop_loss_price = Some(level);
            }
        }
    }
    signal
}

pub(crate) fn compute_vol_target_multiplier_bounded(
    close: f64,
    atr: Option<f64>,
    vol_target_pct: Option<f64>,
    min_mult: f64,
    max_mult: f64,
) -> f64 {
    let min_b = if min_mult.is_finite() && min_mult > 0.0 {
        min_mult
    } else {
        0.35
    };
    let max_b = if max_mult.is_finite() && max_mult >= min_b {
        max_mult
    } else {
        min_b.max(1.25)
    };
    match (vol_target_pct, atr) {
        (Some(vt), Some(a))
            if vt.is_finite()
                && vt > 0.0
                && a.is_finite()
                && a > 0.0
                && close.is_finite()
                && close > 0.0 =>
        {
            let atr_pct = a / close;
            if atr_pct > 1e-9 && atr_pct.is_finite() {
                (vt / atr_pct).clamp(min_b, max_b)
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

#[allow(dead_code)]
pub(crate) fn compute_vol_target_multiplier(
    close: f64,
    atr: Option<f64>,
    vol_target_pct: Option<f64>,
) -> f64 {
    compute_vol_target_multiplier_bounded(close, atr, vol_target_pct, 0.35, 1.25)
}

pub(crate) fn build_vol_scaled_primary_hedge_signal(
    primary: Signal,
    history: &[Kline],
    current_idx: usize,
    vol_target_pct: Option<f64>,
    vol_min_mult: f64,
    vol_max_mult: f64,
) -> HedgeSignal {
    let mut hedge_sig = HedgeSignal::from_net_signal(primary.clone());
    if vol_target_pct.is_some() && current_idx < history.len() {
        let close = history[current_idx].close;
        let atr = TreeBoostSignalDecision::causal_atr_14(history, current_idx);
        let mult = compute_vol_target_multiplier_bounded(
            close,
            atr,
            vol_target_pct,
            vol_min_mult,
            vol_max_mult,
        );
        match primary.direction {
            SignalDirection::Long => {
                hedge_sig.long = HedgeLegAction::open_or_keep(
                    primary.strength,
                    primary.stop_loss_price,
                    primary.take_profit_price,
                    mult,
                );
            }
            SignalDirection::Short => {
                hedge_sig.short = HedgeLegAction::open_or_keep(
                    primary.strength,
                    primary.stop_loss_price,
                    primary.take_profit_price,
                    mult,
                );
            }
            SignalDirection::Flat => {}
        }
    }
    hedge_sig
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_atr_klines(n: usize, base: f64, spread: f64) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 14_400_000,
                open: base,
                high: base + spread,
                low: base - spread,
                close: base,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_profit_lock_trailing_ignores_pre_activation_pullback_and_triggers_post_activation() {
        let mut klines = flat_atr_klines(30, 100.0, 0.5);
        klines[20].close = 100.0;
        klines[21].high = 104.5;
        klines[21].low = 100.0;
        klines[21].close = 102.0;
        klines[22].high = 102.0;
        klines[22].low = 99.0;
        klines[22].close = 99.0;
        klines[23].high = 112.0;
        klines[23].low = 99.0;
        klines[23].close = 111.0;
        klines[24].high = 111.0;
        klines[24].low = 100.0;
        klines[24].close = 100.5;

        let mut dec_lock = TreeBoostSignalDecision::new();
        let s20 = dec_lock.step_with_profit_lock(
            &klines,
            20,
            0.005,
            0.002,
            -0.002,
            2.0,
            None,
            Some(3.0),
            2,
            0.25,
            Some(4.0),
        );
        assert_eq!(s20.direction, SignalDirection::Long);
        assert_eq!(dec_lock.entry_price, Some(100.0));
        assert!(!dec_lock.trailing_armed);

        let s21 = dec_lock.step_with_profit_lock(
            &klines,
            21,
            0.003,
            0.002,
            -0.002,
            2.0,
            None,
            Some(3.0),
            2,
            0.25,
            Some(4.0),
        );
        assert_eq!(s21.direction, SignalDirection::Long);
        assert!(!dec_lock.trailing_armed);

        let atr22 = TreeBoostSignalDecision::causal_atr_14(&klines, 22).unwrap();
        assert!(99.0 <= 104.5 - 3.0 * atr22);
        let s22 = dec_lock.step_with_profit_lock(
            &klines,
            22,
            0.003,
            0.002,
            -0.002,
            2.0,
            None,
            Some(3.0),
            2,
            0.25,
            Some(4.0),
        );
        assert_eq!(s22.direction, SignalDirection::Long);
        assert!(!dec_lock.trailing_armed);

        let _atr23 = TreeBoostSignalDecision::causal_atr_14(&klines, 23).unwrap();
        let s23 = dec_lock.step_with_profit_lock(
            &klines,
            23,
            0.003,
            0.002,
            -0.002,
            2.0,
            None,
            Some(3.0),
            2,
            0.25,
            Some(4.0),
        );
        assert_eq!(s23.direction, SignalDirection::Long);
        assert!(dec_lock.trailing_armed);

        let s24 = dec_lock.step_with_profit_lock(
            &klines,
            24,
            0.003,
            0.002,
            -0.002,
            2.0,
            None,
            Some(3.0),
            2,
            0.25,
            Some(4.0),
        );
        assert_eq!(s24.direction, SignalDirection::Flat);
        assert_eq!(dec_lock.entry_price, None);
        assert!(!dec_lock.trailing_armed);

        let mut dec_immediate = TreeBoostSignalDecision::new();
        assert_eq!(
            dec_immediate
                .step_with_params(&klines, 20, 0.005, 0.002, -0.002, 2.0, None, Some(3.0), 2, 0.25)
                .direction,
            SignalDirection::Long
        );
        assert_eq!(
            dec_immediate
                .step_with_params(&klines, 21, 0.003, 0.002, -0.002, 2.0, None, Some(3.0), 2, 0.25)
                .direction,
            SignalDirection::Long
        );
        assert_eq!(
            dec_immediate
                .step_with_params(&klines, 22, 0.003, 0.002, -0.002, 2.0, None, Some(3.0), 2, 0.25)
                .direction,
            SignalDirection::Flat
        );

        let mut short_klines = flat_atr_klines(30, 100.0, 0.5);
        short_klines[20].close = 100.0;
        short_klines[21].low = 95.5;
        short_klines[21].high = 100.0;
        short_klines[21].close = 98.0;
        short_klines[22].low = 98.0;
        short_klines[22].high = 101.0;
        short_klines[22].close = 101.0;
        short_klines[23].low = 91.0;
        short_klines[23].high = 101.0;
        short_klines[23].close = 92.0;
        short_klines[24].low = 92.0;
        short_klines[24].high = 100.0;
        short_klines[24].close = 99.5;

        let mut dec_short = TreeBoostSignalDecision::new();
        assert_eq!(
            dec_short
                .step_with_profit_lock(
                    &short_klines,
                    20,
                    -0.005,
                    0.002,
                    -0.002,
                    2.0,
                    None,
                    Some(3.0),
                    2,
                    0.25,
                    Some(4.0),
                )
                .direction,
            SignalDirection::Short
        );
        assert_eq!(
            dec_short
                .step_with_profit_lock(
                    &short_klines,
                    21,
                    -0.003,
                    0.002,
                    -0.002,
                    2.0,
                    None,
                    Some(3.0),
                    2,
                    0.25,
                    Some(4.0),
                )
                .direction,
            SignalDirection::Short
        );
        assert!(!dec_short.trailing_armed);
        assert_eq!(
            dec_short
                .step_with_profit_lock(
                    &short_klines,
                    22,
                    -0.003,
                    0.002,
                    -0.002,
                    2.0,
                    None,
                    Some(3.0),
                    2,
                    0.25,
                    Some(4.0),
                )
                .direction,
            SignalDirection::Short
        );
        assert!(!dec_short.trailing_armed);
        assert_eq!(
            dec_short
                .step_with_profit_lock(
                    &short_klines,
                    23,
                    -0.003,
                    0.002,
                    -0.002,
                    2.0,
                    None,
                    Some(3.0),
                    2,
                    0.25,
                    Some(4.0),
                )
                .direction,
            SignalDirection::Short
        );
        assert!(dec_short.trailing_armed);
        assert_eq!(
            dec_short
                .step_with_profit_lock(
                    &short_klines,
                    24,
                    -0.003,
                    0.002,
                    -0.002,
                    2.0,
                    None,
                    Some(3.0),
                    2,
                    0.25,
                    Some(4.0),
                )
                .direction,
            SignalDirection::Flat
        );
    }
}

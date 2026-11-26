use crate::backtest::Strategy;
use crate::models::treeboost_signal_decision::{
    build_vol_scaled_primary_hedge_signal, clamp_signal_max_sl,
};
use crate::models::{TreeBoostHedgeConfig, TreeBoostSignalDecision};
use crate::types::{HedgeSignal, Kline, Signal, SignalDirection};

pub struct OofTestStrategy<'a> {
    pub predictions: &'a [f64],
    pub long_th: f64,
    pub short_th: f64,
    pub warmup_bars: usize,
    pub hedge_config: TreeBoostHedgeConfig,
    pub decision: TreeBoostSignalDecision,
}

impl<'a> Strategy for OofTestStrategy<'a> {
    fn name(&self) -> &str {
        "OofTestStrategy"
    }

    fn clear_cache(&mut self) {
        self.decision.reset();
    }

    fn sync_open_legs(&mut self, long_open: bool, short_open: bool) {
        match self.decision.current_direction() {
            SignalDirection::Long if !long_open => self.decision.reset(),
            SignalDirection::Short if !short_open => self.decision.reset(),
            _ => {}
        }
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            self.decision.reset();
            return Signal::flat(0);
        }
        let ts = history[current_idx].timestamp;
        if current_idx >= self.predictions.len() || current_idx < self.warmup_bars {
            self.decision.reset();
            return Signal::flat(ts);
        }

        let pred = self.predictions[current_idx];
        let sig = self.decision.step_with_profit_lock(
            history,
            current_idx,
            pred,
            self.long_th,
            self.short_th,
            self.hedge_config.primary_sl_atr_mult,
            self.hedge_config.primary_tp_atr_mult,
            self.hedge_config.trailing_atr_mult,
            self.hedge_config.min_hold_bars,
            self.hedge_config.hysteresis_frac,
            self.hedge_config.trail_activate_atr_mult,
        );
        clamp_signal_max_sl(
            sig,
            history[current_idx].close,
            self.hedge_config.max_sl_pct,
        )
    }

    fn evaluate_hedge(&mut self, history: &[Kline], current_idx: usize) -> HedgeSignal {
        let primary = self.evaluate(history, current_idx);
        build_vol_scaled_primary_hedge_signal(
            primary,
            history,
            current_idx,
            self.hedge_config.vol_target_pct,
            self.hedge_config.vol_min_mult,
            self.hedge_config.vol_max_mult,
        )
    }
}

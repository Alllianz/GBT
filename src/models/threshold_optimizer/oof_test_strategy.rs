use crate::backtest::Strategy;
use crate::models::{TreeBoostHedgeConfig, TreeBoostSignalDecision};
use crate::types::{Kline, Signal};

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
        self.decision.step(
            history,
            current_idx,
            pred,
            self.long_th,
            self.short_th,
            self.hedge_config.primary_sl_atr_mult,
            self.hedge_config.min_hold_bars,
        )
    }
}

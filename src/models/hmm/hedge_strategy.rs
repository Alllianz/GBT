use crate::backtest::Strategy;
use crate::features::TechnicalIndicators;
use crate::types::{
    HedgeLegAction, HedgeSignal, Kline, Signal, SignalDirection, Timeframe, TradeRegime,
};

use super::features::HmmFeatureExtractor;
use super::gaussian_hmm::GaussianHmm;
use super::regime::{HmmRegime, RegimePosterior};
use super::strategy_config::HmmHedgeStrategyConfig;

#[derive(Debug, Clone)]
pub struct HmmDualHedgeStrategy {
    name: String,
    config: HmmHedgeStrategyConfig,
    hmm: GaussianHmm,
    base_hmm: Option<GaussianHmm>,
    active_regime: Option<HmmRegime>,
    bars_in_regime: usize,
    last_posterior: Option<RegimePosterior>,
    last_evaluated_idx: Option<usize>,
    last_fit_idx: Option<usize>,
    mr_long_active: bool,
    mr_short_active: bool,
    last_hedge_idx: Option<usize>,
    last_hedge_signal: Option<HedgeSignal>,
}

impl HmmDualHedgeStrategy {
    pub fn new(config: HmmHedgeStrategyConfig) -> Self {
        let hmm = GaussianHmm::new(config.hmm_config);
        Self {
            name: "HMM_Dual_Hedge".to_string(),
            config,
            hmm,
            base_hmm: None,
            active_regime: None,
            bars_in_regime: 0,
            last_posterior: None,
            last_evaluated_idx: None,
            last_fit_idx: None,
            mr_long_active: false,
            mr_short_active: false,
            last_hedge_idx: None,
            last_hedge_signal: None,
        }
    }

    pub fn for_timeframe(tf: Timeframe) -> Self {
        Self::new(HmmHedgeStrategyConfig::for_timeframe(tf))
    }

    pub fn config(&self) -> HmmHedgeStrategyConfig {
        self.config
    }

    pub fn is_fitted(&self) -> bool {
        self.hmm.is_fitted()
    }

    pub fn hmm(&self) -> &GaussianHmm {
        &self.hmm
    }

    pub fn last_posterior(&self) -> Option<RegimePosterior> {
        self.last_posterior
    }

    pub fn active_regime(&self) -> Option<HmmRegime> {
        self.active_regime
    }

    pub fn mr_long_active(&self) -> bool {
        self.mr_long_active
    }

    pub fn mr_short_active(&self) -> bool {
        self.mr_short_active
    }

    pub fn fit_on_klines(&mut self, klines: &[Kline]) -> Result<(), String> {
        if klines.is_empty() {
            return Err("Cannot fit HmmDualHedgeStrategy on empty klines".to_string());
        }
        let obs = HmmFeatureExtractor::extract_causal_window(
            klines,
            klines.len() - 1,
            self.config.vol_window,
        );
        self.hmm.fit_causal(&obs)?;
        self.base_hmm = Some(self.hmm.clone());
        self.reset_runtime_state();
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        self.reset_runtime_state();
    }

    fn reset_runtime_state(&mut self) {
        if let Some(ref base) = self.base_hmm {
            self.hmm = base.clone();
        }
        self.hmm.reset_filter();
        self.active_regime = None;
        self.bars_in_regime = 0;
        self.last_posterior = None;
        self.last_evaluated_idx = None;
        self.last_fit_idx = None;
        self.mr_long_active = false;
        self.mr_short_active = false;
        self.last_hedge_idx = None;
        self.last_hedge_signal = None;
    }

    fn step_regime(&mut self, history: &[Kline], current_idx: usize) -> Option<(HmmRegime, RegimePosterior)> {
        if current_idx >= history.len() {
            return None;
        }
        if self.last_evaluated_idx.map_or(false, |prev| current_idx < prev) {
            self.reset_runtime_state();
        }
        if self.last_evaluated_idx == Some(current_idx) {
            return self.active_regime.zip(self.last_posterior);
        }

        let obs = HmmFeatureExtractor::extract_causal_step(
            history,
            current_idx,
            self.config.vol_window,
        )?;

        let min_obs = 20usize.min(self.config.hmm_config.lookback_window).max(6);
        let refit_interval = self.config.hmm_config.refit_interval;

        if !self.hmm.is_fitted() {
            let window = HmmFeatureExtractor::extract_causal_window(
                history,
                current_idx,
                self.config.vol_window,
            );
            if window.len() < min_obs {
                self.last_evaluated_idx = Some(current_idx);
                return None;
            }
            let start = window.len().saturating_sub(self.config.hmm_config.lookback_window);
            if self.hmm.fit_causal(&window[start..]).is_err() {
                self.last_evaluated_idx = Some(current_idx);
                return None;
            }
            self.hmm.reset_filter();
            let tail_end = window.len().saturating_sub(1);
            if start < tail_end {
                for &prev_obs in &window[start..tail_end] {
                    let _ = self.hmm.forward_step(prev_obs);
                }
            }
            self.last_fit_idx = Some(current_idx);
        } else if refit_interval > 0 {
            match self.last_fit_idx {
                None => {
                    self.last_fit_idx = Some(current_idx);
                }
                Some(prev_fit) if current_idx.saturating_sub(prev_fit) >= refit_interval => {
                    let window = HmmFeatureExtractor::extract_causal_window(
                        history,
                        current_idx,
                        self.config.vol_window,
                    );
                    if window.len() >= min_obs {
                        let start = window.len().saturating_sub(self.config.hmm_config.lookback_window);
                        let mut candidate_hmm = GaussianHmm::new(self.config.hmm_config);
                        if candidate_hmm.fit_causal(&window[start..]).is_ok() {
                            let tail_end = window.len().saturating_sub(1);
                            if start < tail_end {
                                for &prev_obs in &window[start..tail_end] {
                                    let _ = candidate_hmm.forward_step(prev_obs);
                                }
                            }
                            self.hmm = candidate_hmm;
                            self.last_fit_idx = Some(current_idx);
                        }
                    }
                }
                _ => {}
            }
        }

        if self.active_regime.is_some() {
            self.bars_in_regime = self.bars_in_regime.saturating_add(1);
        }

        let posterior = self.hmm.forward_step(obs);
        self.last_posterior = Some(posterior);
        self.last_evaluated_idx = Some(current_idx);

        let candidate = match posterior.dominant {
            HmmRegime::Bull if posterior.confidence >= self.config.bull_confidence_threshold => {
                HmmRegime::Bull
            }
            HmmRegime::Bear if posterior.confidence >= self.config.bear_confidence_threshold => {
                HmmRegime::Bear
            }
            _ => HmmRegime::Sideways,
        };

        let effective = match self.active_regime {
            Some(prev) if prev != candidate && self.bars_in_regime < self.config.min_holding_bars => {
                prev
            }
            Some(prev) if prev == candidate => prev,
            _ => {
                self.active_regime = Some(candidate);
                self.bars_in_regime = 1;
                candidate
            }
        };

        Some((effective, posterior))
    }

    fn causal_zscore(history: &[Kline], idx: usize, window: usize) -> Option<f64> {
        if window < 2 || idx + 1 < window || idx >= history.len() {
            return None;
        }
        let slice = &history[(idx + 1 - window)..=idx];
        let mut sum = 0.0;
        for k in slice {
            if !k.close.is_finite() || k.close <= 0.0 {
                return None;
            }
            sum += k.close;
        }
        let n = window as f64;
        let mean = sum / n;
        let mut var_sum = 0.0;
        for k in slice {
            let diff = k.close - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / (n - 1.0)).max(0.0).sqrt();
        if !std.is_finite() || std <= 1e-12 {
            return None;
        }
        let z = (history[idx].close - mean) / std;
        if z.is_finite() { Some(z) } else { None }
    }

    fn compute_causal_indicators(
        &self,
        history: &[Kline],
        idx: usize,
    ) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        let start = idx.saturating_sub(self.config.ema_slow.saturating_mul(3));
        let slice = &history[start..=idx];

        let mut valid_closes = true;
        let mut closes = Vec::with_capacity(slice.len());
        for k in slice {
            if !k.close.is_finite() || k.close <= 0.0 {
                valid_closes = false;
                break;
            }
            closes.push(k.close);
        }

        let ema_fast = if valid_closes
            && self.config.ema_fast > 0
            && closes.len() >= self.config.ema_fast
        {
            TechnicalIndicators::ema(&closes, self.config.ema_fast)
                .last()
                .copied()
                .filter(|v| v.is_finite())
        } else {
            None
        };

        let ema_slow = if valid_closes
            && self.config.ema_slow > 0
            && closes.len() >= self.config.ema_slow
        {
            TechnicalIndicators::ema(&closes, self.config.ema_slow)
                .last()
                .copied()
                .filter(|v| v.is_finite())
        } else {
            None
        };

        let atr_period = 14usize.min(slice.len());
        let atr = if atr_period >= 2 {
            TechnicalIndicators::atr(slice, atr_period)
                .last()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
        } else if slice.len() == 1 {
            let tr = slice[0].high - slice[0].low;
            if tr.is_finite() && tr > 0.0 { Some(tr) } else { None }
        } else {
            None
        };

        let z = Self::causal_zscore(history, idx, self.config.zscore_window);

        (ema_fast, ema_slow, atr, z)
    }

    fn update_mr_state(&mut self, z: Option<f64>, posterior: RegimePosterior) {
        let p_bear = posterior.probs[HmmRegime::Bear.as_index()];
        let p_bull = posterior.probs[HmmRegime::Bull.as_index()];
        if let Some(z_val) = z {
            self.mr_long_active = if self.mr_long_active {
                z_val < -self.config.mr_z_exit && p_bear < 0.70
            } else {
                z_val <= -self.config.mr_z_entry && p_bear < 0.70
            };
            self.mr_short_active = if self.mr_short_active {
                z_val > self.config.mr_z_exit && p_bull < 0.70
            } else {
                z_val >= self.config.mr_z_entry && p_bull < 0.70
            };
        } else {
            self.mr_long_active = false;
            self.mr_short_active = false;
        }
    }

    fn build_hedge_signal(
        &self,
        ts: i64,
        close: f64,
        regime: HmmRegime,
        posterior: RegimePosterior,
        ema_fast: Option<f64>,
        ema_slow: Option<f64>,
        atr: Option<f64>,
    ) -> HedgeSignal {
        let trend_long = matches!(
            (ema_fast, ema_slow),
            (Some(f), Some(s)) if f > s && regime != HmmRegime::Bear
        );
        let trend_short = matches!(
            (ema_fast, ema_slow),
            (Some(f), Some(s)) if f < s && regime != HmmRegime::Bull
        );

        let p_bear = posterior.probs[HmmRegime::Bear.as_index()];
        let p_bull = posterior.probs[HmmRegime::Bull.as_index()];
        let p_side = posterior.probs[HmmRegime::Sideways.as_index()];

        let long = if trend_long {
            let mult = self.config.bull_exposure.max(0.0);
            if mult > 0.0 {
                let sl = atr.map(|a| close - 3.0 * a);
                HedgeLegAction::open_or_keep_with_regime(
                    posterior.confidence,
                    sl,
                    None,
                    mult,
                    TradeRegime::PrimaryTrend,
                )
            } else {
                HedgeLegAction::Close
            }
        } else if self.mr_long_active {
            let mult = self.config.sideways_exposure.max(0.0);
            if mult > 0.0 {
                let sl = atr.map(|a| close - 2.0 * a);
                let tp = atr.map(|a| close + 2.5 * a);
                HedgeLegAction::open_or_keep_with_regime(
                    p_side.max(p_bull),
                    sl,
                    tp,
                    mult,
                    TradeRegime::MeanReversionHedge,
                )
            } else {
                HedgeLegAction::Close
            }
        } else {
            HedgeLegAction::Close
        };

        let short = if trend_short {
            let mult = self.config.bear_exposure.abs();
            if mult > 0.0 {
                let sl = atr.map(|a| close + 3.0 * a);
                HedgeLegAction::open_or_keep_with_regime(
                    posterior.confidence,
                    sl,
                    None,
                    mult,
                    TradeRegime::PrimaryTrend,
                )
            } else {
                HedgeLegAction::Close
            }
        } else if self.mr_short_active {
            let mult = self.config.sideways_exposure.max(0.0);
            if mult > 0.0 {
                let sl = atr.map(|a| close + 2.0 * a);
                let tp = atr.map(|a| close - 2.5 * a);
                HedgeLegAction::open_or_keep_with_regime(
                    p_side.max(p_bear),
                    sl,
                    tp,
                    mult,
                    TradeRegime::MeanReversionHedge,
                )
            } else {
                HedgeLegAction::Close
            }
        } else {
            HedgeLegAction::Close
        };

        HedgeSignal {
            timestamp: ts,
            long,
            short,
        }
    }
}

impl Strategy for HmmDualHedgeStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn clear_cache(&mut self) {
        self.reset_runtime_state();
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            return Signal::flat(0);
        }
        let ts = history[current_idx].timestamp;
        match self.step_regime(history, current_idx) {
            Some((HmmRegime::Bull, posterior)) => {
                Signal::new(SignalDirection::Long, posterior.confidence, ts)
            }
            Some((HmmRegime::Bear, posterior)) => {
                Signal::new(SignalDirection::Short, posterior.confidence, ts)
            }
            _ => Signal::flat(ts),
        }
    }

    fn evaluate_hedge(&mut self, history: &[Kline], current_idx: usize) -> HedgeSignal {
        if current_idx >= history.len() {
            return HedgeSignal::hold(0);
        }
        if self.last_hedge_idx.map_or(false, |prev| current_idx < prev) {
            self.reset_runtime_state();
        }
        if self.last_hedge_idx == Some(current_idx) {
            return self
                .last_hedge_signal
                .unwrap_or_else(|| HedgeSignal::hold(history[current_idx].timestamp));
        }

        let ts = history[current_idx].timestamp;
        let sig = match self.step_regime(history, current_idx) {
            Some((regime, posterior)) => {
                let (ema_fast, ema_slow, atr, z) =
                    self.compute_causal_indicators(history, current_idx);
                self.update_mr_state(z, posterior);
                let close = history[current_idx].close;
                self.build_hedge_signal(ts, close, regime, posterior, ema_fast, ema_slow, atr)
            }
            None => HedgeSignal::hold(ts),
        };
        self.last_hedge_idx = Some(current_idx);
        self.last_hedge_signal = Some(sig);
        sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::{BacktestConfig, BacktestEngine};
    use crate::models::hmm::HmmConfig;
    use crate::types::{PositionSide, PositionSizing};

    fn synthetic_regime_klines() -> Vec<Kline> {
        let mut closes = Vec::with_capacity(120);
        let mut price = 100.0_f64;
        closes.push(price);

        for i in 1..40 {
            let ret = -0.018 - ((i % 3) as f64) * 0.004;
            price *= ret.exp();
            closes.push(price);
        }
        for i in 40..80 {
            let ret: f64 = if i % 2 == 0 { 0.001 } else { -0.0008 };
            price *= ret.exp();
            closes.push(price);
        }
        for i in 80..120 {
            let ret = 0.018 + ((i % 3) as f64) * 0.004;
            price *= ret.exp();
            closes.push(price);
        }

        closes
            .into_iter()
            .enumerate()
            .map(|(i, c)| Kline {
                timestamp: (i as i64) * 3_600_000,
                open: c,
                high: c * 1.005,
                low: c * 0.995,
                close: c,
                volume: 1000.0,
            })
            .collect()
    }

    fn test_config() -> HmmHedgeStrategyConfig {
        HmmHedgeStrategyConfig {
            hmm_config: HmmConfig {
                max_em_iters: 40,
                tol: 1e-6,
                min_variance: 1e-6,
                lookback_window: 100,
                refit_interval: 20,
            },
            vol_window: 5,
            ema_fast: 4,
            ema_slow: 12,
            zscore_window: 10,
            mr_z_entry: 1.75,
            mr_z_exit: 0.0,
            bull_confidence_threshold: 0.55,
            bear_confidence_threshold: 0.55,
            bull_exposure: 0.70,
            sideways_exposure: 0.40,
            bear_exposure: -0.70,
            min_holding_bars: 2,
        }
    }

    #[test]
    fn test_fit_on_klines_and_causal_no_lookahead() {
        let klines = synthetic_regime_klines();
        let split_t = 70;

        let mut s1 = HmmDualHedgeStrategy::new(test_config());
        s1.fit_on_klines(&klines[..=split_t]).unwrap();

        let mut mutated = klines.clone();
        for k in &mut mutated[split_t + 1..] {
            k.open *= 5.0;
            k.high *= 5.0;
            k.low *= 5.0;
            k.close *= 5.0;
        }

        let mut s2 = HmmDualHedgeStrategy::new(test_config());
        s2.fit_on_klines(&mutated[..=split_t]).unwrap();

        for i in 0..=split_t {
            let sig1 = s1.evaluate_hedge(&klines[..=i], i);
            let sig2 = s2.evaluate_hedge(&mutated[..=i], i);
            assert_eq!(sig1, sig2);
            assert_eq!(s1.last_posterior(), s2.last_posterior());
        }
    }

    #[test]
    fn test_dual_hedge_trend_and_mr_combination() {
        let mut strat = HmmDualHedgeStrategy::new(test_config());
        let atr = Some(2.0);
        let close = 100.0;

        let bull_post = RegimePosterior::from_probs([0.05, 0.15, 0.80]);
        let bull_sig = strat.build_hedge_signal(
            100,
            close,
            HmmRegime::Bull,
            bull_post,
            Some(105.0),
            Some(100.0),
            atr,
        );
        assert!(matches!(
            bull_sig.long,
            HedgeLegAction::OpenOrKeep {
                size_multiplier,
                stop_loss_price: Some(sl),
                take_profit_price: None,
                ..
            } if (size_multiplier - 0.70).abs() < 1e-12 && (sl - 94.0).abs() < 1e-12
        ));
        assert_eq!(bull_sig.short, HedgeLegAction::Close);

        let bear_post = RegimePosterior::from_probs([0.80, 0.15, 0.05]);
        let bear_sig = strat.build_hedge_signal(
            200,
            close,
            HmmRegime::Bear,
            bear_post,
            Some(95.0),
            Some(100.0),
            atr,
        );
        assert_eq!(bear_sig.long, HedgeLegAction::Close);
        assert!(matches!(
            bear_sig.short,
            HedgeLegAction::OpenOrKeep {
                size_multiplier,
                stop_loss_price: Some(sl),
                take_profit_price: None,
                ..
            } if (size_multiplier - 0.70).abs() < 1e-12 && (sl - 106.0).abs() < 1e-12
        ));

        let side_post = RegimePosterior::from_probs([0.10, 0.80, 0.10]);
        strat.update_mr_state(Some(2.0), side_post);
        assert!(strat.mr_short_active());
        assert!(!strat.mr_long_active());

        let combo_sig = strat.build_hedge_signal(
            300,
            close,
            HmmRegime::Sideways,
            side_post,
            Some(105.0),
            Some(100.0),
            atr,
        );
        assert!(matches!(
            combo_sig.long,
            HedgeLegAction::OpenOrKeep {
                size_multiplier,
                stop_loss_price: Some(sl),
                take_profit_price: None,
                ..
            } if (size_multiplier - 0.70).abs() < 1e-12 && (sl - 94.0).abs() < 1e-12
        ));
        assert!(matches!(
            combo_sig.short,
            HedgeLegAction::OpenOrKeep {
                size_multiplier,
                stop_loss_price: Some(sl),
                take_profit_price: Some(tp),
                ..
            } if (size_multiplier - 0.40).abs() < 1e-12
                && (sl - 104.0).abs() < 1e-12
                && (tp - 95.0).abs() < 1e-12
        ));

        strat.update_mr_state(Some(0.5), side_post);
        assert!(strat.mr_short_active());
        strat.update_mr_state(Some(0.0), side_post);
        assert!(!strat.mr_short_active());

        strat.update_mr_state(Some(-2.0), side_post);
        assert!(strat.mr_long_active());
        strat.update_mr_state(Some(-0.5), side_post);
        assert!(strat.mr_long_active());
        strat.update_mr_state(Some(-0.5), bear_post);
        assert!(!strat.mr_long_active());
    }

    #[test]
    fn test_clear_cache_preserves_fitted_model_and_resets_filter() {
        let klines = synthetic_regime_klines();
        let mut strat = HmmDualHedgeStrategy::new(test_config());
        strat.fit_on_klines(&klines).unwrap();
        assert!(strat.is_fitted());

        let mut pass1 = Vec::with_capacity(klines.len());
        for i in 0..klines.len() {
            pass1.push(strat.evaluate_hedge(&klines[..=i], i));
        }

        strat.clear_cache();
        assert!(strat.is_fitted());
        assert!(strat.active_regime().is_none());
        assert!(strat.last_posterior().is_none());
        assert!(!strat.mr_long_active());
        assert!(!strat.mr_short_active());

        for i in 0..klines.len() {
            let sig = strat.evaluate_hedge(&klines[..=i], i);
            assert_eq!(sig, pass1[i]);
        }
    }

    #[test]
    fn test_backtest_engine_hedge_mode_integration() {
        let klines = synthetic_regime_klines();
        let mut strat = HmmDualHedgeStrategy::new(test_config());
        strat.fit_on_klines(&klines).unwrap();

        let cfg = BacktestConfig {
            initial_capital: 10_000.0,
            leverage: 1.0,
            position_sizing: PositionSizing::PercentOfInitialCapital(0.25),
            allow_short: true,
            hedge_mode: true,
            ..Default::default()
        };
        let engine = BacktestEngine::new(cfg);
        let res = engine.run(&mut strat, &klines).unwrap();

        assert_eq!(res.equity_curve.len(), klines.len());
        assert!(!res.trades.is_empty());
        assert!(res.trades.iter().any(|t| t.side == PositionSide::Long));
        assert!(res.trades.iter().any(|t| t.side == PositionSide::Short));
    }

    #[test]
    fn test_causal_rolling_refit_strict_no_lookahead() {
        let klines = synthetic_regime_klines();
        let split_t = 70;
        let mut mutated = klines.clone();
        for k in &mut mutated[split_t + 1..] {
            k.open *= 4.0;
            k.high *= 4.0;
            k.low *= 4.0;
            k.close *= 4.0;
        }

        let mut cfg = test_config();
        cfg.hmm_config.refit_interval = 10;

        let mut s1 = HmmDualHedgeStrategy::new(cfg);
        let mut s2 = HmmDualHedgeStrategy::new(cfg);

        let mut fitted_at_idx = Vec::new();
        for i in 0..=split_t {
            let prev_fit = s1.last_fit_idx;
            let sig1 = s1.evaluate_hedge(&klines, i);
            let sig2 = s2.evaluate_hedge(&mutated, i);
            assert_eq!(sig1, sig2);
            assert_eq!(s1.last_posterior(), s2.last_posterior());
            if s1.last_fit_idx != prev_fit {
                if let Some(fit_i) = s1.last_fit_idx {
                    fitted_at_idx.push(fit_i);
                }
            }
        }

        assert!(fitted_at_idx.len() >= 3);
        assert!(fitted_at_idx.windows(2).all(|w| w[1] - w[0] >= cfg.hmm_config.refit_interval));
    }
}

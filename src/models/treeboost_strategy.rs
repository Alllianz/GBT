use crate::backtest::Strategy;
use crate::features::{FeatureMatrix, H4FeatureMatrix};
use crate::types::{
    HedgeLegAction, HedgeSignal, Kline, Signal, SignalDirection, Timeframe, TradeRegime,
};
use super::treeboost_hedge_config::TreeBoostHedgeConfig;
use super::treeboost_model::TreeBoostModel;
use super::treeboost_mr_indicators::TreeBoostMrIndicators;
use super::treeboost_signal_decision::TreeBoostSignalDecision;

pub struct TreeBoostStrategy {
    name: String,
    model: TreeBoostModel,
    cached_predictions: Option<Vec<f64>>,
    decision: TreeBoostSignalDecision,
    timeframe: Option<Timeframe>,
    hedge_config: TreeBoostHedgeConfig,
    hedge_long_active: bool,
    hedge_short_active: bool,
    last_eval_idx: Option<usize>,
    last_eval_signal: Option<Signal>,
    last_hedge_idx: Option<usize>,
    last_hedge_signal: Option<HedgeSignal>,
}

impl TreeBoostStrategy {
    pub fn new(model: TreeBoostModel) -> Self {
        let timeframe = model.timeframe();
        let hedge_config = TreeBoostHedgeConfig::for_timeframe(timeframe, false);
        Self {
            name: "TreeBoost_GBDT_Strategy".to_string(),
            model,
            cached_predictions: None,
            decision: TreeBoostSignalDecision::new(),
            timeframe,
            hedge_config,
            hedge_long_active: false,
            hedge_short_active: false,
            last_eval_idx: None,
            last_eval_signal: None,
            last_hedge_idx: None,
            last_hedge_signal: None,
        }
    }

    pub fn with_timeframe(mut self, timeframe: Option<Timeframe>) -> Self {
        self.timeframe = timeframe;
        self.hedge_config = TreeBoostHedgeConfig::for_timeframe(timeframe, self.hedge_config.enabled);
        self
    }

    pub fn with_hedge_enabled(mut self, enabled: bool) -> Self {
        self.hedge_config.enabled = enabled;
        self
    }

    pub fn with_hedge_config(mut self, hedge_config: TreeBoostHedgeConfig) -> Self {
        self.hedge_config = hedge_config;
        self
    }

    pub fn set_hedge_config(&mut self, hedge_config: TreeBoostHedgeConfig) {
        self.hedge_config = hedge_config;
        self.reset_runtime_state();
    }

    pub fn hedge_config(&self) -> TreeBoostHedgeConfig {
        self.hedge_config
    }

    pub fn timeframe(&self) -> Option<Timeframe> {
        self.timeframe
    }

    pub fn last_direction(&self) -> SignalDirection {
        self.decision.last_direction()
    }

    pub fn set_cached_predictions(&mut self, preds: Vec<f64>) {
        self.cached_predictions = Some(preds);
        self.reset_runtime_state();
    }

    pub fn clear_cache(&mut self) {
        self.cached_predictions = None;
        self.reset_runtime_state();
    }

    fn reset_runtime_state(&mut self) {
        self.decision.reset();
        self.hedge_long_active = false;
        self.hedge_short_active = false;
        self.last_eval_idx = None;
        self.last_eval_signal = None;
        self.last_hedge_idx = None;
        self.last_hedge_signal = None;
    }

    fn predict_at(&self, history: &[Kline], current_idx: usize, is_h4: bool) -> Option<f64> {
        if let Some(ref preds) = self.cached_predictions {
            if current_idx < preds.len() {
                return Some(preds[current_idx]);
            }
        }
        let feats = if is_h4 {
            H4FeatureMatrix::extract_single_vector(history, current_idx)
        } else {
            FeatureMatrix::extract_single_vector(history, current_idx)
        };
        feats.map(|f| self.model.predict_single(&f).unwrap_or(0.0))
    }

    fn primary_size_multiplier(&self, close: f64, atr: Option<f64>) -> f64 {
        match (self.hedge_config.vol_target_pct, atr) {
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
                    (vt / atr_pct).clamp(0.35, 1.25)
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_mr_active(
        was_active: bool,
        bb_z: Option<f64>,
        rsi: Option<f64>,
        open: f64,
        close: f64,
        is_long: bool,
        entry_z: f64,
        rsi_thresh: f64,
        require_rejection: bool,
    ) -> bool {
        match bb_z {
            Some(z) if was_active => {
                if is_long {
                    z < 0.0
                } else {
                    z > 0.0
                }
            }
            Some(z) => rsi.map_or(false, |r| {
                let rejection_ok = if require_rejection {
                    if is_long {
                        close > open
                    } else {
                        close < open
                    }
                } else {
                    true
                };
                if is_long {
                    z <= -entry_z && r <= rsi_thresh && rejection_ok
                } else {
                    z >= entry_z && r >= rsi_thresh && rejection_ok
                }
            }),
            None => false,
        }
    }

    fn mr_leg(
        &self,
        is_long: bool,
        close: f64,
        atr: Option<f64>,
        bb_z: Option<f64>,
        entry_z: f64,
    ) -> HedgeLegAction {
        if self.hedge_config.hedge_size_mult <= 0.0 {
            return HedgeLegAction::Close;
        }
        let sl = if self.hedge_config.hedge_sl_atr_mult > 0.0 {
            atr.map(|a| {
                if is_long {
                    close - self.hedge_config.hedge_sl_atr_mult * a
                } else {
                    close + self.hedge_config.hedge_sl_atr_mult * a
                }
            })
            .filter(|&p| p > 0.0)
        } else {
            None
        };
        let tp = if self.hedge_config.hedge_tp_atr_mult > 0.0 {
            atr.map(|a| {
                if is_long {
                    close + self.hedge_config.hedge_tp_atr_mult * a
                } else {
                    close - self.hedge_config.hedge_tp_atr_mult * a
                }
            })
            .filter(|&p| p > 0.0)
        } else {
            None
        };
        let h_strength = bb_z
            .map(|zv| (zv.abs() / (entry_z * 2.0)).clamp(0.25, 1.0))
            .unwrap_or(0.5);
        HedgeLegAction::open_or_keep_with_regime(
            h_strength,
            sl,
            tp,
            self.hedge_config.hedge_size_mult,
            TradeRegime::MeanReversionHedge,
        )
    }
}

impl Strategy for TreeBoostStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn clear_cache(&mut self) {
        TreeBoostStrategy::clear_cache(self);
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            return Signal::flat(0);
        }
        if self.last_eval_idx.map_or(false, |prev| current_idx < prev) {
            self.reset_runtime_state();
        }
        if self.last_eval_idx == Some(current_idx) {
            if let Some(ref sig) = self.last_eval_signal {
                return sig.clone();
            }
        }

        let ts = history[current_idx].timestamp;
        let is_h4 = matches!(self.timeframe, Some(Timeframe::H4));
        let warmup_bars = if is_h4 { H4FeatureMatrix::WARMUP_BARS } else { 35 };

        let sig = if !self.model.is_trained() || current_idx < warmup_bars {
            self.decision.reset();
            Signal::flat(ts)
        } else {
            match self.predict_at(history, current_idx, is_h4) {
                Some(predicted_return) => {
                    let long_th = self.model.config().long_threshold;
                    let short_th = self.model.config().short_threshold;
                    let mut step_sig = self.decision.step_with_params(
                        history,
                        current_idx,
                        predicted_return,
                        long_th,
                        short_th,
                        self.hedge_config.primary_sl_atr_mult,
                        self.hedge_config.primary_tp_atr_mult,
                        self.hedge_config.trailing_atr_mult,
                        self.hedge_config.min_hold_bars,
                        self.hedge_config.hysteresis_frac,
                    );
                    if let (Some(pct), Some(sl_price)) = (
                        self.hedge_config
                            .max_sl_pct
                            .filter(|&p| p.is_finite() && p > 0.0),
                        step_sig.stop_loss_price,
                    ) {
                        let close = history[current_idx].close;
                        if close.is_finite() && close > 0.0 {
                            let max_dist = close * pct;
                            let clamped = match step_sig.direction {
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
                                step_sig.stop_loss_price = Some(level);
                            }
                        }
                    }
                    step_sig
                }
                None => {
                    self.decision.reset();
                    Signal::flat(ts)
                }
            }
        };

        self.last_eval_idx = Some(current_idx);
        self.last_eval_signal = Some(sig.clone());
        sig
    }

    fn evaluate_hedge(&mut self, history: &[Kline], current_idx: usize) -> HedgeSignal {
        if !self.hedge_config.enabled {
            let primary = self.evaluate(history, current_idx);
            let mut hedge_sig = HedgeSignal::from_net_signal(primary.clone());
            if self.hedge_config.vol_target_pct.is_some() && current_idx < history.len() {
                let close = history[current_idx].close;
                let atr = TreeBoostSignalDecision::causal_atr_14(history, current_idx);
                let mult = self.primary_size_multiplier(close, atr);
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
            return hedge_sig;
        }
        if current_idx >= history.len() {
            return HedgeSignal::flat(0);
        }
        if self.last_hedge_idx.map_or(false, |prev| current_idx < prev) {
            self.reset_runtime_state();
        }
        if self.last_hedge_idx == Some(current_idx) {
            return self
                .last_hedge_signal
                .unwrap_or_else(|| HedgeSignal::hold(history[current_idx].timestamp));
        }

        let primary = self.evaluate(history, current_idx);
        let ts = history[current_idx].timestamp;
        let open = history[current_idx].open;
        let close = history[current_idx].close;
        let atr = TreeBoostSignalDecision::causal_atr_14(history, current_idx);
        let primary_mult = self.primary_size_multiplier(close, atr);
        let bb_z = TreeBoostMrIndicators::bb_zscore(history, current_idx, 20);
        let rsi = TreeBoostMrIndicators::rsi_14(history, current_idx);
        let entry_z = self.hedge_config.hedge_entry_z.max(1e-6);
        let req_rej = self.hedge_config.mr_require_rejection;

        let hedge_sig = match primary.direction {
            SignalDirection::Long => {
                self.hedge_long_active = false;
                self.hedge_short_active = Self::update_mr_active(
                    self.hedge_short_active,
                    bb_z,
                    rsi,
                    open,
                    close,
                    false,
                    entry_z,
                    self.hedge_config.mr_rsi_high,
                    req_rej,
                );
                let long = HedgeLegAction::open_or_keep(
                    primary.strength,
                    primary.stop_loss_price,
                    primary.take_profit_price,
                    primary_mult,
                );
                let short = if self.hedge_short_active {
                    self.mr_leg(false, close, atr, bb_z, entry_z)
                } else {
                    HedgeLegAction::Close
                };
                HedgeSignal {
                    timestamp: ts,
                    long,
                    short,
                }
            }
            SignalDirection::Short => {
                self.hedge_short_active = false;
                self.hedge_long_active = Self::update_mr_active(
                    self.hedge_long_active,
                    bb_z,
                    rsi,
                    open,
                    close,
                    true,
                    entry_z,
                    self.hedge_config.mr_rsi_low,
                    req_rej,
                );
                let short = HedgeLegAction::open_or_keep(
                    primary.strength,
                    primary.stop_loss_price,
                    primary.take_profit_price,
                    primary_mult,
                );
                let long = if self.hedge_long_active {
                    self.mr_leg(true, close, atr, bb_z, entry_z)
                } else {
                    HedgeLegAction::Close
                };
                HedgeSignal {
                    timestamp: ts,
                    long,
                    short,
                }
            }
            SignalDirection::Flat => {
                self.hedge_long_active = Self::update_mr_active(
                    self.hedge_long_active,
                    bb_z,
                    rsi,
                    open,
                    close,
                    true,
                    entry_z,
                    self.hedge_config.mr_rsi_low,
                    req_rej,
                );
                self.hedge_short_active = Self::update_mr_active(
                    self.hedge_short_active,
                    bb_z,
                    rsi,
                    open,
                    close,
                    false,
                    entry_z,
                    self.hedge_config.mr_rsi_high,
                    req_rej,
                );
                if self.hedge_long_active && self.hedge_short_active {
                    if bb_z.unwrap_or(0.0) < 0.0 {
                        self.hedge_short_active = false;
                    } else {
                        self.hedge_long_active = false;
                    }
                }
                let long = if self.hedge_long_active {
                    self.mr_leg(true, close, atr, bb_z, entry_z)
                } else {
                    HedgeLegAction::Close
                };
                let short = if self.hedge_short_active {
                    self.mr_leg(false, close, atr, bb_z, entry_z)
                } else {
                    HedgeLegAction::Close
                };
                HedgeSignal {
                    timestamp: ts,
                    long,
                    short,
                }
            }
        };

        self.last_hedge_idx = Some(current_idx);
        self.last_hedge_signal = Some(hedge_sig);
        hedge_sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_config::ModelConfig;

    fn sample_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64) * 0.1;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p,
                    volume: 100.0,
                }
            })
            .collect()
    }

    fn sample_trained_model() -> TreeBoostModel {
        let mut model = TreeBoostModel::new(ModelConfig::default());
        let n = 20;
        let f_dim = crate::features::FeatureMatrix::num_features();
        let x: Vec<f64> = (0..n * f_dim).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        model.train_from_matrix(&x, &y, f_dim).unwrap();
        model
    }

    #[test]
    fn test_treeboost_strategy_warmup_and_untrained() {
        let model = TreeBoostModel::new(ModelConfig::default());
        let mut strat = TreeBoostStrategy::new(model);
        let klines = sample_klines(40);
        let sig = strat.evaluate(&klines, 10);
        assert_eq!(sig.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);
    }

    #[test]
    fn test_treeboost_strategy_long_hysteresis() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        preds[37] = 0.001;
        preds[38] = 0.001;
        preds[39] = -0.0001;
        preds[40] = 0.001;
        strat.set_cached_predictions(preds);

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Long);
        assert!(sig36.stop_loss_price.is_some());
        assert!(sig36.stop_loss_price.unwrap() < klines[36].close);
        assert_eq!(strat.last_direction(), SignalDirection::Long);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Long);
        let expected_strength = (0.001 / 0.002f64).clamp(0.25, 1.0);
        assert!((sig37.strength - expected_strength).abs() < 1e-6);
        assert_eq!(strat.last_direction(), SignalDirection::Long);

        let sig38 = strat.evaluate(&klines, 38);
        assert_eq!(sig38.direction, SignalDirection::Long);
        assert_eq!(strat.last_direction(), SignalDirection::Long);

        let sig39 = strat.evaluate(&klines, 39);
        assert_eq!(sig39.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);

        let sig40 = strat.evaluate(&klines, 40);
        assert_eq!(sig40.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);
    }

    #[test]
    fn test_treeboost_strategy_short_hysteresis() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = -0.005;
        preds[37] = -0.001;
        preds[38] = -0.001;
        preds[39] = 0.0001;
        preds[40] = -0.001;
        strat.set_cached_predictions(preds);

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Short);
        assert!(sig36.stop_loss_price.is_some());
        assert!(sig36.stop_loss_price.unwrap() > klines[36].close);
        assert_eq!(strat.last_direction(), SignalDirection::Short);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Short);
        let expected_strength = (0.001 / 0.002f64).clamp(0.25, 1.0);
        assert!((sig37.strength - expected_strength).abs() < 1e-6);
        assert_eq!(strat.last_direction(), SignalDirection::Short);

        let sig38 = strat.evaluate(&klines, 38);
        assert_eq!(sig38.direction, SignalDirection::Short);
        assert_eq!(strat.last_direction(), SignalDirection::Short);

        let sig39 = strat.evaluate(&klines, 39);
        assert_eq!(sig39.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);

        let sig40 = strat.evaluate(&klines, 40);
        assert_eq!(sig40.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);
    }

    #[test]
    fn test_treeboost_strategy_cache_reset_clears_direction() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        strat.set_cached_predictions(preds.clone());

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Long);
        assert_eq!(strat.last_direction(), SignalDirection::Long);

        strat.clear_cache();
        assert_eq!(strat.last_direction(), SignalDirection::Flat);

        preds[37] = 0.001;
        strat.set_cached_predictions(preds);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);

        let sig37 = strat.evaluate(&klines, 37);
        assert_eq!(sig37.direction, SignalDirection::Flat);
        assert_eq!(strat.last_direction(), SignalDirection::Flat);
    }

    #[test]
    fn test_treeboost_strategy_min_hold_bars_and_strength_floor() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model).with_timeframe(Some(Timeframe::H1));
        assert_eq!(strat.hedge_config().min_hold_bars, 8);

        let klines = sample_klines(50);
        let mut preds = vec![0.0; 50];
        preds[36] = 0.0021;
        for idx in 37..=44 {
            preds[idx] = -0.005;
        }
        strat.set_cached_predictions(preds);

        let sig36 = strat.evaluate(&klines, 36);
        assert_eq!(sig36.direction, SignalDirection::Long);
        assert!((sig36.strength - 0.35).abs() < 1e-6);

        for idx in 37..=43 {
            let sig = strat.evaluate(&klines, idx);
            assert_eq!(sig.direction, SignalDirection::Long);
            assert!((sig.strength - 0.25).abs() < 1e-6);
        }

        let sig44 = strat.evaluate(&klines, 44);
        assert_eq!(sig44.direction, SignalDirection::Short);
    }

    #[test]
    fn test_treeboost_evaluate_hedge_disabled_delegates_to_net_signal() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model).with_hedge_enabled(false);
        let klines = sample_klines(45);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        strat.set_cached_predictions(preds);

        let h_sig = strat.evaluate_hedge(&klines, 36);
        assert!(matches!(h_sig.long, HedgeLegAction::OpenOrKeep { size_multiplier, .. } if (size_multiplier - 1.0).abs() < 1e-9));
        assert_eq!(h_sig.short, HedgeLegAction::Close);
    }

    #[test]
    fn test_treeboost_evaluate_hedge_secondary_counter_trend_lifecycle() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model)
            .with_timeframe(Some(Timeframe::D1))
            .with_hedge_enabled(true);
        let mut klines = sample_klines(45);
        for i in 1..36 {
            let prev = klines[i - 1].close;
            let delta = if i % 2 == 0 { 0.002 } else { -0.001 };
            klines[i].close = prev * (1.0 + delta);
            klines[i].open = prev;
            klines[i].high = klines[i].close.max(prev) + 0.3;
            klines[i].low = klines[i].close.min(prev) - 0.3;
        }
        klines[36].close = klines[35].close * 1.001;
        klines[37].close = klines[36].close * 1.12;
        klines[37].open = klines[37].close + 0.3;
        klines[37].high = klines[37].open + 0.5;
        klines[38].close = klines[37].close * 0.99;
        klines[39].close = klines[35].close * 0.96;
        klines[39].low = klines[39].close - 0.5;
        klines[40].close = klines[39].close * 0.999;
        klines[40].open = klines[39].close;
        klines[40].high = klines[40].open + 0.3;
        klines[40].low = klines[40].close - 0.3;

        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        preds[37] = 0.003;
        preds[38] = 0.003;
        preds[39] = 0.003;
        preds[40] = -0.005;
        strat.set_cached_predictions(preds);

        let s36 = strat.evaluate_hedge(&klines, 36);
        assert!(matches!(s36.long, HedgeLegAction::OpenOrKeep { .. }));
        assert_eq!(s36.short, HedgeLegAction::Close);

        let s37 = strat.evaluate_hedge(&klines, 37);
        assert!(matches!(s37.long, HedgeLegAction::OpenOrKeep { .. }));
        assert!(matches!(
            s37.short,
            HedgeLegAction::OpenOrKeep {
                stop_loss_price: Some(_),
                take_profit_price: Some(_),
                size_multiplier,
                ..
            } if (size_multiplier - 0.40).abs() < 1e-9
        ));

        let s38 = strat.evaluate_hedge(&klines, 38);
        assert!(matches!(s38.short, HedgeLegAction::OpenOrKeep { .. }));

        let s39 = strat.evaluate_hedge(&klines, 39);
        assert!(matches!(s39.long, HedgeLegAction::OpenOrKeep { .. }));
        assert_eq!(s39.short, HedgeLegAction::Close);

        let s40 = strat.evaluate_hedge(&klines, 40);
        assert_eq!(s40.long, HedgeLegAction::Close);
        assert!(matches!(s40.short, HedgeLegAction::OpenOrKeep { .. }));
    }

    #[test]
    fn test_treeboost_evaluate_hedge_flat_primary_allows_mr_legs() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model)
            .with_timeframe(Some(Timeframe::D1))
            .with_hedge_enabled(true);
        let mut klines = sample_klines(45);
        for i in 1..36 {
            let prev = klines[i - 1].close;
            let delta = if i % 2 == 0 { 0.002 } else { -0.001 };
            klines[i].close = prev * (1.0 + delta);
            klines[i].open = prev;
            klines[i].high = klines[i].close.max(prev) + 0.3;
            klines[i].low = klines[i].close.min(prev) - 0.3;
        }
        klines[36].close = klines[35].close;
        klines[37].close = klines[36].close * 0.88;
        klines[37].open = klines[37].close - 0.3;
        klines[37].low = klines[37].open - 0.5;
        klines[38].close = klines[37].close * 1.01;
        klines[39].close = klines[35].close * 1.02;

        strat.set_cached_predictions(vec![0.0; 45]);

        let s37 = strat.evaluate_hedge(&klines, 37);
        assert!(matches!(
            s37.long,
            HedgeLegAction::OpenOrKeep {
                stop_loss_price: Some(_),
                take_profit_price: Some(_),
                size_multiplier,
                ..
            } if (size_multiplier - 0.40).abs() < 1e-9
        ));
        assert_eq!(s37.short, HedgeLegAction::Close);

        let s38 = strat.evaluate_hedge(&klines, 38);
        assert!(matches!(s38.long, HedgeLegAction::OpenOrKeep { .. }));
        assert_eq!(s38.short, HedgeLegAction::Close);

        let s39 = strat.evaluate_hedge(&klines, 39);
        assert_eq!(s39.long, HedgeLegAction::Close);
        assert_eq!(s39.short, HedgeLegAction::Close);
    }

    #[test]
    fn test_treeboost_evaluate_hedge_requires_rejection_candle() {
        let model = sample_trained_model();
        let mut strat = TreeBoostStrategy::new(model)
            .with_timeframe(Some(Timeframe::H4))
            .with_hedge_enabled(true);
        let mut klines = sample_klines(45);
        for i in 1..36 {
            let prev = klines[i - 1].close;
            let delta = if i % 2 == 0 { 0.002 } else { -0.001 };
            klines[i].close = prev * (1.0 + delta);
            klines[i].open = prev;
            klines[i].high = klines[i].close.max(prev) + 0.3;
            klines[i].low = klines[i].close.min(prev) - 0.3;
        }
        klines[36].close = klines[35].close;
        klines[37].open = klines[36].close;
        klines[37].close = klines[36].close * 0.85;
        klines[37].low = klines[37].close - 0.5;

        strat.set_cached_predictions(vec![0.0; 45]);
        let s37 = strat.evaluate_hedge(&klines, 37);
        assert_eq!(s37.long, HedgeLegAction::Close);
        assert_eq!(s37.short, HedgeLegAction::Close);
    }

    #[test]
    fn test_treeboost_strategy_max_sl_pct_clamps_primary_sl() {
        let model = sample_trained_model();
        let mut klines = sample_klines(45);
        for k in &mut klines {
            k.high = k.close + 5.0;
            k.low = k.close - 5.0;
        }
        let mut cfg = TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H4), false);
        cfg.primary_sl_atr_mult = 2.2;
        cfg.max_sl_pct = Some(0.028);

        let mut strat_clamped = TreeBoostStrategy::new(model).with_hedge_config(cfg);
        let mut preds = vec![0.0; 45];
        preds[36] = 0.005;
        preds[42] = -0.005;
        strat_clamped.set_cached_predictions(preds.clone());

        let sig_long = strat_clamped.evaluate(&klines, 36);
        assert_eq!(sig_long.direction, SignalDirection::Long);
        let c36 = klines[36].close;
        assert!((sig_long.stop_loss_price.unwrap() - c36 * (1.0 - 0.028)).abs() < 1e-6);

        let mut cfg_unclamped = cfg;
        cfg_unclamped.max_sl_pct = None;
        let mut strat_raw = TreeBoostStrategy::new(sample_trained_model()).with_hedge_config(cfg_unclamped);
        strat_raw.set_cached_predictions(preds);
        let sig_raw_long = strat_raw.evaluate(&klines, 36);
        assert!(sig_raw_long.stop_loss_price.unwrap() < sig_long.stop_loss_price.unwrap());

        for idx in 37..=41 {
            strat_clamped.evaluate(&klines, idx);
        }
        let sig_short = strat_clamped.evaluate(&klines, 42);
        assert_eq!(sig_short.direction, SignalDirection::Short);
        let c42 = klines[42].close;
        assert!((sig_short.stop_loss_price.unwrap() - c42 * (1.0 + 0.028)).abs() < 1e-6);
    }
}

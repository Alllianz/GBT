use rayon::prelude::*;
use crate::backtest::{compute_stagnation_stats, BacktestConfig, BacktestEngine, BacktestResult};
use crate::features::H4FeatureMatrix;
use crate::models::{TreeBoostHedgeConfig, TreeBoostSignalDecision};
use crate::types::{FitnessMetric, Kline, Timeframe};
use super::adaptive_threshold_grid::AdaptiveThresholdGrid;
use super::oof_test_strategy::OofTestStrategy;

pub struct OofCalibrator;

impl OofCalibrator {
    pub const CANDIDATE_THRESHOLDS: [f64; 9] = [
        0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070, 0.0100,
    ];

    fn min_trade_thresholds(num_klines: usize) -> (usize, usize) {
        let min_trades = if num_klines >= 500 {
            (num_klines / 40).clamp(15, 180)
        } else {
            5
        };
        let fallback_min_trades = if num_klines >= 500 {
            (min_trades / 2).max(10)
        } else {
            1
        };
        (min_trades, fallback_min_trades)
    }

    fn min_trade_thresholds_for_timeframe(
        num_klines: usize,
        timeframe: Option<Timeframe>,
    ) -> (usize, usize) {
        if num_klines < 500 {
            return (5, 1);
        }
        match timeframe {
            Some(Timeframe::D1 | Timeframe::D3 | Timeframe::W1) => {
                let min_trades = (num_klines / 40).clamp(25, 180);
                let fallback = (min_trades / 2).max(12);
                (min_trades, fallback)
            }
            Some(_) => {
                let min_trades = (num_klines / 160).clamp(60, 95);
                let fallback_min_trades = (min_trades / 2).max(25);
                (min_trades, fallback_min_trades)
            }
            None => Self::min_trade_thresholds(num_klines),
        }
    }

    fn best_in_tier<T: Copy>(
        candidates: &[(T, f64, usize)],
        trade_floor: usize,
    ) -> Option<(T, f64)> {
        let mut best: Option<T> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &(val, score, trades) in candidates {
            if trades >= trade_floor && score.is_finite() && score > best_score {
                best_score = score;
                best = Some(val);
            }
        }
        best.map(|val| (val, best_score))
    }

    fn select_best_candidate_for_timeframe<T: Copy>(
        candidates: &[(T, f64, usize)],
        num_klines: usize,
        timeframe: Option<Timeframe>,
        default_val: T,
    ) -> (T, f64) {
        let (min_trades, fallback_min_trades) =
            Self::min_trade_thresholds_for_timeframe(num_klines, timeframe);

        for trade_floor in [min_trades, fallback_min_trades, 1] {
            if let Some(hit) = Self::best_in_tier(candidates, trade_floor) {
                return hit;
            }
        }

        let fallback_val = candidates
            .iter()
            .filter(|&&(_, score, _)| score.is_finite())
            .max_by(|a, b| {
                a.2.cmp(&b.2)
                    .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            })
            .or_else(|| candidates.first())
            .map(|&(val, _, _)| val)
            .unwrap_or(default_val);

        (fallback_val, 0.0)
    }

    #[cfg(test)]
    fn select_best_candidate<T: Copy>(
        candidates: &[(T, f64, usize)],
        num_klines: usize,
        default_val: T,
    ) -> (T, f64) {
        Self::select_best_candidate_for_timeframe(candidates, num_klines, None, default_val)
    }

    fn score_backtest_result(res: &BacktestResult, fitness_metric: FitnessMetric) -> f64 {
        let base_fit = fitness_metric.evaluate(&res.metrics);
        if fitness_metric == FitnessMetric::CompositeAllianz && res.metrics.net_profit > 0.0 {
            let stag = compute_stagnation_stats(&res.equity_curve);
            base_fit
                - 0.35 * stag.max_stagnation_ratio
                - 0.25 * stag.total_underwater_ratio
                - 0.25 * (res.metrics.max_drawdown_pct / 100.0).clamp(0.0, 1.0)
        } else {
            base_fit
        }
    }

    pub fn calibrate_symmetric(
        oof_predictions: &[f64],
        klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        let warmup_bars = match base_config.timeframe {
            Some(Timeframe::H4) => H4FeatureMatrix::WARMUP_BARS,
            _ => 60,
        };
        if oof_predictions.is_empty() || klines.len() < 50.max(warmup_bars.saturating_add(2)) {
            return (0.002, -0.002, 0.0);
        }

        let active_oof = if warmup_bars < oof_predictions.len() {
            &oof_predictions[warmup_bars..]
        } else {
            oof_predictions
        };
        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(active_oof);
        let hedge_cfg =
            TreeBoostHedgeConfig::for_timeframe(base_config.timeframe, base_config.hedge_mode);
        let mut eval_cfg = base_config.clone();
        eval_cfg.hedge_mode = base_config.hedge_mode || hedge_cfg.vol_target_pct.is_some();

        let candidates: Vec<(f64, f64, usize)> = grid
            .into_par_iter()
            .map(|th| {
                let mut strat = OofTestStrategy {
                    predictions: oof_predictions,
                    long_th: th,
                    short_th: -th,
                    warmup_bars,
                    hedge_config: hedge_cfg,
                    decision: TreeBoostSignalDecision::new(),
                };
                let engine = BacktestEngine::new(eval_cfg.clone());
                match engine.run_with_warmup(&mut strat, klines, warmup_bars) {
                    Ok(res) => {
                        let score = Self::score_backtest_result(&res, fitness_metric);
                        (th, score, res.metrics.total_trades)
                    }
                    Err(_) => (th, f64::NEG_INFINITY, 0),
                }
            })
            .collect();

        let active_bars = klines.len().saturating_sub(warmup_bars);
        let (final_th, final_score) = Self::select_best_candidate_for_timeframe(
            &candidates,
            active_bars,
            base_config.timeframe,
            0.002,
        );

        (final_th, -final_th, final_score)
    }

    pub fn calibrate_asymmetric(
        oof_predictions: &[f64],
        klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        let warmup_bars = match base_config.timeframe {
            Some(Timeframe::H4) => H4FeatureMatrix::WARMUP_BARS,
            _ => 60,
        };
        Self::calibrate_asymmetric_with_warmup(
            oof_predictions,
            klines,
            fitness_metric,
            base_config,
            warmup_bars,
        )
    }

    pub fn calibrate_asymmetric_with_warmup(
        oof_predictions: &[f64],
        klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
        warmup_bars: usize,
    ) -> (f64, f64, f64) {
        if oof_predictions.is_empty() || klines.len() < 50.max(warmup_bars.saturating_add(2)) {
            return (0.002, -0.002, 0.0);
        }

        let active_oof = if warmup_bars < oof_predictions.len() {
            &oof_predictions[warmup_bars..]
        } else {
            oof_predictions
        };
        let (long_grid, short_grid) =
            AdaptiveThresholdGrid::compute_asymmetric_grids(active_oof);
        let hedge_cfg =
            TreeBoostHedgeConfig::for_timeframe(base_config.timeframe, base_config.hedge_mode);
        let mut eval_cfg = base_config.clone();
        eval_cfg.hedge_mode = base_config.hedge_mode || hedge_cfg.vol_target_pct.is_some();

        let mut all_pairs = Vec::with_capacity(long_grid.len() * short_grid.len());
        let mut pairs = Vec::with_capacity(long_grid.len() * short_grid.len());
        for &l in &long_grid {
            for &s in &short_grid {
                all_pairs.push((l, s));
                let s_abs = s.abs();
                if !(l / s_abs > 2.0 || s_abs / l > 2.0) {
                    pairs.push((l, s));
                }
            }
        }
        let pairs = if pairs.is_empty() { all_pairs } else { pairs };

        let candidates: Vec<((f64, f64), f64, usize)> = pairs
            .into_par_iter()
            .map(|(l_th, s_th)| {
                let mut strat = OofTestStrategy {
                    predictions: oof_predictions,
                    long_th: l_th,
                    short_th: s_th,
                    warmup_bars,
                    hedge_config: hedge_cfg,
                    decision: TreeBoostSignalDecision::new(),
                };
                let engine = BacktestEngine::new(eval_cfg.clone());
                match engine.run_with_warmup(&mut strat, klines, warmup_bars) {
                    Ok(res) => {
                        let score = Self::score_backtest_result(&res, fitness_metric);
                        ((l_th, s_th), score, res.metrics.total_trades)
                    }
                    Err(_) => ((l_th, s_th), f64::NEG_INFINITY, 0),
                }
            })
            .collect();

        let active_bars = klines.len().saturating_sub(warmup_bars);
        let (final_pair, final_score) = Self::select_best_candidate_for_timeframe(
            &candidates,
            active_bars,
            base_config.timeframe,
            (0.002, -0.002),
        );

        (final_pair.0, final_pair.1, final_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::Strategy;
    use crate::types::SignalDirection;

    #[test]
    fn test_oof_empty_predictions_fallback() {
        let (long_th, short_th, score) = OofCalibrator::calibrate_symmetric(
            &[],
            &[],
            FitnessMetric::SharpeRatio,
            &BacktestConfig::default(),
        );
        assert_eq!(long_th, 0.002);
        assert_eq!(short_th, -0.002);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_oof_asymmetric_calibration_adapts_to_skewed_distribution() {
        let n = 150;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.1).sin() * 5.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 1.0,
                    low: p - 1.0,
                    close: p + 0.5,
                    volume: 1000.0,
                }
            })
            .collect();

        let preds: Vec<f64> = (0..n)
            .map(|i| if i % 12 < 6 { 0.0006 } else { -0.0014 })
            .collect();

        let (long_th, short_th, _score) = OofCalibrator::calibrate_asymmetric(
            &preds,
            &klines,
            FitnessMetric::SharpeRatio,
            &BacktestConfig::default(),
        );

        assert!(long_th > 0.0, "long_th debe ser positivo");
        assert!(short_th < 0.0, "short_th debe ser negativo");
    }

    #[test]
    fn test_oof_min_trade_thresholds_scaling() {
        assert_eq!(OofCalibrator::min_trade_thresholds(150), (5, 1));
        assert_eq!(OofCalibrator::min_trade_thresholds(499), (5, 1));
        assert_eq!(OofCalibrator::min_trade_thresholds(500), (15, 10));
        assert_eq!(OofCalibrator::min_trade_thresholds(900), (22, 11));
        assert_eq!(OofCalibrator::min_trade_thresholds(1800), (45, 22));
        assert_eq!(OofCalibrator::min_trade_thresholds(2400), (60, 30));
        assert_eq!(OofCalibrator::min_trade_thresholds(3000), (75, 37));
    }

    #[test]
    fn test_oof_selects_sub_0005_candidates_without_floor_filter() {
        let candidates = vec![
            (0.00015, 2.5, 30),
            (0.0005, 1.2, 20),
            (0.0010, 4.0, 12),
        ];
        let (th_large, score_large) =
            OofCalibrator::select_best_candidate(&candidates, 800, 0.002);
        assert_eq!(th_large, 0.00015);
        assert!((score_large - 2.5).abs() < 1e-9);

        let (th_small, score_small) =
            OofCalibrator::select_best_candidate(&candidates, 200, 0.002);
        assert_eq!(th_small, 0.0010);
        assert!((score_small - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_oof_level2_fallback_when_min_trades_unmet() {
        let large_fallback = vec![
            (0.0002, 2.1, 12),
            (0.0005, 3.5, 4),
            (0.0010, 1.8, 10),
        ];
        let (th_large, score_large) =
            OofCalibrator::select_best_candidate(&large_fallback, 800, 0.002);
        assert_eq!(th_large, 0.0002);
        assert!((score_large - 2.1).abs() < 1e-9);

        let small_fallback = vec![
            (0.0001, 1.7, 2),
            (0.0005, 3.0, 0),
        ];
        let (th_small, score_small) =
            OofCalibrator::select_best_candidate(&small_fallback, 200, 0.002);
        assert_eq!(th_small, 0.0001);
        assert!((score_small - 1.7).abs() < 1e-9);
    }

    #[test]
    fn test_oof_symmetric_calibration_accepts_small_magnitude_predictions() {
        let n = 600;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.25).sin() * 3.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.5,
                    low: p - 0.5,
                    close: p + 0.2,
                    volume: 1000.0,
                }
            })
            .collect();

        let preds: Vec<f64> = (0..n)
            .map(|i| match i % 16 {
                0..=5 => 0.00025,
                8..=13 => -0.00025,
                _ => 0.0,
            })
            .collect();

        let (long_th, short_th, _score) = OofCalibrator::calibrate_symmetric(
            &preds,
            &klines,
            FitnessMetric::SharpeRatio,
            &BacktestConfig::default(),
        );

        assert!(long_th > 0.0 && long_th < 0.0005);
        assert!(short_th < 0.0 && short_th > -0.0005);
    }

    #[test]
    fn test_oof_strategy_hysteresis_and_min_hold_shared_rules() {
        let klines: Vec<Kline> = (0..115)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 100.0,
            })
            .collect();

        let mut preds_h4 = vec![0.0; 115];
        preds_h4[100] = 0.005;
        preds_h4[101] = 0.0012;
        preds_h4[102] = 0.0008;
        preds_h4[103] = 0.0008;
        preds_h4[104] = 0.0008;
        preds_h4[105] = 0.0008;
        preds_h4[106] = 0.0008;
        preds_h4[107] = -0.005;
        preds_h4[108] = -0.0012;
        preds_h4[109] = -0.0008;
        preds_h4[110] = -0.0008;
        preds_h4[111] = -0.0008;
        preds_h4[112] = -0.0008;
        preds_h4[113] = -0.0008;

        let mut strat_h4 = OofTestStrategy {
            predictions: &preds_h4,
            long_th: 0.004,
            short_th: -0.004,
            warmup_bars: 100,
            hedge_config: TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H4), false),
            decision: TreeBoostSignalDecision::new(),
        };
        let s100 = strat_h4.evaluate(&klines, 100);
        assert_eq!(s100.direction, SignalDirection::Long);
        assert!(s100.stop_loss_price.is_some());
        for idx in 101..=105 {
            assert_eq!(strat_h4.evaluate(&klines, idx).direction, SignalDirection::Long);
        }
        assert_eq!(strat_h4.evaluate(&klines, 106).direction, SignalDirection::Flat);
        let s107 = strat_h4.evaluate(&klines, 107);
        assert_eq!(s107.direction, SignalDirection::Short);
        assert!(s107.stop_loss_price.is_some());
        for idx in 108..=112 {
            assert_eq!(strat_h4.evaluate(&klines, idx).direction, SignalDirection::Short);
        }
        assert_eq!(strat_h4.evaluate(&klines, 113).direction, SignalDirection::Flat);

        let mut preds_d1 = vec![0.0; 115];
        preds_d1[35] = 0.005;
        preds_d1[36] = 0.0012;
        preds_d1[37] = 0.0008;
        preds_d1[38] = 0.0008;
        preds_d1[39] = -0.005;
        preds_d1[40] = -0.0012;
        preds_d1[41] = -0.0008;
        preds_d1[42] = -0.0008;

        let mut strat_d1 = OofTestStrategy {
            predictions: &preds_d1,
            long_th: 0.004,
            short_th: -0.004,
            warmup_bars: 35,
            hedge_config: TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::D1), false),
            decision: TreeBoostSignalDecision::new(),
        };
        assert_eq!(strat_d1.evaluate(&klines, 35).direction, SignalDirection::Long);
        assert_eq!(strat_d1.evaluate(&klines, 36).direction, SignalDirection::Long);
        assert_eq!(strat_d1.evaluate(&klines, 37).direction, SignalDirection::Long);
        assert_eq!(strat_d1.evaluate(&klines, 38).direction, SignalDirection::Flat);
        assert_eq!(strat_d1.evaluate(&klines, 39).direction, SignalDirection::Short);
        assert_eq!(strat_d1.evaluate(&klines, 40).direction, SignalDirection::Short);
        assert_eq!(strat_d1.evaluate(&klines, 41).direction, SignalDirection::Short);
        assert_eq!(strat_d1.evaluate(&klines, 42).direction, SignalDirection::Flat);
    }

    #[test]
    fn test_oof_min_trade_thresholds_timeframe_floors() {
        let (d1_min, _) =
            OofCalibrator::min_trade_thresholds_for_timeframe(600, Some(Timeframe::D1));
        assert!(d1_min >= 25);
        let (h4_min, _) =
            OofCalibrator::min_trade_thresholds_for_timeframe(1200, Some(Timeframe::H4));
        assert!(h4_min >= 60);
        let (h1_min, _) =
            OofCalibrator::min_trade_thresholds_for_timeframe(1200, Some(Timeframe::H1));
        assert!(h1_min >= 60);
    }

    #[test]
    fn test_oof_vol_scaled_asymmetric_calibration() {
        let n = 600;
        let klines: Vec<Kline> = (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64 * 0.25).sin() * 4.0;
                Kline {
                    timestamp: (i as i64 + 1) * 3600,
                    open: p,
                    high: p + 0.6,
                    low: p - 0.6,
                    close: p + 0.2,
                    volume: 1000.0,
                }
            })
            .collect();

        let preds: Vec<f64> = (0..n)
            .map(|i| match i % 16 {
                0..=5 => 0.35 + ((i % 6) as f64) * 0.3,
                8..=13 => -0.45 - ((i % 6) as f64) * 0.3,
                _ => 0.0,
            })
            .collect();

        let mut cfg = BacktestConfig::default();
        cfg.timeframe = Some(Timeframe::H4);
        let (long_th, short_th, _score) =
            OofCalibrator::calibrate_asymmetric_with_warmup(&preds, &klines, FitnessMetric::SharpeRatio, &cfg, 100);
        assert!(long_th > 0.05 && long_th < 3.0);
        assert!(short_th < -0.05 && short_th > -3.0);

        let (sym_long, sym_short, _) =
            OofCalibrator::calibrate_symmetric(&preds, &klines, FitnessMetric::SharpeRatio, &cfg);
        assert!(sym_long > 0.0);
        assert!(sym_short < 0.0);
    }

    #[test]
    fn test_oof_tier1_fallback_and_zero_trade_floor() {
        let low_trade_candidates = vec![(0.001, 1.2, 3), (0.002, 2.8, 1), (0.003, 9.9, 0)];
        let (th, score) = OofCalibrator::select_best_candidate_for_timeframe(
            &low_trade_candidates,
            2000,
            Some(Timeframe::H4),
            0.002,
        );
        assert_eq!(th, 0.002);
        assert!((score - 2.8).abs() < 1e-9);

        let zero_trade_candidates = vec![(0.001, 5.0, 0), (0.002, 8.0, 0)];
        let (zero_th, zero_score) = OofCalibrator::select_best_candidate_for_timeframe(
            &zero_trade_candidates,
            2000,
            Some(Timeframe::H4),
            0.002,
        );
        assert_eq!(zero_th, 0.002);
        assert_eq!(zero_score, 0.0);

        assert_eq!(
            OofCalibrator::min_trade_thresholds_for_timeframe(1200, Some(Timeframe::H4)),
            (60, 30)
        );
        assert_eq!(
            OofCalibrator::min_trade_thresholds_for_timeframe(6000, Some(Timeframe::H1)),
            (60, 30)
        );
        assert_eq!(
            OofCalibrator::min_trade_thresholds_for_timeframe(12000, Some(Timeframe::H1)),
            (75, 37)
        );
        assert_eq!(
            OofCalibrator::min_trade_thresholds_for_timeframe(16000, Some(Timeframe::H1)),
            (95, 47)
        );
    }

    #[test]
    fn test_oof_test_strategy_clamps_sl_syncs_legs_and_scales_vol() {
        let klines: Vec<Kline> = (0..110)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 100.0,
                volume: 100.0,
            })
            .collect();
        let mut preds = vec![0.0; 110];
        preds[100] = 0.005;
        preds[101] = 0.0012;

        let mut cfg = TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H4), false);
        cfg.primary_sl_atr_mult = 2.2;
        cfg.max_sl_pct = Some(0.028);
        cfg.vol_target_pct = Some(0.015);

        let mut strat = OofTestStrategy {
            predictions: &preds,
            long_th: 0.004,
            short_th: -0.004,
            warmup_bars: 100,
            hedge_config: cfg,
            decision: TreeBoostSignalDecision::new(),
        };

        let h100 = strat.evaluate_hedge(&klines, 100);
        match h100.long {
            crate::types::HedgeLegAction::OpenOrKeep {
                stop_loss_price: Some(sl),
                size_multiplier,
                ..
            } => {
                assert!((sl - 97.2).abs() < 1e-6);
                assert!((size_multiplier - 0.35).abs() < 1e-6);
            }
            other => panic!("unexpected leg: {:?}", other),
        }

        strat.sync_open_legs(false, false);
        assert_eq!(strat.decision.current_direction(), SignalDirection::Flat);
        let s101 = strat.evaluate(&klines, 101);
        assert_eq!(s101.direction, SignalDirection::Flat);
    }

    #[test]
    fn test_oof_composite_allianz_stagnation_penalty() {
        use crate::backtest::{EquityPoint, MetricsCalculator};

        let curve = vec![
            EquityPoint::new(1000, 0, 10000.0, 10000.0, 0.0, 0.0),
            EquityPoint::new(2000, 1, 11000.0, 11000.0, 0.0, 0.0),
            EquityPoint::new(3000, 2, 10500.0, 10500.0, 500.0, 4.545454545454546),
            EquityPoint::new(4000, 3, 10600.0, 10600.0, 400.0, 3.6363636363636362),
        ];
        let mut metrics = MetricsCalculator::calculate(10000.0, &[], &curve);
        metrics.net_profit = 600.0;
        metrics.net_profit_pct = 6.0;
        metrics.max_drawdown_pct = 10.0;
        metrics.composite_score = 0.8;

        let res = BacktestResult {
            metrics: metrics.clone(),
            trades: Vec::new(),
            equity_curve: curve.clone(),
        };
        let adj = OofCalibrator::score_backtest_result(&res, FitnessMetric::CompositeAllianz);
        let expected = 0.8 - 0.35 * 0.5 - 0.25 * 0.5 - 0.25 * 0.10;
        assert!((adj - expected).abs() < 1e-12);

        let mut loss_metrics = metrics;
        loss_metrics.net_profit = -100.0;
        loss_metrics.composite_score = -0.2;
        let loss_res = BacktestResult {
            metrics: loss_metrics,
            trades: Vec::new(),
            equity_curve: curve,
        };
        let loss_adj =
            OofCalibrator::score_backtest_result(&loss_res, FitnessMetric::CompositeAllianz);
        assert!((loss_adj - (-0.2)).abs() < 1e-12);
    }
}

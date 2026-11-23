//! Calibrador de Umbrales sobre Predicciones Out-of-Fold (OOF Calibrator)
//!
//! REGLA FINANCIERA: Elimina el meta-overfitting de umbrales optimizando tau_long y tau_short
//! exclusivamente sobre vectores de predicción generados fuera de muestra (Out-of-Fold).
//!
//! Cada predicción F_oof(x_t) proviene de un estimador entrenado sin ver el punto t, garantizando
//! que los umbrales seleccionados generalicen fielmente al régimen no observado.
//!
//! Incorpora AdaptiveThresholdGrid para calibración adaptativa a la escala OOF y
//! penalización estricta de inactividad (min_trades) para garantizar significancia estadística.

use rayon::prelude::*;
use crate::backtest::{BacktestConfig, BacktestEngine, Strategy};
use crate::types::{FitnessMetric, Kline, Signal, SignalDirection};
use super::adaptive_threshold_grid::AdaptiveThresholdGrid;

pub struct OofCalibrator;

struct OofTestStrategy<'a> {
    predictions: &'a [f64],
    long_th: f64,
    short_th: f64,
    last_direction: SignalDirection,
}

impl<'a> Strategy for OofTestStrategy<'a> {
    fn name(&self) -> &str {
        "OofTestStrategy"
    }

    fn clear_cache(&mut self) {
        self.last_direction = SignalDirection::Flat;
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        let ts = history[current_idx].timestamp;
        if current_idx >= self.predictions.len() || current_idx < 35 {
            self.last_direction = SignalDirection::Flat;
            return Signal::flat(ts);
        }

        let pred = self.predictions[current_idx];
        if pred > self.long_th {
            self.last_direction = SignalDirection::Long;
            let divisor = (self.long_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { pred / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Long, strength, ts)
        } else if pred < self.short_th {
            self.last_direction = SignalDirection::Short;
            let divisor = (self.short_th * 3.0).abs();
            let raw_strength = if divisor > 1e-9 { pred.abs() / divisor } else { 0.5 };
            let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
            Signal::new(SignalDirection::Short, strength, ts)
        } else if self.last_direction == SignalDirection::Long && pred > 0.0 {
            let strength = (pred / self.long_th.max(1e-9)).clamp(0.01, 1.0);
            Signal::new(SignalDirection::Long, strength, ts)
        } else if self.last_direction == SignalDirection::Short && pred < 0.0 {
            let strength = (pred.abs() / self.short_th.abs().max(1e-9)).clamp(0.01, 1.0);
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            self.last_direction = SignalDirection::Flat;
            Signal::flat(ts)
        }
    }
}

impl OofCalibrator {
    /// Rejilla estándar de umbrales para calibración Out-of-Fold (compatibilidad)
    pub const CANDIDATE_THRESHOLDS: [f64; 9] = [
        0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070, 0.0100,
    ];

    fn min_trade_thresholds(num_klines: usize) -> (usize, usize) {
        let min_trades = if num_klines >= 500 {
            (num_klines / 40).clamp(15, 60)
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

    fn select_best_candidate<T: Copy>(
        candidates: &[(T, f64, usize)],
        num_klines: usize,
        default_val: T,
    ) -> (T, f64) {
        let (min_trades, fallback_min_trades) = Self::min_trade_thresholds(num_klines);

        for trade_floor in [min_trades, fallback_min_trades] {
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

    /// Calibra umbrales simétricos (tau, -tau) sobre predicciones Out-of-Fold
    /// Retorna: (tau_long, tau_short, score_optimo_oof)
    pub fn calibrate_symmetric(
        oof_predictions: &[f64],
        klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        if oof_predictions.is_empty() || klines.len() < 50 {
            return (0.002, -0.002, 0.0);
        }

        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(oof_predictions);

        let candidates: Vec<(f64, f64, usize)> = grid
            .into_par_iter()
            .map(|th| {
                let mut strat = OofTestStrategy {
                    predictions: oof_predictions,
                    long_th: th,
                    short_th: -th,
                    last_direction: SignalDirection::Flat,
                };
                let engine = BacktestEngine::new(base_config.clone());
                match engine.run(&mut strat, klines) {
                    Ok(res) => {
                        let score = fitness_metric.evaluate(&res.metrics);
                        (th, score, res.metrics.total_trades)
                    }
                    Err(_) => (th, f64::NEG_INFINITY, 0),
                }
            })
            .collect();

        let (final_th, final_score) =
            Self::select_best_candidate(&candidates, klines.len(), 0.002);

        (final_th, -final_th, final_score)
    }

    /// Calibra umbrales asimétricos (tau_long > 0, tau_short < 0) sobre predicciones Out-of-Fold
    ///
    /// Permite capturar asimetrías de mercado entre regímenes alcistas y bajistas
    /// sin incurrir en meta-overfitting gracias a la naturaleza OOF de las predicciones.
    pub fn calibrate_asymmetric(
        oof_predictions: &[f64],
        klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        if oof_predictions.is_empty() || klines.len() < 50 {
            return (0.002, -0.002, 0.0);
        }

        let (long_grid, short_grid) = AdaptiveThresholdGrid::compute_asymmetric_grids(oof_predictions);

        let mut pairs = Vec::with_capacity(long_grid.len() * short_grid.len());
        for &l in &long_grid {
            for &s in &short_grid {
                pairs.push((l, s));
            }
        }

        let candidates: Vec<((f64, f64), f64, usize)> = pairs
            .into_par_iter()
            .map(|(l_th, s_th)| {
                let mut strat = OofTestStrategy {
                    predictions: oof_predictions,
                    long_th: l_th,
                    short_th: s_th,
                    last_direction: SignalDirection::Flat,
                };
                let engine = BacktestEngine::new(base_config.clone());
                match engine.run(&mut strat, klines) {
                    Ok(res) => {
                        let score = fitness_metric.evaluate(&res.metrics);
                        ((l_th, s_th), score, res.metrics.total_trades)
                    }
                    Err(_) => ((l_th, s_th), f64::NEG_INFINITY, 0),
                }
            })
            .collect();

        let (final_pair, final_score) =
            Self::select_best_candidate(&candidates, klines.len(), (0.002, -0.002));

        (final_pair.0, final_pair.1, final_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Predicciones con asimetría: alcista moderada (+0.0006) vs bajista fuerte (-0.0014)
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
        assert_eq!(OofCalibrator::min_trade_thresholds(3000), (60, 30));
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
}

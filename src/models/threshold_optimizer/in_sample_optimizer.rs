//! Calibrador y Optimizador Causal de Umbrales In-Sample guiado por la Métrica Fitness
//!
//! REGLA FINANCIERA: La optimización de umbrales se ejecuta estrictamente sobre
//! datos In-Sample (IS - t <= T_IS), sin acceder jamás al conjunto Out-of-Sample (OOS).
//!
//! Incorpora AdaptiveThresholdGrid para adaptar los candidatos a la dispersión real de
//! las predicciones y penaliza la inactividad (min_trades) para impedir que una estrategia
//! sin operaciones obtenga puntaje superior a estrategias con actividad estadística verificable.

use rayon::prelude::*;
use crate::backtest::{BacktestConfig, BacktestEngine, Strategy};
use crate::features::H4FeatureMatrix;
use crate::types::{FitnessMetric, Kline, Signal, SignalDirection, Timeframe};
use super::adaptive_threshold_grid::AdaptiveThresholdGrid;

pub struct ThresholdOptimizer;

struct ThresholdTestStrategy<'a> {
    predictions: &'a [f64],
    threshold: f64,
    warmup_bars: usize,
}

impl<'a> Strategy for ThresholdTestStrategy<'a> {
    fn name(&self) -> &str {
        "ThresholdTest"
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        if current_idx >= history.len() {
            return Signal::flat(0);
        }
        let ts = history[current_idx].timestamp;
        if current_idx >= self.predictions.len() || current_idx < self.warmup_bars {
            return Signal::flat(ts);
        }

        let pred = self.predictions[current_idx];
        if !pred.is_finite() {
            return Signal::flat(ts);
        }

        let denom = self.threshold * 3.0;
        if pred > self.threshold {
            let raw_strength = if denom.is_finite() && denom > 1e-9 {
                pred / denom
            } else {
                0.5
            };
            let strength = if raw_strength.is_finite() {
                raw_strength.clamp(0.1, 1.0)
            } else {
                0.5
            };
            Signal::new(SignalDirection::Long, strength, ts)
        } else if pred < -self.threshold {
            let raw_strength = if denom.is_finite() && denom > 1e-9 {
                pred.abs() / denom
            } else {
                0.5
            };
            let strength = if raw_strength.is_finite() {
                raw_strength.clamp(0.1, 1.0)
            } else {
                0.5
            };
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            Signal::flat(ts)
        }
    }
}

impl ThresholdOptimizer {
    pub const THRESHOLD_GRID: [f64; 8] = [
        0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
    ];

    pub fn optimize_in_sample(
        in_sample_predictions: &[f64],
        train_klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        let warmup_bars = match base_config.timeframe {
            Some(Timeframe::H4) => H4FeatureMatrix::WARMUP_BARS,
            _ => 60,
        };
        if in_sample_predictions.is_empty()
            || train_klines.len() < 50.max(warmup_bars.saturating_add(2))
        {
            return (0.002, -0.002, 0.0);
        }

        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(in_sample_predictions);

        let min_trades = if train_klines.len() < 100 { 3 } else { 5 };

        let candidates: Vec<(f64, f64, usize)> = grid
            .into_par_iter()
            .map(|th| {
                let mut strat = ThresholdTestStrategy {
                    predictions: in_sample_predictions,
                    threshold: th,
                    warmup_bars,
                };
                let engine = BacktestEngine::new(base_config.clone());
                match engine.run_with_warmup(&mut strat, train_klines, warmup_bars) {
                    Ok(res) => {
                        let score = fitness_metric.evaluate(&res.metrics);
                        (th, score, res.metrics.total_trades)
                    }
                    Err(_) => (th, f64::NEG_INFINITY, 0),
                }
            })
            .collect();

        // 4. Selección jerárquica:
        // Nivel 1: Candidatos con trades >= min_trades (mayor score)
        let mut best_th = None;
        let mut best_score = f64::NEG_INFINITY;

        for &(th, score, trades) in &candidates {
            if trades >= min_trades && score.is_finite() && score > best_score {
                best_score = score;
                best_th = Some(th);
            }
        }

        // Nivel 2: Si ninguno alcanzó min_trades, relajar a trades >= 1
        if best_th.is_none() {
            for &(th, score, trades) in &candidates {
                if trades >= 1 && score.is_finite() && score > best_score {
                    best_score = score;
                    best_th = Some(th);
                }
            }
        }

        // Nivel 3: Si ningún candidato generó trades (0 trades), fallback adaptativo sin premiar inactividad
        let final_th = match best_th {
            Some(th) => th,
            None => {
                // Fallback: primer umbral de la grilla adaptativa (el más cercano a la actividad)
                candidates.first().map(|c| c.0).unwrap_or(0.002)
            }
        };

        let final_score = if best_th.is_some() && best_score.is_finite() {
            best_score
        } else {
            0.0
        };

        (final_th, -final_th, final_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_predictions_fallback() {
        let (th_long, th_short, score) = ThresholdOptimizer::optimize_in_sample(
            &[],
            &[],
            FitnessMetric::SharpeRatio,
            &BacktestConfig::default(),
        );
        assert_eq!(th_long, 0.002);
        assert_eq!(th_short, -0.002);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_adaptive_in_sample_selects_active_threshold_when_trades_exist() {
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

        // Predicciones alternantes con suficiente magnitud para activar umbrales bajos pero no altos
        let preds: Vec<f64> = (0..n)
            .map(|i| if i % 10 < 5 { 0.0008 } else { -0.0008 })
            .collect();

        let (th_long, th_short, _score) = ThresholdOptimizer::optimize_in_sample(
            &preds,
            &klines,
            FitnessMetric::SharpeRatio,
            &BacktestConfig::default(),
        );

        assert!(th_long > 0.0, "tau_long debe ser positivo");
        assert!(th_short < 0.0, "tau_short debe ser negativo");
        assert!(th_long <= 0.0015, "tau_long no debe ser inalcanzable ({})", th_long);
    }

    #[test]
    fn test_threshold_strategy_guards_zero_threshold_and_non_finite_pred() {
        let klines: Vec<Kline> = (0..70)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 1000.0,
            })
            .collect();
        let mut preds = vec![0.05; 70];
        preds[65] = f64::NAN;
        preds[66] = f64::INFINITY;

        let mut strat = ThresholdTestStrategy {
            predictions: &preds,
            threshold: 0.0,
            warmup_bars: 60,
        };
        let sig_zero = strat.evaluate(&klines, 61);
        assert_eq!(sig_zero.direction, SignalDirection::Long);
        assert!(sig_zero.strength.is_finite());

        let sig_nan = strat.evaluate(&klines, 65);
        assert_eq!(sig_nan.direction, SignalDirection::Flat);

        let sig_inf = strat.evaluate(&klines, 66);
        assert_eq!(sig_inf.direction, SignalDirection::Flat);

        let sig_oob = strat.evaluate(&klines, 100);
        assert_eq!(sig_oob.direction, SignalDirection::Flat);
    }
}

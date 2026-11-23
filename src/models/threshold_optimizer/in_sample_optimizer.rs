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
use crate::types::{FitnessMetric, Kline, Signal, SignalDirection};
use super::adaptive_threshold_grid::AdaptiveThresholdGrid;

pub struct ThresholdOptimizer;

struct ThresholdTestStrategy<'a> {
    predictions: &'a [f64],
    threshold: f64,
}

impl<'a> Strategy for ThresholdTestStrategy<'a> {
    fn name(&self) -> &str {
        "ThresholdTest"
    }

    fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
        let ts = history[current_idx].timestamp;
        if current_idx >= self.predictions.len() || current_idx < 35 {
            return Signal::flat(ts);
        }

        let pred = self.predictions[current_idx];
        if pred > self.threshold {
            let strength = (pred / (self.threshold * 3.0)).clamp(0.1, 1.0);
            Signal::new(SignalDirection::Long, strength, ts)
        } else if pred < -self.threshold {
            let strength = (pred.abs() / (self.threshold * 3.0)).clamp(0.1, 1.0);
            Signal::new(SignalDirection::Short, strength, ts)
        } else {
            Signal::flat(ts)
        }
    }
}

impl ThresholdOptimizer {
    /// Rejilla estándar de umbrales estáticos para compatibilidad y fallback
    pub const THRESHOLD_GRID: [f64; 8] = [
        0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050, 0.0070,
    ];

    /// Encuentra el umbral óptimo (tau_long, tau_short) que maximiza la métrica fitness en In-Sample
    /// Retorna: (tau_long, tau_short, score_optimo)
    pub fn optimize_in_sample(
        in_sample_predictions: &[f64],
        train_klines: &[Kline],
        fitness_metric: FitnessMetric,
        base_config: &BacktestConfig,
    ) -> (f64, f64, f64) {
        if in_sample_predictions.is_empty() || train_klines.len() < 50 {
            return (0.002, -0.002, 0.0);
        }

        // 1. Obtener rejilla adaptativa basada en la distribución real de predicciones
        let grid = AdaptiveThresholdGrid::compute_symmetric_grid(in_sample_predictions);

        // 2. Umbral mínimo de significancia estadística en operaciones
        let min_trades = if train_klines.len() < 100 { 3 } else { 5 };

        // 3. Evaluación concurrente de todos los umbrales candidatos con Rayon
        // Retorna (th, score, trades_count)
        let candidates: Vec<(f64, f64, usize)> = grid
            .into_par_iter()
            .map(|th| {
                let mut strat = ThresholdTestStrategy {
                    predictions: in_sample_predictions,
                    threshold: th,
                };
                let engine = BacktestEngine::new(base_config.clone());
                match engine.run(&mut strat, train_klines) {
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
        // Con la grilla adaptativa, el umbral debe adaptarse al rango de 0.0008
        assert!(th_long <= 0.0015, "tau_long no debe ser inalcanzable ({})", th_long);
    }
}

use serde::{Deserialize, Serialize};

use crate::types::Timeframe;
use super::config::HmmConfig;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HmmHedgeStrategyConfig {
    pub hmm_config: HmmConfig,
    pub vol_window: usize,
    pub ema_fast: usize,
    pub ema_slow: usize,
    pub zscore_window: usize,
    pub mr_z_entry: f64,
    pub mr_z_exit: f64,
    pub bull_confidence_threshold: f64,
    pub bear_confidence_threshold: f64,
    pub bull_exposure: f64,
    pub sideways_exposure: f64,
    pub bear_exposure: f64,
    pub min_holding_bars: usize,
}

impl Default for HmmHedgeStrategyConfig {
    fn default() -> Self {
        Self::preset_h4()
    }
}

impl HmmHedgeStrategyConfig {
    pub fn preset_d1() -> Self {
        Self {
            hmm_config: HmmConfig {
                max_em_iters: 50,
                tol: 1e-6,
                min_variance: 1e-6,
                lookback_window: 252,
                refit_interval: 20,
            },
            vol_window: 20,
            ema_fast: 12,
            ema_slow: 36,
            zscore_window: 20,
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

    pub fn preset_h4() -> Self {
        Self {
            hmm_config: HmmConfig {
                max_em_iters: 50,
                tol: 1e-6,
                min_variance: 1e-6,
                lookback_window: 360,
                refit_interval: 30,
            },
            vol_window: 30,
            ema_fast: 36,
            ema_slow: 108,
            zscore_window: 30,
            mr_z_entry: 1.75,
            mr_z_exit: 0.0,
            bull_confidence_threshold: 0.55,
            bear_confidence_threshold: 0.55,
            bull_exposure: 0.70,
            sideways_exposure: 0.40,
            bear_exposure: -0.70,
            min_holding_bars: 3,
        }
    }

    pub fn preset_h1() -> Self {
        Self {
            hmm_config: HmmConfig {
                max_em_iters: 50,
                tol: 1e-6,
                min_variance: 1e-6,
                lookback_window: 720,
                refit_interval: 48,
            },
            vol_window: 48,
            ema_fast: 72,
            ema_slow: 216,
            zscore_window: 48,
            mr_z_entry: 2.0,
            mr_z_exit: 0.0,
            bull_confidence_threshold: 0.60,
            bear_confidence_threshold: 0.60,
            bull_exposure: 0.70,
            sideways_exposure: 0.40,
            bear_exposure: -0.70,
            min_holding_bars: 4,
        }
    }

    pub fn for_timeframe(tf: Timeframe) -> Self {
        match tf {
            Timeframe::D1 | Timeframe::D3 | Timeframe::W1 => Self::preset_d1(),
            Timeframe::H2 | Timeframe::H4 | Timeframe::H6 | Timeframe::H8 | Timeframe::H12 => {
                Self::preset_h4()
            }
            _ => Self::preset_h1(),
        }
    }
}

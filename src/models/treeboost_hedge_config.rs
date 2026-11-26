use serde::{Deserialize, Serialize};
use crate::types::Timeframe;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TreeBoostHedgeConfig {
    pub enabled: bool,
    pub hedge_size_mult: f64,
    pub hedge_entry_z: f64,
    pub hedge_sl_atr_mult: f64,
    pub hedge_tp_atr_mult: f64,
    pub primary_sl_atr_mult: f64,
    pub min_hold_bars: usize,
    pub mr_rsi_low: f64,
    pub mr_rsi_high: f64,
    pub primary_tp_atr_mult: Option<f64>,
    pub trailing_atr_mult: Option<f64>,
    pub trail_activate_atr_mult: Option<f64>,
    pub vol_target_pct: Option<f64>,
    pub max_sl_pct: Option<f64>,
    pub mr_require_rejection: bool,
    pub hysteresis_frac: f64,
    pub mr_exit_z: f64,
    pub mr_max_ker: Option<f64>,
    pub mr_block_against_primary: bool,
    pub primary_min_ker: Option<f64>,
    pub primary_cooldown_bars: usize,
    pub mr_vol_target_pct: Option<f64>,
    pub vol_min_mult: f64,
    pub vol_max_mult: f64,
}

impl Default for TreeBoostHedgeConfig {
    fn default() -> Self {
        Self::for_timeframe(None, false)
    }
}

impl TreeBoostHedgeConfig {
    pub fn for_timeframe(tf: Option<Timeframe>, enabled: bool) -> Self {
        let min_hold_bars = match tf {
            Some(Timeframe::D1 | Timeframe::D3 | Timeframe::W1) => 3,
            Some(Timeframe::H2 | Timeframe::H4 | Timeframe::H6 | Timeframe::H8 | Timeframe::H12) => 6,
            Some(Timeframe::H1) => 8,
            Some(_) => 8,
            None => 3,
        };
        let primary_sl_atr_mult = match tf {
            Some(Timeframe::D1 | Timeframe::D3 | Timeframe::W1) => 2.2,
            Some(Timeframe::H2 | Timeframe::H4 | Timeframe::H6 | Timeframe::H8 | Timeframe::H12) => 2.2,
            Some(Timeframe::H1) => 2.2,
            Some(_) => 2.2,
            None => 2.2,
        };
        let (
            hedge_size_mult,
            hedge_entry_z,
            hedge_sl_atr_mult,
            hedge_tp_atr_mult,
            mr_rsi_low,
            mr_rsi_high,
            vol_target_pct,
            max_sl_pct,
        ) = match tf {
            Some(Timeframe::H2 | Timeframe::H4 | Timeframe::H6 | Timeframe::H8 | Timeframe::H12) => (
                0.35,
                2.35,
                1.4,
                2.6,
                28.0,
                72.0,
                Some(0.015),
                Some(0.028),
            ),
            _ => (0.40, 2.0, 1.5, 2.0, 32.0, 68.0, None, None),
        };
        Self {
            enabled,
            hedge_size_mult,
            hedge_entry_z,
            hedge_sl_atr_mult,
            hedge_tp_atr_mult,
            primary_sl_atr_mult,
            min_hold_bars,
            mr_rsi_low,
            mr_rsi_high,
            primary_tp_atr_mult: None,
            trailing_atr_mult: None,
            trail_activate_atr_mult: None,
            vol_target_pct,
            max_sl_pct,
            mr_require_rejection: true,
            hysteresis_frac: 0.25,
            mr_exit_z: 0.0,
            mr_max_ker: None,
            mr_block_against_primary: false,
            primary_min_ker: None,
            primary_cooldown_bars: 0,
            mr_vol_target_pct: None,
            vol_min_mult: 0.35,
            vol_max_mult: 1.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treeboost_hedge_config_presets() {
        let d1 = TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::D1), true);
        assert!(d1.enabled);
        assert_eq!(d1.min_hold_bars, 3);
        assert!((d1.primary_sl_atr_mult - 2.2).abs() < 1e-12);
        assert!((d1.hedge_size_mult - 0.40).abs() < 1e-12);
        assert!((d1.hedge_entry_z - 2.0).abs() < 1e-12);
        assert!((d1.hedge_sl_atr_mult - 1.5).abs() < 1e-12);
        assert!((d1.hedge_tp_atr_mult - 2.0).abs() < 1e-12);
        assert!((d1.mr_rsi_low - 32.0).abs() < 1e-12);
        assert!((d1.mr_rsi_high - 68.0).abs() < 1e-12);

        assert_eq!(d1.vol_target_pct, None);
        assert_eq!(d1.max_sl_pct, None);
        assert_eq!(d1.mr_vol_target_pct, None);

        let h4 = TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H4), false);
        assert!(!h4.enabled);
        assert_eq!(h4.min_hold_bars, 6);
        assert!((h4.primary_sl_atr_mult - 2.2).abs() < 1e-12);
        assert!((h4.hedge_size_mult - 0.35).abs() < 1e-12);
        assert!((h4.hedge_entry_z - 2.35).abs() < 1e-12);
        assert!((h4.hedge_sl_atr_mult - 1.4).abs() < 1e-12);
        assert!((h4.hedge_tp_atr_mult - 2.6).abs() < 1e-12);
        assert!((h4.mr_rsi_low - 28.0).abs() < 1e-12);
        assert!((h4.mr_rsi_high - 72.0).abs() < 1e-12);
        assert_eq!(h4.vol_target_pct, Some(0.015));
        assert_eq!(h4.max_sl_pct, Some(0.028));
        assert_eq!(h4.mr_vol_target_pct, None);
        assert!(h4.mr_require_rejection);
        assert!((h4.hysteresis_frac - 0.25).abs() < 1e-12);

        let h1 = TreeBoostHedgeConfig::for_timeframe(Some(Timeframe::H1), true);
        assert!(h1.enabled);
        assert_eq!(h1.min_hold_bars, 8);
        assert!((h1.primary_sl_atr_mult - 2.2).abs() < 1e-12);
        assert_eq!(h1.vol_target_pct, None);
        assert_eq!(h1.max_sl_pct, None);
        assert_eq!(h1.mr_vol_target_pct, None);

        let none = TreeBoostHedgeConfig::default();
        assert!(!none.enabled);
        assert_eq!(none.min_hold_bars, 3);
        assert_eq!(none.primary_tp_atr_mult, None);
        assert_eq!(none.trailing_atr_mult, None);
        assert_eq!(none.trail_activate_atr_mult, None);
        assert_eq!(d1.trail_activate_atr_mult, None);
        assert_eq!(h4.trail_activate_atr_mult, None);
        assert_eq!(h1.trail_activate_atr_mult, None);
        assert_eq!(none.vol_target_pct, None);
        assert_eq!(none.max_sl_pct, None);
        assert_eq!(none.mr_vol_target_pct, None);
        assert!(none.mr_require_rejection);
        assert!((none.hysteresis_frac - 0.25).abs() < 1e-12);
        assert!((none.mr_exit_z - 0.0).abs() < 1e-12);
        assert_eq!(none.mr_max_ker, None);
        assert!(!none.mr_block_against_primary);
        assert_eq!(none.primary_min_ker, None);
        assert_eq!(none.primary_cooldown_bars, 0);
        assert!((none.vol_min_mult - 0.35).abs() < 1e-12);
        assert!((none.vol_max_mult - 1.25).abs() < 1e-12);
        assert!((h4.vol_min_mult - 0.35).abs() < 1e-12);
        assert!((h4.vol_max_mult - 1.25).abs() < 1e-12);
    }
}

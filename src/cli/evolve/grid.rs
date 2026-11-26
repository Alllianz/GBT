use crate::models::TreeBoostHedgeConfig;
use crate::types::{PositionSizing, Timeframe};
use super::spec::ModelGroupSpec;

#[derive(Clone)]
pub struct EvolveCandidate {
    pub id: usize,
    pub label: String,
    pub group_idx: usize,
    pub is_baseline: bool,
    pub hedge_cfg: TreeBoostHedgeConfig,
    pub position_sizing: Option<PositionSizing>,
}

impl EvolveCandidate {
    #[allow(clippy::too_many_arguments)]
    fn rg_hedge_config(
        z: f64,
        exit_z: f64,
        rsi_low: f64,
        rsi_high: f64,
        h_sz: f64,
        tp: f64,
        h_sl: f64,
        p_sl: f64,
        m_sl: f64,
        vt: f64,
        pk: f64,
        cd: usize,
        mr_vol_target_pct: Option<f64>,
        vol_min_mult: f64,
        vol_max_mult: f64,
        trail_activate_atr_mult: Option<f64>,
        trailing_atr_mult: Option<f64>,
    ) -> TreeBoostHedgeConfig {
        TreeBoostHedgeConfig {
            enabled: true,
            hedge_size_mult: h_sz,
            hedge_entry_z: z,
            hedge_sl_atr_mult: h_sl,
            hedge_tp_atr_mult: tp,
            primary_sl_atr_mult: p_sl,
            min_hold_bars: 5,
            mr_rsi_low: rsi_low,
            mr_rsi_high: rsi_high,
            primary_tp_atr_mult: None,
            trailing_atr_mult,
            trail_activate_atr_mult,
            vol_target_pct: Some(vt),
            max_sl_pct: Some(m_sl),
            mr_require_rejection: true,
            hysteresis_frac: 0.25,
            mr_exit_z: exit_z,
            mr_max_ker: Some(0.42),
            mr_block_against_primary: true,
            primary_min_ker: Some(pk),
            primary_cooldown_bars: cd,
            mr_vol_target_pct,
            vol_min_mult,
            vol_max_mult,
        }
    }

    fn format_cfg_tag(cfg: &TreeBoostHedgeConfig) -> String {
        let hsl_tag = if (cfg.hedge_sl_atr_mult - 1.4).abs() > 1e-9 {
            if (cfg.hedge_sl_atr_mult * 10.0 - (cfg.hedge_sl_atr_mult * 10.0).round()).abs() > 1e-6
            {
                format!("_HSL{}", (cfg.hedge_sl_atr_mult * 100.0).round() as usize)
            } else {
                format!("_HSL{}", (cfg.hedge_sl_atr_mult * 10.0).round() as usize)
            }
        } else {
            String::new()
        };
        let hy_tag = if (cfg.hysteresis_frac - 0.25).abs() > 1e-9 {
            format!("_HY{}", (cfg.hysteresis_frac * 100.0).round() as usize)
        } else {
            String::new()
        };
        let ta_tag = cfg
            .trail_activate_atr_mult
            .map(|v| format!("_TA{}", (v * 10.0).round() as usize))
            .unwrap_or_default();
        let tr_tag = cfg
            .trailing_atr_mult
            .map(|v| format!("_TR{}", (v * 10.0).round() as usize))
            .unwrap_or_default();
        let ptp_tag = cfg
            .primary_tp_atr_mult
            .map(|v| format!("_PTP{}", (v * 10.0).round() as usize))
            .unwrap_or_default();
        let mvt_tag = cfg
            .mr_vol_target_pct
            .map(|v| format!("_MVT{}", (v * 1000.0).round() as usize))
            .unwrap_or_default();
        let vc_tag = if (cfg.vol_min_mult - 0.35).abs() > 1e-9
            || (cfg.vol_max_mult - 1.25).abs() > 1e-9
        {
            format!(
                "_VC{}_{}",
                (cfg.vol_min_mult * 100.0).round() as usize,
                (cfg.vol_max_mult * 100.0).round() as usize
            )
        } else {
            String::new()
        };
        format!(
            "Z{}_EZ{}_R{}_SZ{}_TP{}_PSL{}_MSL{}_VT{}_PK{}_CD{}{}{}{}{}{}{}{}",
            (cfg.hedge_entry_z * 100.0).round() as usize,
            (cfg.mr_exit_z * 10.0).round() as usize,
            cfg.mr_rsi_low as usize,
            (cfg.hedge_size_mult * 100.0).round() as usize,
            (cfg.hedge_tp_atr_mult * 10.0).round() as usize,
            (cfg.primary_sl_atr_mult * 100.0).round() as usize,
            (cfg.max_sl_pct.unwrap_or(0.0) * 1000.0).round() as usize,
            (cfg.vol_target_pct.unwrap_or(0.0) * 1000.0).round() as usize,
            (cfg.primary_min_ker.unwrap_or(0.0) * 100.0).round() as usize,
            cfg.primary_cooldown_bars,
            hsl_tag,
            hy_tag,
            ta_tag,
            tr_tag,
            ptp_tag,
            mvt_tag,
            vc_tag,
        )
    }

    pub fn build_candidates(tf: Timeframe, groups: &[ModelGroupSpec]) -> Vec<Self> {
        let mut base_hedge = TreeBoostHedgeConfig::for_timeframe(Some(tf), true);
        base_hedge.mr_vol_target_pct = None;
        base_hedge.vol_min_mult = 0.35;
        base_hedge.vol_max_mult = 1.25;
        base_hedge.trail_activate_atr_mult = None;
        let poic10 = Some(PositionSizing::PercentOfInitialCapital(0.10));

        let mut out = Vec::with_capacity(groups.len() * 140 + 1);
        let mut id = 0usize;

        let anchors: [(&str, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, Option<f64>); 4] = [
            ("ANC_C110", 1.25, 0.8, 31.0, 69.0, 3.25, 4.0, 1.40, 0.18, 0.20, 1.60, Some(0.024)),
            ("ANC_C116", 1.30, 0.7, 30.0, 70.0, 3.25, 4.0, 1.40, 0.18, 0.20, 1.60, Some(0.020)),
            ("ANC_C118", 1.30, 0.7, 30.0, 70.0, 3.25, 4.0, 1.40, 0.18, 0.20, 1.60, Some(0.024)),
            ("ANC_C046", 1.25, 0.8, 31.0, 69.0, 3.00, 3.6, 1.40, 0.18, 0.20, 1.60, Some(0.024)),
        ];

        let trail_pairs: [(Option<f64>, Option<f64>); 6] = [
            (None, None),
            (Some(4.0), Some(3.0)),
            (Some(5.0), Some(3.5)),
            (Some(6.0), Some(4.0)),
            (Some(8.0), Some(4.5)),
            (Some(10.0), Some(5.0)),
        ];

        let trail_bases: [(f64, f64, f64, f64, f64, f64, f64, Option<f64>, f64, f64); 4] = [
            (1.30, 0.7, 30.0, 70.0, 3.25, 4.0, 0.18, Some(0.020), 0.20, 1.60),
            (1.30, 0.7, 30.0, 70.0, 3.25, 4.0, 0.18, Some(0.024), 0.20, 1.60),
            (1.25, 0.8, 31.0, 69.0, 3.25, 4.0, 0.18, Some(0.024), 0.20, 1.60),
            (1.30, 0.7, 30.0, 70.0, 3.25, 4.0, 0.20, Some(0.020), 0.22, 1.35),
        ];

        let mr_profiles: [(f64, f64, f64, f64); 3] = [
            (1.30, 0.7, 30.0, 70.0),
            (1.28, 0.7, 31.0, 69.0),
            (1.25, 0.8, 31.0, 69.0),
        ];

        for group_idx in 0..groups.len() {
            if group_idx == 0 {
                out.push(Self {
                    id,
                    label: "BASELINE_R2_HEDGE".to_string(),
                    group_idx,
                    is_baseline: true,
                    hedge_cfg: base_hedge,
                    position_sizing: None,
                });
                id += 1;
            }

            let mut seen_cfgs: Vec<TreeBoostHedgeConfig> = Vec::with_capacity(140);

            for &(tag, z, exit_z, r_low, r_high, h_sz, tp, h_sl, pk, vol_min, vol_max, mr_vt) in
                &anchors
            {
                let cfg = Self::rg_hedge_config(
                    z, exit_z, r_low, r_high, h_sz, tp, h_sl, 1.40, 0.025, 0.012, pk, 2, mr_vt,
                    vol_min, vol_max, None, None,
                );
                seen_cfgs.push(cfg);
                out.push(Self {
                    id,
                    label: format!("C{:03}_{}_{}", id, tag, Self::format_cfg_tag(&cfg)),
                    group_idx,
                    is_baseline: false,
                    hedge_cfg: cfg,
                    position_sizing: poic10.clone(),
                });
                id += 1;
            }

            let mut push_grid_cfg = |prefix: &str, cfg: TreeBoostHedgeConfig| {
                if seen_cfgs.contains(&cfg) {
                    return;
                }
                seen_cfgs.push(cfg);
                out.push(Self {
                    id,
                    label: format!("C{:03}_{}_{}", id, prefix, Self::format_cfg_tag(&cfg)),
                    group_idx,
                    is_baseline: false,
                    hedge_cfg: cfg,
                    position_sizing: poic10.clone(),
                });
                id += 1;
            };

            for &(z, exit_z, r_low, r_high, h_sz, tp, pk, mr_vt, vol_min, vol_max) in &trail_bases
            {
                for &(trail_act, trail_mult) in &trail_pairs {
                    let cfg = Self::rg_hedge_config(
                        z, exit_z, r_low, r_high, h_sz, tp, 1.40, 1.40, 0.025, 0.012, pk, 2,
                        mr_vt, vol_min, vol_max, trail_act, trail_mult,
                    );
                    push_grid_cfg("GA", cfg);
                }
            }

            for &(trail_act, trail_mult) in &[(None, None), (Some(6.0), Some(4.0))] {
                for &(z, exit_z, r_low, r_high) in &mr_profiles {
                    for &h_sz in &[3.25_f64, 3.50] {
                        for &mr_vt in &[Some(0.022_f64), Some(0.026)] {
                            for &tp in &[3.8_f64, 4.2] {
                                for &pk in &[0.18_f64, 0.19] {
                                    let cfg = Self::rg_hedge_config(
                                        z, exit_z, r_low, r_high, h_sz, tp, 1.40, 1.40, 0.025,
                                        0.012, pk, 2, mr_vt, 0.20, 1.60, trail_act, trail_mult,
                                    );
                                    push_grid_cfg("GB", cfg);
                                }
                            }
                        }
                    }
                }
            }

            for &(z, exit_z, r_low, r_high, h_sz, tp, pk, mr_vt, _, _) in &trail_bases[..3] {
                for &(trail_act, trail_mult) in &trail_pairs {
                    let cfg = Self::rg_hedge_config(
                        z, exit_z, r_low, r_high, h_sz, tp, 1.40, 1.40, 0.025, 0.012, pk, 2,
                        mr_vt, 0.22, 1.35, trail_act, trail_mult,
                    );
                    push_grid_cfg("GC", cfg);
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_candidates_iteration_23_grid() {
        let groups = ModelGroupSpec::build_model_groups(Timeframe::H4);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].label, "M0_BASE");
        assert_eq!(groups[0].model_cfg.horizon, 12);
        assert_eq!(groups[0].model_cfg.num_trees, 35);
        assert_eq!(groups[0].model_cfg.max_depth, 2);
        assert!((groups[0].model_cfg.learning_rate - 0.020).abs() < 1e-6);
        assert!((groups[0].model_cfg.l2_regularization - 0.16).abs() < 1e-6);
        assert!(groups[0].secondary_model_cfg.is_none());
        assert!((groups[0].primary_weight - 1.0).abs() < 1e-12);

        let cands = EvolveCandidate::build_candidates(Timeframe::H4, &groups);
        assert_eq!(cands.len(), 140);
        assert_eq!(cands.iter().filter(|c| c.is_baseline).count(), 1);
        assert_eq!(cands.iter().filter(|c| c.group_idx == 0).count(), 140);

        let base = cands.iter().find(|c| c.is_baseline).unwrap();
        assert_eq!(base.hedge_cfg.mr_vol_target_pct, None);
        assert_eq!(base.hedge_cfg.trail_activate_atr_mult, None);
        assert_eq!(base.hedge_cfg.trailing_atr_mult, None);
        assert!((base.hedge_cfg.vol_min_mult - 0.35).abs() < 1e-12);
        assert!((base.hedge_cfg.vol_max_mult - 1.25).abs() < 1e-12);

        for tag in ["ANC_C110", "ANC_C116", "ANC_C118", "ANC_C046"] {
            let matches: Vec<&EvolveCandidate> =
                cands.iter().filter(|c| c.label.contains(tag)).collect();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].hedge_cfg.trail_activate_atr_mult, None);
            assert_eq!(matches[0].hedge_cfg.trailing_atr_mult, None);
        }

        let anc_c116 = cands
            .iter()
            .find(|c| c.label.contains("ANC_C116"))
            .unwrap();
        assert!((anc_c116.hedge_cfg.hedge_entry_z - 1.30).abs() < 1e-12);
        assert!((anc_c116.hedge_cfg.mr_exit_z - 0.7).abs() < 1e-12);
        assert!((anc_c116.hedge_cfg.mr_rsi_low - 30.0).abs() < 1e-12);
        assert!((anc_c116.hedge_cfg.hedge_size_mult - 3.25).abs() < 1e-12);
        assert!((anc_c116.hedge_cfg.hedge_tp_atr_mult - 4.0).abs() < 1e-12);
        assert_eq!(anc_c116.hedge_cfg.primary_min_ker, Some(0.18));
        assert_eq!(anc_c116.hedge_cfg.mr_vol_target_pct, Some(0.020));
        assert!((anc_c116.hedge_cfg.vol_min_mult - 0.20).abs() < 1e-12);
        assert!((anc_c116.hedge_cfg.vol_max_mult - 1.60).abs() < 1e-12);

        let non_base: Vec<&EvolveCandidate> = cands.iter().filter(|c| !c.is_baseline).collect();
        for (i, a) in non_base.iter().enumerate() {
            for b in non_base.iter().skip(i + 1) {
                assert_ne!(a.hedge_cfg, b.hedge_cfg);
                assert_ne!(a.label, b.label);
            }
        }

        for &(ta, tr) in &[
            (None, None),
            (Some(4.0), Some(3.0)),
            (Some(5.0), Some(3.5)),
            (Some(6.0), Some(4.0)),
            (Some(8.0), Some(4.5)),
            (Some(10.0), Some(5.0)),
        ] {
            assert_eq!(
                non_base
                    .iter()
                    .filter(|c| c.hedge_cfg.trail_activate_atr_mult == ta
                        && c.hedge_cfg.trailing_atr_mult == tr
                        && (c.hedge_cfg.hedge_tp_atr_mult - 4.0).abs() < 1e-12
                        && (c.hedge_cfg.hedge_size_mult - 3.25).abs() < 1e-12)
                    .count(),
                7
            );
        }

        for &(z, ez, rl, rh) in &[
            (1.30, 0.7, 30.0, 70.0),
            (1.28, 0.7, 31.0, 69.0),
            (1.25, 0.8, 31.0, 69.0),
        ] {
            for &sz in &[3.25, 3.50] {
                for &mvt in &[0.022, 0.026] {
                    for &tp in &[3.8, 4.2] {
                        for &pk in &[0.18, 0.19] {
                            assert!(non_base.iter().any(|c| {
                                (c.hedge_cfg.hedge_entry_z - z).abs() < 1e-12
                                    && (c.hedge_cfg.mr_exit_z - ez).abs() < 1e-12
                                    && (c.hedge_cfg.mr_rsi_low - rl).abs() < 1e-12
                                    && (c.hedge_cfg.mr_rsi_high - rh).abs() < 1e-12
                                    && (c.hedge_cfg.hedge_size_mult - sz).abs() < 1e-12
                                    && c.hedge_cfg.mr_vol_target_pct == Some(mvt)
                                    && (c.hedge_cfg.hedge_tp_atr_mult - tp).abs() < 1e-12
                                    && c.hedge_cfg.primary_min_ker == Some(pk)
                            }));
                        }
                    }
                }
            }
        }
    }
}

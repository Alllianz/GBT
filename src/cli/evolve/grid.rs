use crate::models::TreeBoostHedgeConfig;
use crate::types::Timeframe;
use super::spec::ModelGroupSpec;

#[derive(Clone)]
pub struct EvolveCandidate {
    pub id: usize,
    pub label: String,
    pub group_idx: usize,
    pub is_baseline: bool,
    pub hedge_cfg: TreeBoostHedgeConfig,
}

impl EvolveCandidate {
    pub fn build_candidates(tf: Timeframe, _groups: &[ModelGroupSpec]) -> Vec<Self> {
        let base_hedge = TreeBoostHedgeConfig::for_timeframe(Some(tf), true);
        let mut out = Vec::new();
        let mut id = 0usize;

        out.push(Self {
            id,
            label: "BASELINE_R2_HEDGE".to_string(),
            group_idx: 0,
            is_baseline: true,
            hedge_cfg: base_hedge,
        });
        id += 1;

        let primary_sls: [f64; 4] = [1.4, 1.7, 2.0, 2.2];
        let max_sl_pcts: [Option<f64>; 3] = [None, Some(0.028), Some(0.034)];
        let vol_targets: [Option<f64>; 4] = [None, Some(0.015), Some(0.020), Some(0.025)];
        let trailings: [Option<f64>; 2] = [Some(2.6), Some(3.2)];
        let min_holds = [4usize, 5, 6];
        let mr_zs: [f64; 3] = [2.15, 2.35, 2.50];
        let mr_rsis: [(f64, f64); 2] = [(28.0, 72.0), (30.0, 70.0)];
        let mr_sizes: [f64; 2] = [0.25, 0.35];
        let mr_sls: [f64; 2] = [1.4, 1.6];
        let mr_tps: [f64; 2] = [2.2, 2.6];

        for &p_sl in &primary_sls {
            for &m_sl in &max_sl_pcts {
                for &v_tgt in &vol_targets {
                    if (p_sl - 2.2).abs() < 1e-9 && m_sl.is_none() && v_tgt.is_none() {
                        continue;
                    }
                    let cfg = TreeBoostHedgeConfig {
                        primary_sl_atr_mult: p_sl,
                        max_sl_pct: m_sl,
                        vol_target_pct: v_tgt,
                        min_hold_bars: 5,
                        trailing_atr_mult: None,
                        ..base_hedge
                    };
                    out.push(Self {
                        id,
                        label: format!(
                            "C{:03}_SL{:.1}_M{}_V{}",
                            id,
                            p_sl,
                            m_sl.map_or(0, |v| (v * 1000.0).round() as usize),
                            v_tgt.map_or(0, |v| (v * 1000.0).round() as usize),
                        ),
                        group_idx: 0,
                        is_baseline: false,
                        hedge_cfg: cfg,
                    });
                    id += 1;
                }
            }
        }

        let risk_anchors: [(f64, Option<f64>, Option<f64>); 3] = [
            (1.7, Some(0.028), Some(0.020)),
            (2.0, Some(0.028), Some(0.020)),
            (2.2, Some(0.034), Some(0.025)),
        ];
        for &(p_sl, m_sl, v_tgt) in &risk_anchors {
            for &tr in &trailings {
                for &mh in &min_holds {
                    let cfg = TreeBoostHedgeConfig {
                        primary_sl_atr_mult: p_sl,
                        max_sl_pct: m_sl,
                        vol_target_pct: v_tgt,
                        trailing_atr_mult: tr,
                        min_hold_bars: mh,
                        ..base_hedge
                    };
                    out.push(Self {
                        id,
                        label: format!(
                            "C{:03}_SL{:.1}_M{}_V{}_TR{}_MH{}",
                            id,
                            p_sl,
                            m_sl.map_or(0, |v| (v * 1000.0).round() as usize),
                            v_tgt.map_or(0, |v| (v * 1000.0).round() as usize),
                            tr.map_or(0, |v| (v * 10.0).round() as usize),
                            mh,
                        ),
                        group_idx: 0,
                        is_baseline: false,
                        hedge_cfg: cfg,
                    });
                    id += 1;
                }
            }
        }

        for &(mh, tr) in &[
            (4usize, None),
            (6usize, None),
            (5usize, Some(2.6)),
            (5usize, Some(3.2)),
        ] {
            let cfg = TreeBoostHedgeConfig {
                min_hold_bars: mh,
                trailing_atr_mult: tr,
                ..base_hedge
            };
            out.push(Self {
                id,
                label: format!(
                    "C{:03}_BASE_TR{}_MH{}",
                    id,
                    tr.map_or(0, |v| (v * 10.0).round() as usize),
                    mh,
                ),
                group_idx: 0,
                is_baseline: false,
                hedge_cfg: cfg,
            });
            id += 1;
        }

        let hedge_risk_profiles: [(f64, Option<f64>, Option<f64>, Option<f64>, usize); 6] = [
            (1.7, Some(0.028), Some(0.020), None, 5),
            (2.0, Some(0.028), Some(0.020), Some(3.2), 5),
            (1.7, Some(0.034), Some(0.025), Some(2.6), 4),
            (2.2, Some(0.028), Some(0.015), None, 6),
            (1.4, Some(0.028), Some(0.020), Some(3.2), 5),
            (2.0, Some(0.034), Some(0.025), None, 4),
        ];

        let mut mr_idx = 0usize;
        for &z in &mr_zs {
            for &(rsi_low, rsi_high) in &mr_rsis {
                for &h_sz in &mr_sizes {
                    for &h_sl in &mr_sls {
                        for &h_tp in &mr_tps {
                            let (p_sl, m_sl, v_tgt, tr, mh) =
                                hedge_risk_profiles[mr_idx % hedge_risk_profiles.len()];
                            let cfg = TreeBoostHedgeConfig {
                                enabled: true,
                                hedge_size_mult: h_sz,
                                hedge_entry_z: z,
                                hedge_sl_atr_mult: h_sl,
                                hedge_tp_atr_mult: h_tp,
                                primary_sl_atr_mult: p_sl,
                                min_hold_bars: mh,
                                mr_rsi_low: rsi_low,
                                mr_rsi_high: rsi_high,
                                primary_tp_atr_mult: None,
                                trailing_atr_mult: tr,
                                vol_target_pct: v_tgt,
                                max_sl_pct: m_sl,
                                mr_require_rejection: true,
                                hysteresis_frac: 0.25,
                            };
                            out.push(Self {
                                id,
                                label: format!(
                                    "C{:03}_MR_Z{}_R{}_SZ{}_SL{:.1}_TP{:.1}",
                                    id,
                                    (z * 100.0).round() as usize,
                                    rsi_low as usize,
                                    (h_sz * 100.0).round() as usize,
                                    h_sl,
                                    h_tp,
                                ),
                                group_idx: 0,
                                is_baseline: false,
                                hedge_cfg: cfg,
                            });
                            id += 1;
                            mr_idx += 1;
                        }
                    }
                }
            }
        }

        out
    }
}

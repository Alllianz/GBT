use rusqlite::Connection;
use crate::backtest::{BacktestConfig, BacktestEngine};
use crate::features::TrainTestSplit;
use crate::models::{TreeBoostModel, TreeBoostStrategy};
use crate::storage::KlineRepository;
use crate::types::{FitnessMetric, PositionSizing};
use super::backtest_presenter::BacktestPresenter;
use super::evolve::{EvaluatedCandidate, EvolveCandidate, EvolveReport, ModelGroupSpec};
use super::threshold_calibration_helper::ThresholdCalibrationHelper;

pub struct GbtEvolveRunner;

impl GbtEvolveRunner {
    pub fn run(conn: &Connection, tf_str: &str) -> Result<(), String> {
        let tf = ModelGroupSpec::parse_timeframe(tf_str)?;
        let klines = KlineRepository::load_all(conn, "BTCUSDT", tf)
            .map_err(|e| format!("DB error loading BTCUSDT [{}]: {}", tf, e))?;
        if klines.len() < 100 {
            return Err(format!(
                "Insufficient klines for BTCUSDT [{}]: {} (min 100)",
                tf,
                klines.len()
            ));
        }

        let mut base_bt_cfg = BacktestConfig::default();
        base_bt_cfg.timeframe = Some(tf);
        base_bt_cfg.leverage = 20.0;
        base_bt_cfg.position_sizing = PositionSizing::PercentOfInitialCapital(0.10);
        base_bt_cfg.fitness_metric = FitnessMetric::CompositeAllianz;
        base_bt_cfg.allow_short = true;

        let groups = ModelGroupSpec::build_model_groups(tf);
        let candidates = EvolveCandidate::build_candidates(tf, &groups);
        println!(
            "=== GBT EVOLVE [{}] | Klines: {} | ModelGroups: {} | Candidates: {} ===",
            tf.as_db_str(),
            klines.len(),
            groups.len(),
            candidates.len()
        );

        let mut evaluated: Vec<EvaluatedCandidate> = Vec::with_capacity(candidates.len());

        for (group_idx, group) in groups.iter().enumerate() {
            let group_candidates: Vec<&EvolveCandidate> =
                candidates.iter().filter(|c| c.group_idx == group_idx).collect();
            if group_candidates.is_empty() {
                continue;
            }

            let purge_bars = group
                .model_cfg
                .horizon
                .max(group.secondary_model_cfg.as_ref().map_or(0, |s| s.horizon))
                + 2;
            let split = TrainTestSplit::split_chronological(&klines, 0.70, purge_bars)?;

            if let Some(ref sec_cfg) = group.secondary_model_cfg {
                println!(
                    "\n[Group {}/{}] {} (w={:.2} | P: h={},t={},d={},l2={:.2},lr={:.3} + S: h={},t={},d={},l2={:.2},lr={:.3}) | Evaluating {} candidates (IS + OOF + OOS)...",
                    group_idx + 1,
                    groups.len(),
                    group.label,
                    group.primary_weight,
                    group.model_cfg.horizon,
                    group.model_cfg.num_trees,
                    group.model_cfg.max_depth,
                    group.model_cfg.l2_regularization,
                    group.model_cfg.learning_rate,
                    sec_cfg.horizon,
                    sec_cfg.num_trees,
                    sec_cfg.max_depth,
                    sec_cfg.l2_regularization,
                    sec_cfg.learning_rate,
                    group_candidates.len()
                );
            } else {
                println!(
                    "\n[Group {}/{}] {} (h={}, trees={}, depth={}, l2={:.2}, lr={:.3}) | Evaluating {} candidates (IS + OOF + OOS)...",
                    group_idx + 1,
                    groups.len(),
                    group.label,
                    group.model_cfg.horizon,
                    group.model_cfg.num_trees,
                    group.model_cfg.max_depth,
                    group.model_cfg.l2_regularization,
                    group.model_cfg.learning_rate,
                    group_candidates.len()
                );
            }

            let (model, opt_long, opt_short, best_fit, is_preds, oos_preds) =
                if let Some(ref sec_cfg) = group.secondary_model_cfg {
                    let (model_a, model_b, opt_long, opt_short, best_fit) =
                        ThresholdCalibrationHelper::calibrate_and_train_ensemble_oof(
                            &split.train_klines,
                            &group.model_cfg,
                            &base_bt_cfg,
                            group.primary_weight,
                            || TreeBoostModel::new(group.model_cfg.clone()).with_timeframe(Some(tf)),
                            || TreeBoostModel::new(sec_cfg.clone()).with_timeframe(Some(tf)),
                        )?;
                    let w = group.primary_weight;
                    let is_a = model_a.predict_batch(&split.train_klines)?;
                    let is_b = model_b.predict_batch(&split.train_klines)?;
                    let oos_a = model_a.predict_batch(&split.test_warmup_klines)?;
                    let oos_b = model_b.predict_batch(&split.test_warmup_klines)?;
                    let is_preds: Vec<f64> = is_a
                        .iter()
                        .zip(is_b.iter())
                        .map(|(&a, &b)| w * a + (1.0 - w) * b)
                        .collect();
                    let oos_preds: Vec<f64> = oos_a
                        .iter()
                        .zip(oos_b.iter())
                        .map(|(&a, &b)| w * a + (1.0 - w) * b)
                        .collect();
                    (model_a, opt_long, opt_short, best_fit, is_preds, oos_preds)
                } else {
                    let (model, opt_long, opt_short, best_fit) =
                        ThresholdCalibrationHelper::calibrate_and_train_oof(
                            &split.train_klines,
                            &group.model_cfg,
                            &base_bt_cfg,
                            || TreeBoostModel::new(group.model_cfg.clone()).with_timeframe(Some(tf)),
                        )?;
                    let is_preds = model.predict_batch(&split.train_klines)?;
                    let oos_preds = model.predict_batch(&split.test_warmup_klines)?;
                    (model, opt_long, opt_short, best_fit, is_preds, oos_preds)
                };

            let calibrated_model_cfg = model.config().clone();
            let mut strategy = TreeBoostStrategy::new(model);

            for cand in group_candidates {
                let hedge_mode = cand.hedge_cfg.enabled || cand.hedge_cfg.vol_target_pct.is_some();

                let mut cand_bt_cfg = base_bt_cfg.clone();
                cand_bt_cfg.hedge_mode = hedge_mode;
                if let Some(ps) = &cand.position_sizing {
                    cand_bt_cfg.position_sizing = ps.clone();
                }

                let is_bt_cfg = cand_bt_cfg.clone();
                strategy.set_hedge_config(cand.hedge_cfg);
                strategy.set_cached_predictions(is_preds.clone());
                let is_res = BacktestEngine::new(is_bt_cfg)
                    .run(&mut strategy, &split.train_klines)?;

                let oos_res = if is_res.metrics.final_capital > 0.0 {
                    let mut oos_bt_cfg = cand_bt_cfg.clone();
                    oos_bt_cfg.initial_capital = is_res.metrics.final_capital;
                    strategy.set_hedge_config(cand.hedge_cfg);
                    strategy.set_cached_predictions(oos_preds.clone());
                    Some(BacktestEngine::new(oos_bt_cfg).run_with_warmup(
                        &mut strategy,
                        &split.test_warmup_klines,
                        split.warmup_bars,
                    )?)
                } else {
                    None
                };

                evaluated.push(EvaluatedCandidate::score(
                    cand,
                    &group.label,
                    &calibrated_model_cfg,
                    opt_long,
                    opt_short,
                    best_fit,
                    is_res,
                    oos_res,
                ));
            }
        }

        let baseline_snapshot = evaluated
            .iter()
            .find(|c| c.is_baseline)
            .map(EvolveReport::summarize_line);

        evaluated.sort_by(|a, b| {
            b.passes_gate
                .cmp(&a.passes_gate)
                .then_with(|| b.inst_score.total_cmp(&a.inst_score))
        });

        EvolveReport::print(tf, &evaluated, baseline_snapshot.as_deref());

        let winner = evaluated
            .iter()
            .find(|c| c.passes_gate)
            .or_else(|| evaluated.iter().find(|c| c.is_baseline))
            .or_else(|| evaluated.first());

        if let Some(w) = winner {
            let wf = w.to_walk_forward_result(tf, base_bt_cfg.fitness_metric);
            let model_slug = format!("treeboost_evolve_{}", w.label.to_lowercase());
            BacktestPresenter::present_walk_forward(
                &wf.model_name,
                &model_slug,
                tf,
                &wf,
                w.opt_long,
                w.opt_short,
                w.best_fit,
                base_bt_cfg.fitness_metric,
            )
            .map_err(|e| format!("Error presenting evolve walk-forward dashboard: {}", e))?;
        }

        Ok(())
    }
}

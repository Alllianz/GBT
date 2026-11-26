use crate::types::Timeframe;
use super::eval::EvaluatedCandidate;

pub struct EvolveReport;

impl EvolveReport {
    pub fn summarize_line(c: &EvaluatedCandidate) -> String {
        let is_m = &c.is_res.metrics;
        let oos_part = match c.oos_res.as_ref() {
            Some(oos) => {
                let oos_m = &oos.metrics;
                format!(
                    "OOS(PnL:{:+.1}% PF:{:.2} Comp:{:.4} R2:{:.3} DD:{:.1}% CapDD:{:.1} Stag:{:.1}% Uwf:{:.1}% T:{} Liq:{} Trend:{:+.0}({}) MR:{:+.0}({}) Dual:{:.2}) | Rob:{:.2}x",
                    oos_m.net_profit_pct,
                    oos_m.profit_factor,
                    oos_m.composite_score,
                    oos_m.curve_r2,
                    oos_m.max_drawdown_pct,
                    oos_m.capital_to_maxdd_ratio,
                    c.oos_stag * 100.0,
                    c.oos_underwater_freq * 100.0,
                    oos_m.total_trades,
                    c.oos_liq,
                    c.oos_trend_net_pnl,
                    c.oos_trend_trades,
                    c.oos_mr_net_pnl,
                    c.oos_mr_trades,
                    c.oos_dual_regime_score,
                    c.robustness_ratio,
                )
            }
            None => format!(
                "OOS(SKIPPED:IS_Cap<=${:.2}) | Rob:0.00x",
                is_m.final_capital
            ),
        };
        format!(
            "#{} {} [gate={}|score={:.4}] | IS(PnL:{:+.1}% PF:{:.2} Comp:{:.4} R2:{:.3} DD:{:.1}% CapDD:{:.1} Stag:{:.1}% Uwf:{:.1}% T:{} Liq:{} Trend:{:+.0}({}) MR:{:+.0}({}) Dual:{:.2}) | OOF:{:.4} | {}",
            c.id,
            c.label,
            c.passes_gate,
            c.inst_score,
            is_m.net_profit_pct,
            is_m.profit_factor,
            is_m.composite_score,
            is_m.curve_r2,
            is_m.max_drawdown_pct,
            is_m.capital_to_maxdd_ratio,
            c.is_stag * 100.0,
            c.is_underwater_freq * 100.0,
            is_m.total_trades,
            c.is_liq,
            c.is_trend_net_pnl,
            c.is_trend_trades,
            c.is_mr_net_pnl,
            c.is_mr_trades,
            c.is_dual_regime_score,
            c.best_fit,
            oos_part,
        )
    }

    pub fn print(
        tf: Timeframe,
        evaluated: &[EvaluatedCandidate],
        baseline_summary: Option<&str>,
    ) {
        let passed_count = evaluated.iter().filter(|c| c.passes_gate).count();
        let oos_count = evaluated.iter().filter(|c| c.oos_res.is_some()).count();
        let skipped_oos_count = evaluated.len().saturating_sub(oos_count);
        println!(
            "\n==================== EVOLVE-GBT [{}] REPORT ====================",
            tf.as_db_str()
        );
        println!(
            "Total Evaluated: {} | Passed Gate (IS>=120, IS_PnL>0, IS_Cap>0): {} | OOS Evaluated: {} | OOS Skipped (IS_Cap<=0): {}",
            evaluated.len(),
            passed_count,
            oos_count,
            skipped_oos_count
        );

        if let Some(base_line) = baseline_summary {
            println!("\n[BASELINE CHAMPION (Round 2 hedge=true | IS/OOF/OOS)]");
            println!("  {}", base_line);
        }

        println!("\n[ALL CANDIDATES (Ranked strictly on IS + Purged-CV OOF | IS vs OOS Comparison)]");
        for (rank, cand) in evaluated.iter().enumerate() {
            println!("  #{:03}: {}", rank + 1, Self::summarize_line(cand));
        }

        if let Some(winner) = evaluated.first() {
            println!("\n[WINNING CONFIGURATION]");
            println!(
                "  Candidate: {} (Group: {}, Baseline: {}, Gate: {}, InstScore: {:.4})",
                winner.label, winner.group_label, winner.is_baseline, winner.passes_gate, winner.inst_score
            );
            println!(
                "  ModelConfig: horizon={}, num_trees={}, max_depth={}, lr={:.4}, l1={:.4}, l2={:.4}, Th=+{:.4}/{:.4} (OOF: {:.4})",
                winner.model_cfg.horizon,
                winner.model_cfg.num_trees,
                winner.model_cfg.max_depth,
                winner.model_cfg.learning_rate,
                winner.model_cfg.l1_regularization,
                winner.model_cfg.l2_regularization,
                winner.opt_long,
                winner.opt_short,
                winner.best_fit,
            );
            println!("  HedgeConfig: {:?}", winner.hedge_cfg);

            println!("\n[WINNER IS vs OOS EVALUATION]");
            let is_m = &winner.is_res.metrics;
            println!(
                "  IS (PnL:{:+.1}% PF:{:.2} Comp:{:.4} R2:{:.3} DD:{:.1}% CapDD:{:.1} Stag:{:.1}% Uwf:{:.1}% T:{} Liq:{} Trend:{:+.2}({}) MR:{:+.2}({}) Dual:{:.2})",
                is_m.net_profit_pct,
                is_m.profit_factor,
                is_m.composite_score,
                is_m.curve_r2,
                is_m.max_drawdown_pct,
                is_m.capital_to_maxdd_ratio,
                winner.is_stag * 100.0,
                winner.is_underwater_freq * 100.0,
                is_m.total_trades,
                winner.is_liq,
                winner.is_trend_net_pnl,
                winner.is_trend_trades,
                winner.is_mr_net_pnl,
                winner.is_mr_trades,
                winner.is_dual_regime_score,
            );
            if let Some(ref oos) = winner.oos_res {
                let oos_m = &oos.metrics;
                println!(
                    "  OOS(PnL:{:+.1}% PF:{:.2} Comp:{:.4} R2:{:.3} DD:{:.1}% CapDD:{:.1} Stag:{:.1}% Uwf:{:.1}% T:{} Liq:{} Trend:{:+.2}({}) MR:{:+.2}({}) Dual:{:.2}) | Robustness:{:.2}x",
                    oos_m.net_profit_pct,
                    oos_m.profit_factor,
                    oos_m.composite_score,
                    oos_m.curve_r2,
                    oos_m.max_drawdown_pct,
                    oos_m.capital_to_maxdd_ratio,
                    winner.oos_stag * 100.0,
                    winner.oos_underwater_freq * 100.0,
                    oos_m.total_trades,
                    winner.oos_liq,
                    winner.oos_trend_net_pnl,
                    winner.oos_trend_trades,
                    winner.oos_mr_net_pnl,
                    winner.oos_mr_trades,
                    winner.oos_dual_regime_score,
                    winner.robustness_ratio,
                );
            } else {
                println!(
                    "  OOS skipped (IS final_capital <= 0.0: ${:.2}).",
                    is_m.final_capital
                );
            }
        } else {
            println!("\n[WINNER IS vs OOS EVALUATION]");
            println!("  No candidates evaluated.");
        }
        println!("==================================================================");
    }
}

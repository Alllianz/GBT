use crate::backtest::config::BacktestConfig;
use crate::types::{MarginMode, PositionSide};

pub(crate) fn sanitize_dynamic_sl(
    side: PositionSide,
    stop_loss_price: Option<f64>,
    exec_price: f64,
    config: &BacktestConfig,
    entry_commission_rate: f64,
) -> Option<f64> {
    if !exec_price.is_finite() || exec_price <= 0.0 {
        return None;
    }
    let sl = stop_loss_price.filter(|&v| v.is_finite() && v > 0.0)?;
    let raw_dist_pct = match side {
        PositionSide::Long => (exec_price - sl) / exec_price,
        PositionSide::Short => (sl - exec_price) / exec_price,
    };
    let mut dist_pct = raw_dist_pct;
    let mut modified = false;
    if !dist_pct.is_finite() || dist_pct <= 0.0001 {
        if config.fixed_stop_loss_pct.is_some() {
            return None;
        }
        dist_pct = 0.015;
        modified = true;
    }
    if config.margin_mode == MarginMode::Isolated && config.leverage > 1.0 {
        let fee = if entry_commission_rate.is_finite() && entry_commission_rate >= 0.0 {
            entry_commission_rate
        } else {
            0.0
        };
        let max_dist = ((1.0 / config.leverage) - fee - 0.005 - 0.006).max(0.002);
        if dist_pct > max_dist {
            dist_pct = max_dist;
            modified = true;
        }
    }
    if !modified {
        return Some(sl);
    }
    let guarded = match side {
        PositionSide::Long => exec_price * (1.0 - dist_pct),
        PositionSide::Short => exec_price * (1.0 + dist_pct),
    };
    if guarded.is_finite() && guarded > 0.0 {
        Some(guarded)
    } else {
        None
    }
}

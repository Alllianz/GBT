use crate::backtest::config::BacktestConfig;
use crate::backtest::portfolio::Portfolio;
use crate::microstructure::OrderExecutionType;
use crate::risk::RiskEvaluationContext;
use crate::types::{
    CompletedTrade, HedgeLegAction, HedgeSignal, Kline, MarginMode, Position, PositionSide,
    PositionSizing, TradeRegime,
};
use super::accounting::TradeAccounting;
use super::exits::ExitEvaluator;
use super::margin::MarginCalculator;
use super::sizing::PositionSizer;

pub struct HedgeExecutionEngine;

struct LegOpenSpec {
    side: PositionSide,
    raw_notional: f64,
    dynamic_sl: Option<f64>,
    dynamic_tp: Option<f64>,
    regime: TradeRegime,
}

impl HedgeExecutionEngine {
    pub fn process_hedge_signal_at_open(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        signal: &HedgeSignal,
        current_candle: &Kline,
        candle_idx: usize,
        risk_context: Option<&RiskEvaluationContext>,
    ) -> Vec<CompletedTrade> {
        let mut completed_trades = Vec::new();

        if !current_candle.open.is_finite() || current_candle.open <= 0.0 {
            return completed_trades;
        }
        let exec_price = current_candle.open;

        if matches!(signal.long, HedgeLegAction::Close) {
            if let Some(mut pos) = portfolio.long_position.take() {
                pos.update_extremes(exec_price, exec_price);
                completed_trades.push(TradeAccounting::close_position(
                    portfolio,
                    config,
                    pos,
                    exec_price,
                    current_candle.timestamp,
                    candle_idx,
                    "Hedge Signal Exit".to_string(),
                ));
            }
        }

        if matches!(signal.short, HedgeLegAction::Close) {
            if let Some(mut pos) = portfolio.short_position.take() {
                pos.update_extremes(exec_price, exec_price);
                completed_trades.push(TradeAccounting::close_position(
                    portfolio,
                    config,
                    pos,
                    exec_price,
                    current_candle.timestamp,
                    candle_idx,
                    "Hedge Signal Exit".to_string(),
                ));
            }
        }

        let equity = portfolio.current_equity(exec_price);
        if !equity.is_finite() || equity <= 0.0 {
            return completed_trades;
        }

        let mut updated_trades_storage = Vec::new();
        let updated_risk_context;
        let effective_risk_context: Option<&RiskEvaluationContext> =
            if !completed_trades.is_empty() {
                if let Some(ctx) = risk_context {
                    let peak = ctx.peak_equity.max(equity);
                    let updated_dd = if peak > 0.0 {
                        (peak - equity).max(0.0) / peak
                    } else {
                        0.0
                    };
                    updated_trades_storage
                        .reserve(ctx.closed_trades.len() + completed_trades.len());
                    updated_trades_storage.extend_from_slice(ctx.closed_trades);
                    updated_trades_storage.extend(completed_trades.iter().cloned());

                    updated_risk_context = RiskEvaluationContext {
                        equity,
                        leverage: ctx.leverage,
                        exec_price,
                        closed_trades: &updated_trades_storage,
                        current_volatility: ctx.current_volatility,
                        current_drawdown: updated_dd,
                        signal_strength: ctx.signal_strength,
                        dynamic_sl: ctx.dynamic_sl,
                        annualization_factor: ctx.annualization_factor,
                        timeframe: ctx.timeframe,
                        peak_equity: peak,
                    };
                    Some(&updated_risk_context)
                } else {
                    None
                }
            } else {
                risk_context
            };

        let leverage = config.leverage.max(1.0);
        let committed_margin: f64 = [&portfolio.long_position, &portfolio.short_position]
            .into_iter()
            .flatten()
            .map(|pos| {
                pos.isolated_margin
                    .unwrap_or_else(|| pos.notional_value() / leverage)
            })
            .sum();

        let available_collateral = (portfolio.cash_balance.min(equity) - committed_margin).max(0.0);
        if available_collateral <= 0.0 {
            return completed_trades;
        }

        let entry_exec_type = config
            .microstructure
            .as_ref()
            .map_or(OrderExecutionType::Taker, |ms| ms.entry_order_type);
        let entry_commission_rate = config.get_commission_rate(entry_exec_type);

        let long_spec = if portfolio.long_position.is_none() {
            Self::build_leg_open_spec(
                PositionSide::Long,
                signal.long,
                portfolio.initial_capital,
                equity,
                leverage,
                exec_price,
                config,
                entry_commission_rate,
                effective_risk_context,
            )
        } else {
            None
        };

        let short_spec = if config.allow_short && portfolio.short_position.is_none() {
            Self::build_leg_open_spec(
                PositionSide::Short,
                signal.short,
                portfolio.initial_capital,
                equity,
                leverage,
                exec_price,
                config,
                entry_commission_rate,
                effective_risk_context,
            )
        } else {
            None
        };

        let raw_long = long_spec.as_ref().map_or(0.0, |s| s.raw_notional);
        let raw_short = short_spec.as_ref().map_or(0.0, |s| s.raw_notional);
        let total_raw = raw_long + raw_short;
        if total_raw <= 0.0 {
            return completed_trades;
        }

        let cost_per_notional = (1.0 / leverage) + entry_commission_rate;
        let max_supported_total = if cost_per_notional > 0.0 {
            available_collateral / cost_per_notional
        } else {
            0.0
        };
        if max_supported_total <= 0.0 {
            return completed_trades;
        }

        let joint_scale = if total_raw > max_supported_total {
            max_supported_total / total_raw
        } else {
            1.0
        };

        if let Some(spec) = long_spec {
            if let Some(pos) = Self::open_leg(
                portfolio,
                config,
                &spec,
                joint_scale,
                leverage,
                exec_price,
                equity,
                entry_commission_rate,
                current_candle.timestamp,
                candle_idx,
            ) {
                portfolio.long_position = Some(pos);
            }
        }

        if let Some(spec) = short_spec {
            if let Some(pos) = Self::open_leg(
                portfolio,
                config,
                &spec,
                joint_scale,
                leverage,
                exec_price,
                equity,
                entry_commission_rate,
                current_candle.timestamp,
                candle_idx,
            ) {
                portfolio.short_position = Some(pos);
            }
        }

        completed_trades
    }

    #[allow(clippy::too_many_arguments)]
    fn build_leg_open_spec(
        side: PositionSide,
        action: HedgeLegAction,
        initial_capital: f64,
        equity: f64,
        leverage: f64,
        exec_price: f64,
        config: &BacktestConfig,
        entry_commission_rate: f64,
        base_risk_context: Option<&RiskEvaluationContext>,
    ) -> Option<LegOpenSpec> {
        let HedgeLegAction::OpenOrKeep {
            strength,
            stop_loss_price,
            take_profit_price,
            size_multiplier,
            regime,
        } = action
        else {
            return None;
        };

        if !size_multiplier.is_finite() || size_multiplier <= 0.0 {
            return None;
        }

        let dynamic_sl = super::stop_loss_guard::sanitize_dynamic_sl(
            side,
            stop_loss_price,
            exec_price,
            config,
            entry_commission_rate,
        );

        let dynamic_tp = take_profit_price.filter(|&tp| {
            tp.is_finite()
                && tp > 0.0
                && match side {
                    PositionSide::Long => tp > exec_price,
                    PositionSide::Short => tp < exec_price,
                }
        });

        let leg_risk_ctx = base_risk_context.map(|ctx| {
            let mut cloned = ctx.clone();
            cloned.signal_strength = if strength.is_finite() {
                strength.clamp(0.0, 1.0)
            } else {
                0.0
            };
            cloned.dynamic_sl = dynamic_sl;
            cloned
        });

        let base_notional = PositionSizer::calculate_position_size(
            &config.position_sizing,
            initial_capital,
            equity,
            leverage,
            exec_price,
            dynamic_sl,
            config.fixed_stop_loss_pct,
            leg_risk_ctx.as_ref(),
        );

        let raw_notional = if base_notional.is_finite() && base_notional > 0.0 {
            base_notional * size_multiplier
        } else {
            0.0
        };

        if raw_notional <= 0.0 {
            return None;
        }

        Some(LegOpenSpec {
            side,
            raw_notional,
            dynamic_sl,
            dynamic_tp,
            regime,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn open_leg(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        spec: &LegOpenSpec,
        joint_scale: f64,
        leverage: f64,
        exec_price: f64,
        equity: f64,
        entry_commission_rate: f64,
        timestamp: i64,
        candle_idx: usize,
    ) -> Option<Position> {
        let notional_target = spec.raw_notional * joint_scale;
        let required_margin = notional_target / leverage;
        let commission = notional_target * entry_commission_rate;

        if !notional_target.is_finite()
            || notional_target <= 0.0
            || (required_margin + commission) > portfolio.cash_balance + 1e-9
            || (required_margin + commission) > equity + 1e-9
            || portfolio.cash_balance <= commission
        {
            return None;
        }

        let quantity = notional_target / exec_price;
        if !quantity.is_finite() || quantity <= 0.0 {
            return None;
        }

        if config.margin_mode == MarginMode::Cross {
            portfolio.cash_balance -= commission;
            if portfolio.cash_balance < 0.0 {
                portfolio.cash_balance = 0.0;
            }
        }
        portfolio.total_commissions_paid += commission;

        let stop_loss = spec
            .dynamic_sl
            .or_else(|| {
                config.fixed_stop_loss_pct.map(|pct| match spec.side {
                    PositionSide::Long => {
                        (exec_price * (1.0 - pct)).min(exec_price * (1.0 - 0.0001))
                    }
                    PositionSide::Short => {
                        (exec_price * (1.0 + pct)).max(exec_price * (1.0 + 0.0001))
                    }
                })
            })
            .or_else(|| {
                if let PositionSizing::RiskBasedPercent(_) = config.position_sizing {
                    let sl_dist = 0.02;
                    Some(match spec.side {
                        PositionSide::Long => {
                            (exec_price * (1.0 - sl_dist)).min(exec_price * (1.0 - 0.0001))
                        }
                        PositionSide::Short => {
                            (exec_price * (1.0 + sl_dist)).max(exec_price * (1.0 + 0.0001))
                        }
                    })
                } else {
                    None
                }
            });

        let take_profit = spec.dynamic_tp.or_else(|| {
            config.fixed_take_profit_pct.map(|pct| match spec.side {
                PositionSide::Long => exec_price * (1.0 + pct),
                PositionSide::Short => exec_price * (1.0 - pct),
            })
        });

        let post_entry_equity = portfolio.current_equity(exec_price);
        let margin_equity = match config.margin_mode {
            MarginMode::Cross => post_entry_equity,
            MarginMode::Isolated => (post_entry_equity - portfolio.total_isolated_margin()).max(0.0),
        };
        let (isolated_margin, liq_price) = MarginCalculator::calculate_margin_and_liquidation(
            config.margin_mode,
            spec.side,
            exec_price,
            quantity,
            margin_equity,
            notional_target,
            leverage,
            commission,
        );

        Some(
            Position::new(
                spec.side,
                exec_price,
                quantity,
                timestamp,
                candle_idx,
                stop_loss,
                take_profit,
                isolated_margin,
                liq_price,
                commission,
            )
            .with_regime(spec.regime),
        )
    }

    pub fn check_intra_candle_hedge_exits(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        current_candle: &Kline,
        candle_idx: usize,
    ) -> Vec<CompletedTrade> {
        let mut closed_trades = Vec::new();
        if let Some(trade) =
            Self::check_leg_exit(portfolio, config, true, current_candle, candle_idx)
        {
            closed_trades.push(trade);
        }
        if let Some(trade) =
            Self::check_leg_exit(portfolio, config, false, current_candle, candle_idx)
        {
            closed_trades.push(trade);
        }
        closed_trades
    }

    fn check_leg_exit(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        is_long: bool,
        current_candle: &Kline,
        candle_idx: usize,
    ) -> Option<CompletedTrade> {
        if !current_candle.open.is_finite() || current_candle.open <= 0.0 {
            return None;
        }
        let cash_balance = portfolio.cash_balance;
        let opposing_cross_adjustment = if is_long {
            portfolio.short_position.as_ref()
        } else {
            portfolio.long_position.as_ref()
        }
        .filter(|opp| opp.isolated_margin.is_none())
        .map_or(0.0, |opp| {
            opp.unrealized_pnl(current_candle.open) - (0.005 * opp.quantity * current_candle.open)
        });
        let joint_cross_equity = (cash_balance + opposing_cross_adjustment).max(0.0);

        let slot = if is_long {
            &mut portfolio.long_position
        } else {
            &mut portfolio.short_position
        };
        let pos = slot.as_mut()?;

        if pos.isolated_margin.is_none() {
            pos.liquidation_price = MarginCalculator::calculate_cross_liquidation(
                pos.side,
                pos.entry_price,
                pos.quantity,
                joint_cross_equity,
            );
        } else if let Some(eff_margin) = pos.effective_margin() {
            pos.liquidation_price = MarginCalculator::calculate_isolated_liquidation(
                pos.side,
                pos.entry_price,
                pos.quantity,
                eff_margin,
            );
        }

        let exit_event = ExitEvaluator::evaluate_intra_candle(pos, current_candle);
        if let Some((exit_target_price, reason)) = exit_event {
            let mut pos = slot.take()?;
            Self::update_exit_extremes(&mut pos, current_candle, exit_target_price);
            Some(TradeAccounting::close_position(
                portfolio,
                config,
                pos,
                exit_target_price,
                current_candle.timestamp,
                candle_idx,
                reason.to_string(),
            ))
        } else {
            pos.update_extremes(current_candle.high, current_candle.low);
            None
        }
    }

    fn update_exit_extremes(pos: &mut Position, current_candle: &Kline, exit_target_price: f64) {
        let exited_at_open = (exit_target_price - current_candle.open).abs() < 1e-9;
        if exited_at_open {
            pos.update_extremes(current_candle.open, current_candle.open);
        } else {
            match pos.side {
                PositionSide::Long => {
                    let effective_high = if exit_target_price >= pos.entry_price {
                        current_candle.high.min(exit_target_price)
                    } else {
                        current_candle.high
                    };
                    let effective_low = if exit_target_price <= pos.entry_price {
                        current_candle.low.max(exit_target_price)
                    } else {
                        current_candle.low
                    };
                    pos.update_extremes(effective_high, effective_low);
                }
                PositionSide::Short => {
                    let effective_low = if exit_target_price <= pos.entry_price {
                        current_candle.low.max(exit_target_price)
                    } else {
                        current_candle.low
                    };
                    let effective_high = if exit_target_price >= pos.entry_price {
                        current_candle.high.min(exit_target_price)
                    } else {
                        current_candle.high
                    };
                    pos.update_extremes(effective_high, effective_low);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Kline {
        Kline {
            timestamp: ts,
            open,
            high,
            low,
            close,
            volume: 100.0,
        }
    }

    #[test]
    fn test_hedge_opens_both_legs_and_caps_jointly() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            initial_capital: 1000.0,
            leverage: 10.0,
            commission_rate: 0.0,
            position_sizing: PositionSizing::PercentOfEquity(0.8),
            hedge_mode: true,
            ..Default::default()
        };
        let sig = HedgeSignal {
            timestamp: 1000,
            long: HedgeLegAction::open_or_keep(1.0, None, None, 1.0),
            short: HedgeLegAction::open_or_keep(1.0, None, None, 1.0),
        };
        let c1 = candle(1000, 100.0, 105.0, 95.0, 100.0);
        let closed = HedgeExecutionEngine::process_hedge_signal_at_open(
            &mut portfolio,
            &config,
            &sig,
            &c1,
            0,
            None,
        );
        assert!(closed.is_empty());
        assert!(portfolio.has_long());
        assert!(portfolio.has_short());
        let l = portfolio.long_position.as_ref().unwrap();
        let s = portfolio.short_position.as_ref().unwrap();
        let total_notional = l.notional_value() + s.notional_value();
        assert!((total_notional - 10_000.0).abs() < 1e-4);
        assert!((l.notional_value() - s.notional_value()).abs() < 1e-4);
    }

    #[test]
    fn test_hedge_closes_leg_first_and_runs_independent_exits() {
        let mut portfolio = Portfolio::new(10_000.0);
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            leverage: 2.0,
            commission_rate: 0.0,
            position_sizing: PositionSizing::PercentOfEquity(0.25),
            hedge_mode: true,
            ..Default::default()
        };
        let open_both = HedgeSignal {
            timestamp: 1000,
            long: HedgeLegAction::open_or_keep(1.0, Some(90.0), Some(120.0), 1.0),
            short: HedgeLegAction::open_or_keep(1.0, Some(110.0), Some(80.0), 1.0),
        };
        let c1 = candle(1000, 100.0, 102.0, 98.0, 100.0);
        HedgeExecutionEngine::process_hedge_signal_at_open(
            &mut portfolio,
            &config,
            &open_both,
            &c1,
            0,
            None,
        );
        assert!(portfolio.has_long());
        assert!(portfolio.has_short());

        let c2 = candle(2000, 100.0, 112.0, 99.0, 105.0);
        let exits =
            HedgeExecutionEngine::check_intra_candle_hedge_exits(&mut portfolio, &config, &c2, 1);
        assert_eq!(exits.len(), 1);
        assert_eq!(exits[0].side, PositionSide::Short);
        assert_eq!(exits[0].exit_price, 110.0);
        assert!(portfolio.has_long());
        assert!(!portfolio.has_short());
    }

    #[test]
    fn test_cross_hedge_liquidation_accounts_for_opposing_leg_pnl_and_maint_margin() {
        let mut portfolio = Portfolio::new(200.0);
        let config = BacktestConfig {
            initial_capital: 200.0,
            margin_mode: MarginMode::Cross,
            leverage: 10.0,
            commission_rate: 0.0,
            hedge_mode: true,
            ..Default::default()
        };

        portfolio.long_position = Some(Position::new(
            PositionSide::Long,
            100.0,
            5.0,
            1000,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        ));
        portfolio.short_position = Some(Position::new(
            PositionSide::Short,
            100.0,
            10.0,
            1000,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        ));

        let c_gap = candle(2000, 115.0, 115.0, 90.0, 95.0);
        let exits =
            HedgeExecutionEngine::check_intra_candle_hedge_exits(&mut portfolio, &config, &c_gap, 1);

        assert_eq!(exits.len(), 1);
        assert_eq!(exits[0].side, PositionSide::Long);
        assert_eq!(exits[0].exit_reason, "Liquidated (Margin Call)");
        let expected_joint_collat = 200.0 + (-150.0) - (0.005 * 10.0 * 115.0);
        let expected_liq = (100.0 - (expected_joint_collat / 5.0)) / 0.995;
        assert!((exits[0].exit_price - expected_liq).abs() < 1e-6);
    }
}

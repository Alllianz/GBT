use crate::backtest::config::BacktestConfig;
use crate::backtest::portfolio::Portfolio;
use crate::types::{
    CompletedTrade, Kline, Position, PositionSide, PositionSizing, Signal, SignalDirection,
};
use super::accounting::TradeAccounting;
use super::exits::ExitEvaluator;
use super::margin::MarginCalculator;
use super::sizing::PositionSizer;

pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Procesa una señal pendiente de la vela anterior (t) en el Open de la vela actual (t+1)
    pub fn process_signal_at_open(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        signal: &Signal,
        current_candle: &Kline,
        candle_idx: usize,
    ) -> Option<CompletedTrade> {
        Self::process_signal_at_open_with_risk(
            portfolio,
            config,
            signal,
            current_candle,
            candle_idx,
            None,
        )
    }

    /// Procesa una señal con contexto de riesgo dinámico e histórico causal (t <= T)
    pub fn process_signal_at_open_with_risk(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        signal: &Signal,
        current_candle: &Kline,
        candle_idx: usize,
        risk_context: Option<&crate::risk::RiskEvaluationContext>,
    ) -> Option<CompletedTrade> {
        let mut completed_trade = None;

        if !current_candle.open.is_finite() || current_candle.open <= 0.0 {
            return completed_trade;
        }

        // Si ya hay una posición abierta y la señal exige cambio o reversión (Long <-> Short) o cierre (Flat)
        if let Some(mut pos) = portfolio.active_position.take() {
            let is_opposite = matches!(
                (pos.side, signal.direction),
                (PositionSide::Long, SignalDirection::Short)
                    | (PositionSide::Short, SignalDirection::Long)
            );
            let is_flat = signal.direction == SignalDirection::Flat;

            if is_opposite || is_flat {
                // BUG-32 & C-14: Actualizar extremos MFE/MAE únicamente con el precio Open de la vela de salida
                pos.update_extremes(current_candle.open, current_candle.open);
                let reason = if is_flat {
                    "Signal Exit".to_string()
                } else {
                    "Signal Reversal".to_string()
                };
                let trade = Self::close_position(
                    portfolio,
                    config,
                    pos,
                    current_candle.open,
                    current_candle.timestamp,
                    candle_idx,
                    reason,
                );
                completed_trade = Some(trade);
            } else {
                // Conservar la posición existente si la señal es en la misma dirección
                portfolio.active_position = Some(pos);
            }
        }

        // Si ahora no hay posición y la señal pide entrar
        if portfolio.active_position.is_none() && signal.direction != SignalDirection::Flat {
            let side = match signal.direction {
                SignalDirection::Long => PositionSide::Long,
                SignalDirection::Short => {
                    if !config.allow_short {
                        return completed_trade;
                    }
                    PositionSide::Short
                }
                SignalDirection::Flat => return completed_trade,
            };

            // Directriz institucional de Allianz: GBT no utiliza slippage.
            // El precio de ejecución de entrada es exactamente el precio Open de la vela actual.
            let exec_price = current_candle.open;

            // Validación de Open y precio de ejecución (BUG-25)
            if !current_candle.open.is_finite()
                || current_candle.open <= 0.0
                || !exec_price.is_finite()
                || exec_price <= 0.0
            {
                return completed_trade;
            }

            let equity = portfolio.current_equity(exec_price);
            if equity <= 0.0 {
                return completed_trade;
            }

            let entry_exec_type = if let Some(ref ms) = config.microstructure {
                ms.entry_order_type
            } else {
                crate::microstructure::OrderExecutionType::Taker
            };
            let entry_commission_rate = config.get_commission_rate(entry_exec_type);

            let dynamic_sl = super::stop_loss_guard::sanitize_dynamic_sl(
                side,
                signal.stop_loss_price,
                exec_price,
                config,
                entry_commission_rate,
            );

            let dynamic_tp = signal.take_profit_price.filter(|&tp| {
                tp.is_finite()
                    && tp > 0.0
                    && match side {
                        PositionSide::Long => tp > exec_price,
                        PositionSide::Short => tp < exec_price,
                    }
            });

            // C-06: En una reversión al Open (cerrar la posición vieja y abrir la contraria en la misma vela),
            // el RiskEvaluationContext usado para dimensionar la nueva posición debe reflejar el estado
            // INMEDIATAMENTE POSTERIOR al cierre:
            // 1. Equity/caja tras realizar el PnL y la comisión de salida (portfolio.current_equity(exec_price))
            // 2. El historial de trades incluyendo el trade recién cerrado
            // 3. El drawdown causal actualizado respecto al pico histórico (peak_equity)
            // La volatilidad causal solo usa velas cerradas (< T), ya encapsulada causalmente.
            let mut updated_trades_storage = Vec::new();
            let updated_risk_context;
            let effective_risk_context: Option<&crate::risk::RiskEvaluationContext> =
                if let Some(ref trade) = completed_trade {
                    if let Some(ctx) = risk_context {
                        let post_close_equity = equity;
                        let peak = ctx.peak_equity.max(post_close_equity);
                        let updated_dd = if peak > 0.0 {
                            (peak - post_close_equity).max(0.0) / peak
                        } else {
                            0.0
                        };

                        updated_trades_storage.reserve(ctx.closed_trades.len() + 1);
                        updated_trades_storage.extend_from_slice(ctx.closed_trades);
                        updated_trades_storage.push(trade.clone());

                        updated_risk_context = crate::risk::RiskEvaluationContext {
                            equity: post_close_equity,
                            leverage: ctx.leverage,
                            exec_price,
                            closed_trades: &updated_trades_storage,
                            current_volatility: ctx.current_volatility,
                            current_drawdown: updated_dd,
                            signal_strength: ctx.signal_strength,
                            dynamic_sl,
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

            // Asignación de tamaño con PositionSizer
            let leverage = config.leverage.max(1.0);
            let raw_notional = PositionSizer::calculate_position_size(
                &config.position_sizing,
                portfolio.initial_capital,
                equity,
                leverage,
                exec_price,
                dynamic_sl,
                config.fixed_stop_loss_pct,
                effective_risk_context,
            );

            // BUG-F05: Validación conjunta de margen requerido + comisión estimada <= colateral disponible
            let available_collateral = portfolio.cash_balance.min(equity);
            if available_collateral <= 0.0 || raw_notional <= 0.0 {
                return completed_trade;
            }

            let cost_per_notional = (1.0 / leverage) + entry_commission_rate;
            let max_supported_notional = if cost_per_notional > 0.0 {
                available_collateral / cost_per_notional
            } else {
                0.0
            };

            let notional_target = raw_notional.min(max_supported_notional);
            let required_margin = notional_target / leverage;
            let commission = notional_target * entry_commission_rate;

            // Verificación estricta de solvencia: margen + comisión no deben exceder el balance disponible
            if notional_target <= 0.0
                || (required_margin + commission) > portfolio.cash_balance + 1e-9
                || (required_margin + commission) > equity + 1e-9
                || portfolio.cash_balance <= commission
            {
                return completed_trade;
            }

            let quantity = notional_target / exec_price;

            if quantity > 0.0 {
                // C-34: Se eliminó la creación y descarte inútil (_entry_order, _entry_fill) sin tracking.

                // C-03: En margen Cross, la comisión de entrada se debita del balance libre.
                // En margen Isolated, la comisión sale del margen aislado asignado (no hay doble deducción de cash).
                if config.margin_mode == crate::types::MarginMode::Cross {
                    portfolio.cash_balance -= commission;
                    if portfolio.cash_balance < 0.0 {
                        portfolio.cash_balance = 0.0;
                    }
                }
                portfolio.total_commissions_paid += commission;

                // Bounding de stop loss respecto a exec_price y Open
                let stop_loss = dynamic_sl
                    .or_else(|| {
                        config.fixed_stop_loss_pct.map(|pct| match side {
                            PositionSide::Long => {
                                (exec_price * (1.0 - pct)).min(current_candle.open * (1.0 - 0.0001))
                            }
                            PositionSide::Short => {
                                (exec_price * (1.0 + pct)).max(current_candle.open * (1.0 + 0.0001))
                            }
                        })
                    })
                    .or_else(|| {
                        if let PositionSizing::RiskBasedPercent(_) = config.position_sizing {
                            let sl_dist = 0.02;
                            Some(match side {
                                PositionSide::Long => {
                                    (exec_price * (1.0 - sl_dist)).min(current_candle.open * (1.0 - 0.0001))
                                }
                                PositionSide::Short => {
                                    (exec_price * (1.0 + sl_dist)).max(current_candle.open * (1.0 + 0.0001))
                                }
                            })
                        } else {
                            None
                        }
                    });

                let take_profit = dynamic_tp.or_else(|| {
                    config.fixed_take_profit_pct.map(|pct| match side {
                        PositionSide::Long => exec_price * (1.0 + pct),
                        PositionSide::Short => exec_price * (1.0 - pct),
                    })
                });

                // C-03 & C-13: Cálculo de margen y precio de liquidación con margen efectivo neto de comisión
                let post_entry_equity = portfolio.current_equity(exec_price);
                let (isolated_margin, liq_price) = MarginCalculator::calculate_margin_and_liquidation(
                    config.margin_mode,
                    side,
                    exec_price,
                    quantity,
                    post_entry_equity,
                    notional_target,
                    leverage,
                    commission,
                );

                let new_pos = Position::new(
                    side,
                    exec_price,
                    quantity,
                    current_candle.timestamp,
                    candle_idx,
                    stop_loss,
                    take_profit,
                    isolated_margin,
                    liq_price,
                    commission,
                );
                portfolio.active_position = Some(new_pos);
            }
        }

        completed_trade
    }

    /// Evalúa Stop Loss, Take Profit y Liquidación intra-vela
    pub fn check_intra_candle_exits(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        current_candle: &Kline,
        candle_idx: usize,
    ) -> Option<CompletedTrade> {
        let pos = portfolio.active_position.as_mut()?;

        // C-10: En margen Cross, recalcular causalmente el precio de liquidación con el colateral vigente
        // antes de evaluar los exits de la vela.
        // C-11: En margen Isolated, recalcular causalmente el precio de liquidación con el margen efectivo neto
        // (descontando comisión de apertura y funding devengado) antes de evaluar salidas intra-vela.
        if pos.isolated_margin.is_none() {
            pos.liquidation_price = MarginCalculator::calculate_cross_liquidation(
                pos.side,
                pos.entry_price,
                pos.quantity,
                portfolio.cash_balance,
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
            let mut pos = portfolio.active_position.take()?;

            // C-14: MFE/MAE no deben actualizarse con el High/Low de una vela en la que el trade ya salió al Open (por gap o por señal).
            // Solo cuentan los extremos de las velas en las que la posición estuvo viva, hasta el precio de salida.
            let exited_at_open = (exit_target_price - current_candle.open).abs() < 1e-9;
            if exited_at_open {
                // Salió al Open por Gap: solo registrar el precio de salida Open
                pos.update_extremes(current_candle.open, current_candle.open);
            } else {
                // Salió intra-vela: registrar extremos acotados al precio de salida real
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

            let trade = Self::close_position(
                portfolio,
                config,
                pos,
                exit_target_price,
                current_candle.timestamp,
                candle_idx,
                reason.to_string(),
            );
            Some(trade)
        } else {
            // La posición permaneció viva toda la vela: actualizar extremos con High y Low completos
            pos.update_extremes(current_candle.high, current_candle.low);
            None
        }
    }

    /// Cierra una posición delegando en el subsistema de contabilidad
    #[inline]
    pub fn close_position(
        portfolio: &mut Portfolio,
        config: &BacktestConfig,
        pos: Position,
        target_price: f64,
        exit_timestamp: i64,
        candle_idx: usize,
        reason: String,
    ) -> CompletedTrade {
        TradeAccounting::close_position(
            portfolio,
            config,
            pos,
            target_price,
            exit_timestamp,
            candle_idx,
            reason,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug_f05_no_negative_balance_on_100pct_margin() {
        let mut portfolio = Portfolio::new(1000.0);
        let mut config = BacktestConfig::default();
        config.leverage = 10.0;
        config.position_sizing = PositionSizing::PercentOfEquity(1.0); // 100% equity

        let candle = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 1.0,
        };

        let signal = Signal::new(SignalDirection::Long, 1.0, 1000);
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &signal, &candle, 1);
        assert!(trade.is_none());
        assert!(portfolio.has_position());

        // El balance de caja DEBE ser positivo y cubrir el margen requerido
        assert!(portfolio.cash_balance > 0.0, "Balance de caja no debe ser negativo");
        let pos = portfolio.active_position.as_ref().unwrap();
        let notional = pos.quantity * pos.entry_price;
        let required_margin = notional / config.leverage;
        let fee_rate = config.get_commission_rate(crate::microstructure::OrderExecutionType::Taker);
        let commission = notional * fee_rate;

        // Validar que margen + comisión no superaron los 1000.0 originales
        assert!(required_margin + commission <= 1000.0 + 1e-6);
        assert!(portfolio.cash_balance >= required_margin - 1e-6);
    }

    #[test]
    fn test_c06_reversal_sizes_with_post_close_equity_numerical() {
        // Caso numérico riguroso de Allianz para reversiones al Open:
        // Capital inicial: $10,000. Leverage: 1.0. Comisión: 0.0 para aislar el efecto de sizing.
        let mut portfolio = Portfolio::new(10_000.0);
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            leverage: 1.0,
            commission_rate: 0.0,
            position_sizing: PositionSizing::PercentOfEquity(0.50), // 50% de equity
            ..Default::default()
        };

        // 1. Abrir Long a 100.0: nocional = 10,000 * 0.50 = 5,000 USD (cantidad = 50.0 BTC)
        let c1 = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 10.0,
        };
        let enter_long = Signal::new(SignalDirection::Long, 1.0, 1000);
        let trade1 = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_long, &c1, 0);
        assert!(trade1.is_none());
        assert_eq!(portfolio.active_position.as_ref().unwrap().quantity, 50.0);

        // 2. Siguiente vela abre con gap a la baja en 80.0. Estrategia emite reversión a Short.
        let c2 = Kline {
            timestamp: 2000,
            open: 80.0,
            high: 85.0,
            low: 75.0,
            close: 80.0,
            volume: 10.0,
        };
        let reversal_short = Signal::new(SignalDirection::Short, 1.0, 2000);

        let trade2 = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &reversal_short, &c2, 1);
        assert!(trade2.is_some(), "La reversión debe retornar el trade cerrado");
        let completed = trade2.unwrap();
        // Pérdida realizada: (80.0 - 100.0) * 50.0 = -1,000.0 USD
        assert_eq!(completed.net_pnl, -1000.0);
        assert_eq!(completed.exit_price, 80.0);

        // Balance de caja post-cierre = 10,000 - 1,000 = 9,000 USD
        assert_eq!(portfolio.cash_balance, 9000.0);

        // La nueva posición en Short DEBE dimensionarse estrictamente con el equity post-cierre ($9,000):
        // Nocional esperado = 9,000 * 0.50 = 4,500 USD (y NO 10,000 * 0.50 = 5,000 USD)
        // Cantidad esperada = 4,500 / 80.0 = 56.25 BTC
        assert!(portfolio.has_position());
        let pos_short = portfolio.active_position.as_ref().unwrap();
        assert_eq!(pos_short.side, PositionSide::Short);
        assert_eq!(pos_short.entry_price, 80.0);
        let notional = pos_short.quantity * pos_short.entry_price;
        assert_eq!(notional, 4500.0, "Nocional debe ser 4500.0 (50% de $9,000 post-cierre)");
        assert_eq!(pos_short.quantity, 56.25);
    }

    #[test]
    fn test_c06_reversal_dynamic_risk_incorporates_closed_trade_and_drawdown() {
        use crate::risk::RiskConfig;

        let mut portfolio = Portfolio::new(10_000.0);
        let risk_cfg = RiskConfig {
            kelly_default_fraction: 0.10,
            use_volatility_targeting: false,
            use_conviction: false,
            use_drawdown_governor: true,
            max_tolerable_drawdown: 0.20, // 20% max DD
            drawdown_gamma: 1.0,          // Penalización lineal
            drawdown_min_scale: 0.0,
            ..Default::default()
        };

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            leverage: 1.0,
            commission_rate: 0.0,
            position_sizing: PositionSizing::DynamicRisk(risk_cfg),
            ..Default::default()
        };

        // 1. Abrir Long a 100.0 con 50 BTC ($5,000 nocional)
        let _c1 = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 10.0,
        };
        portfolio.active_position = Some(Position::new(
            PositionSide::Long,
            100.0,
            50.0,
            1000,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        ));

        // 2. Reversión a Short a 80.0:
        // Cierre de Long genera pérdida de -$1,000. Cash baja de $10,000 a $9,000.
        // Peak equity = $10,000 -> Drawdown actualizado = ($10,000 - $9,000) / $10,000 = 10% (0.10)
        // Factor de drawdown governor = 1.0 - (0.10 / 0.20) = 0.50 (50% de reducción)
        // Base weight = 0.10 * 0.50 = 0.05
        // Nocional objetivo Short = $9,000 (equity post-cierre) * 0.05 = $450.0 USD
        let c2 = Kline {
            timestamp: 2000,
            open: 80.0,
            high: 85.0,
            low: 75.0,
            close: 80.0,
            volume: 10.0,
        };
        let rev_sig = Signal::new(SignalDirection::Short, 1.0, 2000);

        let trades: Vec<CompletedTrade> = Vec::new();
        let pre_reversal_ctx = crate::risk::RiskEvaluationContext::new(
            10_000.0, // Contexto original antes del cierre
            1.0,
            80.0,
            &trades,
            None,
            0.0, // 0% DD pre-cierre
            1.0,
            None,
        ).with_peak_equity(10_000.0);

        let trade = ExecutionEngine::process_signal_at_open_with_risk(
            &mut portfolio,
            &config,
            &rev_sig,
            &c2,
            1,
            Some(&pre_reversal_ctx),
        );

        assert!(trade.is_some());
        assert_eq!(portfolio.cash_balance, 9000.0);

        let pos = portfolio.active_position.as_ref().unwrap();
        let notional = pos.quantity * pos.entry_price;
        // Debe ser exactamente 450.0 USD (y no 1000.0 USD si hubiera ignorado el cierre)
        assert!((notional - 450.0).abs() < 1e-4, "Nocional debe ser 450.0 USD, obtenido: {}", notional);
    }

    #[test]
    fn test_c21_volatility_targeting_sizing_across_all_four_allianz_timeframes() {
        use crate::types::Timeframe;

        let timeframes = [
            (Timeframe::M15, 35040.0),
            (Timeframe::H1, 8760.0),
            (Timeframe::H4, 2190.0),
            (Timeframe::D1, 365.0),
        ];

        for (tf, expected_periods) in timeframes {
            assert_eq!(tf.periods_per_year(), expected_periods);
            let ann_factor = tf.annualization_factor();
            assert_eq!(ann_factor, expected_periods.sqrt());

            // Target 20% anual con volatilidad por vela = 0.20 / ann_factor => peso 1.0
            let exact_bar_vol = 0.20 / ann_factor;
            let sizing = PositionSizing::VolatilityTargeting {
                target_annual_vol: 0.20,
                max_leverage: 2.0,
            };

            let trades: Vec<CompletedTrade> = Vec::new();
            let ctx = crate::risk::RiskEvaluationContext::new(
                10_000.0,
                1.0,
                100.0,
                &trades,
                Some(exact_bar_vol),
                0.0,
                1.0,
                None,
            ).with_timeframe(tf);

            let notional = PositionSizer::calculate_position_size(
                &sizing,
                10_000.0,
                10_000.0,
                1.0,
                100.0,
                None,
                None,
                Some(&ctx),
            );

            // Weight 1.0 * equity 10,000 = 10,000.0 USD
            assert!(
                (notional - 10_000.0).abs() < 1e-4,
                "Para timeframe {:?}, notional debe ser 10,000.0, obtenido: {}",
                tf,
                notional
            );
        }
    }
}


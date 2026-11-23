use crate::types::{CompletedTrade, Kline, Signal};
use super::close_equity_evaluator::CloseEquityEvaluator;
use super::config::BacktestConfig;
use super::equity_point::EquityPoint;
use super::execution_engine::ExecutionEngine;
use super::metrics::BacktestMetrics;
use super::metrics_calculator::MetricsCalculator;
use super::portfolio::Portfolio;
use super::strategy::Strategy;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: BacktestMetrics,
    pub trades: Vec<CompletedTrade>,
    pub equity_curve: Vec<EquityPoint>,
}

pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        strategy: &mut dyn Strategy,
        klines: &[Kline],
    ) -> Result<BacktestResult, String> {
        self.run_with_warmup(strategy, klines, 0)
    }

    pub fn run_with_warmup(
        &self,
        strategy: &mut dyn Strategy,
        klines: &[Kline],
        warmup_bars: usize,
    ) -> Result<BacktestResult, String> {
        if klines.len() < warmup_bars + 2 {
            return Err("Se requieren al menos 2 velas para ejecutar el backtest".to_string());
        }

        let mut portfolio = Portfolio::new(self.config.initial_capital);
        let mut trades = Vec::new();
        let mut equity_curve = Vec::with_capacity(klines.len().saturating_sub(warmup_bars));
        let ms_engine = self.config.microstructure.map(crate::microstructure::MicrostructureEngine::new);

        let candle_duration: Option<i64> = if let Some(tf) = self.config.timeframe {
            let is_sec = ms_engine.as_ref().map_or(false, |m| {
                m.funding_simulator.config.timestamp_unit == crate::microstructure::TimestampUnit::Seconds
            });
            if is_sec {
                Some(tf.duration_seconds())
            } else {
                Some(tf.duration_millis())
            }
        } else if klines.len() >= 2 {
            Some((klines[1].timestamp - klines[0].timestamp).abs())
        } else {
            None
        };

        let mut pending_signal: Option<Signal> = None;
        let mut peak_equity = self.config.initial_capital;

        let mut last_eval_idx = 0;
        let mut is_bankrupt = false;

        for i in 0..klines.len() {
            last_eval_idx = i;
            let current_candle = &klines[i];

            // Durante la fase de calentamiento (< warmup_bars), no se procesan órdenes en el Open ni salidas intra-vela.
            // C-33 & C-08: Invocamos strategy.evaluate(&klines[..=i], i) en cada barra de calentamiento pasando estrictamente
            // el slice causal para que las estrategias con estado interno calienten su memoria sin look-ahead.
            if i < warmup_bars {
                let sig = strategy.evaluate(&klines[..=i], i);
                // En la vela inmediatamente anterior al inicio de test (i + 1 == warmup_bars),
                // retenemos la señal para que la primera vela de prueba (en i == warmup_bars)
                // pueda ejecutar la orden al Open si el modelo emite señal.
                if i + 1 == warmup_bars {
                    pending_signal = Some(sig);
                } else {
                    let _ = sig;
                }
                continue;
            }

            let eval_idx = i.saturating_sub(warmup_bars);

            // 1. Si hay una señal generada en el cierre de la vela anterior (t-1), se ejecuta en el Open de la vela actual (t)
            if let Some(signal) = pending_signal.take() {
                // Estimación causal de volatilidad sobre velas cerradas previas (t <= T, velas < i)
                let current_volatility = if i >= 2 {
                    let lookback = i.saturating_sub(20);
                    let window_klines = &klines[lookback..i];
                    if window_klines.len() >= 2 {
                        let closes: Vec<f64> = window_klines.iter().map(|k| k.close).collect();
                        crate::risk::VolatilityTargetingSizer::calculate_realized_volatility(&closes)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let current_eq = portfolio.current_equity(current_candle.open);
                let current_dd = if peak_equity > 0.0 {
                    (peak_equity - current_eq).max(0.0) / peak_equity
                } else {
                    0.0
                };

                let risk_ctx = crate::risk::RiskEvaluationContext {
                    equity: current_eq,
                    leverage: self.config.leverage,
                    exec_price: current_candle.open,
                    closed_trades: &trades,
                    current_volatility,
                    current_drawdown: current_dd,
                    signal_strength: signal.strength,
                    dynamic_sl: signal.stop_loss_price,
                    annualization_factor: self.config.timeframe.map(|tf| tf.annualization_factor()),
                    timeframe: self.config.timeframe,
                    peak_equity,
                };

                if let Some(trade) = ExecutionEngine::process_signal_at_open_with_risk(
                    &mut portfolio,
                    &self.config,
                    &signal,
                    current_candle,
                    eval_idx,
                    Some(&risk_ctx),
                ) {
                    trades.push(trade);
                }
            }

            // Aplicación causal de Funding Rate si se cruza una frontera de 8h UTC (00:00, 08:00, 16:00)
            // o cortes acumulados en temporalidades macro > 8h (C-11).
            if let Some(ref engine) = ms_engine {
                engine.apply_funding_step(&mut portfolio, current_candle.open, current_candle.timestamp, candle_duration);
            }

            // 2. Evaluar condiciones intra-vela (Stop Loss / Take Profit / Liquidación) con los extremos High/Low
            // Las salidas intra-vela por SL, TP y liquidación son eventos de ejecución, NO de valoración de equity.
            if let Some(trade) = ExecutionEngine::check_intra_candle_exits(
                &mut portfolio,
                &self.config,
                current_candle,
                eval_idx,
            ) {
                trades.push(trade);
            }

            // 3. Registrar punto de la curva de equity al CIERRE de la vela actual
            // Directriz de Allianz: Eliminación total de equity intra-vela. Equity, drawdown y peak SOLO al cierre.
            let snapshot = CloseEquityEvaluator::evaluate_candle_close(
                &portfolio,
                current_candle.close,
                current_candle.timestamp,
                eval_idx,
                peak_equity,
                self.config.initial_capital,
            );

            peak_equity = snapshot.peak_equity;
            equity_curve.push(snapshot.point);

            // Control de bancarrota evaluado exclusivamente con la equity al cierre (C-01)
            if snapshot.is_bankrupt {
                is_bankrupt = true;
                break;
            }

            // 4. Generar señal al CIERRE de la vela actual usando estrictamente velas hasta el índice i (C-08: t <= T)
            let sig = strategy.evaluate(&klines[..=i], i);
            pending_signal = Some(sig);
        }

        // Cierre de cualquier posición remanente en la última vela evaluada (BUG-39: Cero Look-Ahead Bias)
        if let Some(pos) = portfolio.active_position.take() {
            let exit_candle = &klines[last_eval_idx];
            let reason = if is_bankrupt {
                "Bankruptcy (Account Insolvent)".to_string()
            } else {
                "End of Data Horizon".to_string()
            };
            let trade = ExecutionEngine::close_position(
                &mut portfolio,
                &self.config,
                pos,
                exit_candle.close,
                exit_candle.timestamp,
                last_eval_idx.saturating_sub(warmup_bars),
                reason,
            );
            trades.push(trade);
        }

        if is_bankrupt {
            portfolio.cash_balance = 0.0;
        }

        // C-22: Actualizar el último punto de la curva de equity con el saldo final tras el cierre forzado o quiebra,
        // garantizando que peak y drawdown sean matemáticamente coherentes con el histórico previo.
        let peak_before_last = if equity_curve.len() > 1 {
            equity_curve[..equity_curve.len() - 1]
                .iter()
                .map(|p| p.equity)
                .fold(self.config.initial_capital, f64::max)
        } else {
            self.config.initial_capital
        };
        if let Some(last_eq) = equity_curve.last_mut() {
            let _ = CloseEquityEvaluator::update_final_horizon_point(
                last_eq,
                peak_before_last,
                portfolio.cash_balance,
            );
        }

        let metrics = MetricsCalculator::calculate(
            self.config.initial_capital,
            &trades,
            &equity_curve,
        );

        Ok(BacktestResult {
            metrics,
            trades,
            equity_curve,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microstructure::MicrostructureConfig;
    use crate::types::{MarginMode, PositionSizing, SignalDirection};

    struct TestWarmupStrategy {
        signal_at_warmup_end: bool,
    }

    impl Strategy for TestWarmupStrategy {
        fn name(&self) -> &str {
            "TestWarmupStrategy"
        }

        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx == 49 {
                self.signal_at_warmup_end = true;
                Signal::new(SignalDirection::Long, 1.0, ts)
            } else {
                Signal::flat(ts)
            }
        }
    }

    fn dummy_kline(timestamp: i64, price: f64) -> Kline {
        Kline {
            timestamp,
            open: price,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume: 100.0,
        }
    }

    #[test]
    fn test_run_with_warmup_executes_pending_signal_at_test_start() {
        let klines: Vec<Kline> = (0..60)
            .map(|i| dummy_kline(i as i64 * 3600, 100.0 + i as f64))
            .collect();

        let engine = BacktestEngine::new(BacktestConfig::default());
        let mut strategy = TestWarmupStrategy { signal_at_warmup_end: false };
        let res = engine.run_with_warmup(&mut strategy, &klines, 50).unwrap();

        assert!(strategy.signal_at_warmup_end);
        assert_eq!(res.equity_curve.len(), 10);
        assert_eq!(res.equity_curve[0].candle_idx, 0);
        assert_eq!(res.equity_curve[9].candle_idx, 9);
        assert!(!res.trades.is_empty());
        assert_eq!(res.trades[0].entry_timestamp, klines[50].timestamp);
        assert_eq!(res.trades[0].entry_candle_idx, 0);
    }

    struct ReversalOpenPeakStrategy;
    impl Strategy for ReversalOpenPeakStrategy {
        fn name(&self) -> &str {
            "ReversalOpenPeakStrategy"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx == 0 {
                Signal::new(SignalDirection::Long, 1.0, ts)
            } else if current_idx == 1 {
                Signal::new(SignalDirection::Short, 1.0, ts)
            } else {
                Signal::flat(ts)
            }
        }
    }

    #[test]
    fn test_peak_equity_updates_only_at_candle_close_not_at_open_reversal() {
        let klines = vec![
            Kline { timestamp: 100, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 200, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 300, open: 200.0, high: 220.0, low: 190.0, close: 220.0, volume: 10.0 },
            Kline { timestamp: 400, open: 220.0, high: 220.0, low: 220.0, close: 220.0, volume: 10.0 },
        ];

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            commission_rate: 0.0,
            position_sizing: PositionSizing::PercentOfEquity(1.0),
            leverage: 1.0,
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);
        let mut strat = ReversalOpenPeakStrategy;
        let res = engine.run(&mut strat, &klines).unwrap();

        assert_eq!(res.equity_curve[2].equity, 18_000.0);
        assert_eq!(res.equity_curve[2].drawdown_nominal, 0.0);
    }

    struct AlwaysLongStrategy;
    impl Strategy for AlwaysLongStrategy {
        fn name(&self) -> &str {
            "AlwaysLong"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            Signal::new(SignalDirection::Long, 1.0, history[current_idx].timestamp)
        }
    }

    #[test]
    fn test_bug39_bankruptcy_exits_without_lookahead() {
        let mut klines = Vec::new();
        for i in 0..10 {
            // En i=2 cae a 0.01 provocando insolvencia / bancarrota al cierre de la vela
            let price = if i <= 1 { 100.0 } else if i == 2 { 0.01 } else { 500.0 };
            klines.push(Kline {
                timestamp: i as i64 * 3600,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: 100.0,
            });
        }

        let config = BacktestConfig {
            initial_capital: 1000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::PercentOfEquity(1.0),
            commission_rate: 0.001,
            leverage: 1.0,
            ..Default::default()
        };
        let engine = BacktestEngine::new(config);
        let mut strategy = AlwaysLongStrategy;
        let res = engine.run(&mut strategy, &klines).unwrap();

        assert!(!res.trades.is_empty());
        let last_trade = res.trades.last().unwrap();
        // Debe salir estrictamente en la vela 2 donde ocurrió la quiebra (t=2*3600), sin mirar velas futuras (t>2)
        assert_eq!(last_trade.exit_timestamp, klines[2].timestamp);
        assert_eq!(last_trade.exit_reason, "Bankruptcy (Account Insolvent)");
    }

    #[test]
    fn test_drawdown_measured_only_at_candle_close() {
        // Vela 0: señal Long
        // Vela 1: entrada al open 100.0, cae a low 80.0 (-20%), sube a high 120.0 (+20%), pero cierra en 100.0 (0%)
        // Vela 2: abre a 100.0, cae y cierra a 90.0 (-10%)
        // Vela 3: abre a 90.0, sube y cierra a 100.0 (0%)
        let klines = vec![
            Kline { timestamp: 100, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 200, open: 100.0, high: 120.0, low: 80.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 300, open: 100.0, high: 101.0, low: 89.0, close: 90.0, volume: 10.0 },
            Kline { timestamp: 400, open: 90.0, high: 101.0, low: 89.0, close: 100.0, volume: 10.0 },
        ];

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::PercentOfEquity(1.0),
            commission_rate: 0.0,
            leverage: 1.0,
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);
        let mut strategy = AlwaysLongStrategy;
        let res = engine.run(&mut strategy, &klines).unwrap();

        // En la vela 1, a pesar de la mecha a low 80.0 (-20%), el close fue 100.0 -> drawdown = 0.0
        let pt1 = &res.equity_curve[1];
        assert_eq!(pt1.equity, 10_000.0, "El equity al cierre es $10,000");
        assert_eq!(
            pt1.drawdown_nominal, 0.0,
            "El drawdown nominal al cierre debe ser 0.0 ignorando mechas intra-vela"
        );
        assert_eq!(
            pt1.drawdown_pct, 0.0,
            "El drawdown porcentual al cierre debe ser 0.0%"
        );

        // En la vela 2, el close fue 90.0 -> drawdown nominal = 1000.0 (10%)
        let pt2 = &res.equity_curve[2];
        assert_eq!(pt2.equity, 9_000.0);
        assert_eq!(pt2.drawdown_nominal, 1000.0);
        assert_eq!(pt2.drawdown_pct, 10.0);

        // En la vela 3, el close volvió a 100.0 -> drawdown nominal = 0.0
        let pt3 = &res.equity_curve[3];
        assert_eq!(pt3.equity, 10_000.0);
        assert_eq!(pt3.drawdown_nominal, 0.0);

        // En las métricas consolidadas, el max drawdown debe ser del 10.0%, medido exclusivamente al close
        assert!(
            (res.metrics.max_drawdown_pct - 10.0).abs() < 1e-6,
            "Las métricas deben reportar el Max Drawdown al Close (10.0%), obtenido: {}",
            res.metrics.max_drawdown_pct
        );
    }

    struct TakeProfitStrategy {
        tp_price: f64,
    }

    impl Strategy for TakeProfitStrategy {
        fn name(&self) -> &str {
            "TakeProfitStrategy"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx == 0 {
                let mut sig = Signal::new(SignalDirection::Long, 1.0, ts);
                sig.take_profit_price = Some(self.tp_price);
                sig
            } else {
                Signal::flat(ts)
            }
        }
    }

    #[test]
    fn test_tp_with_high_wick_does_not_generate_drawdown() {
        // Vela 0: señal Long con TP = 105.0
        // Vela 1: entrada al Open 100.0. High vuela a 150.0. Low 99.0. Close 105.0.
        // El TP se ejecuta a 105.0 intra-vela.
        // Vela 2: vela plana final 105.0.
        let klines = vec![
            Kline { timestamp: 100, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 200, open: 100.0, high: 150.0, low: 99.0, close: 105.0, volume: 10.0 },
            Kline { timestamp: 300, open: 105.0, high: 105.0, low: 105.0, close: 105.0, volume: 10.0 },
        ];

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::PercentOfEquity(1.0),
            commission_rate: 0.0, // Cero comisiones para aislar drawdown
            leverage: 1.0,
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);
        let mut strategy = TakeProfitStrategy { tp_price: 105.0 };
        let res = engine.run(&mut strategy, &klines).unwrap();

        // Verificamos que el trade cerró por TP
        assert_eq!(res.trades.len(), 1);
        assert_eq!(res.trades[0].exit_reason, "Take Profit");
        assert_eq!(res.trades[0].exit_price, 105.0);
        assert!(res.trades[0].net_pnl > 0.0);

        // Al cierre de la vela 1 y en todas las velas, el equity nunca tuvo drawdown respecto al peak
        for pt in &res.equity_curve {
            assert_eq!(
                pt.drawdown_nominal, 0.0,
                "No debe haber drawdown nominal fantasma por el High de 150.0, obtenido: {}",
                pt.drawdown_nominal
            );
            assert_eq!(
                pt.drawdown_pct, 0.0,
                "No debe haber drawdown porcentual fantasma por el High de 150.0, obtenido: {}",
                pt.drawdown_pct
            );
        }

        assert_eq!(
            res.metrics.max_drawdown_pct, 0.0,
            "El Max Drawdown total debe ser 0.0% en una operación puramente ganadora"
        );
    }

    struct StopLossStrategy {
        sl_price: f64,
    }

    impl Strategy for StopLossStrategy {
        fn name(&self) -> &str {
            "StopLossStrategy"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx == 0 {
                let mut sig = Signal::new(SignalDirection::Long, 1.0, ts);
                sig.stop_loss_price = Some(self.sl_price);
                sig
            } else {
                Signal::flat(ts)
            }
        }
    }

    #[test]
    fn test_extreme_low_wick_with_sl_does_not_cause_bankruptcy() {
        // Capital 10,000. Entrada Long a 100.0 con SL a 95.0.
        // Vela 1 tiene una mecha bajista extrema con Low = 0.01 (caída del 99.99%), pero Close en 96.0.
        // El SL se ejecuta a 95.0 preservando ~9,500 de capital.
        // El motor NO debe declarar bancarrota por la mecha extrema (C-01 resuelto).
        let klines = vec![
            Kline { timestamp: 100, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 200, open: 100.0, high: 101.0, low: 0.01, close: 96.0, volume: 10.0 },
            Kline { timestamp: 300, open: 96.0, high: 96.0, low: 96.0, close: 96.0, volume: 10.0 },
        ];

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::PercentOfEquity(1.0),
            commission_rate: 0.0,
            leverage: 1.0,
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);
        let mut strategy = StopLossStrategy { sl_price: 95.0 };
        let res = engine.run(&mut strategy, &klines).unwrap();

        // El trade debió salir por Stop Loss a 95.0
        assert_eq!(res.trades.len(), 1);
        assert_eq!(res.trades[0].exit_reason, "Stop Loss");
        assert_eq!(res.trades[0].exit_price, 95.0);

        // El capital final debe ser de 9,500, NO cero ni bancarrota
        assert!(
            (res.metrics.final_capital - 9500.0).abs() < 1e-6,
            "El capital final debe ser 9500 tras ejecutar SL, obtenido: {}",
            res.metrics.final_capital
        );

        // Se deben haber evaluado todas las velas sin abortar por insolvencia
        assert_eq!(res.equity_curve.len(), 3);
        let last_pt = res.equity_curve.last().unwrap();
        assert_eq!(last_pt.equity, 9500.0);
    }

    #[test]
    fn test_zero_slippage_default_policy() {
        let klines = vec![
            Kline { timestamp: 100, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 200, open: 105.0, high: 110.0, low: 105.0, close: 108.0, volume: 10.0 },
            Kline { timestamp: 300, open: 108.0, high: 108.0, low: 108.0, close: 108.0, volume: 10.0 },
        ];
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            commission_rate: 0.0,
            ..Default::default()
        };
        let engine = BacktestEngine::new(config);
        let mut strategy = AlwaysLongStrategy;
        let res = engine.run(&mut strategy, &klines).unwrap();
        assert!(!res.trades.is_empty());
        // El precio de entrada DEBE ser exactamente el Open de la vela 1 (105.0), sin ningún slippage
        assert_eq!(
            res.trades[0].entry_price, 105.0,
            "La política institucional de Allianz exige terminantemente ejecución exacta a Open sin slippage"
        );
    }

    #[test]
    fn test_funding_rate_accrual_across_8h_boundary_in_backtest() {
        // Base: 2026-01-01 00:00:00 UTC en ms = 1767225600000
        let t0 = 1767225600000_i64;
        let eight_hours = 8 * 3600 * 1000;

        // Velas espaciadas cada 8 horas exactas
        let klines = vec![
            Kline { timestamp: t0, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 }, // Vela 0: emite señal Long
            Kline { timestamp: t0 + eight_hours, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 }, // Vela 1: entrada al Open (08:00 UTC)
            Kline { timestamp: t0 + 2 * eight_hours, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 }, // Vela 2: corte de las 16:00 UTC -> devengo de funding
            Kline { timestamp: t0 + 3 * eight_hours, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 10.0 }, // Vela 3: corte de las 00:00 UTC -> devengo de funding
        ];

        let mut config = BacktestConfig {
            initial_capital: 10_000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::FixedAmount(1000.0), // Nocional $1,000 (10 unidades a $100)
            commission_rate: 0.0, // Sin comisiones para aislar el funding
            leverage: 1.0,
            ..Default::default()
        };

        let mut ms = crate::microstructure::MicrostructureConfig::default();
        ms.fee_schedule.custom_maker_fee = Some(0.0);
        ms.fee_schedule.custom_taker_fee = Some(0.0);
        ms.funding_config.base_funding_rate = 0.001; // 0.1% cada 8h -> $1.00 por corte para $1,000 nocional
        config.microstructure = Some(ms);

        let engine = BacktestEngine::new(config);
        let mut strategy = AlwaysLongStrategy;
        let res = engine.run(&mut strategy, &klines).unwrap();

        assert!(!res.trades.is_empty(), "Debe haberse cerrado el trade al fin del horizonte");
        let trade = &res.trades[0];

        // En la vela 1 entró a las 08:00
        // En la vela 2 (16:00): cruzó corte 16:00 -> pagó $1.00 de funding
        // En la vela 3 (00:00): cruzó corte 00:00 -> pagó $1.00 de funding
        // Total funding acumulado: $2.00
        // Como precio no varió (gross_pnl = 0.0) y comisiones = 0.0, net_pnl = -$2.00
        assert!(
            (trade.net_pnl - (-2.0)).abs() < 1e-4,
            "El trade debe reflejar $2.00 de costos netos por funding rates en los 2 cortes de 8h, obtenido: {}",
            trade.net_pnl
        );
        // Verificación C-18: desglose de funding registrado en CompletedTrade y en BacktestMetrics
        assert!(
            (trade.funding_paid - 2.0).abs() < 1e-4,
            "CompletedTrade debe registrar funding_paid = 2.0, obtenido: {}",
            trade.funding_paid
        );
        assert!(
            (res.metrics.total_funding - 2.0).abs() < 1e-4,
            "BacktestMetrics debe registrar total_funding = 2.0, obtenido: {}",
            res.metrics.total_funding
        );
    }

    struct StatefulCountingStrategy {
        evaluated_indices: Vec<usize>,
    }

    impl Strategy for StatefulCountingStrategy {
        fn name(&self) -> &str {
            "StatefulCountingStrategy"
        }

        fn evaluate(&mut self, _history: &[Kline], current_idx: usize) -> Signal {
            self.evaluated_indices.push(current_idx);
            Signal::flat(0)
        }
    }

    #[test]
    fn test_c33_warmup_bars_evaluates_strategy_streaming() {
        let klines: Vec<Kline> = (0..10)
            .map(|i| Kline {
                timestamp: 1000 + (i as i64 * 3600),
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 100.0,
            })
            .collect();

        let engine = BacktestEngine::new(BacktestConfig::default());
        let mut strategy = StatefulCountingStrategy {
            evaluated_indices: Vec::new(),
        };

        let warmup_bars = 4;
        let res = engine.run_with_warmup(&mut strategy, &klines, warmup_bars);
        assert!(res.is_ok());

        // C-33: Todas las barras de warmup (0, 1, 2, 3) y las de prueba (4, 5, 6, 7, 8, 9)
        // deben haber sido evaluadas para que la memoria causal esté íntegra.
        assert_eq!(
            strategy.evaluated_indices,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Estrategia debe ser evaluada en cada una de las barras de warmup y de test"
        );
    }

    struct C08CausalSliceStrategy {
        evaluated_count: usize,
    }

    impl Strategy for C08CausalSliceStrategy {
        fn name(&self) -> &str {
            "C08CausalSliceStrategy"
        }

        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            assert_eq!(
                history.len(),
                current_idx + 1,
                "C-08 Violación: history.len() ({}) debe ser exactamente current_idx + 1 ({})",
                history.len(),
                current_idx + 1
            );
            self.evaluated_count += 1;
            Signal::flat(history[current_idx].timestamp)
        }
    }

    #[test]
    fn test_c08_strategy_evaluate_receives_strictly_causal_slice() {
        let klines: Vec<Kline> = (0..20)
            .map(|i| Kline {
                timestamp: 1000 + (i as i64 * 3600),
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 100.0,
            })
            .collect();

        let engine = BacktestEngine::new(BacktestConfig::default());
        let mut strategy = C08CausalSliceStrategy { evaluated_count: 0 };
        let res = engine.run_with_warmup(&mut strategy, &klines, 5);
        assert!(res.is_ok());
        assert_eq!(strategy.evaluated_count, 20);
    }

    struct OneShotLongStrategy {
        entry_idx: usize,
    }

    impl Strategy for OneShotLongStrategy {
        fn name(&self) -> &str {
            "OneShotLongStrategy"
        }

        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx >= self.entry_idx {
                Signal::new(SignalDirection::Long, 1.0, ts)
            } else {
                Signal::flat(ts)
            }
        }
    }

    #[test]
    fn test_c11_macro_1d_candle_funding_accrual_three_cuts_at_open() {
        // Simulación con velas diarias (1D) de 24h cada una
        // 2026-01-01 00:00:00 UTC en ms = 1767225600000
        let t0 = 1767225600000_i64;
        let day_ms = 24 * 3600 * 1000;

        let klines = vec![
            Kline { timestamp: t0, open: 100.0, high: 102.0, low: 98.0, close: 100.0, volume: 1000.0 },
            Kline { timestamp: t0 + day_ms, open: 100.0, high: 102.0, low: 98.0, close: 100.0, volume: 1000.0 },
            Kline { timestamp: t0 + 2 * day_ms, open: 100.0, high: 100.0, low: 100.0, close: 100.0, volume: 1000.0 },
        ];

        let mut ms = MicrostructureConfig::default();
        ms.funding_config.enabled = true;
        ms.funding_config.base_funding_rate = 0.001; // 0.1% cada 8h -> $1.00 por corte para $1,000 nocional

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::PercentOfEquity(0.10), // $1,000 nocional (10 unidades a $100)
            commission_rate: 0.0, // Cero comisiones para aislar funding
            leverage: 1.0,
            timeframe: Some(crate::types::Timeframe::D1),
            microstructure: Some(ms),
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);
        let mut strategy = OneShotLongStrategy { entry_idx: 0 };
        let res = engine.run(&mut strategy, &klines).unwrap();

        assert_eq!(res.trades.len(), 1);
        let trade = &res.trades[0];
        // En Día 2: 3 cortes tasados al Open (100.0) -> $3.00
        // En Día 3: 3 cortes tasados al Open (100.0) -> $3.00
        // Total esperado: 6 cortes de 8h = $6.00 de funding acumulado
        assert!(
            (trade.funding_paid - 6.0).abs() < 1e-4,
            "Trade en velas 1D debe reflejar $6.00 de costes por 6 cortes de funding, obtenido: {}",
            trade.funding_paid
        );
        assert!(
            (res.metrics.total_funding - 6.0).abs() < 1e-4,
            "Métricas de 1D deben registrar total_funding = 6.0, obtenido: {}",
            res.metrics.total_funding
        );
    }
}

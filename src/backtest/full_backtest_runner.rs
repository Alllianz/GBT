use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult, EquityPoint};
use crate::backtest::metrics_calculator::MetricsCalculator;
use crate::backtest::strategy::Strategy;
use crate::features::SplitDataset;
use crate::types::Kline;

/// Resultado estructurado de un Backtest Completo Purged (Causal e Institucional)
#[derive(Debug, Clone)]
pub struct FullBacktestResult {
    pub is_result: BacktestResult,
    pub oos_result: BacktestResult,
    pub combined_result: BacktestResult,
    pub purge_bars: usize,
    pub train_end_idx: usize,
    pub test_start_idx: usize,
    pub total_bars: usize,
}

pub struct FullBacktestRunner;

impl FullBacktestRunner {
    /// Ejecuta un backtest de horizonte completo con estricta partición y purga/embargo:
    /// 1. Backtest In-Sample (IS): velas 0..train_end. Al finalizar IS, cualquier posición
    ///    abierta es liquidada/cerrada causalmente ("End of In-Sample Horizon").
    /// 2. Ventana de Purga/Embargo: velas train_end..test_start. Se respeta una cuarentena absoluta
    ///    donde no se toma ninguna operación para evitar cualquier fuga por horizonte de etiquetas (t <= T).
    ///    Durante este lapso, el balance y la curva de equity permanecen inmutables.
    /// 3. Backtest Out-of-Sample (OOS): velas test_start..N. Se ejecuta con período de calentamiento
    ///    y reinicio/limpieza de memoria interna (`clear_cache()`), iniciando con el capital final de IS.
    /// 4. Consolida la curva de equity completa y la totalidad de operaciones ejecutadas para
    ///    generar un reporte completo honesto, unificado y libre de Look-Ahead Bias (BUG-07).
    pub fn run_purged_full_backtest(
        strategy: &mut dyn Strategy,
        split: &SplitDataset,
        full_klines: &[Kline],
        config: &BacktestConfig,
    ) -> Result<FullBacktestResult, String> {
        if full_klines.is_empty() {
            return Err("El conjunto de velas históricas está vacío".to_string());
        }

        // 1. Fase In-Sample
        let engine_is = BacktestEngine::new(config.clone());
        let is_result = engine_is.run(strategy, &split.train_klines)?;

        let is_final_capital = is_result.metrics.final_capital;

        // BUG-F02: Eliminación de resurrección de capital zombi
        // Si la estrategia quebró en In-Sample (Equity <= 0), abortar OOS inmediatamente.
        // No inyectar capital nuevo: consolidar la pérdida total (-100%) y reflejar la ruina en el reporte.
        if is_final_capital <= 0.0 {
            let oos_result = BacktestResult {
                metrics: MetricsCalculator::calculate(0.0, &[], &[]),
                trades: Vec::new(),
                equity_curve: Vec::new(),
            };

            let mut combined_equity = Vec::with_capacity(full_klines.len());
            let mut peak_equity = config.initial_capital;

            for pt in &is_result.equity_curve {
                if pt.equity > peak_equity {
                    peak_equity = pt.equity;
                }
                let dd_nom = (peak_equity - pt.equity).max(0.0);
                let dd_pct = if peak_equity > 0.0 { (dd_nom / peak_equity) * 100.0 } else { 0.0 };
                combined_equity.push(EquityPoint::new(
                    pt.timestamp,
                    pt.candle_idx,
                    pt.balance,
                    pt.equity,
                    dd_nom,
                    dd_pct,
                ));
            }

            for (idx, candle) in full_klines.iter().enumerate().skip(combined_equity.len()) {
                let dd_nom = peak_equity;
                let dd_pct = 100.0;
                combined_equity.push(EquityPoint::new(
                    candle.timestamp,
                    idx,
                    0.0,
                    0.0,
                    dd_nom,
                    dd_pct,
                ));
            }

            let combined_trades = is_result.trades.clone();
            let combined_metrics = MetricsCalculator::calculate(
                config.initial_capital,
                &combined_trades,
                &combined_equity,
            );

            let combined_result = BacktestResult {
                metrics: combined_metrics,
                trades: combined_trades,
                equity_curve: combined_equity,
            };

            return Ok(FullBacktestResult {
                is_result,
                oos_result,
                combined_result,
                purge_bars: split.purge_bars,
                train_end_idx: split.train_end,
                test_start_idx: split.test_start,
                total_bars: full_klines.len(),
            });
        }

        // Si sobrevive a IS, OOS inicia estrictamente con el capital final de IS (sin reinicio a initial_capital)
        let oos_initial_capital = is_final_capital;

        // 2. Fase Out-of-Sample con aislamiento de estado
        let mut oos_config = config.clone();
        oos_config.initial_capital = oos_initial_capital;
        let engine_oos = BacktestEngine::new(oos_config);
        strategy.clear_cache();
        let oos_result = engine_oos.run_with_warmup(strategy, &split.test_warmup_klines, split.warmup_bars)?;

        // 3. Consolidación de Operaciones sin solapamiento
        // C-09: Reindexar monótonamente entry_candle_idx y exit_candle_idx de OOS sumando el offset global
        let mut combined_trades = is_result.trades.clone();
        let oos_offset = split.test_start;
        for mut trade in oos_result.trades.clone() {
            trade.entry_candle_idx += oos_offset;
            trade.exit_candle_idx += oos_offset;
            combined_trades.push(trade);
        }

        // 4. Consolidación de Curva de Equity Causal (IS + Purga/Embargo + OOS)
        let mut combined_equity = Vec::with_capacity(full_klines.len());
        let mut peak_equity = config.initial_capital;

        // Puntos In-Sample (0..train_end)
        for pt in &is_result.equity_curve {
            if pt.equity > peak_equity {
                peak_equity = pt.equity;
            }
            let dd_nom = (peak_equity - pt.equity).max(0.0);
            let dd_pct = if peak_equity > 0.0 { (dd_nom / peak_equity) * 100.0 } else { 0.0 };
            combined_equity.push(EquityPoint::new(
                pt.timestamp,
                pt.candle_idx,
                pt.balance,
                pt.equity,
                dd_nom,
                dd_pct,
            ));
        }

        // Puntos de Cuarentena / Purga (train_end..test_start)
        for (idx, candle) in full_klines
            .iter()
            .enumerate()
            .take(split.test_start.min(full_klines.len()))
            .skip(split.train_end)
        {
            let dd_nom = (peak_equity - is_final_capital).max(0.0);
            let dd_pct = if peak_equity > 0.0 { (dd_nom / peak_equity) * 100.0 } else { 0.0 };
            combined_equity.push(EquityPoint::new(
                candle.timestamp,
                idx,
                is_final_capital,
                is_final_capital,
                dd_nom,
                dd_pct,
            ));
        }

        // Puntos Out-of-Sample (test_start..N)
        // C-23: Medición de drawdown combinado exclusivamente con equity al Close (cero componentes intra-vela)
        for (i, pt) in oos_result.equity_curve.iter().enumerate() {
            let global_idx = split.test_start + i;
            if global_idx >= full_klines.len() {
                break;
            }
            if pt.equity > peak_equity {
                peak_equity = pt.equity;
            }
            let dd_nom = (peak_equity - pt.equity).max(0.0);
            let dd_pct = if peak_equity > 0.0 { (dd_nom / peak_equity) * 100.0 } else { 0.0 };
            combined_equity.push(EquityPoint::new(
                pt.timestamp,
                global_idx,
                pt.balance,
                pt.equity,
                dd_nom,
                dd_pct,
            ));
        }

        // 5. Cálculo de Métricas Consolidadas
        let combined_metrics = MetricsCalculator::calculate(
            config.initial_capital,
            &combined_trades,
            &combined_equity,
        );

        let combined_result = BacktestResult {
            metrics: combined_metrics,
            trades: combined_trades,
            equity_curve: combined_equity,
        };

        Ok(FullBacktestResult {
            is_result,
            oos_result,
            combined_result,
            purge_bars: split.purge_bars,
            train_end_idx: split.train_end,
            test_start_idx: split.test_start,
            total_bars: full_klines.len(),
        })
    }

    /// Formatea un resumen institucional comparativo del backtest completo
    pub fn format_audit_summary(res: &FullBacktestResult) -> String {
        let is = &res.is_result.metrics;
        let oos = &res.oos_result.metrics;
        let full = &res.combined_result.metrics;

        format!(
            "\n===============================================================\n\
             📊 REPORTE DE BACKTEST COMPLETO CON AISLAMIENTO CAUSAL (IS/OOS/PURGA)\n\
             ===============================================================\n\
             Velas Totales: {} | In-Sample (IS): 0..{} | Purga: {} barras ({}..{}) | OOS: {}..{}\n\
             ---------------------------------------------------------------\n\
             MÉTRICA                    | IN-SAMPLE (70%) | OUT-OF-SAMPLE (30%) | HORIZONTE TOTAL\n\
             ---------------------------+-----------------+---------------------+----------------\n\
             Retorno Neto %             | {:>+14.2}% | {:>+18.2}% | {:>+13.2}%\n\
             Capital Final              | ${:>14.2} | ${:>18.2} | ${:>13.2}\n\
             Max Drawdown %             | {:>14.2}% | {:>18.2}% | {:>13.2}%\n\
             Sharpe Ratio               | {:>15.2} | {:>19.2} | {:>14.2}\n\
             Profit Factor              | {:>15.2} | {:>19.2} | {:>14.2}\n\
             Win Rate %                 | {:>14.1}% | {:>18.1}% | {:>13.1}%\n\
             Total Trades               | {:>15} | {:>19} | {:>14}\n\
             ===============================================================",
            res.total_bars,
            res.train_end_idx,
            res.purge_bars,
            res.train_end_idx,
            res.test_start_idx,
            res.test_start_idx,
            res.total_bars,
            is.net_profit_pct, oos.net_profit_pct, full.net_profit_pct,
            is.final_capital, oos.final_capital, full.final_capital,
            is.max_drawdown_pct, oos.max_drawdown_pct, full.max_drawdown_pct,
            is.sharpe_ratio, oos.sharpe_ratio, full.sharpe_ratio,
            is.profit_factor, oos.profit_factor, full.profit_factor,
            is.win_rate_pct, oos.win_rate_pct, full.win_rate_pct,
            is.total_trades, oos.total_trades, full.total_trades,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Signal, SignalDirection};

    struct DummyStepStrategy;
    impl Strategy for DummyStepStrategy {
        fn name(&self) -> &str {
            "DummyStep"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            Signal::flat(history[current_idx].timestamp)
        }
    }

    #[test]
    fn test_bug07_purged_full_backtest_continuity_and_quarantine() {
        let klines: Vec<Kline> = (0..120)
            .map(|i| Kline {
                timestamp: i as i64 * 3600,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 50.0,
            })
            .collect();

        let purge_bars = 5;
        let split = crate::features::TrainTestSplit::split_chronological(&klines, 0.70, purge_bars).unwrap();
        let config = BacktestConfig::default();

        let mut strat = DummyStepStrategy;

        let full_res = FullBacktestRunner::run_purged_full_backtest(
            &mut strat,
            &split,
            &klines,
            &config,
        ).unwrap();

        assert_eq!(full_res.purge_bars, 5);
        assert_eq!(full_res.train_end_idx, split.train_end);
        assert_eq!(full_res.test_start_idx, split.test_start);
        assert_eq!(full_res.combined_result.equity_curve.len(), 120);

        // Verificar que durante la purga (indices train_end..test_start), el balance es idéntico al final de IS
        for idx in split.train_end..split.test_start {
            assert_eq!(full_res.combined_result.equity_curve[idx].balance, full_res.is_result.metrics.final_capital);
        }
    }

    struct BankruptStrategy;
    impl Strategy for BankruptStrategy {
        fn name(&self) -> &str {
            "BankruptStrategy"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            crate::types::Signal::new(
                crate::types::SignalDirection::Long,
                1.0,
                history[current_idx].timestamp,
            )
        }
    }

    #[test]
    fn test_bug_f02_no_zombie_capital_after_is_bankruptcy() {
        // Generar serie donde en IS el precio colapsa a 0.01 causando quiebra
        let mut klines = Vec::new();
        for i in 0..100 {
            let price = if i < 10 { 100.0 } else if i == 10 { 0.01 } else { 100.0 };
            klines.push(Kline {
                timestamp: i as i64 * 3600,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: 50.0,
            });
        }

        let split = crate::features::TrainTestSplit::split_chronological(&klines, 0.70, 5).unwrap();
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            position_sizing: crate::types::PositionSizing::PercentOfEquity(1.0),
            leverage: 1.0,
            ..Default::default()
        };

        let mut strat = BankruptStrategy;
        let full_res = FullBacktestRunner::run_purged_full_backtest(
            &mut strat,
            &split,
            &klines,
            &config,
        ).unwrap();

        // IS quebró: el capital final debe ser <= 0.0
        assert!(full_res.is_result.metrics.final_capital <= 0.0);

        // OOS NO debe haber sido ejecutado con $10,000 zombi
        assert_eq!(full_res.oos_result.metrics.final_capital, 0.0);
        assert_eq!(full_res.oos_result.trades.len(), 0);

        // El resultado consolidado total debe reflejar quiebra completa (-100% PnL y >= 100% MaxDD por liquidación)
        assert_eq!(full_res.combined_result.metrics.final_capital, 0.0);
        assert_eq!(full_res.combined_result.metrics.net_profit_pct, -100.0);
        assert!(full_res.combined_result.metrics.max_drawdown_pct >= 100.0);
    }

    #[test]
    fn test_c23_combined_equity_and_drawdown_strictly_at_close() {
        // Generar 100 velas con mechas extremas en IS y OOS pero closes controlados
        let mut klines = Vec::new();
        for i in 0..100 {
            // Vela 85 (en OOS) cierra a 90.0 (-10%), todas las demás cierran a 100.0
            let close = if i == 85 { 90.0 } else { 100.0 };
            klines.push(Kline {
                timestamp: i as i64 * 3600,
                open: 100.0,
                high: 150.0, // Mecha alcista extrema que NO debe inflar peak
                low: 50.0,   // Mecha bajista extrema que NO debe generar drawdown
                close,
                volume: 50.0,
            });
        }

        let split = crate::features::TrainTestSplit::split_chronological(&klines, 0.70, 5).unwrap();
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            commission_rate: 0.0,
            position_sizing: crate::types::PositionSizing::PercentOfEquity(1.0),
            leverage: 1.0,
            ..Default::default()
        };

        let mut strat = DummyStepStrategy; // Estrategia plana: balance = equity = 10,000 en todo momento
        let full_res = FullBacktestRunner::run_purged_full_backtest(
            &mut strat,
            &split,
            &klines,
            &config,
        ).unwrap();

        // En todas las velas del backtest combinado, el drawdown nominal debe ser 0.0
        // porque la estrategia estuvo Flat y el balance se preservó intacto en $10,000
        for (idx, pt) in full_res.combined_result.equity_curve.iter().enumerate() {
            assert_eq!(
                pt.equity, 10_000.0,
                "Equity en vela {} debe ser exactamente $10,000",
                idx
            );
            assert_eq!(
                pt.drawdown_nominal, 0.0,
                "C-23: Drawdown nominal en vela {} debe ser 0.0 a pesar de mechas de low 50.0",
                idx
            );
            assert_eq!(
                pt.drawdown_pct, 0.0,
                "C-23: Drawdown pct en vela {} debe ser 0.0%",
                idx
            );
        }

        assert_eq!(full_res.combined_result.metrics.max_drawdown_pct, 0.0);
    }

    struct MultiTradeStepStrategy;
    impl Strategy for MultiTradeStepStrategy {
        fn name(&self) -> &str {
            "MultiTradeStep"
        }
        fn evaluate(&mut self, history: &[Kline], current_idx: usize) -> Signal {
            let ts = history[current_idx].timestamp;
            if current_idx % 20 == 2 {
                Signal::new(SignalDirection::Long, 1.0, ts)
            } else if current_idx % 20 == 10 {
                Signal::new(SignalDirection::Flat, 1.0, ts)
            } else {
                Signal::flat(ts)
            }
        }
    }

    #[test]
    fn test_c09_monotonic_entry_candle_idx_in_combined_trades() {
        let klines: Vec<Kline> = (0..150)
            .map(|i| Kline {
                timestamp: (i as i64 + 1) * 3600,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 50.0,
            })
            .collect();

        let split = crate::features::TrainTestSplit::split_chronological_with_warmup(
            &klines,
            0.60,
            5,
            10,
        ).unwrap();

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            commission_rate: 0.0005,
            position_sizing: crate::types::PositionSizing::PercentOfEquity(0.20),
            leverage: 1.0,
            ..Default::default()
        };

        let mut strat = MultiTradeStepStrategy;
        let full_res = FullBacktestRunner::run_purged_full_backtest(
            &mut strat,
            &split,
            &klines,
            &config,
        ).unwrap();

        let combined_trades = &full_res.combined_result.trades;
        assert!(
            combined_trades.len() >= 2,
            "Debe haber operaciones en IS y OOS, obtenido: {}",
            combined_trades.len()
        );

        let mut prev_entry_idx = 0;
        for (trade_idx, trade) in combined_trades.iter().enumerate() {
            if trade_idx > 0 {
                assert!(
                    trade.entry_candle_idx > prev_entry_idx,
                    "C-09 Violación: entry_candle_idx debe ser estrictamente creciente: prev={}, curr={}",
                    prev_entry_idx,
                    trade.entry_candle_idx
                );
            }
            prev_entry_idx = trade.entry_candle_idx;

            let eq_pt = &full_res.combined_result.equity_curve[trade.entry_candle_idx];
            assert_eq!(
                eq_pt.timestamp, trade.entry_timestamp,
                "El timestamp de la vela global {} en equity_curve debe coincidir con entry_timestamp del trade {}",
                trade.entry_candle_idx, trade_idx
            );
        }

        let is_trade_count = full_res.is_result.trades.len();
        assert_eq!(
            full_res.combined_result.trades[is_trade_count].entry_candle_idx,
            full_res.oos_result.trades[0].entry_candle_idx + split.test_start
        );
        for oos_trade in &combined_trades[is_trade_count..] {
            assert!(
                oos_trade.entry_candle_idx >= split.test_start,
                "Trade OOS debe tener entry_candle_idx >= test_start ({}), obtenido: {}",
                split.test_start,
                oos_trade.entry_candle_idx
            );
        }
    }
}

//! Módulo de compatibilidad para el motor de ejecución.
//! La implementación y sus submódulos atómicos residen en `crate::backtest::execution`.

pub use super::execution::ExecutionEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::config::BacktestConfig;
    use crate::backtest::portfolio::Portfolio;
    use crate::microstructure::{MicrostructureConfig, MicrostructureEngine};
    use crate::types::{Kline, MarginMode, PositionSizing, Signal, SignalDirection};

    fn dummy_candle(open: f64) -> Kline {
        Kline {
            timestamp: 1000,
            open,
            high: open + 2.0,
            low: open - 2.0,
            close: open,
            volume: 100.0,
        }
    }

    #[test]
    fn test_bug32_reversal_updates_extremes_with_open_gap() {
        let mut portfolio = Portfolio::new(10000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            ..Default::default()
        };

        // Entrar en Long a 100.0
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // La vela 2 abre con Gap alcista a 150.0 y señal Flat (cierre)
        let exit_sig = Signal::flat(2000);
        let c2 = dummy_candle(150.0);
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &exit_sig, &c2, 1)
            .expect("Debe cerrar la posición");

        // El MFE debe reflejar el Gap de 150.0 (50% de ganancia de excursión favorable máxima)
        assert!((trade.max_favorable_excursion - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_bug33_stop_loss_and_take_profit_bounding_long() {
        let mut portfolio = Portfolio::new(10000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            fixed_stop_loss_pct: Some(0.05), // 5% fallback
            fixed_take_profit_pct: Some(0.10),
            ..Default::default()
        };

        // 1. Long con SL inválido (mayor que open/exec_price) y TP inválido (menor que exec_price)
        let mut sig_invalid = Signal::new(SignalDirection::Long, 1.0, 1000);
        sig_invalid.stop_loss_price = Some(110.0); // Inválido para Long (open = 100.0)
        sig_invalid.take_profit_price = Some(90.0); // Inválido para Long

        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &sig_invalid, &c1, 0);

        let pos = portfolio.active_position.as_ref().unwrap();
        // Debe haber descartado el SL dinámico y usado el fallback fijo de 5% (100 * 0.95 = 95.0)
        assert!((pos.stop_loss.unwrap() - 95.0).abs() < 1e-6);
        // Debe haber descartado el TP dinámico y usado el fallback fijo de 10% (100 * 1.10 = 110.0)
        assert!((pos.take_profit.unwrap() - 110.0).abs() < 1e-6);
    }

    #[test]
    fn test_bug33_stop_loss_and_take_profit_bounding_short() {
        let mut portfolio = Portfolio::new(10000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            allow_short: true,
            fixed_stop_loss_pct: Some(0.05), // 5% fallback
            fixed_take_profit_pct: Some(0.10),
            ..Default::default()
        };

        // Short con SL inválido (menor que exec_price) y TP inválido (mayor que exec_price)
        let mut sig_invalid = Signal::new(SignalDirection::Short, 1.0, 1000);
        sig_invalid.stop_loss_price = Some(90.0); // Inválido para Short (open = 100.0)
        sig_invalid.take_profit_price = Some(110.0); // Inválido para Short

        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &sig_invalid, &c1, 0);

        let pos = portfolio.active_position.as_ref().unwrap();
        // Fallback fijo para Short: 100 * 1.05 = 105.0
        assert!((pos.stop_loss.unwrap() - 105.0).abs() < 1e-6);
        // Fallback fijo para Short: 100 * 0.90 = 90.0
        assert!((pos.take_profit.unwrap() - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_bug38_exact_entry_exit_commission_accounting() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            commission_rate: 0.01,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::FixedAmount(100.0),
            leverage: 1.0,
            ..Default::default()
        };

        // Entrar en Long a 100.0 con 1.0 unidad (notional = 100.0)
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        assert!((portfolio.cash_balance - 999.0).abs() < 1e-6);
        assert!((portfolio.total_commissions_paid - 1.0).abs() < 1e-6);

        // Salir a 110.0 (gross_pnl = +10.0, exit_commission = 1.10, total_commissions = 2.10)
        let exit_sig = Signal::flat(2000);
        let c2 = dummy_candle(110.0);
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &exit_sig, &c2, 1)
            .expect("Debe cerrar");

        assert!((trade.commissions - 2.10).abs() < 1e-6);
        assert!((trade.net_pnl - 7.90).abs() < 1e-6);
        assert!((portfolio.cash_balance - 1007.90).abs() < 1e-6);
        assert!((portfolio.total_commissions_paid - 2.10).abs() < 1e-6);
    }

    #[test]
    fn test_bug04_cross_margin_zero_liquidation_does_not_trigger_spurious_exit() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            margin_mode: MarginMode::Cross,
            position_sizing: PositionSizing::FixedAmount(500.0), // notional 500 con equity 1000 (leverage efectivo 0.5x)
            leverage: 1.0,
            ..Default::default()
        };

        // Long a 100.0, equity $1000, quantity 5.0. Total notional $500.
        // equity / quantity = 200.0 > exec_price (100.0). liq_price DEBE ser 0.0 (imposible de liquidar).
        let sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &sig, &c1, 0);

        let pos = portfolio.active_position.as_ref().unwrap();
        assert_eq!(pos.liquidation_price, 0.0);

        // Vela con caída anómala o precio cero: low = 0.0 no debe liquidar
        let mut c_crash = dummy_candle(50.0);
        c_crash.low = 0.0;
        let exit = ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c_crash, 1);
        assert!(exit.is_none(), "Una posición con liq_price 0.0 en Cross Margin jamás debe liquidarse espuriamente");
        assert!(portfolio.active_position.is_some(), "La posición debe permanecer activa");
    }

    #[test]
    fn test_c03_isolated_numerical_allianz_spec() {
        // Requisito textual de Allianz para C-03:
        // "cuenta de 1000, margen aislado de 100, comisión de entrada de 0.5 y liquidación;
        //  el saldo final debe ser exactamente 900, y net_pnl debe ser exactamente −100."
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            margin_mode: MarginMode::Isolated,
            commission_rate: 0.0005, // 0.05% taker -> en 1000 notional = 0.5 comisión
            position_sizing: PositionSizing::FixedAmount(100.0), // $100 margen
            leverage: 10.0, // 1000 notional (10.0 unidades a $100)
            ..Default::default()
        };

        // 1. Abrir Long a 100.0 en Isolated Margin
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // En Isolated, la comisión de entrada (0.50) sale del margen asignado (100.0),
        // por lo que NO se deduce del cash_balance libre (permanece en 1000.0)
        assert_eq!(portfolio.cash_balance, 1000.0, "cash_balance no debe sufrir doble deducción al abrir");
        let pos = portfolio.active_position.as_ref().unwrap();
        assert_eq!(pos.isolated_margin, Some(100.0));
        assert_eq!(pos.entry_commission, 0.50);
        // Margen efectivo neto = 100.0 - 0.50 = 99.50
        assert_eq!(pos.effective_margin(), Some(99.50));
        // liq_price calculado con margen neto: (100 - (99.5/10)) / 0.995 = 90.05 / 0.995 = 90.5025
        assert!((pos.liquidation_price - (90.05 / 0.995)).abs() < 1e-4);

        // 2. Vela 2 se desploma a 80.0, detonando liquidación forzosa
        let c2 = Kline {
            timestamp: 2000,
            open: 95.0,
            high: 96.0,
            low: 80.0,
            close: 82.0,
            volume: 50.0,
        };
        let trade = ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c2, 1)
            .expect("Debe liquidar la posición");

        // Verificaciones obligatorias de Allianz:
        // net_pnl = -100.0 exactamente
        assert_eq!(trade.net_pnl, -100.0, "net_pnl debe ser exactamente -100");
        // Saldo final = 900.0 exactamente
        assert_eq!(portfolio.cash_balance, 900.0, "Saldo final debe ser exactamente 900");
        assert!(portfolio.active_position.is_none());
    }

    #[test]
    fn test_c03_isolated_normal_exit_accounting_cents_match() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            margin_mode: MarginMode::Isolated,
            commission_rate: 0.001,
            position_sizing: PositionSizing::FixedAmount(100.0),
            leverage: 1.0, // 100 notional, 1.0 unidad a 100.0
            ..Default::default()
        };

        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // Al abrir, cash_balance sigue en 1000.0
        assert_eq!(portfolio.cash_balance, 1000.0);

        // Salir con ganancia al Open de la siguiente vela a 110.0 (gross +10.0, exit comm = 0.11, entry comm = 0.10)
        let exit_sig = Signal::flat(2000);
        let c2 = dummy_candle(110.0);
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &exit_sig, &c2, 1)
            .expect("Debe cerrar");

        let expected_net_pnl = 10.0 - 0.21;
        assert!((trade.net_pnl - expected_net_pnl).abs() < 1e-6);
        assert!((portfolio.cash_balance - (1000.0 + expected_net_pnl)).abs() < 1e-6);
        // Cuadre al centavo entre variación de balance y net_pnl
        assert!(((portfolio.cash_balance - 1000.0) - trade.net_pnl).abs() < 1e-6);
    }

    #[test]
    fn test_c04_same_direction_maintains_position() {
        let mut portfolio = Portfolio::new(1000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            ..Default::default()
        };

        // Entrar en Long
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);
        assert!(portfolio.has_position());

        // Señal en la misma dirección (Long) en la vela 2: debe mantener la posición intacta
        let keep_sig = Signal::new(SignalDirection::Long, 1.0, 2000);
        let c2 = dummy_candle(102.0);
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &keep_sig, &c2, 1);
        assert!(trade.is_none(), "Señal en la misma dirección no debe cerrar la posición");
        assert!(portfolio.has_position(), "Posición debe seguir activa");
        assert_eq!(portfolio.active_position.as_ref().unwrap().entry_price, 100.0);
    }

    #[test]
    fn test_c14_mfe_mae_not_polluted_by_open_gap_exit() {
        let mut portfolio = Portfolio::new(10000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            fixed_stop_loss_pct: Some(0.05), // SL a 95.0
            ..Default::default()
        };

        // Long a 100.0
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // Vela 2 abre con Gap bajista violento a 80.0 (debajo del SL 95.0)
        // con mecha alcista transitoria a 120.0 y caída a 70.0
        let c2_gap = Kline {
            timestamp: 2000,
            open: 80.0,
            high: 120.0,
            low: 70.0,
            close: 75.0,
            volume: 100.0,
        };

        let trade = ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c2_gap, 1)
            .expect("Debe salir por Stop Loss al Open por Gap");

        assert_eq!(trade.exit_price, 80.0);
        // C-14: Al haber salido en el Open a 80.0, los extremos no deben incorporar el High 120.0 ni el Low 70.0
        // MFE debe ser 0.0% (el precio nunca estuvo por encima de 100.0 mientras la posición estuvo viva)
        assert!((trade.max_favorable_excursion - 0.0).abs() < 1e-4, "MFE no debe contaminarse con High de la vela de salida por gap");
        // MAE debe ser 20.0% ((100 - 80) / 100 * 100), NO 30.0% ((100 - 70) / 100)
        assert!((trade.max_adverse_excursion - 20.0).abs() < 1e-4, "MAE debe reflejar la salida al Open (80.0), no el Low (70.0)");
    }

    #[test]
    fn test_c14_intra_candle_sl_exit_clamps_mae() {
        let mut portfolio = Portfolio::new(10000.0);
        let config = BacktestConfig {
            commission_rate: 0.0,
            fixed_stop_loss_pct: Some(0.05), // SL a 95.0
            ..Default::default()
        };

        // Long a 100.0
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, 1000);
        let c1 = dummy_candle(100.0);
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // Vela 2 abre normal a 100.0, sube a 102.0, y luego cae a 85.0
        let c2 = Kline {
            timestamp: 2000,
            open: 100.0,
            high: 102.0,
            low: 85.0,
            close: 88.0,
            volume: 100.0,
        };

        let trade = ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c2, 1)
            .expect("Debe salir por Stop Loss a 95.0");

        assert_eq!(trade.exit_price, 95.0);
        // Favorable estuvo vivo hasta 102.0 (+2%)
        assert!((trade.max_favorable_excursion - 2.0).abs() < 1e-4);
        // C-14: La posición murió en 95.0, por lo que el Low de 85.0 post-stop no debe sobreestimar el MAE
        // MAE debe ser 5.0% ((100 - 95) / 100 * 100), NO 15.0% ((100 - 85) / 100)
        assert!((trade.max_adverse_excursion - 5.0).abs() < 1e-4, "MAE debe acotarse al precio de salida real de Stop Loss (95.0)");
    }

    #[test]
    fn test_c11_isolated_funding_deducted_from_margin_recalculates_liquidation() {
        let mut portfolio = Portfolio::new(1000.0);
        let mut ms = MicrostructureConfig::default();
        ms.funding_config.enabled = true;
        ms.funding_config.base_funding_rate = 0.005; // 0.5% por corte -> en 1000 nocional = $5.00 por corte

        let config = BacktestConfig {
            margin_mode: MarginMode::Isolated,
            commission_rate: 0.0005, // 0.05% -> $0.50 en 1000 nocional
            position_sizing: PositionSizing::FixedAmount(100.0), // $100 margen
            leverage: 10.0, // 1000 nocional (10 unidades a $100)
            microstructure: Some(ms),
            ..Default::default()
        };

        // 1. Abrir Long a $100 en t0 = 00:00 UTC
        let t0 = 1767225600000_i64;
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, t0);
        let c1 = Kline { timestamp: t0, open: 100.0, high: 101.0, low: 99.0, close: 100.0, volume: 10.0 };
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // Al abrir: cash_balance intacto en $1000.0
        assert_eq!(portfolio.cash_balance, 1000.0);
        let initial_liq = portfolio.active_position.as_ref().unwrap().liquidation_price;
        // initial_liq con 99.50 de margen: (100 - 9.95) / 0.995 = 90.05 / 0.995 = 90.50251
        assert!((initial_liq - (90.05 / 0.995)).abs() < 1e-4);

        // 2. Devengo de funding a las 08:00 UTC ($5.00)
        let ms_engine = MicrostructureEngine::new(config.effective_microstructure());
        let t_8h = t0 + 8 * 3600 * 1000;
        let (events, payment) = ms_engine.apply_funding_step(&mut portfolio, 100.0, t_8h, None);
        assert_eq!(events, 1);
        assert_eq!(payment, 5.0);

        // C-11: cash_balance libre NO se debita por el funding (permanece en 1000.0)
        assert_eq!(portfolio.cash_balance, 1000.0, "En Isolated, el funding no se descuenta de cash_balance libre");
        let pos = portfolio.active_position.as_ref().unwrap();
        assert_eq!(pos.accumulated_funding, 5.0);
        // effective_margin se reduce: 100 - 0.50 (comisión) - 5.00 (funding) = 94.50
        assert_eq!(pos.effective_margin(), Some(94.50));

        // 3. Evaluar exits: el recálculo dinámico causal debe actualizar pos.liquidation_price
        // new_liq: (100 - (94.50 / 10)) / 0.995 = (100 - 9.45) / 0.995 = 90.55 / 0.995 = 91.005025
        let c2 = Kline { timestamp: t_8h, open: 98.0, high: 99.0, low: 95.0, close: 97.0, volume: 10.0 };
        ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c2, 1);

        let pos_after = portfolio.active_position.as_ref().unwrap();
        let updated_liq = pos_after.liquidation_price;
        assert!(
            updated_liq > initial_liq,
            "El funding descontado del margen debe haber elevado/acercado el precio de liquidación: updated={}, initial={}",
            updated_liq,
            initial_liq
        );
        assert!((updated_liq - (90.55 / 0.995)).abs() < 1e-4);

        // 4. Salida por Liquidación en Vela 3 con caída a 85.0
        let c3 = Kline { timestamp: t_8h + 3600_000, open: 92.0, high: 92.0, low: 85.0, close: 86.0, volume: 10.0 };
        let trade = ExecutionEngine::check_intra_candle_exits(&mut portfolio, &config, &c3, 2)
            .expect("Debe liquidar la posición");

        // Verificaciones obligatorias de Allianz:
        // Pérdida neta acotada contractualmente al margen asignado de $100 (incluyendo comisiones y funding)
        assert_eq!(trade.net_pnl, -100.0, "net_pnl en liquidación aislada debe ser exactamente -isolated_margin (-100)");
        assert_eq!(portfolio.cash_balance, 900.0, "Saldo final de cuenta debe ser 900.0 exactos");
        assert_eq!(trade.funding_paid, 5.0, "CompletedTrade debe registrar el funding pagado de 5.0");
    }

    #[test]
    fn test_c11_isolated_normal_exit_with_funding_cents_match() {
        let mut portfolio = Portfolio::new(1000.0);
        let mut ms = MicrostructureConfig::default();
        ms.funding_config.enabled = true;
        ms.funding_config.base_funding_rate = 0.002; // 0.2% -> $2.00 en $1000 nocional

        let config = BacktestConfig {
            margin_mode: MarginMode::Isolated,
            commission_rate: 0.001, // 0.1% -> $1.00 en entrada, $1.10 en salida a 110
            position_sizing: PositionSizing::FixedAmount(100.0),
            leverage: 10.0,
            microstructure: Some(ms),
            ..Default::default()
        };

        let t0 = 1767225600000_i64;
        let enter_sig = Signal::new(SignalDirection::Long, 1.0, t0);
        let c1 = Kline { timestamp: t0, open: 100.0, high: 101.0, low: 99.0, close: 100.0, volume: 10.0 };
        ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &enter_sig, &c1, 0);

        // Cobrar funding de $2.00
        let ms_engine = MicrostructureEngine::new(config.effective_microstructure());
        let t_8h = t0 + 8 * 3600 * 1000;
        ms_engine.apply_funding_step(&mut portfolio, 100.0, t_8h, None);
        assert_eq!(portfolio.cash_balance, 1000.0); // intacto

        // Salir al Open a 110.0:
        // gross = (110 - 100) * 10 = +100.0
        // exit_comm = 1100 * 0.001 = 1.10
        // entry_comm = 1.00
        // funding = 2.00
        // net_pnl = 100.0 - 2.10 - 2.00 = 95.90
        let exit_sig = Signal::flat(t_8h + 3600_000);
        let c2 = Kline { timestamp: t_8h + 3600_000, open: 110.0, high: 111.0, low: 109.0, close: 110.0, volume: 10.0 };
        let trade = ExecutionEngine::process_signal_at_open(&mut portfolio, &config, &exit_sig, &c2, 1)
            .expect("Debe cerrar");

        assert!((trade.net_pnl - 95.90).abs() < 1e-6);
        assert!((portfolio.cash_balance - 1095.90).abs() < 1e-6);
        assert!(((portfolio.cash_balance - 1000.0) - trade.net_pnl).abs() < 1e-6);
        assert_eq!(trade.funding_paid, 2.0);
    }
}

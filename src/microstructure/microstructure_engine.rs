use serde::{Deserialize, Serialize};
use crate::backtest::portfolio::Portfolio;
use super::exit_reason::ExitReason;
use super::funding_rate_simulator::FundingRateSimulator;
use super::microstructure_config::MicrostructureConfig;
use super::order_execution_type::OrderExecutionType;

/// Motor central de microestructura y costos de ejecución para Binance Futures USD-M.
/// Orquesta el cómputo de comisiones asimétricas Maker/Taker y la acumulación periódica de Funding Rates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicrostructureEngine {
    pub config: MicrostructureConfig,
    pub funding_simulator: FundingRateSimulator,
}

impl Default for MicrostructureEngine {
    fn default() -> Self {
        let config = MicrostructureConfig::default();
        let funding_simulator = FundingRateSimulator::new(config.funding_config);
        Self {
            config,
            funding_simulator,
        }
    }
}

impl MicrostructureEngine {
    /// Inicializa el motor a partir de la configuración provista.
    pub fn new(config: MicrostructureConfig) -> Self {
        let funding_simulator = FundingRateSimulator::new(config.funding_config);
        Self {
            config,
            funding_simulator,
        }
    }

    /// Calcula la comisión exigible en la apertura de la posición (Entry Fill).
    pub fn calculate_entry_commission(&self, notional: f64) -> f64 {
        self.config
            .fee_schedule
            .calculate_commission(notional, self.config.entry_order_type)
    }

    /// Calcula la comisión aplicable al cierre de la posición en base al motivo causal tipado de salida.
    /// Retorna una tupla: `(comision_monto, tipo_ejecucion_utilizado)`.
    pub fn calculate_exit_commission_typed(&self, notional: f64, exit_reason: ExitReason) -> (f64, OrderExecutionType) {
        let exec_type = self.config.execution_type_for_exit_reason_typed(exit_reason);
        let commission = self.config.fee_schedule.calculate_commission(notional, exec_type);
        (commission, exec_type)
    }

    /// Calcula la comisión aplicable al cierre de la posición en base al motivo causal de salida.
    /// Retorna una tupla: `(comision_monto, tipo_ejecucion_utilizado)`.
    pub fn calculate_exit_commission(&self, notional: f64, exit_reason: &str) -> (f64, OrderExecutionType) {
        let exec_type = self.config.execution_type_for_exit_reason(exit_reason);
        let commission = self.config.fee_schedule.calculate_commission(notional, exec_type);
        (commission, exec_type)
    }

    fn apply_funding_to_position(
        &self,
        pos: &mut crate::types::Position,
        cash_balance: &mut f64,
        total_funding_paid: &mut f64,
        mark_price: f64,
        target_timestamp: i64,
    ) -> (u32, f64) {
        let prev_timestamp = pos.last_funding_timestamp;
        let funding_simulator = FundingRateSimulator::new(self.config.funding_config);
        let (events, payment) = funding_simulator.evaluate_funding_accrual(
            pos.side,
            pos.quantity,
            mark_price,
            prev_timestamp,
            target_timestamp,
            None,
        );

        if events > 0 {
            if pos.isolated_margin.is_none() {
                *cash_balance -= payment;
            }
            *total_funding_paid += payment;
            pos.apply_funding(payment, target_timestamp);
        }

        (events, payment)
    }

    pub fn apply_funding_step(
        &self,
        portfolio: &mut Portfolio,
        mark_price: f64,
        curr_timestamp: i64,
        candle_duration: Option<i64>,
    ) -> (u32, f64) {
        if !self.config.funding_config.enabled {
            return (0, 0.0);
        }

        let _ = candle_duration;
        let target_timestamp = curr_timestamp;

        let mut total_events = 0_u32;
        let mut total_payment = 0.0_f64;

        for slot in [
            &mut portfolio.active_position,
            &mut portfolio.long_position,
            &mut portfolio.short_position,
        ] {
            if let Some(pos) = slot.as_mut() {
                let (events, payment) = self.apply_funding_to_position(
                    pos,
                    &mut portfolio.cash_balance,
                    &mut portfolio.total_funding_paid,
                    mark_price,
                    target_timestamp,
                );
                total_events += events;
                total_payment += payment;
            }
        }

        (total_events, total_payment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Position, PositionSide};

    #[test]
    fn test_microstructure_engine_commission_differentiation() {
        let engine = MicrostructureEngine::default();
        let notional = 10_000.0;

        // Entrada al mercado: Taker (0.05% = $5.00)
        let entry_comm = engine.calculate_entry_commission(notional);
        assert!((entry_comm - 5.0).abs() < 1e-6);

        // Salida por Take Profit: Maker (0.02% = $2.00)
        let (tp_comm, tp_exec) = engine.calculate_exit_commission(notional, "Intra-candle Take Profit Hit");
        assert_eq!(tp_exec, OrderExecutionType::Maker);
        assert!((tp_comm - 2.0).abs() < 1e-6);

        // Salida por Stop Loss: Taker (0.05% = $5.00)
        let (sl_comm, sl_exec) = engine.calculate_exit_commission(notional, "Intra-candle Stop Loss Hit");
        assert_eq!(sl_exec, OrderExecutionType::Taker);
        assert!((sl_comm - 5.0).abs() < 1e-6);

        // Verificación con ExitReason tipado directo (C-20)
        let (tp_typed_comm, tp_typed_exec) = engine.calculate_exit_commission_typed(notional, ExitReason::TakeProfit);
        assert_eq!(tp_typed_exec, OrderExecutionType::Maker);
        assert!((tp_typed_comm - 2.0).abs() < 1e-6);

        let (sl_typed_comm, sl_typed_exec) = engine.calculate_exit_commission_typed(notional, ExitReason::StopLoss);
        assert_eq!(sl_typed_exec, OrderExecutionType::Taker);
        assert!((sl_typed_comm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_microstructure_engine_funding_application() {
        let mut engine = MicrostructureEngine::default();
        // Configurar tasa neutral base 0.0001 (0.01%)
        engine.config.funding_config.base_funding_rate = 0.0001;

        let mut portfolio = Portfolio::new(10_000.0);
        let t0 = 1767225600000_i64; // 00:00 UTC
        let t_8h = t0 + 8 * 3600 * 1000; // 08:00 UTC

        // Crear posición Long de 1 BTC a $50,000 (Nocional = $50,000)
        let pos = Position::new(
            PositionSide::Long,
            50_000.0,
            1.0,
            t0,
            0,
            None,
            None,
            None,
            0.0,
            25.0,
        );
        portfolio.active_position = Some(pos);

        // Al avanzar a las 08:00 UTC con mark_price = $50,000
        // Nocional = $50,000 * 0.0001 = $5.00 pago de funding
        let (events, payment) = engine.apply_funding_step(&mut portfolio, 50_000.0, t_8h, None);

        assert_eq!(events, 1);
        assert!((payment - 5.0).abs() < 1e-6);
        assert!((portfolio.cash_balance - 9995.0).abs() < 1e-6);
        assert!((portfolio.total_funding_paid - 5.0).abs() < 1e-6);

        let active_pos = portfolio.active_position.as_ref().unwrap();
        assert!((active_pos.accumulated_funding - 5.0).abs() < 1e-6);
        assert_eq!(active_pos.last_funding_timestamp, t_8h);
    }

    #[test]
    fn test_microstructure_engine_multi_8h_does_not_advance_past_curr_timestamp() {
        let mut engine = MicrostructureEngine::default();
        engine.config.funding_config.base_funding_rate = 0.001;

        let mut portfolio = Portfolio::new(10_000.0);
        let t0 = 1767225600000_i64;
        let day_ms = 24 * 3600 * 1000;

        let pos = Position::new(
            PositionSide::Long,
            100.0,
            10.0,
            t0,
            0,
            None,
            None,
            None,
            0.0,
            0.0,
        );
        portfolio.active_position = Some(pos);

        let (events_open, payment_open) =
            engine.apply_funding_step(&mut portfolio, 100.0, t0, Some(day_ms));
        assert_eq!(events_open, 0);
        assert_eq!(payment_open, 0.0);
        assert_eq!(
            portfolio.active_position.as_ref().unwrap().last_funding_timestamp,
            t0
        );

        let t_next_day = t0 + day_ms;
        let (events_held, payment_held) =
            engine.apply_funding_step(&mut portfolio, 100.0, t_next_day, Some(day_ms));
        assert_eq!(events_held, 3);
        assert!((payment_held - 3.0).abs() < 1e-6);
        assert_eq!(
            portfolio.active_position.as_ref().unwrap().last_funding_timestamp,
            t_next_day
        );
    }
}

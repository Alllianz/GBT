use serde::{Deserialize, Serialize};
use crate::types::PositionSide;
use super::funding_config::FundingConfig;
use super::timestamp_unit::TimestampUnit;

/// Simulador de liquidación de Tasas de Financiación (Funding Rates) para contratos perpetuos.
/// Garantiza causalidad temporal estricta ($t \le T$) y modelado exacto de los cortes UTC (00:00, 08:00, 16:00).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FundingRateSimulator {
    pub config: FundingConfig,
}

impl Default for FundingRateSimulator {
    fn default() -> Self {
        Self {
            config: FundingConfig::default(),
        }
    }
}

impl FundingRateSimulator {
    /// Crea un nuevo simulador con la configuración de financiación especificada.
    pub fn new(config: FundingConfig) -> Self {
        Self { config }
    }

    /// Asigna una unidad de escala temporal tipada explícita (C-36).
    pub fn with_timestamp_unit(mut self, unit: TimestampUnit) -> Self {
        self.config.timestamp_unit = unit;
        self
    }

    /// Retorna la duración del intervalo de financiación según la unidad temporal tipada configurada (C-36).
    #[inline]
    pub fn interval_duration(&self) -> i64 {
        self.config.interval_duration()
    }

    /// Determina si un timestamp está expresado en milisegundos (Unix ms) o segundos.
    ///
    /// # Obsolescencia (C-36 / S4-H07):
    /// Esta función heurística basada en umbrales estáticos está en desuso debido a que falla
    /// en series sintéticas o relativas que comienzan en $t = 0$. Use `FundingConfig.timestamp_unit`
    /// o `self.interval_duration()` en su lugar.
    #[deprecated(since = "0.2.0", note = "Usar TimestampUnit tipado en FundingConfig")]
    #[inline]
    pub fn is_millisecond_timestamp(timestamp: i64) -> bool {
        timestamp.abs() > 10_000_000_000
    }

    /// Obtiene el tamaño del intervalo de financiación en la unidad configurada.
    /// Mantenido por compatibilidad con firmas previas.
    #[inline]
    pub fn interval_duration_for_timestamp(&self, _timestamp: i64) -> i64 {
        self.interval_duration()
    }

    /// Verifica si un timestamp exacto coincide con la frontera de corte de financiación (00:00, 08:00, 16:00 UTC).
    #[inline]
    pub fn is_funding_boundary(&self, timestamp: i64) -> bool {
        let interval = self.interval_duration();
        if interval <= 0 {
            return false;
        }
        timestamp % interval == 0
    }

    /// Calcula rigurosamente el número de cortes de liquidación de funding ocurridos en el intervalo semi-abierto $(T_{prev}, T_{curr}]$.
    /// $$\text{Events} = \left\lfloor \frac{T_{curr}}{\Delta} \right\rfloor - \left\lfloor \frac{T_{prev}}{\Delta} \right\rfloor$$
    pub fn count_funding_events(&self, prev_timestamp: i64, curr_timestamp: i64) -> u32 {
        if !self.config.enabled || curr_timestamp <= prev_timestamp {
            return 0;
        }

        let interval = self.interval_duration();
        if interval <= 0 {
            return 0;
        }

        let prev_bucket = prev_timestamp.div_euclid(interval);
        let curr_bucket = curr_timestamp.div_euclid(interval);

        (curr_bucket - prev_bucket).max(0) as u32
    }

    /// Computa el flujo de caja de financiación para un evento de liquidación sobre una posición activa:
    /// $$\text{CashflowPayment} = \begin{cases}
    /// +\text{Notional} \times r & \text{si Long (paga si } r > 0 \text{, cobra si } r < 0 \text{)} \\
    /// -\text{Notional} \times r & \text{si Short (cobra si } r > 0 \text{, paga si } r < 0 \text{)}
    /// \end{cases}$$
    ///
    /// # Robustez Numérica (C-26 / S4-H05):
    /// - Si `position_quantity`, `mark_price` o `funding_rate` no son finitos (`NaN` o `\pm\infty`),
    ///   retorna `0.0` para blindar el balance de caja contra contaminación matemática.
    ///
    /// Retorno:
    /// - Valor > 0: El trader debe **pagar** financiación (débito/costo que reduce su saldo).
    /// - Valor < 0: El trader **cobra** financiación (crédito/ingreso que incrementa su saldo).
    #[inline]
    pub fn calculate_single_cashflow(
        side: PositionSide,
        position_quantity: f64,
        mark_price: f64,
        funding_rate: f64,
    ) -> f64 {
        if !position_quantity.is_finite() || !mark_price.is_finite() || !funding_rate.is_finite() {
            return 0.0;
        }

        let notional = (position_quantity * mark_price).abs();
        if !notional.is_finite() {
            return 0.0;
        }

        let payment = match side {
            PositionSide::Long => notional * funding_rate,
            PositionSide::Short => -notional * funding_rate,
        };

        if payment.is_finite() {
            payment
        } else {
            0.0
        }
    }

    /// Evalúa el paso temporal entre `prev_timestamp` y `curr_timestamp` para una posición abierta.
    /// Retorna una tupla: `(eventos_de_funding, costo_total_acumulado)`.
    ///
    /// # Robustez y Sanitización (C-26 / S4-H05):
    /// - Verifica estrictamente la finitud y positividad de `quantity` y `mark_price`.
    /// - Sanitiza `custom_rate` ante `NaN` o valores no finitos, recurriendo a la tasa base de la configuración.
    /// - Neutraliza cualquier posibilidad de emitir `NaN` hacia el balance de portafolio.
    pub fn evaluate_funding_accrual(
        &self,
        side: PositionSide,
        quantity: f64,
        mark_price: f64,
        prev_timestamp: i64,
        curr_timestamp: i64,
        custom_rate: Option<f64>,
    ) -> (u32, f64) {
        if !self.config.enabled
            || !quantity.is_finite()
            || quantity <= 0.0
            || !mark_price.is_finite()
            || mark_price <= 0.0
        {
            return (0, 0.0);
        }

        let events = self.count_funding_events(prev_timestamp, curr_timestamp);
        if events == 0 {
            return (0, 0.0);
        }

        let raw_rate = match custom_rate {
            Some(r) if r.is_finite() => r,
            Some(_) => self.config.base_funding_rate,
            None => self.config.base_funding_rate,
        };
        let clamped_rate = self.config.clamped_rate(raw_rate);

        let single_payment = Self::calculate_single_cashflow(side, quantity, mark_price, clamped_rate);
        let total_payment = single_payment * (events as f64);

        if !total_payment.is_finite() {
            return (events, 0.0);
        }

        (events, total_payment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_boundary_detection_milliseconds() {
        let sim = FundingRateSimulator::default();
        // 2026-01-01 00:00:00 UTC en ms = 1767225600000 (múltiplo exacto de 28_800_000)
        let t_00_00 = 1767225600000_i64;
        let t_08_00 = t_00_00 + 8 * 3600 * 1000;
        let t_04_00 = t_00_00 + 4 * 3600 * 1000;

        assert!(sim.is_funding_boundary(t_00_00));
        assert!(sim.is_funding_boundary(t_08_00));
        assert!(!sim.is_funding_boundary(t_04_00));
    }

    #[test]
    fn test_funding_boundary_detection_seconds() {
        let sim = FundingRateSimulator::default().with_timestamp_unit(TimestampUnit::Seconds);
        let t_00_00_sec = 1767225600_i64;
        let t_08_00_sec = t_00_00_sec + 8 * 3600;
        let t_01_00_sec = t_00_00_sec + 3600;

        assert!(sim.is_funding_boundary(t_00_00_sec));
        assert!(sim.is_funding_boundary(t_08_00_sec));
        assert!(!sim.is_funding_boundary(t_01_00_sec));
    }

    #[test]
    fn test_count_funding_events_hourly_progression() {
        let sim = FundingRateSimulator::default();
        let base_ts = 1767225600000_i64; // 00:00 UTC

        // De 00:00 a 07:00: 0 eventos
        assert_eq!(sim.count_funding_events(base_ts, base_ts + 7 * 3600 * 1000), 0);

        // De 00:00 a 08:00: exactamente 1 evento (corte de las 08:00)
        assert_eq!(sim.count_funding_events(base_ts, base_ts + 8 * 3600 * 1000), 1);

        // De 07:59 a 08:01: exactamente 1 evento
        let t_7_59 = base_ts + (7 * 3600 + 59 * 60) * 1000;
        let t_8_01 = base_ts + (8 * 3600 + 1 * 60) * 1000;
        assert_eq!(sim.count_funding_events(t_7_59, t_8_01), 1);

        // De 00:00 a 24:00 (1 día entero): exactamente 3 eventos (08:00, 16:00, 24:00/00:00)
        assert_eq!(sim.count_funding_events(base_ts, base_ts + 24 * 3600 * 1000), 3);
    }

    #[test]
    fn test_symmetry_long_short_funding_cashflow() {
        let quantity = 1.0;
        let price = 10_000.0;
        let positive_rate = 0.0001; // +0.01% (mercado alcista, Longs pagan a Shorts)

        let long_cost = FundingRateSimulator::calculate_single_cashflow(
            PositionSide::Long,
            quantity,
            price,
            positive_rate,
        );
        let short_cost = FundingRateSimulator::calculate_single_cashflow(
            PositionSide::Short,
            quantity,
            price,
            positive_rate,
        );

        // Long paga $1.0 (+1.0)
        assert!((long_cost - 1.0).abs() < 1e-8);
        // Short cobra $1.0 (-1.0)
        assert!((short_cost - (-1.0)).abs() < 1e-8);
        // Suma cero perfecta
        assert!((long_cost + short_cost).abs() < 1e-8);

        // Tasa negativa: mercado bajista, Shorts pagan a Longs
        let negative_rate = -0.0002;
        let long_cost_neg = FundingRateSimulator::calculate_single_cashflow(
            PositionSide::Long,
            quantity,
            price,
            negative_rate,
        );
        let short_cost_neg = FundingRateSimulator::calculate_single_cashflow(
            PositionSide::Short,
            quantity,
            price,
            negative_rate,
        );

        // Long cobra $2.0 (-2.0)
        assert!((long_cost_neg - (-2.0)).abs() < 1e-8);
        // Short paga $2.0 (+2.0)
        assert!((short_cost_neg - 2.0).abs() < 1e-8);
        // Suma cero perfecta
        assert!((long_cost_neg + short_cost_neg).abs() < 1e-8);
    }

    #[test]
    fn test_c36_synthetic_relative_timestamp_from_zero_milliseconds() {
        // En series sintéticas relativas que inician en t = 0 ms:
        // Una vela o avance de 8 horas equivale a 28_800_000 ms.
        // Con la heurística previa is_millisecond_timestamp (> 10_000_000_000), 28_800_000 se interpretaba
        // como segundos, produciendo 28_800_000 / 28_800 = 1,000 eventos erróneos (S4-H07 / C-36).
        // Con TimestampUnit::Milliseconds tipado, produce exactamente 1 corte de funding.
        let sim = FundingRateSimulator::default(); // default = Milliseconds
        assert_eq!(sim.config.timestamp_unit, TimestampUnit::Milliseconds);

        let t_start = 0_i64;
        let t_8h_ms = 28_800_000_i64; // 8 horas en ms

        let events = sim.count_funding_events(t_start, t_8h_ms);
        assert_eq!(
            events, 1,
            "C-36: Una serie sintética relativa de 8h en ms debe producir exactamente 1 evento, no 1000"
        );

        let (accrual_events, payment) = sim.evaluate_funding_accrual(
            PositionSide::Long,
            1.0,
            10_000.0,
            t_start,
            t_8h_ms,
            None,
        );
        assert_eq!(accrual_events, 1);
        assert!((payment - 1.0).abs() < 1e-8); // 10,000 * 0.0001 = $1.00
    }

    #[test]
    fn test_c26_nan_and_inf_cashflow_sanitization() {
        // Ningún valor NaN o Inf debe generar pagos NaN o Inf en calculate_single_cashflow
        assert_eq!(
            FundingRateSimulator::calculate_single_cashflow(PositionSide::Long, f64::NAN, 10_000.0, 0.0001),
            0.0
        );
        assert_eq!(
            FundingRateSimulator::calculate_single_cashflow(PositionSide::Long, 1.0, f64::NAN, 0.0001),
            0.0
        );
        assert_eq!(
            FundingRateSimulator::calculate_single_cashflow(PositionSide::Long, 1.0, 10_000.0, f64::NAN),
            0.0
        );
        assert_eq!(
            FundingRateSimulator::calculate_single_cashflow(PositionSide::Long, f64::INFINITY, 10_000.0, 0.0001),
            0.0
        );
        assert_eq!(
            FundingRateSimulator::calculate_single_cashflow(PositionSide::Long, 1.0, f64::INFINITY, 0.0001),
            0.0
        );
    }

    #[test]
    fn test_c26_nan_and_inf_accrual_sanitization() {
        let sim = FundingRateSimulator::default();
        let base_ts = 1767225600000_i64;
        let curr_ts = base_ts + 8 * 3600 * 1000;

        // Cantidad NaN
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, f64::NAN, 10_000.0, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        // Precio NaN
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, f64::NAN, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        // Precio Infinito
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, f64::INFINITY, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        // Custom rate NaN (debe neutralizarse recurriendo a base_funding_rate)
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, 10_000.0, base_ts, curr_ts, Some(f64::NAN));
        assert_eq!(ev, 1);
        assert!((pay - 1.0).abs() < 1e-8); // recurre a base rate 0.0001 -> $1.00

        // Custom rate Infinito (debe neutralizarse recurriendo a base_funding_rate)
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, 10_000.0, base_ts, curr_ts, Some(f64::INFINITY));
        assert_eq!(ev, 1);
        assert!((pay - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_c26_negative_and_zero_inputs() {
        let sim = FundingRateSimulator::default();
        let base_ts = 1767225600000_i64;
        let curr_ts = base_ts + 8 * 3600 * 1000;

        // Cantidad <= 0
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 0.0, 10_000.0, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, -1.0, 10_000.0, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        // Precio <= 0
        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, 0.0, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);

        let (ev, pay) = sim.evaluate_funding_accrual(PositionSide::Long, 1.0, -500.0, base_ts, curr_ts, None);
        assert_eq!(ev, 0);
        assert_eq!(pay, 0.0);
    }
}

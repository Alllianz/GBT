use serde::{Deserialize, Serialize};
use super::timestamp_unit::TimestampUnit;

/// Configuración del mecanismo de Tasa de Financiación (Funding Rate) para futuros perpetuos.
/// Modela los intercambios periódicos de flujo de caja entre posiciones Long y Short cada 8 horas en Binance USD-M.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FundingConfig {
    /// Activa o desactiva la simulación de funding rates.
    pub enabled: bool,
    /// Tasa de financiación base (estándar neutral de Binance: 0.0001 = 0.01% por período de 8h).
    pub base_funding_rate: f64,
    /// Límite inferior de saturación de la tasa (estándar Binance: -0.0075 = -0.75%).
    pub clamp_min: f64,
    /// Límite superior de saturación de la tasa (estándar Binance: +0.0075 = +0.75%).
    pub clamp_max: f64,
    /// Intervalo de liquidación en horas (estándar Binance: 8 horas, cortes a las 00:00, 08:00 y 16:00 UTC).
    pub interval_hours: u32,
    /// Unidad de tiempo explícita de los timestamps de la serie (C-36). Por defecto: Milisegundos.
    pub timestamp_unit: TimestampUnit,
}

impl Default for FundingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_funding_rate: 0.0001, // 0.01% por intervalo de 8 horas
            clamp_min: -0.0075,        // -0.75%
            clamp_max: 0.0075,         // +0.75%
            interval_hours: 8,
            timestamp_unit: TimestampUnit::Milliseconds,
        }
    }
}

impl FundingConfig {
    /// Construye una configuración con estado activo y tasa base especificada.
    pub fn new(enabled: bool, base_funding_rate: f64) -> Self {
        Self {
            enabled,
            base_funding_rate,
            ..Default::default()
        }
    }

    /// Asigna una unidad de escala temporal tipada explícita (C-36).
    pub fn with_timestamp_unit(mut self, unit: TimestampUnit) -> Self {
        self.timestamp_unit = unit;
        self
    }

    /// Asigna límites de acotamiento (clamping) validando su orden relativo (`min <= max`).
    pub fn with_clamps(mut self, min: f64, max: f64) -> Result<Self, String> {
        if !min.is_finite() || !max.is_finite() {
            return Err("Los límites de clamping deben ser números finitos (no NaN ni Inf)".to_string());
        }
        if min > max {
            return Err(format!("clamp_min ({}) no puede ser mayor que clamp_max ({})", min, max));
        }
        self.clamp_min = min;
        self.clamp_max = max;
        Ok(self)
    }

    /// Retorna la duración del intervalo de financiación en la unidad temporal configurada (`timestamp_unit`).
    #[inline]
    pub fn interval_duration(&self) -> i64 {
        self.timestamp_unit.interval_duration(self.interval_hours)
    }

    /// Retorna la duración del intervalo de financiación en milisegundos.
    #[inline]
    pub fn interval_millis(&self) -> i64 {
        self.interval_hours as i64 * 3_600_000
    }

    /// Retorna la duración del intervalo de financiación en segundos.
    #[inline]
    pub fn interval_seconds(&self) -> i64 {
        self.interval_hours as i64 * 3_600
    }

    /// Aplica el acotamiento oficial (clamping) de Binance a una tasa de financiación observada.
    ///
    /// # Robustez Numérica y Prevención de Pánicos (C-26 / S4-H05):
    /// - Si `raw_rate` no es un número finito (`NaN` o `\pm\infty`), se rechaza de forma segura
    ///   retornando `self.base_funding_rate` (o `0.0` si la tasa base tampoco fuese finita).
    /// - Si la configuración contiene límites inválidos (`clamp_min > clamp_max` o no finitos),
    ///   se neutraliza el riesgo de pánico en `f64::clamp` retornando la tasa base segura.
    #[inline]
    pub fn clamped_rate(&self, raw_rate: f64) -> f64 {
        let safe_fallback = if self.base_funding_rate.is_finite() {
            self.base_funding_rate
        } else {
            0.0
        };

        if !raw_rate.is_finite() {
            return safe_fallback;
        }

        if !self.clamp_min.is_finite() || !self.clamp_max.is_finite() || self.clamp_min > self.clamp_max {
            return safe_fallback;
        }

        raw_rate.clamp(self.clamp_min, self.clamp_max)
    }

    /// Valida que la configuración cumpla con los invariantes financieros institucionales (C-26):
    /// - `base_funding_rate`, `clamp_min` y `clamp_max` deben ser finitos.
    /// - `clamp_min <= clamp_max`.
    /// - `interval_hours > 0`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.base_funding_rate.is_finite() {
            return Err(format!("base_funding_rate no es finita: {}", self.base_funding_rate));
        }
        if !self.clamp_min.is_finite() {
            return Err(format!("clamp_min no es finita: {}", self.clamp_min));
        }
        if !self.clamp_max.is_finite() {
            return Err(format!("clamp_max no es finita: {}", self.clamp_max));
        }
        if self.clamp_min > self.clamp_max {
            return Err(format!(
                "clamp_min ({}) no puede ser mayor que clamp_max ({})",
                self.clamp_min, self.clamp_max
            ));
        }
        if self.interval_hours == 0 {
            return Err("interval_hours debe ser estrictamente positivo (> 0)".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_config_default() {
        let config = FundingConfig::default();
        assert!(config.enabled);
        assert!((config.base_funding_rate - 0.0001).abs() < 1e-8);
        assert_eq!(config.interval_hours, 8);
        assert_eq!(config.interval_millis(), 28_800_000);
        assert_eq!(config.interval_seconds(), 28_800);
        assert_eq!(config.timestamp_unit, TimestampUnit::Milliseconds);
        assert_eq!(config.interval_duration(), 28_800_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_funding_config_clamping_standard() {
        let config = FundingConfig::default();
        assert_eq!(config.clamped_rate(0.01), 0.0075);
        assert_eq!(config.clamped_rate(-0.01), -0.0075);
        assert_eq!(config.clamped_rate(0.0003), 0.0003);
    }

    #[test]
    fn test_c26_clamping_nan_inf_sanitization() {
        let config = FundingConfig::default();
        // NaN debe neutralizarse retornando la tasa base
        assert_eq!(config.clamped_rate(f64::NAN), config.base_funding_rate);
        // Infinitos deben neutralizarse retornando la tasa base
        assert_eq!(config.clamped_rate(f64::INFINITY), config.base_funding_rate);
        assert_eq!(config.clamped_rate(f64::NEG_INFINITY), config.base_funding_rate);
    }

    #[test]
    fn test_c26_clamping_inverted_bounds_panic_immunity() {
        let mut config = FundingConfig::default();
        // Configuración patológica: clamp_min > clamp_max
        config.clamp_min = 0.01;
        config.clamp_max = -0.01;

        // No debe causar pánico; debe retornar safe_fallback
        assert_eq!(config.clamped_rate(0.0005), config.base_funding_rate);
        assert_eq!(config.clamped_rate(f64::NAN), config.base_funding_rate);
    }

    #[test]
    fn test_c26_validation_invariants() {
        let mut valid_config = FundingConfig::default();
        assert!(valid_config.validate().is_ok());

        // Error: clamp_min > clamp_max
        valid_config.clamp_min = 0.05;
        valid_config.clamp_max = 0.01;
        assert!(valid_config.validate().is_err());

        // Error: interval_hours == 0
        valid_config.clamp_min = -0.01;
        valid_config.interval_hours = 0;
        assert!(valid_config.validate().is_err());

        // Error: NaN en base_funding_rate
        valid_config.interval_hours = 8;
        valid_config.base_funding_rate = f64::NAN;
        assert!(valid_config.validate().is_err());
    }

    #[test]
    fn test_with_timestamp_unit_seconds() {
        let config = FundingConfig::default().with_timestamp_unit(TimestampUnit::Seconds);
        assert_eq!(config.timestamp_unit, TimestampUnit::Seconds);
        assert_eq!(config.interval_duration(), 28_800);
    }
}

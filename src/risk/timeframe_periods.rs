//! Módulo dedicado al cálculo tipado de períodos por año y factores de anualización de volatilidad
//! en mercados de futuros de criptomonedas (Binance Futures USD-M).
//!
//! # Principio Institucional (C-21):
//! En criptomonedas, el mercado opera 24 horas al día, 7 días a la semana, los 365 días del año de forma continua.
//! Por lo tanto, la base de anualización no puede asumir el estándar de bolsas tradicionales (252 días * 24h = 6,048),
//! sino la base real cripto (365 días * 24h = 8,760 horas/año).
//!
//! Fórmula general:
//!   `períodos por año = 365 * 24 * 3600 * 1000 / timeframe_ms`
//!
//! Valores canónicos institucionales de Allianz:
//! - 15m (900,000 ms):   35,040 períodos/año
//! - 1H  (3,600,000 ms):  8,760 períodos/año
//! - 4H  (14,400,000 ms): 2,190 períodos/año
//! - 1D  (86,400,000 ms):   365 períodos/año

use crate::types::Timeframe;
use serde::{Deserialize, Serialize};

/// Días por año en mercados de criptomonedas (operación continua 24/7/365).
pub const CRYPTO_DAYS_PER_YEAR: f64 = 365.0;

/// Horas por día calendario estándar.
pub const HOURS_PER_DAY: f64 = 24.0;

/// Horas por año en mercados de criptomonedas (365 * 24 = 8,760 horas).
pub const CRYPTO_HOURS_PER_YEAR: f64 = CRYPTO_DAYS_PER_YEAR * HOURS_PER_DAY;

/// Segundos por hora estándar.
pub const SECONDS_PER_HOUR: f64 = 3600.0;

/// Segundos totales por año cripto (365 * 24 * 3600 = 31,536,000 segundos).
pub const CRYPTO_SECONDS_PER_YEAR: f64 = CRYPTO_HOURS_PER_YEAR * SECONDS_PER_HOUR;

/// Milisegundos totales por año cripto (365 * 24 * 3600 * 1000 = 31,536,000,000 ms).
pub const CRYPTO_MILLIS_PER_YEAR: f64 = CRYPTO_SECONDS_PER_YEAR * 1000.0;

/// Calcula el número exacto de períodos por año para un timeframe de criptomonedas continuo 24/7.
///
/// Fórmula:
///   `365 * 24 * 3600 * 1000 / timeframe_ms`
#[inline]
pub fn crypto_periods_per_year(timeframe: Timeframe) -> f64 {
    let ms = timeframe.duration_millis() as f64;
    if ms > 0.0 {
        CRYPTO_MILLIS_PER_YEAR / ms
    } else {
        CRYPTO_HOURS_PER_YEAR
    }
}

/// Calcula el factor de anualización de la volatilidad para un timeframe de criptomonedas.
///
/// Fórmula:
///   `sqrt(períodos por año)`
#[inline]
pub fn crypto_annualization_factor(timeframe: Timeframe) -> f64 {
    crypto_periods_per_year(timeframe).sqrt()
}

/// Calcula los períodos por año a partir de una duración arbitraria en milisegundos.
#[inline]
pub fn periods_per_year_from_millis(timeframe_ms: i64) -> f64 {
    if timeframe_ms > 0 {
        CRYPTO_MILLIS_PER_YEAR / (timeframe_ms as f64)
    } else {
        CRYPTO_HOURS_PER_YEAR
    }
}

/// Calcula el factor de anualización de volatilidad a partir de una duración arbitraria en milisegundos.
#[inline]
pub fn annualization_factor_from_millis(timeframe_ms: i64) -> f64 {
    periods_per_year_from_millis(timeframe_ms).sqrt()
}

/// Estructura tipada y explícita para la configuración de anualización de volatilidad.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeframeAnnualization {
    /// Timeframe tipado oficial
    pub timeframe: Timeframe,
    /// Períodos calculados por año calendario continuo (365 días)
    pub periods_per_year: f64,
    /// Factor de escala de desviación estándar / volatilidad (`sqrt(periods_per_year)`)
    pub annualization_factor: f64,
}

impl TimeframeAnnualization {
    /// Construye la configuración de anualización a partir del timeframe tipado.
    pub fn new(timeframe: Timeframe) -> Self {
        let periods = crypto_periods_per_year(timeframe);
        Self {
            timeframe,
            periods_per_year: periods,
            annualization_factor: periods.sqrt(),
        }
    }
}

impl From<Timeframe> for TimeframeAnnualization {
    fn from(tf: Timeframe) -> Self {
        Self::new(tf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c21_exact_periods_per_year_for_allianz_timeframes() {
        // Validación obligatoria de los 4 timeframes de operación de Allianz:
        // 15m => 35040
        let p_15m = crypto_periods_per_year(Timeframe::M15);
        assert_eq!(p_15m, 35040.0, "15m debe tener exactamente 35040 períodos/año");

        // 1H => 8760
        let p_1h = crypto_periods_per_year(Timeframe::H1);
        assert_eq!(p_1h, 8760.0, "1H debe tener exactamente 8760 períodos/año");

        // 4H => 2190
        let p_4h = crypto_periods_per_year(Timeframe::H4);
        assert_eq!(p_4h, 2190.0, "4H debe tener exactamente 2190 períodos/año");

        // 1D => 365
        let p_1d = crypto_periods_per_year(Timeframe::D1);
        assert_eq!(p_1d, 365.0, "1D debe tener exactamente 365 períodos/año");
    }

    #[test]
    fn test_c21_annualization_factors_coherence() {
        assert_eq!(crypto_annualization_factor(Timeframe::M15), (35040.0_f64).sqrt());
        assert_eq!(crypto_annualization_factor(Timeframe::H1), (8760.0_f64).sqrt());
        assert_eq!(crypto_annualization_factor(Timeframe::H4), (2190.0_f64).sqrt());
        assert_eq!(crypto_annualization_factor(Timeframe::D1), (365.0_f64).sqrt());
    }

    #[test]
    fn test_c21_from_millis_calculation() {
        assert_eq!(periods_per_year_from_millis(900_000), 35040.0);
        assert_eq!(periods_per_year_from_millis(3_600_000), 8760.0);
        assert_eq!(periods_per_year_from_millis(14_400_000), 2190.0);
        assert_eq!(periods_per_year_from_millis(86_400_000), 365.0);
    }
}

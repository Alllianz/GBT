use serde::{Deserialize, Serialize};

/// Unidad de escala temporal para timestamps en el motor de backtest y microestructura.
///
/// Sustituye heurísticas numéricas estáticas (como `is_millisecond_timestamp`) por una
/// especificación explícita y fuertemente tipada de la granularidad temporal de la serie.
///
/// En Binance Futures USD-M oficial, los timestamps de velas (Kline/OHLCV) se expresan
/// exclusivamente en milisegundos Unix (`Milliseconds`), el cual actúa como valor predeterminado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum TimestampUnit {
    /// Milisegundos Unix (estándar oficial de Binance Futures USD-M para datos de mercado).
    #[default]
    Milliseconds,
    /// Segundos Unix (utilizado en series agregadas, sintéticas o feeds de baja frecuencia).
    Seconds,
}

impl TimestampUnit {
    /// Retorna la duración de un intervalo de `hours` horas en la unidad representada.
    #[inline]
    pub fn interval_duration(&self, hours: u32) -> i64 {
        match self {
            Self::Milliseconds => hours as i64 * 3_600_000,
            Self::Seconds => hours as i64 * 3_600,
        }
    }

    /// Retorna la cantidad de unidades temporales contenidas en un segundo estándar.
    #[inline]
    pub fn units_per_second(&self) -> i64 {
        match self {
            Self::Milliseconds => 1_000,
            Self::Seconds => 1,
        }
    }

    /// Convierte un timestamp a milisegundos Unix con protección contra desbordamiento (`saturating_mul`).
    #[inline]
    pub fn to_millis(&self, timestamp: i64) -> i64 {
        match self {
            Self::Milliseconds => timestamp,
            Self::Seconds => timestamp.saturating_mul(1_000),
        }
    }

    /// Convierte un timestamp a segundos Unix mediante división euclídea entera.
    #[inline]
    pub fn to_seconds(&self, timestamp: i64) -> i64 {
        match self {
            Self::Milliseconds => timestamp.div_euclid(1_000),
            Self::Seconds => timestamp,
        }
    }

    /// Retorna `true` si la unidad temporal corresponde a milisegundos.
    #[inline]
    pub fn is_milliseconds(&self) -> bool {
        matches!(self, Self::Milliseconds)
    }

    /// Retorna `true` si la unidad temporal corresponde a segundos.
    #[inline]
    pub fn is_seconds(&self) -> bool {
        matches!(self, Self::Seconds)
    }
}

impl std::fmt::Display for TimestampUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Milliseconds => write!(f, "MILLISECONDS"),
            Self::Seconds => write!(f, "SECONDS"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_milliseconds() {
        assert_eq!(TimestampUnit::default(), TimestampUnit::Milliseconds);
        assert!(TimestampUnit::default().is_milliseconds());
        assert!(!TimestampUnit::default().is_seconds());
    }

    #[test]
    fn test_interval_duration_calculation() {
        let ms = TimestampUnit::Milliseconds;
        assert_eq!(ms.interval_duration(8), 28_800_000);
        assert_eq!(ms.interval_duration(1), 3_600_000);
        assert_eq!(ms.interval_duration(0), 0);

        let sec = TimestampUnit::Seconds;
        assert_eq!(sec.interval_duration(8), 28_800);
        assert_eq!(sec.interval_duration(1), 3_600);
        assert_eq!(sec.interval_duration(0), 0);
    }

    #[test]
    fn test_timestamp_conversions() {
        let ms_unit = TimestampUnit::Milliseconds;
        let sec_unit = TimestampUnit::Seconds;

        let t_sec = 1_700_000_000_i64;
        let t_ms = 1_700_000_000_000_i64;

        assert_eq!(ms_unit.to_millis(t_ms), t_ms);
        assert_eq!(ms_unit.to_seconds(t_ms), t_sec);

        assert_eq!(sec_unit.to_millis(t_sec), t_ms);
        assert_eq!(sec_unit.to_seconds(t_sec), t_sec);
    }

    #[test]
    fn test_display_representation() {
        assert_eq!(format!("{}", TimestampUnit::Milliseconds), "MILLISECONDS");
        assert_eq!(format!("{}", TimestampUnit::Seconds), "SECONDS");
    }
}

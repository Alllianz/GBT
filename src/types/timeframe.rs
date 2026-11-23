use std::fmt;
use std::str::FromStr;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    M1,
    M3,
    M5,
    M15,
    M30,
    H1,
    H2,
    H4,
    H6,
    H8,
    H12,
    D1,
    D3,
    W1,
}

impl Timeframe {
    pub fn as_binance_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M3 => "3m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1h",
            Timeframe::H2 => "2h",
            Timeframe::H4 => "4h",
            Timeframe::H6 => "6h",
            Timeframe::H8 => "8h",
            Timeframe::H12 => "12h",
            Timeframe::D1 => "1d",
            Timeframe::D3 => "3d",
            Timeframe::W1 => "1w",
        }
    }

    /// Representación de almacenamiento en base de datos SQLite y convención de persistencia.
    ///
    /// C-30: Se resuelve la colisión semántica entre Binance y la base de datos:
    /// - Los minutos usan 'm' minúscula ("1m", "3m", "5m", "15m", "30m") alineándose con el estándar de Binance.
    /// - Se reserva 'M' mayúscula exclusivamente para marcos mensuales (Binance "1M"), impidiendo colisiones.
    /// - Las horas ("1H".."12H") y días ("1D".."3D") mantienen compatibilidad con el esquema existente.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M3 => "3m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1H",
            Timeframe::H2 => "2H",
            Timeframe::H4 => "4H",
            Timeframe::H6 => "6H",
            Timeframe::H8 => "8H",
            Timeframe::H12 => "12H",
            Timeframe::D1 => "1D",
            Timeframe::D3 => "3D",
            Timeframe::W1 => "1W",
        }
    }

    pub fn duration_millis(&self) -> i64 {
        match self {
            Timeframe::M1 => 60 * 1000,
            Timeframe::M3 => 3 * 60 * 1000,
            Timeframe::M5 => 5 * 60 * 1000,
            Timeframe::M15 => 15 * 60 * 1000,
            Timeframe::M30 => 30 * 60 * 1000,
            Timeframe::H1 => 60 * 60 * 1000,
            Timeframe::H2 => 2 * 60 * 60 * 1000,
            Timeframe::H4 => 4 * 60 * 60 * 1000,
            Timeframe::H6 => 6 * 60 * 60 * 1000,
            Timeframe::H8 => 8 * 60 * 60 * 1000,
            Timeframe::H12 => 12 * 60 * 60 * 1000,
            Timeframe::D1 => 24 * 60 * 60 * 1000,
            Timeframe::D3 => 3 * 24 * 60 * 60 * 1000,
            Timeframe::W1 => 7 * 24 * 60 * 60 * 1000,
        }
    }

    pub fn duration_seconds(&self) -> i64 {
        self.duration_millis() / 1000
    }

    /// Número exacto de períodos por año en mercado de criptomonedas continuo 24/7/365 (C-21).
    ///
    /// Fórmula oficial de Allianz:
    ///   `365 * 24 * 3600 * 1000 / timeframe_ms`
    ///
    /// Cobertura garantizada para la operativa de Allianz:
    /// - 15m => 35,040
    /// - 1H  =>  8,760
    /// - 4H  =>  2,190
    /// - 1D  =>    365
    #[inline]
    pub fn periods_per_year(&self) -> f64 {
        const MS_PER_YEAR_CRYPTO: f64 = 365.0 * 24.0 * 3600.0 * 1000.0;
        let ms = self.duration_millis() as f64;
        if ms > 0.0 {
            MS_PER_YEAR_CRYPTO / ms
        } else {
            365.0 * 24.0
        }
    }

    /// Factor de escala para anualizar la volatilidad del timeframe (`sqrt(periods_per_year)`).
    #[inline]
    pub fn annualization_factor(&self) -> f64 {
        self.periods_per_year().sqrt()
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_db_str())
    }
}

impl FromStr for Timeframe {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        // C-30: Parseo estricto distinguiendo minutos (minúscula 'm') de mes (mayúscula 'M')
        match trimmed {
            // Minutos en convención estándar Binance y alias comunes
            "1m" | "1min" => Ok(Timeframe::M1),
            "3m" | "3min" => Ok(Timeframe::M3),
            "5m" | "5min" => Ok(Timeframe::M5),
            "15m" | "15min" => Ok(Timeframe::M15),
            "30m" | "30min" => Ok(Timeframe::M30),

            // Horas (insensible a mayúsculas)
            "1h" | "1H" => Ok(Timeframe::H1),
            "2h" | "2H" => Ok(Timeframe::H2),
            "4h" | "4H" => Ok(Timeframe::H4),
            "6h" | "6H" => Ok(Timeframe::H6),
            "8h" | "8H" => Ok(Timeframe::H8),
            "12h" | "12H" => Ok(Timeframe::H12),

            // Días y semanas
            "1d" | "1D" => Ok(Timeframe::D1),
            "3d" | "3D" => Ok(Timeframe::D3),
            "1w" | "1W" => Ok(Timeframe::W1),

            // Detección explícita de colisión con convención Binance:
            // '1M' denota 1 mes (Monthly) en Binance, no soportado en GBT.
            "1M" => Err("Colisión semántica Binance: '1M' representa 1 mes (Monthly), no soportado en GBT. Para 1 minuto use '1m'".to_string()),

            // Compatibilidad legacy con base de datos para minutos no ambiguos en mayúscula
            "3M" => Ok(Timeframe::M3),
            "5M" => Ok(Timeframe::M5),
            "15M" => Ok(Timeframe::M15),
            "30M" => Ok(Timeframe::M30),

            _ => Err(format!("Timeframe no reconocido: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c21_timeframe_periods_per_year_for_allianz() {
        assert_eq!(Timeframe::M15.periods_per_year(), 35040.0);
        assert_eq!(Timeframe::H1.periods_per_year(), 8760.0);
        assert_eq!(Timeframe::H4.periods_per_year(), 2190.0);
        assert_eq!(Timeframe::D1.periods_per_year(), 365.0);
    }

    #[test]
    fn test_c21_timeframe_annualization_factors() {
        assert_eq!(Timeframe::M15.annualization_factor(), (35040.0_f64).sqrt());
        assert_eq!(Timeframe::H1.annualization_factor(), (8760.0_f64).sqrt());
        assert_eq!(Timeframe::H4.annualization_factor(), (2190.0_f64).sqrt());
        assert_eq!(Timeframe::D1.annualization_factor(), (365.0_f64).sqrt());
    }

    #[test]
    fn test_c30_binance_vs_db_disambiguation() {
        // Parseo de minutos en estándar Binance ('m' minúscula)
        assert_eq!("1m".parse::<Timeframe>().unwrap(), Timeframe::M1);
        assert_eq!("5m".parse::<Timeframe>().unwrap(), Timeframe::M5);
        assert_eq!("15m".parse::<Timeframe>().unwrap(), Timeframe::M15);

        // '1M' es 1 mes en Binance -> debe retornar error explícito de colisión
        let err_1m = "1M".parse::<Timeframe>();
        assert!(err_1m.is_err());
        assert!(err_1m.unwrap_err().contains("Colisión semántica Binance"));

        // Compatibilidad legacy para otros minutos mayúsculos
        assert_eq!("15M".parse::<Timeframe>().unwrap(), Timeframe::M15);

        // Formato para SQLite en minutos usa 'm' minúscula
        assert_eq!(Timeframe::M1.as_db_str(), "1m");
        assert_eq!(Timeframe::M15.as_db_str(), "15m");

        // Horas y días preservan compatibilidad con tabla existente
        assert_eq!(Timeframe::H1.as_db_str(), "1H");
        assert_eq!(Timeframe::D1.as_db_str(), "1D");
        assert_eq!("1h".parse::<Timeframe>().unwrap(), Timeframe::H1);
        assert_eq!("1H".parse::<Timeframe>().unwrap(), Timeframe::H1);
    }
}

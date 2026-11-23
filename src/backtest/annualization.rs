//! Módulo de inferencia de periodicidad y configuración de anualización para métricas estadísticas.
//!
//! # Responsabilidades (C-37):
//! 1. Inferencia robusta del timeframe y periodicidad a partir de la mediana de deltas entre
//!    timestamps consecutivos en la curva de equity (inmune a gaps temporales por fines de semana o mantenimiento).
//! 2. Filtrado causal contra timestamps no monótonos, duplicados o cero.
//! 3. Soporte transparente para unidades temporales en milisegundos (Binance Futures USD-M) y segundos.
//! 4. Base temporal 24/7 de 365 días para mercados de criptomonedas (31,536,000 segundos/año).

use super::equity_point::EquityPoint;

/// Días por año en mercados de criptomonedas (operación continua 24/7/365).
pub const DAYS_PER_YEAR_CRYPTO: f64 = 365.0;

/// Segundos por día calendario estándar.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Segundos totales en un año cripto (365 * 86,400 = 31,536,000s).
pub const SECONDS_PER_YEAR_CRYPTO: f64 = DAYS_PER_YEAR_CRYPTO * SECONDS_PER_DAY;

/// Número de barras por año por defecto ante series vacías o degeneradas (base diaria 365).
pub const DEFAULT_FALLBACK_BARS_PER_YEAR: f64 = DAYS_PER_YEAR_CRYPTO;

/// Configuración de anualización dimensional derivada de la periodicidad temporal de los datos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnualizationConfig {
    /// Número efectivo de barras o periodos en un año calendario cripto (365 días).
    pub bars_per_year: f64,
    /// Factor de anualización de volatilidad/desviación estándar: `sqrt(bars_per_year)`.
    pub annual_factor: f64,
    /// Duración del intervalo representativo entre barras en segundos.
    pub dt_seconds: f64,
}

impl Default for AnnualizationConfig {
    fn default() -> Self {
        let bars = DEFAULT_FALLBACK_BARS_PER_YEAR;
        Self {
            bars_per_year: bars,
            annual_factor: bars.sqrt(),
            dt_seconds: SECONDS_PER_DAY,
        }
    }
}

/// Inversor e inferidor de periodicidad temporal a partir de series de timestamps.
pub struct AnnualizationInferer;

impl AnnualizationInferer {
    /// Infiere la configuración de anualización a partir de una curva de puntos de equity.
    pub fn infer_from_curve(curve: &[EquityPoint]) -> AnnualizationConfig {
        if curve.len() < 2 {
            return AnnualizationConfig::default();
        }
        let timestamps: Vec<i64> = curve.iter().map(|p| p.timestamp).collect();
        Self::infer_from_timestamps(&timestamps)
    }

    /// Infiere la configuración de anualización a partir de un slice ordenado de timestamps.
    ///
    /// Implementa C-37:
    /// - Utiliza la mediana de diferencias consecutivas positivas `dt = t[i] - t[i-1]`.
    /// - Filtra y descarta diferencias `dt <= 0` (timestamps no monótonos o cero).
    /// - Si la serie está vacía o no tiene deltas válidos, recurre al valor por defecto institucional (365 días).
    pub fn infer_from_timestamps(timestamps: &[i64]) -> AnnualizationConfig {
        if timestamps.len() < 2 {
            return AnnualizationConfig::default();
        }

        // 1. Filtrado de deltas estrictamente positivos entre timestamps consecutivos
        let mut deltas: Vec<i64> = Vec::with_capacity(timestamps.len() - 1);
        for window in timestamps.windows(2) {
            let dt = window[1] - window[0];
            if dt > 0 {
                deltas.push(dt);
            }
        }

        if deltas.is_empty() {
            return AnnualizationConfig::default();
        }

        // 2. Mediana de deltas (C-37): robusta a gaps extremos
        deltas.sort_unstable();
        let mid = deltas.len() / 2;
        let median_dt = if deltas.len() % 2 == 1 {
            deltas[mid] as f64
        } else {
            (deltas[mid - 1] + deltas[mid]) as f64 / 2.0
        };

        if median_dt <= 0.0 || !median_dt.is_finite() {
            return AnnualizationConfig::default();
        }

        // 3. Inferencia de unidad temporal (segundos vs milisegundos)
        let max_ts = *timestamps.iter().max().unwrap_or(&0);
        let dt_seconds = if max_ts >= 100_000_000_000 {
            // Timestamp epoch en milisegundos (ej: >= 1e11, año 1973+)
            median_dt / 1000.0
        } else if max_ts >= 100_000_000 {
            // Timestamp epoch en segundos (1e8 a 1e11)
            median_dt
        } else if median_dt >= 10_000.0 && (median_dt % 1000.0 == 0.0) {
            // Timestamp sintético o relativo en milisegundos (>= 10s expresado en ms)
            median_dt / 1000.0
        } else if median_dt >= 1.0 {
            // Timestamp sintético en segundos
            median_dt
        } else {
            // Caso degenerado sub-segundo o corrupto
            SECONDS_PER_DAY
        };

        let safe_dt = if dt_seconds >= 1.0 && dt_seconds.is_finite() {
            dt_seconds
        } else {
            SECONDS_PER_DAY
        };

        let raw_bars_per_year = SECONDS_PER_YEAR_CRYPTO / safe_dt;
        // Acotamiento institucional del factor para evitar desbordamientos en ticks ultracortos
        let clamped_factor = raw_bars_per_year.sqrt().clamp(1.0, 2000.0);
        let effective_bars_per_year = clamped_factor * clamped_factor;

        AnnualizationConfig {
            bars_per_year: effective_bars_per_year,
            annual_factor: clamped_factor,
            dt_seconds: safe_dt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c37_crypto_365_days_basis() {
        // Intervalo diario: 86400 segundos
        let ts_daily = vec![0, 86400, 172800, 259200];
        let cfg = AnnualizationInferer::infer_from_timestamps(&ts_daily);

        assert_eq!(cfg.dt_seconds, 86400.0);
        assert!((cfg.bars_per_year - 365.0).abs() < 1e-6);
        assert!((cfg.annual_factor - (365.0_f64).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_c37_median_robust_against_gap() {
        // 10 velas horarias de 3600s con un gap de 48 horas (172800s) en el medio
        let mut ts = vec![1_700_000_000_i64];
        for _i in 1..=5 {
            ts.push(ts.last().unwrap() + 3600);
        }
        // Gap de 48 horas
        ts.push(ts.last().unwrap() + 172800);
        for _i in 1..=5 {
            ts.push(ts.last().unwrap() + 3600);
        }

        let cfg = AnnualizationInferer::infer_from_timestamps(&ts);
        // La mediana debe ser exactamente 3600 segundos a pesar del gap masivo
        assert_eq!(cfg.dt_seconds, 3600.0);
        assert!((cfg.bars_per_year - (365.0 * 24.0)).abs() < 1e-6);
    }

    #[test]
    fn test_c37_non_monotonic_and_zero_timestamps() {
        // Timestamps con ceros, retrocesos temporales y duplicados
        let ts = vec![0, 0, 1000, 1000, 500, 2000, 3000, 4000];
        let cfg = AnnualizationInferer::infer_from_timestamps(&ts);

        // Los deltas positivos son: 1000, 1500 (de 500 a 2000), 1000, 1000
        // Ordenados: [1000, 1000, 1000, 1500], mediana = 1000
        assert_eq!(cfg.dt_seconds, 1000.0);
        assert!(cfg.bars_per_year > 0.0);
    }

    #[test]
    fn test_c37_seconds_vs_millis_consistency() {
        let base_s = 1_700_000_000_i64;
        let base_ms = 1_700_000_000_000_i64;

        let ts_s = vec![base_s, base_s + 3600, base_s + 7200, base_s + 10800];
        let ts_ms = vec![base_ms, base_ms + 3_600_000, base_ms + 7_200_000, base_ms + 10_800_000];

        let cfg_s = AnnualizationInferer::infer_from_timestamps(&ts_s);
        let cfg_ms = AnnualizationInferer::infer_from_timestamps(&ts_ms);

        assert_eq!(cfg_s.dt_seconds, cfg_ms.dt_seconds);
        assert_eq!(cfg_s.bars_per_year, cfg_ms.bars_per_year);
        assert_eq!(cfg_s.annual_factor, cfg_ms.annual_factor);
    }

    #[test]
    fn test_c37_empty_and_single_point_fallback() {
        let empty_cfg = AnnualizationInferer::infer_from_timestamps(&[]);
        assert_eq!(empty_cfg.bars_per_year, 365.0);

        let single_cfg = AnnualizationInferer::infer_from_timestamps(&[1_700_000_000]);
        assert_eq!(single_cfg.bars_per_year, 365.0);
    }
}

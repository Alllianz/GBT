use crate::types::{Kline, Timeframe};

/// Representa una discontinuidad temporal (gap) en una serie de velas consecutivas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesGap {
    /// Timestamp de la vela previa a la interrupción (milisegundos)
    pub start_timestamp: i64,
    /// Timestamp de la vela posterior a la reanudación (milisegundos)
    pub end_timestamp: i64,
    /// Duración nominal esperada por barra según el timeframe (milisegundos)
    pub expected_interval_ms: i64,
    /// Cantidad estimada de velas ausentes en el intervalo
    pub missing_bars: usize,
}

/// Detector puro de discontinuidades temporales en series de velas OHLCV.
///
/// Cuantifica huecos temporales en la serie de datos respecto a la periodicidad
/// esperada sin inventar velas ni alterar los precios reales del mercado.
pub struct GapDetector;

impl GapDetector {
    /// Identifica gaps temporales en una serie ordenada de klines.
    /// Retorna una lista con todos los gaps encontrados y las barras faltantes.
    pub fn find_gaps(klines: &[Kline], tf: Timeframe) -> Vec<SeriesGap> {
        let step = tf.duration_millis();
        if step <= 0 || klines.len() < 2 {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        for window in klines.windows(2) {
            let prev_ts = window[0].timestamp;
            let next_ts = window[1].timestamp;
            let delta = next_ts - prev_ts;

            if delta > step {
                let missing = ((delta / step) as usize).saturating_sub(1);
                if missing > 0 {
                    gaps.push(SeriesGap {
                        start_timestamp: prev_ts,
                        end_timestamp: next_ts,
                        expected_interval_ms: step,
                        missing_bars: missing,
                    });
                }
            }
        }

        gaps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_kline(ts: i64) -> Kline {
        Kline {
            timestamp: ts,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 10.0,
        }
    }

    #[test]
    fn test_gap_detector_continuous_series() {
        let step = Timeframe::H1.duration_millis(); // 3_600_000
        let klines = vec![
            dummy_kline(1000),
            dummy_kline(1000 + step),
            dummy_kline(1000 + 2 * step),
            dummy_kline(1000 + 3 * step),
        ];

        let gaps = GapDetector::find_gaps(&klines, Timeframe::H1);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_gap_detector_identifies_missing_bars() {
        let step = Timeframe::H1.duration_millis();
        let klines = vec![
            dummy_kline(1000),
            // Salto de 3 horas (faltan 2 velas)
            dummy_kline(1000 + 3 * step),
            dummy_kline(1000 + 4 * step),
            // Salto de 5 horas (faltan 4 velas)
            dummy_kline(1000 + 9 * step),
        ];

        let gaps = GapDetector::find_gaps(&klines, Timeframe::H1);
        assert_eq!(gaps.len(), 2);

        assert_eq!(gaps[0].start_timestamp, 1000);
        assert_eq!(gaps[0].end_timestamp, 1000 + 3 * step);
        assert_eq!(gaps[0].missing_bars, 2);

        assert_eq!(gaps[1].start_timestamp, 1000 + 4 * step);
        assert_eq!(gaps[1].end_timestamp, 1000 + 9 * step);
        assert_eq!(gaps[1].missing_bars, 4);
    }

    #[test]
    fn test_gap_detector_empty_or_single_candle() {
        let empty: Vec<Kline> = Vec::new();
        assert!(GapDetector::find_gaps(&empty, Timeframe::H1).is_empty());

        let single = vec![dummy_kline(1000)];
        assert!(GapDetector::find_gaps(&single, Timeframe::H1).is_empty());
    }
}

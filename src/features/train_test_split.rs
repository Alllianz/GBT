use crate::types::Kline;

#[derive(Debug, Clone)]
pub struct SplitDataset {
    pub train_klines: Vec<Kline>,
    pub test_klines: Vec<Kline>,
    pub test_warmup_klines: Vec<Kline>,
    pub warmup_bars: usize,
    pub purge_bars: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub train_range_desc: String,
    pub test_range_desc: String,
}

pub struct TrainTestSplit;

impl TrainTestSplit {
    pub fn split_chronological(
        klines: &[Kline],
        train_ratio: f64,
        purge_bars: usize,
    ) -> Result<SplitDataset, String> {
        Self::split_chronological_with_warmup(klines, train_ratio, purge_bars, 200)
    }

    pub fn split_chronological_with_warmup(
        klines: &[Kline],
        train_ratio: f64,
        purge_bars: usize,
        warmup_bars: usize,
    ) -> Result<SplitDataset, String> {
        if !train_ratio.is_finite() || train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err("El parámetro train_ratio debe ser un número finito en el intervalo abierto (0, 1)".to_string());
        }

        let n = klines.len();
        if n < 100 {
            return Err("Se requieren al menos 100 velas para realizar una partición causal válida".to_string());
        }

        let train_end = (n as f64 * train_ratio).round() as usize;
        if train_end == 0 {
            return Err("El conjunto de entrenamiento debe contener al menos 1 vela".to_string());
        }
        let test_start = train_end + purge_bars;

        if test_start >= n {
            return Err("El tamaño de purga excede el conjunto de prueba disponible".to_string());
        }

        let actual_warmup_bars = test_start.min(warmup_bars);
        let test_warmup_start = test_start - actual_warmup_bars;
        let test_warmup_klines = klines[test_warmup_start..n].to_vec();
        let train_klines = klines[0..train_end].to_vec();
        let test_klines = klines[test_start..n].to_vec();

        let train_range_desc = format!(
            "Train: {} velas (idx 0..{})",
            train_klines.len(),
            train_end
        );
        let test_range_desc = format!(
            "Test (OOS): {} velas (idx {}..{}) [Purga: {} barras, Warmup: {} barras]",
            test_klines.len(),
            test_start,
            n,
            purge_bars,
            actual_warmup_bars
        );

        Ok(SplitDataset {
            train_klines,
            test_klines,
            test_warmup_klines,
            warmup_bars: actual_warmup_bars,
            purge_bars,
            train_end,
            test_start,
            train_range_desc,
            test_range_desc,
        })
    }

    pub fn split_with_warmup(
        klines: &[Kline],
        train_ratio: f64,
        purge_bars: usize,
        warmup_bars: usize,
    ) -> Result<SplitDataset, String> {
        Self::split_chronological_with_warmup(klines, train_ratio, purge_bars, warmup_bars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_kline(timestamp: i64, price: f64) -> Kline {
        Kline {
            timestamp,
            open: price,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume: 100.0,
        }
    }

    #[test]
    fn test_split_warmup_preserves_temporal_order() {
        let klines: Vec<Kline> = (0..150)
            .map(|i| dummy_kline(i as i64 * 3600, 100.0 + i as f64))
            .collect();

        let split = TrainTestSplit::split_chronological(&klines, 0.70, 5).unwrap();
        assert_eq!(split.warmup_bars, 110);
        assert_eq!(split.test_warmup_klines.len(), 150);
        assert_eq!(split.test_klines.len(), 150 - 110);
        assert_eq!(split.test_warmup_klines[split.warmup_bars].timestamp, split.test_klines[0].timestamp);
        assert_eq!(split.test_warmup_klines[split.warmup_bars - 1].timestamp, klines[109].timestamp);
    }

    #[test]
    fn test_c27_train_ratio_validation() {
        let klines: Vec<Kline> = (0..150)
            .map(|i| dummy_kline(i as i64 * 3600, 100.0 + i as f64))
            .collect();

        assert!(TrainTestSplit::split_chronological(&klines, f64::NAN, 5).is_err());
        assert!(TrainTestSplit::split_chronological(&klines, f64::INFINITY, 5).is_err());
        assert!(TrainTestSplit::split_chronological(&klines, f64::NEG_INFINITY, 5).is_err());

        assert!(TrainTestSplit::split_chronological(&klines, 0.0, 5).is_err());
        assert!(TrainTestSplit::split_chronological(&klines, -0.5, 5).is_err());
        assert!(TrainTestSplit::split_chronological(&klines, 1.0, 5).is_err());
        assert!(TrainTestSplit::split_chronological(&klines, 1.5, 5).is_err());

        let split_valid = TrainTestSplit::split_chronological(&klines, 0.70, 5);
        assert!(split_valid.is_ok());
    }

    #[test]
    fn test_c29_configurable_warmup_bars() {
        let klines: Vec<Kline> = (0..200)
            .map(|i| dummy_kline(i as i64 * 3600, 100.0 + i as f64))
            .collect();

        let split_default = TrainTestSplit::split_chronological(&klines, 0.70, 5).unwrap();
        assert_eq!(split_default.warmup_bars, 145);

        let split_custom = TrainTestSplit::split_chronological_with_warmup(&klines, 0.70, 5, 80).unwrap();
        assert_eq!(split_custom.warmup_bars, 80);
        assert_eq!(split_custom.test_warmup_klines.len(), 200 - (145 - 80));

        let split_small = TrainTestSplit::split_with_warmup(&klines, 0.70, 5, 15).unwrap();
        assert_eq!(split_small.warmup_bars, 15);
        assert_eq!(split_small.test_warmup_klines.len(), 200 - (145 - 15));
    }
}

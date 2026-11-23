use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Kline {
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Result<Self, String> {
        if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() || !volume.is_finite() {
            return Err("Los valores de la vela deben ser números finitos (no NaN ni Inf)".to_string());
        }
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return Err("Los precios de las velas deben ser estrictamente positivos".to_string());
        }
        if high < low {
            return Err(format!("High ({}) no puede ser menor que Low ({})", high, low));
        }
        let max_oc = open.max(close);
        if high < max_oc {
            return Err(format!(
                "High ({}) no puede ser menor que Open ({}) o Close ({})",
                high, open, close
            ));
        }
        let min_oc = open.min(close);
        if low > min_oc {
            return Err(format!(
                "Low ({}) no puede ser mayor que Open ({}) o Close ({})",
                low, open, close
            ));
        }
        if volume < 0.0 {
            return Err("El volumen no puede ser negativo".to_string());
        }

        Ok(Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        })
    }

    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    #[inline]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    #[inline]
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug41_finite_numbers_validation() {
        assert!(Kline::new(1000, f64::NAN, 105.0, 95.0, 100.0, 10.0).is_err());
        assert!(Kline::new(1000, 100.0, f64::INFINITY, 95.0, 100.0, 10.0).is_err());
        assert!(Kline::new(1000, 100.0, 105.0, f64::NEG_INFINITY, 100.0, 10.0).is_err());
        assert!(Kline::new(1000, 100.0, 105.0, 95.0, f64::NAN, 10.0).is_err());
        assert!(Kline::new(1000, 100.0, 105.0, 95.0, 100.0, f64::NAN).is_err());
        assert!(Kline::new(1000, 100.0, 105.0, 95.0, 100.0, 10.0).is_ok());
    }

    #[test]
    fn test_c15_geometric_ohlc_validation() {
        // High menor que Open
        let err_high_open = Kline::new(1000, 105.0, 100.0, 90.0, 95.0, 10.0);
        assert!(err_high_open.is_err());
        assert!(err_high_open.unwrap_err().contains("High (100) no puede ser menor"));

        // High menor que Close
        let err_high_close = Kline::new(1000, 95.0, 100.0, 90.0, 105.0, 10.0);
        assert!(err_high_close.is_err());
        assert!(err_high_close.unwrap_err().contains("High (100) no puede ser menor"));

        // Low mayor que Open
        let err_low_open = Kline::new(1000, 90.0, 110.0, 95.0, 100.0, 10.0);
        assert!(err_low_open.is_err());
        assert!(err_low_open.unwrap_err().contains("Low (95) no puede ser mayor"));

        // Low mayor que Close
        let err_low_close = Kline::new(1000, 100.0, 110.0, 95.0, 90.0, 10.0);
        assert!(err_low_close.is_err());
        assert!(err_low_close.unwrap_err().contains("Low (95) no puede ser mayor"));

        // Precios no positivos
        assert!(Kline::new(1000, 0.0, 100.0, 0.0, 50.0, 10.0).is_err());
        assert!(Kline::new(1000, -10.0, 100.0, -20.0, 50.0, 10.0).is_err());

        // Volumen negativo
        assert!(Kline::new(1000, 100.0, 110.0, 90.0, 105.0, -1.0).is_err());

        // Vela perfectamente válida y doji
        let valid = Kline::new(1000, 100.0, 110.0, 90.0, 105.0, 50.0);
        assert!(valid.is_ok());
        let doji = Kline::new(1000, 100.0, 100.0, 100.0, 100.0, 0.0);
        assert!(doji.is_ok());
    }
}

use crate::types::Kline;
use serde::Deserialize;

/// Representación exacta del array devuelto por la API REST de Binance:
/// [
///   0: Open time,
///   1: Open,
///   2: High,
///   3: Low,
///   4: Close,
///   5: Volume,
///   6: Close time,
///   7: Quote asset volume,
///   8: Number of trades,
///   9: Taker buy base asset volume,
///   10: Taker buy quote asset volume,
///   11: Ignore
/// ]
#[derive(Debug, Deserialize)]
pub struct RawBinanceKline(
    pub i64,    // 0: Open time
    pub String, // 1: Open
    pub String, // 2: High
    pub String, // 3: Low
    pub String, // 4: Close
    pub String, // 5: Volume
    pub i64,    // 6: Close time
    pub String, // 7: Quote asset volume
    pub i64,    // 8: Number of trades
    pub String, // 9: Taker buy base asset volume
    pub String, // 10: Taker buy quote asset volume
    pub String, // 11: Ignore
);

impl RawBinanceKline {
    pub fn to_kline(&self) -> Option<Kline> {
        let open = self.1.parse::<f64>().ok()?;
        let high = self.2.parse::<f64>().ok()?;
        let low = self.3.parse::<f64>().ok()?;
        let close = self.4.parse::<f64>().ok()?;
        let volume = self.5.parse::<f64>().ok()?;

        Kline::new(self.0, open, high, low, close, volume).ok()
    }

    #[inline]
    pub fn close_time(&self) -> i64 {
        self.6
    }
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    pub direction: SignalDirection,
    pub strength: f64,
    pub timestamp: i64,
    pub stop_loss_price: Option<f64>,
    pub take_profit_price: Option<f64>,
}

impl Signal {
    /// Sanitiza la fuerza de la señal: valores no finitos (NaN, Inf) se normalizan a 0.0
    /// y valores válidos se limitan al intervalo cerrado [0.0, 1.0].
    #[inline]
    fn sanitize_strength(strength: f64) -> f64 {
        if !strength.is_finite() {
            0.0
        } else {
            strength.clamp(0.0, 1.0)
        }
    }

    /// Filtra niveles de precio de SL/TP, descartando valores no finitos (NaN, Inf) o <= 0.0
    #[inline]
    fn sanitize_price_level(price: Option<f64>) -> Option<f64> {
        price.filter(|&p| p.is_finite() && p > 0.0)
    }

    pub fn new(direction: SignalDirection, strength: f64, timestamp: i64) -> Self {
        Self {
            direction,
            strength: Self::sanitize_strength(strength),
            timestamp,
            stop_loss_price: None,
            take_profit_price: None,
        }
    }

    pub fn with_sl_tp(
        direction: SignalDirection,
        strength: f64,
        timestamp: i64,
        stop_loss_price: Option<f64>,
        take_profit_price: Option<f64>,
    ) -> Self {
        Self {
            direction,
            strength: Self::sanitize_strength(strength),
            timestamp,
            stop_loss_price: Self::sanitize_price_level(stop_loss_price),
            take_profit_price: Self::sanitize_price_level(take_profit_price),
        }
    }

    pub fn flat(timestamp: i64) -> Self {
        Self {
            direction: SignalDirection::Flat,
            strength: 0.0,
            timestamp,
            stop_loss_price: None,
            take_profit_price: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c32_signal_strength_sanitization() {
        let sig_nan = Signal::new(SignalDirection::Long, f64::NAN, 1000);
        assert_eq!(sig_nan.strength, 0.0);

        let sig_inf = Signal::new(SignalDirection::Short, f64::INFINITY, 1000);
        assert_eq!(sig_inf.strength, 0.0);

        let sig_neg_inf = Signal::new(SignalDirection::Short, f64::NEG_INFINITY, 1000);
        assert_eq!(sig_neg_inf.strength, 0.0);

        let sig_neg = Signal::new(SignalDirection::Long, -0.5, 1000);
        assert_eq!(sig_neg.strength, 0.0);

        let sig_over = Signal::new(SignalDirection::Long, 1.5, 1000);
        assert_eq!(sig_over.strength, 1.0);

        let sig_ok = Signal::new(SignalDirection::Long, 0.65, 1000);
        assert_eq!(sig_ok.strength, 0.65);
    }

    #[test]
    fn test_c32_signal_sl_tp_filtering() {
        let sig = Signal::with_sl_tp(
            SignalDirection::Long,
            f64::NAN,
            1000,
            Some(f64::NAN),
            Some(-100.0),
        );
        assert_eq!(sig.strength, 0.0);
        assert_eq!(sig.stop_loss_price, None);
        assert_eq!(sig.take_profit_price, None);

        let sig_zero = Signal::with_sl_tp(
            SignalDirection::Long,
            0.5,
            1000,
            Some(0.0),
            Some(f64::INFINITY),
        );
        assert_eq!(sig_zero.stop_loss_price, None);
        assert_eq!(sig_zero.take_profit_price, None);

        let sig_valid = Signal::with_sl_tp(
            SignalDirection::Long,
            0.8,
            1000,
            Some(95.0),
            Some(110.0),
        );
        assert_eq!(sig_valid.strength, 0.8);
        assert_eq!(sig_valid.stop_loss_price, Some(95.0));
        assert_eq!(sig_valid.take_profit_price, Some(110.0));
    }
}

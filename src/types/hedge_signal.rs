use serde::{Deserialize, Serialize};
use crate::types::{Signal, SignalDirection, TradeRegime};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HedgeLegAction {
    Hold,
    Close,
    OpenOrKeep {
        strength: f64,
        stop_loss_price: Option<f64>,
        take_profit_price: Option<f64>,
        size_multiplier: f64,
        #[serde(default)]
        regime: TradeRegime,
    },
}

impl HedgeLegAction {
    #[inline]
    fn sanitize_strength(strength: f64) -> f64 {
        if !strength.is_finite() {
            0.0
        } else {
            strength.clamp(0.0, 1.0)
        }
    }

    #[inline]
    fn sanitize_price_level(price: Option<f64>) -> Option<f64> {
        price.filter(|&p| p.is_finite() && p > 0.0)
    }

    #[inline]
    fn sanitize_size_multiplier(size_multiplier: f64) -> f64 {
        if !size_multiplier.is_finite() {
            0.0
        } else {
            size_multiplier.max(0.0)
        }
    }

    pub fn open_or_keep(
        strength: f64,
        stop_loss_price: Option<f64>,
        take_profit_price: Option<f64>,
        size_multiplier: f64,
    ) -> Self {
        Self::open_or_keep_with_regime(
            strength,
            stop_loss_price,
            take_profit_price,
            size_multiplier,
            TradeRegime::PrimaryTrend,
        )
    }

    pub fn open_or_keep_with_regime(
        strength: f64,
        stop_loss_price: Option<f64>,
        take_profit_price: Option<f64>,
        size_multiplier: f64,
        regime: TradeRegime,
    ) -> Self {
        Self::OpenOrKeep {
            strength: Self::sanitize_strength(strength),
            stop_loss_price: Self::sanitize_price_level(stop_loss_price),
            take_profit_price: Self::sanitize_price_level(take_profit_price),
            size_multiplier: Self::sanitize_size_multiplier(size_multiplier),
            regime,
        }
    }

    #[inline]
    pub fn with_regime(self, target_regime: TradeRegime) -> Self {
        match self {
            Self::OpenOrKeep {
                strength,
                stop_loss_price,
                take_profit_price,
                size_multiplier,
                ..
            } => Self::OpenOrKeep {
                strength,
                stop_loss_price,
                take_profit_price,
                size_multiplier,
                regime: target_regime,
            },
            other => other,
        }
    }

    #[inline]
    pub fn regime(&self) -> TradeRegime {
        match self {
            Self::OpenOrKeep { regime, .. } => *regime,
            Self::Hold | Self::Close => TradeRegime::PrimaryTrend,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HedgeSignal {
    pub timestamp: i64,
    pub long: HedgeLegAction,
    pub short: HedgeLegAction,
}

impl HedgeSignal {
    pub fn hold(timestamp: i64) -> Self {
        Self {
            timestamp,
            long: HedgeLegAction::Hold,
            short: HedgeLegAction::Hold,
        }
    }

    pub fn flat(timestamp: i64) -> Self {
        Self {
            timestamp,
            long: HedgeLegAction::Close,
            short: HedgeLegAction::Close,
        }
    }

    pub fn from_net_signal(signal: Signal) -> Self {
        match signal.direction {
            SignalDirection::Long => Self {
                timestamp: signal.timestamp,
                long: HedgeLegAction::open_or_keep(
                    signal.strength,
                    signal.stop_loss_price,
                    signal.take_profit_price,
                    1.0,
                ),
                short: HedgeLegAction::Close,
            },
            SignalDirection::Short => Self {
                timestamp: signal.timestamp,
                long: HedgeLegAction::Close,
                short: HedgeLegAction::open_or_keep(
                    signal.strength,
                    signal.stop_loss_price,
                    signal.take_profit_price,
                    1.0,
                ),
            },
            SignalDirection::Flat => Self::flat(signal.timestamp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedge_leg_action_sanitization() {
        let action = HedgeLegAction::open_or_keep(
            f64::NAN,
            Some(-10.0),
            Some(f64::INFINITY),
            -2.0,
        );
        assert_eq!(
            action,
            HedgeLegAction::OpenOrKeep {
                strength: 0.0,
                stop_loss_price: None,
                take_profit_price: None,
                size_multiplier: 0.0,
                regime: TradeRegime::PrimaryTrend,
            }
        );

        let clamped = HedgeLegAction::open_or_keep(1.5, Some(95.0), Some(110.0), 2.5);
        assert_eq!(
            clamped,
            HedgeLegAction::OpenOrKeep {
                strength: 1.0,
                stop_loss_price: Some(95.0),
                take_profit_price: Some(110.0),
                size_multiplier: 2.5,
                regime: TradeRegime::PrimaryTrend,
            }
        );
    }

    #[test]
    fn test_hedge_signal_constructors_and_conversion() {
        assert_eq!(
            HedgeSignal::hold(100),
            HedgeSignal {
                timestamp: 100,
                long: HedgeLegAction::Hold,
                short: HedgeLegAction::Hold,
            }
        );

        assert_eq!(
            HedgeSignal::flat(200),
            HedgeSignal {
                timestamp: 200,
                long: HedgeLegAction::Close,
                short: HedgeLegAction::Close,
            }
        );

        let long_net = Signal::with_sl_tp(SignalDirection::Long, 0.8, 300, Some(90.0), Some(120.0));
        assert_eq!(
            HedgeSignal::from_net_signal(long_net),
            HedgeSignal {
                timestamp: 300,
                long: HedgeLegAction::OpenOrKeep {
                    strength: 0.8,
                    stop_loss_price: Some(90.0),
                    take_profit_price: Some(120.0),
                    size_multiplier: 1.0,
                    regime: TradeRegime::PrimaryTrend,
                },
                short: HedgeLegAction::Close,
            }
        );

        let short_net = Signal::new(SignalDirection::Short, 0.6, 400);
        assert_eq!(
            HedgeSignal::from_net_signal(short_net),
            HedgeSignal {
                timestamp: 400,
                long: HedgeLegAction::Close,
                short: HedgeLegAction::OpenOrKeep {
                    strength: 0.6,
                    stop_loss_price: None,
                    take_profit_price: None,
                    size_multiplier: 1.0,
                    regime: TradeRegime::PrimaryTrend,
                },
            }
        );

        assert_eq!(
            HedgeSignal::from_net_signal(Signal::flat(500)),
            HedgeSignal::flat(500)
        );
    }
}

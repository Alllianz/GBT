use serde::{Deserialize, Serialize};

/// Niveles de cliente VIP oficiales en Binance Futures USD-M (USDT-margined).
/// Cada nivel define tasas porcentuales decrecientes para órdenes pasivas (Maker) y agresivas (Taker).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum BinanceVipTier {
    #[default]
    Vip0,
    Vip1,
    Vip2,
    Vip3,
    Vip4,
    Vip5,
    Vip6,
    Vip7,
    Vip8,
    Vip9,
}

impl BinanceVipTier {
    /// Devuelve el nivel numérico correspondiente (0 a 9).
    pub fn level(&self) -> u8 {
        match self {
            Self::Vip0 => 0,
            Self::Vip1 => 1,
            Self::Vip2 => 2,
            Self::Vip3 => 3,
            Self::Vip4 => 4,
            Self::Vip5 => 5,
            Self::Vip6 => 6,
            Self::Vip7 => 7,
            Self::Vip8 => 8,
            Self::Vip9 => 9,
        }
    }

    /// Construye el tier correspondiente a partir de un entero. Niveles >= 9 se limitan a Vip9.
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Self::Vip0,
            1 => Self::Vip1,
            2 => Self::Vip2,
            3 => Self::Vip3,
            4 => Self::Vip4,
            5 => Self::Vip5,
            6 => Self::Vip6,
            7 => Self::Vip7,
            8 => Self::Vip8,
            _ => Self::Vip9,
        }
    }

    /// Tasa de comisión oficial para órdenes pasivas Maker en Binance USD-M Futures.
    /// Expresada en fracción decimal (ej. 0.0002 = 0.020%).
    pub fn maker_fee_rate(&self) -> f64 {
        match self {
            Self::Vip0 => 0.00020, // 0.020%
            Self::Vip1 => 0.00016, // 0.016%
            Self::Vip2 => 0.00014, // 0.014%
            Self::Vip3 => 0.00012, // 0.012%
            Self::Vip4 => 0.00010, // 0.010%
            Self::Vip5 => 0.00008, // 0.008%
            Self::Vip6 => 0.00006, // 0.006%
            Self::Vip7 => 0.00004, // 0.004%
            Self::Vip8 => 0.00002, // 0.002%
            Self::Vip9 => 0.00000, // 0.000% (cero comisión Maker para ultra high-frequency / MM)
        }
    }

    /// Tasa de comisión oficial para órdenes agresivas Taker en Binance USD-M Futures.
    /// Expresada en fracción decimal (ej. 0.0005 = 0.050%).
    pub fn taker_fee_rate(&self) -> f64 {
        match self {
            Self::Vip0 => 0.00050, // 0.050%
            Self::Vip1 => 0.00040, // 0.040%
            Self::Vip2 => 0.00035, // 0.035%
            Self::Vip3 => 0.00032, // 0.032%
            Self::Vip4 => 0.00030, // 0.030%
            Self::Vip5 => 0.00027, // 0.027%
            Self::Vip6 => 0.00025, // 0.025%
            Self::Vip7 => 0.00023, // 0.023%
            Self::Vip8 => 0.00020, // 0.020%
            Self::Vip9 => 0.00017, // 0.017%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vip_tier_default_is_vip0() {
        assert_eq!(BinanceVipTier::default(), BinanceVipTier::Vip0);
        assert_eq!(BinanceVipTier::default().level(), 0);
    }

    #[test]
    fn test_official_binance_fee_rates() {
        // VIP 0
        assert!((BinanceVipTier::Vip0.maker_fee_rate() - 0.00020).abs() < 1e-8);
        assert!((BinanceVipTier::Vip0.taker_fee_rate() - 0.00050).abs() < 1e-8);

        // VIP 5
        assert!((BinanceVipTier::Vip5.maker_fee_rate() - 0.00008).abs() < 1e-8);
        assert!((BinanceVipTier::Vip5.taker_fee_rate() - 0.00027).abs() < 1e-8);

        // VIP 9 (Maker gratis)
        assert!((BinanceVipTier::Vip9.maker_fee_rate() - 0.0).abs() < 1e-8);
        assert!((BinanceVipTier::Vip9.taker_fee_rate() - 0.00017).abs() < 1e-8);
    }

    #[test]
    fn test_from_level_conversion() {
        assert_eq!(BinanceVipTier::from_level(0), BinanceVipTier::Vip0);
        assert_eq!(BinanceVipTier::from_level(4), BinanceVipTier::Vip4);
        assert_eq!(BinanceVipTier::from_level(9), BinanceVipTier::Vip9);
        assert_eq!(BinanceVipTier::from_level(99), BinanceVipTier::Vip9);
    }
}

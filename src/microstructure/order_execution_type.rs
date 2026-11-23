use serde::{Deserialize, Serialize};

/// Modalidad de ejecución de una orden en el motor de emparejamiento (Matching Engine).
/// Determina si la orden provee liquidez pasiva (Maker) o remueve liquidez agresivamente (Taker).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum OrderExecutionType {
    /// Orden agresiva que cruza el spread de mercado de forma inmediata (Market Order, Stop Market).
    #[default]
    Taker,
    /// Orden pasiva que se deposita en el libro de órdenes proveyendo liquidez (Limit Order, Take Profit Limit).
    Maker,
}

impl OrderExecutionType {
    /// Retorna `true` si la ejecución es pasiva como creador de mercado (Maker).
    #[inline]
    pub fn is_maker(&self) -> bool {
        matches!(self, Self::Maker)
    }

    /// Retorna `true` si la ejecución es agresiva como tomador de liquidez (Taker).
    #[inline]
    pub fn is_taker(&self) -> bool {
        matches!(self, Self::Taker)
    }

    /// Construye la modalidad a partir de si la orden es Maker (`true`) o Taker (`false`).
    #[inline]
    pub fn from_is_maker(is_maker: bool) -> Self {
        if is_maker {
            Self::Maker
        } else {
            Self::Taker
        }
    }

    /// Retorna la representación estática en cadena de texto ("MAKER" o "TAKER").
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Maker => "MAKER",
            Self::Taker => "TAKER",
        }
    }
}

impl std::fmt::Display for OrderExecutionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Maker => write!(f, "MAKER"),
            Self::Taker => write!(f, "TAKER"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_execution_type_predicates() {
        let maker = OrderExecutionType::Maker;
        let taker = OrderExecutionType::Taker;

        assert!(maker.is_maker());
        assert!(!maker.is_taker());

        assert!(taker.is_taker());
        assert!(!taker.is_maker());
    }

    #[test]
    fn test_default_is_taker() {
        assert_eq!(OrderExecutionType::default(), OrderExecutionType::Taker);
    }

    #[test]
    fn test_from_is_maker_and_as_str() {
        assert_eq!(OrderExecutionType::from_is_maker(true), OrderExecutionType::Maker);
        assert_eq!(OrderExecutionType::from_is_maker(false), OrderExecutionType::Taker);
        assert_eq!(OrderExecutionType::Maker.as_str(), "MAKER");
        assert_eq!(OrderExecutionType::Taker.as_str(), "TAKER");
    }
}

use serde::{Deserialize, Serialize};
use super::order_execution_type::OrderExecutionType;
use super::vip_tier::BinanceVipTier;

/// Estructura de comisiones del exchange (Binance USD-M Futures).
/// Soporta tiers VIP oficiales, descuento opcional del 10% al pagar con BNB, y tarifas personalizadas.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BinanceFeeSchedule {
    pub vip_tier: BinanceVipTier,
    /// Descuento oficial del 10% en Binance Futures USD-M al activar el pago de comisiones con BNB.
    pub bnb_discount_enabled: bool,
    /// Tasa personalizada opcional para Maker (si se especifica, prevalece sobre el tier VIP).
    pub custom_maker_fee: Option<f64>,
    /// Tasa personalizada opcional para Taker (si se especifica, prevalece sobre el tier VIP).
    pub custom_taker_fee: Option<f64>,
}

impl Default for BinanceFeeSchedule {
    fn default() -> Self {
        Self {
            vip_tier: BinanceVipTier::Vip0,
            bnb_discount_enabled: false,
            custom_maker_fee: None,
            custom_taker_fee: None,
        }
    }
}

impl BinanceFeeSchedule {
    /// Crea un cronograma de comisiones a partir de un nivel VIP y la opción de descuento BNB.
    pub fn new(vip_tier: BinanceVipTier, bnb_discount_enabled: bool) -> Self {
        Self {
            vip_tier,
            bnb_discount_enabled,
            custom_maker_fee: None,
            custom_taker_fee: None,
        }
    }

    /// Crea un cronograma con comisiones personalizadas explícitas (Maker y Taker).
    pub fn with_custom_fees(maker_fee: f64, taker_fee: f64) -> Self {
        Self {
            vip_tier: BinanceVipTier::Vip0,
            bnb_discount_enabled: false,
            custom_maker_fee: Some(maker_fee),
            custom_taker_fee: Some(taker_fee),
        }
    }

    /// Retorna la tasa efectiva de comisión para órdenes Maker considerando el nivel VIP y el descuento BNB.
    #[inline]
    pub fn effective_maker_fee_rate(&self) -> f64 {
        let base_fee = self.custom_maker_fee.unwrap_or_else(|| self.vip_tier.maker_fee_rate());
        if self.bnb_discount_enabled {
            base_fee * 0.90
        } else {
            base_fee
        }
    }

    /// Retorna la tasa efectiva de comisión para órdenes Taker considerando el nivel VIP y el descuento BNB.
    #[inline]
    pub fn effective_taker_fee_rate(&self) -> f64 {
        let base_fee = self.custom_taker_fee.unwrap_or_else(|| self.vip_tier.taker_fee_rate());
        if self.bnb_discount_enabled {
            base_fee * 0.90
        } else {
            base_fee
        }
    }

    /// Retorna la tasa de comisión aplicable según el tipo de ejecución de la orden (`Maker` o `Taker`).
    #[inline]
    pub fn fee_rate_for(&self, execution_type: OrderExecutionType) -> f64 {
        match execution_type {
            OrderExecutionType::Maker => self.effective_maker_fee_rate(),
            OrderExecutionType::Taker => self.effective_taker_fee_rate(),
        }
    }

    /// Calcula la comisión en unidades de cuenta a partir del nocional transaccionado:
    /// $$\text{Commission} = |\text{Notional}| \times \text{FeeRate}$$
    #[inline]
    pub fn calculate_commission(&self, notional: f64, execution_type: OrderExecutionType) -> f64 {
        let rate = self.fee_rate_for(execution_type);
        notional.abs() * rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vip0_standard_fees() {
        let schedule = BinanceFeeSchedule::default();
        assert!((schedule.effective_maker_fee_rate() - 0.00020).abs() < 1e-8);
        assert!((schedule.effective_taker_fee_rate() - 0.00050).abs() < 1e-8);

        let comm_maker = schedule.calculate_commission(10_000.0, OrderExecutionType::Maker);
        let comm_taker = schedule.calculate_commission(10_000.0, OrderExecutionType::Taker);
        assert!((comm_maker - 2.0).abs() < 1e-6);
        assert!((comm_taker - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bnb_discount_10_percent() {
        let schedule = BinanceFeeSchedule::new(BinanceVipTier::Vip0, true);
        // 0.00020 * 0.90 = 0.00018
        // 0.00050 * 0.90 = 0.00045
        assert!((schedule.effective_maker_fee_rate() - 0.00018).abs() < 1e-8);
        assert!((schedule.effective_taker_fee_rate() - 0.00045).abs() < 1e-8);

        let comm_taker = schedule.calculate_commission(10_000.0, OrderExecutionType::Taker);
        assert!((comm_taker - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_custom_fees_override() {
        let schedule = BinanceFeeSchedule::with_custom_fees(0.0001, 0.0003);
        assert!((schedule.effective_maker_fee_rate() - 0.0001).abs() < 1e-8);
        assert!((schedule.effective_taker_fee_rate() - 0.0003).abs() < 1e-8);
    }
}

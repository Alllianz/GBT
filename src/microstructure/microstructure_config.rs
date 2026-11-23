use serde::{Deserialize, Serialize};
use super::exit_reason::ExitReason;
use super::fee_schedule::BinanceFeeSchedule;
use super::funding_config::FundingConfig;
use super::order_execution_type::OrderExecutionType;
use super::vip_tier::BinanceVipTier;

/// Configuración integral de microestructura y costes de ejecución institucional.
/// Integra tarifas por nivel VIP, tasas Maker/Taker diferenciadas y simulación de Funding Rates cada 8h.
///
/// DIRECTRIZ INSTITUCIONAL ESTRICTA DE ALLIANZ:
/// El proyecto GBT opera sin slippage (slippage = 0.0 absoluto en ejecución).
/// Cualquier campo `slippage_rate` en configuraciones serializadas legadas es ignorado causalmente.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicrostructureConfig {
    /// Cronograma oficial de comisiones Binance Futures USD-M.
    pub fee_schedule: BinanceFeeSchedule,
    /// Configuración de liquidación periódica de Tasas de Financiación.
    pub funding_config: FundingConfig,
    /// Tipo de ejecución asignado a las órdenes de apertura (Open). Por defecto: Taker.
    pub entry_order_type: OrderExecutionType,
    /// Tipo de ejecución asignado a las órdenes de Take Profit (órdenes límite pasivas). Por defecto: Maker.
    pub tp_order_type: OrderExecutionType,
    /// Tipo de ejecución asignado a órdenes de Stop Loss y Liquidación (órdenes de protección al mercado). Por defecto: Taker.
    pub sl_order_type: OrderExecutionType,
    /// Tipo de ejecución asignado a salidas por horizonte temporal o reversión de señal. Por defecto: Taker.
    pub time_exit_order_type: OrderExecutionType,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            fee_schedule: BinanceFeeSchedule::default(),
            funding_config: FundingConfig::default(),
            entry_order_type: OrderExecutionType::Taker,
            tp_order_type: OrderExecutionType::Maker,
            sl_order_type: OrderExecutionType::Taker,
            time_exit_order_type: OrderExecutionType::Taker,
        }
    }
}

impl MicrostructureConfig {
    /// Construye una configuración con el cronograma de comisiones y financiamiento provistos.
    pub fn new(fee_schedule: BinanceFeeSchedule, funding_config: FundingConfig) -> Self {
        Self {
            fee_schedule,
            funding_config,
            ..Default::default()
        }
    }

    /// Construye una configuración VIP 0 estándar (Maker 0.02%, Taker 0.05%, Funding 8h).
    pub fn vip_zero() -> Self {
        Self::default()
    }

    /// Activa el descuento oficial del 10% de Binance por tenencia/pago con BNB.
    pub fn with_bnb_discount(mut self) -> Self {
        self.fee_schedule.bnb_discount_enabled = true;
        self
    }

    /// Asigna un nivel VIP específico de Binance.
    pub fn with_vip_tier(mut self, tier: BinanceVipTier) -> Self {
        self.fee_schedule.vip_tier = tier;
        self
    }

    /// Determina el tipo de orden (`Maker` o `Taker`) a partir del enum tipado `ExitReason`.
    /// Implementa la conexión tipada C-20:
    /// - `TakeProfit` -> `self.tp_order_type` (Maker por defecto)
    /// - `StopLoss` / `Liquidation` -> `self.sl_order_type` (Taker por defecto)
    /// - `Signal` / `EndOfData` / `Other` -> `self.time_exit_order_type` (Taker por defecto)
    #[inline]
    pub fn execution_type_for_exit_reason_typed(&self, reason: ExitReason) -> OrderExecutionType {
        reason.execution_type_with_rules(
            self.tp_order_type,
            self.sl_order_type,
            self.time_exit_order_type,
        )
    }

    /// Determina causalmente el tipo de orden (`Maker` o `Taker`) en función del motivo de salida del trade,
    /// empleando el parser léxico tipado `ExitReason::from_reason_str_lenient`.
    /// Elimina completamente la fragilidad de subcadenas `"tp"` (C-20 / S4-H04).
    #[inline]
    pub fn execution_type_for_exit_reason(&self, reason: &str) -> OrderExecutionType {
        let typed_reason = ExitReason::from_reason_str_lenient(reason);
        self.execution_type_for_exit_reason_typed(typed_reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microstructure_config_defaults() {
        let config = MicrostructureConfig::default();
        assert_eq!(config.fee_schedule.vip_tier, BinanceVipTier::Vip0);
        assert!(!config.fee_schedule.bnb_discount_enabled);
        assert!(config.funding_config.enabled);
        assert_eq!(config.entry_order_type, OrderExecutionType::Taker);
        assert_eq!(config.tp_order_type, OrderExecutionType::Maker);
        assert_eq!(config.sl_order_type, OrderExecutionType::Taker);
        assert_eq!(config.time_exit_order_type, OrderExecutionType::Taker);
    }

    #[test]
    fn test_execution_type_for_exit_reason() {
        let config = MicrostructureConfig::default();

        assert_eq!(
            config.execution_type_for_exit_reason("Intra-candle Take Profit Hit"),
            OrderExecutionType::Maker
        );
        assert_eq!(
            config.execution_type_for_exit_reason("Intra-candle Stop Loss Triggered"),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason("Liquidation Triggered"),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason("Signal Reversal / Exit"),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason("End of Data Horizon"),
            OrderExecutionType::Taker
        );

        // Inmunidad a subcadenas falsas positivas de "tp" (C-20)
        assert_eq!(
            config.execution_type_for_exit_reason("Endpoint Reached"),
            OrderExecutionType::Taker,
            "Endpoint Reached no debe clasificarse como Take Profit Maker"
        );
        assert_eq!(
            config.execution_type_for_exit_reason("FastPath Exit"),
            OrderExecutionType::Taker,
            "FastPath Exit no debe clasificarse como Take Profit Maker"
        );
    }

    #[test]
    fn test_execution_type_for_exit_reason_typed() {
        let config = MicrostructureConfig::default();
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::TakeProfit),
            OrderExecutionType::Maker
        );
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::StopLoss),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::Liquidation),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::Signal),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::EndOfData),
            OrderExecutionType::Taker
        );
        assert_eq!(
            config.execution_type_for_exit_reason_typed(ExitReason::Other),
            OrderExecutionType::Taker
        );
    }

    #[test]
    fn test_fluent_builder_with_bnb() {
        let config = MicrostructureConfig::vip_zero()
            .with_vip_tier(BinanceVipTier::Vip3)
            .with_bnb_discount();

        assert_eq!(config.fee_schedule.vip_tier, BinanceVipTier::Vip3);
        assert!(config.fee_schedule.bnb_discount_enabled);
    }

    #[test]
    fn test_legacy_json_deserialization_ignores_slippage_field() {
        let legacy_json = r#"{
            "fee_schedule": {
                "vip_tier": "Vip0",
                "bnb_discount_enabled": false,
                "custom_maker_fee": null,
                "custom_taker_fee": null
            },
            "funding_config": {
                "enabled": true,
                "base_funding_rate": 0.0001,
                "clamp_min": -0.0075,
                "clamp_max": 0.0075,
                "interval_hours": 8,
                "timestamp_unit": "Milliseconds"
            },
            "entry_order_type": "Taker",
            "tp_order_type": "Maker",
            "sl_order_type": "Taker",
            "time_exit_order_type": "Taker",
            "slippage_rate": 0.0002
        }"#;

        let parsed: Result<MicrostructureConfig, _> = serde_json::from_str(legacy_json);
        assert!(parsed.is_ok(), "Debe deserializar con éxito ignorando slippage_rate");
        let cfg = parsed.unwrap();
        assert_eq!(cfg.entry_order_type, OrderExecutionType::Taker);
    }
}

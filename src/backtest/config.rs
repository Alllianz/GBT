use serde::{Deserialize, Serialize};
use crate::microstructure::{MicrostructureConfig, OrderExecutionType};
use crate::types::{FitnessMetric, MarginMode, PositionSizing, Timeframe};

/// Configuración integral para el motor de backtest institucional.
///
/// DIRECTRIZ INSTITUCIONAL ESTRICTA DE ALLIANZ:
/// El proyecto GBT opera sin slippage. El parámetro `slippage_rate` ha sido
/// completamente eliminado de la arquitectura. En caso de deserializar configuraciones
/// legadas en JSON que aún incluyan `slippage_rate`, el campo es ignorado automáticamente por serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub leverage: f64,
    pub margin_mode: MarginMode,
    pub position_sizing: PositionSizing,
    pub commission_rate: f64, // Ej: 0.0005 para 0.05% (tasa plana de fallback cuando microstructure es None)
    pub fixed_stop_loss_pct: Option<f64>,
    pub fixed_take_profit_pct: Option<f64>,
    pub allow_short: bool,
    pub fitness_metric: FitnessMetric,
    pub microstructure: Option<MicrostructureConfig>,
    /// Timeframe explícito y tipado del backtest para dimensionamiento y anualización de volatilidad (C-21)
    #[serde(default)]
    pub timeframe: Option<Timeframe>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            leverage: 1.0,
            margin_mode: MarginMode::default(),
            position_sizing: PositionSizing::PercentOfEquity(0.25),
            commission_rate: 0.0005, // 0.05%
            fixed_stop_loss_pct: None,
            fixed_take_profit_pct: None,
            allow_short: true,
            fitness_metric: FitnessMetric::SharpeRatio,
            microstructure: Some(MicrostructureConfig::default()),
            timeframe: Some(Timeframe::H1),
        }
    }
}

impl BacktestConfig {
    pub fn new(initial_capital: f64, leverage: f64) -> Self {
        Self {
            initial_capital,
            leverage,
            ..Default::default()
        }
    }

    /// Retorna la configuración activa de microestructura o una instancia por defecto compatible.
    /// Fuente única de verdad: `MicrostructureConfig` y su `BinanceFeeSchedule` (C-19 / S4-H03).
    /// Si el usuario configuró una tasa de comisión personalizada en `commission_rate`,
    /// se propaga causalmente a `fee_schedule` respetando la asimetría Maker/Taker
    /// y el descuento BNB sin aplanar las tarifas.
    #[inline]
    pub fn effective_microstructure(&self) -> MicrostructureConfig {
        let mut ms = self.microstructure.unwrap_or_default();
        if (self.commission_rate - 0.0005).abs() > 1e-9 {
            if ms.fee_schedule.custom_taker_fee.is_none() {
                if self.commission_rate == 0.0 {
                    ms.fee_schedule.custom_taker_fee = Some(0.0);
                    ms.fee_schedule.custom_maker_fee = Some(0.0);
                } else {
                    ms.fee_schedule.custom_taker_fee = Some(self.commission_rate);
                    if ms.fee_schedule.custom_maker_fee.is_none() {
                        let maker_ratio = ms.fee_schedule.vip_tier.maker_fee_rate()
                            / ms.fee_schedule.vip_tier.taker_fee_rate();
                        ms.fee_schedule.custom_maker_fee = Some(self.commission_rate * maker_ratio);
                    }
                }
            }
        }
        ms
    }

    /// Obtiene la tasa de comisión aplicable según el tipo de orden (Maker / Taker).
    /// Fuente única de verdad: `FeeSchedule` de la microestructura (C-19 / S4-H03).
    /// Garantiza que el nivel VIP y el descuento BNB no sean jamás anulados por una tasa plana.
    #[inline]
    pub fn get_commission_rate(&self, exec_type: OrderExecutionType) -> f64 {
        self.effective_microstructure()
            .fee_schedule
            .fee_rate_for(exec_type)
    }

    /// Asigna de forma explícita y tipada el timeframe para el backtest (C-21).
    pub fn with_timeframe(mut self, timeframe: Timeframe) -> Self {
        self.timeframe = Some(timeframe);
        self
    }

    /// Períodos calculados por año según el timeframe configurado (C-21).
    pub fn periods_per_year(&self) -> f64 {
        self.timeframe
            .map(|tf| tf.periods_per_year())
            .unwrap_or(8760.0)
    }

    /// Factor de anualización de la volatilidad para este backtest (C-21).
    pub fn annualization_factor(&self) -> f64 {
        self.periods_per_year().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microstructure::BinanceVipTier;

    #[test]
    fn test_backtest_config_default_commission() {
        let config = BacktestConfig::default();
        // Por defecto en VIP0: Maker = 0.02%, Taker = 0.05%
        assert!((config.get_commission_rate(OrderExecutionType::Maker) - 0.00020).abs() < 1e-8);
        assert!((config.get_commission_rate(OrderExecutionType::Taker) - 0.00050).abs() < 1e-8);
    }

    #[test]
    fn test_c19_commission_single_source_of_truth_with_bnb_discount() {
        let mut config = BacktestConfig::default();
        // Si el usuario configura commission_rate = 0.0004 con descuento BNB,
        // la tasa taker pasa a ser 0.0004 * 0.90 = 0.00036 y maker 0.00016 * 0.90 = 0.000144:
        config.commission_rate = 0.0004;
        let mut ms = MicrostructureConfig::default();
        ms.fee_schedule.bnb_discount_enabled = true;
        config.microstructure = Some(ms);

        let maker_rate = config.get_commission_rate(OrderExecutionType::Maker);
        let taker_rate = config.get_commission_rate(OrderExecutionType::Taker);

        assert!(
            (maker_rate - 0.000144).abs() < 1e-8,
            "Maker rate con descuento BNB debe ser 0.000144, obtenido: {}",
            maker_rate
        );
        assert!(
            (taker_rate - 0.000360).abs() < 1e-8,
            "Taker rate con descuento BNB debe ser 0.000360, obtenido: {}",
            taker_rate
        );
    }

    #[test]
    fn test_c19_commission_with_vip_tier() {
        let mut config = BacktestConfig::default();
        let mut ms = MicrostructureConfig::default();
        ms.fee_schedule.vip_tier = BinanceVipTier::Vip3; // VIP3: Maker = 0.012% (0.00012), Taker = 0.032% (0.00032)
        config.microstructure = Some(ms);

        assert!((config.get_commission_rate(OrderExecutionType::Maker) - 0.00012).abs() < 1e-8);
        assert!((config.get_commission_rate(OrderExecutionType::Taker) - 0.00032).abs() < 1e-8);
    }

    #[test]
    fn test_deserialization_ignores_legacy_slippage_field() {
        let legacy_json = r#"{
            "initial_capital": 10000.0,
            "leverage": 1.0,
            "margin_mode": "Isolated",
            "position_sizing": { "PercentOfEquity": 0.25 },
            "commission_rate": 0.0005,
            "slippage_rate": 0.0002,
            "fixed_stop_loss_pct": null,
            "fixed_take_profit_pct": null,
            "allow_short": true,
            "fitness_metric": "SharpeRatio",
            "microstructure": null
        }"#;

        let parsed: Result<BacktestConfig, _> = serde_json::from_str(legacy_json);
        assert!(parsed.is_ok(), "Debe deserializar con éxito ignorando el campo legacy slippage_rate");
        let cfg = parsed.unwrap();
        assert_eq!(cfg.initial_capital, 10000.0);
    }
}

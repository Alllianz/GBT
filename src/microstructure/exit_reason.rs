use serde::{Deserialize, Serialize};
use std::str::FromStr;
use super::order_execution_type::OrderExecutionType;

/// Motivo causal y tipado del cierre de una posición en el motor de trading.
///
/// Resuelve la fragilidad técnica de la clasificación basada en subcadenas ambiguas (C-20 / S4-H04)
/// y proporciona una clasificación determinista e institucional de las órdenes de salida
/// hacia sus respectivas modalidades de ejecución en el libro (`Maker` o `Taker`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum ExitReason {
    /// Cierre por toma de beneficios (orden límite pasiva en el libro de órdenes).
    #[default]
    TakeProfit,
    /// Cierre por límite de pérdidas (orden de protección agresiva al mercado).
    StopLoss,
    /// Cierre forzoso por insolvencia de margen / llamada de margen (Margin Call).
    Liquidation,
    /// Cierre activado por reversión de señal de la estrategia algorítmica.
    Signal,
    /// Cierre forzoso por finalización de la serie temporal o ventana de backtest.
    EndOfData,
    /// Cualquier otro motivo no catalogado (se asume ejecución agresiva conservadora).
    Other,
}

impl ExitReason {
    /// Retorna la modalidad de ejecución predeterminada oficial de Binance Futures USD-M:
    /// - `TakeProfit`: Maker (0.02% en VIP 0).
    /// - `StopLoss`, `Liquidation`, `Signal`, `EndOfData`, `Other`: Taker (0.05% en VIP 0).
    #[inline]
    pub fn default_execution_type(&self) -> OrderExecutionType {
        match self {
            Self::TakeProfit => OrderExecutionType::Maker,
            Self::StopLoss
            | Self::Liquidation
            | Self::Signal
            | Self::EndOfData
            | Self::Other => OrderExecutionType::Taker,
        }
    }

    /// Mapea el motivo de salida al tipo de ejecución aplicando las reglas de configuración personalizadas:
    /// - `TakeProfit` delega en `tp_order_type`.
    /// - `StopLoss` y `Liquidation` delegan en `sl_order_type`.
    /// - `Signal`, `EndOfData` y `Other` delegan en `time_exit_order_type`.
    #[inline]
    pub fn execution_type_with_rules(
        &self,
        tp_order_type: OrderExecutionType,
        sl_order_type: OrderExecutionType,
        time_exit_order_type: OrderExecutionType,
    ) -> OrderExecutionType {
        match self {
            Self::TakeProfit => tp_order_type,
            Self::StopLoss | Self::Liquidation => sl_order_type,
            Self::Signal | Self::EndOfData | Self::Other => time_exit_order_type,
        }
    }

    /// Retorna la representación canónica en texto estático.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TakeProfit => "Take Profit",
            Self::StopLoss => "Stop Loss",
            Self::Liquidation => "Liquidation",
            Self::Signal => "Signal",
            Self::EndOfData => "End of Data",
            Self::Other => "Other",
        }
    }

    /// Retorna `true` si la salida corresponde a una toma de ganancias.
    #[inline]
    pub fn is_take_profit(&self) -> bool {
        matches!(self, Self::TakeProfit)
    }

    /// Retorna `true` si la salida corresponde a un stop loss.
    #[inline]
    pub fn is_stop_loss(&self) -> bool {
        matches!(self, Self::StopLoss)
    }

    /// Retorna `true` si la salida fue por liquidación forzosa.
    #[inline]
    pub fn is_liquidation(&self) -> bool {
        matches!(self, Self::Liquidation)
    }

    /// Retorna `true` si la salida fue detonada por señal de la estrategia.
    #[inline]
    pub fn is_signal(&self) -> bool {
        matches!(self, Self::Signal)
    }

    /// Retorna `true` si la salida ocurrió por final del horizonte de datos.
    #[inline]
    pub fn is_end_of_data(&self) -> bool {
        matches!(self, Self::EndOfData)
    }

    /// Parser flexible y seguro ante falsos positivos de subcadenas (C-20 / S4-H04).
    ///
    /// No busca subcadenas crudas como `"tp"`, sino frases completas o tokens aislados
    /// con delimitadores léxicos (espacios, paréntesis, guiones, corchetes), evitando clasificar
    /// erróneamente cadenas como `"Endpoint Reached"`, `"FastPath Exit"` o `"Footprint Stop"`
    /// como `TakeProfit`.
    pub fn from_reason_str_lenient(reason: &str) -> Self {
        let lower = reason.to_lowercase();

        // 1. Detección de Take Profit
        if lower.contains("take profit")
            || lower.contains("take_profit")
            || lower.contains("takeprofit")
            || contains_isolated_token(&lower, "tp")
        {
            return Self::TakeProfit;
        }

        // 2. Detección de Stop Loss
        if lower.contains("stop loss")
            || lower.contains("stop_loss")
            || lower.contains("stoploss")
            || contains_isolated_token(&lower, "sl")
        {
            return Self::StopLoss;
        }

        // 3. Detección de Liquidación
        if lower.contains("liquidat")
            || lower.contains("margin call")
            || lower.contains("bankrupt")
            || lower.contains("insolvent")
        {
            return Self::Liquidation;
        }

        // 4. Detección de Señal / Reversión
        if lower.contains("signal") || lower.contains("reversal") {
            return Self::Signal;
        }

        // 5. Detección de Fin de Datos / Horizonte
        if lower.contains("end of data")
            || lower.contains("end_of_data")
            || lower.contains("horizon")
            || lower.contains("time exit")
            || lower.contains("timeout")
        {
            return Self::EndOfData;
        }

        Self::Other
    }
}

/// Comprueba si el texto contiene el token `token` aislado por límites no alfanuméricos
/// (ej. inicio/fin de cadena, espacios, puntuación como `()`, `[]`, `_`, `-`, `/`).
fn contains_isolated_token(text: &str, token: &str) -> bool {
    let token_len = token.len();
    if text.len() < token_len {
        return false;
    }

    for (idx, _) in text.match_indices(token) {
        let before_ok = if idx == 0 {
            true
        } else {
            let prev_char = text[..idx].chars().next_back().unwrap();
            !prev_char.is_alphanumeric()
        };

        let after_idx = idx + token_len;
        let after_ok = if after_idx >= text.len() {
            true
        } else {
            let next_char = text[after_idx..].chars().next().unwrap();
            !next_char.is_alphanumeric()
        };

        if before_ok && after_ok {
            return true;
        }
    }

    false
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ExitReason {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim().to_lowercase();
        match trimmed.as_str() {
            "takeprofit" | "take_profit" | "take profit" | "tp" => Ok(Self::TakeProfit),
            "stoploss" | "stop_loss" | "stop loss" | "sl" => Ok(Self::StopLoss),
            "liquidation" | "liquidated" | "margin_call" | "margin call" | "bankrupt" | "insolvent" => Ok(Self::Liquidation),
            "signal" | "signal_reversal" | "signal reversal" | "reversal" => Ok(Self::Signal),
            "endofdata" | "end_of_data" | "end of data" | "horizon" | "time_exit" | "time exit" => {
                Ok(Self::EndOfData)
            }
            "other" => Ok(Self::Other),
            _ => {
                let parsed = Self::from_reason_str_lenient(s);
                if parsed != Self::Other || trimmed == "other" {
                    Ok(parsed)
                } else {
                    Err(format!("Motivo de salida desconocido: '{}'", s))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_execution_type_mapping() {
        assert_eq!(
            ExitReason::TakeProfit.default_execution_type(),
            OrderExecutionType::Maker
        );
        assert_eq!(
            ExitReason::StopLoss.default_execution_type(),
            OrderExecutionType::Taker
        );
        assert_eq!(
            ExitReason::Liquidation.default_execution_type(),
            OrderExecutionType::Taker
        );
        assert_eq!(
            ExitReason::Signal.default_execution_type(),
            OrderExecutionType::Taker
        );
        assert_eq!(
            ExitReason::EndOfData.default_execution_type(),
            OrderExecutionType::Taker
        );
        assert_eq!(
            ExitReason::Other.default_execution_type(),
            OrderExecutionType::Taker
        );
    }

    #[test]
    fn test_custom_rules_execution_type_mapping() {
        // En una configuración invertida o personalizada:
        let tp_type = OrderExecutionType::Taker;
        let sl_type = OrderExecutionType::Maker;
        let time_type = OrderExecutionType::Taker;

        assert_eq!(
            ExitReason::TakeProfit.execution_type_with_rules(tp_type, sl_type, time_type),
            OrderExecutionType::Taker
        );
        assert_eq!(
            ExitReason::StopLoss.execution_type_with_rules(tp_type, sl_type, time_type),
            OrderExecutionType::Maker
        );
        assert_eq!(
            ExitReason::Liquidation.execution_type_with_rules(tp_type, sl_type, time_type),
            OrderExecutionType::Maker
        );
        assert_eq!(
            ExitReason::Signal.execution_type_with_rules(tp_type, sl_type, time_type),
            OrderExecutionType::Taker
        );
    }

    #[test]
    fn test_c20_anti_false_positive_substring_immunity() {
        // Casos citados explícitamente en el reporte de auditoría S4-H04 / C-20:
        // Cadenas que contienen "tp" embebido en palabras técnicas no deben clasificarse como TakeProfit.
        let false_tp_cases = [
            "FastPath Exit",
            "Endpoint Reached",
            "Output Barrier",
            "Footprint Stop",
            "PostPort Timeout",
            "Http Client Error",
        ];

        for case in false_tp_cases {
            let reason = ExitReason::from_reason_str_lenient(case);
            assert_ne!(
                reason,
                ExitReason::TakeProfit,
                "Falso positivo detectado para '{}': clasificado erróneamente como TakeProfit",
                case
            );
            assert_eq!(
                reason.default_execution_type(),
                OrderExecutionType::Taker,
                "El caso '{}' debió mapearse a Taker",
                case
            );
        }
    }

    #[test]
    fn test_lenient_parsing_legitimate_take_profit() {
        let valid_tp_cases = [
            "Take Profit",
            "take profit",
            "Intra-candle Take Profit Hit",
            "Hit TP target",
            "Exit on TP",
            "Exit (TP)",
            "[TP] Execution",
            "TP",
        ];

        for case in valid_tp_cases {
            let reason = ExitReason::from_reason_str_lenient(case);
            assert_eq!(
                reason,
                ExitReason::TakeProfit,
                "Fallo al reconocer caso legítimo de TakeProfit: '{}'",
                case
            );
            assert_eq!(reason.default_execution_type(), OrderExecutionType::Maker);
        }
    }

    #[test]
    fn test_lenient_parsing_legitimate_stop_loss() {
        let valid_sl_cases = [
            "Stop Loss",
            "stop loss",
            "Intra-candle Stop Loss Triggered",
            "Stop Loss (Ambiguous High/Low)",
            "Hit SL level",
            "Exit on SL",
            "Exit [SL]",
            "SL",
        ];

        for case in valid_sl_cases {
            let reason = ExitReason::from_reason_str_lenient(case);
            assert_eq!(
                reason,
                ExitReason::StopLoss,
                "Fallo al reconocer caso legítimo de StopLoss: '{}'",
                case
            );
            assert_eq!(reason.default_execution_type(), OrderExecutionType::Taker);
        }
    }

    #[test]
    fn test_lenient_parsing_other_standard_reasons() {
        assert_eq!(
            ExitReason::from_reason_str_lenient("Liquidated (Margin Call)"),
            ExitReason::Liquidation
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("bankrupt"),
            ExitReason::Liquidation
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("insolvent"),
            ExitReason::Liquidation
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("Account bankrupt on margin spike"),
            ExitReason::Liquidation
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("Trader insolvent"),
            ExitReason::Liquidation
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("Signal Reversal / Exit"),
            ExitReason::Signal
        );
        assert_eq!(
            ExitReason::from_reason_str_lenient("End of Data Horizon"),
            ExitReason::EndOfData
        );
    }

    #[test]
    fn test_from_str_and_display() {
        assert_eq!(ExitReason::from_str("Take Profit").unwrap(), ExitReason::TakeProfit);
        assert_eq!(ExitReason::from_str("tp").unwrap(), ExitReason::TakeProfit);
        assert_eq!(ExitReason::from_str("sl").unwrap(), ExitReason::StopLoss);
        assert_eq!(ExitReason::from_str("liquidation").unwrap(), ExitReason::Liquidation);
        assert_eq!(ExitReason::from_str("bankrupt").unwrap(), ExitReason::Liquidation);
        assert_eq!(ExitReason::from_str("insolvent").unwrap(), ExitReason::Liquidation);
        assert_eq!(ExitReason::from_str("signal").unwrap(), ExitReason::Signal);
        assert_eq!(ExitReason::from_str("horizon").unwrap(), ExitReason::EndOfData);

        assert_eq!(format!("{}", ExitReason::TakeProfit), "Take Profit");
        assert_eq!(format!("{}", ExitReason::StopLoss), "Stop Loss");
    }
}

use crate::types::Kline;

/// Representa una muestra individual etiquetada para el meta-modelo secundario
#[derive(Debug, Clone, PartialEq)]
pub struct MetaSample {
    /// Índice temporal original $t$ de la muestra en la serie
    pub index: usize,
    /// Predicción continua generada por el modelo primario $\hat{y}_{\text{primary}, t}$
    pub primary_prediction: f64,
    /// Dirección propuesta por el modelo primario ($+1.0$ Long, $-1.0$ Short)
    pub primary_direction: f64,
    /// Retorno neto realizado de la posición tras deducir comisiones roundtrip (sin slippage)
    pub realized_net_return: f64,
    /// Si la operación fue estrictamente rentable tras comisiones
    pub is_winner: bool,
    /// Etiqueta objetivo binaria para el meta-modelo ($1.0$ si fue rentable, $0.0$ si fue perdedora)
    pub meta_label: f64,
}

/// Constructor de Etiquetas Causal para Meta-Labeling (Marcos López de Prado)
///
/// Convierte las predicciones Out-of-Fold del modelo primario y los retornos netos realizados
/// en un problema binario causal para el Gatekeeper secundario:
/// - $y_{\text{meta}, t} = 1.0$ si la señal primaria ($d_t \neq 0$) generó un retorno neto positivo ($r_{\text{net}} > 0$).
/// - $y_{\text{meta}, t} = 0.0$ si la señal primaria fue un falso positivo ($r_{\text{net}} \le 0$).
///
/// REGLA FINANCIERA: Las muestras donde el modelo primario no operó ($d_t = 0$) se excluyen
/// del entrenamiento del meta-modelo, ya que el Gatekeeper solo actúa como filtro sobre decisiones tomadas.
pub struct MetaTargetBuilder;

impl MetaTargetBuilder {
    /// Construye muestras de meta-etiquetado directamente a partir de las velas y el horizonte causal.
    ///
    /// - Precio de entrada: $P_{\text{entry}} = \text{klines}[t + 1].\text{open}$ (ejecución causal a la siguiente vela).
    /// - Precio de salida: $P_{\text{exit}} = \text{klines}[t + \text{horizon}].\text{close}$.
    /// - Costo de comisión roundtrip: $2 \times \text{commission\_rate}$ (sin slippage).
    pub fn build_from_klines(
        primary_predictions: &[f64],
        klines: &[Kline],
        horizon: usize,
        commission_rate: f64,
        long_threshold: f64,
        short_threshold: f64,
    ) -> Vec<MetaSample> {
        let n = primary_predictions.len().min(klines.len());
        if n <= horizon || horizon == 0 {
            return Vec::new();
        }

        let valid_n = n.saturating_sub(horizon);
        let roundtrip_fee = commission_rate * 2.0;
        let mut samples = Vec::new();

        for i in 0..valid_n {
            let pred = primary_predictions[i];
            let direction: f64 = if pred > long_threshold {
                1.0
            } else if pred < short_threshold {
                -1.0
            } else {
                0.0
            };

            // Solo procesamos instantes donde el modelo primario activó una señal
            if direction.abs() < 1e-6 {
                continue;
            }

            let p_entry = klines[i + 1].open;
            let p_exit = klines[i + horizon].close;

            if p_entry <= 0.0 || p_exit <= 0.0 {
                continue;
            }

            let realized_net_return = if direction > 0.0 {
                // Operación en Long: beneficio si p_exit > p_entry
                ((p_exit - p_entry) / p_entry) - roundtrip_fee
            } else {
                // Operación en Short: beneficio si p_entry > p_exit
                ((p_entry - p_exit) / p_entry) - roundtrip_fee
            };

            let is_winner = realized_net_return > 0.0;
            let meta_label = if is_winner { 1.0 } else { 0.0 };

            samples.push(MetaSample {
                index: i,
                primary_prediction: pred,
                primary_direction: direction,
                realized_net_return,
                is_winner,
                meta_label,
            });
        }

        samples
    }

    /// Construye muestras de meta-etiquetado a partir de vectores ya calculados de predicción y retornos Long netos.
    ///
    /// `long_net_returns` debe contener $((P_{\text{exit}} - P_{\text{entry}}) / P_{\text{entry}}) - \text{roundtrip\_fee}$.
    pub fn build_from_returns(
        primary_predictions: &[f64],
        long_net_returns: &[f64],
        long_threshold: f64,
        short_threshold: f64,
        roundtrip_fee: f64,
    ) -> Vec<MetaSample> {
        let n = primary_predictions.len().min(long_net_returns.len());
        let mut samples = Vec::new();

        for i in 0..n {
            let pred = primary_predictions[i];
            let direction: f64 = if pred > long_threshold {
                1.0
            } else if pred < short_threshold {
                -1.0
            } else {
                0.0
            };

            if direction.abs() < 1e-6 {
                continue;
            }

            let long_ret = long_net_returns[i];
            // Reconstruir el retorno de precio bruto y computar el neto de la posición Short
            let realized_net_return = if direction > 0.0 {
                long_ret
            } else {
                let gross_price_return = long_ret + roundtrip_fee;
                (-gross_price_return) - roundtrip_fee
            };

            let is_winner = realized_net_return > 0.0;
            let meta_label = if is_winner { 1.0 } else { 0.0 };

            samples.push(MetaSample {
                index: i,
                primary_prediction: pred,
                primary_direction: direction,
                realized_net_return,
                is_winner,
                meta_label,
            });
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_target_builder_long_and_short() {
        let preds = vec![0.005, -0.006, 0.001, 0.008, -0.004];
        let long_th = 0.003;
        let short_th = -0.003;
        let roundtrip_fee = 0.0008;

        // Caso 0: Long con ret bruto +0.02 -> neto +0.0192 (Winner)
        // Caso 1: Short con ret precio bruto -0.015 -> ret short neto +0.0142 (Winner)
        // Caso 2: Neutral (pred 0.001 entre -0.003 y 0.003) -> Ignorado
        // Caso 3: Long con ret bruto -0.01 -> neto -0.0108 (Loser)
        // Caso 4: Short con ret precio bruto +0.01 -> ret short neto -0.0108 (Loser)
        let long_net_returns = vec![
            0.0200 - roundtrip_fee,
            -0.0150 - roundtrip_fee,
            0.0050,
            -0.0100 - roundtrip_fee,
            0.0100 - roundtrip_fee,
        ];

        let samples = MetaTargetBuilder::build_from_returns(
            &preds,
            &long_net_returns,
            long_th,
            short_th,
            roundtrip_fee,
        );

        assert_eq!(samples.len(), 4, "Debe omitir la muestra neutral");

        // Muestra 0: Long ganador
        assert_eq!(samples[0].index, 0);
        assert_eq!(samples[0].primary_direction, 1.0);
        assert!(samples[0].realized_net_return > 0.0);
        assert_eq!(samples[0].meta_label, 1.0);

        // Muestra 1: Short ganador (precio cayó)
        assert_eq!(samples[1].index, 1);
        assert_eq!(samples[1].primary_direction, -1.0);
        assert!(samples[1].realized_net_return > 0.0);
        assert_eq!(samples[1].meta_label, 1.0);

        // Muestra 2 (índice 3 original): Long perdedor
        assert_eq!(samples[2].index, 3);
        assert_eq!(samples[2].primary_direction, 1.0);
        assert!(samples[2].realized_net_return < 0.0);
        assert_eq!(samples[2].meta_label, 0.0);

        // Muestra 3 (índice 4 original): Short perdedor (precio subió)
        assert_eq!(samples[3].index, 4);
        assert_eq!(samples[3].primary_direction, -1.0);
        assert!(samples[3].realized_net_return < 0.0);
        assert_eq!(samples[3].meta_label, 0.0);
    }
}

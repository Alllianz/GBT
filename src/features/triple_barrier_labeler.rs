use rayon::prelude::*;
use crate::types::Kline;
use super::technical_indicators::TechnicalIndicators;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetLabel {
    Short = 0,
    Long = 1,
    Neutral = 2,
}

pub struct TripleBarrierLabeler;

impl TripleBarrierLabeler {
    /// Genera etiquetas discretas (0: Short, 1: Long, 2: Neutral) usando Triple Barrier Method en paralelo.
    /// BUG-F01: Deduce comisiones netas roundtrip (SIN SLIPPAGE por orden de Allianz).
    pub fn label_classification(
        klines: &[Kline],
        horizon: usize,
        multiplier: f64,
        commission_rate: f64,
    ) -> Vec<TargetLabel> {
        let n = klines.len();
        if n <= horizon || horizon == 0 {
            return vec![TargetLabel::Neutral; n];
        }

        let roundtrip_fee = commission_rate * 2.0;
        let atrs = TechnicalIndicators::atr(klines, 14);
        // BUG-40: Alinear el horizonte de forma causal y consistente con la entrada al Open
        let valid_n = n.saturating_sub(horizon);

        let mut labels: Vec<TargetLabel> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let entry_price = klines[i + 1].open;
                if entry_price <= 0.0 {
                    return TargetLabel::Neutral;
                }
                let atr = atrs[i].max(entry_price * 0.002);
                let upper_barrier = entry_price + multiplier * atr;
                let lower_barrier = entry_price - multiplier * atr;

                for step in 0..horizon {
                    let candle = &klines[i + 1 + step];
                    let hit_upper = candle.high >= upper_barrier;
                    let hit_lower = candle.low <= lower_barrier;

                    if hit_upper && hit_lower {
                        return TargetLabel::Neutral;
                    } else if hit_upper {
                        let gross_return = (upper_barrier - entry_price) / entry_price;
                        return if gross_return - roundtrip_fee > 0.0 {
                            TargetLabel::Long
                        } else {
                            TargetLabel::Neutral
                        };
                    } else if hit_lower {
                        let gross_return = (entry_price - lower_barrier) / entry_price;
                        return if gross_return - roundtrip_fee > 0.0 {
                            TargetLabel::Short
                        } else {
                            TargetLabel::Neutral
                        };
                    }
                }

                // Si no toca ninguna barrera, evaluar retorno causal neto tras comisiones al cierre de la vela (i + horizon)
                let p_exit = klines[i + horizon].close;
                let long_net = ((p_exit - entry_price) / entry_price) - roundtrip_fee;
                let short_net = ((entry_price - p_exit) / entry_price) - roundtrip_fee;

                if long_net > 0.0 && long_net > short_net {
                    TargetLabel::Long
                } else if short_net > 0.0 && short_net > long_net {
                    TargetLabel::Short
                } else {
                    TargetLabel::Neutral
                }
            })
            .collect();

        labels.resize(n, TargetLabel::Neutral);
        labels
    }

    /// Variante con barreras porcentuales fijas pt (profit taking) y sl (stop loss).
    /// BUG-03: Corrige el etiquetado asimétrico falso-positivo en posiciones Short.
    /// BUG-F01: Deduce comisiones netas roundtrip (SIN SLIPPAGE conforme a la orden de Allianz).
    pub fn label_classification_barriers(
        klines: &[Kline],
        horizon: usize,
        pt: f64,
        sl: f64,
        commission_rate: f64,
    ) -> Vec<TargetLabel> {
        let n = klines.len();
        if n <= horizon || horizon == 0 || pt <= 0.0 || sl <= 0.0 {
            return vec![TargetLabel::Neutral; n];
        }

        let roundtrip_fee = commission_rate * 2.0;
        let valid_n = n.saturating_sub(horizon);
        let mut labels: Vec<TargetLabel> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let entry_price = klines[i + 1].open;
                if entry_price <= 0.0 {
                    return TargetLabel::Neutral;
                }

                // Barreras de Take Profit y Stop Loss específicas por dirección
                let long_tp = entry_price * (1.0 + pt);
                let long_sl = entry_price * (1.0 - sl);
                let short_tp = entry_price * (1.0 - pt);
                let short_sl = entry_price * (1.0 + sl);

                let mut long_active = true;
                let mut short_active = true;

                for step in 0..horizon {
                    let candle = &klines[i + 1 + step];

                    // Evaluación causal de eventos para Long
                    let hit_long_tp = long_active && candle.high >= long_tp;
                    let hit_long_sl = long_active && candle.low <= long_sl;

                    // Evaluación causal de eventos para Short
                    let hit_short_tp = short_active && candle.low <= short_tp;
                    let hit_short_sl = short_active && candle.high >= short_sl;

                    // Detección de ambigüedad intra-vela (TP y SL tocados en la misma vela)
                    let long_ambiguous = hit_long_tp && hit_long_sl;
                    let short_ambiguous = hit_short_tp && hit_short_sl;

                    // Solo califica como ganador si tras comisiones netas el retorno es estrictamente positivo
                    let long_won = hit_long_tp && !hit_long_sl && (pt - roundtrip_fee > 0.0);
                    let short_won = hit_short_tp && !hit_short_sl && (pt - roundtrip_fee > 0.0);

                    if long_won && short_won {
                        return TargetLabel::Neutral;
                    } else if long_won {
                        return TargetLabel::Long;
                    } else if short_won {
                        return TargetLabel::Short;
                    }

                    // Actualizar estado si se tocó Stop Loss o hubo ambigüedad
                    if hit_long_sl || long_ambiguous {
                        long_active = false;
                    }
                    if hit_short_sl || short_ambiguous {
                        short_active = false;
                    }

                    // Si ambas hipótesis sufrieron Stop Loss, no hay oportunidad viable
                    if !long_active && !short_active {
                        return TargetLabel::Neutral;
                    }
                }

                // Barrera vertical causal (expiración temporal tras `horizon` velas)
                let p_exit = klines[i + horizon].close;
                let long_net = ((p_exit - entry_price) / entry_price) - roundtrip_fee;
                let short_net = ((entry_price - p_exit) / entry_price) - roundtrip_fee;

                if long_active && !short_active && long_net > 0.0 {
                    TargetLabel::Long
                } else if short_active && !long_active && short_net > 0.0 {
                    TargetLabel::Short
                } else if long_active && short_active {
                    if long_net > 0.0 && long_net > short_net {
                        TargetLabel::Long
                    } else if short_net > 0.0 && short_net > long_net {
                        TargetLabel::Short
                    } else {
                        TargetLabel::Neutral
                    }
                } else {
                    TargetLabel::Neutral
                }
            })
            .collect();

        labels.resize(n, TargetLabel::Neutral);
        labels
    }

    pub fn label_regression(klines: &[Kline], horizon: usize, commission_rate: f64) -> Vec<f64> {
        let n = klines.len();
        if n <= horizon || horizon == 0 {
            return vec![0.0; n];
        }

        let roundtrip_fee = commission_rate * 2.0;

        let valid_n = n.saturating_sub(horizon);
        let mut returns: Vec<f64> = (0..valid_n)
            .into_par_iter()
            .map(|i| {
                let p_entry = klines[i + 1].open;
                let p_exit = klines[i + horizon].close;
                if p_entry > 0.0 {
                    let gross_return = (p_exit - p_entry) / p_entry;
                    if gross_return > roundtrip_fee {
                        gross_return - roundtrip_fee
                    } else if gross_return < -roundtrip_fee {
                        gross_return + roundtrip_fee
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();

        returns.resize(n, 0.0);
        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kline(ts: i64, open: f64, high: f64, low: f64, close: f64) -> Kline {
        Kline {
            timestamp: ts,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_barrier_long_take_profit() {
        // Vela 0: base
        // Vela 1: entrada al open 100.0, sube a high 104.5 (TP 4%), low 99.5
        // Vela 2: vela de cierre
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 104.5, 99.5, 104.0),
            make_kline(220, 104.0, 105.0, 103.0, 104.0),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_eq!(labels[0], TargetLabel::Long);
    }

    #[test]
    fn test_barrier_short_take_profit() {
        // Vela 0: base
        // Vela 1: entrada al open 100.0, cae a low 95.5 (TP 4% en 96.0), high 100.5 (no toca SL 102.0)
        // Vela 2: vela de cierre
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 100.5, 95.5, 95.8),
            make_kline(220, 95.8, 96.0, 94.0, 95.0),
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_eq!(labels[0], TargetLabel::Short);
    }

    #[test]
    fn test_barrier_short_not_triggered_by_long_sl_distance() {
        // BUG-03: Antes, con pt=4% y sl=2%, si el precio caía un 2.5% (a 97.5),
        // se activaba erróneamente lower_barrier (entry * 0.98) y se etiquetaba como Short en el paso 0,
        // a pesar de que el TP de Short requería 4% (<= 96.0). Si luego el precio rebotaba
        // y cerraba en 100.5 (pérdida para Short), el modelo recibía un falso positivo.
        // Ahora, para ser Short por barrera horizontal debe descender al menos pt (4%, <= 96.0).
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 100.2, 97.5, 98.0), // cae 2.5%, toca SL de Long (98.0), pero NO TP de Short (96.0)
            make_kline(220, 98.0, 101.0, 98.0, 100.5), // rebota y cierra por encima de entry_price (100.5 > 100.0)
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        // Debe ser Neutral, NO Short
        assert_eq!(labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_barrier_short_stopped_out_before_drop() {
        // Si el precio primero sube a 102.5 (SL de Short en 102.0), el Short es detenido.
        // Aunque en la siguiente vela caiga a 94.0, ya no puede considerarse un trade exitoso.
        let klines = vec![
            make_kline(100, 100.0, 101.0, 99.0, 100.0),
            make_kline(160, 100.0, 102.5, 99.5, 102.0), // sube a 102.5 -> Toca SL de Short
            make_kline(220, 102.0, 102.0, 94.0, 94.5),  // cae fuerte después
        ];

        let labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.04, 0.02, 0.0005);
        assert_ne!(labels[0], TargetLabel::Short);
        assert_eq!(labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_bug_f01_fee_deduction_prevents_false_positive_gains() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.9, 100.02),
            make_kline(220, 100.02, 100.06, 99.98, 100.05),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        assert_eq!(reg_targets[0], 0.0);

        let class_labels = TripleBarrierLabeler::label_classification_barriers(&klines, 2, 0.02, 0.02, 0.0005);
        assert_eq!(class_labels[0], TargetLabel::Neutral);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_positive() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.5, 99.9, 100.1),
            make_kline(220, 100.1, 100.5, 100.0, 100.3),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        let expected = (100.3 - 100.0) / 100.0 - 0.0010;
        assert!((reg_targets[0] - expected).abs() < 1e-9);
        assert!(reg_targets[0] > 0.0);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_negative() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.5, 99.9),
            make_kline(220, 99.9, 100.0, 99.4, 99.6),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        let expected = (99.6 - 100.0) / 100.0 + 0.0010;
        assert!((reg_targets[0] - expected).abs() < 1e-9);
        assert!(reg_targets[0] < 0.0);
    }

    #[test]
    fn test_regression_symmetric_fee_deadzone_neutral() {
        let klines = vec![
            make_kline(100, 100.0, 100.1, 99.9, 100.0),
            make_kline(160, 100.0, 100.1, 99.9, 100.0),
            make_kline(220, 100.0, 100.05, 99.95, 99.95),
        ];

        let reg_targets = TripleBarrierLabeler::label_regression(&klines, 2, 0.0005);
        assert_eq!(reg_targets[0], 0.0);
    }
}

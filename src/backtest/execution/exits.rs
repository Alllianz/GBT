use crate::types::{Kline, Position, PositionSide};

pub struct ExitEvaluator;

impl ExitEvaluator {
    /// Evalúa si durante la vela actual se produce una salida por Liquidación (Margin Call),
    /// Stop Loss o Take Profit, modelando los Gaps de apertura.
    ///
    /// C-28: Sanitiza estrictamente `stop_loss` y `take_profit`: valores no finitos (`NaN`, `±Inf`)
    /// o `<= 0.0` se descartan de forma documentada tratándose como `None`, imposibilitando salidas espurias.
    pub fn evaluate_intra_candle(
        pos: &Position,
        current_candle: &Kline,
    ) -> Option<(f64, &'static str)> {
        if !current_candle.open.is_finite() || current_candle.open <= 0.0 {
            return None;
        }

        let valid_sl = pos.stop_loss.filter(|&sl| sl.is_finite() && sl > 0.0);
        let valid_tp = pos.take_profit.filter(|&tp| tp.is_finite() && tp > 0.0);

        match pos.side {
            PositionSide::Long => {
                let sl_before_liq = valid_sl.is_some_and(|sl| {
                    pos.liquidation_price.is_finite()
                        && pos.liquidation_price > 0.0
                        && current_candle.open > sl
                        && sl > pos.liquidation_price
                });

                if !sl_before_liq
                    && pos.liquidation_price.is_finite()
                    && pos.liquidation_price > 0.0
                    && current_candle.low.is_finite()
                    && current_candle.low > 0.0
                    && current_candle.low <= pos.liquidation_price
                {
                    let liq_target = pos.liquidation_price.min(current_candle.open).max(0.00000001);
                    return Some((liq_target, "Liquidated (Margin Call)"));
                }
            }
            PositionSide::Short => {
                let sl_before_liq = valid_sl.is_some_and(|sl| {
                    pos.liquidation_price.is_finite()
                        && pos.liquidation_price > 0.0
                        && current_candle.open < sl
                        && sl < pos.liquidation_price
                });

                if !sl_before_liq
                    && pos.liquidation_price.is_finite()
                    && pos.liquidation_price > 0.0
                    && current_candle.high.is_finite()
                    && current_candle.high > 0.0
                    && current_candle.high >= pos.liquidation_price
                {
                    let liq_target = pos.liquidation_price.max(current_candle.open);
                    return Some((liq_target, "Liquidated (Margin Call)"));
                }
            }
        }

        match pos.side {
            PositionSide::Long => {
                let hit_sl = valid_sl.is_some_and(|sl| current_candle.low.is_finite() && current_candle.low <= sl);
                let hit_tp = valid_tp.is_some_and(|tp| current_candle.high.is_finite() && current_candle.high >= tp);

                // Prevención de Gap Risk Look-Ahead: si la vela abre por debajo del Stop Loss, se ejecuta en Open
                let sl_exec = valid_sl.map(|sl| sl.min(current_candle.open));
                let tp_exec = valid_tp.map(|tp| tp.max(current_candle.open));

                if hit_sl && hit_tp {
                    Some((sl_exec.unwrap(), "Stop Loss (Ambiguous High/Low)"))
                } else if hit_sl {
                    Some((sl_exec.unwrap(), "Stop Loss"))
                } else if hit_tp {
                    Some((tp_exec.unwrap(), "Take Profit"))
                } else {
                    None
                }
            }
            PositionSide::Short => {
                let hit_sl = valid_sl.is_some_and(|sl| current_candle.high.is_finite() && current_candle.high >= sl);
                let hit_tp = valid_tp.is_some_and(|tp| current_candle.low.is_finite() && current_candle.low <= tp);

                // Prevención de Gap Risk Look-Ahead: si la vela abre por encima del Stop Loss, se ejecuta en Open
                let sl_exec = valid_sl.map(|sl| sl.max(current_candle.open));
                let tp_exec = valid_tp.map(|tp| tp.min(current_candle.open));

                if hit_sl && hit_tp {
                    Some((sl_exec.unwrap(), "Stop Loss (Ambiguous High/Low)"))
                } else if hit_sl {
                    Some((sl_exec.unwrap(), "Stop Loss"))
                } else if hit_tp {
                    Some((tp_exec.unwrap(), "Take Profit"))
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_position(
        side: PositionSide,
        entry: f64,
        sl: Option<f64>,
        tp: Option<f64>,
        liq: f64,
    ) -> Position {
        Position::new(
            side,
            entry,
            1.0,
            1000,
            0,
            sl,
            tp,
            None,
            liq,
            0.0,
        )
    }

    #[test]
    fn test_c28_nan_and_non_positive_sl_tp_are_ignored() {
        let pos_long = dummy_position(
            PositionSide::Long,
            100.0,
            Some(f64::NAN),
            Some(-50.0),
            0.0,
        );

        let candle = Kline {
            timestamp: 2000,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 10.0,
        };

        // Ningún SL ni TP espurio debe dispararse
        let result = ExitEvaluator::evaluate_intra_candle(&pos_long, &candle);
        assert!(result.is_none(), "SL con NaN y TP con <= 0 deben ignorarse por C-28");

        let pos_short = dummy_position(
            PositionSide::Short,
            100.0,
            Some(-10.0), // SL inválido (<= 0)
            Some(f64::INFINITY), // TP inválido
            0.0,
        );
        let result_short = ExitEvaluator::evaluate_intra_candle(&pos_short, &candle);
        assert!(result_short.is_none(), "SL <= 0 y TP infinito deben ignorarse por C-28");
    }

    #[test]
    fn test_c28_valid_sl_and_tp_behave_correctly() {
        let pos = dummy_position(
            PositionSide::Long,
            100.0,
            Some(95.0),
            Some(105.0),
            90.0,
        );

        // Vela que toca el TP a 106.0
        let candle_tp = Kline {
            timestamp: 2000,
            open: 100.0,
            high: 106.0,
            low: 98.0,
            close: 104.0,
            volume: 10.0,
        };
        let (exit_price, reason) = ExitEvaluator::evaluate_intra_candle(&pos, &candle_tp)
            .expect("Debe ejecutar TP");
        assert_eq!(exit_price, 105.0);
        assert_eq!(reason, "Take Profit");

        // Vela con Gap bajista por debajo del SL (abre en 92.0, SL es 95.0)
        let candle_gap_sl = Kline {
            timestamp: 3000,
            open: 92.0,
            high: 94.0,
            low: 91.0,
            close: 93.0,
            volume: 10.0,
        };
        let (exit_price_gap, reason_gap) = ExitEvaluator::evaluate_intra_candle(&pos, &candle_gap_sl)
            .expect("Debe ejecutar SL al Open por Gap");
        assert_eq!(exit_price_gap, 92.0);
        assert_eq!(reason_gap, "Stop Loss");
    }

    #[test]
    fn test_sl_before_liquidation_long_and_short() {
        let pos_long = dummy_position(
            PositionSide::Long,
            100.0,
            Some(95.0),
            Some(110.0),
            80.0,
        );
        let candle_drop = Kline {
            timestamp: 2000,
            open: 98.0,
            high: 99.0,
            low: 75.0,
            close: 76.0,
            volume: 10.0,
        };
        let (exit_price, reason) = ExitEvaluator::evaluate_intra_candle(&pos_long, &candle_drop)
            .expect("Debe ejecutar SL antes de Liquidacion");
        assert_eq!(exit_price, 95.0);
        assert_eq!(reason, "Stop Loss");

        let pos_short = dummy_position(
            PositionSide::Short,
            100.0,
            Some(105.0),
            Some(90.0),
            120.0,
        );
        let candle_rally = Kline {
            timestamp: 3000,
            open: 102.0,
            high: 125.0,
            low: 101.0,
            close: 124.0,
            volume: 10.0,
        };
        let (exit_price_s, reason_s) = ExitEvaluator::evaluate_intra_candle(&pos_short, &candle_rally)
            .expect("Debe ejecutar SL antes de Liquidacion");
        assert_eq!(exit_price_s, 105.0);
        assert_eq!(reason_s, "Stop Loss");
    }

    #[test]
    fn test_liquidation_priority_retained_edge_cases() {
        let pos_no_sl = dummy_position(
            PositionSide::Long,
            100.0,
            None,
            Some(110.0),
            80.0,
        );
        let candle_liq = Kline {
            timestamp: 2000,
            open: 95.0,
            high: 96.0,
            low: 75.0,
            close: 76.0,
            volume: 10.0,
        };
        let (p_no_sl, r_no_sl) = ExitEvaluator::evaluate_intra_candle(&pos_no_sl, &candle_liq)
            .expect("Debe liquidar sin SL");
        assert_eq!(p_no_sl, 80.0);
        assert_eq!(r_no_sl, "Liquidated (Margin Call)");

        let pos_gap_liq = dummy_position(
            PositionSide::Long,
            100.0,
            Some(95.0),
            Some(110.0),
            80.0,
        );
        let candle_gap = Kline {
            timestamp: 3000,
            open: 78.0,
            high: 79.0,
            low: 70.0,
            close: 71.0,
            volume: 10.0,
        };
        let (p_gap, r_gap) = ExitEvaluator::evaluate_intra_candle(&pos_gap_liq, &candle_gap)
            .expect("Debe liquidar con Gap de apertura bajo liq");
        assert_eq!(p_gap, 78.0);
        assert_eq!(r_gap, "Liquidated (Margin Call)");

        let pos_short_gap_liq = dummy_position(
            PositionSide::Short,
            100.0,
            Some(105.0),
            Some(90.0),
            120.0,
        );
        let candle_short_gap = Kline {
            timestamp: 3100,
            open: 125.0,
            high: 130.0,
            low: 124.0,
            close: 128.0,
            volume: 10.0,
        };
        let (p_sgap, r_sgap) = ExitEvaluator::evaluate_intra_candle(&pos_short_gap_liq, &candle_short_gap)
            .expect("Debe liquidar short con Gap de apertura sobre liq");
        assert_eq!(p_sgap, 125.0);
        assert_eq!(r_sgap, "Liquidated (Margin Call)");

        let pos_liq_first = dummy_position(
            PositionSide::Long,
            100.0,
            Some(85.0),
            Some(110.0),
            90.0,
        );
        let candle_lf = Kline {
            timestamp: 4000,
            open: 98.0,
            high: 99.0,
            low: 80.0,
            close: 82.0,
            volume: 10.0,
        };
        let (p_lf, r_lf) = ExitEvaluator::evaluate_intra_candle(&pos_liq_first, &candle_lf)
            .expect("Debe liquidar si liq_price es alcanzado antes de SL");
        assert_eq!(p_lf, 90.0);
        assert_eq!(r_lf, "Liquidated (Margin Call)");
    }
}

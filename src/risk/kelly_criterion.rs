use crate::types::CompletedTrade;

/// Dimensionador de posición basado en el Criterio de Kelly Fraccional (Fractional Kelly).
/// Modula la exposición a partir del edge empírico causal (win rate y payoff ratio)
/// observado en una ventana móvil retrospectiva de trades cerrados.
#[derive(Debug, Clone, PartialEq)]
pub struct FractionalKellySizer {
    /// Multiplicador fraccional defensivo lambda (ej: 0.25 para Quarter Kelly)
    pub fraction: f64,
    /// Tamaño máximo de la ventana móvil de trades históricos
    pub window: usize,
    /// Límite inferior de asignación fraccional
    pub min_fraction: f64,
    /// Límite superior de asignación fraccional
    pub max_fraction: f64,
    /// Cantidad mínima de trades cerrados requeridos para estimación estadística
    pub min_trades: usize,
    /// Fracción conservadora por defecto cuando no se alcanza el mínimo de trades
    pub default_fraction: f64,
}

impl FractionalKellySizer {
    pub fn new(
        fraction: f64,
        window: usize,
        min_fraction: f64,
        max_fraction: f64,
        min_trades: usize,
        default_fraction: f64,
    ) -> Self {
        let safe_fraction = if fraction.is_finite() && fraction > 0.0 {
            fraction.clamp(0.01, 1.0)
        } else {
            0.25
        };
        let safe_min = if min_fraction.is_finite() && min_fraction >= 0.0 {
            min_fraction.clamp(0.0, 1.0)
        } else {
            0.02
        };
        let safe_max = if max_fraction.is_finite() && max_fraction >= safe_min {
            max_fraction.clamp(safe_min, 1.0)
        } else {
            0.25_f64.max(safe_min)
        };
        let safe_default = if default_fraction.is_finite() && default_fraction >= 0.0 {
            default_fraction.clamp(safe_min, safe_max)
        } else {
            safe_min
        };

        Self {
            fraction: safe_fraction,
            window: window.max(5),
            min_fraction: safe_min,
            max_fraction: safe_max,
            min_trades: min_trades.max(2),
            default_fraction: safe_default,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.25, 30, 0.02, 0.25, 20, 0.05)
    }

    /// Calcula la fracción de capital recomendada a partir de los trades cerrados previos (t <= T).
    ///
    /// Fórmula:
    ///   f* = lambda * ((p * b - q) / b)
    /// donde:
    ///   p: tasa de victorias (win rate)
    ///   q: tasa de pérdidas (1.0 - p)
    ///   b: ratio de payoff (ganancia promedio / pérdida promedio)
    pub fn calculate_fraction(&self, closed_trades: &[CompletedTrade]) -> f64 {
        if closed_trades.is_empty() {
            return self.default_fraction;
        }

        let start_idx = closed_trades.len().saturating_sub(self.window);
        let sample = &closed_trades[start_idx..];

        let mut win_count = 0usize;
        let mut loss_count = 0usize;
        let mut total_win_pnl = 0.0f64;
        let mut total_loss_pnl = 0.0f64;

        for trade in sample {
            if !trade.net_pnl.is_finite() {
                continue;
            }
            if trade.net_pnl > 0.0 {
                win_count += 1;
                total_win_pnl += trade.net_pnl;
            } else if trade.net_pnl < 0.0 {
                loss_count += 1;
                total_loss_pnl += trade.net_pnl.abs();
            }
        }

        let total_valid = win_count + loss_count;
        if total_valid < self.min_trades {
            return self.default_fraction;
        }

        // Si no hubo pérdidas en la muestra (100% win rate), el ratio b tendería a infinito.
        // Asignamos la fracción máxima configurada del lambda de Kelly.
        if loss_count == 0 {
            return (self.fraction * 1.0).clamp(self.min_fraction, self.max_fraction);
        }

        // Si no hubo ganancias (0% win rate), edge estrictamente negativo: asignación mínima
        if win_count == 0 {
            return self.min_fraction;
        }

        let avg_win = total_win_pnl / win_count as f64;
        let avg_loss = total_loss_pnl / loss_count as f64;

        if !avg_win.is_finite() || !avg_loss.is_finite() || avg_loss <= 0.0 {
            return self.min_fraction;
        }

        let b = avg_win / avg_loss;
        if !b.is_finite() || b <= 0.0 {
            return self.min_fraction;
        }

        let p = win_count as f64 / total_valid as f64;
        let q = 1.0 - p;

        // Edge esperado: (p * b - q)
        let edge = p * b - q;
        if edge <= 0.0 {
            return self.min_fraction;
        }

        let full_kelly = edge / b;
        let scaled_kelly = self.fraction * full_kelly;

        if !scaled_kelly.is_finite() {
            return self.min_fraction;
        }

        scaled_kelly.clamp(self.min_fraction, self.max_fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PositionSide;

    fn make_trade(net_pnl: f64) -> CompletedTrade {
        CompletedTrade {
            side: PositionSide::Long,
            entry_timestamp: 1000,
            exit_timestamp: 2000,
            entry_price: 100.0,
            exit_price: 100.0 + net_pnl,
            quantity: 1.0,
            gross_pnl: net_pnl,
            commissions: 0.0,
            funding_paid: 0.0,
            net_pnl,
            return_pct: net_pnl / 100.0,
            duration_bars: 1,
            duration_millis: 1000,
            max_favorable_excursion: 0.0,
            max_adverse_excursion: 0.0,
            exit_reason: "Test".to_string(),
            entry_candle_idx: 0,
            exit_candle_idx: 1,
            regime: crate::types::TradeRegime::PrimaryTrend,
        }
    }

    #[test]
    fn test_empty_trades_returns_default() {
        let sizer = FractionalKellySizer::with_defaults();
        assert_eq!(sizer.calculate_fraction(&[]), sizer.default_fraction);
    }

    #[test]
    fn test_insufficient_trades_returns_default() {
        let sizer = FractionalKellySizer::with_defaults();
        let trades: Vec<CompletedTrade> = (0..10).map(|_| make_trade(10.0)).collect();
        assert_eq!(sizer.calculate_fraction(&trades), sizer.default_fraction);
    }

    #[test]
    fn test_positive_edge_kelly_calculation() {
        // 20 trades: 12 ganadores ($20) y 8 perdedores ($10)
        // p = 12/20 = 0.60, q = 0.40, avg_win = 20, avg_loss = 10 -> b = 2.0
        // full_kelly = (0.60 * 2.0 - 0.40) / 2.0 = (1.20 - 0.40) / 2.0 = 0.80 / 2.0 = 0.40
        // lambda = 0.25 -> f* = 0.25 * 0.40 = 0.10
        let sizer = FractionalKellySizer::new(0.25, 30, 0.02, 0.25, 20, 0.05);
        let mut trades = Vec::new();
        for _ in 0..12 {
            trades.push(make_trade(20.0));
        }
        for _ in 0..8 {
            trades.push(make_trade(-10.0));
        }

        let f = sizer.calculate_fraction(&trades);
        assert!((f - 0.10).abs() < 1e-4);
    }

    #[test]
    fn test_negative_edge_returns_min_fraction() {
        // p = 0.30, b = 1.0 -> edge = 0.30 - 0.70 = -0.40 <= 0 -> min_fraction (0.02)
        let sizer = FractionalKellySizer::new(0.25, 30, 0.02, 0.25, 20, 0.05);
        let mut trades = Vec::new();
        for _ in 0..6 {
            trades.push(make_trade(10.0));
        }
        for _ in 0..14 {
            trades.push(make_trade(-10.0));
        }

        let f = sizer.calculate_fraction(&trades);
        assert_eq!(f, 0.02);
    }

    #[test]
    fn test_all_winners_capped_properly() {
        let sizer = FractionalKellySizer::new(0.5, 30, 0.02, 0.25, 20, 0.05);
        let trades: Vec<CompletedTrade> = (0..25).map(|_| make_trade(15.0)).collect();
        let f = sizer.calculate_fraction(&trades);
        assert_eq!(f, 0.25); // Clamped at max_fraction
    }
}

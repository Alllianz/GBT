use serde::{Deserialize, Serialize};
use std::fmt;
use crate::backtest::BacktestMetrics;

/// Métricas de Fitness para evaluación, ranking y optimización de modelos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FitnessMetric {
    /// 0.5 * Linealidad R² + 0.5 * (Capital Final / Max Drawdown Nominal)
    #[default]
    CompositeAllianz,
    /// Sharpe Ratio Anualizado
    SharpeRatio,
    /// Sortino Ratio Anualizado
    SortinoRatio,
    /// Recovery Factor: Beneficio Neto / Max Drawdown Nominal
    RecoveryFactor,
    /// Linealidad R² de la Curva de Balance (0.0 a 1.0)
    LinearityR2,
    /// Calmar Ratio: Beneficio Neto % / Max Drawdown %
    CalmarRatio,
    /// Profit Factor: Ganancia Bruta / Pérdida Bruta
    ProfitFactor,
    /// Retorno Neto Total Porcentual
    NetProfitPct,
}

impl fmt::Display for FitnessMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitnessMetric::CompositeAllianz => write!(f, "Score Compuesto Allianz [0.5*R² + 0.5*(Cap/MaxDD)]"),
            FitnessMetric::SharpeRatio => write!(f, "Sharpe Ratio (Anualizado)"),
            FitnessMetric::SortinoRatio => write!(f, "Sortino Ratio (Anualizado)"),
            FitnessMetric::RecoveryFactor => write!(f, "Recovery Factor (Net Profit / MaxDD)"),
            FitnessMetric::LinearityR2 => write!(f, "Linealidad de Balance R²"),
            FitnessMetric::CalmarRatio => write!(f, "Calmar Ratio (Retorno% / MaxDD%)"),
            FitnessMetric::ProfitFactor => write!(f, "Profit Factor"),
            FitnessMetric::NetProfitPct => write!(f, "Retorno Neto %"),
        }
    }
}

impl FitnessMetric {
    /// Devuelve el nombre corto legible
    pub fn name(&self) -> &'static str {
        match self {
            FitnessMetric::CompositeAllianz => "Score Allianz (0.5 R² + 0.5 Cap/DD)",
            FitnessMetric::SharpeRatio => "Sharpe Ratio Anual",
            FitnessMetric::SortinoRatio => "Sortino Ratio Anual",
            FitnessMetric::RecoveryFactor => "Recovery Factor",
            FitnessMetric::LinearityR2 => "Linealidad R²",
            FitnessMetric::CalmarRatio => "Calmar Ratio",
            FitnessMetric::ProfitFactor => "Profit Factor",
            FitnessMetric::NetProfitPct => "Retorno Neto %",
        }
    }

    /// Devuelve la fórmula matemática explicativa
    pub fn formula(&self) -> &'static str {
        match self {
            FitnessMetric::CompositeAllianz => "0.5 * clamp(R², 0, 1) + 0.5 * (Cap_final / max(1.0, MaxDD_nom))",
            FitnessMetric::SharpeRatio => "if Mean_Ret >= 0 { (Mean_Ret / Std_Ret) * sqrt(annual_bars) } else { (Mean_Ret * Std_Ret) * sqrt(annual_bars) }",
            FitnessMetric::SortinoRatio => "if Mean_Ret >= 0 { (Mean_Ret / Downside_Std) * sqrt(annual_bars) } else { (Mean_Ret * Downside_Std) * sqrt(annual_bars) }",
            FitnessMetric::RecoveryFactor => "Net_Profit / max(1.0, MaxDD_nom)",
            FitnessMetric::LinearityR2 => "1 - sum(residuals²) / sum((equity - mean)²)",
            FitnessMetric::CalmarRatio => "if Net_Profit >= 0 { Net_Profit_pct / max(0.01, MaxDD_pct) } else { Net_Profit_pct * (max(0.01, MaxDD_pct) / 100) }",
            FitnessMetric::ProfitFactor => "Gross_Profit / max(0.01, Gross_Loss)",
            FitnessMetric::NetProfitPct => "((Cap_final - Cap_init) / Cap_init) * 100",
        }
    }

    /// Evalúa y devuelve el valor numérico escalar de la métrica fitness seleccionada
    pub fn evaluate(&self, m: &BacktestMetrics) -> f64 {
        match self {
            FitnessMetric::CompositeAllianz => m.composite_score,
            FitnessMetric::SharpeRatio => m.sharpe_ratio,
            FitnessMetric::SortinoRatio => m.sortino_ratio,
            FitnessMetric::RecoveryFactor => m.recovery_factor,
            FitnessMetric::LinearityR2 => m.curve_r2,
            FitnessMetric::CalmarRatio => {
                let safe_dd = m.max_drawdown_pct.max(0.01);
                if m.net_profit_pct >= 0.0 {
                    m.net_profit_pct / safe_dd
                } else {
                    // BUG-06: Normalización monótona institucional para retornos negativos.
                    // Evita la inversión matemática donde dividir por un max_dd mayor
                    // producía un valor más cercano a 0 favoreciendo estrategias con pérdidas catastróficas.
                    // A mayor drawdown con pérdidas, mayor penalización continua:
                    // Ret = -10%, DD = 10% -> Calmar = -1.0
                    // Ret = -10%, DD = 50% -> Calmar = -5.0 (-5.0 < -1.0, penalización monótona correcta)
                    m.net_profit_pct * (safe_dd / 100.0)
                }
            }
            FitnessMetric::ProfitFactor => m.profit_factor,
            FitnessMetric::NetProfitPct => m.net_profit_pct,
        }
    }

    /// Evalúa todas las métricas fitness soportadas para auditoría comparativa
    pub fn evaluate_all(m: &BacktestMetrics) -> Vec<(FitnessMetric, &'static str, f64)> {
        vec![
            (FitnessMetric::CompositeAllianz, "Score Compuesto Allianz", FitnessMetric::CompositeAllianz.evaluate(m)),
            (FitnessMetric::SharpeRatio, "Sharpe Ratio Anual", FitnessMetric::SharpeRatio.evaluate(m)),
            (FitnessMetric::SortinoRatio, "Sortino Ratio Anual", FitnessMetric::SortinoRatio.evaluate(m)),
            (FitnessMetric::RecoveryFactor, "Recovery Factor", FitnessMetric::RecoveryFactor.evaluate(m)),
            (FitnessMetric::LinearityR2, "Linealidad R²", FitnessMetric::LinearityR2.evaluate(m)),
            (FitnessMetric::CalmarRatio, "Calmar Ratio", FitnessMetric::CalmarRatio.evaluate(m)),
            (FitnessMetric::ProfitFactor, "Profit Factor", FitnessMetric::ProfitFactor.evaluate(m)),
            (FitnessMetric::NetProfitPct, "Retorno Neto %", FitnessMetric::NetProfitPct.evaluate(m)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_metrics(net_profit_pct: f64, max_dd_pct: f64) -> BacktestMetrics {
        BacktestMetrics {
            initial_capital: 10000.0,
            final_capital: 10000.0 * (1.0 + net_profit_pct / 100.0),
            net_profit: 10000.0 * (net_profit_pct / 100.0),
            net_profit_pct,
            total_trades: 10,
            winning_trades: 5,
            losing_trades: 5,
            win_rate_pct: 50.0,
            gross_profit: 1000.0,
            gross_loss: 1000.0,
            profit_factor: 1.0,
            avg_trade_pnl: 0.0,
            avg_win_pnl: 200.0,
            avg_loss_pnl: 200.0,
            payoff_ratio: 1.0,
            max_drawdown_nominal: 10000.0 * (max_dd_pct / 100.0),
            max_drawdown_pct: max_dd_pct,
            recovery_factor: 1.0,
            total_commissions: 10.0,
            total_funding: 0.0,
            sharpe_ratio: 1.0,
            sortino_ratio: 1.0,
            curve_r2: 0.95,
            capital_to_maxdd_ratio: 1.0,
            composite_score: 1.0,
            avg_duration_bars: 5.0,
            max_consecutive_losses: 2,
        }
    }

    #[test]
    fn test_bug06_calmar_ratio_monotonic_penalization_in_losses() {
        let metric = FitnessMetric::CalmarRatio;

        // Dos estrategias con idéntica pérdida (-10%), pero B sufrió mucho mayor drawdown (50% vs 10%)
        let strategy_a = dummy_metrics(-10.0, 10.0);
        let strategy_b = dummy_metrics(-10.0, 50.0);

        let score_a = metric.evaluate(&strategy_a);
        let score_b = metric.evaluate(&strategy_b);

        // Score A: -10.0 * (10.0 / 100.0) = -1.0
        // Score B: -10.0 * (50.0 / 100.0) = -5.0
        assert!((score_a - (-1.0)).abs() < 1e-6);
        assert!((score_b - (-5.0)).abs() < 1e-6);

        // La estrategia con mayor drawdown DEBE tener un score MENOR (más negativo), penalizando el riesgo
        assert!(score_a > score_b, "Estrategia con menor drawdown debe clasificar mejor que estrategia con mayor drawdown en pérdidas");
    }

    #[test]
    fn test_calmar_ratio_profitable_strategy() {
        let metric = FitnessMetric::CalmarRatio;

        let strat_high_gain = dummy_metrics(20.0, 10.0); // 20 / 10 = 2.0
        let strat_low_gain = dummy_metrics(10.0, 10.0);  // 10 / 10 = 1.0

        let score_high = metric.evaluate(&strat_high_gain);
        let score_low = metric.evaluate(&strat_low_gain);

        assert!((score_high - 2.0).abs() < 1e-6);
        assert!((score_low - 1.0).abs() < 1e-6);
        assert!(score_high > score_low);
    }
}

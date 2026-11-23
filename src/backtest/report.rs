use super::engine::BacktestResult;
use crate::types::FitnessMetric;
use std::fmt::Write;

pub struct BacktestReport;

impl BacktestReport {
    pub fn format_summary(result: &BacktestResult) -> String {
        let m = &result.metrics;
        let mut out = String::new();

        let _ = writeln!(out, "==================================================");
        let _ = writeln!(out, "        GBT - REPORTE DE BACKTEST INSTITUCIONAL   ");
        let _ = writeln!(out, "==================================================");
        let _ = writeln!(out, "Capital Inicial:        ${:.2}", m.initial_capital);
        let _ = writeln!(out, "Capital Final:          ${:.2}", m.final_capital);
        let _ = writeln!(out, "Beneficio Neto Total:   ${:.2} ({:+.2}%)", m.net_profit, m.net_profit_pct);
        let _ = writeln!(out, "Total Comisiones:       ${:.2}", m.total_commissions);
        let _ = writeln!(out, "Total Funding:          ${:.2}", m.total_funding);
        let _ = writeln!(out, "--------------------------------------------------");
        let _ = writeln!(out, "Total de Operaciones:   {}", m.total_trades);
        let _ = writeln!(out, "Operaciones Ganadoras:  {} ({:.2}%)", m.winning_trades, m.win_rate_pct);
        let _ = writeln!(out, "Operaciones Perdedoras: {}", m.losing_trades);
        let _ = writeln!(out, "Profit Factor:          {:.3}", m.profit_factor);
        let _ = writeln!(out, "Payoff Ratio:           {:.3}", m.payoff_ratio);
        let _ = writeln!(out, "Ganancia Promedio:      ${:.2}", m.avg_win_pnl);
        let _ = writeln!(out, "Pérdida Promedio:       ${:.2}", m.avg_loss_pnl);
        let _ = writeln!(out, "--------------------------------------------------");
        let _ = writeln!(out, "Max Drawdown Nominal:   ${:.2}", m.max_drawdown_nominal);
        let _ = writeln!(out, "Max Drawdown %:         {:.2}%", m.max_drawdown_pct);
        let _ = writeln!(out, "Recovery Factor:        {:.3}", m.recovery_factor);
        let _ = writeln!(out, "Capital / MaxDD Ratio:  {:.3}", m.capital_to_maxdd_ratio);
        let _ = writeln!(out, "Score Compuesto:        {:.4} [0.5*R² + 0.5*(Cap/MaxDD)]", m.composite_score);
        let _ = writeln!(out, "Sharpe Ratio (Anual):   {:.3}", m.sharpe_ratio);
        let _ = writeln!(out, "Sortino Ratio (Anual):  {:.3}", m.sortino_ratio);
        let _ = writeln!(out, "Consistencia Curva R²:  {:.4}", m.curve_r2);
        let _ = writeln!(out, "Racha Máx Pérdidas:     {}", m.max_consecutive_losses);
        let _ = writeln!(out, "Duración Media (Velas): {:.1}", m.avg_duration_bars);
        let _ = writeln!(out, "==================================================");

        out
    }

    /// Genera una tabla comparativa de todas las métricas de fitness disponibles
    pub fn format_fitness_comparison(result: &BacktestResult, active_metric: FitnessMetric) -> String {
        let m = &result.metrics;
        let mut out = String::new();

        let _ = writeln!(out, "\n==========================================================================");
        let _ = writeln!(out, "            AUDITORÍA COMPARATIVA DE MÉTRICAS FITNESS                    ");
        let _ = writeln!(out, "==========================================================================");
        let _ = writeln!(out, "{:<4} {:<32} {:<14} FÓRMULA / CRITERIO", "ACT", "MÉTRICA FITNESS", "PUNTAJE");
        let _ = writeln!(out, "--------------------------------------------------------------------------");

        for (metric, label, score) in FitnessMetric::evaluate_all(m) {
            let active_indicator = if metric == active_metric { ">>> " } else { "    " };
            let _ = writeln!(
                out,
                "{}{:<32} {:<14.4} {}",
                active_indicator,
                label,
                score,
                metric.formula()
            );
        }
        let _ = writeln!(out, "==========================================================================\n");

        out
    }
}

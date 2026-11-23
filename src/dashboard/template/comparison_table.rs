use crate::backtest::WalkForwardResult;
use super::escape::escape_html;

pub fn render_comparison_table(wf: &WalkForwardResult) -> String {
    let is_m = &wf.is_result.metrics;
    let oos_m = &wf.oos_result.metrics;

    let fitness_name = escape_html(wf.fitness_metric.name());
    let is_fitness_score = wf.fitness_metric.evaluate(is_m);
    let oos_fitness_score = wf.fitness_metric.evaluate(oos_m);

    let is_ret_color = if is_m.net_profit_pct >= 0.0 { "green" } else { "red" };
    let oos_ret_color = if oos_m.net_profit_pct >= 0.0 { "green" } else { "red" };

    let sharpe_ratio_deg = if is_m.sharpe_ratio.abs() > 1e-6 {
        oos_m.sharpe_ratio / is_m.sharpe_ratio
    } else {
        1.0
    };
    let dd_ratio_deg = if is_m.max_drawdown_pct > 0.0 {
        oos_m.max_drawdown_pct / is_m.max_drawdown_pct
    } else {
        1.0
    };

    format!(
        r#"
    <!-- Tabla Comparativa Exhaustiva IS vs OOS -->
    <h3 style="margin-bottom: 12px;">📊 Auditoría Cuantitativa Comparativa (IS vs OOS)</h3>
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Métrica Financiera</th>
                <th class="num">In-Sample (Entrenamiento)</th>
                <th class="num">Out-of-Sample (Prueba Ciega)</th>
                <th class="num">Ratio de Robustez / Degradación</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>🎯 Métrica Fitness Activa: {fitness_name}</strong></td>
                <td class="num purple"><strong>{is_fitness_score:.4}</strong></td>
                <td class="num blue"><strong>{oos_fitness_score:.4}</strong></td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td><strong>Beneficio Neto Total %</strong></td>
                <td class="num {is_ret_color}"><strong>{is_ret:+0.2}%</strong></td>
                <td class="num {oos_ret_color}"><strong>{oos_ret:+0.2}%</strong></td>
                <td class="num purple"><strong>{robustness:.2}x</strong></td>
            </tr>
            <tr>
                <td><strong>Score Compuesto [0.5*R² + 0.5*(Cap/MaxDD)]</strong></td>
                <td class="num purple"><strong>{is_composite:.4}</strong></td>
                <td class="num blue"><strong>{oos_composite:.4}</strong></td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Ratio Capital / MaxDD Nominal</td>
                <td class="num">{is_cap_maxdd:.3}</td>
                <td class="num">{oos_cap_maxdd:.3}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Capital Final ($)</td>
                <td class="num">${is_final_cap:.2}</td>
                <td class="num">${oos_final_cap:.2}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Sharpe Ratio (Anualizado)</td>
                <td class="num">{is_sharpe:.3}</td>
                <td class="num">{oos_sharpe:.3}</td>
                <td class="num">{sharpe_ratio_deg:.2}x</td>
            </tr>
            <tr>
                <td>Sortino Ratio (Anualizado)</td>
                <td class="num">{is_sortino:.3}</td>
                <td class="num">{oos_sortino:.3}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Drawdown Máximo %</td>
                <td class="num red">{is_dd:.2}%</td>
                <td class="num red">{oos_dd:.2}%</td>
                <td class="num">{dd_ratio_deg:.2}x</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td class="num">{is_pf:.3}</td>
                <td class="num">{oos_pf:.3}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Win Rate %</td>
                <td class="num">{is_wr:.2}%</td>
                <td class="num">{oos_wr:.2}%</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Payoff Ratio (Win / Loss)</td>
                <td class="num">{is_payoff:.3}</td>
                <td class="num">{oos_payoff:.3}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Coeficiente R² de Linealidad</td>
                <td class="num">{is_r2:.4}</td>
                <td class="num">{oos_r2:.4}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Total de Operaciones</td>
                <td class="num">{is_trades}</td>
                <td class="num">{oos_trades}</td>
                <td class="num">-</td>
            </tr>
            <tr>
                <td>Total Comisiones Pagadas</td>
                <td class="num">${is_commissions:.2}</td>
                <td class="num">${oos_commissions:.2}</td>
                <td class="num">-</td>
            </tr>
        </tbody>
    </table>
        "#,
        fitness_name = fitness_name,
        is_fitness_score = is_fitness_score,
        oos_fitness_score = oos_fitness_score,
        is_ret = is_m.net_profit_pct,
        is_ret_color = is_ret_color,
        oos_ret = oos_m.net_profit_pct,
        oos_ret_color = oos_ret_color,
        robustness = wf.metrics.robustness_ratio,
        is_composite = is_m.composite_score,
        oos_composite = oos_m.composite_score,
        is_cap_maxdd = is_m.capital_to_maxdd_ratio,
        oos_cap_maxdd = oos_m.capital_to_maxdd_ratio,
        is_final_cap = is_m.final_capital,
        oos_final_cap = oos_m.final_capital,
        is_sharpe = is_m.sharpe_ratio,
        oos_sharpe = oos_m.sharpe_ratio,
        sharpe_ratio_deg = sharpe_ratio_deg,
        is_sortino = is_m.sortino_ratio,
        oos_sortino = oos_m.sortino_ratio,
        is_dd = is_m.max_drawdown_pct,
        oos_dd = oos_m.max_drawdown_pct,
        dd_ratio_deg = dd_ratio_deg,
        is_pf = is_m.profit_factor,
        oos_pf = oos_m.profit_factor,
        is_wr = is_m.win_rate_pct,
        oos_wr = oos_m.win_rate_pct,
        is_payoff = is_m.payoff_ratio,
        oos_payoff = oos_m.payoff_ratio,
        is_r2 = is_m.curve_r2,
        oos_r2 = oos_m.curve_r2,
        is_trades = is_m.total_trades,
        oos_trades = oos_m.total_trades,
        is_commissions = is_m.total_commissions,
        oos_commissions = oos_m.total_commissions,
    )
}

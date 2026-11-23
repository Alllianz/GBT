use crate::backtest::WalkForwardResult;
use super::escape::escape_html;

pub fn render_header(wf: &WalkForwardResult, gen_date: &str) -> String {
    let model_name = escape_html(&wf.model_name);
    let timeframe = escape_html(&wf.timeframe);

    format!(
        r#"
    <div class="header">
        <div>
            <h1>🚀 GBT Dashboard Institucional: {model_name}</h1>
            <div style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">
                Timeframe: <strong>{timeframe}</strong> | Evaluación Temporal: <strong>In-Sample (IS) vs Out-of-Sample (OOS)</strong>
            </div>
        </div>
        <div class="badge">Generado: {gen_date} UTC</div>
    </div>
        "#
    )
}

pub fn render_fitness_banner(wf: &WalkForwardResult) -> String {
    let fitness_name = escape_html(wf.fitness_metric.name());
    let fitness_formula = escape_html(wf.fitness_metric.formula());
    let is_m = &wf.is_result.metrics;
    let oos_m = &wf.oos_result.metrics;
    let is_fitness_score = wf.fitness_metric.evaluate(is_m);
    let oos_fitness_score = wf.fitness_metric.evaluate(oos_m);

    format!(
        r#"
    <!-- Banner de Métrica Fitness Utilizada -->
    <div class="fitness-banner">
        <div class="fitness-icon">🎯</div>
        <div class="fitness-details">
            <div class="fitness-title">Métrica Fitness de Optimización: <strong class="blue">{fitness_name}</strong></div>
            <div class="fitness-formula">Fórmula: {fitness_formula}</div>
        </div>
        <div class="fitness-scores">
            <div class="fitness-pill is">Puntaje IS: {is_fitness_score:.4}</div>
            <div class="fitness-pill oos">Puntaje OOS: {oos_fitness_score:.4}</div>
        </div>
    </div>
        "#
    )
}

pub fn render_comparison_cards(wf: &WalkForwardResult) -> String {
    let is_m = &wf.is_result.metrics;
    let oos_m = &wf.oos_result.metrics;

    let is_ret_color = if is_m.net_profit_pct >= 0.0 { "green" } else { "red" };
    let oos_ret_color = if oos_m.net_profit_pct >= 0.0 { "green" } else { "red" };

    format!(
        r#"
    <!-- Comparativa de Bloques IS vs OOS -->
    <div class="comparison-bar">
        <div class="period-card is">
            <h3>
                <span>🧠 In-Sample (IS - Entrenamiento)</span>
                <span class="purple">{is_trades} Trades</span>
            </h3>
            <div class="period-metrics">
                <div class="p-metric">
                    <div class="lbl">Retorno Total</div>
                    <div class="val {is_ret_color}">{is_ret:+0.2}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Sharpe Ratio</div>
                    <div class="val">{is_sharpe:.2}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Max Drawdown</div>
                    <div class="val red">{is_dd:.2}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Profit Factor</div>
                    <div class="val">{is_pf:.2}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Win Rate</div>
                    <div class="val">{is_wr:.1}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Consistencia R²</div>
                    <div class="val">{is_r2:.3}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Score Compuesto</div>
                    <div class="val purple">{is_composite:.2}</div>
                </div>
            </div>
        </div>

        <div class="period-card oos">
            <h3>
                <span>🛡️ Out-of-Sample (OOS - Prueba Ciega)</span>
                <span class="blue">{oos_trades} Trades</span>
            </h3>
            <div class="period-metrics">
                <div class="p-metric">
                    <div class="lbl">Retorno Total</div>
                    <div class="val {oos_ret_color}">{oos_ret:+0.2}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Sharpe Ratio</div>
                    <div class="val">{oos_sharpe:.2}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Max Drawdown</div>
                    <div class="val red">{oos_dd:.2}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Profit Factor</div>
                    <div class="val">{oos_pf:.2}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Win Rate</div>
                    <div class="val">{oos_wr:.1}%</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Consistencia R²</div>
                    <div class="val">{oos_r2:.3}</div>
                </div>
                <div class="p-metric">
                    <div class="lbl">Score Compuesto</div>
                    <div class="val blue">{oos_composite:.2}</div>
                </div>
            </div>
        </div>
    </div>
        "#,
        is_trades = is_m.total_trades,
        is_ret = is_m.net_profit_pct,
        is_ret_color = is_ret_color,
        is_sharpe = is_m.sharpe_ratio,
        is_dd = is_m.max_drawdown_pct,
        is_pf = is_m.profit_factor,
        is_wr = is_m.win_rate_pct,
        is_r2 = is_m.curve_r2,
        is_composite = is_m.composite_score,
        oos_trades = oos_m.total_trades,
        oos_ret = oos_m.net_profit_pct,
        oos_ret_color = oos_ret_color,
        oos_sharpe = oos_m.sharpe_ratio,
        oos_dd = oos_m.max_drawdown_pct,
        oos_pf = oos_m.profit_factor,
        oos_wr = oos_m.win_rate_pct,
        oos_r2 = oos_m.curve_r2,
        oos_composite = oos_m.composite_score,
    )
}

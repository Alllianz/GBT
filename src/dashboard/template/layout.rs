use chrono::Utc;
use crate::backtest::{FullBacktestResult, WalkForwardResult};
use crate::types::FitnessMetric;
use super::charts::{render_charts_containers, render_charts_script, TimelineSeries};
use super::comparison_table::render_comparison_table;
use super::escape::escape_html;
use super::metrics_cards::{render_comparison_cards, render_fitness_banner, render_header};
use super::styles::get_styles;
use super::trade_table::render_trades_section;

pub fn render_full_dashboard(wf: &WalkForwardResult) -> String {
    let model_name = escape_html(&wf.model_name);
    let timeframe = escape_html(&wf.timeframe);
    let gen_date = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

    let css = get_styles();
    let header_html = render_header(wf, &gen_date);
    let fitness_html = render_fitness_banner(wf);
    let comparison_cards_html = render_comparison_cards(wf);
    let charts_containers = render_charts_containers();
    let comparison_table_html = render_comparison_table(wf);
    let trades_section_html = render_trades_section(wf);

    let timeline = TimelineSeries::build(wf);
    let charts_script = render_charts_script(&timeline);

    format!(
        r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GBT Dashboard - {model_name} [{timeframe}] - IS & OOS</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
{css}
    </style>
</head>
<body>
{header_html}
{fitness_html}
{comparison_cards_html}
{charts_containers}
{comparison_table_html}
{trades_section_html}
{charts_script}
</body>
</html>"#
    )
}

pub fn render_purged_full_dashboard(
    model_name: &str,
    timeframe: &str,
    full: &FullBacktestResult,
    fitness_metric: FitnessMetric,
) -> String {
    let wf = WalkForwardResult::new(
        model_name,
        timeframe,
        full.is_result.clone(),
        full.oos_result.clone(),
        fitness_metric,
    );

    let escaped_model = escape_html(model_name);
    let escaped_tf = escape_html(timeframe);
    let gen_date = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

    let css = get_styles();
    let header_html = render_header(&wf, &gen_date);
    let fitness_html = render_fitness_banner(&wf);
    let comparison_cards_html = render_comparison_cards(&wf);
    let charts_containers = render_charts_containers();
    let comparison_table_html = render_comparison_table(&wf);
    let trades_section_html = render_trades_section(&wf);

    let timeline = TimelineSeries::build_full(full);
    let charts_script = render_charts_script(&timeline);

    format!(
        r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GBT Dashboard - {escaped_model} [{escaped_tf}] - Purged Full Backtest</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
{css}
    </style>
</head>
<body>
{header_html}
{fitness_html}
{comparison_cards_html}
{charts_containers}
{comparison_table_html}
{trades_section_html}
{charts_script}
</body>
</html>"#
    )
}

use chrono::{TimeZone, Utc};
use crate::backtest::{FullBacktestResult, WalkForwardResult};

pub struct TimelineSeries {
    pub labels: Vec<String>,
    pub is_equity: Vec<String>,
    pub oos_equity: Vec<String>,
    pub is_dd: Vec<String>,
    pub oos_dd: Vec<String>,
    pub purge_equity: Option<Vec<String>>,
    pub purge_dd: Option<Vec<String>>,
}

impl TimelineSeries {
    pub fn build(wf: &WalkForwardResult) -> Self {
        let mut labels = Vec::new();
        let mut is_equity = Vec::new();
        let mut oos_equity = Vec::new();
        let mut is_dd = Vec::new();
        let mut oos_dd = Vec::new();

        // 1. Agregar puntos de In-Sample (IS)
        let n_is = wf.is_result.equity_curve.len();
        for (i, p) in wf.is_result.equity_curve.iter().enumerate() {
            let d = Utc
                .timestamp_millis_opt(p.timestamp)
                .single()
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| p.candle_idx.to_string());

            labels.push(format!("\"{}\"", d));
            is_equity.push(format!("{:.2}", p.equity));

            // BUG-21: Evitar cero negativo "-0.00%" en Chart.js cuando el drawdown es 0
            if p.drawdown_pct < 1e-4 {
                is_dd.push("0.00".to_string());
            } else {
                is_dd.push(format!("-{:.2}", p.drawdown_pct));
            }

            // Para que OOS empiece exactamente en el punto final de IS de forma continua:
            if i == n_is.saturating_sub(1) {
                oos_equity.push(format!("{:.2}", p.equity));
                oos_dd.push("0.00".to_string());
            } else {
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());
            }
        }

        // 2. Agregar puntos de Out-of-Sample (OOS) continuando desde el capital final de IS
        for p in &wf.oos_result.equity_curve {
            let d = Utc
                .timestamp_millis_opt(p.timestamp)
                .single()
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| p.candle_idx.to_string());

            labels.push(format!("\"{}\"", d));
            is_equity.push("null".to_string());
            oos_equity.push(format!("{:.2}", p.equity));
            is_dd.push("null".to_string());

            // BUG-21: Evitar cero negativo "-0.00%" en Chart.js
            if p.drawdown_pct < 1e-4 {
                oos_dd.push("0.00".to_string());
            } else {
                oos_dd.push(format!("-{:.2}", p.drawdown_pct));
            }
        }

        Self {
            labels,
            is_equity,
            oos_equity,
            is_dd,
            oos_dd,
            purge_equity: None,
            purge_dd: None,
        }
    }

    /// Construye la serie temporal completa para Backtest Purged con visualización explícita
    /// de la fase de Purga/Embargo (Cuarentena Causal) sin solapamiento
    pub fn build_full(full: &FullBacktestResult) -> Self {
        let curve = &full.combined_result.equity_curve;
        let n = curve.len();
        let mut labels = Vec::with_capacity(n);
        let mut is_equity = Vec::with_capacity(n);
        let mut purge_equity = Vec::with_capacity(n);
        let mut oos_equity = Vec::with_capacity(n);
        let mut is_dd = Vec::with_capacity(n);
        let mut purge_dd = Vec::with_capacity(n);
        let mut oos_dd = Vec::with_capacity(n);

        for (i, p) in curve.iter().enumerate() {
            let d = Utc
                .timestamp_millis_opt(p.timestamp)
                .single()
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| p.candle_idx.to_string());

            labels.push(format!("\"{}\"", d));

            let dd_str = if p.drawdown_pct < 1e-4 {
                "0.00".to_string()
            } else {
                format!("-{:.2}", p.drawdown_pct)
            };
            let eq_str = format!("{:.2}", p.equity);

            if i < full.train_end_idx {
                // In-Sample puro
                is_equity.push(eq_str);
                is_dd.push(dd_str);
                purge_equity.push("null".to_string());
                purge_dd.push("null".to_string());
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());
            } else if i == full.train_end_idx {
                // Punto de empalme IS -> Purga (ambos datasets comparten el punto para continuidad visual)
                is_equity.push(eq_str.clone());
                is_dd.push(dd_str.clone());
                purge_equity.push(eq_str);
                purge_dd.push(dd_str);
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());
            } else if i < full.test_start_idx {
                // Purga / Embargo pura (capital en cuarentena estricta sin trading)
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push(eq_str);
                purge_dd.push(dd_str);
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());
            } else if i == full.test_start_idx {
                // Punto de empalme Purga -> OOS
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push(eq_str.clone());
                purge_dd.push(dd_str.clone());
                oos_equity.push(eq_str);
                oos_dd.push(dd_str);
            } else {
                // Out-of-Sample puro
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push("null".to_string());
                purge_dd.push("null".to_string());
                oos_equity.push(eq_str);
                oos_dd.push(dd_str);
            }
        }

        Self {
            labels,
            is_equity,
            oos_equity,
            is_dd,
            oos_dd,
            purge_equity: Some(purge_equity),
            purge_dd: Some(purge_dd),
        }
    }
}

pub fn render_charts_containers() -> &'static str {
    r#"
    <!-- Gráfico 1: Curva de Balance Continua IS + OOS -->
    <div class="chart-box">
        <div class="chart-header">
            <div class="chart-title">📈 Curva de Balance Continua: In-Sample (Azul Claro) ➔ Purga (Ámbar) ➔ Out-of-Sample (Azul Francia)</div>
            <div class="chart-controls">
                <div class="chart-legend-custom">Línea continua sin fugas temporales (t &le; T)</div>
                <button id="equityScaleToggle" class="scale-toggle-btn" type="button" aria-pressed="false" onclick="toggleEquityScale()">Escala Log</button>
            </div>
        </div>
        <div class="chart-wrapper">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <!-- Gráfico 2: Drawdown Submarino IS vs OOS -->
    <div class="chart-box">
        <div class="chart-header">
            <div class="chart-title">🌊 Drawdown Submarino Comparativo (%)</div>
            <div class="chart-legend-custom">Profundidad de caídas temporales</div>
        </div>
        <div class="chart-wrapper-small">
            <canvas id="ddChart"></canvas>
        </div>
    </div>
    "#
}

pub fn render_charts_script(series: &TimelineSeries) -> String {
    let labels_str = series.labels.join(",");
    let is_eq_str = series.is_equity.join(",");
    let oos_eq_str = series.oos_equity.join(",");
    let is_dd_str = series.is_dd.join(",");
    let oos_dd_str = series.oos_dd.join(",");

    let purge_eq_dataset = if let Some(ref p_eq) = series.purge_equity {
        let p_eq_str = p_eq.join(",");
        format!(
            r#",
                    {{
                        label: 'Purga / Embargo (Cuarentena Causal)',
                        data: [{p_eq_str}],
                        borderColor: '#d29922',
                        backgroundColor: 'rgba(210, 153, 34, 0.12)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: true,
                        tension: 0.05,
                        pointRadius: 0
                    }}"#
        )
    } else {
        String::new()
    };

    let purge_dd_dataset = if let Some(ref p_dd) = series.purge_dd {
        let p_dd_str = p_dd.join(",");
        format!(
            r#",
                    {{
                        label: 'Drawdown Purga (%)',
                        data: [{p_dd_str}],
                        borderColor: '#d29922',
                        backgroundColor: 'rgba(210, 153, 34, 0.20)',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        fill: true,
                        pointRadius: 0
                    }}"#
        )
    } else {
        String::new()
    };

    format!(
        r#"
    <script>
        const labels = [{labels_str}];

        // Gráfico 1: Curva de Balance
        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctxEquity, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'In-Sample (IS - Entrenamiento)',
                        data: [{is_eq_str}],
                        borderColor: '#60a5fa',
                        backgroundColor: 'rgba(96, 165, 250, 0.07)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.05,
                        pointRadius: 0
                    }}{purge_eq_dataset},
                    {{
                        label: 'Out-of-Sample (OOS - Validación Ciega)',
                        data: [{oos_eq_str}],
                        borderColor: '#0055FF',
                        backgroundColor: 'rgba(0, 85, 255, 0.16)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.05,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{
                    legend: {{ labels: {{ color: '#8a94a6', font: {{ size: 12 }} }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#8a94a6', maxTicksLimit: 14 }}, grid: {{ color: '#121826' }} }},
                    y: {{ ticks: {{ color: '#8a94a6' }}, grid: {{ color: '#121826' }} }}
                }}
            }}
        }});

        const rawEquityDatasets = equityChart.data.datasets.map(ds => ds.data.slice());
        let equityScaleMode = 'linear';
        const hasPositiveEquity = rawEquityDatasets.some(arr => arr.some(v => v !== null && Number.isFinite(v) && v > 0));
        const scaleToggleBtn = document.getElementById('equityScaleToggle');
        if (scaleToggleBtn && !hasPositiveEquity) {{
            scaleToggleBtn.disabled = true;
            scaleToggleBtn.title = 'Escala logarítmica no disponible sin valores positivos';
        }}

        function toggleEquityScale() {{
            if (!hasPositiveEquity) return;
            const toLog = equityScaleMode === 'linear';
            equityScaleMode = toLog ? 'logarithmic' : 'linear';

            equityChart.data.datasets.forEach((ds, idx) => {{
                const raw = rawEquityDatasets[idx];
                ds.data = toLog ? raw.map(v => (v !== null && Number.isFinite(v) && v > 0) ? v : null) : raw.slice();
                ds.fill = toLog ? 'start' : true;
            }});

            equityChart.options.scales.y.type = equityScaleMode;
            equityChart.options.scales.y.ticks = toLog
                ? {{
                    color: '#8a94a6',
                    callback: function(value) {{
                        return Number(value).toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
                    }}
                }}
                : {{ color: '#8a94a6' }};

            equityChart.update();

            if (scaleToggleBtn) {{
                scaleToggleBtn.textContent = toLog ? 'Escala Lineal' : 'Escala Log';
                scaleToggleBtn.classList.toggle('active', toLog);
                scaleToggleBtn.setAttribute('aria-pressed', toLog ? 'true' : 'false');
            }}
        }}

        // Gráfico 2: Drawdown Submarino
        const ctxDd = document.getElementById('ddChart').getContext('2d');
        new Chart(ctxDd, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Drawdown IS (%)',
                        data: [{is_dd_str}],
                        borderColor: '#f85149',
                        backgroundColor: 'rgba(248, 81, 73, 0.15)',
                        borderWidth: 1.5,
                        fill: true,
                        pointRadius: 0
                    }}{purge_dd_dataset},
                    {{
                        label: 'Drawdown OOS (%)',
                        data: [{oos_dd_str}],
                        borderColor: '#ff7b72',
                        backgroundColor: 'rgba(255, 123, 114, 0.25)',
                        borderWidth: 1.5,
                        fill: true,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#8a94a6' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#8a94a6', maxTicksLimit: 14 }}, grid: {{ color: '#121826' }} }},
                    y: {{ ticks: {{ color: '#8a94a6' }}, grid: {{ color: '#121826' }} }}
                }}
            }}
        }});
    </script>
        "#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::metrics_calculator::MetricsCalculator;
    use crate::backtest::{BacktestResult, EquityPoint};
    use crate::types::FitnessMetric;

    fn make_dummy_result(equity_values: &[f64]) -> BacktestResult {
        let curve: Vec<EquityPoint> = equity_values
            .iter()
            .enumerate()
            .map(|(i, &eq)| EquityPoint::new(1_700_000_000_000 + (i as i64 * 3600_000), i, eq, eq, 0.0, 0.0))
            .collect();

        BacktestResult {
            metrics: MetricsCalculator::calculate(10_000.0, &[], &curve),
            trades: Vec::new(),
            equity_curve: curve,
        }
    }

    #[test]
    fn test_timeline_series_build_walk_forward() {
        let is_res = make_dummy_result(&[10_000.0, 10_200.0, 10_500.0]);
        let oos_res = make_dummy_result(&[10_500.0, 10_800.0]);
        let wf = WalkForwardResult::new("TestModel", "1h", is_res, oos_res, FitnessMetric::CalmarRatio);

        let series = TimelineSeries::build(&wf);
        assert_eq!(series.labels.len(), 5);
        assert_eq!(series.is_equity.len(), 5);
        assert_eq!(series.oos_equity.len(), 5);
        assert!(series.purge_equity.is_none());
    }

    #[test]
    fn test_timeline_series_build_full() {
        let is_res = make_dummy_result(&[10_000.0, 10_200.0]);
        let oos_res = make_dummy_result(&[10_200.0, 10_600.0]);
        let comb_res = make_dummy_result(&[10_000.0, 10_200.0, 10_200.0, 10_200.0, 10_600.0]);

        let full = FullBacktestResult {
            is_result: is_res,
            oos_result: oos_res,
            combined_result: comb_res,
            purge_bars: 1,
            train_end_idx: 1,
            test_start_idx: 3,
            total_bars: 5,
        };

        let series = TimelineSeries::build_full(&full);
        assert_eq!(series.labels.len(), 5);
        assert!(series.purge_equity.is_some());
        assert!(series.purge_dd.is_some());
        let script = render_charts_script(&series);
        assert!(script.contains("Purga / Embargo"));
        assert!(script.contains("toggleEquityScale"));
        assert!(script.contains("'logarithmic'"));
        let containers = render_charts_containers();
        assert!(containers.contains("equityScaleToggle"));
    }
}

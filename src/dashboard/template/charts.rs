use chrono::{TimeZone, Utc};
use crate::backtest::{FullBacktestResult, WalkForwardResult};
use super::capital_mode_replay::CapitalModeReplayer;

pub struct TimelineSeries {
    pub labels: Vec<String>,
    pub base_capital: f64,
    pub is_equity: Vec<String>,
    pub oos_equity: Vec<String>,
    pub is_dd: Vec<String>,
    pub oos_dd: Vec<String>,
    pub purge_equity: Option<Vec<String>>,
    pub purge_dd: Option<Vec<String>>,
    pub pct_is_equity: Vec<String>,
    pub pct_oos_equity: Vec<String>,
    pub pct_is_dd: Vec<String>,
    pub pct_oos_dd: Vec<String>,
    pub pct_purge_equity: Option<Vec<String>>,
    pub pct_purge_dd: Option<Vec<String>>,
}

impl TimelineSeries {
    fn format_dd(dd_pct: f64) -> String {
        if !dd_pct.is_finite() || dd_pct < 1e-4 {
            "0.00".to_string()
        } else {
            format!("-{:.2}", dd_pct)
        }
    }

    fn format_eq(eq: f64) -> String {
        if eq.is_finite() {
            format!("{:.2}", eq.max(0.0))
        } else {
            "0.00".to_string()
        }
    }

    fn format_ts(ts: i64, idx: usize) -> String {
        let d = Utc
            .timestamp_millis_opt(ts)
            .single()
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| idx.to_string());
        format!("\"{}\"", d)
    }

    pub fn build(wf: &WalkForwardResult) -> Self {
        let is_init = CapitalModeReplayer::resolve_initial_capital(&wf.is_result, 10_000.0);
        let oos_init = CapitalModeReplayer::resolve_initial_capital(&wf.oos_result, is_init);

        let (nom_is, nom_peak) = CapitalModeReplayer::normalize_nominal_segment(
            &wf.is_result.equity_curve,
            is_init,
            is_init,
            is_init,
            is_init,
        );
        let (nom_oos, _) = CapitalModeReplayer::normalize_nominal_segment(
            &wf.oos_result.equity_curve,
            oos_init,
            is_init,
            nom_is.final_capital,
            nom_peak,
        );

        let (pct_is, pct_peak) = CapitalModeReplayer::replay_percentual_segment(
            &wf.is_result,
            is_init,
            is_init,
            is_init,
        );
        let (pct_oos, _) = CapitalModeReplayer::replay_percentual_segment(
            &wf.oos_result,
            oos_init,
            pct_is.final_capital,
            pct_peak,
        );

        let n_is = wf.is_result.equity_curve.len();
        let n_oos = wf.oos_result.equity_curve.len();
        let total = n_is + n_oos;

        let mut labels = Vec::with_capacity(total);
        let mut is_equity = Vec::with_capacity(total);
        let mut oos_equity = Vec::with_capacity(total);
        let mut is_dd = Vec::with_capacity(total);
        let mut oos_dd = Vec::with_capacity(total);
        let mut pct_is_equity = Vec::with_capacity(total);
        let mut pct_oos_equity = Vec::with_capacity(total);
        let mut pct_is_dd = Vec::with_capacity(total);
        let mut pct_oos_dd = Vec::with_capacity(total);

        for (i, p) in wf.is_result.equity_curve.iter().enumerate() {
            labels.push(Self::format_ts(p.timestamp, p.candle_idx));
            is_equity.push(Self::format_eq(nom_is.equity[i]));
            is_dd.push(Self::format_dd(nom_is.drawdown_pct[i]));
            pct_is_equity.push(Self::format_eq(pct_is.equity[i]));
            pct_is_dd.push(Self::format_dd(pct_is.drawdown_pct[i]));

            if i + 1 == n_is && n_oos > 0 {
                oos_equity.push(Self::format_eq(nom_is.equity[i]));
                oos_dd.push(Self::format_dd(nom_is.drawdown_pct[i]));
                pct_oos_equity.push(Self::format_eq(pct_is.equity[i]));
                pct_oos_dd.push(Self::format_dd(pct_is.drawdown_pct[i]));
            } else {
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());
                pct_oos_equity.push("null".to_string());
                pct_oos_dd.push("null".to_string());
            }
        }

        for (j, p) in wf.oos_result.equity_curve.iter().enumerate() {
            labels.push(Self::format_ts(p.timestamp, p.candle_idx));
            is_equity.push("null".to_string());
            is_dd.push("null".to_string());
            pct_is_equity.push("null".to_string());
            pct_is_dd.push("null".to_string());

            oos_equity.push(Self::format_eq(nom_oos.equity[j]));
            oos_dd.push(Self::format_dd(nom_oos.drawdown_pct[j]));
            pct_oos_equity.push(Self::format_eq(pct_oos.equity[j]));
            pct_oos_dd.push(Self::format_dd(pct_oos.drawdown_pct[j]));
        }

        Self {
            labels,
            base_capital: is_init,
            is_equity,
            oos_equity,
            is_dd,
            oos_dd,
            purge_equity: None,
            purge_dd: None,
            pct_is_equity,
            pct_oos_equity,
            pct_is_dd,
            pct_oos_dd,
            pct_purge_equity: None,
            pct_purge_dd: None,
        }
    }

    pub fn build_full(full: &FullBacktestResult) -> Self {
        let curve = &full.combined_result.equity_curve;
        let n = curve.len();

        let is_init = CapitalModeReplayer::resolve_initial_capital(&full.is_result, 10_000.0);
        let oos_init = CapitalModeReplayer::resolve_initial_capital(&full.oos_result, is_init);

        let (nom_is, nom_peak) = CapitalModeReplayer::normalize_nominal_segment(
            &full.is_result.equity_curve,
            is_init,
            is_init,
            is_init,
            is_init,
        );
        let (nom_oos, _) = CapitalModeReplayer::normalize_nominal_segment(
            &full.oos_result.equity_curve,
            oos_init,
            is_init,
            nom_is.final_capital,
            nom_peak,
        );

        let (pct_is, pct_peak) = CapitalModeReplayer::replay_percentual_segment(
            &full.is_result,
            is_init,
            is_init,
            is_init,
        );
        let (pct_oos, _) = CapitalModeReplayer::replay_percentual_segment(
            &full.oos_result,
            oos_init,
            pct_is.final_capital,
            pct_peak,
        );

        let nom_is_last = nom_is.final_capital;
        let nom_is_last_dd = if nom_peak > 0.0 {
            ((nom_peak - nom_is_last).max(0.0) / nom_peak) * 100.0
        } else {
            0.0
        };
        let pct_is_last = pct_is.final_capital;
        let pct_is_last_dd = if pct_peak > 0.0 {
            ((pct_peak - pct_is_last).max(0.0) / pct_peak) * 100.0
        } else {
            0.0
        };

        let mut labels = Vec::with_capacity(n);
        let mut is_equity = Vec::with_capacity(n);
        let mut purge_equity = Vec::with_capacity(n);
        let mut oos_equity = Vec::with_capacity(n);
        let mut is_dd = Vec::with_capacity(n);
        let mut purge_dd = Vec::with_capacity(n);
        let mut oos_dd = Vec::with_capacity(n);

        let mut pct_is_equity = Vec::with_capacity(n);
        let mut pct_purge_equity = Vec::with_capacity(n);
        let mut pct_oos_equity = Vec::with_capacity(n);
        let mut pct_is_dd = Vec::with_capacity(n);
        let mut pct_purge_dd = Vec::with_capacity(n);
        let mut pct_oos_dd = Vec::with_capacity(n);

        for (i, p) in curve.iter().enumerate() {
            labels.push(Self::format_ts(p.timestamp, p.candle_idx));

            let (nom_eq_val, nom_dd_val, pct_eq_val, pct_dd_val) = if i < full.train_end_idx {
                let n_eq = nom_is.equity.get(i).copied().unwrap_or(0.0);
                let n_dd = nom_is.drawdown_pct.get(i).copied().unwrap_or(100.0);
                let p_eq = pct_is.equity.get(i).copied().unwrap_or(0.0);
                let p_dd = pct_is.drawdown_pct.get(i).copied().unwrap_or(100.0);
                (n_eq, n_dd, p_eq, p_dd)
            } else if i < full.test_start_idx {
                (nom_is_last, nom_is_last_dd, pct_is_last, pct_is_last_dd)
            } else {
                let j = i - full.test_start_idx;
                let n_eq = nom_oos.equity.get(j).copied().unwrap_or(0.0);
                let n_dd = nom_oos.drawdown_pct.get(j).copied().unwrap_or(100.0);
                let p_eq = pct_oos.equity.get(j).copied().unwrap_or(0.0);
                let p_dd = pct_oos.drawdown_pct.get(j).copied().unwrap_or(100.0);
                (n_eq, n_dd, p_eq, p_dd)
            };

            let nom_eq_str = Self::format_eq(nom_eq_val);
            let nom_dd_str = Self::format_dd(nom_dd_val);
            let pct_eq_str = Self::format_eq(pct_eq_val);
            let pct_dd_str = Self::format_dd(pct_dd_val);

            if i < full.train_end_idx {
                is_equity.push(nom_eq_str);
                is_dd.push(nom_dd_str);
                purge_equity.push("null".to_string());
                purge_dd.push("null".to_string());
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());

                pct_is_equity.push(pct_eq_str);
                pct_is_dd.push(pct_dd_str);
                pct_purge_equity.push("null".to_string());
                pct_purge_dd.push("null".to_string());
                pct_oos_equity.push("null".to_string());
                pct_oos_dd.push("null".to_string());
            } else if i == full.train_end_idx {
                is_equity.push(nom_eq_str.clone());
                is_dd.push(nom_dd_str.clone());
                purge_equity.push(nom_eq_str);
                purge_dd.push(nom_dd_str);
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());

                pct_is_equity.push(pct_eq_str.clone());
                pct_is_dd.push(pct_dd_str.clone());
                pct_purge_equity.push(pct_eq_str);
                pct_purge_dd.push(pct_dd_str);
                pct_oos_equity.push("null".to_string());
                pct_oos_dd.push("null".to_string());
            } else if i < full.test_start_idx {
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push(nom_eq_str);
                purge_dd.push(nom_dd_str);
                oos_equity.push("null".to_string());
                oos_dd.push("null".to_string());

                pct_is_equity.push("null".to_string());
                pct_is_dd.push("null".to_string());
                pct_purge_equity.push(pct_eq_str);
                pct_purge_dd.push(pct_dd_str);
                pct_oos_equity.push("null".to_string());
                pct_oos_dd.push("null".to_string());
            } else if i == full.test_start_idx {
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push(nom_eq_str.clone());
                purge_dd.push(nom_dd_str.clone());
                oos_equity.push(nom_eq_str);
                oos_dd.push(nom_dd_str);

                pct_is_equity.push("null".to_string());
                pct_is_dd.push("null".to_string());
                pct_purge_equity.push(pct_eq_str.clone());
                pct_purge_dd.push(pct_dd_str.clone());
                pct_oos_equity.push(pct_eq_str);
                pct_oos_dd.push(pct_dd_str);
            } else {
                is_equity.push("null".to_string());
                is_dd.push("null".to_string());
                purge_equity.push("null".to_string());
                purge_dd.push("null".to_string());
                oos_equity.push(nom_eq_str);
                oos_dd.push(nom_dd_str);

                pct_is_equity.push("null".to_string());
                pct_is_dd.push("null".to_string());
                pct_purge_equity.push("null".to_string());
                pct_purge_dd.push("null".to_string());
                pct_oos_equity.push(pct_eq_str);
                pct_oos_dd.push(pct_dd_str);
            }
        }

        Self {
            labels,
            base_capital: is_init,
            is_equity,
            oos_equity,
            is_dd,
            oos_dd,
            purge_equity: Some(purge_equity),
            purge_dd: Some(purge_dd),
            pct_is_equity,
            pct_oos_equity,
            pct_is_dd,
            pct_oos_dd,
            pct_purge_equity: Some(pct_purge_equity),
            pct_purge_dd: Some(pct_purge_dd),
        }
    }
}

pub fn render_charts_containers() -> &'static str {
    r#"
    <div class="chart-box">
        <div class="chart-header">
            <div class="chart-title">📈 Curva de Balance Continua: In-Sample (Azul Claro) ➔ Purga (Ámbar) ➔ Out-of-Sample (Azul Francia)</div>
            <div class="chart-controls">
                <div id="capitalModeSubtitle" class="chart-legend-custom">Base normalizada (Nominal fijo · t &le; T)</div>
                <button id="capitalModeToggle" class="scale-toggle-btn" type="button" aria-pressed="false" onclick="toggleCapitalMode()">Capital: Nominal</button>
                <button id="equityScaleToggle" class="scale-toggle-btn" type="button" aria-pressed="false" onclick="toggleEquityScale()">Escala Log</button>
            </div>
        </div>
        <div class="chart-wrapper">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

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
    let base_capital = series.base_capital;
    let labels_str = series.labels.join(",");
    let is_eq_str = series.is_equity.join(",");
    let oos_eq_str = series.oos_equity.join(",");
    let is_dd_str = series.is_dd.join(",");
    let oos_dd_str = series.oos_dd.join(",");

    let pct_is_eq_str = series.pct_is_equity.join(",");
    let pct_oos_eq_str = series.pct_oos_equity.join(",");
    let pct_is_dd_str = series.pct_is_dd.join(",");
    let pct_oos_dd_str = series.pct_oos_dd.join(",");

    let has_purge = series.purge_equity.is_some();

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

    let nom_purge_js = if let Some(ref p_eq) = series.purge_equity {
        format!("[{}]", p_eq.join(","))
    } else {
        "null".to_string()
    };
    let pct_purge_js = if let Some(ref p_eq) = series.pct_purge_equity {
        format!("[{}]", p_eq.join(","))
    } else {
        "null".to_string()
    };
    let nom_purge_dd_js = if let Some(ref p_dd) = series.purge_dd {
        format!("[{}]", p_dd.join(","))
    } else {
        "null".to_string()
    };
    let pct_purge_dd_js = if let Some(ref p_dd) = series.pct_purge_dd {
        format!("[{}]", p_dd.join(","))
    } else {
        "null".to_string()
    };

    format!(
        r#"
    <script>
        const labels = [{labels_str}];
        const baseCapital = {base_capital:.2};
        const hasPurge = {has_purge};

        const nominalEquityDatasets = [
            [{is_eq_str}],
            ...(hasPurge ? [{nom_purge_js}] : []),
            [{oos_eq_str}]
        ];
        const percentualEquityDatasets = [
            [{pct_is_eq_str}],
            ...(hasPurge ? [{pct_purge_js}] : []),
            [{pct_oos_eq_str}]
        ];

        const nominalDdDatasets = [
            [{is_dd_str}],
            ...(hasPurge ? [{nom_purge_dd_js}] : []),
            [{oos_dd_str}]
        ];
        const percentualDdDatasets = [
            [{pct_is_dd_str}],
            ...(hasPurge ? [{pct_purge_dd_js}] : []),
            [{pct_oos_dd_str}]
        ];

        const idxIs = 0;
        const idxOos = hasPurge ? 2 : 1;
        const idxLineIs = idxOos + 1;
        const idxLineOos = idxOos + 2;
        const idxLineGlobal = idxOos + 3;

        function findEndpoints(arr, requirePositive) {{
            let first = null;
            let last = null;
            for (let i = 0; i < arr.length; i++) {{
                const v = arr[i];
                if (v !== null && Number.isFinite(v) && (!requirePositive || v > 0)) {{
                    if (first === null) first = {{ idx: i, val: v }};
                    last = {{ idx: i, val: v }};
                }}
            }}
            return {{ first, last }};
        }}

        function makeLineData(len, p1, p2) {{
            const out = new Array(len).fill(null);
            if (p1 && p2) {{
                out[p1.idx] = p1.val;
                out[p2.idx] = p2.val;
            }}
            return out;
        }}

        function computeYBounds(arrays, isLog) {{
            let minV = Infinity;
            let maxV = -Infinity;
            for (const arr of arrays) {{
                for (const v of arr) {{
                    if (v !== null && Number.isFinite(v) && (!isLog || v > 0)) {{
                        if (v < minV) minV = v;
                        if (v > maxV) maxV = v;
                    }}
                }}
            }}
            if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {{
                return {{ min: undefined, max: undefined }};
            }}
            if (isLog) {{
                return {{
                    min: Math.max(minV * 0.96, 1e-6),
                    max: maxV * 1.04
                }};
            }}
            const span = maxV - minV;
            const pad = span > 1e-9 ? span * 0.05 : Math.max(Math.abs(maxV) * 0.05, 1.0);
            return {{
                min: minV - pad,
                max: maxV + pad
            }};
        }}

        const initBounds = computeYBounds(nominalEquityDatasets, false);
        const initIsEp = findEndpoints(nominalEquityDatasets[idxIs], false);
        const initOosEp = findEndpoints(nominalEquityDatasets[idxOos], false);

        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctxEquity, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'In-Sample (IS - Entrenamiento)',
                        data: nominalEquityDatasets[idxIs].slice(),
                        borderColor: '#60a5fa',
                        backgroundColor: 'rgba(96, 165, 250, 0.07)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.05,
                        pointRadius: 0
                    }}{purge_eq_dataset},
                    {{
                        label: 'Out-of-Sample (OOS - Validación Ciega)',
                        data: nominalEquityDatasets[idxOos].slice(),
                        borderColor: '#0055FF',
                        backgroundColor: 'rgba(0, 85, 255, 0.16)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.05,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Recta IS (Inicio ➔ Fin)',
                        data: makeLineData(labels.length, initIsEp.first, initIsEp.last),
                        borderColor: '#93c5fd',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        borderDash: [6, 4],
                        fill: false,
                        spanGaps: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#93c5fd'
                    }},
                    {{
                        label: 'Recta OOS (Inicio ➔ Fin)',
                        data: makeLineData(labels.length, initOosEp.first, initOosEp.last),
                        borderColor: '#38bdf8',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        borderDash: [6, 4],
                        fill: false,
                        spanGaps: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#38bdf8'
                    }},
                    {{
                        label: 'Recta Global (IS Inicio ➔ OOS Fin)',
                        data: makeLineData(labels.length, initIsEp.first, initOosEp.last),
                        borderColor: '#f0f4ff',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        fill: false,
                        spanGaps: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#f0f4ff'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{
                    legend: {{ labels: {{ color: '#8a94a6', font: {{ size: 12 }} }} }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                const v = ctx.parsed.y;
                                if (v === null || !Number.isFinite(v)) return null;
                                const pct = baseCapital > 0 ? ((v - baseCapital) / baseCapital) * 100 : 0;
                                const sign = pct >= 0 ? '+' : '';
                                return ctx.dataset.label + ': $' + Number(v).toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) + ' (' + sign + pct.toFixed(2) + '%)';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#8a94a6', maxTicksLimit: 14 }}, grid: {{ color: '#121826' }} }},
                    y: {{
                        beginAtZero: false,
                        min: initBounds.min,
                        max: initBounds.max,
                        ticks: {{
                            color: '#8a94a6',
                            callback: function(value) {{
                                return '$' + Number(value).toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
                            }}
                        }},
                        grid: {{ color: '#121826' }}
                    }}
                }}
            }}
        }});

        const ctxDd = document.getElementById('ddChart').getContext('2d');
        const ddChart = new Chart(ctxDd, {{
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

        let equityScaleMode = 'linear';
        let capitalMode = 'nominal';
        const scaleToggleBtn = document.getElementById('equityScaleToggle');
        const capitalModeBtn = document.getElementById('capitalModeToggle');
        const capitalSubtitle = document.getElementById('capitalModeSubtitle');

        function activeEquitySource() {{
            return capitalMode === 'percentual' ? percentualEquityDatasets : nominalEquityDatasets;
        }}

        function activeDdSource() {{
            return capitalMode === 'percentual' ? percentualDdDatasets : nominalDdDatasets;
        }}

        function hasPositiveValues(datasets) {{
            return datasets.some(arr => arr.some(v => v !== null && Number.isFinite(v) && v > 0));
        }}

        function syncCharts() {{
            const eqSrc = activeEquitySource();
            const ddSrc = activeDdSource();
            const hasPositive = hasPositiveValues(eqSrc);

            if (scaleToggleBtn) {{
                scaleToggleBtn.disabled = !hasPositive;
                if (!hasPositive) {{
                    scaleToggleBtn.title = 'Escala logarítmica no disponible sin valores positivos';
                }} else {{
                    scaleToggleBtn.title = '';
                }}
            }}

            if (!hasPositive && equityScaleMode === 'logarithmic') {{
                equityScaleMode = 'linear';
            }}

            const toLog = equityScaleMode === 'logarithmic';

            for (let idx = 0; idx < eqSrc.length; idx++) {{
                const raw = eqSrc[idx];
                const ds = equityChart.data.datasets[idx];
                ds.data = toLog
                    ? raw.map(v => (v !== null && Number.isFinite(v) && v > 0) ? v : null)
                    : raw.slice();
                ds.fill = toLog ? 'start' : true;
            }}

            const isEp = findEndpoints(eqSrc[idxIs], toLog);
            const oosEp = findEndpoints(eqSrc[idxOos], toLog);
            equityChart.data.datasets[idxLineIs].data = makeLineData(labels.length, isEp.first, isEp.last);
            equityChart.data.datasets[idxLineOos].data = makeLineData(labels.length, oosEp.first, oosEp.last);
            equityChart.data.datasets[idxLineGlobal].data = makeLineData(labels.length, isEp.first, oosEp.last);

            const bounds = computeYBounds(eqSrc, toLog);
            equityChart.options.scales.y.type = equityScaleMode;
            equityChart.options.scales.y.beginAtZero = false;
            equityChart.options.scales.y.min = bounds.min;
            equityChart.options.scales.y.max = bounds.max;
            equityChart.options.scales.y.ticks = {{
                color: '#8a94a6',
                callback: function(value) {{
                    return '$' + Number(value).toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
                }}
            }};
            equityChart.update();

            for (let idx = 0; idx < ddSrc.length; idx++) {{
                ddChart.data.datasets[idx].data = ddSrc[idx].slice();
            }}
            ddChart.update();

            if (scaleToggleBtn) {{
                scaleToggleBtn.textContent = toLog ? 'Escala Lineal' : 'Escala Log';
                scaleToggleBtn.classList.toggle('active', toLog);
                scaleToggleBtn.setAttribute('aria-pressed', toLog ? 'true' : 'false');
            }}

            const isPct = capitalMode === 'percentual';
            if (capitalModeBtn) {{
                capitalModeBtn.textContent = isPct ? 'Capital: Percentual' : 'Capital: Nominal';
                capitalModeBtn.classList.toggle('active', isPct);
                capitalModeBtn.setAttribute('aria-pressed', isPct ? 'true' : 'false');
            }}
            if (capitalSubtitle) {{
                capitalSubtitle.textContent = isPct
                    ? 'Replay compuesto (fracción de capital total · t ≤ T)'
                    : 'Base normalizada (Nominal fijo · t ≤ T)';
            }}
        }}

        function toggleEquityScale() {{
            if (!hasPositiveValues(activeEquitySource())) return;
            equityScaleMode = equityScaleMode === 'linear' ? 'logarithmic' : 'linear';
            syncCharts();
        }}

        function toggleCapitalMode() {{
            capitalMode = capitalMode === 'nominal' ? 'percentual' : 'nominal';
            syncCharts();
        }}

        syncCharts();
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
            .map(|(i, &eq)| {
                EquityPoint::new(
                    1_700_000_000_000 + (i as i64 * 3600_000),
                    i,
                    eq,
                    eq,
                    0.0,
                    0.0,
                )
            })
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
        assert_eq!(series.pct_is_equity.len(), 5);
        assert_eq!(series.pct_oos_equity.len(), 5);
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
        assert!(series.pct_purge_equity.is_some());
        assert!(series.pct_purge_dd.is_some());
        let script = render_charts_script(&series);
        assert!(script.contains("Purga / Embargo"));
        assert!(script.contains("toggleEquityScale"));
        assert!(script.contains("toggleCapitalMode"));
        assert!(script.contains("Recta IS (Inicio ➔ Fin)"));
        assert!(script.contains("Recta OOS (Inicio ➔ Fin)"));
        assert!(script.contains("Recta Global (IS Inicio ➔ OOS Fin)"));
        assert!(script.contains("'logarithmic'"));
        let containers = render_charts_containers();
        assert!(containers.contains("equityScaleToggle"));
        assert!(containers.contains("capitalModeToggle"));
    }
}

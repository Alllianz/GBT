/// Retorna la hoja de estilos CSS responsive optimizada para tema oscuro institucional
pub fn get_styles() -> &'static str {
    r#"
        :root {
            --bg-color: #000000;
            --card-bg: #050505;
            --card-header: #0a0a0a;
            --border-color: #141b2d;
            --text-primary: #f0f4ff;
            --text-secondary: #8a94a6;
            --accent-is: #3b82f6;
            --accent-oos: #0055ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .header h1 { margin: 0; font-size: 24px; font-weight: 600; }
        .header .badge {
            background: #050505;
            border: 1px solid rgba(0, 85, 255, 0.45);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            color: #2e7dff;
        }

        /* Banner de Métrica Fitness */
        .fitness-banner {
            display: flex;
            align-items: center;
            background: linear-gradient(90deg, #000000 0%, #050b1a 100%);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-oos);
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 24px;
            gap: 16px;
        }
        .fitness-icon { font-size: 28px; }
        .fitness-details { flex: 1; }
        .fitness-title { font-size: 15px; font-weight: 600; color: var(--text-primary); }
        .fitness-formula {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
            font-family: monospace;
            background: #000000;
            border: 1px solid var(--border-color);
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        .fitness-scores { display: flex; gap: 12px; }
        .fitness-pill {
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            text-align: center;
        }
        .fitness-pill.is {
            background: rgba(59, 130, 246, 0.14);
            color: var(--accent-is);
            border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .fitness-pill.oos {
            background: rgba(0, 85, 255, 0.18);
            color: #2e7dff;
            border: 1px solid rgba(0, 85, 255, 0.45);
        }

        /* Barra de comparación rápida IS vs OOS */
        .comparison-bar {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }
        .period-card {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 18px;
            border: 1px solid var(--border-color);
        }
        .period-card.is { border-left: 5px solid var(--accent-is); }
        .period-card.oos { border-left: 5px solid var(--accent-oos); }
        .period-card h3 { margin: 0 0 12px 0; font-size: 16px; display: flex; justify-content: space-between; }
        .period-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        .p-metric .lbl { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; }
        .p-metric .val { font-size: 18px; font-weight: bold; margin-top: 3px; }

        /* Contenedores de Gráficos */
        .chart-box {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            gap: 12px;
            flex-wrap: wrap;
        }
        .chart-title { font-size: 16px; font-weight: 600; margin: 0; }
        .chart-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .chart-legend-custom { font-size: 12px; color: var(--text-secondary); }
        .scale-toggle-btn {
            background: var(--card-header);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .scale-toggle-btn:hover:not(:disabled) {
            color: var(--text-primary);
            border-color: var(--accent-oos);
        }
        .scale-toggle-btn.active {
            background: rgba(0, 85, 255, 0.18);
            color: #2e7dff;
            border-color: var(--accent-oos);
        }
        .scale-toggle-btn:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .chart-wrapper { height: 380px; position: relative; }
        .chart-wrapper-small { height: 180px; position: relative; }

        /* Tabla Comparativa Exhaustiva */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
        }
        .metrics-table th, .metrics-table td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        .metrics-table th { background: var(--card-header); color: var(--text-secondary); }
        .metrics-table tr:hover { background: rgba(0, 85, 255, 0.08); }
        .metrics-table .num { text-align: right; }

        /* Pestañas de Trades */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        .tab-btn {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }
        .tab-btn.active {
            background: rgba(0, 85, 255, 0.18);
            color: var(--text-primary);
            border-color: var(--accent-oos);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .trades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        .trades-table th, .trades-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        .trades-table th { background: var(--card-header); color: var(--text-secondary); }
        .trades-table tr:hover { background: rgba(0, 85, 255, 0.08); }

        .badge.long { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); padding: 2px 6px; border-radius: 4px; }
        .badge.short { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); padding: 2px 6px; border-radius: 4px; }
        .green { color: var(--accent-green); }
        .red { color: var(--accent-red); }
        .purple { color: var(--accent-is); }
        .blue { color: var(--accent-oos); }
        .yellow { color: var(--accent-yellow); }
    "#
}

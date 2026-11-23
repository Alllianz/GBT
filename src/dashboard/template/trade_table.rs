use chrono::{TimeZone, Utc};
use crate::backtest::WalkForwardResult;
use crate::types::CompletedTrade;
use super::escape::escape_html;

pub fn render_trade_rows(trades: &[CompletedTrade]) -> String {
    let mut rows = String::new();
    for (idx, t) in trades.iter().enumerate() {
        let entry_d = Utc
            .timestamp_millis_opt(t.entry_timestamp)
            .single()
            .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();
        let exit_d = Utc
            .timestamp_millis_opt(t.exit_timestamp)
            .single()
            .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();

        let pnl_class = if t.net_pnl >= 0.0 { "green" } else { "red" };
        let side_str = format!("{:?}", t.side);

        rows.push_str(&format!(
            "<tr>
                <td>#{}</td>
                <td><span class='badge {}'>{}</span></td>
                <td>{}</td>
                <td>{:.2}</td>
                <td>{}</td>
                <td>{:.2}</td>
                <td class='{}'>${:.2} ({:+.2}%)</td>
                <td>${:.2}</td>
                <td>{:.2}%</td>
                <td>{} barras</td>
                <td>{}</td>
            </tr>\n",
            idx + 1,
            side_str.to_lowercase(),
            side_str,
            entry_d,
            t.entry_price,
            exit_d,
            t.exit_price,
            pnl_class,
            t.net_pnl,
            t.return_pct,
            t.commissions,
            t.max_adverse_excursion,
            t.duration_bars,
            escape_html(&t.exit_reason)
        ));
    }
    rows
}

pub fn render_trades_section(wf: &WalkForwardResult) -> String {
    let is_trades_count = wf.is_result.trades.len();
    let oos_trades_count = wf.oos_result.trades.len();

    let is_rows = render_trade_rows(&wf.is_result.trades);
    let oos_rows = render_trade_rows(&wf.oos_result.trades);

    format!(
        r#"
    <!-- Pestañas para ver Trades IS o Trades OOS -->
    <div style="margin-top: 32px;">
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('oos-tab', this)">Ver Trades Out-of-Sample (OOS) [{oos_trades_count}]</button>
            <button class="tab-btn" onclick="switchTab('is-tab', this)">Ver Trades In-Sample (IS) [{is_trades_count}]</button>
        </div>

        <div id="oos-tab" class="tab-content active">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Lado</th>
                        <th>Fecha Entrada</th>
                        <th>Precio Entrada</th>
                        <th>Fecha Salida</th>
                        <th>Precio Salida</th>
                        <th>PnL Neto</th>
                        <th>Comisión</th>
                        <th>MAE %</th>
                        <th>Duración</th>
                        <th>Motivo Cierre</th>
                    </tr>
                </thead>
                <tbody>
                    {oos_rows}
                </tbody>
            </table>
        </div>

        <div id="is-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Lado</th>
                        <th>Fecha Entrada</th>
                        <th>Precio Entrada</th>
                        <th>Fecha Salida</th>
                        <th>Precio Salida</th>
                        <th>PnL Neto</th>
                        <th>Comisión</th>
                        <th>MAE %</th>
                        <th>Duración</th>
                        <th>Motivo Cierre</th>
                    </tr>
                </thead>
                <tbody>
                    {is_rows}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function switchTab(tabId, btn) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }}
    </script>
        "#
    )
}

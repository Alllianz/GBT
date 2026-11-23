# Briefing de Implementación: Subagente C (Ronda 5)
## Ámbito: Dashboard y Handlers de CLI (BUG-42, BUG-43, BUG-46)

### 1. BUG-42: Sanitización de Nombres de Archivo en `EquityCurveExporter`
- **Archivo:** `src/dashboard/equity_curve_exporter.rs`.
- **Acción:**
  - Implementar o reutilizar una función de sanitización de nombres:
    ```rust
    fn sanitize_filename(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
                _ => c,
            })
            .collect()
    }
    ```
  - En `save_json` y `save_csv`, aplicar `sanitize_filename(model_name)` antes de construir el nombre de archivo, evitando fallos con `ERROR_INVALID_NAME` en Windows cuando el modelo contenga `:`, `/`, etc.

### 2. BUG-43: Escape de Entidades HTML en `HtmlTemplate`
- **Archivo:** `src/dashboard/html_template.rs`.
- **Acción:**
  - Implementar una función de escape HTML:
    ```rust
    fn escape_html(input: &str) -> String {
        input
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
    ```
  - Aplicar `escape_html` sobre `model_name` y sobre `t.exit_reason` en la tabla de trades y títulos antes de interpolar en la plantilla HTML.

### 3. BUG-46: Validación de Solvencia de Capital antes de Iniciar OOS
- **Archivos:**
  - `src/cli/baseline_handler.rs`
  - `src/cli/cart_handler.rs`
  - `src/cli/forust_handler.rs`
  - `src/cli/treeboost_handler.rs`
  - `src/cli/random_forest_handler.rs`
  - `src/cli/native_gbdt_handler.rs`
- **Acción:**
  - En la transición In-Sample a Out-of-Sample:
    ```rust
    let oos_capital = if is_result.metrics.final_capital > 0.0 {
        is_result.metrics.final_capital
    } else {
        println!("  ⚠️ ALERTA: Capital IS agotado o insolvente (${:.2}). Reiniciando OOS con capital base (${:.2})", 
            is_result.metrics.final_capital, bt_cfg.initial_capital);
        bt_cfg.initial_capital
    };
    oos_cfg.initial_capital = oos_capital;
    ```
    Evita divisiones por cero (`0.0 / 0.0 = NaN`) y quiebra instantánea en la primera vela de OOS si IS sufrió una pérdida catastrófica.

### Verificación:
- Ejecutar `cargo check` y `cargo test --no-run` para asegurar compilación limpia (0 errores).

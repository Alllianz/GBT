# Reporte de Implementación: Subagente C (Ronda 5)

## Misión Asignada
Resolución de bugs de Dashboard y Handlers de CLI:
- **BUG-42:** Sanitización de Nombres de Archivo en `EquityCurveExporter`
- **BUG-43:** Escape de Entidades HTML en `HtmlTemplate`
- **BUG-46:** Validación de Solvencia de Capital antes de Iniciar OOS

---

## Modificaciones Realizadas

### 1. BUG-42: Sanitización de Nombres en `EquityCurveExporter`
- **Archivo:** `src/dashboard/equity_curve_exporter.rs`
- **Implementación:**
  - Se agregó la función pública `sanitize_filename(name: &str) -> String`, que convierte a minúsculas y reemplaza caracteres no válidos (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`, y `' '`) por `_`.
  - Se integró `sanitize_filename` en `save_json` y `save_csv` para limpiar `model_name`, `timeframe` y `segment`.
  - Se añadieron pruebas unitarias `test_sanitize_filename`.

### 2. BUG-43: Escape de Entidades HTML en `HtmlTemplate`
- **Archivo:** `src/dashboard/html_template.rs`
- **Implementación:**
  - Se implementó la función pública `escape_html(input: &str) -> String` que reemplaza `&`, `<`, `>`, `"`, y `'` por sus entidades HTML seguras (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&#39;`).
  - Se aplicó `escape_html` sobre `exit_reason` en la tabla de trades (IS y OOS), y sobre `model_name`, `timeframe`, `fitness_name` y `fitness_formula` al formatear la plantilla.
  - Se removió un fragmento de CSS huérfano/inválido presente en la plantilla.
  - Se añadieron pruebas unitarias `test_escape_html`.
  - Se corrigió el conteo de caracteres en el test existente de `dashboard_generator.rs`.

### 3. BUG-46: Validación de Solvencia de Capital antes de OOS
- **Archivos:**
  - `src/cli/baseline_handler.rs`
  - `src/cli/cart_handler.rs`
  - `src/cli/forust_handler.rs`
  - `src/cli/treeboost_handler.rs`
  - `src/cli/random_forest_handler.rs`
  - `src/cli/native_gbdt_handler.rs`
- **Implementación:**
  - En la transición IS -> OOS de cada handler, se agregó la comprobación de que `is_result.metrics.final_capital > 0.0`.
  - Si el capital final es `<= 0.0` (quiebra o balance nulo), se emite una advertencia en consola y se reinicia el capital inicial de OOS con `bt_cfg.initial_capital`.
  - Esto previene divisiones por cero (`0.0 / 0.0 = NaN`), métricas degradadas con valores inválidos y quiebras artificiales inmediatas en OOS.

---

## Verificación
- `cargo check`: Completado exitosamente con código 0 (sin errores ni advertencias en los módulos modificados).
- `cargo test --no-run`: Compiló exitosamente todas las pruebas unitarias y ejecutables (`lib.rs` y `main.rs`).
- Pruebas unitarias de `dashboard` ejecutadas: 3 passed; 0 failed.

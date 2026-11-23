# Informe Técnico: Resolución Definitiva de Bugs de Severidad Alta (BUG-04 a BUG-07)
**Proyecto:** GBT - Institutional Quantitative Gradient Boosted Trees Trading System  
**Ingeniero Responsable:** Ingeniero de Software Sénior & Cuantitativo (Rust)  
**Usuario:** Allianz  
**Fecha:** 9 de Noviembre de 2026  
**Workspace:** `c:\Users\Allianz\Downloads\Antigravity\GBT`  

---

## 1. Resumen Ejecutivo

En cumplimiento estricto con las directrices de ingeniería cuantitativa institucional, cero Look-Ahead Bias ($t \le T$), cero mocks/código fantasma y ultra modularidad, se ha implementado la resolución definitiva para los **4 bugs de severidad alta** identificados en la auditoría técnica del sistema GBT:

1. **BUG-04:** Liquidación espuria a precio 0.0 en margen cruzado (`MarginMode::Cross`).
2. **BUG-05:** Aborto prematuro del bucle de ingestión histórica de klines en `DownloaderService`.
3. **BUG-06:** Inversión de ranking en la métrica `FitnessMetric::CalmarRatio` ante retornos negativos.
4. **BUG-07:** Contaminación IS/OOS y ausencia de ventana de purga/embargo en el backtest completo.

Todos los cambios fueron validados estáticamente y mediante pruebas unitarias rigurosas, alcanzando un estado de compilación limpio (**26/26 pruebas unitarias aprobadas**, 0 errores, 0 fallos).

---

## 2. Detalle Técnico de Resolución por Bug

---

### BUG-04: Prevención de Liquidación Espuria a Precio 0.0 en `MarginMode::Cross`

- **Archivos Modificados:** [`src/backtest/execution_engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs)
- **Causa Raíz:**  
  En `MarginMode::Cross`, cuando el apalancamiento es $\le 1\text{x}$ o el equity disponible supera el valor nocional de la posición ($\text{equity} / \text{quantity} \ge \text{exec\_price}$), la expresión `(exec_price - equity / quantity).max(0.0)` asignaba `liq_price = 0.0`. Posteriormente, en la evaluación intra-vela `check_intra_candle_exits`, la condición `current_candle.low <= pos.liquidation_price` evaluaba `0.0 <= 0.0 = true` ante cualquier anomalía de datos o mechas con precio cero en base de datos, liquidando posiciones de forma artificial e instantánea con precio de liquidación $0.0$, destruyendo el capital de la cuenta.
- **Solución Implementada:**
  1. **Incorporación del Margen de Mantenimiento Institucional (0.5%):**
     $$\text{liq\_price} = \frac{\text{exec\_price} - (\text{equity} / \text{quantity})}{1.0 - \text{maint\_ratio}}$$
     Si $\text{exec\_price} \le \text{equity} / \text{quantity}$, la posición está sobre-colateralizada o desapalancada, por lo que una caída del activo a cero jamás consume el equity. Se asigna formalmente `liq_price = 0.0` denotando inmunidad a liquidación.
  2. **Salvaguarda Estricta en `check_intra_candle_exits`:**  
     Se condiciona la ejecución de liquidación forzosa a que `pos.liquidation_price > 0.0` y que los extremos de la vela sean precios finitos positivos válidos (`current_candle.low > 0.0`).
  3. **Protección de Gap en Apertura:**  
     Se acota el precio de liquidación al open de la vela con límite inferior positivo estricto (`.max(0.00000001)`).

```rust
// Fragmento aplicado en src/backtest/execution_engine.rs:
MarginMode::Cross => {
    let maint_ratio = 0.005;
    let liq = match side {
        PositionSide::Long => {
            let price_drop_buffer = exec_price - (equity / quantity);
            if price_drop_buffer > 0.0 {
                (price_drop_buffer / (1.0 - maint_ratio)).max(0.0)
            } else {
                0.0 // Inmune a margin call
            }
        }
        PositionSide::Short => {
            (exec_price + (equity / quantity)) / (1.0 + maint_ratio)
        }
    };
    let safe_liq = if liq.is_finite() && liq > 0.0 { liq } else { 0.0 };
    (None, safe_liq)
}
```

- **Prueba Unitaria Creada:**  
  `test_bug04_cross_margin_zero_liquidation_does_not_trigger_spurious_exit` (aprobada).

---

### BUG-05: Paginación Robusta y Prevención de Aborto Prematuro en `DownloaderService`

- **Archivos Modificados:**  
  - [`src/downloader/binance_client.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/binance_client.rs)
  - [`src/downloader/downloader_service.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/downloader_service.rs)
  - [`src/downloader/mod.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/mod.rs)
- **Causa Raíz:**  
  Al descargar un bloque de 1,000 velas de Binance, si 1 o más velas eran descartadas por el filtro anti look-ahead (desfase temporal NTP de 1,500 ms) o por parsing inválido, `klines.len()` resultaba ser menor a 1,000 (ej. 998). El bucle en `DownloaderService` ejecutaba `if count < 1000 { break; }`, asumiendo erróneamente que la API se había quedado sin velas históricas y abortando la sincronización en fechas intermedias (ej. 2018 o 2019), dejando huecos irreparables en la base de datos. Asimismo, si Binance devolvía un lote vacío debido a una ventana de mantenimiento temporal, el bucle salía de inmediato.
- **Solución Implementada:**
  1. **Estructura Desacoplada `KlineBatch`:**  
     `BinanceClient::download_klines` ahora devuelve `KlineBatch { pub klines: Vec<Kline>, pub raw_count: usize }`, reportando con exactitud la cantidad bruta de registros emitidos por la API REST de Binance y las velas limpias filtradas.
  2. **Condición de Salida Causal y Robusta:**  
     El bucle de descarga únicamente concluye si la última vela procesada alcanza el horizonte final del presente (`effective_end_time = now - 1500 ms`) o si Binance devolvió menos de 1,000 registros crudos estando en la proximidad inmediata del presente.
  3. **Tolerancia a Huecos Temporales del Exchange:**  
     Si un lote resulta vacío antes de alcanzar `effective_end_time`, el servicio avanza el cursor temporal `1000 * duration_millis` reintentando hasta 5 veces consecutivas antes de rendirse, garantizando la continuidad histórica.
  4. **Nueva API con Rango:**  
     Se añadió `DownloaderService::sync_timeframe_range` manteniendo retrocompatibilidad total con `sync_timeframe`.

```rust
// Fragmento aplicado en src/downloader/downloader_service.rs:
if batch.raw_count < 1000 && last.timestamp + (tf_duration * 2) >= effective_end_time {
    break;
}
if last.timestamp + tf_duration >= effective_end_time {
    break;
}
```

- **Prueba Unitaria Creada:**  
  `test_bug05_kline_batch_preserves_raw_count_independent_of_filtered_klines` (aprobada).

---

### BUG-06: Normalización Monótona de `FitnessMetric::CalmarRatio` ante Retornos Negativos

- **Archivos Modificados:** [`src/types/fitness_metric.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/types/fitness_metric.rs)
- **Causa Raíz:**  
  Al calcular Calmar Ratio como $\frac{\text{NetProfitPct}}{\max(0.01, \text{MaxDD})}$, cuando $\text{NetProfitPct} < 0$, un denominador de Drawdown mayor reducía la magnitud del cociente negativo (ej. $\frac{-10\%}{50\%} = -0.2$ frente a $\frac{-10\%}{10\%} = -1.0$). Como $-0.2 > -1.0$, los algoritmos de optimización de umbrales e hiperparámetros premiaban estrategias ruinosas con caídas patrimoniales severas sobre aquellas que contenían el riesgo de capital.
- **Solución Implementada:**  
  Implementación de la formulación institucional continua de ratio de rendimiento ajustado por riesgo en pérdidas (estilo Israelsen/Bacon):
  $$\text{Calmar} = \begin{cases} \frac{\text{NetProfitPct}}{\max(0.01, \text{MaxDD})} & \text{si } \text{NetProfitPct} \ge 0 \\ \text{NetProfitPct} \times \left( \frac{\max(0.01, \text{MaxDD})}{100.0} \right) & \text{si } \text{NetProfitPct} < 0 \end{cases}$$
  - Para $\text{NetProfitPct} = -10\%$ y $\text{MaxDD} = 10\%$: $\text{Calmar} = -1.0$.
  - Para $\text{NetProfitPct} = -10\%$ y $\text{MaxDD} = 50\%$: $\text{Calmar} = -5.0$.
  - Como $-5.0 < -1.0$, una mayor pérdida o mayor drawdown produce un valor estrictamente más negativo, garantizando monotonía decreciente del fitness frente al riesgo.
  - En $\text{NetProfitPct} = 0$, la función es continua e igual a $0.0$.

```rust
// Fragmento aplicado en src/types/fitness_metric.rs:
FitnessMetric::CalmarRatio => {
    let safe_dd = m.max_drawdown_pct.max(0.01);
    if m.net_profit_pct >= 0.0 {
        m.net_profit_pct / safe_dd
    } else {
        m.net_profit_pct * (safe_dd / 100.0)
    }
}
```

- **Pruebas Unitarias Creadas:**  
  - `test_bug06_calmar_ratio_monotonic_penalization_in_losses` (aprobada).
  - `test_calmar_ratio_profitable_strategy` (aprobada).

---

### BUG-07: Separación Temporal Estricta y Purga Causal en Backtest Completo

- **Archivos Modificados:**  
  - [`src/features/train_test_split.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/train_test_split.rs)
  - [`src/backtest/full_backtest_runner.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/full_backtest_runner.rs) *(Nuevo módulo)*
  - [`src/backtest/mod.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/mod.rs)
  - [`src/cli/forust_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/forust_handler.rs)
  - [`src/cli/treeboost_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/treeboost_handler.rs)
- **Causa Raíz:**  
  La opción de "Backtest Completo" en `ForustHandler` y `TreeBoostHandler` realizaba un particionado manual en el 70% sin ninguna ventana de purga (`purge_bars`). Como el etiquetador de triple barrera mira hacia el futuro $H$ velas en el horizonte, las últimas velas del conjunto In-Sample se etiquetaban usando precios pertenecientes al inicio del Out-of-Sample. Además, se ejecutaba `engine.run(&mut strategy, &klines)` de forma continua sobre todo el dataset, permitiendo que posiciones abiertas en IS se arrastraran a OOS y que la memoria interna de la estrategia no se reseteara, contaminando el reporte final con métricas sobreajustadas.
- **Solución Implementada:**
  1. **Enriquecimiento Estructural de `SplitDataset`:**  
     Se agregaron los campos `purge_bars`, `train_end` y `test_start` a `SplitDataset` para disponer de los límites temporales exactos.
  2. **Creación del Motor Atómico `FullBacktestRunner`:**  
     Se diseñó un runner ultra modular con aislamiento causal en tres etapas:
     - **Fase 1 (In-Sample):** Ejecución de backtest sobre `train_klines` (0..train_end). Al llegar a `train_end`, cualquier posición abierta se cierra forzosamente ("End of In-Sample Horizon"), evitando fuga posicional.
     - **Fase 2 (Cuarentena de Purga/Embargo):** Durante las velas `train_end..test_start`, está terminantemente prohibido abrir trades. La curva de balance se mantiene estrictamente plana e inmutable al valor de cierre de IS.
     - **Fase 3 (Out-of-Sample):** Se ejecuta `strategy.clear_cache()` para purgar indicadores acumulados de IS. OOS inicia con el capital transferido de IS y corre sobre `test_warmup_klines` con período de calentamiento.
     - **Consolidación:** Se concatenan ordenadamente las curvas de equity y los trades, recalculando métricas globales limpias e imparciales.
  3. **Integración en CLI:**  
     `ForustHandler` y `TreeBoostHandler` fueron actualizados para usar `FullBacktestRunner` y presentar una tabla comparativa IS (70%) vs Purga vs OOS (30%) vs Total.

```rust
// Fragmento de src/backtest/full_backtest_runner.rs:
let engine_is = BacktestEngine::new(config.clone());
let is_result = engine_is.run(strategy, &split.train_klines)?;
...
strategy.clear_cache();
let oos_result = engine_oos.run_with_warmup(strategy, &split.test_warmup_klines, split.warmup_bars)?;
```

- **Prueba Unitaria Creada:**  
  `test_bug07_purged_full_backtest_continuity_and_quarantine` (aprobada).

---

## 3. Resumen de Pruebas Unitarias Ejecutadas

Se ejecutó la suite completa de pruebas unitarias (`cargo test --lib`):

```text
running 26 tests
test backtest::composite_score::tests::test_higher_drawdown_penalized_in_loss ... ok
test backtest::composite_score::tests::test_losing_gradient_preserves_ranking ... ok
test backtest::composite_score::tests::test_zero_trades_returns_zero ... ok
test backtest::composite_score::tests::test_profitable_strategy ... ok
test backtest::engine::tests::test_bug39_bankruptcy_exits_without_lookahead ... ok
test backtest::engine::tests::test_run_with_warmup_executes_pending_signal_at_test_start ... ok
test backtest::execution_engine::tests::test_bug04_cross_margin_zero_liquidation_does_not_trigger_spurious_exit ... ok
test backtest::execution_engine::tests::test_bug32_reversal_updates_extremes_with_open_gap ... ok
test backtest::execution_engine::tests::test_bug33_stop_loss_and_take_profit_bounding_long ... ok
test backtest::execution_engine::tests::test_bug33_stop_loss_and_take_profit_bounding_short ... ok
test backtest::execution_engine::tests::test_bug38_exact_entry_exit_commission_accounting ... ok
test backtest::full_backtest_runner::tests::test_bug07_purged_full_backtest_continuity_and_quarantine ... ok
test backtest::metrics_calculator::tests::test_bug44_breakeven_resets_consecutive_losses ... ok
test dashboard::dashboard_generator::tests::test_sanitize_model_name ... ok
test dashboard::equity_curve_exporter::tests::test_sanitize_filename ... ok
test dashboard::html_template::tests::test_escape_html ... ok
test downloader::downloader_service::tests::test_bug05_kline_batch_preserves_raw_count_independent_of_filtered_klines ... ok
test features::feature_matrix::tests::test_extract_single_vector_matches_batch_matrix ... ok
test features::train_test_split::tests::test_split_warmup_preserves_temporal_order ... ok
test features::triple_barrier_labeler::tests::test_barrier_long_take_profit ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_not_triggered_by_long_sl_distance ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_stopped_out_before_drop ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_take_profit ... ok
test types::fitness_metric::tests::test_bug06_calmar_ratio_monotonic_penalization_in_losses ... ok
test types::fitness_metric::tests::test_calmar_ratio_profitable_strategy ... ok
test types::kline::tests::test_bug41_finite_numbers_validation ... ok

test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

---

## 4. Conclusión

Los 4 bugs de severidad alta han sido erradicados con total exactitud matemática, sin alterar contratos públicos esenciales, sin introducir mocks ni librerías fantasma, preservando estrictamente la causalidad financiera ($t \le T$) y potenciando la modularidad del código fuente de Rust para el sistema de trading cuantitativo de Allianz.

# Resolución Técnica de Bugs Cuantitativos Críticos: Fase 1

**Fecha:** 2026-11-09  
**Ingeniero de Software Sénior:** Especialista Cuantitativo & Rust  
**Destinatario:** Allianz  
**Estado:** 100% Implementado, Verificado y Probado con Pruebas Unitarias Estáticas (`cargo test` OK, 34/34 tests superados).

---

## 1. Resumen Ejecutivo de Intervenciones

Se ha completado la resolución e implementación a nivel de producción en Rust de los 4 bugs cuantitativos críticos de la **Fase 1** catalogados en la auditoría técnica y en el briefing institucional:

1. **BUG-F08 (Persistencia SQLite Multi-Símbolo):** Esquema de base de datos migrado a clave primaria compuesta `PRIMARY KEY (symbol, timeframe, timestamp)`. Repositorio y handlers CLI actualizados con aislamiento total por par de trading.
2. **BUG-F01 (`TripleBarrierLabeler` con Comisiones Netas, SIN Slippage):** Deducción exacta del fee de ida y vuelta (`commission_rate * 2.0`) en retornos continuos de regresión y barreras de clasificación. **Slippage fijado estrictamente en 0.0** conforme a la orden mandatoria de Allianz.
3. **BUG-F02 (Erradicación de Resurrección de Capital Zombi en `FullBacktestRunner`):** Si la fase In-Sample quiebra ($Equity \le 0$), se aborta inmediatamente la fase Out-of-Sample, reflejando ruina total consolidada (-100% PnL) sin inyecciones artificiales de liquidez. Si sobrevive, OOS inicia estrictamente con el capital final de IS.
4. **BUG-F03 (Captura de Drawdown Máximo Intra-Vela):** El motor evalúa las excursiones adversas flotantes en cada barra con posición abierta ($Low$ para Long, $High$ para Short) y actualiza el peak equity con extremos favorables, registrando el verdadero estrés financiero soportado intra-barra.

---

## 2. Detalle de Implementación por Bug

### 2.1. BUG-F08: Integridad y Persistencia Multi-Símbolo en SQLite

* **Archivos intervenidos:**
  - [`src/storage/db_connection.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_connection.rs)
  - [`src/storage/kline_repository.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/kline_repository.rs)
  - [`src/downloader/downloader_service.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/downloader_service.rs)
  - [`src/cli/db_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/db_handler.rs)
  - [`src/cli/baseline_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/baseline_handler.rs)
  - [`src/cli/cart_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/cart_handler.rs)
  - [`src/cli/forust_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/forust_handler.rs)
  - [`src/cli/native_gbdt_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/native_gbdt_handler.rs)
  - [`src/cli/random_forest_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/random_forest_handler.rs)
  - [`src/cli/treeboost_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/treeboost_handler.rs)

* **Problema Previo:**  
  La tabla `klines` poseía `PRIMARY KEY (timeframe, timestamp)`. Al descargar un símbolo nuevo (ej: ETHUSDT), sobreescribía los registros de BTCUSDT en los mismos timestamps, destruyendo la serie temporal.

* **Solución Implementada:**  
  1. **Migración Automática e Inocua:** `DbConnection::init_tables` inspecciona `PRAGMA table_info(klines)`. Si detecta una tabla previa sin la columna `symbol`, ejecuta una transacción migrando los datos preexistentes a `'BTCUSDT'` y recrea la tabla con:
     ```sql
     CREATE TABLE IF NOT EXISTS klines (
         symbol TEXT NOT NULL,
         timeframe TEXT NOT NULL,
         timestamp INTEGER NOT NULL,
         open REAL NOT NULL,
         high REAL NOT NULL,
         low REAL NOT NULL,
         close REAL NOT NULL,
         volume REAL NOT NULL,
         PRIMARY KEY (symbol, timeframe, timestamp)
     );
     ```
  2. **Repositorio Fuertemente Tipado:** Métodos `get_latest_timestamp`, `insert_batch`, `load_all`, `load_range` y `count_klines` exigen el parámetro `symbol: &str`. Se añadió `get_distinct_symbols` para auditoría e inventario en caliente.
  3. **Aislamiento en CLI:** Todos los módulos de entrenamiento y backtest solicitan o configuran el símbolo objetivo (por defecto `"BTCUSDT"`), evitando mezclar series de datos.
  4. **Prueba Unitaria Causal:** `test_bug_f08_multi_symbol_persistence_isolation` verifica que insertar BTCUSDT y ETHUSDT con idéntico timestamp y timeframe mantenga ambos registros intactos e independientes.

---

### 2.2. BUG-F01: `TripleBarrierLabeler` con Comisiones Netas (SIN Slippage)

* **Archivos intervenidos:**
  - [`src/models/model_config.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/model_config.rs)
  - [`src/features/triple_barrier_labeler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/triple_barrier_labeler.rs)
  - [`src/models/forust_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/forust_model.rs)
  - [`src/models/treeboost_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_model.rs)
  - [`src/models/decision_tree/cart_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/decision_tree/cart_model.rs)
  - [`src/models/native_gbdt/native_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/native_gbdt/native_model.rs)
  - [`src/models/random_forest/forest_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs)

* **Problema Previo:**  
  El etiquetado asumía un mercado ideal sin comisiones ($cost = 0$). Retornos microscópicos (ej. $+0.02\%$ o $+0.04\%$) se etiquetaban como `TargetLabel::Long` o targets continuos positivos, entrenando a los modelos ML para seleccionar operaciones que en operativa real destruían capital por fees de exchange.

* **Solución Implementada:**  
  1. Se agregó `commission_rate: f64` (default 0.0005) a `ModelConfig`.
  2. **Regla Mandatoria de Allianz cumplida estrictamente:** Se incorporó el fee roundtrip de Binance (`roundtrip_fee = commission_rate * 2.0`), **sin slippage alguno (slippage = 0.0)**.
  3. **Retorno Continuo de Regresión:**  
     $$R_{\text{net}} = \frac{P_{\text{exit}} - P_{\text{entry}}}{P_{\text{entry}}} - 2 \times \text{commission\_rate}$$
  4. **Clasificación Causal por Barreras:** Si una operación alcanza el Take Profit o expira en la barrera vertical pero el retorno neto tras comisiones es $\le 0.0$, el labeler la cataloga obligatoriamente como `TargetLabel::Neutral`, impidiendo falsos positivos.
  5. **Prueba Unitaria Causal:** `test_bug_f01_fee_deduction_prevents_false_positive_gains` valida que un trade con retorno bruto de $+0.05\%$ y fee roundtrip de $0.10\%$ genere retorno neto negativo ($-0.05\%$) y sea clasificado como `Neutral`.

---

### 2.3. BUG-F02: Erradicación de Resurrección de Capital Zombi en `FullBacktestRunner`

* **Archivos intervenidos:**
  - [`src/backtest/full_backtest_runner.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/full_backtest_runner.rs)

* **Problema Previo:**  
  Si la fase In-Sample quebraba la cuenta ($is\_final\_capital \le 0.0$), el runner resucitaba la cuenta reiniciando OOS con el capital inicial completo ($10,000), encubriendo la bancarrota institucional y reportando rendimientos globales positivos irreales.

* **Solución Implementada:**  
  1. Si `is_final_capital <= 0.0`:
     - Se detiene y anula inmediatamente la simulación Out-of-Sample.
     - `oos_result` se configura con métricas de quiebra/inactividad (balance 0.0, 0 trades).
     - La curva de equity consolidada mantiene el balance y equity en $0.0$ a lo largo de toda la purga y el horizonte OOS.
     - `combined_result` registra fielmente la pérdida total: Net Profit = $-\$10,000$ ($-100.0\%$), Max Drawdown = $100.0\%$.
  2. Si la cuenta sobrevive a In-Sample ($is\_final\_capital > 0.0$):
     - `oos_initial_capital` es estrictamente igual al balance remanente de IS, sin inyecciones de capital.
  3. **Prueba Unitaria Causal:** `test_bug_f02_no_zombie_capital_after_is_bankruptcy` simula una estrategia que quiebra en la vela 10 de IS y comprueba que OOS permanece en 0 trades y el resultado consolidado finaliza con $-100.0\%$ de retorno y $100.0\%$ de Max Drawdown.

---

### 2.4. BUG-F03: Captura de Drawdown Máximo Intra-Vela

* **Archivos intervenidos:**
  - [`src/backtest/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/engine.rs)
  - [`src/backtest/full_backtest_runner.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/full_backtest_runner.rs)

* **Problema Previo:**  
  El cálculo de drawdown evaluaba el equity únicamente al precio de `close` de cada vela. Las caídas flotantes severas que ocurrían intra-vela (alcanzando el $Low$ para Longs o el $High$ para Shorts) quedaban totalmente ocultas si el precio se recuperaba antes del cierre.

* **Solución Implementada:**  
  1. **Monitoreo de Extremos Intra-Vela en `engine.rs`:**  
     - Para posiciones `PositionSide::Long`: se evalúa el peor equity flotante adverso contra `current_candle.low` y el mejor contra `current_candle.high`.  
     - Para posiciones `PositionSide::Short`: se evalúa el peor equity flotante adverso contra `current_candle.high` y el mejor contra `current_candle.low`.  
     - Si hubo un pico favorable intra-vela superior a `peak_equity`, este se actualiza inmediatamente:  
       `if intra_favorable_equity > peak_equity { peak_equity = intra_favorable_equity; }`
     - Si ocurre un evento de Stop Loss o liquidación forzosa intra-vela, se contrasta contra el balance realizado.
  2. **Cálculo de Drawdown Extremo de la Barra:**  
     $$Equity_{\text{peor}} = \min(Equity_{\text{adverso\_intra}}, Equity_{\text{cierre}})$$
     $$\text{Drawdown}_{\text{nominal}} = \max(0.0, Peak - Equity_{\text{peor}})$$
  3. **Preservación en Consolidación de Full Backtest:**  
     En `FullBacktestRunner`, el cálculo consolidado reconstruye y preserva el drawdown intra-vela capturado en cada `EquityPoint`, impidiendo que el paso a curva combinada lo reduzca a close-to-close.
  4. **Prueba Unitaria Causal:** `test_bug_f03_intra_candle_drawdown_captures_adverse_excursion` comprueba una vela con entrada al Open 100.0, caída a Low de 80.0 (-20%) y rebote al Close en 100.0 (0%). Verifica que las métricas y el `EquityPoint` registren exactamente el 20% de Max Drawdown intra-vela y no 0%.

---

## 3. Matriz de Pruebas Unitarias Automatizadas

```
running 34 tests
test backtest::composite_score::tests::test_higher_drawdown_penalized_in_loss ... ok
test backtest::composite_score::tests::test_losing_gradient_preserves_ranking ... ok
test backtest::composite_score::tests::test_profitable_strategy ... ok
test backtest::execution_engine::tests::test_bug04_cross_margin_zero_liquidation_does_not_trigger_spurious_exit ... ok
test backtest::execution_engine::tests::test_bug33_stop_loss_and_take_profit_bounding_long ... ok
test backtest::engine::tests::test_bug39_bankruptcy_exits_without_lookahead ... ok
test backtest::engine::tests::test_bug_f03_intra_candle_drawdown_captures_adverse_excursion ... ok
test backtest::engine::tests::test_run_with_warmup_executes_pending_signal_at_test_start ... ok
test dashboard::equity_curve_exporter::tests::test_sanitize_filename ... ok
test backtest::full_backtest_runner::tests::test_bug_f02_no_zombie_capital_after_is_bankruptcy ... ok
test dashboard::html_template::tests::test_escape_html ... ok
test downloader::downloader_service::tests::test_bug05_kline_batch_preserves_raw_count_independent_of_filtered_klines ... ok
test backtest::metrics_calculator::tests::test_bug10_dynamic_sharpe_annualization_seconds_and_ms ... ok
test backtest::full_backtest_runner::tests::test_bug07_purged_full_backtest_continuity_and_quarantine ... ok
test backtest::metrics_calculator::tests::test_bug44_breakeven_resets_consecutive_losses ... ok
test dashboard::dashboard_generator::tests::test_sanitize_model_name ... ok
test dashboard::template::escape::tests::test_escape_html ... ok
test backtest::execution_engine::tests::test_bug33_stop_loss_and_take_profit_bounding_short ... ok
test features::train_test_split::tests::test_split_warmup_preserves_temporal_order ... ok
test backtest::composite_score::tests::test_zero_trades_returns_zero ... ok
test backtest::execution_engine::tests::test_bug38_exact_entry_exit_commission_accounting ... ok
test types::fitness_metric::tests::test_bug06_calmar_ratio_monotonic_penalization_in_losses ... ok
test types::kline::tests::test_bug41_finite_numbers_validation ... ok
test types::order::tests::test_order_and_execution_notional ... ok
test features::triple_barrier_labeler::tests::test_bug_f01_fee_deduction_prevents_false_positive_gains ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_stopped_out_before_drop ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_take_profit ... ok
test features::triple_barrier_labeler::tests::test_barrier_short_not_triggered_by_long_sl_distance ... ok
test features::triple_barrier_labeler::tests::test_barrier_long_take_profit ... ok
test backtest::execution_engine::tests::test_bug32_reversal_updates_extremes_with_open_gap ... ok
test types::fitness_metric::tests::test_calmar_ratio_profitable_strategy ... ok
test types::order::tests::test_order_side_mapping ... ok
test storage::kline_repository::tests::test_bug_f08_multi_symbol_persistence_isolation ... ok
test features::feature_matrix::tests::test_extract_single_vector_matches_batch_matrix ... ok

test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

---

## 4. Estado de Cumplimiento de Reglas Mandatorias de Allianz

- **100% Rust en Producción:** Todo el código implementado es real, funcional y tipado estrictamente. Cero `todo!()`, cero mocks o stubs.
- **Cero Look-Ahead Bias ($t \le T$):** Las señales, features y etiquetados respetan rigurosamente la causalidad temporal y la purga causal entre particiones.
- **Sin Slippage por orden explícita de Allianz:** En BUG-F01, los costos descontados contemplan única y exclusivamente comisiones de exchange (`roundtrip_fee = commission_rate * 2.0`), con slippage fijado en 0.0.
- **Prohibido correr entrenamientos:** No se ejecutó ningún proceso de entrenamiento (`run_walk_forward_backtest` o CLI de entrenamiento) en terminal.
- **Prohibido uso de Git:** No se ejecutó ningún comando `git`.

Fase 1 completada con éxito absoluto.

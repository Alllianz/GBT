# Briefing de Misión: Auditoría Forense y Búsqueda Exhaustiva de Bugs - Ronda 6

## Objetivo
Realizar una auditoría institucional exhaustiva, rigurosa e implacable sobre la totalidad del código fuente en Rust del proyecto `GBT` (`c:\Users\Allianz\Downloads\Antigravity\GBT`).
En las rondas 1 a 5 se han identificado y resuelto satisfactoriamente 46 bugs (BUG-01 a BUG-46).
Esta ronda 6 debe auditar en profundidad la integración de todos los componentes, la interacción de las soluciones previas y someter todo el codebase a estrés analítico para descubrir cualquier bug residual o certificar la integridad matemática, causal y de ingeniería de nivel institucional.

## Restricciones Absolutas
1. **PROHIBIDO ejecutar o probar entrenamientos** (`cargo run`, tests que ejecuten optimizaciones/entrenamientos). Solo se permite `cargo check` o `cargo test --no-run` para verificación estática.
2. **PROHIBIDO el uso de Git**.
3. **Cero Look-Ahead Bias ($t \le T$)**: Verificar escrupulosamente que ninguna serie temporal, indicador, etiqueta, feature o cálculo de balance acceda a información futura.
4. **Cero Código Fantasma**: No se toleran mocks ni funciones vacías.
5. **Ultra-Modularidad**: Estricta separación de responsabilidades.

## Áreas y Componentes a Auditar
1. **Motor de Backtesting y Ejecución (`src/backtest/`)**:
   - `execution_engine.rs`: Ejecución de Market/Limit/Stop orders, slippage, comisiones dobles (apertura y cierre), margen aislado vs cruzado, trailing stop, take-profit, stop-loss, MFE/MAE.
   - `engine.rs`: Bucle temporal, control estricto de causalidad ante bancarrota, cálculo de curva de equity, sincronización de índices IS/OOS.
   - `metrics_calculator.rs`: Sharpe, Sortino, Calmar, Max Drawdown, Composite Score, Profit Factor, rachas consecutivas, casos con varianza cero o 0 trades.

2. **Modelos de Machine Learning y Estrategias (`src/models/`)**:
   - `treeboost_model.rs` & `treeboost_strategy.rs`
   - `forust_model.rs` & `forust_strategy.rs`
   - `decision_tree/cart_model.rs` & `cart_strategy.rs`
   - `random_forest/forest_model.rs` & `forest_strategy.rs`
   - `native_gbdt/native_model.rs` & `native_strategy.rs`
   - `ensemble.rs`
   - Finitud de predicciones (`.is_finite()`), protección de `.clamp()`, cálculo de fuerza de señal, división por cero en umbrales, consistencia de serialización JSON.

3. **Ingeniería de Características y Etiquetado (`src/features/`)**:
   - `triple_barrier_labeler.rs`: Sincronización causal del horizonte temporal ($t \le T$), alineación de entrada al Open ($t+1$) y salida al Close ($t+H$), evaluación intra-bar de barreras superior e inferior.
   - `technical_indicators.rs`: ROC, EMA, RSI, ATR, SMA de volumen, determinismo temporal y causalidad.
   - `fractional_diff.rs`, `orderbook_features.rs`, `features_calculator.rs`.

4. **Descargador, Ingesta y Base de Datos (`src/downloader/`, `src/storage/`, `src/data/`)**:
   - `binance_client.rs`: Paginación, descarte de velas vivas, buffer de reloj NTP (1500 ms), rate limits.
   - `downloader_service.rs`: Sincronización continua, control de interrupción graceful.
   - `types/kline.rs`: Validación de finitud numérica ante NaN / Inf.
   - `storage/kline_repository.rs`: Transacciones SQLite, integridad referencial.

5. **Optimización, Walk-Forward, Dashboards y Handlers CLI (`src/optimization/`, `src/dashboard/`, `src/cli/`)**:
   - `dashboard_generator.rs`, `equity_curve_exporter.rs`, `html_template.rs`: Sanitización de rutas en Windows, escape seguro de entidades HTML.
   - `walk_forward.rs`, `hyperopt.rs`, `threshold_optimizer.rs`.
   - Handlers CLI (`baseline_handler.rs`, `cart_handler.rs`, `forust_handler.rs`, `treeboost_handler.rs`, `random_forest_handler.rs`, `native_gbdt_handler.rs`): Transición IS -> OOS, persistencia de métricas.

## Entregable Requerido
Escribir el informe exhaustivo en el archivo físico:
`docs/auditoria_sexta_ronda_bugs.md`

Incluir:
- Relación de componentes auditados y metodología.
- Lista detallada de nuevos bugs encontrados (BUG-47 en adelante), con severidad, archivo, líneas, causa raíz, impacto y recomendación.
- Si tras la auditoría exhaustiva de todos los módulos no se detectan nuevos bugs, redactar la **Certificación Formal de Integridad Institucional Cuantitativa**, detallando por qué el sistema es seguro y robusto en producción.

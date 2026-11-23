# Briefing de Misión: Auditoría Forense y Búsqueda Exhaustiva de Bugs - Ronda 5

## Objetivo
Realizar una auditoría institucional exhaustiva e implacable sobre la totalidad del código fuente en Rust del proyecto `GBT` (`c:\Users\Allianz\Downloads\Antigravity\GBT`).
En las rondas 1 a 4 se identificaron y resolvieron 37 bugs (BUG-01 a BUG-37). Esta ronda 5 debe auditar las interacciones de los cambios aplicados y someter todo el sistema a un escrutinio profundo para identificar cualquier bug remanente o certificar la integridad total.

## Restricciones Absolutas
1. **PROHIBIDO ejecutar o probar entrenamientos** (`cargo run`, tests que corran entrenamiento). Solo se permite `cargo check` o `cargo test --no-run` para verificación estática de sintaxis y tipos.
2. **PROHIBIDO el uso de Git**.
3. **Cero Look-Ahead Bias ($t \le T$)**: El sistema financiero no puede bajo ninguna circunstancia acceder a información del futuro ($t > T$).
4. **Cero Código Fantasma**: No se aceptan mocks ni estructuras vacías; todo debe ser 100% código funcional de producción.
5. **Ultra-Modularidad**: Estándar estricto de separación de responsabilidades.

## Áreas Críticas a Auditar
1. **Motor de Backtesting y Ejecución (`src/backtest/`)**:
   - `execution_engine.rs`: Lógica de ejecución de órdenes (Market, Limit, Stop), trailing stops, take-profit, stop-loss, liquidación, cálculo de margen aislado vs cruzado, comisiones, slippage, mark-to-market.
   - `engine.rs`: Secuencia temporal de velas, warmup period, evaluación OOS, actualización de curvas de equity al cierre del backtest.
   - `metrics_calculator.rs`: Sharpe, Sortino, Calmar, Max Drawdown, Composite Score, Profit Factor con casos de borde (0 operaciones, retornos constantes, varianza nula).

2. **Modelos de Machine Learning (`src/models/`)**:
   - `treeboost_model.rs`, `cart_model.rs`, `forest_model.rs`, `forust_model.rs`, `native_model.rs`, `ensemble.rs`.
   - Manejo de NaN, infinitos, features no finitas, off-by-one en divisiones de nodos (`min_samples_leaf`), determinismo, serialización/deserialización.

3. **Ingeniería de Características y Etiquetado (`src/features/`)**:
   - `technical_indicators.rs`, `triple_barrier_labeler.rs`, `fractional_diff.rs`, `orderbook_features.rs`, `features_calculator.rs`.
   - Prevención de fuga temporal: alineación entre el momento de generación de la señal y la ejecución real (ej. señal al cierre de $t$, ejecución en apertura de $t+1$).
   - Verificación de lags y horizontes temporales.

4. **Descargador e Ingesta de Datos (`src/downloader/`, `src/data/`)**:
   - `binance_client.rs`, `downloader_service.rs`, `historical_data.rs`.
   - Paginación de klines, gaps temporales, drifts de reloj (NTP), rate limits, manejo de errores de red y parseo.

5. **Optimización, Walk-Forward y CLI (`src/optimization/`, `src/cli/`, `src/dashboard/`)**:
   - `walk_forward.rs`, `hyperopt.rs`, `monte_carlo.rs`.
   - `handlers/`: `train_handler.rs`, `backtest_handler.rs`, `baseline_handler.rs`, `live_handler.rs`, `menu.rs`.
   - `dashboard_generator.rs`: Generación de reportes, sanitización de nombres de archivo y exportación de métricas.

## Entregable Requerido
Escribir el reporte completo en el archivo físico:
`docs/auditoria_quinta_ronda_bugs.md`

El reporte debe incluir:
- Metodología y subsistemas auditados.
- Lista detallada de bugs encontrados (si los hay), con ID (BUG-38, etc.), severidad, archivo, líneas, causa raíz, impacto financiero y propuesta de solución.
- Si no se detectan nuevos bugs, justificación detallada y exhaustiva de verificación por subsistema que certifique la solidez matemática, temporal y de concurrencia.

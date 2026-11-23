# Reporte de Auditoría Técnica y Diagnóstico de Bugs
**Proyecto:** GBT - Institutional Quantitative Gradient Boosted Trees Trading System  
**Auditor:** Auditor Técnico Sénior (Rust & Cuantitativo)  
**Usuario:** Allianz  
**Fecha:** 9 de Noviembre de 2026  
**Workspace:** `c:\Users\Allianz\Downloads\Antigravity\GBT`  

---

## 1. Resumen Ejecutivo

Se ha completado una auditoría exhaustiva y milimétrica del código fuente del sistema cuantitativo **GBT**. La revisión cubrió todos los submódulos del workspace: motor de backtesting (`src/backtest/`), ingeniería de variables (`src/features/`), modelos de machine learning (`src/models/`), capa de persistencia (`src/storage/`), servicio de ingestión de datos (`src/downloader/`), interfaz de línea de comandos (`src/cli/`), tablero de control (`src/dashboard/`) y concurrencia (`src/threading/`).

El sistema exhibe un diseño técnico avanzado con paralelización en Rayon (20 hilos), soporte para modelos institucionales (Forust-ML, TreeBoost AutoML, Native GBDT, Random Forest y Decision Tree CART), y estrictas reglas de no-acceso al futuro en la mayor parte de los cálculos temporales. No obstante, se han detectado **16 vulnerabilidades y bugs**, que van desde **inversiones matemáticas críticas en la función de fitness** hasta **asimetrías en el cálculo de features causales** y **riesgo de liquidación espuria por precios nulos**.

### Resumen de Hallazgos por Severidad

| Severidad | Cantidad | Descripción General |
| :--- | :---: | :--- |
| 🔴 **Crítica** | **3** | Inversión matemática en el score compuesto de pérdidas, asimetría de features entre entrenamiento e inferencia, y clasificación asimétrica errónea en Short trades. |
| 🟠 **Alta** | **4** | Riesgo de liquidación a precio $0.0 en Cross Margin, corte prematuro del downloader histórico, ratio Calmar invertido en rendimientos negativos, y contaminación IS/OOS en el backtest completo. |
| 🟡 **Media** | **5** | Opción `allow_short` inaccesible en CLI, inconsistencia de `Default` en `MarginMode`, asunción de unidad de tiempo en factor de anualización Sharpe, índice duplicado en SQLite, y código inalcanzable de fallback en estrategias. |
| 🟢 **Baja / Arquitectura** | **4** | Archivos monolíticos que violan ultra modularidad (`html_template.rs`, `execution_engine.rs`), tipos de órdenes huérfanos sin uso, y 20 advertencias de compilador/Clippy. |

---

## 2. Tabla Consolidada de Hallazgos

| ID | Severidad | Módulo / Archivo | Líneas | Título / Naturaleza del Bug |
| :--- | :---: | :--- | :--- | :--- |
| **BUG-01** | 🔴 Crítica | `src/backtest/composite_score.rs` | L52–L57 | **Inversión Matemática del Gradiente de Pérdida en Composite Score** |
| **BUG-02** | 🔴 Crítica | `src/features/feature_matrix.rs` | L38–L50, L118–L135 | **Discrepancia Numérica de Features entre Entrenamiento e Inferencia Causal** |
| **BUG-03** | 🔴 Crítica | `src/features/triple_barrier_labeler.rs` | L86–L106 | **Etiquetado Asimétrico Falso-Positivo en Trades Short** |
| **BUG-04** | 🟠 Alta | `src/backtest/execution_engine.rs` | L172–L176, L224–L227 | **Liquidación Espuria a Precio 0.0 en MarginMode::Cross** |
| **BUG-05** | 🟠 Alta | `src/downloader/downloader_service.rs` | L50–L73 | **Aborto Prematuro de Ingestión Histórica de Velas** |
| **BUG-06** | 🟠 Alta | `src/types/fitness_metric.rs` | L84–L87 | **Inversión del Ratio Calmar con Retornos Negativos** |
| **BUG-07** | 🟠 Alta | `src/cli/forust_handler.rs`, `treeboost_handler.rs` | L150–L177 | **Contaminación y Falta de Purga en Backtest Completo** |
| **BUG-08** | 🟡 Media | `src/cli/config_handler.rs` | L26, L30–L106 | **Omisión de Parámetro `allow_short` en Menú Interactivo** |
| **BUG-09** | 🟡 Media | `src/types/margin_mode.rs` vs `src/backtest/config.rs` | L13–L17 vs L23 | **Inconsistencia de Variante `Default` en `MarginMode`** |
| **BUG-10** | 🟡 Media | `src/backtest/metrics_calculator.rs` | L198–L213 | **Asunción Rígida de Timestamps en Milisegundos en Anualización** |
| **BUG-11** | 🟡 Media | `src/storage/db_connection.rs` | L31, L37 | **Índice B-Tree Duplicado en SQLite sobre Tabla `klines`** |
| **BUG-12** | 🟡 Media | `src/models/*_strategy.rs` | L56–L63 | **Fallback Muerto / Código Inalcanzable en Estrategias GBDT** |
| **BUG-13** | 🟢 Baja | `src/types/order.rs` | L1–L34 | **Estructuras de Órdenes Huérfanas (Código Pasivo)** |
| **BUG-14** | 🟢 Baja | `src/dashboard/html_template.rs` | L1–L780 | **Monolito de Plantilla HTML (780 líneas / 31 KB)** |
| **BUG-15** | 🟢 Baja | `src/backtest/execution_engine.rs` | L1–L512 | **Sobrecarga de Responsabilidades en ExecutionEngine (512 líneas)** |
| **BUG-16** | 🟢 Baja | Varios archivos | Diversas | **20 Advertencias de Linter Clippy (`too_many_arguments`, `useless_format`, etc.)** |

---

## 3. Diagnóstico Detallado Pilar por Pilar

### Pilar 1: Detección de Look-Ahead Bias (Finanzas Causal t <= T)

1. **Alineación Temporal en Backtesting:**
   - La arquitectura central en `src/backtest/engine.rs` respeta el principio causal: las señales se calculan al **cierre** de la vela $t$ (`strategy.evaluate(klines, i)`) y se ejecutan estrictamente en el **Open** de la vela $t+1$ (`ExecutionEngine::process_signal_at_open`).
   - Los extremos intra-vela (`High` y `Low`) se evalúan únicamente para salidas por Stop Loss y Take Profit después de haber ejecutado la orden en el Open, modelando correctamente los Gaps de apertura.

2. **Vulnerabilidad BUG-02 (Discrepancia en Feature Matrix):**
   - En entrenamiento, `FeatureMatrix::build` calcula EMAs y RSI sobre el vector completo desde el índice 0.
   - En inferencia individual causal (`FeatureMatrix::extract_single_vector`), se extrae una ventana acotada de las últimas 80 velas.
   - Debido a que una EMA de periodo 30 inicializa su SMA en `idx - 80` en lugar de la vela 0, el valor numérico difiere hasta en un 3.5% entre lo que el modelo aprendió y lo que recibe en tiempo de inferencia individual.

3. **Vulnerabilidad BUG-03 (Etiquetado Asimétrico en Short):**
   - `TripleBarrierLabeler::label_classification_barriers` define `upper_barrier = entry_price * (1.0 + pt)` y `lower_barrier = entry_price * (1.0 - sl)`.
   - Cuando se evalúa un Short, una caída de precio toca `lower_barrier`, pero si `pt > sl` (ej. TP 4%, SL 2%), el trade es etiquetado como `TargetLabel::Short` habiendo avanzado solo un 2%, cuando el TP del Short requería un 4%.

---

### Pilar 2: Detección de Bugs Lógicos y Estabilidad Numérica

1. **Vulnerabilidad BUG-01 (Inversión Matemática en Composite Score - CRÍTICO):**
   - En `CompositeScoreCalculator::calculate`:
     ```rust
     let net_pnl = final_capital - initial_capital; // Negativo cuando hay pérdidas
     let loss_ratio = net_pnl / safe_max_dd;
     ```
     Al dividir un número negativo (`net_pnl`) por un divisor positivo grande (`safe_max_dd`), el resultado se aproxima a cero (es MENOS negativo).
     - Pérdida de -$100 con Drawdown de $100: $\text{ratio} = -1.0$.
     - Pérdida de -$100 con Drawdown de $1,000: $\text{ratio} = -0.1$.
     Como $-0.1 > -1.0$, una estrategia desastrosa con 10 veces más drawdown recibe una mejor puntuación que una estrategia conservadora. Durante la búsqueda de hiperparámetros y calibración en `ThresholdOptimizer`, el sistema tiende a favorecer parámetros que provocan mayores caídas patrimoniales.

2. **Vulnerabilidad BUG-04 (Liquidación Espuria a Cero en Cross Margin):**
   - En `MarginMode::Cross`, cuando el apalancamiento es 1x o el tamaño de la posición es menor que el balance de la cuenta, `(exec_price - equity/quantity).max(0.0)` produce `liq_price = 0.0`.
   - Luego, la condición `current_candle.low <= pos.liquidation_price` evalúa `0.0 <= 0.0 = true` ante cualquier anomalía o vela con precio cero en base de datos, liquidando la posición injustificadamente.

3. **Vulnerabilidad BUG-05 (Downloader Premature Abort):**
   - En `DownloaderService::sync_timeframe`, si una sola vela dentro de un bloque de 1,000 es filtrada por el búfer de tiempo o error de parsing, `count` pasa a 999. La condición `if count < 1000 { break; }` aborta toda la sincronización histórica creyendo erróneamente que Binance no tiene más velas.

4. **Vulnerabilidad BUG-06 (Calmar Ratio Invertido):**
   - Misma distorsión que BUG-01: cuando `net_profit_pct < 0`, dividir por un `max_dd` mayor produce un número más cercano a 0, premiando carteras de mayor riesgo.

---

### Pilar 3: Código Fantasma y Mocks (Cero Mocks)

1. **Vulnerabilidad BUG-13 (Estructuras de Órdenes Sin Uso):**
   - `src/types/order.rs` contiene `OrderRequest` y `ExecutionResult`. Ningún componente del motor las instancia ni las procesa; el motor opera directamente sobre `Signal` y `CompletedTrade`. Debe decidirse si se integran funcionalmente o se eliminan para no inducir a confusión arquitectónica.
2. **Vulnerabilidad BUG-12 (Ramas de Fallback en Estrategias):**
   - Las llamadas a `predict_single` en los bloques `Err(_)` de las estrategias nunca se alcanzan en operación normal, ya que `predict_batch` opera de forma determinista en memoria sobre datos ya validados.

---

### Pilar 4: Ultra Modularidad y Arquitectura de Rust

1. **Vulnerabilidad BUG-14 (Monolito HTML):**
   - `src/dashboard/html_template.rs` acumula 780 líneas combinando lógica de marcado HTML, scripts embebidos de Chart.js y estilos CSS. Debe modularizarse en submódulos especializados (`src/dashboard/template/`).
2. **Vulnerabilidad BUG-15 (Sobrecarga en ExecutionEngine):**
   - `src/backtest/execution_engine.rs` acumula 512 líneas gestionando ejecución de señales, liquidación, Stop Loss, Take Profit, comisiones y cálculo de excursiones. Debe desacoplarse en subestructuras dedicadas.

---

### Pilar 5: Compilación Estática y Linter Clippy

- El proyecto compila con código de salida 0 mediante `cargo check` y `cargo test --no-run`.
- Se detectaron **20 advertencias de Clippy**:
  - `too_many_arguments`: `Position::new` (10 argumentos) y `CartTree::build_tree` (11 argumentos).
  - `derivable_impls`: `MarginMode` y `FitnessMetric` tienen implementaciones manuales de `Default` en lugar de `#[derive(Default)]`.
  - `match_like_matches_macro`: Lógica de cierre en `ExecutionEngine::process_signal_at_open`.
  - `unnecessary_map_or`: Uso repetido de `map_or(false, ...)` en lugar del idiomático `is_some_and(...)`.
  - `useless_format`: Llamadas redundantes a `format!("{}", e)` en lugar de `e.to_string()`.
  - `new_without_default`: `InteractiveApp::new()` sin implementar `Default`.

---

## 4. Fichas Técnicas de los Bugs Críticos y Altos

---

### Ficha Técnica BUG-01
- **Archivo:** [`src/backtest/composite_score.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs#L52-L57)
- **Severidad:** 🔴 Crítica
- **Impacto:** Distorsiona la optimización de hiperparámetros y calibración de umbrales en `ThresholdOptimizer`, favoreciendo estrategias con drawdowns severos durante periodos con pérdidas.
- **Causa Raíz:**
  ```rust
  let net_pnl = final_capital - initial_capital; // Valor negativo
  let loss_ratio = net_pnl / safe_max_dd;        // DIVISIÓN PROBLEMÁTICA
  let bounded_loss = loss_ratio.max(-100.0);
  let penalty_score = (0.5 * linearity) + (0.5 * bounded_loss);
  ```
- **Solución Propuesta:**
  Penalizar el drawdown de forma directamente proporcional al capital cuando hay pérdidas, garantizando monotonía estricta:
  ```rust
  let net_pnl = final_capital - initial_capital; // Negativo
  // Penalización compuesta: pérdida acumulada más el drawdown sufrido
  let loss_impact = (net_pnl - safe_max_dd) / initial_capital;
  let bounded_loss = loss_impact.clamp(-100.0, 0.0);
  let penalty_score = (0.5 * linearity) + (0.5 * bounded_loss);
  ```

---

### Ficha Técnica BUG-02
- **Archivo:** [`src/features/feature_matrix.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/feature_matrix.rs#L118-L135)
- **Severidad:** 🔴 Crítica
- **Impacto:** Divergencia matemática entre la distribución de features de entrenamiento y la de inferencia en tiempo real o pruebas vela a vela.
- **Causa Raíz:**
  `extract_single_vector` toma únicamente una ventana de 80 velas (`idx.saturating_sub(80)`). La EMA de periodo 30 inicializa su SMA sobre las primeras 30 velas de esa ventana reducida, generando un desfase respecto a la EMA calculada globalmente desde la vela 0 en `build()`.
- **Solución Propuesta:**
  Permitir que la inferencia individual causal pase el historial acumulado necesario o un estado incremental de indicadores (`EmaState`, `RsiState`), o garantizar que el tamaño del búfer de calentamiento sea lo suficientemente amplio (ej. 250 velas) para que la diferencia exponencial sea matemáticamente despreciable ($< 10^{-6}$).

---

### Ficha Técnica BUG-03
- **Archivo:** [`src/features/triple_barrier_labeler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/triple_barrier_labeler.rs#L86-L117)
- **Severidad:** 🔴 Crítica
- **Impacto:** Generación errónea de señales de entrenamiento para operaciones en corto cuando se configuran barreras asimétricas de Take Profit y Stop Loss.
- **Causa Raíz:**
  Se utilizan barreras estáticas centradas en una posición larga: `upper_barrier = entry * (1 + pt)` y `lower_barrier = entry * (1 - sl)`. Si el precio baja un `sl`%, se cataloga como `TargetLabel::Short`, cuando para un Short un movimiento a la baja representa Take Profit que requiere descender `pt`%.
- **Solución Propuesta:**
  Definir objetivos simétricos o parametrizar barreras independientes para la hipótesis Long y la hipótesis Short.

---

### Ficha Técnica BUG-04
- **Archivo:** [`src/backtest/execution_engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L173,L224-L227)
- **Severidad:** 🟠 Alta
- **Impacto:** Liquidación forzosa inmediata a precio 0.0 y bancarrota artificial de la cuenta.
- **Causa Raíz:**
  En `MarginMode::Cross` con bajo apalancamiento, `liq_price` se calcula como 0.0. Posteriormente, en `check_intra_candle_exits`, la condición `current_candle.low <= pos.liquidation_price` evalúa `true` si una vela tiene `low == 0.0`.
- **Solución Propuesta:**
  ```rust
  PositionSide::Long => {
      if pos.liquidation_price > 0.0 && current_candle.low <= pos.liquidation_price {
          let liq_target = pos.liquidation_price.min(current_candle.open);
          exit_event = Some((liq_target, "Liquidated (Margin Call)"));
      }
  }
  ```

---

### Ficha Técnica BUG-05
- **Archivo:** [`src/downloader/downloader_service.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/downloader_service.rs#L70-L73)
- **Severidad:** 🟠 Alta
- **Impacto:** Interrupción silenciosa y prematura de la descarga de datos históricos de Binance.
- **Causa Raíz:**
  El bucle evalúa `count < 1000` sobre las velas ya filtradas (`klines.len()`), en lugar de evaluar si la API de Binance devolvió menos de 1,000 registros brutos.
- **Solución Propuesta:**
  `BinanceClient::download_klines` debe devolver tanto las velas limpias como la cantidad bruta recibida de la API REST de Binance. El corte del bucle debe depender exclusivamente de si `raw_count < 1000`.

---

## 5. Plan de Acción y Priorización para Implementación

1. **Fase 1 (Corrección Matemática y Causal):**
   - Reparar `CompositeScoreCalculator` eliminando la inversión en retornos negativos (BUG-01).
   - Ajustar el etiquetado en `TripleBarrierLabeler` para simetría Long/Short (BUG-03).
   - Proteger la comprobación de precio de liquidación en `ExecutionEngine` contra ceros (BUG-04).
   - Corregir la condición de parada del bucle en `DownloaderService` (BUG-05).
   - Corregir el cálculo de `CalmarRatio` en `FitnessMetric` (BUG-06).

2. **Fase 2 (Interfaz y Consistencia):**
   - Exponer la configuración de `allow_short` en `ConfigHandler` (BUG-08).
   - Homogeneizar el `Default` de `MarginMode` en `BacktestConfig` y su enum (BUG-09).
   - Eliminar el índice duplicado en `DbConnection` (BUG-11).

3. **Fase 3 (Refactorización y Ultra Modularidad):**
   - Descomponer `src/dashboard/html_template.rs` en submódulos especializados de renderizado.
   - Descomponer `src/backtest/execution_engine.rs` en módulos atómicos de ejecución, salidas y contabilidad.
   - Corregir las 20 advertencias de Clippy.

---
*Reporte generado de forma autónoma bajo los estándares cuantitativos y de ingeniería de software en Rust de Allianz.*

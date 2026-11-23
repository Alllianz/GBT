# AUDITORÍA FORENSE CUANTITATIVA INSTITUCIONAL DE TERCERA RONDA - BUGS, MATEMÁTICAS Y CAUSALIDAD (GBT)
**Destinatario:** Allianz  
**Auditor:** Subagente Auditor Cuantitativo Forense  
**Fecha:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Veracidad y Rigor Matemático Institucional, Ultra Modularidad en Rust.

---

## 1. RESUMEN EJECUTIVO DE LA TERCERA RONDA

Estimado **Allianz**, tras la consolidación de las mejoras de la segunda ronda, se ha ejecutado una auditoría forense exhaustiva de nivel institucional sobre la totalidad del código fuente en `src/`. El objetivo central ha sido localizar bugs sutiles de segundo orden, desalineaciones de índices residuales, riesgos de división por cero o propagación de `NaN`, vulnerabilidades ante valores extremos y verificar con rigor matemático la causalidad temporal estricta ($t \le T$).

Se han identificado **8 nuevos hallazgos** (2 Críticos, 3 Altos, 3 Medios):
- **Bugs Críticos:** Desalineación de índice en el descuento de filas válidas en el entrenamiento de los 5 modelos de Machine Learning (inyección de una muestra fantasma con target nulo `0.0`), e incoherencia de liquidación inmediata de Stop Loss intra-vela si el slippage supera o iguala la distancia del SL.
- **Bugs Altos:** Potencial división por cero o inyección de infinito en `PositionSizing::RiskBasedPercent` e ignorado de stop loss dinámico, riesgo de equity infinito por `open <= 0.0` o `exec_price == 0.0` en `ExecutionEngine`, y paradoja matemática en `MetricsCalculator::calculate_sharpe_sortino` donde estrategias con 100% de operaciones ganadoras reciben un Sortino Ratio de `0.0`.
- **Bugs Medios:** Propagación silenciosa de `NaN` en `calculate_r2` y `CompositeScoreCalculator`, llamadas desprotegidas a `.unwrap()` en runtime susceptibles de provocar pánico, e inconsistencia de desacoplamiento temporal en la ventana de purga de Walk-Forward.

---

## 2. TABLA RESUMEN DE HALLAZGOS (TERCERA RONDA)

| ID | Severidad | Módulo / Archivo | Componente / Función | Descripción Resumida |
|---|---|---|---|---|
| **BUG-22** | **CRÍTICO** | `models::*` | `train()` en los 5 modelos GBDT/CART | Desalineación de índice tras `label_regression`: inyección de una muestra residual con target `0.0` al final del set de entrenamiento (`horizon` vs `horizon + 1`). |
| **BUG-23** | **CRÍTICO** | `backtest::execution_engine` | `process_signal_at_open` y `check_intra_candle_exits` | Disparo incoherente inmediato de Stop Loss en la vela de apertura si `slippage_rate >= fixed_stop_loss_pct`. |
| **BUG-24** | **ALTO** | `backtest::execution_engine` | `PositionSizing::RiskBasedPercent` | Vulnerabilidad ante `sl_dist <= 0.0` (produce `f64::INFINITY`) e ignorado del Stop Loss dinámico provisto por la señal. |
| **BUG-25** | **ALTO** | `backtest::execution_engine` | `process_signal_at_open` y `close_position` | Multiplicador de cantidad infinita (`quantity = inf`) si `exec_price <= 0.0` o `open <= 0.0`, inflando el equity a infinito. |
| **BUG-26** | **ALTO** | `backtest::metrics_calculator` | `calculate_sharpe_sortino` | Paradoja del Sortino Ratio: si no existen retornos negativos (100% win rate), el ratio asigna `0.0` en lugar de una puntuación máxima. |
| **BUG-27** | **MEDIO** | `backtest::metrics_calculator` / `composite_score` | `calculate_r2` y `calculate` | Propagación de `NaN` a través de comparaciones de punto flotante en IEEE 754, corrompiendo el Score Compuesto. |
| **BUG-28** | **MEDIO** | `backtest::engine` / `downloader::downloader_service` | `engine.rs:122`, `downloader_service.rs:30` | Uso de `.unwrap()` sin manejo defensivo en rutas críticas de fin de datos y sincronización. |
| **BUG-29** | **MEDIO** | `cli::*` / `features::train_test_split` | Handlers y split cronológico | Inconsistencia entre el período de lookback de los indicadores y la ventana de purga (`purge_bars`) en validaciones Walk-Forward. |

---

## 3. ANÁLISIS FORENSE DETALLADO DE CADA HALLAZGO

---

### [BUG-22] CRÍTICO: Desalineación de Descuento de Filas Válidas en los 5 Modelos de Machine Learning (Target Residual Nulo)

- **Ubicación:** 
  - `src/models/treeboost_model.rs:32`
  - `src/models/forust_model.rs:48`
  - `src/models/decision_tree/cart_model.rs:32`
  - `src/models/random_forest/forest_model.rs:34`
  - `src/models/native_gbdt/native_model.rs:33`

#### Diagnóstico del Problema:
En la auditoría previa, se corrigió `TripleBarrierLabeler::label_regression` (BUG-17) para eliminar el sesgo Close-to-Open, alineando la base al Open de la siguiente vela:
```rust
let valid_n = n.saturating_sub(horizon + 1);
let mut returns: Vec<f64> = (0..valid_n)
    .into_par_iter()
    .map(|i| {
        let p_entry = klines[i + 1].open;
        let p_exit = klines[i + 1 + horizon].close;
        ...
    })
    .collect();
returns.resize(n, 0.0);
```
Dado que para la muestra $i$, el precio de salida se toma en el índice $i + 1 + horizon$, el índice máximo evaluado es $valid\_n - 1 + 1 + horizon = valid\_n + horizon = n - 1$. Por lo tanto, el número de retornos calculados de forma causal es estrictamente $n - (horizon + 1)$. Los índices desde $n - horizon - 1$ hasta $n - 1$ son rellenados con `0.0` mediante `returns.resize(n, 0.0)`.

Sin embargo, los 5 modelos de predicción mantuvieron la lógica anterior:
```rust
let valid_rows = rows.saturating_sub(self.config.horizon); // O rows.saturating_sub(horizon)
let y = TripleBarrierLabeler::label_regression(train_klines, self.config.horizon);
let x_slice = &matrix_flat[0..(valid_rows * f_dim)];
let y_slice = &y[0..valid_rows];
```
Dado que `valid_rows` es igual a $n - horizon$, resulta ser exactamente **1 fila mayor** que `valid_n` ($n - horizon - 1$).
La muestra situada en el índice $i = n - horizon - 1$ toma sus características explicativas $X_i$ de la serie temporal real, pero su etiqueta $y_i$ en `y_slice` corresponde al relleno por defecto `0.0`.

#### Impacto Financiero y Cuantitativo:
1. **Contaminación del Conjunto de Entrenamiento:** En cada entrenamiento, la última muestra histórica antes de la purga es emparejada con un retorno artificialmente nulo (0.0000%), forzando al modelo a aprender una relación falsa entre las condiciones de mercado de esa vela y una inmovilidad de precio inexistente.
2. **Distorsión del Sesgo Base en Native GBDT:** En `native_model.rs`, la predicción base inicial se calcula como:
   `self.base_score = y_slice.iter().sum::<f64>() / valid_rows as f64;`
   La muestra nula arrastra hacia abajo el promedio real de retornos del mercado, sesgando todas las hojas de los árboles subsecuentes.
3. **Muestreo Corrupto en Random Forest:** En `forest_model.rs`, el algoritmo de Bootstrap con reemplazo selecciona aleatoriamente índices entre `0..valid_rows`. La muestra corrupta puede ser seleccionada múltiples veces en múltiples árboles, diluyendo la capacidad predictiva del ensamble.

#### Código Vulnerable:
```rust
// treeboost_model.rs:32, forust_model.rs:48, cart_model.rs:32, forest_model.rs:34, native_model.rs:33
let valid_rows = rows.saturating_sub(self.config.horizon);
```

#### Código Corregido Recomendado:
```rust
// Debe descontarse horizon + 1 para concordar con la etiqueta causal de regresión
let valid_rows = rows.saturating_sub(self.config.horizon + 1);
```

---

### [BUG-23] CRÍTICO: Disparo Incoherente Inmediato de Stop Loss cuando `slippage_rate >= fixed_stop_loss_pct`

- **Ubicación:** `src/backtest/execution_engine.rs:61-64, 109-113, 205-218`

#### Diagnóstico del Problema:
Examinemos el ciclo temporal de una vela $t$ en `engine.rs` y `execution_engine.rs`:
1. Al inicio de la vela $t$, si hay una señal Long pendiente generada al cierre de $t-1$:
   ```rust
   let exec_price = current_candle.open * (1.0 + config.slippage_rate);
   ```
2. El Stop Loss se asigna en:
   ```rust
   let stop_loss = exec_price * (1.0 - config.fixed_stop_loss_pct.unwrap());
   ```
   Sustituyendo:
   $$\text{stop\_loss} = \text{open} \times (1.0 + \text{slippage\_rate}) \times (1.0 - \text{fixed\_stop\_loss\_pct})$$
   Si por ejemplo el usuario configura un Stop Loss del 0.10% (`0.001`) y el slippage estimado del mercado es de 0.15% (`0.0015`):
   $$(1.0 + 0.0015) \times (1.0 - 0.001) = 1.0015 \times 0.999 = 1.0004985 > 1.0$$
   ¡El precio del Stop Loss queda por **encima** del precio de apertura $\text{open}$ de la vela!
3. Inmediatamente a continuación en el mismo tick temporal de la vela $t$, `engine.rs` llama a `ExecutionEngine::check_intra_candle_exits`:
   ```rust
   let hit_sl = pos.stop_loss.map_or(false, |sl| current_candle.low <= sl);
   ```
   Puesto que en cualquier vela japonesa $\text{low} \le \text{open}$, y aquí $\text{open} < \text{stop\_loss}$, la condición $\text{low} \le \text{stop\_loss}$ se evalúa como **`true` de forma instantánea e ineludible en el 100% de los casos**.
4. La orden que acaba de entrar en el Open se liquida por Stop Loss en la mismísima vela de entrada, cobrando doble slippage y doble comisión.

#### Impacto Financiero:
- Destrucción inmediata del capital en estrategias scalping o con stops de volatilidad ceñidos.
- Generación de secuencias continuas de pérdidas ficticias con `duration_bars == 0`.
- Imposibilidad de que cualquier trade prospere si el mercado tiene slippage representativo.

#### Solución Recomendada:
El Stop Loss contractualmente no puede calcularse como un porcentaje flotante por encima del Open en posiciones Long (o por debajo en Short). El precio de activación del Stop Loss debe referenciarse a la estructura de precios del subyacente o acotarse estrictamente para que $\text{stop\_loss} \le \text{current\_candle.open} \times (1.0 - \epsilon)$:
```rust
let stop_loss = match side {
    PositionSide::Long => (exec_price * (1.0 - pct)).min(current_candle.open * (1.0 - pct)),
    PositionSide::Short => (exec_price * (1.0 + pct)).max(current_candle.open * (1.0 + pct)),
};
```

---

### [BUG-24] ALTO: Vulnerabilidad ante `sl_dist <= 0.0` e Ignorado del Stop Loss Dinámico en `RiskBasedPercent`

- **Ubicación:** `src/backtest/execution_engine.rs:84-89`

#### Diagnóstico del Problema:
En `ExecutionEngine::process_signal_at_open`:
```rust
PositionSizing::RiskBasedPercent(risk_pct) => {
    let sl_dist = config.fixed_stop_loss_pct.unwrap_or(0.02);
    let risk_amount = equity * risk_pct.clamp(0.001, 0.10);
    let pos_size = risk_amount / sl_dist;
    pos_size.min(equity * leverage)
}
```
Existen dos deficiencias graves:
1. **Riesgo de División por Cero / Infinito:** Si en la configuración el usuario desactiva el stop loss fijo fijando `fixed_stop_loss_pct = Some(0.0)`:
   `sl_dist` es `0.0`. `risk_amount / 0.0` genera `f64::INFINITY`. Si `sl_dist` resulta ser negativo o `NaN`, `pos_size` se vuelve `NaN`.
   Bajo el estándar IEEE 754 de Rust, la validación posterior:
   `if notional_target <= 0.0 || required_margin > equity || portfolio.cash_balance <= commission`
   evalúa todas sus condiciones a `false` frente a `NaN`, permitiendo la apertura de órdenes descalibradas.
2. **Desconexión con el Stop Loss de la Señal:** Si una estrategia avanzada (como ATR-trailing o Stop dinámico de volatilidad) emite una señal con `signal.stop_loss_price`, `RiskBasedPercent` la ignora por completo y utiliza el valor fijo de `config.fixed_stop_loss_pct` o el `0.02` por defecto. Si el modelo requería un stop loss amplio del 6%, pero el dimensionador calcula la posición asumiendo un stop del 2%, el tamaño de la posición asignado será **el triple del riesgo permitido**.

#### Solución Recomendada:
Calcular `sl_dist` dinámicamente evaluando primero la distancia porcentual al `signal.stop_loss_price`, respaldado por `config.fixed_stop_loss_pct`, e imponer un piso mínimo infranqueable:
```rust
PositionSizing::RiskBasedPercent(risk_pct) => {
    let dynamic_sl_pct = signal.stop_loss_price.map(|sl| {
        (exec_price - sl).abs() / exec_price
    });
    let sl_dist = dynamic_sl_pct
        .or(config.fixed_stop_loss_pct)
        .unwrap_or(0.02)
        .max(0.001); // Guarda infranqueable contra división por cero
    let risk_amount = equity * risk_pct.clamp(0.001, 0.10);
    let pos_size = risk_amount / sl_dist;
    pos_size.min(equity * leverage)
}
```

---

### [BUG-25] ALTO: Multiplicador de Cantidad Infinita por Precio No Positivo

- **Ubicación:** `src/backtest/execution_engine.rs:60-64, 100-102`

#### Diagnóstico del Problema:
Si en los datos históricos descargados de Binance existe una vela anómala con `open <= 0.0`:
```rust
let exec_price = match side {
    PositionSide::Long => current_candle.open * (1.0 + config.slippage_rate),
    PositionSide::Short => current_candle.open * (1.0 - config.slippage_rate),
};
let quantity = notional_target / exec_price;
```
Si `exec_price == 0.0`:
`quantity = notional_target / 0.0 = f64::INFINITY`.
La comprobación existente en la línea 102 es:
```rust
if quantity > 0.0 {
    portfolio.cash_balance -= commission;
    ...
```
En Rust, `f64::INFINITY > 0.0` evalúa a **`true`**.
La posición se abre con `quantity = INFINITY` y `entry_price = 0.0`.
En las velas subsiguientes, la función `unrealized_pnl` calcula:
$$\text{pnl} = (\text{close} - 0.0) \times \infty = \infty$$
El equity del portafolio se dispara inmediatamente a infinito, corrompiendo todos los cálculos del Sharpe Ratio, Sortino Ratio, Drawdown y Composite Score.

#### Solución Recomendada:
Validación estricta previa:
```rust
if current_candle.open <= 1e-8 || exec_price <= 1e-8 {
    return completed_trade;
}
```

---

### [BUG-26] ALTO: Paradoja Numérica del Sortino Ratio en Estrategias con Cero Pérdidas

- **Ubicación:** `src/backtest/metrics_calculator.rs:186-192, 219-223`

#### Diagnóstico del Problema:
En `MetricsCalculator::calculate_sharpe_sortino`:
```rust
let downside_variance = returns
    .iter()
    .filter(|&&r| r < 0.0)
    .map(|&r| r.powi(2))
    .sum::<f64>()
    / n;
let downside_std_dev = downside_variance.sqrt();
...
let sortino = if downside_std_dev > 1e-9 {
    (mean / downside_std_dev) * annual_factor
} else {
    0.0
};
```
Si una estrategia o conjunto de parámetros logra una curva donde **ninguna vela experimenta retorno negativo** (típico en periodos de fuerte tendencia alcista con buen filtro de entrada, o tras pocas operaciones altamente ganadoras):
`downside_variance` es `0.0`, y `downside_std_dev` es `0.0`.
Como `downside_std_dev > 1e-9` es `false`, el código asigna `sortino = 0.0`.

#### Impacto Financiero:
- En `threshold_optimizer.rs`, cuando el usuario selecciona la métrica fitness `SortinoRatio`, los umbrales que producen curvas perfectas sin pérdidas reciben un score de `0.0`.
- El optimizador descarta la mejor configuración posible y selecciona umbrales mediocres que sí incurrieron en pérdidas simplemente porque estos últimos producen un denominador mayor a cero.
- En contraste, `profit_factor` y `recovery_factor` manejan este caso correctamente devolviendo `99.99`.

#### Solución Recomendada:
```rust
let sortino = if downside_std_dev > 1e-9 {
    (mean / downside_std_dev) * annual_factor
} else if mean > 0.0 {
    99.99 * annual_factor.clamp(1.0, 10.0)
} else {
    0.0
};
```

---

### [BUG-27] MEDIO: Propagación de `NaN` en `calculate_r2` y `CompositeScoreCalculator`

- **Ubicación:** `src/backtest/metrics_calculator.rs:256-266` y `src/backtest/composite_score.rs:25`

#### Diagnóstico del Problema:
En `MetricsCalculator::calculate_r2`:
```rust
if s_xx <= 1e-12 || s_yy <= 1e-12 {
    return 0.0;
}
let slope = s_xy / s_xx;
if slope <= 0.0 {
    return 0.0;
}
let r2 = (s_xy * s_xy) / (s_xx * s_yy);
r2.clamp(0.0, 1.0)
```
Si por alguna anomalía de datos el balance en algún punto contiene `NaN`:
`s_xx` o `s_yy` se vuelven `NaN`.
En Rust, toda comparación de desigualdad contra `NaN` retorna `false`:
- `NaN <= 1e-12` -> `false`.
- `slope <= 0.0` (donde slope es NaN) -> `false`.
`r2` se calcula como `NaN`.
La función estándar `f64::clamp(self, min, max)` retorna `self` si `self` es `NaN`.
Por ende, `calculate_r2` devuelve `NaN`.
Posteriormente, en `CompositeScoreCalculator::calculate`:
`let linearity = curve_r2.clamp(0.0, 1.0);`
devuelve `NaN`, y el score final deviene en `NaN`.

#### Solución Recomendada:
```rust
if !s_xx.is_finite() || !s_yy.is_finite() || s_xx <= 1e-12 || s_yy <= 1e-12 {
    return 0.0;
}
let slope = s_xy / s_xx;
if !slope.is_finite() || slope <= 0.0 {
    return 0.0;
}
let r2 = (s_xy * s_xy) / (s_xx * s_yy);
if !r2.is_finite() {
    0.0
} else {
    r2.clamp(0.0, 1.0)
}
```

---

### [BUG-28] MEDIO: Llamadas Desprotegidas a `.unwrap()` en Rutas de Ejecución Críticas

- **Ubicación:** 
  - `src/backtest/engine.rs:122`: `let last_candle = klines.last().unwrap();`
  - `src/backtest/execution_engine.rs:240`: `let pos = portfolio.active_position.take().unwrap();`
  - `src/downloader/downloader_service.rs:30`: `Utc.with_ymd_and_hms(2017, 8, 17, 0, 0, 0).unwrap()`

#### Diagnóstico del Problema:
Aunque en los flujos normales las condiciones de longitud previas mitigan el error, en un sistema de producción institucional el uso de `.unwrap()` es una práctica de riesgo ante refactorizaciones futuras o modificaciones de firmas concurrentes.
Por ejemplo, si en una futura modificación de `engine.rs` se introduce un filtro previo sobre `klines` o se manipula la posición en un hilo asíncrono, `.unwrap()` causará un pánico incontrolable que tumbará el servicio.

#### Solución Recomendada:
Sustituir por comprobación segura:
```rust
// engine.rs
if let Some(pos) = portfolio.active_position.take() {
    if let Some(last_candle) = klines.last() {
        let trade = ExecutionEngine::close_position(
            &mut portfolio, &self.config, pos, last_candle.close,
            last_candle.timestamp, klines.len().saturating_sub(1),
            "End of Data Horizon".to_string(),
        );
        trades.push(trade);
    }
}
```

---

### [BUG-29] MEDIO: Inconsistencia en la Ventana de Purga respecto al Período de Indicadores

- **Ubicación:** `src/cli/baseline_handler.rs:36-38`, `src/features/train_test_split.rs:27-35`

#### Diagnóstico del Problema:
En `baseline_handler.rs`:
```rust
let fast_p = InputReader::read_with_default("Período EMA Rápida", 9);
let slow_p = InputReader::read_with_default("Período EMA Lenta", 21);
...
let purge_bars = 5;
```
La purga entre In-Sample y Out-of-Sample está fijada en solo 5 velas. Sin embargo, la estrategia de referencia evalúa una EMA lenta de 21 periodos (o más si el usuario configura 50 o 100).
Si la purga es de 5 barras, la memoria exponencial de la EMA en las primeras velas de prueba retiene una dependencia matemática directa con los precios finales del conjunto In-Sample.
Para un aislamiento causal perfecto de nivel institucional, la purga debe ser:
$$\text{purge\_bars} = \text{horizon} + \max(\text{indicator\_lookback}, 14) + 1$$

#### Solución Recomendada:
Calcular dinámicamente `purge_bars` en función del lookback máximo de los indicadores activos.

---

## 4. VERIFICACIÓN METICULOSA DE LOS 5 FOCOS OBLIGATORIOS

### 1. Desalineación de Índices tras el cambio de `label_regression`:
- **Estado:** Confirmado BUG-22.
- **Detalle:** `triple_barrier_labeler.rs` genera etiquetas de regresión válidas únicamente para $n - (horizon + 1)$ elementos. Los 5 modelos (`treeboost_model.rs`, `forust_model.rs`, `cart_model.rs`, `forest_model.rs`, `native_model.rs`) truncaban con `rows.saturating_sub(horizon)`. Esto colaba invariablemente **1 muestra corrupta residual con target 0.0** al final de cada set de entrenamiento.

### 2. Robustez ante valores extremos, NaNs, ceros y divisiones por cero:
- **Estado:** Confirmados BUG-24, BUG-25, BUG-26 y BUG-27.
- **Detalle:**
  - `PositionSizing::RiskBasedPercent`: susceptible a división por cero y generación de `f64::INFINITY` si `sl_dist <= 0.0`.
  - `MetricsCalculator`: división por cero encubierta en Sortino cuando `downside_std_dev == 0.0`, penalizando con `0.0` a las estrategias más rentables.
  - `calculate_r2`: permeabilidad ante `NaN` que contamina el `CompositeScoreCalculator`.
  - `ExecutionEngine`: si `open <= 0.0` o `exec_price == 0.0`, `quantity` se vuelve infinito, inflando el portafolio sin control.

### 3. Manejo de errores y `unwrap()` / `panic!`:
- **Estado:** Confirmado BUG-28.
- **Detalle:** Se localizaron llamadas a `.unwrap()` en `engine.rs:122`, `execution_engine.rs:240`, y `downloader_service.rs:30`. Se proponen reemplazos 100% seguros con `if let Some(...)` y `unwrap_or(...)`.

### 4. Concurrencia y Datos (Rayon, `Arc<AtomicUsize>`, Polars):
- **Estado:** Auditado y Seguro.
- **Detalle:**
  - Rayon y Polars están sincronizados a través de `ThreadingConfig::initialize_global_pool(20)` y `AtomicUsize`.
  - No existen bloqueos mutuos (`deadlocks`), ya que el motor no utiliza primitivas bloqueantes de cerrojo (`std::sync::Mutex` o `RwLock`) dentro de los bucles de Rayon.
  - En `ThresholdOptimizer::optimize_in_sample`, las evaluaciones de la rejilla se distribuyen en paralelo sobre vectores de predicción inmutables (`&'a [f64]`), sin contención ni clonaciones redundantes.
  - `EndpointPool` utiliza `Arc<AtomicUsize>` con semántica `Relaxed`, lo cual es adecuado y óptimo para balanceo de carga round-robin sin contención de bus de memoria.

### 5. Backtesting y Causalidad Cero Look-Ahead ($t \le T$):
- **Estado:** Verificado y Validado Matemáticamente (con la salvedad de BUG-23).
- **Detalle:**
  - **Secuencia Causal del Bucle:**
    1. Señal generada en el Cierre de la vela $t-1$.
    2. Ejecución física al precio de Apertura (`Open`) de la vela $t$ con slippage y comisión.
    3. Evaluación intra-vela de Stop Loss, Take Profit y Liquidación usando los extremos `High` y `Low` de la vela $t$. Si coinciden SL y TP en la misma vela, se asume conservadoramente la salida por Stop Loss.
    4. Cierre contable de balance y registro de curva de equity al precio de Cierre (`Close`) de la vela $t$.
    5. Evaluación de la estrategia al Cierre de la vela $t$ con histórico causal $[0..=t]$ para emisión de señal dirigida a la vela $t+1$.
  - No existe fuga temporal de datos futuros en la simulación. La única corrección crítica requerida es BUG-23 para evitar que un slippage mayor al SL provoque ejecuciones en la misma barra por desajuste de precio relativo.

---

## 5. HOJA DE RUTA DE IMPLEMENTACIÓN

Para la fase de implementación a cargo del subagente implementador:
1. **Fase 1 (Alineación de Modelos - BUG-22):** Actualizar en los 5 modelos el cálculo de filas válidas a `valid_rows = rows.saturating_sub(self.config.horizon + 1)`.
2. **Fase 2 (Motor de Ejecución - BUG-23, BUG-24, BUG-25):** 
   - Proteger `exec_price <= 0.0` y `open <= 0.0`.
   - Implementar cálculo acotado del Stop Loss en `process_signal_at_open` para que no supere el Open en Longs ni perfore el Open en Shorts.
   - Robustecer `RiskBasedPercent` integrando `signal.stop_loss_price` y piso mínimo `max(0.001)`.
3. **Fase 3 (Métricas Financieras - BUG-26, BUG-27):** 
   - Añadir valor límite de Sortino (`99.99 * annual_factor`) cuando `downside_std_dev <= 1e-9` y `mean > 0.0`.
   - Proteger `calculate_r2` contra valores no finitos (`is_finite()`).
4. **Fase 4 (Resiliencia de Código - BUG-28, BUG-29):** Eliminar `.unwrap()` residuales y ajustar la purga temporal en handlers.
5. **Fase 5 (Compilación y Verificación):** Comprobar tipos y ausencia de advertencias con `cargo check`.

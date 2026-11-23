# AUDITORÍA FORENSE CUANTITATIVA INSTITUCIONAL DE CUARTA RONDA - BUGS, MATEMÁTICAS Y CAUSALIDAD (GBT)
**Destinatario:** Allianz  
**Auditor:** Subagente Auditor Cuantitativo Forense  
**Fecha:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Veracidad y Rigor Matemático Institucional, Ultra Modularidad en Rust.

---

## 1. RESUMEN EJECUTIVO DE LA CUARTA RONDA

Estimado **Allianz**, como Director Cuantitativo Forense, he completado una exhaustiva inspección de Nivel Institucional sobre la totalidad de la arquitectura del proyecto GBT. Esta cuarta ronda ha auditado con lupa quirúrgica las correcciones implementadas previamente (BUG-22 al BUG-29), la estabilidad matemática de los indicadores técnicos y de la matriz de características causales en casos de borde (ej. series de exactamente 35 velas), los motores de inferencia (TreeBoost/Polars, Forust/Rayon y Native GBDT), la contabilidad de reversiones y sincronización de índices en el motor de Backtest, la sanitización en el generador de Dashboards y la resiliencia temporal de la sincronización con Binance ante desviaciones de reloj local (NTP drift).

El sistema exhibe hoy una solidez cuantitativa y causal notable:
- La sincronización `horizon + 1` en los 5 modelos de aprendizaje automático eliminó con éxito el sesgo de target nulo residual en el entrenamiento.
- La deducción de comisiones y ejecución con slippage en reversiones (Long a Short en la misma barra) es justa, transparente y conserva la solvencia de la cuenta.
- El modelo `Native GBDT` implementa de forma impecable el algoritmo de Friedman con shrinkage de learning rate y ajuste CART de residuales.

No obstante, se han detectado **8 nuevos hallazgos** (2 Altos, 4 Medios y 2 Bajos) que deben ser subsanados para alcanzar el estándar definitivo de producción institucional:

1. **Vulnerabilidad ante Desviación de Reloj Local (NTP Drift) en Binance Client [BUG-30 - ALTO]:** Filtrar velas con `raw.close_time() <= now_ms` del reloj local puede inyectar velas incompletas en la base de datos (si el reloj está adelantado) que nunca más se corregirán, o truncar prematuramente la sincronización (si el reloj está atrasado).
2. **Falta de Acotamiento Inferior y Bounding Incompleto en Señales Dinámicas de SL y TP [BUG-33 - ALTO]:** Si una señal emite un `stop_loss_price <= 0.0` o `NaN`, o un `take_profit_price` ilógico (por debajo de Open en Long o por encima en Short), la posición puede quedar sin Stop Loss o cerrarse espuriamente en la vela de apertura.
3. **Falla en Creación de Archivos en Windows por Caracteres No Sanitizados en `model_name` [BUG-31 - MEDIO]:** En `DashboardGenerator`, si el modelo incluye caracteres como `/`, `:`, `\` o `?` (ej. `Baseline_EMA_9/21`), `File::create` entra en pánico en sistemas Windows.
4. **Desalineación de MFE / MAE en Salidas en Open por Señal [BUG-32 - MEDIO]:** Cuando una posición se cierra en el Open por cambio de señal (`process_signal_at_open`), no se actualizan los extremos con el precio Open de la vela actual, distorsionando MFE y MAE ante aperturas con Gap.
5. **Ambigüedad y Solapamiento de Índices en `EquityPoint.candle_idx` en Validación Out-of-Sample (OOS) [BUG-34 - MEDIO]:** En `run_with_warmup`, el índice de vela de OOS inicia en `warmup_bars` (50), solapándose con los índices 50..300 de In-Sample en los archivos CSV.
6. **Propagación Potencial de Floats No Finitos (`NaN`/`Inf`) desde Inferencia de `TreeBoostModel` [BUG-35 - MEDIO]:** La conversión directa `v as f64` desde el vector de predicciones `f32` de Polars/TreeBoost no filtra valores no finitos.
7. **Silenciamiento Asimétrico en `TechnicalIndicators::roc` [BUG-36 - BAJO]:** Ante valores previos `<= 0.0`, `roc` fija silenciosamente `0.0` sin validar series oscilatorias ni períodos nulos.
8. **Inconsistencia Metodológica Residual en `label_classification` [BUG-37 - BAJO]:** `label_classification` retiene la referencia a `klines[i].close` y `valid_n = n - horizon`, en contraste con la causalidad Open-to-Close de `label_regression`.

---

## 2. TABLA RESUMEN DE HALLAZGOS (CUARTA RONDA)

| ID | Severidad | Módulo / Archivo | Componente / Función | Descripción Resumida |
|---|---|---|---|---|
| **BUG-30** | **ALTO** | `downloader::binance_client` / `downloader_service` | `download_klines` y `sync_timeframe` | Riesgo de corrupción histórica por NTP drift: si el reloj local está adelantado, guarda velas incompletas que jamás se re-descargan; si está atrasado, trunca la descarga. |
| **BUG-31** | **MEDIO** | `dashboard::dashboard_generator` / `html_template` | `generate_and_save` y `render` | Pánico en Windows si `model_name` contiene `/`, `:`, `\` o `?`, y falta de escape de entidades HTML en `<title>` y tablas. |
| **BUG-32** | **MEDIO** | `backtest::execution_engine` | `process_signal_at_open` y `close_position` | Subestimación de MFE/MAE: omisión de actualización de extremos con `current_candle.open` al cerrar por reversión o salida de señal. |
| **BUG-33** | **ALTO** | `backtest::execution_engine` | `process_signal_at_open` y `check_intra_candle_exits` | Falta de acotamiento inferior contra valores nulos o negativos en `signal.stop_loss_price`, y ausencia de bounding contra Open en `take_profit_price`. |
| **BUG-34** | **MEDIO** | `backtest::engine` / `dashboard::equity_curve_exporter` | `run_with_warmup` y `save_csv` | Solapamiento de `EquityPoint.candle_idx`: en OOS comienza en `warmup_bars` (50) colisionando con los índices de In-Sample. |
| **BUG-35** | **MEDIO** | `models::treeboost_model` | `predict_batch` y `predict_single` | Propagación de `NaN`/`Inf` al convertir directamente `preds` de `f32` a `f64` sin comprobación defensiva `.is_finite()`. |
| **BUG-36** | **BAJO** | `features::technical_indicators` | `roc` y `rsi` | Silenciamiento asimétrico ante precios nulos o negativos en `roc` y falta de validación de `period == 0`. |
| **BUG-37** | **BAJO** | `features::triple_barrier_labeler` | `label_classification` | Discrepancia metodológica: `label_classification` mantiene entrada al Close y `valid_n = n - horizon`, desacoplada de la semántica del motor. |

---

## 3. ANÁLISIS FORENSE METICULOSO DE CADA HALLAZGO

---

### [BUG-30] ALTO: Corrupción de Datos Históricos por Desviación de Reloj Local (NTP Drift)

- **Ubicación:** `src/downloader/binance_client.rs:56-61` y `src/downloader/downloader_service.rs:36-48`

#### Diagnóstico del Problema:
En `binance_client.rs`:
```rust
let now_ms = chrono::Utc::now().timestamp_millis();
let klines: Vec<Kline> = raw_list
    .into_iter()
    .filter(|raw| raw.close_time() <= now_ms)
    .filter_map(|raw| raw.to_kline())
    .collect();
return Ok(klines);
```
Este filtro busca evitar Look-Ahead Bias descartando la vela que se encuentra actualmente en formación en Binance. Sin embargo, depende **exclusivamente del reloj del sistema operativo local** (`Utc::now()`).

Existen dos escenarios de desviación temporal (NTP drift):
1. **Reloj Local Adelantado (Fast Clock):** Si el reloj de la máquina de Allianz está 30 segundos o 2 minutos adelantado respecto a los servidores de Binance (situación común en Windows sin sincronización horaria forzada), la vela viva en formación satisface `raw.close_time() <= now_ms`. El cliente la interpreta como una vela cerrada y la guarda en SQLite con volumen parcial e incompleto y precio de cierre flotante.
   - En `downloader_service.rs:36-38`:
     ```rust
     let last_ts = KlineRepository::get_latest_timestamp(conn, tf)?;
     let start_time = match last_ts {
         Some(ts) => Some(ts + 1),
         ...
     ```
     La próxima sincronización pedirá velas a partir de `last_ts + 1`. ¡La vela incompleta **jamás es corregida ni re-descargada**, quedando permanentemente corrupta en la base de datos institucional!
2. **Reloj Local Atrasado (Slow Clock):** Si el reloj local está 5 segundos atrasado, una vela que acaba de cerrar en Binance tiene `close_time > now_ms`. Es descartada por el filtro, lo que provoca que el lote devuelva menos de 1000 velas o quede vacío, finalizando la sincronización prematuramente y dejando un vacío temporal.

#### Solución Recomendada:
1. Validar la hora del servidor a través de la cabecera HTTP `Date` de la respuesta de Binance o del endpoint `/api/v3/time` para calcular el `server_time_offset`.
2. En `downloader_service.rs`, al iniciar una sincronización, retroceder una vela (`last_ts` en lugar de `last_ts + 1`) y actualizar mediante `INSERT OR REPLACE` para re-sincronizar siempre la última vela cerrada por si quedó trunca.

---

### [BUG-31] MEDIO: Ruptura de Creación de Archivos en Windows por Caracteres No Sanitizados en `model_name`

- **Ubicación:** `src/dashboard/dashboard_generator.rs:22-28` y `src/dashboard/html_template.rs:168, 351`

#### Diagnóstico del Problema:
En `dashboard_generator.rs`:
```rust
let filename = format!(
    "dashboard_{}_{}_{}.html",
    wf.model_name.to_lowercase().replace(' ', "_"),
    wf.timeframe,
    now_str
);
let path = dir.join(&filename);
let mut file = File::create(&path)?;
```
Si el usuario o el pipeline evalúa un modelo cuyo nombre contiene caracteres reservados del sistema de archivos como:
- Barras: `Baseline EMA 9/21`
- Dos puntos: `DecisionTree: Depth=4`
- Caracteres especiales: `AutoML (Linear/Tree)`
En sistemas Windows, los caracteres `\`, `/`, `:`, `*`, `?`, `"`, `<`, `>`, `|` están terminantemente prohibidos en nombres de archivo. `File::create(&path)` retorna un error de E/S del sistema operativo (`ERROR_INVALID_NAME`: *El nombre de archivo o la sintaxis de la etiqueta del volumen no son válidos*), provocando el colapso de la generación del dashboard.

Adicionalmente, en `html_template.rs`:
```rust
<title>GBT Dashboard - {model_name} [{timeframe}] - IS & OOS</title>
<h1>🚀 GBT Dashboard Institucional: {model_name}</h1>
```
Si `model_name` o `t.exit_reason` contienen entidades HTML (`<`, `>`, `&`), se inyectan en el DOM sin escapar, corrompiendo la tabla o el título.

#### Solución Recomendada:
Sanitizar rigurosamente `filename` reemplazando cualquier carácter no alfanumérico por un guión bajo:
```rust
let safe_model = wf.model_name
    .chars()
    .map(|c| if c.is_alphanumeric() { c } else { '_' })
    .collect::<String>()
    .to_lowercase();
```

---

### [BUG-32] MEDIO: Desalineación de MFE / MAE en Salidas en Open por Señal

- **Ubicación:** `src/backtest/execution_engine.rs:34-44`

#### Diagnóstico del Problema:
En `process_signal_at_open`:
```rust
if should_close {
    let trade = Self::close_position(
        portfolio,
        config,
        pos,
        current_candle.open,
        current_candle.timestamp,
        candle_idx,
        "Signal Reversal / Exit".to_string(),
    );
    completed_trade = Some(trade);
}
```
La posición `pos` fue actualizada por última vez en la vela $t-1$ mediante `check_intra_candle_exits(klines[t-1])`.
En la vela $t$, el precio abre en `current_candle.open`. Si el mercado experimenta una brecha de apertura (Gap Up o Gap Down) respecto al cierre de $t-1$, el precio de salida efectivo es `current_candle.open * (1.0 \pm slippage)`.
Sin embargo, `pos.update_extremes` **no fue invocado** con `current_candle.open`.

#### Consecuencia Financiera:
Si una posición Long entró en 100.0, su máximo previo fue 102.0, y en la vela $t$ abre con un fuerte Gap Alcista a 106.0 liquidando la posición:
`pos.max_favorable_price` permanece en 102.0 (+2.0%).
El cálculo de MFE en `close_position`:
`mfe = (pos.max_favorable_price - pos.entry_price) / pos.entry_price * 100.0;`
reporta un MFE de +2.0%, ¡a pesar de que la operación cerró con una ganancia real de +6.0%! El MFE queda por debajo del retorno real del trade.

#### Solución Recomendada:
Actualizar los extremos de la posición con `current_candle.open` antes de cerrar:
```rust
let mut pos = pos;
pos.update_extremes(current_candle.open, current_candle.open);
```

---

### [BUG-33] ALTO: Falta de Acotamiento Inferior y Bounding Incompleto en Señales Dinámicas de SL y TP

- **Ubicación:** `src/backtest/execution_engine.rs:124-153`

#### Diagnóstico del Problema:
Examinemos la lógica de Stop Loss dinámico provisto por la señal:
```rust
let stop_loss = signal
    .stop_loss_price
    .map(|sl| match side {
        PositionSide::Long => sl.min(current_candle.open * (1.0 - 0.0001)),
        PositionSide::Short => sl.max(current_candle.open * (1.0 + 0.0001)),
    })
```
1. **Caso Long con SL Inválido o Negativo:** Si la señal emite un valor anómalo como `stop_loss_price = Some(0.0)` o un valor negativo:
   `sl.min(current_candle.open * 0.9999)` selecciona `0.0`.
   El Stop Loss queda configurado en `$0.00`. Dado que en finanzas los precios de activos son positivos, la condición `low <= 0.0` nunca se cumple. La posición queda desprovista de Stop Loss en el mercado.
2. **Caso Short con SL Nulo o Menor a Open:** Si la señal Short emite `stop_loss_price = Some(0.0)` o un precio por debajo del open (ej: 95.0 cuando open es 100.0):
   `sl.max(current_candle.open * 1.0001)` lo fuerza a `open * 1.0001`.
   En el tick de apertura o primera fluctuación normal hacia arriba de la vela, `high >= open * 1.0001` se dispara de inmediato, liquidando la posición Short en la misma vela por error de señal.
3. **Take Profit sin Bounding contra Open:**
   ```rust
   let take_profit = signal.take_profit_price.or_else(|| { ... });
   ```
   Si la señal emite un Take Profit dinámico erróneo (para Long menor a Open, o para Short mayor a Open):
   En `check_intra_candle_exits`:
   `hit_tp = current_candle.high >= tp;` (para Long). Si `tp <= open`, como en toda vela `high >= open`, ¡la posición se cierra instantáneamente con Take Profit falso en la vela 0!

#### Solución Recomendada:
Validar que `signal.stop_loss_price` y `signal.take_profit_price` sean finitos, positivos y se encuentren en el lado correcto del precio de ejecución:
```rust
let stop_loss = signal
    .stop_loss_price
    .filter(|&sl| sl.is_finite() && sl > 0.0)
    .and_then(|sl| match side {
        PositionSide::Long => {
            if sl < current_candle.open {
                Some(sl.min(current_candle.open * (1.0 - 0.0001)))
            } else {
                None // SL ilógico descartado, cae al fixed_sl de respaldo
            }
        }
        PositionSide::Short => {
            if sl > current_candle.open {
                Some(sl.max(current_candle.open * (1.0 + 0.0001)))
            } else {
                None
            }
        }
    })
    ...
```

---

### [BUG-34] MEDIO: Ambigüedad y Solapamiento de Índices en `EquityPoint.candle_idx` en Validación Out-of-Sample

- **Ubicación:** `src/backtest/engine.rs:101-108` y `src/dashboard/equity_curve_exporter.rs:48-62`

#### Diagnóstico del Problema:
En `run_with_warmup`:
```rust
for i in 0..klines.len() {
    if i < warmup_bars {
        ...
        continue;
    }
    ...
    equity_curve.push(EquityPoint::new(
        current_candle.timestamp,
        i,
        portfolio.cash_balance,
        current_eq,
        dd_nominal,
        dd_pct,
    ));
}
```
En Out-of-Sample (OOS), `klines` es `split.test_warmup_klines` y `warmup_bars` es 50.
El primer punto de la curva OOS tiene `candle_idx = 50`. El siguiente tiene 51, 52...
Mientras tanto, en In-Sample (IS), `run` ejecuta con `warmup_bars = 0`, asignando `candle_idx` desde 0 hasta `train_end` (ej. 700).

#### Consecuencia:
Si Allianz o un script cuantitativo consume los archivos CSV exportados (`equity_treeboost_1h_IS.csv` y `equity_treeboost_1h_OOS.csv`) y los une por `candle_idx`, encontrará que las filas `50..300` existen duplicadas en ambos conjuntos, pero correspondientes a fechas totalmente distintas (ej. año 2021 vs año 2023).

#### Solución Recomendada:
Normalizar `candle_idx` en `run_with_warmup` para que refleje el índice ordinal continuo del período de prueba (restando `warmup_bars`), o bien documentar explícitamente en el CSV el índice relativo (`test_bar_idx = i - warmup_bars`).

---

### [BUG-35] MEDIO: Propagación de Floats No Finitos (`NaN`/`Inf`) desde Inferencia de `TreeBoostModel`

- **Ubicación:** `src/models/treeboost_model.rs:97-101, 132`

#### Diagnóstico del Problema:
En `treeboost_model.rs`:
```rust
pub fn predict_batch(&self, klines: &[Kline]) -> Result<Vec<f64>, String> {
    ...
    let preds = model
        .predict(&df)
        .map_err(|e| format!("Error en inferencia batch de TreeBoost: {:?}", e))?;

    Ok(preds.into_iter().map(|v| v as f64).collect())
}
```
En `treeboost`, `model.predict(&df)` retorna un `Vec<f32>`.
Si por alguna condición de volatilidad extrema, convergencia numérica o datos planos el modelo devuelve un `f32::NAN` o desborda a `f32::INFINITY`, la conversión directa `v as f64` produce `f64::NAN` o `f64::INFINITY`.

Posteriormente, en `ThresholdOptimizer::optimize_in_sample`:
`let (opt_long, opt_short, best_fit) = ThresholdOptimizer::optimize_in_sample(&is_preds, ...);`
Las comparaciones IEEE 754 de punto flotante frente a `NaN` retornan `false`, corrompiendo silenciosamente el ordenamiento de los umbrales candidatos o provocando selecciones subóptimas.

#### Solución Recomendada:
Aplicar saneamiento defensivo inmediato en la conversión:
```rust
Ok(preds
    .into_iter()
    .map(|v| if v.is_finite() { v as f64 } else { 0.0 })
    .collect())
```

---

### [BUG-36] BAJO: Silenciamiento Asimétrico en `TechnicalIndicators::roc`

- **Ubicación:** `src/features/technical_indicators.rs:106-116`

#### Diagnóstico del Problema:
```rust
pub fn roc(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    for i in period..n {
        let prev = values[i - period];
        if prev > 0.0 {
            out[i] = (values[i] - prev) / prev;
        }
    }
    out
}
```
Si `period == 0`, `i - period` es `i`, y calcula `(values[i] - values[i]) / values[i] = 0.0`.
Si `prev <= 0.0`, el bloque `if prev > 0.0` se salta y deja `out[i] = 0.0`.
Si este indicador se utiliza en el futuro sobre una serie que puede oscilar en torno a cero o tomar valores negativos (como un histograma MACD o retornos acumulados), silencia por completo las tasas de cambio de valores negativos.

#### Solución Recomendada:
Comprobar `period == 0` al inicio y usar `prev.abs() > 1e-12`:
```rust
if n == 0 || period == 0 || n <= period {
    return vec![0.0; n];
}
```

---

### [BUG-37] BAJO: Inconsistencia Metodológica Residual en `label_classification`

- **Ubicación:** `src/features/triple_barrier_labeler.rs:27-38`

#### Diagnóstico del Problema:
En la segunda y tercera ronda se corrigió `label_regression` para anclar la entrada al Open de la siguiente vela (`klines[i + 1].open`) y descontar `horizon + 1` muestras válidas.
Sin embargo, `label_classification` mantuvo:
```rust
let valid_n = n - horizon;
...
let entry_price = klines[i].close;
```
Si bien actualmente los modelos de Machine Learning en `src/models/` operan bajo la modalidad de regresión continua para calibrar umbrales óptimos mediante `ThresholdOptimizer`, si en futuras iteraciones se activa un clasificador multinomial (Short/Long/Neutral), este mantendrá el sesgo de entrada en Close frente al motor de ejecución que compra al Open.

#### Solución Recomendada:
Alinear `label_classification` para que utilice `klines[i + 1].open` y `valid_n = n.saturating_sub(horizon + 1)` de forma idéntica a `label_regression`.

---

## 4. VERIFICACIÓN METICULOSA DE LOS 6 FOCOS OBLIGATORIOS

### 1. Verificación de Correcciones Previas (BUG-22 al BUG-29):
- **Estado:** Confirmado y Auditado.
- **Detalle:**
  - Los 5 modelos (`TreeBoost`, `Forust`, `Decision Tree CART`, `Random Forest`, `Native GBDT`) descuentan con exactitud matemática `self.config.horizon + 1` en su entrenamiento. No existen muestras espurias con target nulo `0.0`.
  - En `predict_batch`, no se descuenta `horizon + 1` porque la inferencia requiere estimar el retorno de cada vela causalmente hasta el instante presente ($t \le T$). La lógica es 100% causal y matemáticamente irreprochable.
  - En `execution_engine.rs`, el bounding de Stop Loss contra `Open` previene la liquidación inmediata en velas normales, pero requiere la protección de BUG-33 ante señales dinámicas anómalas.

### 2. Seguridad en Tipos, Memoria y Desbordamiento (Series Cortas y 35 Velas):
- **Estado:** Totalmente Seguro contra Desbordamientos y Pánicos.
- **Detalle:**
  - Con exactamente 35 velas (`klines.len() == 35`), `FeatureMatrix::build` no colapsa ni genera vectores vacíos: ejecuta los 5 indicadores técnicos concurrentemente (períodos 10, 30, 14, 14, 20 $\le 30 \le 35$), asigna exactamente los 35 elementos y construye una matriz plana de $35 \times 11 = 385$ floats.
  - `FeatureMatrix::extract_single_vector(klines, 34)` construye una ventana de exactamente 35 velas (`slice.len() == 35`), extrayendo el vector causal final de 11 características sin desbordamiento de índice. Para `idx < 34`, retorna `None` de forma segura.

### 3. Motores de Inferencia y Modelos:
- **Estado:** Auditado y Verificado.
- **Detalle:**
  - **TreeBoost / Polars:** La construcción del DataFrame utiliza `Series` de floats primitivos sin bitmaps de nulos. Se detectó la necesidad de filtrar `preds` no finitos con `.is_finite()` (BUG-35).
  - **Forust / Rayon:** La inferencia en paralelo `booster.predict(&matrix, true)` opera sobre referencias inmutables `&[f64]`. No existen condiciones de carrera ni contención de cerrojos.
  - **Native GBDT:** El algoritmo de Friedman para regresión MSE está implementado de forma canónica. El shrinkage por learning rate $F_m = F_{m-1} + \nu \cdot h_m$ y el cálculo de residuales $y_i - \hat{y}_i$ son matemáticamente correctos y libres de sesgo.

### 4. Backtest Engine y Contabilidad de Portafolio:
- **Estado:** Validado Financieramente.
- **Detalle:**
  - En reversiones de posición (Long cerrado por señal Short):
    1. La posición Long se cierra al precio `Open * (1.0 - slippage)`.
    2. Se deduce la comisión de salida: `exit_notional * commission_rate`.
    3. Se acredita/debe el PnL neto al balance en efectivo.
    4. La nueva posición Short se abre al precio `Open * (1.0 - slippage)`.
    5. Se deduce la comisión de entrada: `entry_notional * commission_rate`.
    6. Ambas comisiones y ambos slippages son contabilizados íntegramente. Si la posición Short toca su SL en la misma vela, se liquida en el mismo ciclo.
  - Se identificó la discrepancia en MFE/MAE (BUG-32) y la ambigüedad en `candle_idx` de OOS (BUG-34).

### 5. Dashboard y Seguridad Web:
- **Estado:** Seguro contra Inyección JS, pero Vulnerable a Fallas de Ruta en Windows.
- **Detalle:**
  - Los scripts de Chart.js inyectan matrices de valores numéricos puros y fechas UTC (`"YYYY-MM-DD HH:MM"`). No hay inyección directa de código JavaScript ejecutable desde variables de usuario.
  - Se confirmó BUG-31: nombres de modelos con caracteres especiales rompen `File::create` en Windows.

### 6. Sincronización de Datos con Binance:
- **Estado:** Vulnerabilidad Crítica de Sincronización Identificada (BUG-30).
- **Detalle:**
  - La condición `raw.close_time() <= now_ms` vincula la validez de los datos al reloj del hardware local. Un adelanto horario en la máquina local guarda velas en formación en la base de datos de SQLite, corrompiendo irreversiblemente el histórico.

---

## 5. HOJA DE RUTA RECOMENDADA PARA EL SUBAGENTE IMPLEMENTADOR

Para la fase de implementación de la Ronda 4:
1. **Fase 1 (Downloader Resiliente - BUG-30):**
   - En `binance_client.rs`, añadir un margen de guarda de 2 segundos (`raw.close_time() + 2000 <= now_ms`) o sincronizar el timestamp con la cabecera `Date` HTTP de Binance.
   - En `downloader_service.rs`, comenzar la descarga desde `last_ts` (no `last_ts + 1`) con `INSERT OR REPLACE` para revalidar y sobreescribir la última vela guardada.
2. **Fase 2 (Generador de Dashboard y Sanitización de Rutas - BUG-31):**
   - Sanitizar `wf.model_name` en `DashboardGenerator` filtrando caracteres no alfanuméricos (`[^a-zA-Z0-9_-]`).
   - Escapar entidades XML/HTML en `html_template.rs`.
3. **Fase 3 (Motor de Ejecución y Stops Dinámicos - BUG-32, BUG-33):**
   - En `ExecutionEngine::process_signal_at_open`, actualizar extremos con `current_candle.open` antes de cerrar posiciones.
   - Validar que `signal.stop_loss_price` y `signal.take_profit_price` sean finitos, positivos y consistentes con la dirección de la posición.
4. **Fase 4 (Normalización de Índices OOS y Filtro TreeBoost - BUG-34, BUG-35):**
   - Normalizar `candle_idx` en `run_with_warmup` o diferenciar `test_bar_idx` en los reportes CSV.
   - Filtrar con `.is_finite()` las predicciones de `TreeBoostModel`.
5. **Fase 5 (Refinamiento de Indicadores y Etiquetas - BUG-36, BUG-37):**
   - Añadir salvaguardas en `TechnicalIndicators::roc` y sincronizar `label_classification` con `label_regression`.
6. **Fase 6 (Verificación Estática):**
   - Ejecutar `cargo check` y confirmar ausencia absoluta de advertencias y errores de compilación.

# AUDITORÍA FORENSE CUANTITATIVA INSTITUCIONAL DE QUINTA RONDA - BUGS, MATEMÁTICAS Y CAUSALIDAD (GBT)
**Destinatario:** Allianz  
**Auditor:** Subagente Auditor Cuantitativo Forense (Ronda 5)  
**Fecha:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Rigor Matemático Institucional, Contabilidad Financiera Exacta, Ultra Modularidad en Rust.

---

## 1. RESUMEN EJECUTIVO DE LA QUINTA RONDA

Estimado **Allianz**, como Director Cuantitativo Forense, he llevado a cabo una auditoría implacable y exhaustiva sobre la totalidad del código fuente en Rust del ecosistema **GBT** (`c:\Users\Allianz\Downloads\Antigravity\GBT`). Tras 4 rondas de desarrollo y correcciones previas (BUG-01 al BUG-37), esta quinta ronda se enfocó en auditar las interacciones de los subsistemas acoplados, la contabilidad financiera de doble partida en ejecución de órdenes y comisiones, los límites de causalidad temporal ($t \le T$) en salidas anómalas (bancarrota), el horizonte temporal de etiquetado en series temporales, la sanitización de exportación en sistemas Windows y la seguridad numérica ante valores no finitos (`NaN`/`Infinity`).

El sistema exhibe un nivel sobresaliente de ingeniería:
- La paralelización de indicadores técnicos y generación de matrices de características con Rayon en 20 hilos es completamente determinista y causal.
- La ejecución en apertura de vela ($t+1$) con slippage y comprobaciones intra-vela de extremos High/Low respeta escrupulosamente la flecha del tiempo.
- La optimización de umbrales en In-Sample (`ThresholdOptimizer`) se encuentra estrictamente aislada del conjunto Out-of-Sample.

No obstante, esta auditoría forense profunda ha descubierto **9 nuevos hallazgos críticos y cuantitativos** (2 Críticos/Altos, 5 Medios y 2 Bajos) catalogados desde **BUG-38** hasta **BUG-46**, los cuales deben ser corregidos para garantizar un motor de grado institucional inmune a distorsiones contables y sesgos temporales:

1. **Omisión de Comisión de Entrada en `CompletedTrade` y Subestimación del 50% de Comisiones Totales [BUG-38 - CRÍTICO/ALTO]:** Al abrir una posición se deduce la comisión del capital, pero `CompletedTrade` solo registra la comisión de salida, inflando el `net_pnl` de cada operación y reportando en `MetricsCalculator` exactamente la mitad de las comisiones reales pagadas.
2. **Look-Ahead Bias Temporal en Cierre Forzado por Bancarrota [BUG-39 - CRÍTICO/ALTO]:** Si la cuenta quiebra en la vela $i$ (ej. vela 25 de 1000), el motor rompe el bucle pero fuera de él liquida la posición con `klines.last()`, tomando el precio y la fecha del futuro lejano (vela 999), violando la causalidad $t \le T$.
3. **Off-by-One en Horizonte de `TripleBarrierLabeler` y Omisión de Vela de Entrada en Clasificación [BUG-40 - MEDIO]:** La salida en `label_regression` evalúa $H+1$ velas en lugar de $H$ (ej. horizonte 1 evalúa 2 velas). En `label_classification` se omite por completo la vela `i + 1` en la verificación de barreras.
4. **Bypass de Validación Numérica en `Kline::new` ante `NaN`/`Infinity` [BUG-41 - MEDIO]:** Las comparaciones IEEE 754 de orden con `NaN` devuelven `false`, permitiendo que velas con precios corruptos pasen como válidas.
5. **Omisión de Sanitización de `model_name` en `EquityCurveExporter` [BUG-42 - MEDIO]:** Guardar curvas en CSV/JSON con nombres de modelos que contengan caracteres prohibidos (`/`, `:`, `\`) falla en sistemas Windows con `ERROR_INVALID_NAME`.
6. **Ausencia de Escape de Entidades HTML en `HtmlTemplate::render` [BUG-43 - BAJO]:** Nombres de modelos o motivos de salida con `<`, `>`, `&` se inyectan sin escapar en el DOM del reporte web.
7. **Distorsión de Racha de Pérdidas Consecutivas ante Breakeven [BUG-44 - BAJO]:** Operaciones con `net_pnl == 0.0` no reinician `current_consecutive_losses`, uniendo rachas independientes.
8. **Riesgo de Pánico Fatal en Inferencia por `NaN.clamp()` en Modelos Secundarios [BUG-45 - MEDIO]:** `Forust`, `CART`, `Random Forest` y `Native GBDT` no sanean predicciones con `.is_finite()`, arriesgando pánicos en Rust al invocar `.clamp()` sobre `NaN`.
9. **Propagación de Quiebra No Acotada en Transición IS -> OOS [BUG-46 - MEDIO]:** Si In-Sample termina con capital $\le 0$, OOS se inicializa con capital insolvente, generando división por cero (`0.0 / 0.0 = NaN`).

---

## 2. TABLA RESUMEN DE HALLAZGOS (QUINTA RONDA)

| ID | Severidad | Módulo / Archivo | Componente / Función | Descripción Resumida |
|---|---|---|---|---|
| **BUG-38** | **ALTO** | `backtest::execution_engine` / `types::position` / `metrics_calculator` | `process_signal_at_open`, `close_position`, `MetricsCalculator::calculate` | Omisión de `entry_commission` en `CompletedTrade::net_pnl`, `commissions` y `return_pct`. Subestimación del 50% de comisiones totales en métricas y desalineación contable $\sum \text{trade.net\_pnl} \ne \Delta \text{Equity}$. |
| **BUG-39** | **ALTO** | `backtest::engine` | `run_with_warmup` | Fuga temporal (Look-Ahead Bias $t \le T$): al quebrar la cuenta en la vela $i$, el motor sale del bucle y liquida con `klines.last().close` (vela final del histórico, en el futuro), resucitando cuentas quebradas. |
| **BUG-40** | **MEDIO** | `features::triple_barrier_labeler` / `models::*` | `label_regression`, `label_classification`, `train` en los 5 modelos | Off-by-one: salida en `i + 1 + horizon` evalúa $H+1$ velas en vez de $H$. `label_classification` omite la vela de apertura `i + 1`. Desalineación de `valid_rows` en los 5 modelos. |
| **BUG-41** | **MEDIO** | `types::kline` | `Kline::new` | Falla en validación IEEE 754: comparaciones `<`, `<=` contra `NaN` devuelven `false`, permitiendo instanciar velas corruptas con `NaN` o `Infinity`. |
| **BUG-42** | **MEDIO** | `dashboard::equity_curve_exporter` | `save_json` y `save_csv` | Omisión de sanitización de `model_name`: nombres con `/`, `:`, `\` o `?` fallan con error de sistema operativo en Windows (`ERROR_INVALID_NAME`). |
| **BUG-43** | **BAJO** | `dashboard::html_template` | `render` | Inyección de variables de usuario (`model_name`, `exit_reason`) sin escapar entidades HTML (`<`, `>`, `&`), con riesgo de romper el DOM. |
| **BUG-44** | **BAJO** | `backtest::metrics_calculator` | `calculate` | Trades con `net_pnl == 0.0` (breakeven) no reinician `current_consecutive_losses`, concatenando falsamente pérdidas separadas en una sola racha. |
| **BUG-45** | **MEDIO** | `models::*` / `models::*_strategy` | Inferencia en `Forust`, `CART`, `Random Forest`, `Native GBDT` | Falta de saneamiento `.is_finite()` en predicciones de los 4 modelos no-TreeBoost, provocando pánico en Rust al invocar `.clamp()` sobre `NaN`. |
| **BUG-46** | **MEDIO** | `cli::*_handler` | `run_walk_forward_backtest` | Propagación de quiebra no acotada: si IS termina con `final_capital <= 0.0`, OOS arranca con capital nulo/negativo, generando división por cero y `NaN` en métricas. |

---

## 3. ANÁLISIS FORENSE METICULOSO DE CADA HALLAZGO

---

### [BUG-38] ALTO: Omisión de la Comisión de Entrada en `CompletedTrade` y Subestimación del 50% de Comisiones

- **Ubicación:** 
  - `src/types/position.rs:10-22`
  - `src/backtest/execution_engine.rs:138-140, 315-335`
  - `src/backtest/metrics_calculator.rs:38, 41-53`

#### Diagnóstico del Problema:
Examinemos el ciclo de vida contable de una operación:
1. En `ExecutionEngine::process_signal_at_open`:
   ```rust
   let commission = notional_target * config.commission_rate;
   ...
   portfolio.cash_balance -= commission;
   portfolio.total_commissions_paid += commission;
   ```
   La comisión de apertura (entrada) es deducida inmediatamente del efectivo del portafolio. Sin embargo, la estructura `Position` **no almacena** el monto de comisión de entrada cobrado.
2. En `ExecutionEngine::close_position`:
   ```rust
   let exit_notional = exit_price * pos.quantity;
   let commission = exit_notional * config.commission_rate; // SOLO comisión de salida!
   ...
   let net_pnl = if let Some(isolated_margin) = pos.isolated_margin {
       let uncapped_pnl = gross_pnl - commission;
       ...
   } else {
       gross_pnl - commission
   };
   portfolio.cash_balance += net_pnl;
   portfolio.total_commissions_paid += commission;
   ```
   La variable `commission` calculada en `close_position` es **exclusivamente la comisión de salida**.
   El `net_pnl` del trade se calcula como `gross_pnl - exit_commission`.
   ¡La comisión de entrada jamás es descontada de `CompletedTrade.net_pnl`!
3. Al instanciar `CompletedTrade`:
   ```rust
   CompletedTrade {
       ...
       gross_pnl,
       commissions: commission, // SOLO comisión de salida!
       net_pnl,                 // Omite la comisión de entrada!
       return_pct,              // Inflado!
       ...
   }
   ```
4. En `MetricsCalculator::calculate`:
   ```rust
   let mut total_commissions = 0.0;
   for t in trades {
       total_commissions += t.commissions;
       ...
   ```
   `total_commissions` suma únicamente `t.commissions`, omitiendo todas las comisiones de entrada.

#### Consecuencia Financiera y Cuantitativa:
1. **Subestimación del 50% en Comisiones:** En un backtest de 1,000 operaciones donde se pagaron \$2,000 en comisiones totales (1,000 de entrada + 1,000 de salida), el reporte de métricas institucionales informa que se pagaron solo \$1,000.
2. **Sobreestimación del Retorno por Trade:** Una operación en la que el precio no se movió (`gross_pnl = 0.0`) y pagó \$10 al entrar y \$10 al salir tiene una pérdida real de -\$20. En el registro de trades aparece con `net_pnl = -$10.0`, ocultando la mitad de la pérdida real.
3. **Ruptura de la Ecuación Contable:**
   $\sum \text{trade.net\_pnl} \ne \text{final\_capital} - \text{initial\_capital}$.
   El saldo final del portafolio sí descuenta ambas comisiones, pero la suma de los trades reporta un valor superior, destruyendo la reconciliación contable de auditoría.

#### Solución Recomendada:
1. Agregar el campo `entry_commission: f64` a `Position`.
2. Al abrir la posición en `process_signal_at_open`, guardar `pos.entry_commission = commission`.
3. En `close_position`:
   ```rust
   let exit_commission = exit_notional * config.commission_rate;
   let total_trade_commissions = pos.entry_commission + exit_commission;
   let uncapped_pnl = gross_pnl - total_trade_commissions;
   let net_pnl = if let Some(isolated_margin) = pos.isolated_margin {
       if uncapped_pnl < -isolated_margin { -isolated_margin } else { uncapped_pnl }
   } else {
       uncapped_pnl
   };
   // Como la comisión de entrada ya fue restada de cash_balance al abrir, al balance solo se acredita (gross_pnl - exit_commission):
   portfolio.cash_balance += gross_pnl - exit_commission;
   portfolio.total_commissions_paid += exit_commission;
   ```
4. Asignar en `CompletedTrade`: `commissions: total_trade_commissions`, `net_pnl`.

---

### [BUG-39] ALTO: Ruptura de Causalidad Temporal (Look-Ahead Bias $t \le T$) en Salida por Bancarrota

- **Ubicación:** `src/backtest/engine.rs:112-136`

#### Diagnóstico del Problema:
En `BacktestEngine::run_with_warmup`:
```rust
for i in 0..klines.len() {
    let current_candle = &klines[i];
    ...
    // Si la cuenta llega a balance <= 0 (bancarrota), detener la simulación
    if portfolio.is_bankrupt(current_candle.close) {
        break;
    }
    ...
}

// Cierre de cualquier posición remanente en la última vela
if let Some(pos) = portfolio.active_position.take() {
    let last_candle = match klines.last() {
        Some(c) => c,
        None => return Err("Klines vacío".to_string()),
    };
    let trade = ExecutionEngine::close_position(
        &mut portfolio,
        &self.config,
        pos,
        last_candle.close,
        last_candle.timestamp,
        klines.len() - 1,
        "End of Data Horizon".to_string(),
    );
    trades.push(trade);
```
Si la cuenta cae en quiebra en la vela $i=25$ de una serie temporal de 1,000 velas:
1. `portfolio.is_bankrupt` evalúa `true`.
2. Se ejecuta `break;` inmediatamente, saliendo del bucle en $i=25$.
3. Fuera del bucle, la posición abierta remanente es liquidada con `klines.last()`, que es la vela $i=999$ (¡meses o años en el futuro!).
4. La operación se cierra al precio `klines[999].close`, con fecha `klines[999].timestamp` y razón `"End of Data Horizon"`.

#### Consecuencia Financiera y Cuantitativa:
1. **Look-Ahead Bias Violento:** El sistema financiero accede a datos posteriores al instante temporal simulado ($t > T$).
2. **Resurrección Ficticia de Cuentas Quebradas:** Si el activo experimentó un ciclo alcista posterior, la posición quebrada en la vela 25 se liquida a los precios récord de la vela 999, generando utilidades espurias y "reviviendo" la cuenta, falseando por completo las métricas de quiebra y riesgo institucional.
3. **Desincronización de Series:** La curva de balance `equity_curve` se detiene en la vela 25, pero el trade registrado indica que cerró en la vela 999 con fecha falsa.

#### Solución Recomendada:
Llevar un registro del último índice y vela evaluada `last_evaluated_candle`. Si el bucle termina (por bancarrota o al agotar datos), cerrar la posición usando exactamente `last_evaluated_candle`, con motivo `"Bankruptcy (Account Insolvent)"` si quebró, o `"End of Data Horizon"` si completó el histórico:
```rust
let mut last_eval_idx = 0;
let mut is_bankrupt_exit = false;

for i in 0..klines.len() {
    last_eval_idx = i;
    ...
    if portfolio.is_bankrupt(current_candle.close) {
        is_bankrupt_exit = true;
        break;
    }
    ...
}

if let Some(pos) = portfolio.active_position.take() {
    let exit_candle = &klines[last_eval_idx];
    let reason = if is_bankrupt_exit {
        "Bankruptcy (Account Insolvent)".to_string()
    } else {
        "End of Data Horizon".to_string()
    };
    let trade = ExecutionEngine::close_position(
        &mut portfolio,
        &self.config,
        pos,
        exit_candle.close,
        exit_candle.timestamp,
        last_eval_idx,
        reason,
    );
    trades.push(trade);
    ...
}
```

---

### [BUG-40] MEDIO: Off-by-One en el Horizonte de `TripleBarrierLabeler` y Omisión de Vela de Entrada

- **Ubicación:** 
  - `src/features/triple_barrier_labeler.rs:27-50, 69-80`
  - `src/models/treeboost_model.rs:32`
  - `src/models/forust_model.rs:48`
  - `src/models/decision_tree/cart_model.rs:32`
  - `src/models/random_forest/forest_model.rs:34`
  - `src/models/native_gbdt/native_model.rs:33`

#### Diagnóstico del Problema:
1. **En `label_regression`:**
   ```rust
   let valid_n = n.saturating_sub(horizon + 1);
   ...
   let p_entry = klines[i + 1].open;
   let p_exit = klines[i + 1 + horizon].close;
   ```
   La señal se emite al cierre de $i$. La orden se ejecuta en la apertura de $i + 1$.
   La vela $i + 1$ es la **primera vela** en la que la posición está viva.
   Si el horizonte de mantenimiento nominal configurado es $H=1$ vela:
   El código evalúa $p\_exit = klines[i + 1 + 1].close = klines[i + 2].close$.
   ¡Esto abarca la vela $i+1$ y la vela $i+2$, resultando en un horizonte real de 2 velas!
   Para un horizonte $H=5$, evalúa $i + 1 + 5 = i + 6$, abarcando 6 velas completas ($i+1, i+2, i+3, i+4, i+5, i+6$).
   El cierre tras $H$ velas transcurridas a partir de la apertura de $i + 1$ es estrictamente `klines[i + horizon].close`.
2. **En `label_classification`:**
   ```rust
   for j in 1..=horizon {
       let candle = &klines[i + 1 + j];
       let hit_upper = candle.high >= upper_barrier;
       let hit_lower = candle.low <= lower_barrier;
       ...
   }
   ```
   El bucle empieza en $j=1$, evaluando `klines[i + 2]`.
   ¡La vela de entrada $i + 1$ (donde la posición abrió en el Open) es completamente ignorada!
   Si en la misma vela de apertura $i + 1$ el mercado se desploma o dispara tocando la barrera, la etiqueta no lo detecta.
3. **Muestras Válidas:**
   Con salida en `klines[i + horizon].close`, el índice máximo evaluado para $i = n - 1 - horizon$ es $(n - 1 - horizon) + horizon = n - 1$. Por lo tanto, el número de muestras causales válidas es exactamente `valid_n = n.saturating_sub(horizon)`.

#### Consecuencia:
Desalineación metodológica: el modelo aprende patrones correspondientes a un horizonte $H+1$ mientras el usuario y la configuración especifican $H$. Además, en clasificación se omiten las liquidaciones que ocurren durante la vela de apertura.

#### Solución Recomendada:
1. En `label_regression`: fijar `let p_exit = klines[i + horizon].close;` y `let valid_n = n.saturating_sub(horizon);`.
2. En `label_classification`: iterar `for j in 0..horizon { let candle = &klines[i + 1 + j]; ... }` y `valid_n = n.saturating_sub(horizon)`.
3. Sincronizar los 5 modelos (`valid_rows = rows.saturating_sub(self.config.horizon)`).

---

### [BUG-41] MEDIO: Bypass de Validación Numérica en `Kline::new` ante `NaN` e `Infinity`

- **Ubicación:** `src/types/kline.rs:14-24`

#### Diagnóstico del Problema:
```rust
impl Kline {
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Result<Self, String> {
        if high < low {
            return Err(...);
        }
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return Err(...);
        }
        if volume < 0.0 {
            return Err(...);
        }
        Ok(Self { timestamp, open, high, low, close, volume })
    }
}
```
Bajo el estándar IEEE 754 de punto flotante en Rust:
- `f64::NAN < low` -> `false`
- `f64::NAN <= 0.0` -> `false`
- `f64::NAN < 0.0` -> `false`
Si un precio o volumen recibido de la API o la base de datos es `NaN`:
¡Ninguno de los `if` se activa! `Kline::new` retorna exitosamente `Ok(Kline { open: NaN, ... })`.

#### Consecuencia Cuantitativa:
El `NaN` se propaga de forma invisible hacia los indicadores técnicos (EMA, RSI, ATR), convirtiendo vectores enteros de características en `NaN`. Posteriormente, los árboles de decisión y optimizadores sufren corrupción o pánicos.

#### Solución Recomendada:
Validar explícitamente la finitud de todos los campos numéricos al inicio de `Kline::new`:
```rust
if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() || !volume.is_finite() {
    return Err("Los valores de la vela deben ser números finitos (no NaN ni Inf)".to_string());
}
```

---

### [BUG-42] MEDIO: Omisión de Sanitización de `model_name` en `EquityCurveExporter`

- **Ubicación:** `src/dashboard/equity_curve_exporter.rs:25, 44`

#### Diagnóstico del Problema:
En `save_json` y `save_csv`:
```rust
let filename = format!("equity_{}_{}_{}_{}.json", model_name, timeframe, segment, now_str);
...
let filename = format!("equity_{}_{}_{}_{}.csv", model_name, timeframe, segment, now_str);
```
En la Ronda 4 se introdujo `DashboardGenerator::sanitize_model_name`, pero en `EquityCurveExporter` se omitió por completo.
Si el usuario evalúa una estrategia cuyo nombre incluye caracteres especiales (ej: `"Baseline EMA (9/21)"`, `"Model: CART Depth=4"`, `"Tree/Boost"`), la ruta generada contiene caracteres ilegales en Windows (`/`, `:`, `\`, `*`, `?`, `"`, `<`, `>`, `|`).
`File::create(&path)` retorna de inmediato `std::io::Error` (`ERROR_INVALID_NAME`), abortando el proceso.

#### Solución Recomendada:
Crear o invocar la función de sanitización en `EquityCurveExporter`:
```rust
fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            _ => c,
        })
        .collect()
}
```

---

### [BUG-43] BAJO: Ausencia de Escape de Entidades HTML en `HtmlTemplate::render`

- **Ubicación:** `src/dashboard/html_template.rs:110, 158, 168, 199`

#### Diagnóstico del Problema:
En `HtmlTemplate::render`:
```rust
<title>GBT Dashboard - {model_name} [{timeframe}] - IS & OOS</title>
...
<td>{t.exit_reason}</td>
```
Los textos `model_name` y `t.exit_reason` (ej. motivos de salida como `"Stop Loss < 100"`) se inyectan en bruto en el HTML generado. Si contienen caracteres reservados como `<`, `>`, o `&`, pueden romper la sintaxis del DOM en el navegador web o alterar el renderizado visual.

#### Solución Recomendada:
Implementar una función utilitaria `escape_html`:
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
y envolver `model_name` y `t.exit_reason` antes de inyectarlos en el HTML.

---

### [BUG-44] BAJO: Distorsión en el Conteo de `max_consecutive_losses` ante Operaciones en Breakeven

- **Ubicación:** `src/backtest/metrics_calculator.rs:41-53`

#### Diagnóstico del Problema:
```rust
for t in trades {
    if t.net_pnl > 0.0 {
        winning_trades += 1;
        gross_profit += t.net_pnl;
        current_consecutive_losses = 0;
    } else if t.net_pnl < 0.0 {
        losing_trades += 1;
        gross_loss += t.net_pnl.abs();
        current_consecutive_losses += 1;
        if current_consecutive_losses > max_consecutive_losses {
            max_consecutive_losses = current_consecutive_losses;
        }
    }
}
```
Si una operación cierra exactamente en equilibrio (`t.net_pnl == 0.0`), ni el bloque `if` ni el `else if` se ejecutan.
`current_consecutive_losses` conserva su valor acumulado anterior.
Si una pérdida ocurrió antes del breakeven y otra pérdida ocurrió después, se cuentan como 2 pérdidas consecutivas, a pesar de haber mediado una operación neutra.

#### Solución Recomendada:
Si la definición de racha de pérdidas es estrictamente consecutiva, una operación no perdedora (`net_pnl >= 0.0`) debe reiniciar el contador:
```rust
if t.net_pnl > 0.0 {
    winning_trades += 1;
    gross_profit += t.net_pnl;
    current_consecutive_losses = 0;
} else if t.net_pnl < 0.0 {
    losing_trades += 1;
    gross_loss += t.net_pnl.abs();
    current_consecutive_losses += 1;
    if current_consecutive_losses > max_consecutive_losses {
        max_consecutive_losses = current_consecutive_losses;
    }
} else {
    current_consecutive_losses = 0; // Breakeven interrumpe la racha de pérdidas
}
```

---

### [BUG-45] MEDIO: Riesgo de Pánico Fatal por `NaN.clamp()` en las Estrategias Secundarias

- **Ubicación:**
  - `src/models/forust_model.rs:83-88, 106-107`
  - `src/models/decision_tree/cart_model.rs:60-66, 75`
  - `src/models/random_forest/forest_model.rs:84-85, 107-108`
  - `src/models/native_gbdt/native_model.rs:98-100, 115-117`
  - `src/models/forust_strategy.rs:70, 73`
  - `src/models/decision_tree/cart_strategy.rs:70, 73`
  - `src/models/random_forest/forest_strategy.rs:70, 73`
  - `src/models/native_gbdt/native_strategy.rs:70, 73`

#### Diagnóstico del Problema:
En la ronda 4, BUG-35 protegió `TreeBoostModel` saneando las predicciones con `.filter(|v| v.is_finite())`.
Sin embargo, en los otros cuatro modelos (`Forust`, `CART`, `Random Forest`, `Native GBDT`), `predict_batch` y `predict_single` devuelven floats crudos sin filtrar finitud.
En sus respectivas estrategias:
```rust
if predicted_return > long_th {
    let strength = (predicted_return / (long_th * 3.0)).clamp(0.1, 1.0);
    ...
```
Si por alguna condición de borde matemática o división por cero interna un modelo emite `NaN`, o si `long_th == 0.0` produciendo `NaN`, la función `.clamp(0.1, 1.0)` de la librería estándar de Rust produce un **pánico fatal e irrecuperable** (`panics if self is NaN`).

#### Solución Recomendada:
1. En los 4 modelos, sanear las predicciones en `predict_batch` y `predict_single`:
   `preds.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect()`
2. En las 4 estrategias, validar defensivamente antes del `clamp`:
   ```rust
   let divisor = (long_th * 3.0).abs();
   let raw_strength = if divisor > 1e-9 { predicted_return / divisor } else { 0.5 };
   let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
   ```

---

### [BUG-46] MEDIO: Propagación de Quiebra No Acotada en Transición IS -> OOS

- **Ubicación:** 
  - `src/cli/baseline_handler.rs:50`
  - `src/cli/cart_handler.rs:64`
  - `src/cli/forust_handler.rs:88`
  - `src/cli/treeboost_handler.rs:88`
  - `src/cli/random_forest_handler.rs:64`
  - `src/cli/native_gbdt_handler.rs:64`

#### Diagnóstico del Problema:
Al concluir la evaluación In-Sample (IS), los handlers ejecutan:
```rust
println!("  -> OOS inicia con el capital final de IS: ${:.2}", is_result.metrics.final_capital);
let mut oos_cfg = bt_cfg.clone();
oos_cfg.initial_capital = is_result.metrics.final_capital;
let engine_oos = BacktestEngine::new(oos_cfg);
```
Si un modelo o conjunto de parámetros sufrió pérdidas catastróficas en IS y terminó con `final_capital <= 0.0`:
1. `oos_cfg.initial_capital` se fija en `0.0` o en un número negativo.
2. Al ejecutar OOS en `engine.rs:112`:
   `portfolio.is_bankrupt(current_candle.close)` evalúa `0.0 <= 0.0` -> `true` en la vela 0.
3. El backtest OOS aborta inmediatamente en la primera vela.
4. En `MetricsCalculator::calculate`:
   `let net_profit_pct = (net_profit / initial_capital) * 100.0;`
   Como `initial_capital == 0.0`, produce `0.0 / 0.0 = NaN`.
   Todas las métricas de OOS se corrompen con `NaN`.

#### Solución Recomendada:
Comprobar si el capital final de IS es solvente antes de iniciar OOS:
```rust
let oos_capital = if is_result.metrics.final_capital > 0.0 {
    is_result.metrics.final_capital
} else {
    println!("  ⚠️ ALERTA: Capital IS agotado o insolvente (${:.2}). Reiniciando OOS con capital base de prueba (${:.2})", 
        is_result.metrics.final_capital, bt_cfg.initial_capital);
    bt_cfg.initial_capital
};
oos_cfg.initial_capital = oos_capital;
```

---

## 4. VERIFICACIÓN METICULOSA POR SUBSISTEMA

### 4.1 Motor de Backtesting y Ejecución (`src/backtest/`):
- **Cero Look-Ahead Bias ($t \le T$):**
  - Se confirmó que las órdenes generadas al cierre de la vela $t$ se ejecutan exclusivamente en la apertura de la vela $t+1$ (`process_signal_at_open`).
  - Se detectó y resolvió conceptualmente BUG-39 (Look-Ahead Bias en salidas por bancarrota usando `klines.last()`).
- **Contabilidad de Portafolio:**
  - Se detectó BUG-38: omisión sistemática de la comisión de apertura en `CompletedTrade` y `MetricsCalculator`.
  - El mecanismo de Margen Aislado (`Isolated`) acota contractualmente la pérdida máxima no realizada y realizada al margen comprometido.
  - La verificación de liquidación intra-vela y brechas de apertura (Gap Open) modela fielmente la microestructura de mercado.

### 4.2 Modelos de Machine Learning (`src/models/`):
- **TreeBoost (AutoML / LinearThenTree):**
  - Implementación con Polars y Series primitivas sin bitmaps de nulos. Se ratificó el saneamiento de predicciones de BUG-35.
- **Forust (LightGBM en Rust puro):**
  - `ObjectiveType::SquaredLoss` opera de forma estable. Se detectó la necesidad de extender el saneamiento `.is_finite()` de predicciones (BUG-45).
- **Decision Tree CART, Random Forest y Native GBDT:**
  - En `CartTree`, el algoritmo de búsqueda de corte por varianza con sumas acumulativas en O(N) es matemáticamente exacto.
  - El submuestreo de variables por nodo en Random Forest (Breiman 2001) utiliza un generador pseudoaleatorio determinista por nodo (Xorshift64) basado en seeds inmutables, garantizando reproducibilidad sin contención de bloqueos.
  - En `NativeGbdtModel`, la inferencia vectorial paralela con Rayon en 20 hilos es completamente thread-safe.

### 4.3 Ingeniería de Características y Etiquetado (`src/features/`):
- **Matriz de Características (`FeatureMatrix`):**
  - El cálculo concurrente de los 5 indicadores técnicos (`ema10`, `ema30`, `rsi14`, `atr14`, `vol_sma20`) con `rayon::join` es 100% causal.
  - El vector causal de cada vela utiliza estrictamente información hasta el cierre de dicha vela.
- **Etiquetado Causal (`TripleBarrierLabeler`):**
  - Se diagnosticó el bug off-by-one en el horizonte de regresión y la omisión de la vela $i+1$ en clasificación (BUG-40).

### 4.4 Ingesta de Datos y Almacenamiento (`src/downloader/`, `src/storage/`):
- **Descargador de Binance:**
  - El filtro anti Look-Ahead descarta velas vivas en formación.
  - La rotación atómica de endpoints en `EndpointPool` previene bloqueos por rate limits HTTP 429.
- **Repositorio SQLite:**
  - `KlineRepository` implementa transacciones atómicas seguras e inserciones idempotentes con `INSERT OR REPLACE INTO klines`.
  - Se detectó la falta de filtro de finitud en `Kline::new` (BUG-41).

### 4.5 Generación de Dashboards y Reportes (`src/dashboard/`, `src/cli/`):
- **Dashboards HTML:**
  - `DashboardGenerator` sanitiza adecuadamente nombres de archivo en disco para Windows.
  - Se detectó la omisión de dicha sanitización en `EquityCurveExporter` (BUG-42) y la falta de escape de entidades HTML (BUG-43).
- **Consistencia de Walk-Forward:**
  - Se detectó el riesgo de propagación de quiebra no acotada al inicializar OOS con capital no positivo (BUG-46).

---

## 5. HOJA DE RUTA RECOMENDADA PARA EL SUBAGENTE IMPLEMENTADOR

Para subsanar de forma definitiva los 9 hallazgos de la Ronda 5:

1. **Fase 1: Corrección Contable de Comisiones y Retornos (BUG-38):**
   - Modificar `Position` en `src/types/position.rs` para incluir `entry_commission: f64`.
   - Modificar `ExecutionEngine::process_signal_at_open` para registrar `pos.entry_commission`.
   - Modificar `ExecutionEngine::close_position` para calcular comisiones totales (`entry + exit`), deducir correctamente el PnL neto y ajustar el saldo en efectivo sin doble deducción.
2. **Fase 2: Blindaje de Causalidad Temporal Anti Look-Ahead (BUG-39):**
   - Modificar `BacktestEngine::run_with_warmup` en `src/backtest/engine.rs` para que las salidas por bancarrota o fin de datos se liquiden estrictamente con la última vela evaluada en el bucle (`klines[last_eval_idx]`), eliminando todo acceso a `klines.last()`.
3. **Fase 3: Sincronización Matemática del Horizonte Temporal (BUG-40):**
   - En `TripleBarrierLabeler`, corregir `p_exit = klines[i + horizon].close` y `valid_n = n.saturating_sub(horizon)`.
   - En `label_classification`, iterar `0..horizon` evaluando desde la vela $i+1$.
   - Sincronizar `valid_rows` en `treeboost_model.rs`, `forust_model.rs`, `cart_model.rs`, `forest_model.rs` y `native_model.rs`.
4. **Fase 4: Validación Numérica y Robustez de Tipos (BUG-41, BUG-45):**
   - Añadir validación `.is_finite()` en `Kline::new`.
   - Saneamiento `.is_finite()` en `ForustModel`, `DecisionTreeModel`, `RandomForestModel` y `NativeGbdtModel`, y protección contra división por cero en las 4 estrategias.
5. **Fase 5: Sanitización en Windows y Escape HTML (BUG-42, BUG-43):**
   - Sanitizar nombres de archivo en `EquityCurveExporter::save_json` y `save_csv`.
   - Escapar entidades HTML en `HtmlTemplate::render`.
6. **Fase 6: Métricas y Resiliencia en Handlers (BUG-44, BUG-46):**
   - Reiniciar `current_consecutive_losses` en operaciones breakeven (`net_pnl >= 0.0`).
   - Validar solvencia de capital antes de inicializar OOS en los 6 handlers CLI.
7. **Fase 7: Verificación Estática:**
   - Ejecutar `cargo check` y `cargo test --no-run` confirmando cero advertencias y cero errores.

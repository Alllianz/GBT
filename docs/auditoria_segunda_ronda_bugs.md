# AUDITORÍA PROFUNDA DE SEGUNDA RONDA - CÓDIGO, EJECUCIÓN Y MODELOS (GBT)
**Destinatario:** Allianz  
**Fecha:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Veracidad Institucional, Ultra Modularidad en Rust.

---

## 1. RESUMEN EJECUTIVO DE LA SEGUNDA RONDA

Tras la resolución exitosa de la primera fase de bugs, se ha llevado a cabo una segunda inspección exhaustiva de nivel forense sobre el 100% de los flujos de ejecución, tipos financieros, controladores de terminal y métodos de inferencia.

Se han descubierto **8 nuevos hallazgos** (3 Críticos, 2 Altos, 2 Medios, 1 Bajo):
- **Bugs Críticos:** Desconexión contable entre equity final y cierre forzoso de trades, relleno de liquidación con gap irreal, y falsa bancarrota en margen aislado por PnL no realizado no acotado.
- **Bugs Altos:** Sesgo de brecha de ejecución en el cálculo del target de entrenamiento (Open vs Close gap), y caída fatal de la app de terminal interactiva ante errores controlables.
- **Bugs Medios:** Lógica de inicialización en optimizador de umbrales y rotación estática del pool de endpoints de Binance.
- **Bugs Bajos:** Formateo cosmético de drawdown negativo (`-0.00%`) en Chart.js.

---

## 2. TABLA RESUMEN DE HALLAZGOS (SEGUNDA RONDA)

| ID | Severidad | Módulo / Componente | Descripción Resumida |
|---|---|---|---|
| **BUG-14** | **CRÍTICO** | `backtest::engine` | Desconexión entre el último punto de la curva de balance y el cierre forzoso final |
| **BUG-15** | **CRÍTICO** | `backtest::execution_engine` | Relleno ficticio de liquidación forzosa ante gaps de apertura en Cross Margin |
| **BUG-16** | **CRÍTICO** | `types::position` | Falsa bancarrota y distorsión de equity en Margen Aislado (`unrealized_pnl` no acotado) |
| **BUG-17** | **ALTO** | `features::triple_barrier_labeler` | Sesgo de ejecución en la etiqueta de regresión (Close a Close vs Open real) |
| **BUG-18** | **ALTO** | `cli::interactive_app` | Caída abrupta (crash) de la terminal interactiva ante cualquier error con operador `?` |
| **BUG-19** | **MEDIO** | `models::threshold_optimizer` | Inicialización de score con `-999.0` y falta de filtro `is_finite()` |
| **BUG-20** | **MEDIO** | `downloader::endpoint_pool` | Rotación inoperativa de endpoints ante rate limits de Binance (cursor no atómico) |
| **BUG-21** | **BAJO** | `dashboard::html_template` | Formateo cosmético de cero negativo (`"-0.00%"`) en tooltips de Chart.js |

---

## 3. ANÁLISIS DETALLADO DE CADA HALLAZGO

---

### [BUG-14] CRÍTICO: Desconexión entre el Último Punto de la Curva de Balance y el Cierre Forzoso Final

- **Ubicación:** `src/backtest/engine.rs:120-145`

#### Diagnóstico del Problema:
Al concluir el bucle temporal del backtest (`for i in 0..klines.len()`):
1. En la última vela (`i = n - 1`), se registra el punto de equity en `equity_curve` con la posición activa aún abierta (`current_eq = cash_balance + unrealized_pnl`). Este valor **NO** contempla slippage de salida ni comisión de cierre.
2. Después del bucle, el motor ejecuta el cierre forzoso:
   ```rust
   if let Some(pos) = portfolio.active_position.take() {
       let trade = ExecutionEngine::close_position(
           &mut portfolio, &self.config, pos, last_candle.close, ...
       );
       trades.push(trade);
   }
   ```
   `close_position` deduce la comisión de salida y aplica el slippage, disminuyendo `portfolio.cash_balance`.
3. Acto seguido se calcula `MetricsCalculator::calculate`:
   ```rust
   let final_capital = equity_curve.last().map(|p| p.equity).unwrap_or(initial_capital);
   ```
   `equity_curve.last()` **NUNCA** fue actualizado con el saldo real resultante del cierre de la posición.

#### Impacto Financiero:
- `metrics.final_capital` y `metrics.net_profit` reportan una cifra inflada que no descuenta los costes de la última salida.
- En la validación Walk-Forward, el conjunto Out-of-Sample arranca con un capital inicial (`oos_cfg.initial_capital`) mayor que el dinero efectivo que realmente posee la cuenta.
- El archivo CSV/JSON de curvas de balance termina en un valor discrepante con el historial de trades.

#### Solución Recomendada:
Al cerrar la posición en la última vela, actualizar inmediatamente `equity_curve.last_mut()` con `portfolio.cash_balance` y recalcular el drawdown final.

---

### [BUG-15] CRÍTICO: Relleno Ficticio de Liquidación Forzosa ante Gaps de Apertura

- **Ubicación:** `src/backtest/execution_engine.rs:170-184`

#### Diagnóstico del Problema:
En `check_intra_candle_exits`:
```rust
match pos.side {
    PositionSide::Long => {
        if current_candle.low <= pos.liquidation_price {
            exit_event = Some((pos.liquidation_price, "Liquidated (Margin Call)"));
        }
    }
    PositionSide::Short => {
        if current_candle.high >= pos.liquidation_price {
            exit_event = Some((pos.liquidation_price, "Liquidated (Margin Call)"));
        }
    }
}
```
Si una posición Long tiene precio de liquidación en `$90.00` y la siguiente vela abre directamente con un Gap bajista en `$80.00` (con máximo en `$81.00` y mínimo en `$75.00`):
La condición `low <= liquidation_price` se cumple. El motor ejecuta la liquidación forzosa a `$90.00 * (1 - slippage) = $89.98`.
En una cuenta con Margen Cruzado (Cross Margin), el exchange liquida la posición al precio de mercado de apertura (`$80.00` o peor), consumiendo el saldo excedente. Llenar la orden a `$90.00` cuando el precio real de mercado nunca superó los `$81.00` es un relleno físicamente imposible.

#### Impacto Financiero:
Oculta pérdidas catastróficas por gap en liquidaciones de futuros, falseando la gestión de riesgo en apalancamiento alto.

#### Solución Recomendada:
Ajustar el precio objetivo de liquidación al gap de apertura:
- Long: `pos.liquidation_price.min(current_candle.open)`.
- Short: `pos.liquidation_price.max(current_candle.open)`.

---

### [BUG-16] CRÍTICO: Falsa Bancarrota y Distorsión de Equity en Margen Aislado (`unrealized_pnl` no acotado)

- **Ubicación:** `src/types/position.rs:72-78`

#### Diagnóstico del Problema:
```rust
pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
    match self.side {
        PositionSide::Long => (current_price - self.entry_price) * self.quantity,
        PositionSide::Short => (self.entry_price - current_price) * self.quantity,
    }
}
```
En modo de Margen Aislado (`isolated_margin = Some(margin)`), la responsabilidad financiera máxima del operador ante un intercambio está estrictamente acotada al margen asignado a la posición (`margin`).
Sin embargo, `Position::unrealized_pnl` calcula el PnL no realizado de forma lineal e infinita.
Si una posición de $1,000 de margen apalancada sufre un movimiento adverso severo (ej. -$5,000 no realizado en la vela):
1. `portfolio.current_equity` resta $5,000 en lugar del tope de $1,000.
2. Si la cuenta tenía $3,000 de saldo disponible, `current_equity` pasa a ser negativo.
3. `is_bankrupt` evalúa como verdadero y aborta todo el backtest con bancarrota falsa, cuando en realidad la posición aislada habría sido liquidada perdiendo únicamente sus $1,000 de margen y preservando los $2,000 restantes de la cuenta.

#### Impacto Financiero:
Terminaciones tempranas espurias de simulaciones válidas y curvas de balance que muestran drawdowns mucho más profundos de lo contractualmente posible en Isolated Margin.

#### Solución Recomendada:
Acotar el PnL no realizado si hay margen aislado:
```rust
let raw = match self.side { ... };
if let Some(margin) = self.isolated_margin {
    raw.max(-margin)
} else {
    raw
}
```

---

### [BUG-17] ALTO: Sesgo de Brecha de Ejecución en la Etiqueta de Regresión

- **Ubicación:** `src/features/triple_barrier_labeler.rs:69-77`

#### Diagnóstico del Problema:
```rust
let p_entry = klines[i].close;
let p_exit = klines[i + horizon].close;
if p_entry > 0.0 {
    (p_exit - p_entry) / p_entry
}
```
El modelo se entrena para predecir el retorno entre el cierre de la vela actual `close[i]` y el cierre futuro `close[i + horizon]`.
Sin embargo, el motor de ejecución causal genera la señal al cierre de la vela $i$ y **entra en la posición al precio de apertura de la vela siguiente (`open[i + 1]`)**.
Si existe una discrepancia o gap entre `close[i]` y `open[i + 1]`, el modelo intenta aprender un movimiento de precios que ocurre **antes de que la orden sea ejecutada**, creando un desfase entre la función objetivo del modelo y el retorno realmente capturable en el mercado.

#### Impacto Financiero:
Degradación del rendimiento predictivo en producción y discrepancia entre el retorno esperado por el algoritmo de machine learning y el retorno neto del backtest.

#### Solución Recomendada:
Alinear la base del retorno a la apertura de la vela de ejecución:
```rust
let p_entry = klines[i + 1].open;
let p_exit = klines[i + 1 + horizon].close;
```

---

### [BUG-18] ALTO: Caída Fatal de la Terminal Interactiva ante Errores de Usuario o Red

- **Ubicación:** `src/cli/interactive_app.rs:34-57`

#### Diagnóstico del Problema:
En `InteractiveApp::run`:
```rust
match option.as_str() {
    "1" => DbHandler::handle(conn).await?,
    "2" => ForustHandler::handle(conn, ... )?,
    "3" => TreeBoostHandler::handle(conn, ... )?,
    ...
}
```
Si ocurre cualquier error operacional (por ejemplo, si el usuario selecciona evaluar un timeframe que aún no ha sido descargado en la base de datos o si falla una petición HTTP), el operador `?` propaga el error fuera del bucle `run()`, provocando que el programa termine bruscamente (`Process exited with code 1`).

#### Impacto en Usabilidad:
El usuario es expulsado de la terminal cuantitativa y se ve obligado a reiniciar `cargo run` manualmente tras cada error menor.

#### Solución Recomendada:
Capturar los errores dentro del bucle interactivo mostrando un mensaje descriptivo y permitiendo al usuario continuar en el menú principal:
```rust
if let Err(e) = ForustHandler::handle(...) {
    eprintln!("\n❌ Error durante la operación: {}\n", e);
}
```

---

### [BUG-19] MEDIO: Riesgo de Inicialización en el Calibrador de Umbrales

- **Ubicación:** `src/models/threshold_optimizer.rs:79-88`

#### Diagnóstico del Problema:
`best_score` se inicializa rígidamente en `-999.0` y se comparan los candidatos con `score > best_score`.
Si todos los candidatos obtienen puntuaciones inferiores a `-999.0` (posible en métricas como Retorno Neto % ante pérdidas severas o Sharpe negativo extremo), ninguno es seleccionado y se retorna un umbral predeterminado arbitrario. Además, no se filtra si `score.is_nan()` o `score.is_infinite()`.

#### Solución Recomendada:
Inicializar con `f64::NEG_INFINITY` y filtrar con `score.is_finite()`.

---

### [BUG-20] MEDIO: Ineficacia de Rotación de Endpoints ante Rate Limits de Binance

- **Ubicación:** `src/downloader/endpoint_pool.rs` y `src/downloader/binance_client.rs:35`

#### Diagnóstico del Problema:
`EndpointPool` implementa un pool de URLs de Binance (`api.binance.com`, `api1`, `api2`, `api3`, `api-gcp`).
Sin embargo, `BinanceClient::download_klines` itera siempre sobre `self.endpoints.all()`, comenzando incondicionalmente por el primer endpoint (`api.binance.com`).
Si el primer endpoint recibe un bloqueo temporal (código HTTP 429 - Too Many Requests), la siguiente petición vuelve a intentar primero con el endpoint bloqueado en vez de alternar de forma round-robin concurrente con un contador atómico.

#### Solución Recomendada:
Utilizar `AtomicUsize` en `EndpointPool` para rotar de manera concurrente el endpoint inicial de cada petición.

---

### [BUG-21] BAJO: Formato de Cero Negativo (`"-0.00%"`) en Dashboard HTML

- **Ubicación:** `src/dashboard/html_template.rs:29, 53`

#### Diagnóstico del Problema:
Al construir las series de drawdown para Chart.js:
`is_dd_series.push(format!("-{:.2}", p.drawdown_pct));`
Cuando el drawdown es `0.0`, formatea como `"-0.00"`, lo que provoca que en la gráfica y los tooltips interactivos se muestre `"-0.00%"`.

#### Solución Recomendada:
Validar si el valor es prácticamente cero (`p.drawdown_pct < 1e-4`) y emitir `"0.00"`.

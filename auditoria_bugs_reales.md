# Auditoría Forense de Bugs Reales en el Núcleo Cuantitativo, Financiero y de Ejecución (GBT)

**Fecha:** 2026-11-09  
**Auditor:** Auditor Técnico Sénior (Rust & Cuantitativo)  
**Destinatario:** Allianz  
**Repositorio auditado:** `c:\Users\Allianz\Downloads\Antigravity\GBT`  
**Estado de Compilación Estática:** Compilación limpia (`cargo check` OK en target dev).  

---

## 1. Resumen Ejecutivo de la Auditoría

Se ha completado una inspección forense exhaustiva de 89 archivos fuente del proyecto GBT, focalizada en **fallas reales de ingeniería financiera, matemática, ejecución de órdenes, modelado GBDT y lógica de sistemas**, descartando cualquier elemento cosmético o de interfaz gráfica.

A pesar de que el código compila y cuenta con tests unitarios para bugs previamente catalogados, la auditoría ha detectado **10 fallas críticas reales** que comprometen la validez de los backtests, la integridad de los datos financieros, el cálculo del drawdown, la calibración de modelos y la solvencia en cuentas apalancadas.

### Matriz de Severidad de Hallazgos

| ID | Área Forense | Hallazgo Crítico | Severidad | Impacto Financiero / Sistémico |
|---|---|---|---|---|
| **BUG-F01** | Eje 1 (Look-Ahead / Costos) | `TripleBarrierLabeler` ignora comisiones y slippage en etiquetado continuo y discreto | **CRÍTICA** | Modelos ML entrenan clasificando operaciones perdedoras reales como targets ganadores |
| **BUG-F02** | Eje 1 (Backtest / Sesgo) | Resurrección de capital zombi en `FullBacktestRunner` tras quiebra en In-Sample (IS) | **CRÍTICA** | Estrategias que quiebran la cuenta en IS reportan retornos netos positivos globales |
| **BUG-F03** | Eje 2 (Drawdown / Equity) | Ceguera total a la excursión intra-vela ($Low / High$) en el cálculo de Drawdown Máximo | **CRÍTICA** | Oculta caídas flotantes catastróficas intra-barra; DD reportado es solo close-to-close |
| **BUG-F04** | Eje 2 (Margen / Contabilidad) | `process_signal_at_open` calcula precio de liquidación con Equity previo a comisiones | **ALTA** | Liquidation Price artificialmente optimista en apalancamiento alto |
| **BUG-F05** | Eje 2 (Margen / Solvencia) | Verificación de capacidad financiera no exige `Margen + Comisión` simultáneos | **ALTA** | Permite entrar en posiciones con balance libre negativo (overleveraged) desde barra 0 |
| **BUG-F06** | Eje 3 (Métricas / Optimización) | Inversión paradójica de Sharpe y Sortino con retornos negativos en `ThresholdOptimizer` | **CRÍTICA** | El optimizador favorece estrategias con volatilidad bajista extrema cuando $R < 0$ |
| **BUG-F07** | Eje 3 (Score Compuesto) | Escalamiento asimétrico en `CompositeScoreCalculator` anula Linealidad $R^2$ (<1% peso) | **ALTA** | Score final gobernado al 99% por Capital/MaxDD; estrategias perdedoras obtienen score positivo |
| **BUG-F08** | Eje 4 (Persistencia / DB) | Esquema SQLite carece de columna `symbol`; descarga multi-activo sobreescribe velas | **CRÍTICA** | Contaminación y mutación corrupta de la serie temporal si se consulta más de un símbolo |
| **BUG-F09** | Eje 3 (ML / Series de Tiempo) | Bootstrap sampling IID con reemplazo en `RandomForestModel` sobre etiquetas solapadas | **MEDIA-ALTA** | Fuga de información por autocorrelación serial entre árboles del ensamble |
| **BUG-F10** | Eje 2 (Sizing / Numérico) | `PositionSizing::FixedAmount` sin saneamiento anti-NaN ni cota de positividad | **MEDIA** | Riesgo de propagación de NaNs o nocionales negativos en la asignación de tamaño |

---

## 2. Análisis Forense Detallado por Ejes

---

### EJE 1: Sesgos Financieros, Look-Ahead Bias y Etiquetado Causal ($t \le T$)

#### [BUG-F01] `TripleBarrierLabeler` ignora comisiones y slippage en el etiquetado continuo y discreto

* **Ubicación exacta:**  
  [`src/features/triple_barrier_labeler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/triple_barrier_labeler.rs#L180-L194) (Líneas 180–194 y 55–64).
* **Diagnóstico Técnico:**  
  En el cálculo del target de regresión continua:
  ```rust
  let p_entry = klines[i + 1].open;
  let p_exit = klines[i + horizon].close;
  if p_entry > 0.0 {
      (p_exit - p_entry) / p_entry
  } else {
      0.0
  }
  ```
  Y en la variante discreta de clasificación:
  ```rust
  let p_exit = klines[i + horizon].close;
  if p_exit > entry_price { TargetLabel::Long }
  else if p_exit < entry_price { TargetLabel::Short }
  else { TargetLabel::Neutral }
  ```
* **Causa Raíz:**  
  La fórmula asume frictionless trading ($cost = 0$). No deduce la comisión de entrada (`commission_rate`), la comisión de salida, ni el spread/slippage bid-ask (`slippage_rate`).
* **Impacto Financiero:**  
  Si la comisión estándar taker de Binance Futures es de 0.04% por lado (0.08% roundtrip) y el slippage conservador es de 0.02% por lado (0.04% roundtrip), el costo total de fricción es del **0.12%**.  
  Cualquier variación positiva de precio menor a 0.12% (por ejemplo, $+0.05\%$) es en la realidad financiera una **operación destructora de capital**.  
  Sin embargo, el labeler le asigna una etiqueta positiva ($+0.0005$ o `TargetLabel::Long`). Los modelos de árboles de decisión (Forust, TreeBoost, CART, Native GBDT y Random Forest) minimizan el error cuadrático ajustándose a retornos ilusorios que en operativa real desangran la cuenta en fees.
* **Mitigación de Producción Recomendada:**  
  Deducir el costo total de ida y vuelta (`roundtrip_cost = 2.0 * (commission_rate + slippage_rate)`) en el retorno de regresión y en los umbrales de las barreras horizontales:
  ```rust
  let gross_return = (p_exit - p_entry) / p_entry;
  let net_return = gross_return - roundtrip_friction;
  ```

---

#### [BUG-F02] Resurrección de Capital Zombi en `FullBacktestRunner` tras quiebra en In-Sample (IS)

* **Ubicación exacta:**  
  [`src/backtest/full_backtest_runner.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/full_backtest_runner.rs#L46-L56) (Líneas 46–56 y 104–136).
* **Diagnóstico Técnico:**  
  Al encadenar la fase In-Sample (IS) con la fase Out-of-Sample (OOS), el runner implementa la siguiente lógica:
  ```rust
  let is_final_capital = is_result.metrics.final_capital;
  let oos_initial_capital = if is_final_capital > 0.0 {
      is_final_capital
  } else {
      config.initial_capital // <-- RESURRECCIÓN DE CUENTA QUEBRADA
  };
  ...
  let mut combined_trades = is_result.trades.clone();
  combined_trades.extend(oos_result.trades.clone());
  ...
  let combined_metrics = MetricsCalculator::calculate(
      config.initial_capital,
      &combined_trades,
      &combined_equity,
  );
  ```
* **Causa Raíz:**  
  Si la estrategia resulta insolvente en In-Sample ($Equity \le 0$), el sistema resetea el capital a $10,000 en lugar de abortar el backtest o fijar el capital en 0.  
  Luego, en la consolidación de la curva de equity:
  1. IS termina en $0.
  2. La purga/cuarentena se mantiene en $0.
  3. En `test_start`, la curva salta instantáneamente de $0 a $10,000 (generación de dinero de la nada).
  4. Si en OOS la estrategia genera $+2,000$, el punto final de equity es $\$12,000$.
  5. `MetricsCalculator` computa: $\text{Net Profit} = \$12,000 - \$10,000 = +\$2,000$ ($+20\%$).
* **Impacto Financiero:**  
  Falso positivo institucional crítico. Una estrategia catastrófica que perdió el 100% del fondo de inversión en el 70% inicial de los datos es reportada a Allianz en el resumen como una **estrategia ganadora rentable (+20% neto)**.
* **Mitigación de Producción Recomendada:**  
  Si `is_final_capital <= 0.0`, la simulación OOS debe abortar inmediatamente registrando quiebra definitiva (`is_bankrupt = true`), o la fase OOS debe permanecer con balance 0.0 sin ejecutar operaciones.

---

### EJE 2: Motor de Ejecución, Contabilidad de PnL y Margen

#### [BUG-F03] Ceguera total a la excursión intra-vela ($Low / High$) en el cálculo de Drawdown Máximo

* **Ubicación exacta:**  
  [`src/backtest/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/engine.rs#L93-L113) (Líneas 93–113) y [`src/backtest/metrics_calculator.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics_calculator.rs#L104-L115) (Líneas 104–115).
* **Diagnóstico Técnico:**  
  En el bucle principal de barras:
  ```rust
  // 3. Registrar punto de la curva de equity al cierre de la vela actual
  let current_eq = portfolio.current_equity(current_candle.close);
  if current_eq > peak_equity {
      peak_equity = current_eq;
  }
  let dd_nominal = (peak_equity - current_eq).max(0.0);
  let dd_pct = if peak_equity > 0.0 {
      (dd_nominal / peak_equity) * 100.0
  } else {
      0.0
  };
  ```
  Y luego `MetricsCalculator::calculate` calcula `max_drawdown_nominal` y `max_drawdown_pct` exclusivamente iterando sobre los puntos de `equity_curve`.
* **Causa Raíz:**  
  El motor computa el equity **única y exclusivamente al precio de `close`**.  
  Si una posición Long abierta sufre un desplome violento durante la vela alcanzando el `low` (por ejemplo, una caída flotante del -35%), pero el precio rebota antes de expirar el período cerrando a solo -2%, el valor registrado en `equity_curve` es únicamente el equity al `close` (-2%).
* **Impacto Financiero:**  
  El Drawdown reportado es **Close-to-Close Drawdown**, no **True Peak-to-Trough Equity Drawdown**.  
  Ignora caídas flotantes catastróficas que en un exchange real habrían detonado margin calls o liquidaciones forzosas en brokers de futuros. Esto adultera las métricas de Calmar Ratio, Recovery Factor, Sharpe Ratio y el Score Compuesto Allianz.
* **Mitigación de Producción Recomendada:**  
  Para toda posición abierta en la barra, calcular el equity adverso extremo intra-vela:
  - Para Long: `adverse_eq = portfolio.current_equity(current_candle.low)`
  - Para Short: `adverse_eq = portfolio.current_equity(current_candle.high)`  
  Actualizar `peak_equity` y registrar el drawdown máximo alcanzado dentro de la vela antes del cierre.

---

#### [BUG-F04] `process_signal_at_open` calcula el precio de liquidación con Equity previo a comisiones

* **Ubicación exacta:**  
  [`src/backtest/execution/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/engine.rs#L84-L202) (Líneas 84, 155–156 y 194–202).
* **Diagnóstico Técnico:**  
  Secuencia cronológica del código:
  ```rust
  let equity = portfolio.current_equity(exec_price); // Línea 84: Equity pre-comisión
  ...
  let commission = notional_target * config.commission_rate;
  portfolio.cash_balance -= commission;              // Línea 155: Se descuenta comisión
  portfolio.total_commissions_paid += commission;
  ...
  let (isolated_margin, liq_price) = MarginCalculator::calculate_margin_and_liquidation(
      config.margin_mode,
      side,
      exec_price,
      quantity,
      equity, // Línea 199: BUG - Pasa el equity obsoleto sin descontar la comisión
      notional_target,
      leverage,
  );
  ```
* **Causa Raíz:**  
  El cálculo de `liq_price` en Cross Margin utiliza la fórmula `exec_price - (equity / quantity)`. Al pasar el `equity` capturado en la línea 84, el colateral disponible para absorber pérdidas se sobreestima en una magnitud exactamente igual a `commission`.
* **Impacto Financiero:**  
  En estrategias con alto apalancamiento (20x a 50x) o con comisiones elevadas, el precio de liquidación calculado queda más alejado del precio de entrada de lo que realmente está. La posición sobrevive a caídas de mercado que en la realidad habrían provocado la liquidación total de la cuenta.
* **Mitigación de Producción Recomendada:**  
  Pasar el equity real post-comisión: `let post_entry_equity = portfolio.current_equity(exec_price);` a `calculate_margin_and_liquidation`.

---

#### [BUG-F05] Verificación de capacidad financiera no exige `Margen + Comisión` simultáneos

* **Ubicación exacta:**  
  [`src/backtest/execution/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/engine.rs#L128-L133) (Líneas 128–133).
* **Diagnóstico Técnico:**  
  ```rust
  let required_margin = notional_target / leverage;
  let commission = notional_target * config.commission_rate;

  if notional_target <= 0.0
      || required_margin > equity
      || portfolio.cash_balance <= commission
  {
      return completed_trade;
  }
  ```
* **Causa Raíz:**  
  Las restricciones de capital se validan de forma desacoplada:
  1. `required_margin > equity` (¿Alcanza el balance total para el margen?)
  2. `portfolio.cash_balance <= commission` (¿Alcanza el balance solo para la comisión?)
  No se comprueba la suma: `required_margin + commission > portfolio.cash_balance`.
* **Impacto Financiero:**  
  Ejemplo numérico: Con Capital = $\$1,000$, Leverage = $10\text{x}$, Sizing = $100\%$ de equity:
  - `notional_target` = $\$10,000$
  - `required_margin` = $\$1,000$
  - `commission` = $\$4.00$ ($0.04\%$)
  - Comprobación: $1000 > 1000$ (Falso). $1000 \le 4$ (Falso).
  - La orden se aprueba.
  - Se debita la comisión: `cash_balance` pasa a $\$996.00$.
  - Margen comprometido: $\$1,000.00$.
  - **Margen libre resultante: $-\$4.00$ (Déficit de colateral en barra 0 sin fluctuación de precio).**
* **Mitigación de Producción Recomendada:**  
  Requerir contractualmente:
  ```rust
  if required_margin + commission > portfolio.cash_balance {
      return completed_trade;
  }
  ```

---

### EJE 3: Modelado Algorítmico, GBDT e Inferencia Causal

#### [BUG-F06] Inversión paradójica de Sharpe y Sortino con retornos negativos en `ThresholdOptimizer`

* **Ubicación exacta:**  
  [`src/backtest/metrics_calculator.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics_calculator.rs#L188-L242) (Líneas 188–242) y [`src/models/threshold_optimizer.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/threshold_optimizer.rs#L60-L88) (Líneas 60–88).
* **Diagnóstico Técnico:**  
  En el cálculo de Sortino:
  ```rust
  let downside_variance = returns.iter().filter(|&&r| r < 0.0).map(|&r| r.powi(2)).sum::<f64>() / n;
  let downside_std_dev = downside_variance.sqrt();
  ...
  let sortino = if downside_std_dev > 1e-9 {
      (mean / downside_std_dev) * annual_factor
  } else if mean > 0.0 {
      99.99
  } else {
      0.0
  };
  ```
  Y en `ThresholdOptimizer::optimize_in_sample`:
  ```rust
  for (th, score) in candidates {
      if score.is_finite() && score > best_score {
          best_score = score;
          best_th = th;
      }
  }
  ```
* **Causa Raíz:**  
  Paradoja clásica de ratios financieros de Sharpe / Sortino cuando el numerador ($\mu$) es negativo:
  - Supongamos Estrategia 1 (Pérdidas estables y controladas): $\mu = -0.01$, $\sigma_{down} = 0.005$.  
    $\text{Sortino}_1 = \frac{-0.01}{0.005} \times \text{ann} = -2.0 \times \text{ann}$.
  - Supongamos Estrategia 2 (Pérdidas caóticas con volatilidad destructiva): $\mu = -0.01$, $\sigma_{down} = 0.05$.  
    $\text{Sortino}_2 = \frac{-0.01}{0.05} \times \text{ann} = -0.2 \times \text{ann}$.
  Dado que $-0.2 > -2.0$, **el optimizador concluye que la Estrategia 2 es superior a la Estrategia 1.**
* **Impacto Financiero:**  
  Cuando un modelo en fase In-Sample experimenta retornos medios negativos en la rejilla de umbrales, el optimizador selecciona activamente el umbral que **maximiza la volatilidad de las pérdidas** (la más riesgosa y descontrolada), porque un denominador grande reduce el valor absoluto del ratio negativo.  
  *(Nota institucional: Este problema ya había sido corregido para Calmar Ratio en BUG-06 mediante penalización monótona, pero fue omitido en Sharpe y Sortino).*
* **Mitigación de Producción Recomendada:**  
  Implementar normalización monótona para retornos medios negativos en la evaluación fitness:
  ```rust
  if mean >= 0.0 {
      (mean / downside_std_dev) * annual_factor
  } else {
      mean * downside_std_dev * annual_factor // Monótonamente más negativo a mayor volatilidad
  }
  ```

---

#### [BUG-F07] Escalamiento asimétrico en `CompositeScoreCalculator` anula Linealidad $R^2$ (<1% de peso)

* **Ubicación exacta:**  
  [`src/backtest/composite_score.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs#L40-L60) (Líneas 40–60).
* **Diagnóstico Técnico:**  
  La fórmula solicitada era: $\text{Score} = 0.5 \times R^2 + 0.5 \times (\text{Capital} / \text{MaxDD})$.  
  En el código:
  ```rust
  let linearity = curve_r2.clamp(0.0, 1.0); // Rango: [0.0, 1.0] -> 0.5 * linearity está en [0.0, 0.5]
  let raw_ratio = final_capital / safe_max_dd;
  let capital_to_maxdd = raw_ratio.min(100.0); // Rango: [0.0, 100.0] -> 0.5 * ratio está en [0.0, 50.0]
  let composite_score = (0.5 * linearity) + (0.5 * capital_to_maxdd);
  ```
  Y en la rama de pérdidas:
  ```rust
  let net_pnl = final_capital - initial_capital;
  let loss_impact = (net_pnl - safe_max_dd) / initial_capital;
  let bounded_loss = loss_impact.clamp(-100.0, 0.0);
  let penalty_score = (0.5 * linearity) + (0.5 * bounded_loss);
  ```
* **Causa Raíz:**  
  1. **Desequilibrio de magnitud:** El término de capital aporta hasta **50.0 puntos**, mientras que la linealidad aporta un máximo de **0.5 puntos**. La linealidad de balance representa el **0.99%** de la decisión; el score está gobernado exclusivamente por el ratio de drawdown.
  2. **Score positivo para cuentas perdedoras:** Con `initial_capital = $10,000`, `net_pnl = -$500`, `safe_max_dd = $500`:
     `loss_impact = (-500 - 500) / 10000 = -0.1`.
     `0.5 * bounded_loss = -0.05`.
     Si `linearity = 0.9`, `0.5 * linearity = +0.45`.
     `penalty_score = +0.45 - 0.05 = +0.40` **(Puntaje positivo para una estrategia que perdió dinero).**
* **Impacto Financiero:**  
  Estrategias destructoras de capital pero con caídas suaves y ordenadas obtienen puntajes de optimización positivos y clasifican por encima de estrategias rentables con menor $R^2$.
* **Mitigación de Producción Recomendada:**  
  Normalizar el ratio $\text{Capital} / \text{MaxDD}$ a escala $[0.0, 1.0]$ o usar retornos porcentuales, y forzar que toda pérdida neta resulte estrictamente en un puntaje negativo:
  ```rust
  if final_capital < initial_capital {
      return (loss_ratio, -1.0 * (1.0 - linearity + loss_impact.abs()));
  }
  ```

---

#### [BUG-F08] Esquema SQLite carece de columna `symbol`; descarga multi-activo sobreescribe velas

* **Ubicación exacta:**  
  [`src/storage/db_connection.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_connection.rs#L22-L34) (Líneas 22–34), [`src/storage/kline_repository.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/kline_repository.rs#L21-L49) (Líneas 21–49) y [`src/cli/db_handler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/db_handler.rs#L62-L89) (Líneas 62–89).
* **Diagnóstico Técnico:**  
  El esquema de base de datos define:
  ```sql
  CREATE TABLE IF NOT EXISTS klines (
      timeframe TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      open REAL NOT NULL,
      high REAL NOT NULL,
      low REAL NOT NULL,
      close REAL NOT NULL,
      volume REAL NOT NULL,
      PRIMARY KEY (timeframe, timestamp)
  )
  ```
  Y la inserción en lote ejecuta:
  ```sql
  INSERT OR REPLACE INTO klines (timeframe, timestamp, open, high, low, close, volume)
  VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
  ```
* **Causa Raíz:**  
  **No existe la columna `symbol` en la tabla `klines`**, a pesar de que el menú interactivo CLI le pregunta al usuario el símbolo (`BTCUSDT`, `ETHUSDT`, etc.).  
  La clave primaria es únicamente `(timeframe, timestamp)`.
* **Impacto Financiero y Sistémico:**  
  Si un analista descarga primero `BTCUSDT` en `1h` (por ejemplo, precios alrededor de $\$60,000$), y luego sincroniza `ETHUSDT` en `1h` (precios alrededor de $\$2,500$):
  - `ETHUSDT` **sobreescribe físicamente** los registros de `BTCUSDT` para los mismos timestamps.
  - Al ejecutar cualquier backtest (Forust, TreeBoost, etc.), `KlineRepository::load_all(conn, tf)` carga una serie corrupta con saltos brutales entre $\$60,000$ y $\$2,500$.
  - Se destruye la integridad de la base de datos de klines.
* **Mitigación de Producción Recomendada:**  
  Migrar la tabla agregando la columna `symbol TEXT NOT NULL` y redefinir la clave primaria a:
  `PRIMARY KEY (symbol, timeframe, timestamp)`.

---

#### [BUG-F09] Bootstrap sampling IID con reemplazo en `RandomForestModel` sobre etiquetas solapadas

* **Ubicación exacta:**  
  [`src/models/random_forest/forest_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs#L61-L73) (Líneas 61–73).
* **Diagnóstico Técnico:**  
  En el bucle de entrenamiento del Random Forest:
  ```rust
  let mut bootstrap_indices = Vec::with_capacity(valid_rows);
  for _ in 0..valid_rows {
      let random_idx = (rng() as usize) % valid_rows;
      bootstrap_indices.push(random_idx);
  }
  ```
* **Causa Raíz:**  
  El muestreo bootstrap tradicional (Breiman, 2001) asume observaciones independientes e idénticamente distribuidas (IID).  
  En finanzas y series temporales con etiquetado de Triple Barrier sobre un horizonte de $H$ velas (ej. 5 o 10 barras), las etiquetas adyacentes $y_t$ y $y_{t+1}$ comparten $H-1$ barras de retornos futuros, generando una fuerte autocorrelación serial.  
  El bootstrap aleatorio simple genera redundancia y sesgo optimista artificial en el ensamble.
* **Impacto Financiero:**  
  Sobreajuste severo del Random Forest en In-Sample. El ensamble memoriza patrones espurios causados por el solapamiento temporal de las etiquetas, colapsando su rendimiento en Out-of-Sample.
* **Mitigación de Producción Recomendada:**  
  Implementar *Sequential Bootstrapping* (López de Prado, *Advances in Financial Machine Learning*) o *Block Bootstrapping* con bloques de longitud $\ge H$.

---

#### [BUG-F10] `PositionSizing::FixedAmount` sin saneamiento anti-NaN ni cota de positividad

* **Ubicación exacta:**  
  [`src/backtest/execution/sizing.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/sizing.rs#L23-L26) (Líneas 23–26).
* **Diagnóstico Técnico:**  
  ```rust
  PositionSizing::FixedAmount(amount) => {
      let target = amount * leverage;
      target.min(equity * leverage)
  }
  ```
* **Causa Raíz:**  
  Mientras que `PercentOfInitialCapital`, `PercentOfEquity` y `RiskBasedPercent` aplican `.clamp()`, `FixedAmount` toma `amount` directamente sin verificar que sea finito ni positivo (`amount > 0.0`).
* **Impacto Financiero:**  
  Si por error de entrada se pasa un monto negativo o `f64::NAN`, el cálculo de nocional genera valores anómalos o comportamientos indefinidos en la asignación de posiciones.
* **Mitigación de Producción Recomendada:**  
  Sanitizar con: `amount.max(0.0)`. Si no es finito, fallback a 0.0.

---

## 3. Plan de Mitigación y Acción Inmediata

1. **Corrección de `TripleBarrierLabeler`:** Incorporar `commission_rate` y `slippage_rate` en los métodos `label_regression` y `label_classification`.
2. **Corrección de `FullBacktestRunner`:** Abortar la fase OOS si la estrategia quiebra en IS; no inyectar capital zombi.
3. **Corrección de Drawdown en `engine.rs`:** Computar el drawdown intra-vela utilizando el extremo adverso ($Low$ para Long, $High$ para Short).
4. **Corrección de Margen en `engine.rs`:** Utilizar el equity posterior a comisiones para el cálculo del precio de liquidación y verificar `required_margin + commission <= cash_balance`.
5. **Corrección en `db_connection.rs` y `kline_repository.rs`:** Incorporar la columna `symbol` en la tabla `klines` y en todas las consultas SQL.
6. **Saneamiento en `composite_score.rs` y `metrics_calculator.rs`:** Normalizar el peso de linealidad respecto al ratio de capital y corregir la inversión de Sharpe/Sortino en pérdidas.

---

*Reporte técnico finalizado y documentado con rigor milimétrico según las directivas de Allianz.*

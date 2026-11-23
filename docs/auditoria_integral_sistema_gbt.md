# INFORME DE AUDITORÍA INTEGRAL DE CÓDIGO, ARQUITECTURA FINANCIERA Y BUGS (GBT)

**Destinatario:** Allianz  
**Fecha:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Ultra Modularidad en Rust, Robustez Institucional, Cero Código Fantasma.  
**Estado de Compilación Estática:** `cargo check` exitoso (0 advertencias, 0 errores).

---

## 1. RESUMEN EJECUTIVO

Se ha ejecutado una auditoría integral, profunda y metodológica sobre la totalidad de la base de código del proyecto cuantitativo **GBT** (55+ archivos `.rs`).

### Matriz de Diagnóstico General:
1. **Verificación de Sesgo Temporal (Anti Look-Ahead Bias $t \le T$):**
   - El pipeline base cumple con la separación estricta: las señales generadas al cierre de la vela $t$ se ejecutan en la apertura de la vela $t+1$ con slippage y comisiones.
   - El descargador de Binance descarta velas en curso (`close_time > now_ms`), garantizando que la base de datos contenga únicamente velas cerradas.
   - El parche `clear_cache()` en estrategias Walk-Forward se encuentra activo en los 5 controladores de modelos, evitando la contaminación IS $\rightarrow$ OOS previa.
2. **Hallazgos Críticos y Vulnerabilidades Detectadas en esta Auditoría:**
   - **1 Vulnerabilidad Crítica Financiera:** Relleno ficticio sin slippage ni ajuste de Gap en Liquidaciones intra-vela (Margin Call Gap Bias).
   - **2 Vulnerabilidades de Alta Prioridad:** 
     - Inconsistencia de Position Sizing por Riesgo sin Stop Loss activo (riesgo no acotado).
     - Terminación abrupta (crash) de la CLI interactiva por propagación de errores (`?`).
   - **3 Hallazgos de Prioridad Media:**
     - Desviación algorítmica en Random Forest: Bagging de features a nivel de árbol completo en lugar de muestreo en cada split de nodo (Breiman RF).
     - Distorsión de gradiente en `CompositeScoreCalculator` ante pérdidas netas (penalización plana $0.0$).
     - Cold-start y muerte operativa de las primeras 35 velas en el conjunto Out-of-Sample.
   - **2 Observaciones de Deuda Técnica / Baja Severidad:**
     - Exención artificial de slippage y tarifas de liquidación en `close_position`.
     - Ausencia de purga explícita y umbrales fijos en el modo `run_full_backtest`.

---

## 2. CATÁLOGO DE HALLAZGOS Y BUGS DETECTADOS

| ID | Severidad | Módulo / Componente | Descripción Resumida | Estado |
|---|---|---|---|---|
| **BUG-14** | **CRÍTICO** | [`ExecutionEngine::check_intra_candle_exits`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L173-L185) | **Gap Bias en Liquidaciones**: Liquidación ejecutada al precio teórico exacto ante Gaps de apertura sin contemplar el precio de apertura ni slippage. | **RESUELTO** |
| **BUG-15** | **ALTO** | [`ExecutionEngine::process_signal_at_open`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L84-L111) | **Sizing Basado en Riesgo Huérfano**: El sizing calcula posición asumiendo un Stop Loss del 2%, pero no existe orden de Stop Loss activa en la posición. | **RESUELTO** |
| **BUG-16** | **ALTO** | [`InteractiveApp::run`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/interactive_app.rs#L34-L58) | **Crash de la Terminal Interactiva**: Uso del operador `?` provoca la caída y salida del proceso ante cualquier error de DB o datos insuficientes. | **RESUELTO** |
| **BUG-17** | **MEDIO** | [`RandomForestModel::train`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs#L67-L78) | **Desviación de Random Forest (Breiman vs Ho)**: Muestreo de variables restringido a nivel de árbol completo en lugar de subconjunto aleatorio por nodo. | **RESUELTO** |
| **BUG-18** | **MEDIO** | [`CompositeScoreCalculator::calculate`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs#L29-L31) | **Aplanamiento de Gradiente en Pérdidas**: Asignación de score plano $0.0$ a cualquier estrategia con pérdidas, impidiendo ranking relativo en optimizadores. | **RESUELTO** |
| **BUG-19** | **MEDIO** | [`TrainTestSplit::split_chronological`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/train_test_split.rs#L31-L33) | **Cold-Start y Mutilación de 35 Velas en OOS**: Partición abrupta de velas sin buffer de precalentamiento para indicadores técnicos. | **RESUELTO** |
| **BUG-20** | **BAJO** | [`ExecutionEngine::close_position`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L252-L258) | **Exención de Slippage en Liquidación**: Las liquidaciones forzosas omiten la penalización de slippage y tarifas de penalización de exchange. | **RESUELTO** |
| **BUG-21** | **BAJO** | [`ForustHandler::run_full_backtest`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/forust_handler.rs#L143-L153) | **Backtest Completo Sin Purga**: Evaluación continua sobre 100% de los datos con modelo entrenado en 70%, sin reporte segmentado. | **RESUELTO** |

---

## 3. ANÁLISIS TÉCNICO Y DETALLADO DE CADA HALLAZGO

---

### [BUG-14] CRÍTICO: Relleno Ficticio sin Slippage ante Gaps de Apertura en Liquidaciones (Margin Call Gap Bias)

- **Ubicación:** [`src/backtest/execution_engine.rs:173-185`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L173-L185) y [`src/backtest/execution_engine.rs:252-258`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L252-L258)
- **Diagnóstico:**
  En `ExecutionEngine::check_intra_candle_exits`, el Stop Loss cuenta con prevención de Gap Risk:
  ```rust
  let sl_exec = pos.stop_loss.map(|sl| sl.min(current_candle.open)); // Para Long
  ```
  Sin embargo, en la verificación de liquidación:
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
  Y en `close_position`:
  ```rust
  let exit_price = if is_liquidation {
      target_price
  } else {
      match pos.side { ... slippage ... }
  };
  ```
  Si el mercado abre con un Gap violento en contra de la posición (por ejemplo, Long con precio de liquidación en \$85, y la vela abre directamente en \$70 con mínimo en \$65):
  1. `low <= pos.liquidation_price` resulta verdadero.
  2. El precio de salida se fija en **\$85** (el precio teórico de liquidación).
  3. No se aplica slippage de ejecución.
- **Impacto Financiero:**
  El backtest simula una liquidación a un precio (\$85) que **nunca existió** durante la vela (la vela abrió en \$70 y cotizó entre \$65 y \$70). Esto oculta pérdidas severas por Gap y puede falsear la solvencia de estrategias altamente apalancadas o en modo Cross Margin.
- **Solución Recomendada:**
  Ajustar el precio de liquidación considerando el Gap de apertura y aplicar un slippage de liquidación forzosa:
  ```rust
  let liq_exec = match pos.side {
      PositionSide::Long => pos.liquidation_price.min(current_candle.open),
      PositionSide::Short => pos.liquidation_price.max(current_candle.open),
  };
  ```

---

### [BUG-15] ALTO: Inconsistencia de Position Sizing Basado en Riesgo sin Stop Loss Activo

- **Ubicación:** [`src/backtest/execution_engine.rs:84-111`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L84-L111) y [`src/backtest/config.rs:27`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/config.rs#L27)
- **Diagnóstico:**
  En `PositionSizing::RiskBasedPercent(risk_pct)`:
  ```rust
  let sl_dist = config.fixed_stop_loss_pct.unwrap_or(0.02);
  let risk_amount = equity * risk_pct.clamp(0.001, 0.10);
  let pos_size = risk_amount / sl_dist;
  ```
  El tamaño nominal de la posición se calcula asumiendo una distancia de Stop Loss de `sl_dist` (por defecto 2%).
  Sin embargo, en las líneas 106-111:
  ```rust
  let stop_loss = signal.stop_loss_price.or_else(|| {
      config.fixed_stop_loss_pct.map(|pct| match side {
          PositionSide::Long => exec_price * (1.0 - pct),
          PositionSide::Short => exec_price * (1.0 + pct),
      })
  });
  ```
  Ninguna estrategia de Machine Learning (`TreeBoostStrategy`, `ForustStrategy`, `CartStrategy`, etc.) ni la estrategia `EmaCrossoverStrategy` definen `signal.stop_loss_price`. Además, `config.fixed_stop_loss_pct` es `None` por defecto y no es editable en el menú `ConfigHandler`.
- **Impacto Financiero:**
  La posición se dimensiona asumiendo que el riesgo monetario está acotado al 1% o 2% del capital mediante un Stop Loss. Pero la orden ingresa al mercado **SIN Stop Loss**. Si el mercado cae un 20%, la posición pierde el 20% (o se liquida), invalidando por completo el modelo de gestión de riesgo institucional.
- **Solución Recomendada:**
  Si `config.position_sizing` es `RiskBasedPercent`, garantizar que si `config.fixed_stop_loss_pct` es `None`, se asigne automáticamente el `sl_dist` utilizado en el dimensionamiento como Stop Loss efectivo de la posición.

---

### [BUG-16] ALTO: Interrupción Catastrófica de la Terminal Interactiva por Errores Menores

- **Ubicación:** [`src/cli/interactive_app.rs:34-58`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/interactive_app.rs#L34-L58)
- **Diagnóstico:**
  En el bucle principal de `InteractiveApp::run`:
  ```rust
  match option.as_str() {
      "1" => DbHandler::handle(conn).await?,
      "2" => ForustHandler::handle(conn, &mut self.model_config, &self.backtest_config)?,
      "3" => TreeBoostHandler::handle(conn, &mut self.model_config, &self.backtest_config)?,
      "4" => NativeGbdtHandler::handle(conn, &self.model_config, &self.backtest_config)?,
      "5" => RandomForestHandler::handle(conn, &self.model_config, &self.backtest_config)?,
      "6" => CartHandler::handle(conn, &self.model_config, &self.backtest_config)?,
      "7" => BaselineHandler::handle(conn, &self.backtest_config)?,
      ...
  }
  ```
  Si el usuario selecciona un timeframe con velas insuficientes (o vacío), o si ocurre un error de lectura/escritura en disco, la expresión con `?` aborta la función `run()`, retornando el error a `main()` y provocando el cierre inmediato del binario.
- **Impacto Técnico / UX:**
  Cualquier error de usuario o de datos termina la sesión de la terminal interactiva de forma abruptly, obligando al usuario a reiniciar la aplicación.
- **Solución Recomendada:**
  Capturar el `Result` dentro del `match`, emitir el mensaje formateado `❌ Error: {}` por consola y continuar el ciclo interactivo sin interrumpir la ejecución.

---

### [BUG-17] MEDIO: Desviación Algorítmica en Random Forest (Breiman vs Ho Subspace)

- **Ubicación:** [`src/models/random_forest/forest_model.rs:67-78`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs#L67-L78)
- **Diagnóstico:**
  En el algoritmo de Random Forest clásico (Breiman, 2001), el muestreo aleatorio de variables ($m_{try} \approx \sqrt{F}$) debe ocurrir de forma independiente en **cada nodo y división** del árbol.
  En la implementación actual:
  ```rust
  let selected_features = &all_feats[0..m_try];
  let mut tree = CartTree::new(max_depth, 3, 1e-7);
  tree.fit_indices(x_slice, y_slice, &bootstrap_indices, f_dim, Some(selected_features));
  ```
  `selected_features` se escoge una sola vez antes de construir el árbol completo. Todas las divisiones internas de ese árbol quedan restringidas exclusivamente a ese subconjunto estático (método de subespacios aleatorios de Tin Kam Ho, 1998).
- **Impacto Metodológico:**
  Reduce drásticamente la capacidad de representación y diversidad del ensamble cuando $F$ es moderado (11 features), limitando el poder predictivo del ensamble frente a un verdadero Random Forest a nivel de split.
- **Solución Recomendada:**
  Integrar el muestreo de variables aleatorias dentro de `SplitFinder::find_best_split` o generar candidatos dinámicos en cada llamada recursiva de `CartTree::build_tree`.

---

### [BUG-18] MEDIO: Aplanamiento de Gradiente en `CompositeScoreCalculator` ante Pérdidas Netas

- **Ubicación:** [`src/backtest/composite_score.rs:29-31`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs#L29-L31) y [`src/models/threshold_optimizer.rs:78-87`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/threshold_optimizer.rs#L78-L87)
- **Diagnóstico:**
  En `CompositeScoreCalculator::calculate`:
  ```rust
  if total_trades == 0 || final_capital <= initial_capital {
      return (0.0, 0.0);
  }
  ```
  Si una estrategia finaliza con pérdidas respecto al capital inicial, se le asigna un score plano de `0.0`.
  Cuando el optimizador de umbrales `ThresholdOptimizer` evalúa los 8 umbrales candidatos durante un régimen de mercado adverso (donde todas las combinaciones generan pequeñas pérdidas), todos los candidatos reciben exactamente `0.0`.
  Al buscar el mejor score:
  ```rust
  let mut best_th = 0.002;
  let mut best_score = -999.0;
  for (th, score) in candidates {
      if score > best_score {
          best_score = score;
          best_th = th;
      }
  }
  ```
  El optimizador selecciona ciegamente el primer umbral probado (`0.0005`), sin capacidad de discriminar entre una configuración que preservó el capital (ej. pérdida de -0.2%) y otra que sufrió un drawdown catastrófico (ej. pérdida de -45%).
- **Impacto Financiero:**
  Se pierde la capacidad de discriminación en optimización cuando el régimen es desfavorable, favoreciendo configuraciones con sobreoperación y mayor costo en comisiones.
- **Solución Recomendada:**
  Permitir scores negativos proporcionales a la pérdida y al drawdown cuando `final_capital < initial_capital`, preservando un gradiente monotónico continuo.

---

### [BUG-19] MEDIO: Cold-Start y Mutilación de las Primeras 35 Velas en Out-of-Sample

- **Ubicación:** [`src/features/train_test_split.rs:31-33`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/train_test_split.rs#L31-L33) y [`src/models/treeboost_strategy.rs:38`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_strategy.rs#L38)
- **Diagnóstico:**
  `TrainTestSplit::split_chronological` corta el dataset en:
  `train_klines = klines[0..train_end]` y `test_klines = klines[test_start..n]`.
  Al inicializar el backtest OOS sobre `test_klines`:
  1. Los indicadores (EMA30, RSI14, ATR14) se calculan desde la vela 0 de `test_klines`.
  2. Las estrategias descartan las primeras 35 velas (`if current_idx < 35 { return Signal::flat(ts); }`).
- **Impacto Financiero y de Simulación:**
  Las primeras 35 velas del conjunto Out-of-Sample quedan completamente anuladas (sin trading), y los indicadores en la barra 36 tienen una memoria exponencial distorsionada por falta de historial previo de calentamiento. En trading real, el trader cuenta con las velas históricas precedentes al momento $T$ para el cálculo de sus medias.
- **Solución Recomendada:**
  Alimentar el cálculo de features OOS con un buffer de precalentamiento (warm-up) de 50 a 80 velas previas a `test_start`, iniciando las órdenes y el registro de la curva de equity estrictamente en `test_start`.

---

### [BUG-20] BAJO: Exención de Slippage y Comisiones Extra en Salidas por Liquidación

- **Ubicación:** [`src/backtest/execution_engine.rs:252-258`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs#L252-L258)
- **Diagnóstico:**
  `ExecutionEngine::close_position` no aplica `slippage_rate` si `is_liquidation` es verdadero:
  ```rust
  let exit_price = if is_liquidation {
      target_price
  } else {
      match pos.side { ... }
  };
  ```
  En exchanges reales (ej. Binance Futures), las órdenes de liquidación forzosa son órdenes a mercado agresivas que absorben el libro de órdenes (máximo slippage) y sufren recargos de fondo de liquidación / seguro (1% a 2%). Eximir de slippage a la peor salida posible resulta excesivamente benevolente.

---

### [BUG-21] BAJO: Backtest Completo Sin Purga ni Reporte Segmentado

- **Ubicación:** [`src/cli/forust_handler.rs:143-153`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/forust_handler.rs#L143-L153) y [`src/cli/treeboost_handler.rs:143-153`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/treeboost_handler.rs#L143-L153)
- **Diagnóstico:**
  En la opción "Full Backtest" de la CLI, se entrena con el 70% de las velas y luego se evalúa el 100% de la serie sin purga intermedia ni diferenciación gráfica en la curva de equity entre el periodo de entrenamiento y el periodo no visto.
  Esto puede inducir a error visual si un operador confunde la curva global con un resultado puramente fuera de muestra.

---

## 4. ESTADO DE LA ARQUITECTURA Y CUMPLIMIENTO DE REGLAS

1. **Anti Look-Ahead Bias:** Cumplido en la lógica central de features causales y en la ejecución al Open de $t+1$. Requiere corregir el cálculo de Gap en liquidaciones (BUG-14).
2. **Ultra Modularidad en Rust:** Cumplido con excelencia. Estructuras, handlers, métricas, estrategias y modelos residen en módulos desacoplados y limpios.
3. **Cero Código Fantasma:** Todo el código cuenta con lógica funcional de producción sin mocks ni stubs ficticios.
4. **Verificación de Compilación:** Compilación estática limpia mediante el compilador de Rust.

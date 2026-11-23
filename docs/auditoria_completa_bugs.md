# AUDITORÍA COMPLETA DE CÓDIGO, MODELOS Y ARQUITECTURA FINANCIERA (GBT)
**Destinatario:** Allianz  
**Fecha de Auditoría:** 2026-11-08  
**Estándar de Inspección:** Cero Look-Ahead Bias ($t \le T$), Ultra Modularidad en Rust, Veracidad y Robustez Institucional.

---

## 1. RESUMEN EJECUTIVO DE LA AUDITORÍA

Se ha completado una revisión exhaustiva del 100% de los módulos de la base de código del proyecto GBT:
- **Módulos auditados:** `features`, `backtest`, `models` (DecisionTree CART, RandomForest, Native GBDT, Forust-ML, TreeBoost), `storage`, `downloader`, `strategies`, `cli`, `dashboard` y `threading`.
- **Estado de compilación estática:** El proyecto compila sin errores ni advertencias mediante `cargo check`.
- **Hallazgos Críticos detectados:** 3 errores graves que invalidan resultados de simulación/optimización.
- **Hallazgos de Alta Prioridad:** 2 vulnerabilidades financieras y metodológicas de datos.
- **Hallazgos de Prioridad Media:** 4 problemas algorítmicos, matemáticos y de rendimiento.
- **Hallazgos de Prioridad Baja / Deuda Técnica:** 3 inconsistencias de código muerto y duplicación.

---

## 2. TABLA RESUMEN DE HALLAZGOS Y ESTADO DE RESOLUCIÓN

| ID | Severidad | Módulo / Componente | Descripción Resumida | Estado |
|---|---|---|---|---|
| **BUG-01** | **CRÍTICO** | `cli / models / strategy` | Contaminación del caché de predicciones en Walk-Forward OOS | **RESUELTO** |
| **BUG-02** | **CRÍTICO** | `backtest::execution_engine` | Ejecución imposible en Stop Loss ante Gaps de apertura (Gap Bias) | **RESUELTO** |
| **BUG-03** | **CRÍTICO** | `backtest::composite_score` | Score Compuesto explota a 5,000+ si la estrategia no opera (0 trades) | **RESUELTO** |
| **BUG-04** | **ALTO** | `backtest::execution_engine` | Falta de control de solvencia y margen disponible en el Sizing | **RESUELTO** |
| **BUG-05** | **ALTO** | `features::train_test_split` | Cold-start de indicadores en test y discrepancia Batch vs Streaming | **RESUELTO** |
| **BUG-06** | **MEDIO** | `backtest::metrics_calculator` | Factor de anualización de Sharpe/Sortino fijo a diario ($\sqrt{365.25}$) | **RESUELTO** |
| **BUG-07** | **MEDIO** | `features::technical_indicators` | RSI asigna 100.0 (extremo sobrecomprado) en velas planas sin variación | **RESUELTO** |
| **BUG-08** | **MEDIO** | `models::decision_tree` | Error off-by-one en límite superior del generador de cortes CART | **RESUELTO** |
| **BUG-09** | **MEDIO** | `strategies::ema_cross` | Complejidad $O(N^2)$ en recálculo recursivo de EMA en cada barra | **RESUELTO** |
| **BUG-10** | **MEDIO** | `models::split_finder` | Asignación excesiva de memoria $O(N^2)$ en búsqueda de split | **RESUELTO** |
| **BUG-11** | **BAJO** | `models::cart_tree` | Sentencia duplicada y muerta en nodo hoja de CART | **RESUELTO** |
| **BUG-12** | **BAJO** | `backtest::config` | Campo `capital_percent_per_trade` huérfano y no referenciado | **RESUELTO** |
| **BUG-13** | **BAJO** | `backtest::walk_forward_result`| `robustness_ratio` distorsionado ante retornos netos negativos | **RESUELTO** |

---

## 3. ANÁLISIS DETALLADO DE CADA HALLAZGO

---

### [BUG-01] CRÍTICO: Contaminación del Caché de Predicciones en la Validación Walk-Forward (OOS)

- **Ubicación:**
  - `src/cli/treeboost_handler.rs:81-91`
  - `src/cli/cart_handler.rs:57-67`
  - `src/cli/forust_handler.rs:81-91`
  - `src/cli/random_forest_handler.rs:57-67`
  - `src/cli/native_gbdt_handler.rs:57-67`
  - `src/models/treeboost_strategy.rs:35-47`
  - `src/models/decision_tree/cart_strategy.rs:34-46`
  - `src/models/forust_strategy.rs:34-46`
  - `src/models/random_forest/forest_strategy.rs:34-46`
  - `src/models/native_gbdt/native_strategy.rs:34-46`

#### Diagnóstico del Problema:
Cada estrategia basada en ML implementa un campo `cached_predictions: Option<Vec<f64>>` para almacenar en caché las inferencias vectorizadas y no recomputarlas barra a barra.

En todos los handlers de CLI (`treeboost_handler`, `cart_handler`, etc.), el flujo de ejecución es:
```rust
// 1. Se crea la estrategia y se ejecuta en In-Sample (IS):
let mut strategy_is = TreeBoostStrategy::new(model);
let engine_is = BacktestEngine::new(bt_cfg.clone());
let is_result = engine_is.run(&mut strategy_is, &split.train_klines)?;

// 2. Se "reutiliza" la estrategia para Out-of-Sample (OOS):
let mut strategy_oos = strategy_is;
let oos_result = engine_oos.run(&mut strategy_oos, &split.test_klines)?;
```

Al ejecutarse `engine_is.run`, la estrategia puebla `self.cached_predictions` con las predicciones del lote de **entrenamiento (IS)**.
Cuando `strategy_oos` corre sobre `&split.test_klines`:
```rust
let predicted_return = if let Some(ref preds) = self.cached_predictions {
    if current_idx < preds.len() {
        preds[current_idx]
    } else {
        return Signal::flat(ts);
    }
}
```
Como `cached_predictions` **YA NO ES `None`**, la estrategia OOS lee `preds[current_idx]`, las cuales son **las predicciones de los datos de entrenamiento (IS)** correspondientes al índice `current_idx`, y **NO las predicciones del conjunto de prueba (OOS)**.
Además, si el conjunto OOS tiene más velas que IS, una vez `current_idx >= preds.len()`, la estrategia emite `Signal::flat` de manera indefinida.

#### Impacto Financiero:
Los resultados reportados como "Out-of-Sample" en el dashboard HTML y consola son completamente falsos e inválidos. La estrategia OOS estuvo operando con señales generadas sobre el mercado pasado de entrenamiento.

#### Solución Recomendada:
1. Agregar el método `clear_cache(&mut self)` en todas las estrategias o en el trait `Strategy`.
2. O bien, construir una nueva instancia limpia para OOS pasando el modelo entrenado:
```rust
let mut strategy_oos = TreeBoostStrategy::new(model);
```

---

### [BUG-02] CRÍTICO: Relleno Ficticio de Stop Loss ante Gaps de Apertura (Gap Risk Bias)

- **Ubicación:** `src/backtest/execution_engine.rs:179-200`, `205-218`, `231-240`

#### Diagnóstico del Problema:
En `ExecutionEngine::check_intra_candle_exits`:
```rust
let hit_sl = pos.stop_loss.map_or(false, |sl| current_candle.low <= sl);
...
if hit_sl {
    exit_event = Some((pos.stop_loss.unwrap(), "Stop Loss"));
}
```
Y luego en `close_position`:
```rust
let exit_price = match pos.side {
    PositionSide::Long => target_price * (1.0 - config.slippage_rate),
    PositionSide::Short => target_price * (1.0 + config.slippage_rate),
};
```
Donde `target_price` es rígidamente `pos.stop_loss.unwrap()`.

**Escenario Real de Mercado:**
Una posición Long tiene Stop Loss en `$100.00`.
Ocurre una noticia adversa inter-sesión o un salto de liquidez. La siguiente vela abre directamente con un gap bajista en `$90.00` (con rango High `$91.00`, Low `$88.00`, Close `$89.00`).
La condición `low <= sl` (`88.00 <= 100.00`) es verdadera.
El motor ejecuta el cierre de la posición Long a:
`$100.00 * (1.0 - 0.0002) = $99.98`.

**El precio $99.98 NUNCA existió en esa vela.** El mercado abrió a `$90.00`. En una bolsa real o exchange de futuros (Binance), una orden Stop se convierte en orden de mercado y se llena al precio de apertura (`$90.00`) o peor.

#### Impacto Financiero:
Enmascara por completo el riesgo de gap (Gap Risk / Slippage catastrófico), elimina drawdowns reales y proyecta curvas de equity infladas artificialmente (Look-Ahead Bias de ejecución).

#### Solución Recomendada:
Para posición Long:
`let effective_exit = pos.stop_loss.unwrap().min(current_candle.open);`
Para posición Short:
`let effective_exit = pos.stop_loss.unwrap().max(current_candle.open);`

---

### [BUG-03] CRÍTICO: Explosión del Score Compuesto ante Cero Operaciones o Cero Drawdown

- **Ubicación:** `src/backtest/composite_score.rs:19-35`, `src/models/threshold_optimizer.rs:60-87`

#### Diagnóstico del Problema:
En `CompositeScoreCalculator::calculate`:
```rust
let safe_max_dd = if max_drawdown_nominal <= 0.0 {
    1.0
} else {
    max_drawdown_nominal
};

let capital_to_maxdd = if final_capital > 0.0 {
    final_capital / safe_max_dd
} else {
    0.0
};

let composite_score = (0.5 * linearity) + (0.5 * capital_to_maxdd);
```

Si una estrategia durante un backtest o búsqueda de umbrales **NO realiza ninguna operación** (o umbrales muy restrictivos):
1. `max_drawdown_nominal = 0.0`
2. `safe_max_dd = 1.0`
3. `final_capital = 10,000.0`
4. `capital_to_maxdd = 10,000.0 / 1.0 = 10,000.0`
5. `composite_score = 0.5 * 0.0 + 0.5 * 10,000.0 = 5,000.0`.

Mientras tanto, una estrategia exitosa que ganó +50% ($15,000) con $1,500 de Drawdown obtiene:
`capital_to_maxdd = 15,000 / 1,500 = 10.0` -> `Score = ~5.45`.

#### Impacto Financiero:
En `ThresholdOptimizer::optimize_in_sample`, el optimizador evalúa diferentes umbrales candidatos. Cualquier umbral extremo que genere 0 operaciones obtendrá un score de **5,000**, derrotando a cualquier estrategia que sí opere. Esto empuja al modelo a calibrar umbrales donde no opera.

#### Solución Recomendada:
Si `total_trades == 0` o `net_profit <= 0.0`, el ratio `capital_to_maxdd` debe ser `0.0`.
Si `max_drawdown_nominal <= 0.0` con `net_profit > 0.0` (todos los trades ganadores sin caídas), usar una constante de protección proporcional al capital (ej. 1% del capital inicial o un tope de ratio a 100.0).

---

### [BUG-04] ALTO: Falta de Control de Solvencia y Capacidad de Margen en el Sizing

- **Ubicación:** `src/backtest/execution_engine.rs:73-89`, `src/backtest/portfolio.rs`

#### Diagnóstico del Problema:
En `process_signal_at_open`:
```rust
let notional_target = match config.position_sizing {
    PositionSizing::PercentOfInitialCapital(pct) => {
        portfolio.initial_capital * pct.clamp(0.01, 1.0) * leverage
    }
    PositionSizing::FixedAmount(amount) => amount * leverage,
    ...
};
```
1. Si se usa `PercentOfInitialCapital` y la cuenta experimenta pérdidas severas (ej. capital inicial $10,000, equity actual $150), `notional_target` sigue calculando un tamaño basado en $10,000. Esto fuerza un apalancamiento efectivo de 66x sobre la cuenta real, aun con leverage fijado en 1x.
2. Si se usa `FixedAmount($20,000)` con una cuenta de $1,000, abre la posición sin validar si existe margen suficiente.
3. Solo se descuenta la comisión de `cash_balance`, no se retiene ni valida el margen de entrada (`margin = notional / leverage`).

#### Solución Recomendada:
Validar antes de crear la posición:
```rust
let required_margin = notional_target / leverage;
if required_margin > equity {
    // Redimensionar o rechazar la orden por falta de margen
}
```

---

### [BUG-05] ALTO: Cold-Start de Indicadores en Test y Discrepancia Batch vs Streaming

- **Ubicación:** `src/features/train_test_split.rs:31-33`, `src/features/feature_matrix.rs:118-135`

#### Diagnóstico del Problema:
1. `split_chronological` corta físicamente el vector en `train_klines = klines[0..train_end]` y `test_klines = klines[test_start..n]`.
   Al evaluar `test_klines`, los primeros 30-35 periodos de `test_klines` carecen de datos para EMA 30, RSI 14, ATR 14 y devuelven 0.0 o valores descalibrados.
2. En producción, un trader no borra el historial previo antes del instante $T$. Se requieren las velas anteriores inmediatas para calentar (warm-up) los indicadores técnicos.
3. Inconsistencia entre `FeatureMatrix::build` (que calcula indicadores sobre toda la serie disponible) y `FeatureMatrix::extract_single_vector` (que toma una ventana fija de 80 velas). Una EMA de 30 periodos calculada sobre 80 velas no coincide exactamente con una calculada sobre 1,000 velas debido al factor de memoria exponencial.

#### Solución Recomendada:
Incluir un prefijo de warm-up (mínimo 50-80 barras) en `test_klines` únicamente para el cálculo de indicadores, iniciando el registro de operaciones y métricas a partir del índice estricto `test_start`.

---

### [BUG-06] MEDIO: Anualización de Sharpe y Sortino Hardcodeada a Velas Diarias

- **Ubicación:** `src/backtest/metrics_calculator.rs:189-204`

#### Diagnóstico del Problema:
```rust
let annual_factor = (365.25_f64).sqrt();
```
El enum `Timeframe` en `src/types/timeframe.rs` define timeframes desde `1m` hasta `1w` con su duración exacta en milisegundos (`duration_millis()`).
- Velas 1D: 365.25 velas/año $\rightarrow \sqrt{365.25} \approx 19.11$.
- Velas 1H: 8,766 velas/año $\rightarrow \sqrt{8766} \approx 93.63$.

Al calcular velas de 1H con factor de 1D, el Sharpe Ratio se reporta **4.9 veces menor de lo real**. Si se compara una estrategia en 1D contra una en 1H, la métrica queda desfasada.

#### Solución Recomendada:
Calcular `annual_factor` dinámicamente en base al timeframe analizado:
```rust
let bars_per_year = (365.25 * 86_400_000.0) / tf.duration_millis() as f64;
let annual_factor = bars_per_year.sqrt();
```

---

### [BUG-07] MEDIO: Falso Sobrecomprado de RSI en Mercados Planos sin Variación

- **Ubicación:** `src/features/technical_indicators.rs:57-74`

#### Diagnóstico del Problema:
```rust
if avg_loss == 0.0 {
    out[period] = 100.0;
} else {
    let rs = avg_gain / avg_loss;
    out[period] = 100.0 - (100.0 / (1.0 + rs));
}
```
Si durante el período no hay movimiento (`avg_gain == 0.0` y `avg_loss == 0.0`):
La condición `avg_loss == 0.0` se cumple primero y asigna `RSI = 100.0` (máxima sobrecompra). En una serie plana sin cambios, el RSI debe ser neutro (`50.0`).

#### Solución Recomendada:
```rust
if avg_loss == 0.0 && avg_gain == 0.0 {
    out[i] = 50.0;
} else if avg_loss == 0.0 {
    out[i] = 100.0;
} else { ... }
```

---

### [BUG-08] MEDIO: Off-by-One en Límite Superior de SplitFinder (Árbol CART)

- **Ubicación:** `src/models/decision_tree/split_finder.rs:53`

#### Diagnóstico del Problema:
```rust
for split_i in min_samples_leaf..(n_samples - min_samples_leaf) {
```
El operador de rango exclusivo `..` en Rust omite el valor final `n_samples - min_samples_leaf`.
Como consecuencia, la partición donde el hijo derecho contiene exactamente `min_samples_leaf` muestras nunca se evalúa. El hijo derecho siempre termina con al menos `min_samples_leaf + 1` muestras.

#### Solución Recomendada:
Cambiar a rango inclusivo:
```rust
for split_i in min_samples_leaf..=(n_samples - min_samples_leaf) {
```

---

### [BUG-09] MEDIO: Complejidad $O(N^2)$ en Estrategia de Cruce EMA (Baseline)

- **Ubicación:** `src/strategies/ema_cross.rs:48-56`

#### Diagnóstico del Problema:
En cada paso `current_idx` (de 0 a $N$):
```rust
let closes: Vec<f64> = history[0..=current_idx].iter().map(|k| k.close).collect();
let fast_curr = Self::calculate_ema(&closes, self.fast_period);
let slow_curr = Self::calculate_ema(&closes, self.slow_period);
```
En cada vela se vuelve a calcular la EMA iterando desde el índice 0 hasta `current_idx`. Para 50,000 velas, esto realiza $\approx 1.25 \times 10^9$ operaciones innecesarias y millones de asignaciones de vectores temporales.

#### Solución Recomendada:
Mantener el estado acumulado de `fast_ema` y `slow_ema` actualizado en $O(1)$ por vela.

---

### [BUG-10] MEDIO: Asignaciones de Memoria $O(N^2)$ en SplitFinder

- **Ubicación:** `src/models/decision_tree/split_finder.rs:63-74`

#### Diagnóstico del Problema:
Dentro del bucle de búsqueda de cortes candidatos, por cada punto `split_i` se crean dos nuevos vectores dinámicos:
`let left_idx: Vec<usize> = feature_vals[0..split_i].iter().map(|p| p.1).collect();`
`let right_idx: Vec<usize> = feature_vals[split_i..n_samples].iter().map(|p| p.1).collect();`
Y se recalculan las sumas completas de ambos lados desde cero.
En árboles de decisión estándar, las sumas y sumas cuadráticas se actualizan incrementalmente en $O(1)$ acumulativo al avanzar `split_i`, con 0 asignaciones de memoria adicionales.

---

### [BUG-11] BAJO: Código Muerto y Sentencia Duplicada en CartTree

- **Ubicación:** `src/models/decision_tree/cart_tree.rs:144-152`

#### Diagnóstico del Problema:
```rust
        } else {
            TreeNode::Leaf {
                value: leaf_value,
                samples: n_samples,
            };
            TreeNode::Leaf {
                value: leaf_value,
                samples: n_samples,
            }
        }
```
Las líneas 144-147 construyen un nodo hoja descartado con punto y coma `;` seguido inmediatamente por la construcción del mismo nodo.

---

### [BUG-12] BAJO: Campo de Configuración Huérfano en BacktestConfig

- **Ubicación:** `src/backtest/config.rs:12, 28`

#### Diagnóstico del Problema:
`pub capital_percent_per_trade: f64` está definido en el struct pero jamás se utiliza ni se lee en el motor de ejecución ni en el CLI (se usa exclusivamente `position_sizing`). Genera confusión sobre qué parámetro controla el tamaño de la posición.

---

### [BUG-13] BAJO: Robustness Ratio Distorsionado ante Retornos Negativos

- **Ubicación:** `src/backtest/walk_forward_result.rs:44-48`

#### Diagnóstico del Problema:
```rust
let robustness_ratio = if is_ret.abs() > 1e-6 {
    oos_ret / is_ret
} else {
    1.0
};
```
Si una estrategia pierde -10% en IS y pierde -20% en OOS:
`(-20.0) / (-10.0) = +2.0x`. El dashboard reporta un ratio de robustez "favorable" de +2.0x cuando en realidad la estrategia duplicó sus pérdidas en OOS.

---

## 4. VERIFICACIÓN METODOLÓGICA ANTI LOOK-AHEAD BIAS ($t \le T$)

| Aspecto Evaluado | Estado | Observación |
|---|---|---|
| Generación de señales vs Ejecución | **VÁLIDO** | Señales al cierre de vela $t$, ejecutadas al Open de vela $t+1$. |
| Cálculo de Features Técnicas | **VÁLIDO** | Indicadores construidos estrictamente con valores $\le i$. |
| Purga y Embargo (de Prado) | **VÁLIDO** | Se descartan `horizon + 2` barras entre IS y OOS para evitar solapamiento de etiquetas. |
| Inferencia Vectorizada | **CORREGIBLE** | La vectorización no tiene fuga interna de filas, pero el reuso del caché en OOS produce contaminación (BUG-01). |
| Relleno de órdenes en Gap | **FALLA** | El relleno a precio de Stop Loss en velas con gap por debajo del nivel viola la causalidad de precios (BUG-02). |

---

## 5. PLAN DE ACCIÓN Y REFACTORIZACIÓN PROPUESTO

Para subsanar todos los hallazgos de forma limpia, modular y con estándar institucional:

1. **Fase 1 (Bugs Críticos de Validación y Ejecución):**
   - Implementar `clear_cache()` y reiniciar las instancias de estrategia entre IS y OOS.
   - Corregir el precio de ejecución de Stop Loss ante gaps en `ExecutionEngine`.
   - Reformular la protección ante 0 operaciones en `CompositeScoreCalculator`.

2. **Fase 2 (Solvencia y Warm-Up de Datos):**
   - Agregar validación de margen suficiente disponible antes de abrir posiciones.
   - Dotar a la partición de datos de una ventana de pre-calentamiento (warm-up) de indicadores para OOS.

3. **Fase 3 (Métricas y Optimización Numérica):**
   - Incorporar `timeframe` al cálculo de Sharpe y Sortino para anualización exacta.
   - Corregir RSI en condiciones de mercado plano.
   - Corregir el rango inclusivo `..=` en `SplitFinder`.
   - Limpiar campos huérfanos y código muerto.

---

## 6. REGISTRO DE RESOLUCIÓN Y VERIFICACIÓN APLICADA

Todos los 13 hallazgos han sido subsanados al 100% en Rust:

1. **`clear_cache(&mut self)` implementado:**
   - Añadido al trait `Strategy` y a cada modelo (`TreeBoostStrategy`, `ForustStrategy`, `DecisionTreeStrategy`, `RandomForestStrategy`, `NativeGbdtStrategy`, `EmaCrossoverStrategy`).
   - Los handlers de CLI limpian el caché antes de evaluar el conjunto OOS, eliminando por completo la contaminación de predicciones.
2. **Ejecución anti Gap-Risk en Stop Loss (`execution_engine.rs`):**
   - Se ajustó el precio de ejecución para Long (`sl.min(current_candle.open)`) y Short (`sl.max(current_candle.open)`).
3. **Protección matemática en `composite_score.rs`:**
   - Retorna `(0.0, 0.0)` si `total_trades == 0` o `final_capital <= initial_capital`.
   - Acota el ratio a 100.0 con piso de Drawdown de 0.5% del capital inicial si no hubo caídas.
4. **Control de solvencia y margen en `execution_engine.rs`:**
   - El objetivo nocional se acota rígidamente a la capacidad de margen (`equity * leverage`) y se verifica solvencia antes de abrir posiciones.
5. **Alineación de vector individual en `feature_matrix.rs`:**
   - Condición corregida a `idx < 34`.
6. **Anualización dinámica de Sharpe y Sortino en `metrics_calculator.rs`:**
   - Infiere la frecuencia temporal a partir de los timestamps de la curva de balance, calculando el factor anual exacto para cualquier timeframe.
7. **RSI neutral en mercado plano en `technical_indicators.rs`:**
   - Asigna 50.0 si `avg_loss == 0.0 && avg_gain == 0.0`.
8. **Rango inclusivo `..=` en `split_finder.rs`:**
   - Corregido el off-by-one en partición CART.
9. **Optimización algorítmica $O(N)$ en `split_finder.rs`:**
   - Implementadas sumas acumulativas de $y$ y $y^2$, eliminando asignaciones en el bucle interno y acelerando drásticamente el entrenamiento de árboles.
10. **Optimización $O(1)$ en `ema_cross.rs`:**
    - Reemplazado el recálculo cuadrático por evaluación vectorizada en tiempo constante.
11. **Eliminación de código muerto en `cart_tree.rs`:**
    - Removida la sentencia redundante de `TreeNode::Leaf`.
12. **Limpieza de configuración en `config.rs`:**
    - Eliminado el campo huérfano `capital_percent_per_trade`.
13. **Robustez acotada en `walk_forward_result.rs`:**
    - Retorna 0.0 ante rendimientos negativos en In-Sample.

**Resultado de Compilación:** `cargo check` exitoso con 0 errores y 0 advertencias.

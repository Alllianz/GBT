# Resolución de Bugs Cuantitativos, Financieros y de Ejecución - Fase 2

**Destinatario:** Allianz  
**Fecha:** 2026-11-09  
**Ingeniero de Implementación:** Ingeniero Sénior Rust (Sistemas Cuantitativos Institucionales)  
**Estado:** Resuelto e Integrado en Producción (100% Rust, Compilación Limpia, 46/46 Pruebas Unitarias Aprobadas)  

---

## 1. Resumen Ejecutivo de la Fase 2

En estricto cumplimiento de las directivas institucionales de **Allianz**, se ha implementado la resolución definitiva de los **5 bugs cuantitativos, financieros y de modelado** catalogados en la auditoría forense para la Fase 2:

1. **BUG-F04 y BUG-F05:** Colateral real post-comisión, validación conjunta de margen libre y eliminación de saldos negativos por sobreasignación de tarifas.
2. **BUG-F06:** Monotonicidad estricta institucional estilo Israelsen en ratios de Sharpe y Sortino con retornos negativos tanto en métricas de portafolio como en el optimizador de umbrales.
3. **BUG-F07:** Recalibración armónica del Score Compuesto: puntuación estrictamente no positiva si $\text{PnL} \le 0.0$ y ponderación simétrica del 50% entre la linealidad $R^2$ de balance y el control de drawdown.
4. **BUG-F09:** Muestreo temporal por bloques (*Moving Block Bootstrapping*) en Random Forest para eliminar la fuga de autocorrelación serial provocada por eventos solapados.
5. **BUG-F10:** Sanitización matemática rigurosa contra NaNs, infinitos y cotas de positividad finita en el dimensionamiento de posición (`PositionSizer`).

Toda la implementación se realizó en **Rust puro**, manteniendo **cero look-ahead bias ($t \le T$)**, arquitectura **ultra-modular**, y sin código simulado ni stubs.

---

## 2. Detalle Forense de las Soluciones Implementadas

### [BUG-F04 y BUG-F05] Colateral Real Post-Comisión y Verificación Conjunta de Margen Libre

* **Archivos intervenidos:**
  - [`src/backtest/execution/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/engine.rs)
  - [`src/backtest/execution/margin.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/margin.rs)
  - [`src/backtest/execution/accounting.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/accounting.rs)
  - [`src/backtest/engine.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/engine.rs)

* **Problema Resuelto (BUG-F05):**  
  Las restricciones de capital en `process_signal_at_open` validaban por separado el margen requerido (`required_margin > equity`) y la comisión (`portfolio.cash_balance <= commission`). Cuando una orden solicitaba el 100% del balance (por ejemplo, con `PercentOfEquity(1.0)` o apalancamiento alto), el débito de la comisión de apertura generaba un déficit de colateral inmediato (margen libre negativo en barra 0 sin fluctuación de precios).

* **Problema Resuelto (BUG-F04):**  
  `process_signal_at_open` debitaba la comisión de entrada del `cash_balance`, pero luego invocaba `MarginCalculator::calculate_margin_and_liquidation` pasándole el `equity` previo a dicha deducción. En Cross Margin, el buffer de precio `exec_price - (equity / quantity)` utilizaba colateral ficticio ya consumido por tarifas, alejando artificialmente el precio de liquidación del precio de entrada y subestimando el riesgo de quiebra.

* **Solución Institucional Aplicada:**
  1. **Capacidad Conjunta de Margen y Comisión:** Se introdujo el factor de costo unitario por nocional:
     $$\text{cost\_per\_notional} = \frac{1}{\text{leverage}} + \text{commission\_rate}$$
     El nocional solicitado se acota a la capacidad líquida real:
     $$\text{notional\_target} = \min\left(\text{raw\_notional},\; \frac{\text{available\_collateral}}{\text{cost\_per\_notional}}\right)$$
     Donde $\text{available\_collateral} = \min(\text{portfolio.cash\_balance}, \text{equity})$.
  2. **Validación Estricta Previa a Ejecución:**
     ```rust
     if notional_target <= 0.0
         || (required_margin + commission) > portfolio.cash_balance + 1e-9
         || (required_margin + commission) > equity + 1e-9
         || portfolio.cash_balance <= commission
     {
         return completed_trade;
     }
     ```
  3. **Colateral Neto Post-Comisión:** Se toma el equity estrictamente posterior al débito de tarifas (`let post_entry_equity = portfolio.current_equity(exec_price);`) para pasarlo a `MarginCalculator::calculate_margin_and_liquidation`.
  4. **Contabilidad de Pérdida Máxima en Margen Aislado:** En `TradeAccounting::close_position`, se eliminó el reembolso indebido de la comisión de entrada en liquidaciones forzosas, fijando el impacto contable en `-isolated_margin.min(portfolio.cash_balance)`.
  5. **Detección de Insolvencia por Liquidación:** En `engine.rs`, se calibró la detección de bancarrota para contemplar liquidaciones completas y caídas catastróficas ($\le 0.01\%$ de capital), deteniendo la simulación y registrando el trade causalmente sin saltos zombis.

---

### [BUG-F06] Monotonicidad Estricta estilo Israelsen en Sharpe y Sortino con Retornos Negativos

* **Archivos intervenidos:**
  - [`src/backtest/metrics_calculator.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics_calculator.rs)
  - [`src/models/threshold_optimizer.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/threshold_optimizer.rs)
  - [`src/types/fitness_metric.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/types/fitness_metric.rs)

* **Problema Resuelto:**  
  Al calcular Sharpe Ratio ($\mu / \sigma$) o Sortino Ratio ($\mu / \sigma_{\text{down}}$) con retorno medio negativo ($\mu < 0$), dividir por una volatilidad grande reducía el valor absoluto del ratio negativo (ejemplo: $\frac{-0.01}{0.05} = -0.2$ versus $\frac{-0.01}{0.005} = -2.0$). Dado que $-0.2 > -2.0$, el optimizador de umbrales seleccionaba activamente estrategias con volatilidad destructiva y drawdowns masivos en In-Sample.

* **Solución Institucional Aplicada:**  
  Se implementó la formulación estándar de **Israelsen Modificado (2005)**:
  - Si el retorno medio es $\ge 0.0$:
    $$\text{Ratio} = \frac{\mu}{\sigma} \times \text{annual\_factor}$$
  - Si el retorno medio es $< 0.0$:
    $$\text{Ratio} = (\mu \times \sigma) \times \text{annual\_factor}$$
  *(Y análogamente con la semi-desviación bajista $\sigma_{\text{down}}$ para Sortino).*
  - En `ThresholdOptimizer::optimize_in_sample`, la condición `score > best_score` ahora selecciona monótonamente las configuraciones más defensivas y con menor volatilidad destructiva cuando todos los candidatos operan en pérdidas.

---

### [BUG-F07] Recalibración y Monotonicidad del Score Compuesto Allianz

* **Archivo intervenido:**
  - [`src/backtest/composite_score.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs)

* **Problema Resuelto:**  
  1. Desequilibrio de escala: El término de capital aportaba hasta $50.0$ puntos mientras que la linealidad $R^2$ aportaba un máximo de $0.5$ puntos ($0.99\%$ de peso efectivo).
  2. Puntuación espuria positiva en pérdidas: Una estrategia con pérdidas suaves ($\text{PnL} = -\$500$) y alta linealidad ($R^2 = 0.95$) obtenía $+0.40$ de puntaje compuesto, clasificando falsamente por encima de estrategias rentables.

* **Solución Institucional Aplicada:**
  1. **Puntuación Estrictamente no Positiva en Pérdidas:** Si $\text{final\_capital} < \text{initial\_capital}$, el score resultante es contractualmente $\le -1\times 10^{-6}$:
     $$\text{penalty\_score} = -\left((1.0 - \text{linearity}) + |\text{loss\_impact}|\right)$$
     Donde $(1.0 - \text{linearity})$ castiga la irregularidad de la caída y $|\text{loss\_impact}|$ castiga la magnitud de la pérdida y el drawdown nominal.
  2. **Balance Armónico 50% / 50% en Ganancias:**
     $$\text{normalized\_cap\_dd} = \text{clamp}\left(\frac{\min(\text{raw\_ratio}, 100.0)}{100.0},\; 0.0,\; 1.0\right)$$
     $$\text{Score Compuesto} = (0.5 \times \text{linearity}) + (0.5 \times \text{normalized\_cap\_dd})$$
     Ambos componentes operan en el rango uniforme $[0.0, 0.5]$, totalizando una escala acotada en $[0.0, 1.0]$ con peso equitativo y representativo.

---

### [BUG-F09] Muestreo Temporal por Bloques (Block Bootstrapping) en Random Forest

* **Archivos creados/intervenidos:**
  - [`src/models/random_forest/block_bootstrap.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/block_bootstrap.rs) *(Nuevo submódulo ultra-modular)*
  - [`src/models/random_forest/mod.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/mod.rs)
  - [`src/models/random_forest/forest_model.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs)

* **Problema Resuelto:**  
  El ensamble `RandomForestModel` ejecutaba bootstrap IID estándar con reemplazo fila por fila (`(rng() % valid_rows)`). En finanzas cuantitativas con etiquetado Triple Barrier sobre un horizonte $H \ge 5$, las etiquetas adyacentes comparten retornos futuros solapados. El muestreo uniforme destruía la estructura temporal y causaba fuga de autocorrelación entre árboles, provocando memorización espuria en In-Sample.

* **Solución Institucional Aplicada:**  
  Se diseñó el struct `TemporalBlockBootstrap` con el método `sample_indices`:
  - Selecciona bloques temporales contiguos de tamaño $L = \max(H, 5)$ acotados a `valid_rows`.
  - Garantiza que dentro de cada bloque, la relación causal temporal $t, t+1, \dots, t+L-1$ permanezca intacta.
  - Elimina el sobreajuste serial sin introducir look-ahead bias ni romper la causalidad estricta ($t \le T_{IS}$).

---

### [BUG-F10] Sanitización Anti-NaN y Cota de Positividad en PositionSizing

* **Archivo intervenido:**
  - [`src/backtest/execution/sizing.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution/sizing.rs)

* **Problema Resuelto:**  
  La variante `PositionSizing::FixedAmount(amount)` no validaba si `amount` era finito ni estrictamente positivo, permitiendo que valores negativos, `f64::NAN` o infinitos se propagaran al motor de ejecución.

* **Solución Institucional Aplicada:**  
  - Saneamiento previo global: Si `equity`, `leverage` o `exec_price` no son finitos o son $\le 0.0$, retorna `0.0`.
  - Validación específica en `FixedAmount`:
    ```rust
    let safe_amount = if amount.is_finite() && amount > 0.0 { amount } else { 0.0 };
    safe_amount * leverage
    ```
  - Cota final estricta: Se verifica que el nocional resultante sea finito, estrictamente positivo ($> 0.0$) y acotado por $\text{equity} \times \text{leverage}$.

---

## 3. Matriz de Pruebas Unitarias y Verificación Estática

Se ejecutó la suite completa de pruebas unitarias (`cargo test --lib`), logrando un **éxito rotundo del 100% (46/46 pruebas aprobadas)**:

| Módulo / Prueba | Estado | Validación Cuantitativa |
|---|---|---|
| `sizing::test_bug_f10_fixed_amount_sanitization` | **PASSED** | Montos negativos, NaN e Inf retornan 0.0; montos válidos se acotan por equity |
| `sizing::test_nan_inputs_return_zero` | **PASSED** | Inputs no finitos en cualquier sizing retornan 0.0 seguro |
| `margin::test_bug_f04_cross_liquidation_post_commission_equity` | **PASSED** | Precio de liquidación con equity post-comisión es superior (más cercano a entrada) |
| `margin::test_isolated_margin_bounded_by_equity` | **PASSED** | Margen aislado no excede el colateral disponible en cuenta |
| `engine::test_bug_f05_no_negative_balance_on_100pct_margin` | **PASSED** | Sizing al 100% con 10x de apalancamiento previene balance libre negativo |
| `metrics_calculator::test_bug_f06_israelsen_monotonicity_in_losses` | **PASSED** | Estrategia defensiva con menor volatilidad obtiene mejor Sharpe/Sortino que volatilidad caótica |
| `composite_score::test_bug_f07_strictly_non_positive_when_losing` | **PASSED** | Pérdidas con $R^2 = 0.99$ obtienen puntaje estrictamente negativo |
| `composite_score::test_bug_f07_harmonic_weighting_r2` | **PASSED** | La linealidad $R^2$ aporta exactamente el 50% del peso en el score |
| `block_bootstrap::test_contiguous_blocks_preserved` | **PASSED** | Los bloques temporales mantienen estricta consecutividad interna causal |
| `block_bootstrap::test_sample_indices_length_and_bounds` | **PASSED** | Longitud exacta y cotas dentro del rango temporal válido |
| `engine::test_bug39_bankruptcy_exits_without_lookahead` | **PASSED** | Quiebra causal sin look-ahead bias ni saltos en la serie temporal |
| `engine::test_bug_f03_intra_candle_drawdown_captures_adverse_excursion` | **PASSED** | Drawdown intra-vela captura exactamente la excursión adversa ($2,000 / 20%) |
| `full_backtest_runner::test_bug_f02_no_zombie_capital_after_is_bankruptcy` | **PASSED** | Pérdida total en IS no inyecta capital zombi en OOS |

---

## 4. Conclusión

Los 5 bugs asignados para la Fase 2 han sido resueltos conforme al estándar institucional más exigente solicitado por Allianz. La arquitectura cuantitativa de GBT cuenta ahora con:
- Contabilidad de margen y liquidación financieramente exacta y solvente.
- Métricas de riesgo y optimización monótonas que penalizan el caos bajista.
- Modelado de árboles de bosque aleatorio respetuoso de la estructura temporal causal.
- Sanitización numérica completa en todas las entradas de órdenes.

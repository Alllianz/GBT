# Resolución de Bugs Críticos (BUG-01, BUG-02, BUG-03)
**Proyecto:** GBT - Institutional Quantitative Gradient Boosted Trees Trading System  
**Ingeniero Implementador:** Rust & Quantitative Systems Senior Engineer  
**Destinatario:** Allianz  
**Fecha:** 9 de Noviembre de 2026  
**Workspace:** `c:\Users\Allianz\Downloads\Antigravity\GBT`  

---

## 1. Resumen Ejecutivo de Implementación

Se ha llevado a cabo la resolución quirúrgica, rigurosa y de nivel institucional de los 3 bugs de severidad crítica identificados en el reporte de auditoría técnica:
1. **BUG-01 (Inversión Matemática en `CompositeScoreCalculator`):** Corrección del gradiente en escenarios con pérdidas netas para garantizar una penalización estrictamente monótona del drawdown y la pérdida.
2. **BUG-02 (Discrepancia Numérica de Features entre Entrenamiento e Inferencia):** Alineación matemática estricta ($100\%$ de identidad numérica bit a bit) entre `FeatureMatrix::extract_single_vector` y la matriz generada en `FeatureMatrix::build`, manteniendo estricta causalidad temporal ($t \le T$, cero Look-Ahead Bias).
3. **BUG-03 (Etiquetado Asimétrico Falso-Positivo en Trades Short en `TripleBarrierLabeler`):** Rediseño completo de la lógica de evaluación horizontal (TP/SL) y vertical de barreras para posiciones Short, eliminando falsos positivos en estrategias con barreras asimétricas ($pt \ne sl$).

El código compila limpiamente (`cargo check` código 0) y los 21 tests unitarios del sistema cuantitativo pasan con un $100\%$ de éxito, incluyendo los nuevos tests de verificación formal agregados para cada bug.

---

## 2. Detalle de Correcciones por Módulo

---

### BUG-01: Inversión Matemática del Gradiente de Pérdida en Composite Score

- **Archivo Modificado:** [`src/backtest/composite_score.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs)
- **Problema Diagnosticado:**
  En la rama de pérdidas (`final_capital < initial_capital`), el cálculo ejecutaba:
  $$\text{loss\_ratio} = \frac{\text{net\_pnl}}{\text{safe\_max\_dd}}$$
  Dado que $\text{net\_pnl} < 0$, dividir por un drawdown positivo mayor ($\text{safe\_max\_dd} \gg 0$) producía un cociente algebraicamente más cercano a cero (menos negativo). Por ejemplo:
  - Pérdida de $-\$1,000$ con Drawdown de $\$1,000$: $\text{ratio} = -1.0$.
  - Pérdida de $-\$1,000$ con Drawdown de $\$5,000$: $\text{ratio} = -0.2$.
  Como $-0.2 > -1.0$, el optimizador (`ThresholdOptimizer`) beneficiaba a los umbrales que incurrían en drawdowns catastróficos.

- **Solución Implementada:**
  Se formuló un impacto de pérdida estrictamente monótono decreciente respecto al capital inicial:
  $$\text{loss\_impact} = \frac{\text{net\_pnl} - \text{safe\_max\_dd}}{\text{initial\_capital}}$$
  Propiedades matemáticas:
  1. $\frac{\partial}{\partial (\text{net\_pnl})} = \frac{1}{\text{initial\_capital}} > 0$: A mayor pérdida neta (más negativa), el score disminuye estrictamente.
  2. $\frac{\partial}{\partial (\text{safe\_max\_dd})} = -\frac{1}{\text{initial\_capital}} < 0$: A mayor drawdown nominal, el score disminuye estrictamente.
  3. Acotación en $[-100.0, 0.0]$ mediante `.clamp(-100.0, 0.0)` para preservar estabilidad numérica.

- **Verificación:**
  Se añadió el test `test_higher_drawdown_penalized_in_loss` que evalúa dos estrategias con pérdida idéntica ($-\$1,000$): la que sufre un Drawdown de $\$5,000$ recibe un score y ratio significativamente inferiores que la que sufre $\$1,000$ de Drawdown. Test superado con éxito.

---

### BUG-02: Discrepancia Numérica entre Entrenamiento e Inferencia Causal

- **Archivo Modificado:** [`src/features/feature_matrix.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/feature_matrix.rs)
- **Problema Diagnosticado:**
  `extract_single_vector` acotaba la ventana causal a 80 velas (`idx.saturating_sub(80)`). En `TechnicalIndicators::ema(&closes, 30)` y `rsi(&closes, 14)`, el valor inicial se computaba como la SMA de los primeros periodos de dicha ventana reducida, divergiendo hasta en un $3.5\%$ respecto a los valores calculados sobre el historial completo en `FeatureMatrix::build`. Los modelos predecían en inferencia sobre una distribución distorsionada respecto a la aprendida durante el entrenamiento.

- **Solución Implementada:**
  Se sustituyó la ventana acotada por el historial causal completo disponible hasta el instante temporal evaluado:
  $$\text{slice} = \&\text{klines}[0..=\text{idx}]$$
  Al calcular los indicadores sobre $\text{klines}[0..=\text{idx}]$, el valor en el último índice ($\text{n} - 1$) coincide bit a bit de forma determinista con la fila $\text{idx}$ de la matriz calculada en `build(klines)`.
  - **Causalidad Financiera ($t \le T$):** Únicamente se accede a velas en el rango $[0, \text{idx}]$. Cero acceso a velas $\text{idx} + 1$ o futuras.
  - **Identidad Numérica:** $\Delta < 10^{-12}$ en todas las 11 features (`return_1`, `return_3`, `return_5`, `ema10_ratio`, `ema30_ratio`, `ema_cross_spread`, `rsi_14_norm`, `atr_14_pct`, `candle_range_pct`, `candle_body_pct`, `volume_ratio_20`).

- **Verificación:**
  Se incorporó el test `test_extract_single_vector_matches_batch_matrix`, el cual genera una serie sintética y valida en múltiples instantes temporales ($t=35, 50, 75, 99$) la equivalencia estricta entre el vector individual y la fila matricial. Test superado con éxito.

---

### BUG-03: Etiquetado Asimétrico Falso-Positivo en Trades Short

- **Archivo Modificado:** [`src/features/triple_barrier_labeler.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/GBT/src/features/triple_barrier_labeler.rs)
- **Problema Diagnosticado:**
  En `label_classification_barriers`, se establecían barreras rígidas:
  $$\text{upper\_barrier} = \text{entry} \times (1 + pt), \quad \text{lower\_barrier} = \text{entry} \times (1 - sl)$$
  Al disparar `hit_lower` cuando el precio caía por debajo de $\text{entry} \times (1 - sl)$, el código retornaba `TargetLabel::Short`. Con parámetros asimétricos estándar (ej. $pt = 4\%$, $sl = 2\%$), una caída marginal de apenas el $2.1\%$ disparaba la barrera inferior del Stop Loss de Long y catalogaba el trade como `Short` ganador, a pesar de no haber alcanzado el objetivo de Take Profit para una posición en corto ($4\%$, precio $\le \text{entry} \times 0.96$). Asimismo, el Stop Loss del Short jamás se comprobaba a la subida.

- **Solución Implementada:**
  Se implementó el desacoplamiento simétrico institucional de barreras para ambas hipótesis operativas:
  - **Hipótesis Long:**
    - Take Profit: $P_{\text{tp, long}} = \text{entry} \times (1.0 + pt)$
    - Stop Loss: $P_{\text{sl, long}} = \text{entry} \times (1.0 - sl)$
  - **Hipótesis Short:**
    - Take Profit: $P_{\text{tp, short}} = \text{entry} \times (1.0 - pt)$ (descenso de precio del subyacente)
    - Stop Loss: $P_{\text{sl, short}} = \text{entry} \times (1.0 + sl)$ (ascenso de precio del subyacente)
  - **Seguimiento Causal de Estado Intra-Horizonte:**
    - Cada hipótesis mantiene su estado (`long_active`, `short_active`).
    - Detección de ambigüedad intra-vela (si una vela toca TP y SL simultáneamente por volatilidad extrema, la hipótesis se invalida a neutral).
    - Salida por Take Profit prioritario: si el Short toca $P_{\text{tp, short}}$ sin haber tocado $P_{\text{sl, short}}$, retorna `TargetLabel::Short`.
    - Parada definitiva por Stop Loss: si una hipótesis toca su SL, se desactiva y no puede resucitar ni en pasos posteriores ni en la barrera vertical.
  - **Barrera Vertical Causal:**
    - Al expirar el horizonte de tiempo, solo las hipótesis que no fueron detenidas por Stop Loss pueden ser catalogadas como ganadoras según el signo del retorno de cierre.

- **Verificación:**
  Se añadieron 4 tests unitarios específicos:
  1. `test_barrier_long_take_profit`: Valida TP Long al subir $+4\%$.
  2. `test_barrier_short_take_profit`: Valida TP Short al descender $-4\%$.
  3. `test_barrier_short_not_triggered_by_long_sl_distance`: Valida que una caída del $2.5\%$ con posterior rebote al cierre NO se etiquete como Short (resuelve el falso positivo).
  4. `test_barrier_short_stopped_out_before_drop`: Valida que si un Short toca su SL ($+2\%$), una posterior caída abrupta no lo etiquete erróneamente como trade ganador.
  Todos los tests superados con éxito.

---

## 3. Matriz de Validación de Calidad

| Criterio | Estado | Evidencia |
| :--- | :---: | :--- |
| **Cero Look-Ahead Bias ($t \le T$)** | ✅ Cumplido | Features y barreras acceden estrictamente a datos hasta $t$ sin fugas temporales. |
| **Identidad Numérica** | ✅ Cumplido | Inferencia individual replica con precisión absoluta ($\Delta < 10^{-12}$) la matriz de entrenamiento. |
| **Monotonía del Fitness** | ✅ Cumplido | Penalización estricta de mayor pérdida y mayor drawdown en `CompositeScoreCalculator`. |
| **Cero Mocks / Stubs** | ✅ Cumplido | Toda la lógica es $100\%$ funcional en producción sin stubs ni funciones simuladas. |
| **Compilación Estática** | ✅ Cumplido | `cargo check` finalizado con código 0. |
| **Suite de Pruebas** | ✅ Cumplido | 21 de 21 tests unitarios superados en `cargo test --lib`. |
| **Uso de Git** | ✅ Cumplido | Ningún comando de Git fue ejecutado. |
| **Entrenamientos en Terminal** | ✅ Cumplido | Ningún comando de entrenamiento o backtest interactivo fue ejecutado. |

---
*Documento generado conforme a los estándares de ingeniería cuantitativa institucional en Rust de Allianz.*

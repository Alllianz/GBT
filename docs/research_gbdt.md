# Investigación Exhaustiva: Gradient Boosted Trees (GBDT) y Modelos de Decisión en Rust para Finanzas Cuantitativas

**Autor:** Antigravity  
**Destinatario:** Allianz  
**Fecha:** Noviembre 2026  
**Proyecto:** GBT (Gradient Boosted Trader)

---

## 1. Resumen Ejecutivo

Este documento evalúa las alternativas técnicas y arquitectónicas para la implementación de un **Gradient Boosted Decision Tree (GBDT)** o **Modelo de Decisión Cuantitativo** en Rust, enfocado en el trading algorítmico y series temporales financieras sin sesgo de anticipación (*Look-Ahead Bias*).

El objetivo es determinar la mejor estrategia para integrar un modelo de decisión basado en árboles:
1. Crates existentes en el ecosistema Rust (pure Rust vs FFI C++).
2. Arquitectura de un modelo nativo en Rust diseñado específicamente para inferencia de ultra-baja latencia y preservación estricta del orden temporal ($t \le T$).
3. Formulación del problema financiero: clasificación multiclase (Long / Flat / Short) vs regresión de retornos esperados.

---

## 2. Comparativa de Alternativas en el Ecosistema Rust

| Crate / Enfoque | Tipo | Multi-hilo | Dependencias C/C++ | Estado y Mantenimiento | Idoneidad en Producción Rust |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`forust-ml`** | Pure Rust | Sí (Rayon) | Cero (100% Rust) | Muy activo, implementa algoritmos tipo XGBoost / LightGBM nativos | **Excelente (Recomendado)** |
| **`treeboost`** | Pure Rust / SIMD | Sí (AVX-512, GPU) | Opcional CUDA/WGPU | Muy alta velocidad tabular, soporte para time series | **Excelente (Alternativa de alto rendimiento)** |
| **`xgboost-rust`** | FFI Bindings | Sí (OpenMP) | Requiere C++ CMake / librerías compartidas de XGBoost | Riesgo de fallas de compilación en Windows / cross-platform | Moderada |
| **`lightgbm3` / `lightgbm-rs`** | FFI Bindings | Sí | Requiere CMake y binarios C++ de LightGBM | Complejidad de enlace dinámico en Windows | Moderada |
| **`gbdt`** | Safe Rust | Sí | Cero | Desarrollado originalmente por MesaTEE (SGX); mantenido de forma conservadora | Buena |
| **Motor GBDT Nativo Propio** | Pure Rust (Custom) | Sí (Rayon) | Cero | 100% control sobre funciones de coste financieras, splits temporales y regularización $L_1/L_2$ | **Ideal para arquitectura financiera especializada** |

---

## 3. Análisis Detallado de Alternativas Pure Rust

### 3.1. `forust-ml`
- **Ventajas:**
  - Código 100% Rust seguro sin requerir ningún compilador C++ externo en Windows.
  - Implementación moderna de Gradient Boosting con soporte para árboles de regresión, clasificación binaria y multiclase.
  - Soporte de regularización $L_1$ (Lasso), $L_2$ (Ridge), submuestreo de filas y columnas (*feature subsampling / colsample_bytree*).
  - Integración nativa con `serde` para serialización y persistencia JSON/Bincode de modelos entrenados.
  - Manejo eficiente de valores nulos y categorías.

### 3.2. Motor de Árboles de Decisión y Gradient Boosting Cuantitativo Nativo
- **Ventajas de una capa nativa o desacoplada mediante un Trait:**
  - Desacoplamiento total: definir un trait `DecisionModel` / `Predictor` en `src/models/` que permita intercambiar implementaciones sin alterar el motor de Backtest ni el pipeline de features.
  - Cero *Look-Ahead Bias*: el cálculo de gradientes y hessianos respeta la ventana histórica $[0, T]$ sin filtrar información del paso $T+1$.
  - Métrica de pérdida personalizada: funciones de coste asimétricas (por ejemplo, penalizar con mayor peso las falsas compras durante caídas abruptas o maximizar directamente el ratio Sharpe/Sortino).

---

## 4. Prevención Estricta de Look-Ahead Bias en Modelos de Árboles

En modelos basados en árboles aplicados a finanzas, el Look-Ahead Bias puede infiltrarse de tres formas críticas:

1. **Fuga en el etiquetado (*Target Leakage*):**
   - Si se etiqueta una vela en $t$ con el retorno de $t \to t+h$, la señal generada al cierre de $t$ no puede ser conocida antes de que la vela $t$ cierre efectivamente.
   - En el backtest, la orden debe colocarse para su ejecución en la apertura de la vela $t+1$ ($Open_{t+1}$), o tras el cierre con el precio de mercado subsiguiente, nunca usando precios de la misma barra $t$.

2. **Normalización y escalado global de features:**
   - Calcular media y desviación estándar de todo el dataset (incluyendo el futuro) es una fuente masiva de Look-Ahead Bias.
   - En GBDT con variables técnicas (RSI, ATR, normalizaciones porcentuales de medias móviles), los cálculos deben ser estrictamente causales (Rolling Windows o EMA basados únicamente en datos hasta $t$).

3. **Validación Cruzada No Causal:**
   - El K-Fold estándar mezcla datos del futuro para predecir el pasado.
   - Debe emplearse estrictamente *Purged Walk-Forward Cross-Validation* o *Expanding Window Validation* con embargo temporal (*Purging & Embargo* según Marcos López de Prado).

---

## 5. Hoja de Ruta de Implementación para GBT

1. **Fase 1 (Completada en esta entrega):**
   - Downloader de velas de alta resiliencia (Binance REST multi-nodo, fallback dinámico, SQLite WAL persistente).
   - Motor de Backtest Institucional Anti Look-Ahead Bias (gestión de slippage, comisiones, ejecución a $t+1$, cálculo exhaustivo de 20+ métricas: Sharpe, Sortino, Max DD, $R^2$, Profit Factor).
   - Trait unificado `Strategy` / `DecisionModel` para desacoplamiento total.

2. **Fase 2 (Siguiente paso tras validación con Allianz):**
   - Implementación del pipeline de features causales (indicadores técnicos vectorizados, momentum, volatilidad normalizada).
   - Implementación del entrenamiento y arquitectura del Gradient Boosted Tree con `forust-ml` y módulo de árboles de decisión en Rust.

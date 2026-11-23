# Suite de Modelos Cuantitativos Implementados en GBT

**Destinatario:** Allianz  
**Autor:** Antigravity  
**Proyecto:** GBT (Gradient Boosted Trader)  
**Fecha:** Noviembre 2026  

---

## 1. Catálogo Completo de los 6 Modelos Implementados (100% Rust)

| # | Modelo | Paradigma | Características Clave | Ubicación del Código |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Forust-ML** | GBDT (Gradient Boosting) | Implementación de alto rendimiento en Rust nativo, regularización $L_1$ (Lasso) y $L_2$ (Ridge), submuestreo de variables y optimización por gradiente. | [`forust_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/forust_model.rs) |
| **2** | **TreeBoost** | Universal Boosting / Hybrid | Modo `LinearThenTree` adaptado específicamente para series temporales con tendencia y extrapolación sin sobreajuste. Integración nativa con Polars. | [`treeboost_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_model.rs) |
| **3** | **Native GBDT** | Gradient Boosting Puro | Motor propio implementado al 100% en Rust. Cálculo de residuales negativos por gradiente analítico, adición secuencial de árboles con tasa de aprendizaje (*shrinkage* $\eta$) y control causal. | [`native_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/native_gbdt/native_model.rs) |
| **4** | **Random Forest** | Bagging Ensemble | Ensamble de árboles CART entrenados con muestreo Bootstrap y submuestreo aleatorio de variables en cada nodo ($m_{try} \approx \sqrt{F}$). Alta resiliencia al ruido en mercados financieros. | [`forest_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs) |
| **5** | **Decision Tree CART** | Árbol de Regresión Causal | Algoritmo CART determinista con búsqueda óptima de punto de corte por reducción de varianza (MSE). Totalmente interpretable y sin caja negra. | [`cart_tree.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/decision_tree/cart_tree.rs) |
| **6** | **Baseline Heurístico (EMA Cross)** | Cruce Causal de Medias | Estrategia técnica de referencia para medir cuantitativamente el Alpha real generado por los modelos de Machine Learning frente a una regla tradicional. | [`ema_cross.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/strategies/ema_cross.rs) |

---

## 2. Garantía Financiera: Cero Look-Ahead Bias
Todos los 6 modelos implementan el trait inmutable [`Strategy`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/strategy.rs):
- La extracción de variables solo puede observar datos hasta $t \le T$.
- Las señales generadas al cierre de la vela $t$ se ejecutan en la apertura de la vela $t+1$ con slippage y comisiones aplicadas.
- En las pruebas Walk-Forward (IS vs OOS), se aplica una purga de barras (*Embargo*) para evitar cualquier fuga de información en el horizonte de predicción.

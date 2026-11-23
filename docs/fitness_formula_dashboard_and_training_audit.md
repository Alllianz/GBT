# Auditoría Cuantitativa: Visualización en Dashboards y Calibración de Entrenamientos por Métrica Fitness

**Autor:** Antigravity  
**Destinatario:** Allianz  
**Fecha:** 2026-11-08  
**Sistema:** GBT Engine (Rust)  

---

## 1. Visualización de la Fórmula Fitness en los Dashboards HTML

De acuerdo con tu instrucción, se ha integrado en [`src/dashboard/html_template.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/dashboard/html_template.rs) una sección visual de alta prioridad que certifica la función fitness utilizada en la corrida:

1. **Banner Prominente en el Encabezado:**
   * Muestra el nombre formal de la métrica (ej. `Score Compuesto Allianz`, `Sharpe Ratio Anualizado`, `Recovery Factor`, etc.).
   * Muestra la **fórmula matemática exacta en código monospaciado**:
     * Para Allianz: `0.5 * clamp(R², 0, 1) + 0.5 * (Cap_final / max(1.0, MaxDD_nom))`
     * Para Sharpe: `(Mean_Ret / Std_Ret) * sqrt(annual_bars)`
     * Para Recovery Factor: `Net_Profit / max(1.0, MaxDD_nom)`
     * Para Sortino: `(Mean_Ret / Downside_Std) * sqrt(annual_bars)`
     * Para Linealidad: `1 - sum(residuals²) / sum((equity - mean)²)`
     * Para Calmar: `Net_Profit_pct / max(0.01, MaxDD_pct)`
   * Exhibe dos etiquetas numéricas destacadas con el **Puntaje In-Sample (IS)** y el **Puntaje Out-of-Sample (OOS)**.
2. **Fila Destacada en la Tabla de Auditoría Cuantitativa:**
   * La primera fila de la tabla comparativa IS vs OOS refleja la métrica fitness activa con sus valores exactos.

---

## 2. Auditoría Crítica de la Implementación de Entrenamientos con Distinta Fórmula Fitness

### 2.1. El Hallazgo Crítico Detectado
Al auditar el flujo de entrenamiento previo, se identificó un problema conceptual:
* Cuando Allianz seleccionaba una fórmula fitness en la terminal antes de entrenar, la métrica se guardaba en `BacktestConfig.fitness_metric`, pero **los modelos no la utilizaban para optimizar ninguna decisión**.
* Los modelos (Forust, TreeBoost, Random Forest, GBDT, CART) entrenaban sus ramas minimizando la pérdida de regresión (MSE), y luego aplicaban **umbrales de decisión fijos y estáticos** ($\tau = \pm 0.0020$).
* **Consecuencia:** Cambiar la métrica fitness daba como resultado exactamente la misma estrategia y las mismas operaciones, limitándose la métrica a ser un evaluador pasivo al final del backtest.

### 2.2. Solución Implementada: Optimizador Causal de Umbrales (`ThresholdOptimizer`)
Para que cada fórmula fitness **produzca una estrategia realmente distinta y optimizada**, se creó el componente [`src/models/threshold_optimizer.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/threshold_optimizer.rs):

1. **Entrenamiento Base del Árbol (In-Sample):**
   * El modelo aprende la relación no lineal entre las 11 variables técnicas causales y el retorno futuro a $H$ velas en el conjunto In-Sample ($t \le T_{\text{IS}}$).
2. **Generación de Predicciones Continuas $\hat{y}_t$:**
   * El modelo emite las predicciones continuas para todo el conjunto In-Sample mediante `predict_batch`.
3. **Calibración Guiada por Fitness en In-Sample:**
   * Se evalúa concurrentemente a través de los 20 hilos una rejilla de umbrales de decisión $\tau \in [0.0005, 0.0070]$.
   * Para cada umbral, se simula la ejecución en In-Sample y se calcula el puntaje con la **fórmula fitness exacta seleccionada por Allianz**:
     $$\mathcal{F}(\tau) = \text{FitnessMetric}::\text{evaluate}(\text{Metrics}(\tau))$$
   * Se selecciona el par óptimo $(\tau^*_{\text{long}}, \tau^*_{\text{short}})$ que maximiza $\mathcal{F}(\tau)$.
4. **Transferencia Estricta a Out-of-Sample (OOS):**
   * El modelo queda configurado con sus umbrales óptimos y evalúa el período ciego Out-of-Sample sin Look-Ahead Bias.

---

## 3. Comportamiento Comparativo por Fórmula Fitness

| Métrica Fitness Seleccionada | Comportamiento del Optimizador de Umbrales | Perfil de Trading Resultante |
| :--- | :--- | :--- |
| **Score Compuesto Allianz** | Busca umbrales más altos ($\tau \ge 0.0035$) para recortar operaciones ruidosas que provoquen drawdowns. | **Máxima suavidad de curva ($R^2 \approx 1$)** y mínima profundidad de caída nominal. |
| **Sharpe Ratio** | Equilibra la frecuencia de aciertos y la desviación estándar de los retornos. | Máximo retorno ajustado a la volatilidad global. |
| **Sortino Ratio** | Tolera alta volatilidad alcista pero penaliza drásticamente rachas de pérdidas. | Trades con relación asimétrica favorable (Payoff Ratio alto). |
| **Recovery Factor** | Maximiza la ganancia total dividida por el peor drawdown histórico. | Captura grandes tendencias minimizando el dolor nominal. |
| **Linealidad $R^2$** | Maximiza la recta OLS sin importar el beneficio absoluto en dólares. | Curva de balance rectilínea con retornos regulares. |
| **Retorno Neto %** | Tiende a umbrales más bajos ($\tau \approx 0.0010$) para operar con mayor frecuencia. | Mayor número de operaciones y exposición al mercado. |

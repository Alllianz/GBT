# Diagnóstico de Sobreajuste (Overfitting) en Modelos de Árboles y Soluciones Cuantitativas

**Destinatario:** Allianz  
**Autor:** Antigravity  
**Fecha:** Noviembre 2026  
**Proyecto:** GBT (Gradient Boosted Trader)  

---

## 1. La Pregunta Fundamental: ¿Están Overfitteados los Árboles?

**Respuesta directa y transparente:** **SÍ, categóricamente.**  
Los resultados empíricos de los dashboards demuestran que los modelos basados en Boosting puro (`Native GBDT`, `TreeBoost`) han sufrido un **sobreajuste severo**.

Es vital aclarar un concepto financiero clave:  
> **El split IS / OOS no causa el overfitting; el split IS / OOS es el auditor implacable que lo saca a la luz.**

Si solo hubiésemos mirado el In-Sample (IS), habríamos creído falsamente que teníamos una "máquina de hacer dinero infinito" (+81,000% y +931,000%). El período Out-of-Sample (OOS) demostró que el modelo memorizó el pasado en lugar de aprender patrones predictivos generalizables.

---

## 2. Evidencia Empírica de los Dashboards

| Modelo | Timeframe | In-Sample (IS) | Out-of-Sample (OOS) | Ratio de Robustez (OOS / IS) | Diagnóstico Clínico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Native GBDT** | 1D | **+931,731%** | **-35.65%** | **-0.00003x** | **Overfitting Extremo (Memorización pura)** |
| **TreeBoost** | 1D | **+81,959%** | **-17.36%** | **-0.0002x** | **Overfitting Severo** |
| **TreeBoost** | 4H | **+1,020%** | **-44.99%** | **-0.04x** | **Overfitting por sobreoperación** |
| **Decision Tree CART** | 1D | **+778%** | **-25.36%** | **-0.03x** | **Pérdida de generalización** |
| **Random Forest** | 1D | **+1,527%** | **-3.49%** | **-0.002x** | **Resiliente (Sharpe OOS positivo +0.14)** |
| **Random Forest** | 4H | **+3,485%** | **-5.43%** | **-0.001x** | **Resiliente (Profit Factor OOS 1.01)** |
| **Baseline (EMA 9/21)**| 1D | **+605%** | **+19.81%** | **+0.03x** | **Robusto (Alpha positivo y consistente)** |

---

## 3. ¿Por Qué los Modelos de Boosting Sobreajustan en Series Temporales Cripto?

### 3.1. Razón 1: No Estacionariedad y Cambio de Régimen Macro
- **In-Sample (2017 a ~2023)**: Mercado dominado por ciclos de halving minoristas, volatilidad desbordada y tendencias parabólicas continuas.
- **Out-of-Sample (2023 a 2026)**: Maduración institucional con ETFs spot de Bitcoin, creación/redención de arbitraje institucional y compresión de volatilidad.
- Un árbol de decisión busca umbrales numéricos rígidos ($Feature_k \le \theta$). Cuando la distribución de las variables se desplaza (*concept drift* o *covariate shift*), las reglas fijadas en 2018 pierden validez en 2025.

### 3.2. Razón 2: La Naturaleza Recursiva de Corrección de Residuales en Boosting
- El Gradient Boosting entrena cada nuevo árbol para corregir **exclusivamente los errores del árbol anterior**:
  $$r_{im} = y_i - F_{m-1}(x_i)$$
- En finanzas, gran parte de los retornos a corto plazo son **ruido estocástico puro** (ruido blanco).
- Al iterar 100 o 150 rondas, el algoritmo inevitablemente termina ajustando árboles enteros para "predecir" mechas erráticas y ruido de liquidaciones que jamás se volverán a repetir de forma idéntica en el futuro.

### 3.3. Razón 3: Por qué Random Forest resistió mucho mejor
- A diferencia del Boosting, **Random Forest no corrige residuales en serie**, sino que entrena árboles independientes en paralelo mediante muestreo aleatorio (*Bagging* y *Feature Subsampling*).
- Al promediar decenas de árboles independientes, el ruido aleatorio de cada árbol tiende a cancelarse mutuamente ($\text{E}[error] \to 0$), preservando solo la señal persistente. Por eso Random Forest apenas cedió un -3.49% en 1D y -5.43% en 4H.

---

## 4. Soluciones Cuantitativas para Erradicar el Overfitting en GBT

Para convertir estos modelos en sistemas con Alpha positivo real fuera de muestra, se deben implementar las siguientes 4 reformas institucionales:

### 4.1. Hiperparámetros de "Alta Regularización" (Árboles Poco Profundos)
- Reducir drásticamente la capacidad de memorización del boosting:
  - `max_depth`: Reducir de 4-5 a **2 o 3** (*shallow trees* o *stumps*). Un árbol de profundidad 2 solo puede hacer 3 cortes lógicos, forzándolo a aprender patrones macro.
  - `num_trees`: Limitar a **30 - 50 árboles** (en lugar de 150).
  - `learning_rate`: Bajar de 0.05 a **0.01 - 0.02** (*shrinkage* ultra conservador).
  - `min_samples_leaf`: Subir de 5 a **30 o 50 velas**, impidiendo que un nodo terminal memorice anomalías estadísticas.

### 4.2. Walk-Forward Reentrenable (Rolling Walk-Forward Causal)
- En lugar de entrenar una sola vez sobre 5 años y evaluar a ciegas sobre 2 años, el modelo debe implementar una **Ventana Rodante Causal**:
  - Entrena en los últimos 12 meses $\to$ Predice los próximos 3 meses.
  - Rueda 3 meses hacia adelante $\to$ Reentrena en los últimos 12 meses $\to$ Predice los siguientes 3 meses.
  - Así el modelo se adapta continuamente a la volatilidad y régimen contemporáneos de Bitcoin.

### 4.3. Filtro de Régimen de Volatilidad (Filtro ADX / ATR)
- No forzar al modelo a predecir en todo momento:
  - Solo emitir señales si la tendencia está confirmada ($ADX > 22$) o si la volatilidad normalizada está en rango operativo.
  - En mercados laterales picados, permanecer en `Flat` (efectivo), reduciendo el desgaste de comisiones.

### 4.4. Estandarización Z-Score Causal de Features
- Normalizar las variables técnicas no en función de su valor nominal, sino de su Z-Score rodante respecto a las últimas 100 barras:
  $$Z_t = \frac{X_t - \mu_{t-100..t}}{\sigma_{t-100..t}}$$
  Esto hace que las variables sean estacionarias a lo largo de los años, independientemente de si Bitcoin cotiza a \$10,000 o a \$90,000.

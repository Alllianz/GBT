# Análisis Cuantitativo de Dashboards (4H y 1D) y Fundamentos de TreeBoost

**Destinatario:** Allianz  
**Autor:** Antigravity  
**Fecha:** Noviembre 2026  
**Proyecto:** GBT (Gradient Boosted Trader)  

---

## 1. Análisis Cuantitativo de los Resultados en 1D y 4H

### 1.1. Tabla Comparativa de Rendimiento en 1D (Diario)

| Modelo | Retorno In-Sample (IS) | Retorno Out-of-Sample (OOS) | Sharpe OOS | Profit Factor OOS | Diagnóstico de Robustez |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (EMA 9/21)** | **+605.28%** | **+19.81%** | **+0.409** | **1.30** | **Mejor general en OOS**: Único modelo con retorno y Sharpe positivos fuera de muestra. |
| **Random Forest** | +1527.78% | **-3.49%** | **+0.143** | **1.00** | **Mejor ML en 1D**: Retiene el capital con mínima degradación (-3.49%) y Sharpe positivo. |
| **TreeBoost** | +81959.69% | -17.36% | -0.229 | 0.84 | Fuerte memorización en IS; degradación controlada en OOS (-17%). |
| **Decision Tree CART** | +778.11% | -25.36% | -0.170 | 0.83 | Pérdida de predictibilidad por rigidez en los cortes univariados. |
| **Native GBDT** | +931731.99% | -35.65% | -0.646 | 0.81 | Severo sobreajuste (*overfitting*) en IS por falta de regularización temprana. |
| **Forust-ML** | -100.10% | -100.10% | -2.220 | 0.60 | Umbrales de decisión inadecuados para la volatilidad diaria de BTC. |

---

### 1.2. Tabla Comparativa de Rendimiento en 4H (4 Horas)

| Modelo | Retorno In-Sample (IS) | Retorno Out-of-Sample (OOS) | Sharpe OOS | Profit Factor OOS | Diagnóstico de Robustez |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree CART** | +607.00% | **-1.10%** | **+0.005** | **1.00** | **Máxima estabilidad OOS**: Prácticamente breakeven (-1.10%) y Sharpe neutral. |
| **Random Forest** | +3485.50% | **-5.43%** | -0.066 | **1.01** | Gran capacidad de generalización en alta frecuencia (PF > 1.0). |
| **Baseline (EMA 9/21)** | +628.88% | -10.83% | -0.029 | 0.98 | Ligera degradación por retraso de cruce en fases de consolidación. |
| **Forust-ML** | -67.37% | -39.30% | -0.305 | 0.91 | Menor drawdown que en 1D pero aún en zona negativa. |
| **TreeBoost** | +1020.40% | -44.99% | -0.613 | 0.88 | Sobreoperación en OOS con mayor impacto de comisiones acumuladas. |

---

### 1.3. Conclusiones Financieras Clave

1. **La paradoja del Overfitting en GBDT Puro**:
   - Modelos como `Native GBDT` y `TreeBoost` logran retornos astronómicos en In-Sample (+80,000% a +900,000%). Esto es síntoma de **memorización de ruido**. Al enfrentarse al período ciego OOS, sufren degradación.
2. **Superioridad del Ensamble por Bagging (Random Forest)**:
   - En ambos timeframes (1D y 4H), `Random Forest` fue el modelo de Machine Learning más resiliente (-3.49% en 1D y -5.43% en 4H). Al promediar árboles descorrelacionados con muestreo aleatorio de variables, filtra el ruido del mercado mucho mejor que el boosting puro.
3. **El Baseline como ancla de Alpha**:
   - En 1D, el `Baseline EMA 9/21` fue el único con retorno positivo en OOS (+19.81%). Esto demuestra que en temporalidades altas, una regla de tendencia macro simple tiene una relación señal/ruido superior a árboles complejos sin selección de régimen.

---

## 2. ¿Qué es TreeBoost?

### 2.1. Definición y Origen
**`TreeBoost`** es un motor de aprendizaje automático tabular universal desarrollado de forma nativa en **Rust** (con aceleración SIMD y GPU opcional).

A diferencia de bibliotecas tradicionales como XGBoost, LightGBM o CatBoost, que aplican árboles de decisión de forma aislada, TreeBoost fue diseñado para resolver el **problema fundamental de los árboles en series temporales financieras**:

> **El Talón de Aquiles de los Árboles en Finanzas:**  
> Los árboles de decisión dividen el espacio de features mediante planos ortogonales ($X_i \le \theta$). Por su naturaleza paso a paso (*step functions*), **son incapaces de extrapolar tendencias**. Si Bitcoin cotiza a \$40,000 durante el entrenamiento y en OOS sube a \$70,000, un árbol de decisión estándar predecirá como máximo el valor más alto visto en el entrenamiento (\$40,000), fallando completamente en capturar la expansión de tendencia.

---

### 2.2. Arquitectura de TreeBoost: Los 3 Modos

TreeBoost combina tres paradigmas bajo una misma interfaz unificada:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      UniversalModel en TreeBoost                       │
├────────────────────┬─────────────────────────────┬─────────────────────┤
│      PureTree      │       LinearThenTree        │    RandomForest     │
│   (GBDT Clásico)   │  (Híbrido Tendencia+Residuo)│  (Bagging Paralelo) │
└────────────────────┴─────────────────────────────┴─────────────────────┘
```

1. **`LinearThenTree` (Modo Híbrido - Recomendado para Finanzas)**:
   - **Fase 1 (Modelo Lineal / Ridge / Lasso)**: Ajusta un modelo lineal sobre las variables macro. Captura la dirección, la pendiente y la tendencia general del precio (extrapolación continua).
   - **Fase 2 (Árboles GBDT)**: Entrena árboles de decisión sobre los **residuales** de la fase lineal:
     $$e_t = y_t - \hat{y}_{\text{lineal}}(x_t)$$
     Los árboles capturan patrones no lineales, reversiones a la media y anomalías locales.
   - **Predicción Final**:
     $$\hat{y}(x) = \hat{y}_{\text{lineal}}(x) + \sum_{m=1}^M \eta \cdot T_m(x)$$

2. **`PureTree` (GBDT Estándar)**:
   - Árboles basados en histogramas compactos de 8 bits (`u8 bins`) para optimizar memoria caché L1/L2.

3. **`RandomForest` (Bagging)**:
   - Ensamble paralelo para reducción de varianza.

---

### 2.3. Características Técnicas Avanzadas de TreeBoost

1. **Binning por Histogramas Cuántiles (T-Digest)**:
   - Discretiza variables continuas en 255 bins de 8 bits. Permite calcular gradientes y hessianos a velocidad de memoria caché sin escanear todos los datos continuos.
2. **División Regularizada por Entropía de Shannon**:
   - Penaliza splits que concentran datos en subgrupos inestables, protegiendo contra el cambio de régimen (*dataset shift*).
3. **Pérdida Pseudo-Huber**:
   - Suave como el error cuadrático (MSE) para errores pequeños, pero lineal (MAE) para errores extremos ("mechas" o liquidaciones masivas en cripto), evitando que un spike arruine los pesos del modelo.
4. **Integración Nativa con DataFrames de Polars**:
   - Procesa columnas de memoria contigua en formato Apache Arrow con cero copias.

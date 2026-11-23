# Métricas Matemáticas que Optimizan Random Forest y TreeBoost

**Destinatario:** Allianz  
**Autor:** Antigravity  
**Proyecto:** GBT (Gradient Boosted Trader)  
**Fecha:** Noviembre 2026  

---

## 1. Random Forest: Optimización de Varianza y MSE

En la arquitectura de GBT ([`forest_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/random_forest/forest_model.rs) y [`split_finder.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/decision_tree/split_finder.rs)), **Random Forest opera en dos niveles de optimización simultáneos**:

### 1.1. Nivel Local de Nodo: Reducción de Varianza (MSE)
En cada división (*split*) de cada árbol individual, el algoritmo evalúa un subconjunto aleatorio de variables ($m_{try} \approx \sqrt{F}$) y busca el umbral $\theta$ que maximiza la **reducción de impureza de varianza**:

$$\Delta I = \text{Var}(S) - \left( \frac{|S_L|}{|S|}\text{Var}(S_L) + \frac{|S_R|}{|S|}\text{Var}(S_R) \right)$$

donde la varianza de un conjunto de velas $S$ es:
$$\text{Var}(S) = \frac{1}{|S|}\sum_{i \in S} (y_i - \bar{y}_S)^2$$

- **Objetivo**: Separar las velas históricas en dos grupos donde los retornos futuros sean lo más homogéneos posible (agrupar retornos positivos altos a un lado y retornos negativos o neutros al otro).

### 1.2. Nivel Global de Ensamble: Minimización de la Varianza del Estimador
A diferencia de los modelos de Boosting, Random Forest **no optimiza una función de coste mediante gradientes secuenciales**. Su optimización es estadística mediante el teorema de reducción de varianza de Chebyshev / Bagging:

$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$

- $B$: Número de árboles en el bosque (ej: 50 árboles).
- $\sigma^2$: Varianza de un árbol individual.
- $\rho$: Correlación entre las predicciones de distintos árboles.

Al introducir **Feature Subsampling** (elegir solo un subconjunto de variables en cada split) y **Bootstrap Sampling** (muestreo con reemplazo), Random Forest reduce deliberadamente la correlación $\rho \to 0$. A medida que $\rho$ disminuye, la varianza del error total se reduce drásticamente sin aumentar el sesgo (*bias*), lo que explica por qué fue el modelo con menor degradación en OOS.

---

## 2. TreeBoost: Optimización Híbrida ElasticNet + Pseudo-Huber + Entropía

En [`treeboost_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_model.rs), **TreeBoost opera bajo el modo híbrido `LinearThenTree`**, optimizando métricas distintas en cada una de sus fases:

### 2.1. Fase 1: Optimización de Tendencia Macro (ElasticNet / Coordinate Descent)
La base lineal de TreeBoost optimiza la función de pérdida regularizada con penalización combinada $L_1$ (Lasso) y $L_2$ (Ridge):

$$\min_{w, b} \left[ \frac{1}{2N}\sum_{i=1}^N \Big(y_i - (w^T x_i + b)\Big)^2 + \lambda_1 \|w\|_1 + \frac{\lambda_2}{2} \|w\|_2^2 \right]$$

- **Qué busca**: Encontrar una pendiente macro continua que capture la inercia general del precio y permita extrapolar más allá del rango de precios visto en el entrenamiento.

### 2.2. Fase 2: Optimización de Residuales con Pérdida Pseudo-Huber
Sobre los residuales $e_i = y_i - \hat{y}_{\text{lineal}}(x_i)$, TreeBoost entrena árboles GBDT optimizando la **función de pérdida Pseudo-Huber**:

$$L_{\delta}(y, \hat{y}) = \delta^2 \left( \sqrt{1 + \left(\frac{y - \hat{y}}{\delta}\right)^2} - 1 \right)$$

Esta función es clave para finanzas cuantitativas:
1. **Para errores pequeños ($|e| \ll \delta$)**: Se comporta como **Error Cuadrático Medio (MSE)**:
   $$L_{\delta}(y, \hat{y}) \approx \frac{1}{2}(y - \hat{y})^2$$
   Optimiza la media condicional de los retornos.
2. **Para errores extremos o eventos de cola ($|e| \gg \delta$)**: Se comporta como **Error Absoluto Medio (MAE)**:
   $$L_{\delta}(y, \hat{y}) \approx \delta |y - \hat{y}|$$
   Los gradientes se saturan a una constante fija $\pm \delta$, volviendo al modelo **inmune a mechas anormales de liquidación o anomalías atípicas** del mercado.

### 2.3. Criterio de Split con Regularización por Entropía de Shannon
Al buscar el mejor split en cada nodo, TreeBoost no solo evalúa el gradiente y hessiano ($G^2 / (H + \lambda)$), sino que resta una penalización basada en la **entropía de Shannon**:

$$\text{Gain}_{\text{TreeBoost}} = \text{Gain}_{\text{GBDT}} - \gamma \cdot H(\text{distribución de split})$$

Esto castiga severamente divisiones que crean ramas con muy pocas muestras desbalanceadas, evitando que el árbol memorice velas individuales aisladas.

---

## 3. Resumen Comparativo de Métricas

| Algoritmo | Métrica de Ajuste Principal | Comportamiento ante Outliers / Mechas | Objetivo Financiero |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **Reducción de Varianza (MSE)** sobre el promedio de árboles Bootstrap | Muy Robusto (el promedio cancela ruido extremo) | Minimizar la varianza del error y evitar el sobreajuste. |
| **TreeBoost** | **Pseudo-Huber Loss** + **ElasticNet** en residuales + **Entropía** | Robusto a picos de volatilidad (gradientes acotados) | Extrapolar tendencia macro y modelar no linealidades locales. |

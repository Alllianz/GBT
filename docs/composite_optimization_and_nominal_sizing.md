# Análisis Cuantitativo: Métrica Compuesta de Optimización, Capital Nominal y Margen Cruzado

**Autor:** Antigravity  
**Destinatario:** Allianz  
**Fecha:** 2026-11-08  
**Sistema:** GBT Engine (Rust)  

---

## 1. Planteamiento de la Función Objetivo Compuesta

Se evalúa la adopción de una función objetivo híbrida definida como:

$$\mathcal{S}_{\text{compuesto}} = 0.5 \cdot \text{Linealidad } (R^2) + 0.5 \cdot \left( \frac{\text{Capital Total}}{\text{Máximo Drawdown Nominal}} \right)$$

Donde:
1. **$\text{Linealidad } (R^2) \in [0, 1]$:** Coeficiente de determinación de la regresión de mínimos cuadrados ordinarios (OLS) sobre la trayectoria de balance $W_t$ en función del índice temporal $t$:
   $$R^2 = 1 - \frac{\sum_{t=1}^T (W_t - \hat{W}_t)^2}{\sum_{t=1}^T (W_t - \bar{W})^2}$$
   Mide la suavidad de la curva de balance. Una trayectoria que crece como una línea recta perfecta tiene $R^2 = 1.0$.

2. **Ratio de Capital a Drawdown Nominal:**
   $$\mathcal{R}_{\text{Cap/DD}} = \frac{C_{\text{final}}}{\max(1.0, \text{MaxDD}_{\text{nominal}})} = \frac{C_0 + \text{Beneficio Neto}}{\text{MaxDD}_{\text{nominal}}} = \frac{C_0}{\text{MaxDD}_{\text{nominal}}} + \text{Recovery Factor}$$
   Mide la asimetría entre el volumen de capital preservado/generado y el peor dolor nominal experimentado desde cualquier cresta histórica.

---

## 2. ¿Qué sucede con los Árboles (RF, TreeBoost, GBDT, CART)?

### 2.1. Desconexión Fundamental: Pérdidas Aditivas Locales vs Métricas de Trayectoria Global
En algoritmos basados en árboles de decisión (CART, Random Forest, GBDT, TreeBoost, Forust):
- La optimización intra-árbol durante la búsqueda de divisiones (`split_finder`) opera bajo **funciones de pérdida aditivas muestra por muestra**:
  $$\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = \sum_{i=1}^N \ell(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$
  Tanto el Mean Squared Error (MSE), como la Entropía Cruzada (LogLoss) y la Pseudo-Huber Loss se calculan fila por fila de forma independiente. Cada muestra aporta un gradiente local $g_i = \partial \ell / \partial \hat{y}_i$ y un hessiano $h_i = \partial^2 \ell / \partial \hat{y}_i^2$.

- En contraste, la métrica propuesta $\mathcal{S}_{\text{compuesto}}$ es una **métrica de serie temporal patológica y no aditiva**:
  - **Sensibilidad al Orden Temporal:** Si se permutan aleatoriamente las muestras de entrenamiento, el MSE o LogLoss no cambia en absoluto. Sin embargo, el Max Drawdown y el $R^2$ de la curva de equity cambian drásticamente.
  - **No Diferenciabilidad:** El operador de Max Drawdown ($\max_t (\max_{s \le t} W_s - W_t)$) posee subgradientes nulos casi en todas partes y discontinuidades agudas, impidiendo su uso directo como gradiente analítico dentro de la división de nodos de un árbol.

### 2.2. Impacto en la Optimización de Hiperparámetros, Umbrales y Poda
Cuando esta métrica se utiliza como criterio de optimización en **selección de modelos, poda, ajuste de hiperparámetros (learning rate, profundidad, L2) y búsqueda de umbral óptimo de señal ($\tau$)**, los efectos sobre los árboles son profundos:

1. **Eliminación del Sesgo de "Racha Afortunada" (Lucky-Run Overfitting):**
   - Modelos como el GBDT Nativo o TreeBoost a menudo maximizan el beneficio acumulado en In-Sample capturando movimientos erráticos de alta volatilidad donde 2 operaciones atípicas multiplican el capital por $10\times$, a pesar de sufrir drawdowns del $40\%$.
   - Con $\mathcal{S}_{\text{compuesto}}$, una curva con desplomes intermedios ve colapsar su $R^2$ a valores de $0.30 - 0.50$ y su ratio de MaxDD explota a la baja. El optimizador rechaza categóricamente estos árboles sobreajustados.

2. **Incentivo a la Regularización y Menor Profundidad:**
   - Para maximizar $R^2$, el modelo necesita que casi todas las semanas generen rendimientos positivos estables. Esto solo se logra reduciendo la varianza del modelo (árboles de profundidad moderada $3-5$, mayor número de árboles en Random Forest para suavizar la frontera de decisión, o fuerte subsampling de características).

3. **Calibración Conservadora de Umbrales:**
   - La métrica obliga al modelo a no disparar órdenes en condiciones dudosas. Al penalizar severamente el Drawdown en el denominador, el umbral de disparo de probabilidad (ej. $\tau \ge 0.65$ en vez de $0.50$) se desplaza hacia regímenes de alta convicción estadística.

---

## 3. Dinámica Financiera: Capital Nominal vs Capital Compuesto

Allianz especificó la siguiente configuración para la optimización y simulación:
- **Capital Inicial:** $\$10,000$ USD.
- **Position Sizing:** $10\%$ Nominal Fijo.
- **Margen Comprometido:** $\$1,000$ USD fijos por operación.
- **Apalancamiento:** $10\times$.
- **Volumen Nocional:** $\$1,000 \times 10 = \$10,000$ USD constantes.

### 3.1. Diferencias Críticas entre Sizing Nominal y Compuesto

| Característica | Sizing Compuesto (% del Equity Dinámico) | Sizing Nominal Fijo (% del Capital Inicial) |
| :--- | :--- | :--- |
| **Volumen de Exposición** | Crece exponencialmente con las ganancias ($V_t = W_t \cdot f \cdot L$). | Constante e inmutable ($V_t = C_0 \cdot f \cdot L = \$10,000$). |
| **Efecto de Rachas Ganadoras** | Infla la posición en los techos del mercado; la primera pérdida destruye una suma nominal masiva. | Mantiene el riesgo en dólares idéntico; las ganancias quedan aseguradas en el balance. |
| **Apalancamiento Efectivo con Crecimiento** | Constante ($10\times$ sobre el equity). | **Decreciente:** Si la cuenta sube a $\$20,000$, la exposición es de $\$10,000$ (apalancamiento efectivo $0.5\times$). |
| **Linealidad de la Curva ($R^2$)** | La curva compuesta es exponencial (curva cuadrática/exponencial en gráfico lineal, distorsionando el $R^2$). | **Intrínsecamente lineal:** cada $1\%$ de variación del activo genera exactamente $\$100$ USD de PnL constante. |
| **Riesgo de Ruina Asintótico** | Constante si el sizing es subóptimo (criterio de Kelly). | **Tiende a 0** a medida que el balance de la cuenta se expande. |

### 3.2. Sinergia Matemática con el Score Compuesto
Al operar con capital nominal constante:
- Cada operación aporta un delta lineal a la cuenta: $\Delta W_t = \pm V_{\text{nominal}} \cdot r_t - \text{Comisión}$.
- El cálculo del $R^2$ OLS mide con absoluta fidelidad si la ventaja probabilística del modelo (edge) es regular a lo largo del tiempo, sin verse distorsionada por el efecto de bola de nieve del interés compuesto.
- El denominador $\text{MaxDD}_{\text{nominal}}$ se vuelve comparable trade a trade en términos absolutos de dólares.

---

## 4. Arquitectura de Operatoria en Margen Cruzado (`Cross Margin`)

### 4.1. Mecánica frente al Margen Aislado (`Isolated`)
1. **Margen Aislado (`Isolated`):**
   - El colateral de la posición se limita estrictamente a los $\$1,000$ de margen inicial.
   - Con apalancamiento $10\times$, el margen de mantenimiento suele ser de $\approx 0.5\%$.
   - Una excursión adversa en el precio de sólo:
     $$\Delta P_{\text{liq}} = \frac{1}{\text{Leverage}} - \text{MaintMargin} = \frac{1}{10} - 0.005 = 9.5\%$$
     provoca la **liquidación forzada inmediata** y la pérdida total de los $\$1,000$, aunque el trader conserve $\$9,000$ de capital líquido en la cuenta.
   - En criptoactivos como BTC/USDT, mechazos transitorios de volatilidad en velas de 4H liquidan posiciones aisladas que luego terminan en la dirección prevista.

2. **Margen Cruzado (`Cross`):**
   - Todo el patrimonio neto de la cuenta ($\text{Equity} = \text{Cash} + \text{PnL no realizado}$) respalda la posición abierta.
   - El precio de liquidación se expande enormemente:
     $$P_{\text{liq}}^{\text{Long}} = P_{\text{entry}} - \frac{\text{Equity}}{\text{Cantidad}} = P_{\text{entry}} \cdot \left(1 - \frac{\text{Equity}}{V_{\text{nocional}}}\right)$$
     Si $\text{Equity} = \$10,000$ y el volumen nocional es de $\$10,000$, ¡el precio tendría que caer un $100\%$ hasta cero para que ocurra una liquidación!
   - La posición tiene espacio holgado para respirar frente al ruido del mercado.

### 4.2. Tratamiento de Pérdidas y Comisiones en el Motor GBT
En el motor institucional implementado en Rust ([`execution_engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs)):
- La comisión de entrada y salida se descuenta directamente de `portfolio.cash_balance`.
- Al cierre de la operación, `pos.isolated_margin` es `None`, por lo que el PnL neto (`gross_pnl - commissions`) se suma o resta directamente del balance general de la cuenta.
- No existen límites arbitrarios de margen: las pérdidas son asumidas íntegramente por el patrimonio de la cuenta, garantizando una simulación idéntica a una cuenta de futuros institucionales.

---

## 5. Implementación en el Código de GBT

Se integraron los siguientes componentes modulares:
1. **[`src/types/position_sizing.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/position_sizing.rs):**
   - Variante `PositionSizing::PercentOfInitialCapital(f64)`.
   - Por defecto: `0.10` ($10\%$ del capital inicial).
2. **[`src/backtest/config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/config.rs):**
   - Configuración predeterminada: `initial_capital: 10_000.0`, `leverage: 10.0`, `margin_mode: MarginMode::Cross`, `position_sizing: PositionSizing::PercentOfInitialCapital(0.10)`.
3. **[`src/backtest/execution_engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs):**
   - Dimensionamiento exacto: `portfolio.initial_capital * pct * leverage`.
   - Liquidación cruzada sobre balance global.
4. **[`src/backtest/composite_score.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/composite_score.rs):**
   - Cálculo del score $0.5 \cdot R^2 + 0.5 \cdot (\text{Capital} / \text{MaxDD})$.
5. **[`src/backtest/report.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/report.rs) y [`src/dashboard/html_template.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/dashboard/html_template.rs):**
   - Visualización del Score Compuesto y Ratio Capital/MaxDD en la terminal y en los dashboards interactivos HTML.

# Guía del Sistema de Selección y Auditoría de Métricas Fitness

**Autor:** Antigravity  
**Destinatario:** Allianz  
**Fecha:** 2026-11-08  
**Sistema:** GBT Engine (Rust)  

---

## 1. Estado Predeterminado Restaurado

De acuerdo con tus indicaciones, se ha restaurado la configuración original del sistema como predeterminada:
* **Capital Inicial:** `$10,000.0` USD.
* **Apalancamiento:** `1.0x` (Spot / Desapalancado).
* **Modo de Margen:** `MarginMode::Isolated` (Aislado).
* **Position Sizing:** `PositionSizing::PercentOfEquity(0.25)` ($25\%$ del equity por operación).
* **Métrica Fitness Predeterminada:** `FitnessMetric::SharpeRatio` (Sharpe Ratio Anualizado).

---

## 2. Catálogo de Métricas Fitness Disponibles

El motor cuantitativo GBT incluye el módulo [`src/types/fitness_metric.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/fitness_metric.rs) con 8 funciones objetivo para evaluar, comparar y rankear cualquier modelo o estrategia:

| ID | Nombre de Métrica | Fórmula Matemática | Propósito Cuantitativo |
| :---: | :--- | :--- | :--- |
| **1** | **Sharpe Ratio** (Estándar) | $\frac{\mathbb{E}[R_t]}{\sigma(R_t)} \times \sqrt{N_{\text{barras}}}$ | Mide el exceso de retorno ajustado por la volatilidad total (alcista y bajista). |
| **2** | **Score Compuesto Allianz** | $0.5 \cdot \text{clamp}(R^2, 0, 1) + 0.5 \cdot \left(\frac{C_{\text{final}}}{\max(1.0, \text{MaxDD}_{\text{nom}})}\right)$ | Penaliza caídas intermedias y premia la suavidad y rectitud de la curva de balance sin desvíos. |
| **3** | **Sortino Ratio** | $\frac{\mathbb{E}[R_t]}{\sigma_{\text{downside}}(R_t)} \times \sqrt{N_{\text{barras}}}$ | Similar a Sharpe, pero penaliza únicamente la volatilidad negativa o pérdidas. |
| **4** | **Recovery Factor** | $\frac{\text{Beneficio Neto}}{\max(1.0, \text{MaxDD}_{\text{nominal}})}$ | Eficiencia de recuperación: cuánto capital gana la estrategia por cada dólar de caída máxima. |
| **5** | **Linealidad $R^2$** | $1 - \frac{\sum (W_t - \hat{W}_t)^2}{\sum (W_t - \bar{W})^2}$ | Grado de rectitud de la trayectoria de balance (OLS); $1.0$ indica ganancia perfectamente constante. |
| **6** | **Calmar Ratio** | $\frac{\text{Beneficio Neto \%}}{\max(0.01, \text{MaxDD \%})}$ | Relación entre retorno porcentual total y máxima caída porcentual. |
| **7** | **Profit Factor** | $\frac{\text{Ganancia Bruta}}{\max(0.01, \text{Pérdida Bruta})}$ | Ratio entre el dinero ganado en trades ganadores y el dinero perdido en trades perdedores. |
| **8** | **Retorno Neto %** | $\frac{C_{\text{final}} - C_0}{C_0} \times 100$ | Retorno acumulado simple sin ajuste de riesgo. |

---

## 3. Cómo Probar y Comparar las Métricas en la Terminal

1. **Configuración Activa:** En el menú principal (`cargo run`), ingresa a la opción `[8] Config`.
2. **Selección Interactiva:** Si decides editar los valores, el sistema te solicitará:
   ```text
   Selecciona Métrica Fitness a Evaluar/Probar:
     [1] Sharpe Ratio Anualizado [Estándar]
     [2] Score Compuesto Allianz [0.5*R² + 0.5*(Cap/MaxDD Nominal)]
     [3] Sortino Ratio (Penaliza solo volatilidad bajista)
     [4] Recovery Factor (Beneficio Neto / Max Drawdown Nominal)
     [5] Linealidad R² de Balance (Suavidad OLS)
     [6] Calmar Ratio (Retorno% / Max Drawdown%)
     [7] Profit Factor (Ganancia Bruta / Pérdida Bruta)
     [8] Retorno Neto %
   ```
3. **Auditoría Comparativa en Tiempo Real:** Cada vez que ejecutas cualquier modelo (Forust, TreeBoost, GBDT Nativo, Random Forest, CART o Baseline), el motor imprime una **tabla comparativa simultánea** con todas las métricas evaluadas sobre el conjunto Out-of-Sample (OOS), señalando con `>>>` la métrica activa seleccionada:
   ```text
   ==========================================================================
               AUDITORÍA COMPARATIVA DE MÉTRICAS FITNESS                    
   ==========================================================================
   ACT  MÉTRICA FITNESS                  PUNTAJE        FÓRMULA / CRITERIO
   --------------------------------------------------------------------------
   >>>  Sharpe Ratio Anual               1.8420         (Mean_Ret / Std_Ret) * sqrt(annual_bars)
        Score Compuesto Allianz          6.4210         0.5 * clamp(R², 0, 1) + 0.5 * (Cap_final / max(1.0, MaxDD_nom))
        Sortino Ratio Anual              2.9150         (Mean_Ret / Downside_Std) * sqrt(annual_bars)
        Recovery Factor                  4.8500         Net_Profit / max(1.0, MaxDD_nom)
        Linealidad R²                    0.8920         1 - sum(residuals²) / sum((equity - mean)²)
        Calmar Ratio                     3.1200         Net_Profit_pct / max(0.01, MaxDD_pct)
        Profit Factor                    1.6500         Gross_Profit / max(0.01, Gross_Loss)
        Retorno Neto %                   38.5000        ((Cap_final - Cap_init) / Cap_init) * 100
   ==========================================================================
   ```

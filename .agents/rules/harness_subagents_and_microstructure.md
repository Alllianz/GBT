---
trigger: always_on
description: Directrices del Antigravity Harness para selección de modelo de subagentes (Gemini 3.8 Flash) y política de microestructura sin slippage.
---

# DIRECTRICES DEL ANTIGRAVITY HARNESS - PROYECTO GBT

1. **Modelo Exclusivo para Subagentes (`invoke_subagent`):**
   - Todo subagente despachado por el Orquestador Principal debe ejecutarse SIEMPRE bajo **Gemini 3.8 Flash** (`Model="flash"` o `Model="flash_lite"`).
   - El nivel de esfuerzo/capacidad (*High*, *Medium* o *Low*) queda a criterio estratégico del Orquestador según la complejidad algorítmica o arquitectónica de la tarea asignada.

2. **Política de Ejecución sin Slippage:**
   - En el motor de microestructura y backtesting de GBT, queda excluido el modelado o aplicación de *slippage* (deslizamiento de precios), salvo orden explícita en contrario de Allianz.
   - Los costos operativos se centran exclusivamente en la estructura real de comisiones (Maker/Taker, niveles VIP, descuentos BNB) y tasas de financiación (*Funding Rates* cada 8h en futuros perpetuos USD-M).

3. **Evaluación OOS sobre Todos los Candidatos (Comparación IS/OOS):**
   - En `gbt_evolve` y búsquedas de grilla/candidatos, la evaluación *Out-Of-Sample* (OOS) debe ejecutarse SIEMPRE sobre **todos los candidatos** (no solo sobre el ganador de IS/OOF), para poder comparar consistentemente el desempeño IS vs OOS de cada configuración.

4. **Fórmula Fitness (Linealidad + Doble Régimen Tendencia & Reversión a la Media):**
   - La función objetivo / Fitness (`CompositeAllianz` e `inst_score` en `gbt_evolve`) debe incluir explícitamente la **linealidad de la curva de equity** ($R^2$) y priorizar estrategias rentables **tanto en tendencia (Primary GBT) como en reversión a la media (MR Hedge)**, penalizando estrategias unidimensionales que dependan de un solo régimen.

5. **Estética de Dashboards (Negro Puro + Azul Francia Vibrante):**
   - Todos los dashboards HTML generados deben mantener fondo **negro puro** (`#000000` / tarjetas `#050505`) con detalles y acentos principales en **azul Francia vibrante** (`#0055FF` / `#1D63FF`).



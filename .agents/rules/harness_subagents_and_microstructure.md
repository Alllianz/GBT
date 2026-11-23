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

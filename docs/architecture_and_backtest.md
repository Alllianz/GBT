# Arquitectura Institucional de GBT (Gradient Boosted Trader)

**Autor:** Antigravity  
**Destinatario:** Allianz  
**Fecha:** Noviembre 2026  

---

## 1. Módulos Implementados

### 1.1. Tipos de Dominio (`src/types/`)
- [`timeframe.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/timeframe.rs): Enumeración estricta de temporalidades (`1m` hasta `1w`) con conversiones seguras y duraciones en milisegundos/segundos.
- [`kline.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/kline.rs): Estructura inmutable de velas financieras con validación geométrica (`high >= low`, precios estrictamente positivos).
- [`signal.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/signal.rs): Señales de trading (`Long`, `Short`, `Flat`) emitidas al cierre de barra $t$.
- [`order.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/order.rs): Tipos de órdenes y resultados de ejecución.
- [`position.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/position.rs): Estado de posición viva con tracking intra-vela de precios extremos ($MFE / MAE$).
- [`trade.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/trade.rs): Registro completo de cada operación finalizada con PnL neto, comisiones, retorno %, pips y duración en barras.

### 1.2. Almacenamiento Persistente SQLite (`src/storage/`)
- [`db_config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_config.rs): Configuración de conexiones, WAL mode y timeout.
- [`db_connection.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_connection.rs): Inicialización del schema e índices `(timeframe, timestamp)`.
- [`kline_repository.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/kline_repository.rs): Inserción masiva en transacciones ACID con `INSERT OR REPLACE`, lectura por rangos y consulta del último timestamp disponible.

### 1.3. Downloader de Velas Resiliente (`src/downloader/`)
- Basado en la arquitectura probada de **Astro Evo**, adaptada a tipado ultra modular.
- [`endpoint_pool.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/endpoint_pool.rs): Pool rotativo con 5 endpoints de Binance para eludir bloqueos o caídas de nodo.
- [`raw_binance_kline.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/raw_binance_kline.rs): Deserializador del array crudo de la API REST.
- [`binance_client.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/binance_client.rs): Descarga paginada en bloques de hasta 1000 velas con filtro temporal estricto anti Look-Ahead ($close\_time \le now\_ms$).
- [`downloader_service.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/downloader_service.rs): Orquestador de sincronización que detecta el último punto temporal en la base de datos y continúa la descarga hasta el presente.

### 1.4. Motor de Backtest Institucional Anti Look-Ahead (`src/backtest/`)
- [`config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/config.rs): Parámetros de prueba (capital, apalancamiento, comisiones, slippage, stop loss/take profit fijos o dinámicos).
- [`equity_point.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/equity_point.rs): Registro granular vela a vela del balance, equity y drawdown.
- [`portfolio.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/portfolio.rs): Balance de cuenta, margen utilizado y control de quiebra.
- [`strategy.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/strategy.rs): Trait causal que garantiza que cualquier estrategia o modelo solo consuma datos hasta $t \le T$.
- [`execution_engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs):
  1. Ejecución estricta a $t+1$: Señal generada en $t$ se procesa en el `Open` de $t+1$ con slippage y comisiones aplicadas.
  2. Detección intra-vela de SL/TP con los extremos `High` y `Low`, aplicando criterio financiero conservador ante toques ambiguos.
- [`metrics.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics.rs) y [`metrics_calculator.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics_calculator.rs):
  - Retornos brutos y netos.
  - Win rate y Profit factor.
  - Payoff ratio (Avg Win / Avg Loss).
  - Max Drawdown nominal y porcentual.
  - Sharpe y Sortino ratio anualizados.
  - Consistencia y linealidad de la curva de balance ($R^2$).
- [`engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/engine.rs): Bucle de simulación temporal causal.
- [`report.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/report.rs): Renderizado de reporte financiero en consola y markdown.

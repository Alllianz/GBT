# Guía de Uso de la Terminal Interactiva de GBT

**Destinatario:** Allianz  
**Autor:** Antigravity  
**Proyecto:** GBT (Gradient Boosted Trader)  
**Fecha:** Noviembre 2026  

---

## 1. Modos de Ejecución

Para iniciar la terminal interactiva cuantitativa en cualquier momento, basta con ejecutar:
```bash
cargo run
```

---

## 2. Menú Principal de Navegación

```text
===============================================================
       🚀 GBT (Gradient Boosted Trader) - TERMINAL CUANTITATIVA 
===============================================================
  [1] DB         - Gestión y Sincronización de Velas SQLite
  [2] Forust-ML  - Gradient Boosted Decision Tree (Causal Engine)
  [3] TreeBoost  - Universal Tabular & LinearThenTree Boosting
  [4] Baseline   - Estrategia Heurística de Referencia (EMA Cross)
  [5] Config     - Configuración Financiera (Capital, Apalancamiento)
  [0] Salir
===============================================================
```

---

## 3. Submenús y Capacidades

### 3.1. [1] DB (Gestión de Velas SQLite)
- **Ver inventario:** Muestra la cantidad total de velas almacenadas en `candles.db` por cada timeframe (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, etc.) y la última fecha UTC registrada.
- **Descargar/Sincronizar timeframe:** Conecta con el pool resiliente de Binance (5 endpoints fallback) y descarga de manera secuencial hasta el presente, filtrando velas no cerradas para prevenir Look-Ahead Bias.
- **Sincronización masiva:** Actualiza los timeframes institucionales estándar (`15m`, `1h`, `4h`, `1d`).

### 3.2. [2] Forust-ML (GBDT Causal Puro en Rust)
- **Walk-Forward Causal Backtest:** Divide el dataset cronológicamente en In-Sample (Train) y Out-of-Sample (Test) con purga de barras intermedia para prevenir data leakage.
- **Backtest Completo:** Realiza la simulación causal a lo largo de todo el horizonte histórico disponible.
- **Ajuste de Hiperparámetros:** Permite configurar número de árboles, profundidad máxima, learning rate, regularizaciones $L_1/L_2$, horizonte temporal y umbrales de decisión para Long y Short.

### 3.3. [3] TreeBoost (Universal Tabular & LinearThenTree Boosting)
- **AutoML & Boosting Híbrido:** Diseñado específicamente para series temporales con tendencia y extrapolación sin sobreajuste.
- **Inferencia Causal:** Predicción en tiempo real sin fugas futuras ($t \le T$).

### 3.4. [4] Baseline (Estrategia Heurística EMA Cross)
- Estrategia de cruce de medias móviles exponenciales causales para comparar el alpha real aportado por los modelos de árboles frente a una heurística tradicional.

### 3.5. [5] Config (Parámetros Financieros)
- Configuración en tiempo de ejecución de: Capital inicial, apalancamiento, comisiones maker/taker, slippage por operación, fracción de equity arriesgada y permiso de operaciones Short.

---

## 4. Estructura Ultra Modular de Archivos

- **Dominio y Tipos:**
  - [`timeframe.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/timeframe.rs)
  - [`kline.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/kline.rs)
  - [`signal.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/signal.rs)
  - [`order.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/order.rs)
  - [`position.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/position.rs)
  - [`trade.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/types/trade.rs)
- **Base de Datos SQLite:**
  - [`db_config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_config.rs)
  - [`db_connection.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/db_connection.rs)
  - [`kline_repository.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/storage/kline_repository.rs)
- **Downloader Resiliente:**
  - [`endpoint_pool.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/endpoint_pool.rs)
  - [`raw_binance_kline.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/raw_binance_kline.rs)
  - [`binance_client.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/binance_client.rs)
  - [`downloader_service.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/downloader/downloader_service.rs)
- **Features Causales ($t \le T$):**
  - [`technical_indicators.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/features/technical_indicators.rs)
  - [`feature_matrix.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/features/feature_matrix.rs)
  - [`triple_barrier_labeler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/features/triple_barrier_labeler.rs)
  - [`train_test_split.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/features/train_test_split.rs)
- **Modelos de Decisión y Boosting:**
  - [`model_config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/model_config.rs)
  - [`forust_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/forust_model.rs)
  - [`forust_strategy.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/forust_strategy.rs)
  - [`treeboost_model.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_model.rs)
  - [`treeboost_strategy.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/models/treeboost_strategy.rs)
- **Motor Institucional de Backtesting:**
  - [`config.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/config.rs)
  - [`portfolio.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/portfolio.rs)
  - [`equity_point.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/equity_point.rs)
  - [`execution_engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/execution_engine.rs)
  - [`strategy.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/strategy.rs)
  - [`metrics.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics.rs)
  - [`metrics_calculator.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/metrics_calculator.rs)
  - [`engine.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/engine.rs)
  - [`report.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/backtest/report.rs)
- **Terminal Interactiva:**
  - [`menu_printer.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/menu_printer.rs)
  - [`input_reader.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/input_reader.rs)
  - [`db_handler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/db_handler.rs)
  - [`forust_handler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/forust_handler.rs)
  - [`treeboost_handler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/treeboost_handler.rs)
  - [`baseline_handler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/baseline_handler.rs)
  - [`config_handler.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/config_handler.rs)
  - [`interactive_app.rs`](file:///C:/Users/Allianz/Downloads/Antigravity/GBT/src/cli/interactive_app.rs)

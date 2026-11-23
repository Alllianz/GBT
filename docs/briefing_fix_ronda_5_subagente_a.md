# Briefing de Implementación: Subagente A (Ronda 5)
## Ámbito: Motor de Ejecución, Portafolio, Métricas y Tipos de Datos (BUG-38, BUG-39, BUG-41, BUG-44)

### 1. BUG-38: Contabilidad Exacta de Comisiones de Entrada
- **Archivos:** `src/types/position.rs`, `src/backtest/execution_engine.rs`, `src/backtest/metrics_calculator.rs`.
- **Acción:**
  1. En `src/types/position.rs`: Añadir `pub entry_commission: f64` al struct `Position` y actualizar su constructor o inicialización (`entry_commission: 0.0` o pasado como argumento).
  2. En `src/backtest/execution_engine.rs`:
     - En `process_signal_at_open`: cuando se crea una nueva posición (`Position { ... }`), asignar `entry_commission: commission`.
     - En `close_position`:
       - Calcular `let exit_commission = exit_notional * config.commission_rate;`.
       - Calcular `let total_commissions = pos.entry_commission + exit_commission;`.
       - Deducir el total de comisiones del PnL neto:
         `let uncapped_pnl = gross_pnl - total_commissions;`
         `let net_pnl = if let Some(isolated_margin) = pos.isolated_margin { if uncapped_pnl < -isolated_margin { -isolated_margin } else { uncapped_pnl } } else { uncapped_pnl };`
       - Como la comisión de entrada ya se descontó de `portfolio.cash_balance` en `process_signal_at_open`, al cerrar la posición solo se impacta el balance con `gross_pnl - exit_commission`:
         `portfolio.cash_balance += gross_pnl - exit_commission;`
         `portfolio.total_commissions_paid += exit_commission;`
       - En `CompletedTrade`:
         `commissions: total_commissions,`
         `net_pnl,`
         `return_pct: (net_pnl / pos.notional_value) * 100.0,`
  3. En `src/backtest/metrics_calculator.rs`: `total_commissions` ya suma `t.commissions`, que ahora contiene tanto entrada como salida.

### 2. BUG-39: Eliminación de Look-Ahead Bias en Salida por Bancarrota
- **Archivo:** `src/backtest/engine.rs`.
- **Acción:**
  - En `run_with_warmup`:
    - Rastrear el último índice evaluado en el bucle: `let mut last_eval_idx = 0; let mut is_bankrupt = false;`.
    - Dentro del bucle `for i in 0..klines.len()`: `last_eval_idx = i;`. Si quiebra (`portfolio.is_bankrupt(...)`): `is_bankrupt = true; break;`.
    - Al salir del bucle, si queda una posición activa `if let Some(pos) = portfolio.active_position.take()`:
      - Utilizar `let exit_candle = &klines[last_eval_idx];`.
      - Si `is_bankrupt`: razón `"Bankruptcy (Account Insolvent)"`. Si no: `"End of Data Horizon"`.
      - Cerrar la posición con `exit_candle.close`, `exit_candle.timestamp`, `last_eval_idx`.
      - ¡Eliminar cualquier referencia a `klines.last()` que lea el futuro!

### 3. BUG-41: Validación de Finitud Numérica en `Kline::new`
- **Archivo:** `src/types/kline.rs`.
- **Acción:**
  - En `Kline::new`:
    - Agregar validación explícita al inicio:
      ```rust
      if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() || !volume.is_finite() {
          return Err("Los valores de la vela deben ser números finitos (no NaN ni Inf)".to_string());
      }
      ```

### 4. BUG-44: Manejo de Breakeven en Racha de Pérdidas Consecutivas
- **Archivo:** `src/backtest/metrics_calculator.rs`.
- **Acción:**
  - En el cálculo de trades de `MetricsCalculator::calculate`:
    - Si `t.net_pnl > 0.0`: `winning_trades += 1; gross_profit += t.net_pnl; current_consecutive_losses = 0;`.
    - Si `t.net_pnl < 0.0`: `losing_trades += 1; gross_loss += t.net_pnl.abs(); current_consecutive_losses += 1; if current_consecutive_losses > max_consecutive_losses { max_consecutive_losses = current_consecutive_losses; }`.
    - Else (`t.net_pnl == 0.0`): `current_consecutive_losses = 0;` (una operación sin pérdidas interrumpe la racha).

### Verificación:
- Ejecutar `cargo check` y `cargo test --no-run` para asegurar compilación limpia (0 errores).

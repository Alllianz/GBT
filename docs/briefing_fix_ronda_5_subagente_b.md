# Briefing de Implementación: Subagente B (Ronda 5)
## Ámbito: Features, Etiquetado y Modelos de ML (BUG-40, BUG-45)

### 1. BUG-40: Horizonte Temporal Causal en `TripleBarrierLabeler` y Sincronización en Modelos
- **Archivos:**
  - `src/features/triple_barrier_labeler.rs`
  - `src/models/treeboost_model.rs`
  - `src/models/forust_model.rs`
  - `src/models/decision_tree/cart_model.rs`
  - `src/models/random_forest/forest_model.rs`
  - `src/models/native_gbdt/native_model.rs`
- **Acción:**
  1. En `src/features/triple_barrier_labeler.rs`:
     - En `label_regression`:
       - `let valid_n = n.saturating_sub(horizon);`
       - Para cada `i` en `0..valid_n`:
         - `let p_entry = klines[i + 1].open;` (ejecución al open de i+1)
         - `let p_exit = klines[i + horizon].close;` (cierre tras transcurrir `horizon` velas desde el open)
         - `labels.push((p_exit - p_entry) / p_entry);`
     - En `label_classification`:
       - `let valid_n = n.saturating_sub(horizon);`
       - Para cada `i` en `0..valid_n`:
         - Entrada en `let entry_price = klines[i + 1].open;`
         - Barreras: `upper = entry_price * (1.0 + pt); lower = entry_price * (1.0 - sl);`
         - Bucle de barreras evaluando desde la vela $i+1$:
           `for step in 0..horizon { let candle = &klines[i + 1 + step]; ... }`
         - Si toca barrera superior primero: `labels.push(1);`, inferior: `-1`, si no toca ninguna: evaluar retorno causal `p_exit = klines[i + horizon].close;`.
  2. En los 5 modelos (`treeboost_model.rs`, `forust_model.rs`, `cart_model.rs`, `forest_model.rs`, `native_model.rs`):
     - Sincronizar `let valid_rows = rows.saturating_sub(self.config.horizon);` (o equivalente) para que coincida exactamente con las filas etiquetadas generadas por `label_regression` o `label_classification`.

### 2. BUG-45: Saneamiento de Predicciones `.is_finite()` y Protección ante Pánico en Estrategias
- **Archivos:**
  - `src/models/forust_model.rs` y `src/models/forust_strategy.rs`
  - `src/models/decision_tree/cart_model.rs` y `src/models/decision_tree/cart_strategy.rs`
  - `src/models/random_forest/forest_model.rs` y `src/models/random_forest/forest_strategy.rs`
  - `src/models/native_gbdt/native_model.rs` y `src/models/native_gbdt/native_strategy.rs`
- **Acción:**
  1. En cada uno de los 4 modelos (`forust_model.rs`, `cart_model.rs`, `forest_model.rs`, `native_model.rs`):
     - En `predict_single`: asegurarse de que si el valor devuelto no es finito (`!val.is_finite()`), retorne `0.0` (o maneje `Ok(0.0)` según la firma).
     - En `predict_batch`: mapear las predicciones con `.map(|v| if v.is_finite() { v } else { 0.0 })`.
  2. En cada una de las 4 estrategias (`*_strategy.rs`):
     - Antes de invocar `.clamp(0.1, 1.0)`, evitar división por cero y NaN:
       ```rust
       let divisor = (long_th * 3.0).abs();
       let raw_strength = if divisor > 1e-9 { predicted_return / divisor } else { 0.5 };
       let strength = if raw_strength.is_finite() { raw_strength.clamp(0.1, 1.0) } else { 0.5 };
       ```
       (Aplicar análogo para `short_th`).

### Verificación:
- Ejecutar `cargo check` y `cargo test --no-run` para asegurar compilación limpia (0 errores).

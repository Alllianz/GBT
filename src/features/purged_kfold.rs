use super::embargo::TemporalEmbargo;

/// Partición de Validación Cruzada con Purga y Embargo Temporal (Purged Fold)
#[derive(Debug, Clone)]
pub struct PurgedFold {
    pub fold_index: usize,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_count: usize,
    pub embargoed_count: usize,
    pub test_start: usize,
    pub test_end: usize,
}

/// Particionador Temporal Purged K-Fold Cross-Validation (López de Prado, 2018)
///
/// Divide una serie de tiempo en K folds contiguos garantizando:
/// 1. **Purga Anterior:** Elimina observaciones de entrenamiento cuyo horizonte de etiqueta
///    (t + horizon) solape con el inicio del fold de prueba (evita fuga de retornos futuros).
/// 2. **Embargo Posterior:** Bloquea observaciones de entrenamiento posteriores al fold de prueba
///    durante `embargo_bars` (anula la fuga por memoria autorregresiva de EMA/ATR/MACD).
#[derive(Debug, Clone)]
pub struct PurgedKFold {
    pub n_splits: usize,
    pub horizon: usize,
    pub embargo: TemporalEmbargo,
}

impl PurgedKFold {
    pub fn new(n_splits: usize, horizon: usize, embargo_bars: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            horizon: horizon.max(1),
            embargo: TemporalEmbargo::new(embargo_bars),
        }
    }

    /// Genera la lista de K particiones purgadas con embargo
    pub fn split(&self, total_samples: usize) -> Result<Vec<PurgedFold>, String> {
        if total_samples < 50 {
            return Err("Se requieren al menos 50 muestras para Purged K-Fold CV".to_string());
        }

        let fold_size = total_samples / self.n_splits;
        let post_exclusion = self.horizon.max(self.embargo.embargo_bars);
        if fold_size <= post_exclusion {
            return Err(format!(
                "Muestras por fold ({}) insuficientes frente a horizonte ({}) y embargo ({})",
                fold_size, self.horizon, self.embargo.embargo_bars
            ));
        }

        let mut folds = Vec::with_capacity(self.n_splits);

        for k in 0..self.n_splits {
            let test_start = k * fold_size;
            let test_end = if k == self.n_splits - 1 {
                total_samples
            } else {
                (k + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            let mut train_indices = Vec::with_capacity(total_samples - test_indices.len());
            let mut purged_count = 0;
            let mut embargoed_count = 0;

            for i in 0..total_samples {
                if i >= test_start && i < test_end {
                    // Muestra pertenece al conjunto de prueba
                    continue;
                }

                if i < test_start {
                    // Purga previa: si el horizonte de la muestra solapa con el fold de test
                    if (i + self.horizon) >= test_start {
                        purged_count += 1;
                        continue;
                    }
                } else {
                    // Purga post-test y embargo (C-07):
                    // La última muestra de test es (test_end - 1), cuya etiqueta se extiende temporalmente
                    // hasta (test_end - 1 + horizon). Además, el embargo autorregresivo bloquea [test_end, test_end + embargo_bars).
                    // Por tanto, se excluye rigurosamente del entrenamiento la ventana [test_end, test_end + max(embargo_bars, horizon)).
                    if i < test_end.saturating_add(post_exclusion) {
                        if i < test_end.saturating_add(self.horizon) {
                            purged_count += 1;
                        } else {
                            embargoed_count += 1;
                        }
                        continue;
                    }
                }

                train_indices.push(i);
            }

            if train_indices.is_empty() {
                return Err(format!(
                    "Fold {} quedó sin muestras de entrenamiento tras aplicar purga y embargo",
                    k
                ));
            }

            folds.push(PurgedFold {
                fold_index: k,
                train_indices,
                test_indices,
                purged_count,
                embargoed_count,
                test_start,
                test_end,
            });
        }

        Ok(folds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purged_kfold_no_overlap_and_purging() {
        let n = 200;
        let horizon = 5;
        let embargo_bars = 10;
        let pkf = PurgedKFold::new(4, horizon, embargo_bars);
        let folds = pkf.split(n).unwrap();

        assert_eq!(folds.len(), 4);

        for fold in folds {
            // 1. Cero solapamiento entre train y test
            for &test_idx in &fold.test_indices {
                assert!(!fold.train_indices.contains(&test_idx));
            }

            // 2. Ninguna muestra de train previa al test solapa su horizonte con el test
            for &train_idx in &fold.train_indices {
                if train_idx < fold.test_start {
                    assert!(train_idx + horizon < fold.test_start);
                }
            }

            // 3. Ninguna muestra de train posterior al test cae en la zona de embargo
            for &train_idx in &fold.train_indices {
                if train_idx >= fold.test_end {
                    assert!(train_idx >= fold.test_end + embargo_bars);
                }
            }
        }
    }

    #[test]
    fn test_c07_purged_kfold_post_test_purging_when_embargo_less_than_horizon() {
        let n = 200;
        let horizon = 15;
        let embargo_bars = 5;
        let pkf = PurgedKFold::new(4, horizon, embargo_bars);
        let folds = pkf.split(n).unwrap();

        assert_eq!(folds.len(), 4);

        for fold in folds {
            // Ninguna muestra de train posterior al test debe caer en [test_end, test_end + max(embargo, horizon))
            let post_exclusion = horizon.max(embargo_bars);
            for &train_idx in &fold.train_indices {
                if train_idx >= fold.test_end {
                    assert!(
                        train_idx >= fold.test_end + post_exclusion,
                        "Fuga detectada: muestra {} dentro de la zona de purga post-test [{}, {})",
                        train_idx,
                        fold.test_end,
                        fold.test_end + post_exclusion
                    );
                }
            }
        }
    }
}

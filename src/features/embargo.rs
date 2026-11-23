/// Zona de Enfriamiento Temporal (Temporal Embargo)
///
/// Implementa la técnica institucional de Embargo (López de Prado, 2018):
/// Anula la fuga de información provocada por la correlación serial residual
/// y las colas de memoria exponencial de indicadores autorregresivos (ej. EMA30, ATR14, MACD).
///
/// Si un conjunto de prueba u OOF finaliza en el índice `T_test_end`, cualquier muestra
/// en el intervalo `[T_test_end, T_test_end + embargo_bars)` contiene en sus variables
/// predictoras información generada a partir de los precios del período de evaluación.
/// El embargo bloquea rigurosamente el uso de estas muestras en el entrenamiento posterior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalEmbargo {
    pub embargo_bars: usize,
}

impl Default for TemporalEmbargo {
    fn default() -> Self {
        // 30 barras de enfriamiento por defecto para cubrir los periodos de EMA30 y ATR14
        Self { embargo_bars: 30 }
    }
}

impl TemporalEmbargo {
    pub fn new(embargo_bars: usize) -> Self {
        Self { embargo_bars }
    }

    /// Construye el embargo a partir de un porcentaje del total de barras
    pub fn from_ratio(ratio: f64, total_bars: usize) -> Self {
        let bars = ((total_bars as f64) * ratio.clamp(0.0, 0.20)).round() as usize;
        Self {
            embargo_bars: bars.max(1),
        }
    }

    /// Retorna el rango prohibido por embargo [inicio_embargo, fin_embargo)
    #[inline]
    pub fn embargo_range(&self, test_end_idx: usize, total_len: usize) -> (usize, usize) {
        let start = test_end_idx.min(total_len);
        let embargo_end = start.saturating_add(self.embargo_bars).min(total_len);
        (start, embargo_end)
    }

    /// Retorna el primer índice seguro posterior a la zona de prueba y su embargo
    #[inline]
    pub fn safe_start_idx(&self, test_end_idx: usize, total_len: usize) -> usize {
        test_end_idx.saturating_add(self.embargo_bars).min(total_len)
    }

    /// Evalúa si un índice temporal cae dentro de la zona de enfriamiento
    #[inline]
    pub fn is_in_embargo(&self, index: usize, test_end_idx: usize) -> bool {
        index >= test_end_idx && index < test_end_idx.saturating_add(self.embargo_bars)
    }

    /// Filtra y excluye cualquier índice comprendido en la zona de embargo
    pub fn filter_indices(&self, candidate_indices: &[usize], test_end_idx: usize) -> Vec<usize> {
        let embargo_end = test_end_idx.saturating_add(self.embargo_bars);
        candidate_indices
            .iter()
            .copied()
            .filter(|&idx| idx < test_end_idx || idx >= embargo_end)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embargo_range_and_filtering() {
        let embargo = TemporalEmbargo::new(10);
        let test_end = 50;
        let total = 100;

        let (emb_start, emb_end) = embargo.embargo_range(test_end, total);
        assert_eq!(emb_start, 50);
        assert_eq!(emb_end, 60);

        let safe = embargo.safe_start_idx(test_end, total);
        assert_eq!(safe, 60);

        // Índices en la vecindad
        let candidates: Vec<usize> = (45..65).collect();
        let filtered = embargo.filter_indices(&candidates, test_end);

        // Deben conservarse 45..50 y 60..65
        assert_eq!(filtered.len(), 10);
        assert!(filtered.iter().all(|&idx| idx < 50 || idx >= 60));
    }

    #[test]
    fn test_c31_embargo_range_no_inverted_range_or_overflow() {
        let embargo = TemporalEmbargo::new(15);
        let total = 100;

        // test_end_idx > total_len: no debe invertir el rango (start <= end)
        let (start, end) = embargo.embargo_range(150, total);
        assert_eq!(start, 100);
        assert_eq!(end, 100);
        assert!(start <= end);

        // Desbordamiento cercano a usize::MAX
        let (start_overflow, end_overflow) = embargo.embargo_range(usize::MAX - 5, total);
        assert_eq!(start_overflow, 100);
        assert_eq!(end_overflow, 100);

        let safe_overflow = embargo.safe_start_idx(usize::MAX - 5, total);
        assert_eq!(safe_overflow, 100);

        assert!(!embargo.is_in_embargo(50, usize::MAX - 5));
    }
}

//! Módulo de cálculo institucional de Sharpe Ratio y Sortino Ratio.
//!
//! # Responsabilidades:
//! - C-12: Formulación corregida y dimensionalmente consistente de Israelsen Modificado para retornos negativos.
//!   Penalización estrictamente monótona: ante una misma pérdida media anual, mayor riesgo/volatilidad
//!   genera una puntuación más negativa (peor ranking). La anualización opera sobre cantidades anualizadas
//!   ($\mu_{\text{anual}} \times \sigma_{\text{anual}} = \mu \cdot \sigma \cdot N^{1.5}$), evitando que los scores en pérdida
//!   queden artificialmente colapsados a cero ($10^{-5}$).
//! - C-39: Protección ante curvas ultra-cortas (menos de 3 retornos válidos) o varianza nula ($\le 10^{-9}$),
//!   devolviendo un valor neutro documentado `0.0` y nunca el ratio espurio `99.99`.

use super::annualization::{AnnualizationConfig, AnnualizationInferer};
use super::equity_point::EquityPoint;

/// Calculador especializado para ratios ajustados por riesgo (Sharpe y Sortino).
pub struct SharpeSortinoCalculator;

impl SharpeSortinoCalculator {
    /// Calcula la tupla `(sharpe_ratio, sortino_ratio)` a partir de una serie de puntos de equity,
    /// deduciendo dinámicamente la configuración de anualización a partir de sus timestamps.
    pub fn calculate(curve: &[EquityPoint]) -> (f64, f64) {
        let config = AnnualizationInferer::infer_from_curve(curve);
        Self::calculate_with_config(curve, &config)
    }

    /// Calcula la tupla `(sharpe_ratio, sortino_ratio)` usando una configuración de anualización explícita.
    pub fn calculate_with_config(curve: &[EquityPoint], config: &AnnualizationConfig) -> (f64, f64) {
        // C-39: Protección ante curvas demasiado cortas. Para calcular retornos y desviación muestral
        // con significancia estadística básica se requieren al menos 3 retornos (mínimo 4 puntos de equity).
        if curve.len() < 4 {
            return (0.0, 0.0);
        }

        let mut returns = Vec::with_capacity(curve.len() - 1);
        for i in 1..curve.len() {
            let prev = curve[i - 1].equity;
            let curr = curve[i].equity;
            if prev > 0.0 && prev.is_finite() && curr.is_finite() {
                returns.push((curr - prev) / prev);
            }
        }

        // C-39: Si hay menos de 3 retornos computables, devolver 0.0 documentado
        if returns.len() < 3 {
            return (0.0, 0.0);
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        if !mean.is_finite() {
            return (0.0, 0.0);
        }

        let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let downside_variance = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|&r| r.powi(2))
            .sum::<f64>()
            / n;
        let downside_std_dev = downside_variance.sqrt();

        // Anualización dimensional de variables (C-12):
        // Retorno medio anualizado = media * barras_por_año
        // Desviación estándar anualizada = desv * sqrt(barras_por_año)
        let annual_mean = mean * config.bars_per_year;
        let annual_std = std_dev * config.annual_factor;
        let annual_downside_std = downside_std_dev * config.annual_factor;

        // C-12 y C-39:
        // 1. Ganancia o breakeven (annual_mean >= 0.0):
        //    - Con varianza normal (annual_std > 1e-9): Sharpe clásico = annual_mean / annual_std
        //    - Con varianza nula (annual_std <= 1e-9): 0.0 neutral (C-39: NUNCA 99.99 espurio)
        // 2. Pérdida neta (annual_mean < 0.0):
        //    - Formulación corregida de Israelsen: annual_mean * annual_std
        //    - Mayor volatilidad penaliza el score haciéndolo más negativo de forma estrictamente monótona.
        let sharpe = if annual_mean >= 0.0 {
            if annual_std > 1e-9 {
                annual_mean / annual_std
            } else {
                0.0
            }
        } else {
            annual_mean * annual_std.max(1e-9)
        };

        let sortino = if annual_mean >= 0.0 {
            if annual_downside_std > 1e-9 {
                annual_mean / annual_downside_std
            } else {
                0.0
            }
        } else {
            annual_mean * annual_downside_std.max(1e-9)
        };

        (
            if sharpe.is_finite() { sharpe } else { 0.0 },
            if sortino.is_finite() { sortino } else { 0.0 },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c39_short_curve_returns_zero() {
        let ts = 1_700_000_000_i64;

        // Curva de 1 punto (0 retornos)
        let c1 = vec![EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0)];
        assert_eq!(SharpeSortinoCalculator::calculate(&c1), (0.0, 0.0));

        // Curva de 2 puntos (1 retorno) -> Antiguamente devolvía 99.99 espurio
        let c2 = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 1050.0, 1050.0, 0.0, 0.0),
        ];
        assert_eq!(SharpeSortinoCalculator::calculate(&c2), (0.0, 0.0));

        // Curva de 3 puntos (2 retornos)
        let c3 = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 1050.0, 1050.0, 0.0, 0.0),
            EquityPoint::new(ts + 7200, 2, 1100.0, 1100.0, 0.0, 0.0),
        ];
        assert_eq!(SharpeSortinoCalculator::calculate(&c3), (0.0, 0.0));
    }

    #[test]
    fn test_c39_zero_variance_returns_zero_never_99_99() {
        let ts = 1_700_000_000_i64;
        // 4 puntos con retornos idénticos: 1000 -> 1010 (+1%), 1010 -> 1020.1 (+1%), 1020.1 -> 1030.301 (+1%)
        // Varianza = 0.0
        let c = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 1010.0, 1010.0, 0.0, 0.0),
            EquityPoint::new(ts + 7200, 2, 1020.1, 1020.1, 0.0, 0.0),
            EquityPoint::new(ts + 10800, 3, 1030.301, 1030.301, 0.0, 0.0),
        ];

        let (sharpe, sortino) = SharpeSortinoCalculator::calculate(&c);
        assert_eq!(sharpe, 0.0, "Varianza nula debe retornar 0.0 y jamás 99.99");
        assert_eq!(sortino, 0.0, "Varianza bajista nula debe retornar 0.0");
    }

    #[test]
    fn test_c12_israelsen_monotonicity_higher_volatility_worse_score() {
        let ts = 1_700_000_000_i64;

        // Estrategia A: Pérdida suave y controlada
        let curve_a = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 990.0, 990.0, 10.0, 1.0),
            EquityPoint::new(ts + 7200, 2, 980.0, 980.0, 20.0, 2.0),
            EquityPoint::new(ts + 10800, 3, 970.0, 970.0, 30.0, 3.0),
        ];

        // Estrategia B: Misma pérdida neta final pero volatilidad caótica destructiva
        let curve_b = vec![
            EquityPoint::new(ts, 0, 1000.0, 1000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 900.0, 900.0, 100.0, 10.0),
            EquityPoint::new(ts + 7200, 2, 1050.0, 1050.0, 0.0, 0.0),
            EquityPoint::new(ts + 10800, 3, 970.0, 970.0, 80.0, 7.6),
        ];

        let (sharpe_a, sortino_a) = SharpeSortinoCalculator::calculate(&curve_a);
        let (sharpe_b, sortino_b) = SharpeSortinoCalculator::calculate(&curve_b);

        assert!(sharpe_a < 0.0 && sharpe_b < 0.0);
        assert!(sortino_a < 0.0 && sortino_b < 0.0);

        // Bajo Israelsen modificado:
        // A debe ser superior (menos negativo) que B
        assert!(
            sharpe_a > sharpe_b,
            "Sharpe A ({}) debe ser mayor (menos castigado) que Sharpe B ({})",
            sharpe_a,
            sharpe_b
        );
        assert!(
            sortino_a > sortino_b,
            "Sortino A ({}) debe ser mayor (menos castigado) que Sortino B ({})",
            sortino_a,
            sortino_b
        );
    }

    #[test]
    fn test_c12_dimensional_scale_consistency() {
        let ts = 1_700_000_000_i64;
        // Estrategia horaria perdedora moderada (-0.05% por hora, ~-438% anualizado)
        let curve = vec![
            EquityPoint::new(ts, 0, 10_000.0, 10_000.0, 0.0, 0.0),
            EquityPoint::new(ts + 3600, 1, 9_995.0, 9_995.0, 5.0, 0.05),
            EquityPoint::new(ts + 7200, 2, 9_990.0, 9_990.0, 10.0, 0.10),
            EquityPoint::new(ts + 10800, 3, 9_980.0, 9_980.0, 20.0, 0.20),
        ];

        let (sharpe, _) = SharpeSortinoCalculator::calculate(&curve);

        // La métrica anualizada en pérdida no debe estar colapsada a cero infinitesimal (-0.00001)
        // Debe tener un orden de magnitud sustancial comparable con la escala de retornos
        assert!(
            sharpe < -0.01,
            "El Sharpe ratio corregido no debe colapsar a 0: {}",
            sharpe
        );
    }
}

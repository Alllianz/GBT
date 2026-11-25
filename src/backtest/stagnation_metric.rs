use super::equity_point::EquityPoint;

pub fn max_stagnation_ratio(equity_curve: &[EquityPoint]) -> (usize, f64) {
    if equity_curve.is_empty() {
        return (0, 0.0);
    }

    let mut peak = if equity_curve[0].equity.is_finite() {
        equity_curve[0].equity
    } else {
        0.0
    };
    let mut current_underwater = 0usize;
    let mut max_underwater = 0usize;

    for pt in equity_curve.iter().skip(1) {
        let eq = if pt.equity.is_finite() {
            pt.equity
        } else {
            peak
        };
        if eq > peak {
            peak = eq;
            current_underwater = 0;
        } else {
            current_underwater += 1;
            if current_underwater > max_underwater {
                max_underwater = current_underwater;
            }
        }
    }

    let ratio = max_underwater as f64 / (equity_curve.len().max(1) as f64);
    (max_underwater, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(idx: usize, eq: f64) -> EquityPoint {
        EquityPoint::new((idx as i64 + 1) * 1000, idx, eq, eq, 0.0, 0.0)
    }

    #[test]
    fn test_max_stagnation_ratio_empty_and_monotonic() {
        assert_eq!(max_stagnation_ratio(&[]), (0, 0.0));
        let mono = vec![pt(0, 100.0), pt(1, 110.0), pt(2, 120.0)];
        assert_eq!(max_stagnation_ratio(&mono), (0, 0.0));
    }

    #[test]
    fn test_max_stagnation_ratio_underwater_window() {
        let curve = vec![
            pt(0, 100.0),
            pt(1, 110.0),
            pt(2, 105.0),
            pt(3, 108.0),
            pt(4, 109.0),
            pt(5, 115.0),
        ];
        let (bars, ratio) = max_stagnation_ratio(&curve);
        assert_eq!(bars, 3);
        assert!((ratio - 0.5).abs() < 1e-12);
    }
}

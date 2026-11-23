use crate::types::{MarginMode, PositionSide};

pub struct MarginCalculator;

impl MarginCalculator {
    /// Recalcula causalmente el precio de liquidación en margen Cross en función
    /// del colateral vigente (C-10) y la cantidad de la posición.
    pub fn calculate_cross_liquidation(
        side: PositionSide,
        exec_price: f64,
        quantity: f64,
        collateral: f64,
    ) -> f64 {
        if !quantity.is_finite()
            || quantity <= 0.0
            || !exec_price.is_finite()
            || exec_price <= 0.0
        {
            return 0.0;
        }

        let maint_ratio = 0.005;
        let safe_collat = if collateral.is_finite() { collateral.max(0.0) } else { 0.0 };
        let liq = match side {
            PositionSide::Long => {
                let price_drop_buffer = exec_price - (safe_collat / quantity);
                if price_drop_buffer > 0.0 {
                    (price_drop_buffer / (1.0 - maint_ratio)).max(0.0)
                } else {
                    // Posición sobrecolateralizada o desapalancada (<= 1x):
                    // el colateral disponible absorbe una caída total a 0.0 sin liquidación.
                    0.0
                }
            }
            PositionSide::Short => {
                (exec_price + (safe_collat / quantity)) / (1.0 + maint_ratio)
            }
        };
        if liq.is_finite() && liq > 0.0 { liq } else { 0.0 }
    }

    /// Recalcula causalmente el precio de liquidación en margen Aislado en función
    /// del margen efectivo neto vigente (neto de comisión de entrada y de funding devengado) (C-11).
    pub fn calculate_isolated_liquidation(
        side: PositionSide,
        exec_price: f64,
        quantity: f64,
        effective_margin: f64,
    ) -> f64 {
        if !quantity.is_finite()
            || quantity <= 0.0
            || !exec_price.is_finite()
            || exec_price <= 0.0
        {
            return 0.0;
        }

        let maint_ratio = 0.005;
        let safe_margin = if effective_margin.is_finite() { effective_margin.max(0.0) } else { 0.0 };
        let liq = match side {
            PositionSide::Long => {
                let price_drop_buffer = exec_price - (safe_margin / quantity);
                if price_drop_buffer > 0.0 {
                    (price_drop_buffer / (1.0 - maint_ratio)).max(0.0)
                } else {
                    0.0
                }
            }
            PositionSide::Short => {
                (exec_price + (safe_margin / quantity)) / (1.0 + maint_ratio)
            }
        };
        if liq.is_finite() && liq > 0.0 { liq } else { 0.0 }
    }

    /// Calcula el margen aislado asignado (si aplica) y el precio de liquidación
    /// respetando estrictamente el modo de margen y previniendo liquidaciones espurias a 0.0 (BUG-04).
    ///
    /// - C-03: En margen aislado, la comisión de entrada se descuenta del margen asignado
    ///   obteniendo el margen efectivo neto `(assigned_margin - entry_commission)`.
    /// - C-13: Si el margen aislado está limitado por la equity disponible (margen < notional/leverage),
    ///   el precio de liquidación se formula con el margen REAL asignado neto de comisión.
    /// - C-10: En margen Cross, delega en `calculate_cross_liquidation`.
    pub fn calculate_margin_and_liquidation(
        margin_mode: MarginMode,
        side: PositionSide,
        exec_price: f64,
        quantity: f64,
        equity: f64,
        notional_target: f64,
        leverage: f64,
        entry_commission: f64,
    ) -> (Option<f64>, f64) {
        if !equity.is_finite()
            || equity <= 0.0
            || !quantity.is_finite()
            || quantity <= 0.0
            || !exec_price.is_finite()
            || exec_price <= 0.0
            || !leverage.is_finite()
            || leverage <= 0.0
        {
            return (None, 0.0);
        }

        match margin_mode {
            MarginMode::Cross => {
                let safe_liq = Self::calculate_cross_liquidation(side, exec_price, quantity, equity);
                (None, safe_liq)
            }
            MarginMode::Isolated => {
                // Margen asignado limitado por la equity disponible (C-13)
                let assigned_margin = (notional_target / leverage).min(equity);
                // Margen efectivo neto tras absorber la comisión de apertura (C-03)
                let effective_margin = (assigned_margin - entry_commission).max(0.0);
                let safe_liq = Self::calculate_isolated_liquidation(side, exec_price, quantity, effective_margin);
                (Some(assigned_margin), safe_liq)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug_f04_cross_liquidation_post_commission_equity() {
        // Ejecución Long con apalancamiento 10x
        let exec_price = 100.0;
        let notional = 10_000.0;
        let quantity = notional / exec_price; // 100 units
        let pre_comm_equity = 1_000.0;
        let commission = 4.0;
        let post_comm_equity = pre_comm_equity - commission; // 996.0

        // Con equity pre-comisión: 100 - (1000/100) = 90.0 buffer -> 90 / 0.995 = 90.452
        let (_, liq_pre) = MarginCalculator::calculate_margin_and_liquidation(
            MarginMode::Cross,
            PositionSide::Long,
            exec_price,
            quantity,
            pre_comm_equity,
            notional,
            10.0,
            0.0,
        );

        // Con equity real post-comisión: 100 - (996/100) = 90.04 buffer -> 90.04 / 0.995 = 90.492
        let (_, liq_post) = MarginCalculator::calculate_margin_and_liquidation(
            MarginMode::Cross,
            PositionSide::Long,
            exec_price,
            quantity,
            post_comm_equity,
            notional,
            10.0,
            0.0,
        );

        // El precio de liquidación post-comisión DEBE ser superior (más cercano al precio de entrada)
        // reflejando menor colateral disponible para absorber caídas
        assert!(
            liq_post > liq_pre,
            "Precio de liquidación post-comisión ({}) debe ser superior al pre-comisión ({})",
            liq_post,
            liq_pre
        );
    }

    #[test]
    fn test_isolated_margin_bounded_by_equity() {
        let (margin, liq) = MarginCalculator::calculate_margin_and_liquidation(
            MarginMode::Isolated,
            PositionSide::Long,
            100.0,
            10.0,
            50.0, // Solo $50 de equity disponible (C-13)
            1_000.0,
            10.0, // Margen teórico 100.0
            0.0,
        );
        assert_eq!(margin, Some(50.0));
        assert!(liq > 0.0);
        // buffer: 100 - (50 / 10) = 95.0 -> liq = 95.0 / 0.995 = 95.477
        assert!((liq - (95.0 / 0.995)).abs() < 1e-4);
    }

    #[test]
    fn test_c03_isolated_commission_reduces_effective_margin_and_tightens_liquidation() {
        let exec_price = 100.0;
        let quantity = 10.0;
        let equity = 1000.0;
        let notional = 1000.0;
        let leverage = 10.0; // assigned_margin = 100.0

        let (_, liq_zero_comm) = MarginCalculator::calculate_margin_and_liquidation(
            MarginMode::Isolated,
            PositionSide::Long,
            exec_price,
            quantity,
            equity,
            notional,
            leverage,
            0.0,
        );

        let (_, liq_with_comm) = MarginCalculator::calculate_margin_and_liquidation(
            MarginMode::Isolated,
            PositionSide::Long,
            exec_price,
            quantity,
            equity,
            notional,
            leverage,
            0.50, // Comisión de entrada $0.50
        );

        // Con comisión de entrada, el margen efectivo neto es menor (99.50 vs 100.0),
        // por lo que el precio de liquidación Long DEBE ser superior (más cercano a la entrada 100.0)
        assert!(
            liq_with_comm > liq_zero_comm,
            "Liq con comisión ({}) debe ser superior a liq sin comisión ({})",
            liq_with_comm,
            liq_zero_comm
        );
        // 100 - (99.5 / 10) = 90.05 -> 90.05 / 0.995 = 90.5025
        assert!((liq_with_comm - (90.05 / 0.995)).abs() < 1e-4);
    }

    #[test]
    fn test_c10_cross_liquidation_dynamic_recalculation() {
        let side = PositionSide::Long;
        let exec_price = 100.0;
        let quantity = 10.0;
        let initial_collateral = 1000.0;

        let liq_initial = MarginCalculator::calculate_cross_liquidation(
            side,
            exec_price,
            quantity,
            initial_collateral,
        );
        // equity / quantity = 100.0 == exec_price -> price_drop_buffer = 0.0 -> liq = 0.0 (no liquidable)
        assert_eq!(liq_initial, 0.0);

        // Tras pagos sucesivos de funding, el colateral se reduce a $500
        let reduced_collateral = 500.0;
        let liq_after_funding = MarginCalculator::calculate_cross_liquidation(
            side,
            exec_price,
            quantity,
            reduced_collateral,
        );
        // price_drop_buffer = 100 - 50 = 50.0 -> liq = 50.0 / 0.995 = 50.2512
        assert!(liq_after_funding > 0.0);
        assert!((liq_after_funding - (50.0 / 0.995)).abs() < 1e-4);
    }
}

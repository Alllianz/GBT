use crate::backtest::BacktestConfig;
use crate::types::{FitnessMetric, MarginMode, PositionSizing};
use super::input_reader::InputReader;

pub struct ConfigHandler;

impl ConfigHandler {
    pub fn handle(cfg: &mut BacktestConfig) {
        println!("\n--- PARÁMETROS FINANCIEROS Y MÉTRICA FITNESS ---");
        println!("  1. Capital Inicial:          ${:.2}", cfg.initial_capital);
        println!("  2. Apalancamiento:           {:.1}x", cfg.leverage);
        println!("  3. Modo de Margen:           {}", cfg.margin_mode);
        println!("  4. Tamaño de Posición:       {:?}", cfg.position_sizing);
        println!("  5. Comisión Maker/Taker:     {:.4} ({:.2}%)", cfg.commission_rate, cfg.commission_rate * 100.0);
        let sl_display = match cfg.fixed_stop_loss_pct {
            Some(sl) => format!("{:.4} ({:.2}%)", sl, sl * 100.0),
            None => "Desactivado (None)".to_string(),
        };
        println!("  6. Stop Loss Fijo:           {}", sl_display);
        let tp_display = match cfg.fixed_take_profit_pct {
            Some(tp) => format!("{:.4} ({:.2}%)", tp, tp * 100.0),
            None => "Desactivado (None)".to_string(),
        };
        println!("  7. Take Profit Fijo:         {}", tp_display);
        println!("  8. Permitir Operaciones Short: {}", if cfg.allow_short { "Sí" } else { "No" });
        println!("  9. Métrica Fitness Activa:   {}", cfg.fitness_metric);

        let modify = InputReader::read_with_default("¿Deseas editar estos valores? (s/n)", "n".to_string());
        if modify.to_lowercase() == "s" {
            cfg.initial_capital = InputReader::read_with_default("Capital inicial ($)", cfg.initial_capital);
            cfg.leverage = InputReader::read_with_default("Apalancamiento (ej: 1.0 = spot/aislado, 5.0, 10.0)", cfg.leverage);

            println!("\nSelecciona Modo de Margen:");
            println!("  [1] Isolated (Aislado) - Solo el margen asignado está en riesgo [Previo/Estándar]");
            println!("  [2] Cross (Cruzado) - Todo el balance de la cuenta respalda la posición");
            let m_choice = InputReader::read_with_default("Modo de Margen (1 o 2)", 1);
            cfg.margin_mode = if m_choice == 2 {
                MarginMode::Cross
            } else {
                MarginMode::Isolated
            };

            println!("\nSelecciona Tipo de Tamaño de Posición (Position Sizing):");
            println!("  [1] Porcentaje Dinámico del Equity Acumulado (ej: 0.25 = 25% compuesto) [Previo/Estándar]");
            println!("  [2] Porcentaje Nominal del Capital Inicial (ej: 0.10 = 10% margen fijo * Leverage)");
            println!("  [3] Monto Fijo en USD (ej: 1000.0)");
            println!("  [4] Basado en Riesgo de Stop Loss (ej: 0.01 = 1% riesgo)");
            let s_choice = InputReader::read_with_default("Tipo de Sizing (1, 2, 3 o 4)", 1);
            cfg.position_sizing = match s_choice {
                2 => {
                    let pct = InputReader::read_with_default("Fracción de capital inicial (ej: 0.10 = 10%)", 0.10);
                    PositionSizing::PercentOfInitialCapital(pct)
                }
                3 => {
                    let amt = InputReader::read_with_default("Monto fijo en USD (ej: 1000.0)", 1000.0);
                    PositionSizing::FixedAmount(amt)
                }
                4 => {
                    let risk = InputReader::read_with_default("Fracción de riesgo (ej: 0.01 = 1%)", 0.01);
                    PositionSizing::RiskBasedPercent(risk)
                }
                _ => {
                    let pct = InputReader::read_with_default("Fracción de equity compuesto (ej: 0.25 = 25%)", 0.25);
                    PositionSizing::PercentOfEquity(pct)
                }
            };

            let current_sl = cfg.fixed_stop_loss_pct.unwrap_or(0.0);
            let sl_input = InputReader::read_with_default(
                "Stop Loss Fijo (ej: 0.02 para 2%, o 0 para desactivar)",
                current_sl,
            );
            cfg.fixed_stop_loss_pct = if sl_input > 0.0 { Some(sl_input) } else { None };

            let current_tp = cfg.fixed_take_profit_pct.unwrap_or(0.0);
            let tp_input = InputReader::read_with_default(
                "Take Profit Fijo (ej: 0.04 para 4%, o 0 para desactivar)",
                current_tp,
            );
            cfg.fixed_take_profit_pct = if tp_input > 0.0 { Some(tp_input) } else { None };

            let default_short = if cfg.allow_short { "s" } else { "n" };
            let short_input = InputReader::read_with_default(
                "¿Permitir operaciones en Corto (Short)? (s/n)",
                default_short.to_string(),
            );
            cfg.allow_short = short_input.to_lowercase() == "s";

            println!("\nSelecciona Métrica Fitness a Evaluar/Probar:");
            println!("  [1] Sharpe Ratio Anualizado [Estándar]");
            println!("  [2] Score Compuesto Allianz [0.5*R² + 0.5*(Cap/MaxDD Nominal)]");
            println!("  [3] Sortino Ratio (Penaliza solo volatilidad bajista)");
            println!("  [4] Recovery Factor (Beneficio Neto / Max Drawdown Nominal)");
            println!("  [5] Linealidad R² de Balance (Suavidad OLS)");
            println!("  [6] Calmar Ratio (Retorno% / Max Drawdown%)");
            println!("  [7] Profit Factor (Ganancia Bruta / Pérdida Bruta)");
            println!("  [8] Retorno Neto %");
            let fit_choice = InputReader::read_with_default("Métrica Fitness (1 a 8)", 1);
            cfg.fitness_metric = match fit_choice {
                2 => FitnessMetric::CompositeAllianz,
                3 => FitnessMetric::SortinoRatio,
                4 => FitnessMetric::RecoveryFactor,
                5 => FitnessMetric::LinearityR2,
                6 => FitnessMetric::CalmarRatio,
                7 => FitnessMetric::ProfitFactor,
                8 => FitnessMetric::NetProfitPct,
                _ => FitnessMetric::SharpeRatio,
            };

            cfg.commission_rate = InputReader::read_with_default("Tasa comisión (0.0004 = 0.04%)", cfg.commission_rate);
            println!("✅ Parámetros de simulación y métrica fitness actualizados.");
        }
    }
}

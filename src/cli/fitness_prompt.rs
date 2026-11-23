use crate::types::FitnessMetric;
use super::input_reader::InputReader;

pub struct FitnessPrompt;

impl FitnessPrompt {
    /// Pregunta interactivamente a Allianz qué métrica fitness desea utilizar antes de iniciar la ejecución/entrenamiento
    pub fn prompt_selection(current: FitnessMetric) -> FitnessMetric {
        println!();
        println!("---------------------------------------------------------------");
        println!("  🎯 SELECCIÓN DE MÉTRICA FITNESS PARA ESTE ENTRENAMIENTO/TEST ");
        println!("---------------------------------------------------------------");
        println!("  [1] Sharpe Ratio Anualizado [Estándar de volatilidad]");
        println!("  [2] Score Compuesto Allianz [0.5*R² + 0.5*(Cap/MaxDD Nominal)]");
        println!("  [3] Sortino Ratio (Penaliza únicamente volatilidad bajista)");
        println!("  [4] Recovery Factor (Beneficio Neto / Max Drawdown Nominal)");
        println!("  [5] Linealidad R² de Balance (Consistencia de trayectoria OLS)");
        println!("  [6] Calmar Ratio (Retorno% / Max Drawdown%)");
        println!("  [7] Profit Factor (Ganancia Bruta / Pérdida Bruta)");
        println!("  [8] Retorno Neto %");
        println!("---------------------------------------------------------------");

        let default_opt = match current {
            FitnessMetric::SharpeRatio => 1,
            FitnessMetric::CompositeAllianz => 2,
            FitnessMetric::SortinoRatio => 3,
            FitnessMetric::RecoveryFactor => 4,
            FitnessMetric::LinearityR2 => 5,
            FitnessMetric::CalmarRatio => 6,
            FitnessMetric::ProfitFactor => 7,
            FitnessMetric::NetProfitPct => 8,
        };

        let choice = InputReader::read_with_default("Selecciona Métrica Fitness (1 a 8)", default_opt);
        let selected = match choice {
            2 => FitnessMetric::CompositeAllianz,
            3 => FitnessMetric::SortinoRatio,
            4 => FitnessMetric::RecoveryFactor,
            5 => FitnessMetric::LinearityR2,
            6 => FitnessMetric::CalmarRatio,
            7 => FitnessMetric::ProfitFactor,
            8 => FitnessMetric::NetProfitPct,
            _ => FitnessMetric::SharpeRatio,
        };

        println!("✅ Métrica fitness fijada para esta sesión: {}\n", selected);
        selected
    }
}

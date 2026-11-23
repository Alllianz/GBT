use std::path::PathBuf;
use crate::backtest::{BacktestReport, FullBacktestResult, FullBacktestRunner, WalkForwardResult};
use crate::dashboard::{DashboardGenerator, EquityCurveExporter};
use crate::types::{FitnessMetric, Timeframe};

/// Presentador y coordinador de exportaciones, dashboards y reportes institucionales para backtesting
pub struct BacktestPresenter;

impl BacktestPresenter {
    /// Exporta curvas (IS/OOS CSV + OOS JSON), genera Dashboard HTML e imprime reportes para Walk-Forward
    pub fn present_walk_forward(
        model_name: &str,
        model_slug: &str,
        tf: Timeframe,
        wf: &WalkForwardResult,
        opt_long: f64,
        opt_short: f64,
        best_fit: f64,
        fitness_metric: FitnessMetric,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let tf_str = tf.as_db_str();

        // 1. Exportación sin fallos silenciosos
        let is_csv = EquityCurveExporter::save_csv(&wf.is_result.equity_curve, model_slug, tf_str, "IS")?;
        let oos_csv = EquityCurveExporter::save_csv(&wf.oos_result.equity_curve, model_slug, tf_str, "OOS")?;
        let _oos_json = EquityCurveExporter::save_json(&wf.oos_result.equity_curve, model_slug, tf_str, "OOS")?;

        // 2. Generación de Dashboard HTML interactivo
        let dashboard_path = DashboardGenerator::generate_and_save(wf)?;

        // 3. Impresión estructurada en consola
        println!("\n==================================================");
        println!("           RESUMEN WALK-FORWARD IS vs OOS         ");
        println!("==================================================");
        println!("Modelo Evaluado:            {}", model_name);
        println!("Timeframe:                  {}", tf_str);
        println!("Calibración Umbrales:       Long > +{:.4}, Short < {:.4} (Score: {:.4})", opt_long, opt_short, best_fit);
        println!("Retorno In-Sample (IS):     {:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.is_net_profit_pct, wf.metrics.is_sharpe, wf.metrics.is_max_dd_pct);
        println!("Retorno Out-of-Sample (OOS):{:+.2}% (Sharpe: {:.2}, DD: {:.2}%)", wf.metrics.oos_net_profit_pct, wf.metrics.oos_sharpe, wf.metrics.oos_max_dd_pct);
        println!("Profit Factor (OOS):        {:.2}", wf.metrics.oos_profit_factor);
        println!("Win Rate (OOS):             {:.1}%", wf.metrics.oos_win_rate_pct);
        println!("Consistencia R² (OOS):      {:.4}", wf.metrics.oos_curve_r2);
        println!("Ratio de Robustez (OOS/IS): {:.2}x", wf.metrics.robustness_ratio);
        println!("--------------------------------------------------");
        println!("💾 Equity Curve IS guardada en:  {}", is_csv.display());
        println!("💾 Equity Curve OOS guardada en: {}", oos_csv.display());
        println!("🌐 DASHBOARD HTML GENERADO EN:   {}", dashboard_path.display());
        println!("==================================================");

        let report_oos = BacktestReport::format_summary(&wf.oos_result);
        println!("\nReporte Detallado Out-of-Sample (OOS):\n{}", report_oos);
        println!("{}", BacktestReport::format_fitness_comparison(&wf.oos_result, fitness_metric));

        Ok(dashboard_path)
    }

    /// Exporta curvas (FULL/IS/OOS CSV + OOS JSON), genera Dashboard HTML interactivo con visualización
    /// de purga e imprime auditoría causal institucional completa para Backtest Completo
    pub fn present_full_backtest(
        model_name: &str,
        model_slug: &str,
        tf: Timeframe,
        full_result: &FullBacktestResult,
        opt_long: f64,
        opt_short: f64,
        best_fit: f64,
        fitness_metric: FitnessMetric,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let tf_str = tf.as_db_str();

        // 1. Exportaciones completas requeridas (FULL, IS, OOS CSV y OOS JSON) sin ignorar errores
        let full_csv = EquityCurveExporter::save_csv(&full_result.combined_result.equity_curve, model_slug, tf_str, "FULL")?;
        let is_csv = EquityCurveExporter::save_csv(&full_result.is_result.equity_curve, model_slug, tf_str, "IS")?;
        let oos_csv = EquityCurveExporter::save_csv(&full_result.oos_result.equity_curve, model_slug, tf_str, "OOS")?;
        let _oos_json = EquityCurveExporter::save_json(&full_result.oos_result.equity_curve, model_slug, tf_str, "OOS")?;

        // 2. Generación y guardado del Dashboard HTML interactivo en dashboards/
        let dashboard_path = DashboardGenerator::generate_and_save_full(model_name, tf_str, full_result, fitness_metric)?;

        // 3. Confirmación en consola de archivos generados
        println!("\n==================================================");
        println!("       EXPORTACIONES Y DASHBOARD INTERACTIVO      ");
        println!("==================================================");
        println!("💾 Curva de equity consolidada (con Purga/Embargo) guardada en: {}", full_csv.display());
        println!("💾 Equity Curve IS guardada en:                     {}", is_csv.display());
        println!("💾 Equity Curve OOS guardada en:                    {}", oos_csv.display());
        println!("🌐 DASHBOARD HTML GENERADO EN:                      {}", dashboard_path.display());
        println!("==================================================");
        println!("Calibración Umbrales: Long > +{:.4}, Short < {:.4} (Score: {:.4})", opt_long, opt_short, best_fit);

        // 4. Reporte resumido causal (comparativa de 7 filas IS / Purga / OOS / Total)
        let audit_summary = FullBacktestRunner::format_audit_summary(full_result);
        println!("{}", audit_summary);

        // 5. Reporte institucional completo OOS y auditoría de fitness
        let report_oos = BacktestReport::format_summary(&full_result.oos_result);
        println!("\nReporte Detallado Out-of-Sample (OOS):\n{}", report_oos);
        println!("{}", BacktestReport::format_fitness_comparison(&full_result.oos_result, fitness_metric));

        Ok(dashboard_path)
    }
}

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use chrono::Utc;
use crate::backtest::{FullBacktestResult, WalkForwardResult};
use crate::types::FitnessMetric;
use super::html_template::HtmlTemplate;

pub struct DashboardGenerator;

impl DashboardGenerator {
    pub fn ensure_dashboard_dir() -> std::io::Result<PathBuf> {
        let dir = PathBuf::from("dashboards");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn sanitize_model_name(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
                _ => c,
            })
            .collect()
    }

    pub fn generate_and_save(
        wf: &WalkForwardResult,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let dir = Self::ensure_dashboard_dir()?;
        let now_str = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let sanitized_model = Self::sanitize_model_name(&wf.model_name);
        let filename = format!(
            "dashboard_{}_{}_{}.html",
            sanitized_model,
            wf.timeframe,
            now_str
        );
        let path = dir.join(&filename);

        let html_content = HtmlTemplate::render(wf);
        let mut file = File::create(&path)?;
        file.write_all(html_content.as_bytes())?;

        Ok(path)
    }

    /// Genera y guarda el Dashboard HTML interactivo completo para una ejecución Purged Full Backtest
    pub fn generate_and_save_full(
        model_name: &str,
        timeframe: &str,
        full: &FullBacktestResult,
        fitness_metric: FitnessMetric,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let dir = Self::ensure_dashboard_dir()?;
        let now_str = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let sanitized_model = Self::sanitize_model_name(model_name);
        let filename = format!(
            "dashboard_{}_{}_full_{}.html",
            sanitized_model,
            timeframe,
            now_str
        );
        let path = dir.join(&filename);

        let html_content = HtmlTemplate::render_purged_full(model_name, timeframe, full, fitness_metric);
        let mut file = File::create(&path)?;
        file.write_all(html_content.as_bytes())?;

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_model_name() {
        let dirty = "Model: XGBoost / Test \\ Version * \"special\" ? <fast> | space";
        let clean = DashboardGenerator::sanitize_model_name(dirty);
        assert_eq!(clean, "model__xgboost___test___version____special_____fast____space");
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '] {
            assert!(!clean.contains(ch));
        }
    }
}

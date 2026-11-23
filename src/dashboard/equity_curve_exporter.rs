use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use chrono::{TimeZone, Utc};
use crate::backtest::EquityPoint;

pub struct EquityCurveExporter;

/// BUG-42: Sanitiza caracteres inválidos en sistemas de archivos (especialmente Windows: / \ : * ? " < > | y espacios)
pub fn sanitize_filename(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            _ => c,
        })
        .collect()
}

impl EquityCurveExporter {
    pub fn ensure_output_dir() -> std::io::Result<PathBuf> {
        let dir = PathBuf::from("output/equity_curves");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Guarda la curva de balance/equity en JSON con escritura en búfer optimizada
    pub fn save_json(
        curve: &[EquityPoint],
        model_name: &str,
        timeframe: &str,
        segment: &str, // "IS", "OOS", "FULL"
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let dir = Self::ensure_output_dir()?;
        let now_str = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let clean_model = sanitize_filename(model_name);
        let clean_tf = sanitize_filename(timeframe);
        let clean_seg = sanitize_filename(segment);
        let filename = format!("equity_{}_{}_{}_{}.json", clean_model, clean_tf, clean_seg, now_str);
        let path = dir.join(&filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, curve)?;
        writer.flush()?;

        Ok(path)
    }

    /// Guarda la curva de balance/equity en CSV con escritura en búfer optimizada para series grandes
    pub fn save_csv(
        curve: &[EquityPoint],
        model_name: &str,
        timeframe: &str,
        segment: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let dir = Self::ensure_output_dir()?;
        let now_str = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let clean_model = sanitize_filename(model_name);
        let clean_tf = sanitize_filename(timeframe);
        let clean_seg = sanitize_filename(segment);
        let filename = format!("equity_{}_{}_{}_{}.csv", clean_model, clean_tf, clean_seg, now_str);
        let path = dir.join(&filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "candle_idx,timestamp,utc_date,balance,equity,drawdown_nominal,drawdown_pct")?;

        for p in curve {
            let date_str = Utc
                .timestamp_millis_opt(p.timestamp)
                .single()
                .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "N/A".to_string());

            writeln!(
                writer,
                "{},{},{},{:.4},{:.4},{:.4},{:.4}",
                p.candle_idx, p.timestamp, date_str, p.balance, p.equity, p.drawdown_nominal, p.drawdown_pct
            )?;
        }
        writer.flush()?;

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Baseline:EMA/1h*test?"), "baseline_ema_1h_test_");
        assert_eq!(sanitize_filename("Model Name with Spaces"), "model_name_with_spaces");
        assert_eq!(sanitize_filename("Clean_Name-123"), "clean_name-123");
    }
}

use rusqlite::Connection;
use crate::backtest::BacktestConfig;
use crate::features::{SplitDataset, TrainTestSplit};
use crate::models::ModelConfig;
use crate::storage::KlineRepository;
use crate::types::{Kline, Timeframe};
use super::fitness_prompt::FitnessPrompt;
use super::input_reader::InputReader;

/// Paquete con el dataset cargado y particionado causalmente para evaluación de backtest
#[derive(Debug, Clone)]
pub struct BacktestDatasetBundle {
    pub symbol: String,
    pub timeframe: Timeframe,
    pub klines: Vec<Kline>,
    pub split: SplitDataset,
    pub bt_cfg: BacktestConfig,
}

/// Asistente CLI modular para solicitud de parámetros, carga de klines y partición causal
pub struct BacktestDatasetHelper;

impl BacktestDatasetHelper {
    /// Prepara el dataset para Backtest Walk-Forward (solicitando opcionalmente ratio IS interactivo)
    pub fn prepare_for_walk_forward(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<Option<BacktestDatasetBundle>, Box<dyn std::error::Error + Send + Sync>> {
        let (symbol, tf, active_bt_cfg) = Self::prompt_symbol_timeframe_and_fitness(bt_cfg)?;
        let klines = KlineRepository::load_all(conn, &symbol, tf)?;
        if klines.len() < 100 {
            println!(
                "Se requieren al menos 100 velas en la DB para [{}] (disponibles: {}). Descarga velas primero en el menú [1].",
                tf,
                klines.len()
            );
            return Ok(None);
        }

        let train_ratio = InputReader::read_with_default(
            "Proporción In-Sample (IS) (ej: 0.70 para 70% Train)",
            0.70,
        );
        let purge_bars = m_cfg.horizon + 2;

        let split = TrainTestSplit::split_chronological(&klines, train_ratio, purge_bars)?;
        println!("\n📊 {}", split.train_range_desc);
        println!("📊 {}", split.test_range_desc);

        Ok(Some(BacktestDatasetBundle {
            symbol,
            timeframe: tf,
            klines,
            split,
            bt_cfg: active_bt_cfg,
        }))
    }

    /// Prepara el dataset para Backtest Completo Purged (70% IS / Purga / 30% OOS estándar)
    pub fn prepare_for_full_backtest(
        conn: &Connection,
        m_cfg: &ModelConfig,
        bt_cfg: &BacktestConfig,
    ) -> Result<Option<BacktestDatasetBundle>, Box<dyn std::error::Error + Send + Sync>> {
        let (symbol, tf, active_bt_cfg) = Self::prompt_symbol_timeframe_and_fitness(bt_cfg)?;
        let klines = KlineRepository::load_all(conn, &symbol, tf)?;
        if klines.len() < 100 {
            println!(
                "Se requieren al menos 100 velas en la DB para [{}] (disponibles: {}).",
                tf,
                klines.len()
            );
            return Ok(None);
        }

        let train_ratio = 0.70;
        let purge_bars = m_cfg.horizon + 2;

        let split = TrainTestSplit::split_chronological(&klines, train_ratio, purge_bars)?;
        println!("\n📊 {}", split.train_range_desc);
        println!("📊 {}", split.test_range_desc);

        Ok(Some(BacktestDatasetBundle {
            symbol,
            timeframe: tf,
            klines,
            split,
            bt_cfg: active_bt_cfg,
        }))
    }

    fn prompt_symbol_timeframe_and_fitness(
        bt_cfg: &BacktestConfig,
    ) -> Result<(String, Timeframe, BacktestConfig), Box<dyn std::error::Error + Send + Sync>> {
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let tf_str = InputReader::read_with_default("Timeframe para evaluación", "1h".to_string());
        let tf: Timeframe = tf_str.parse()?;

        let mut active_cfg = bt_cfg.clone();
        active_cfg.timeframe = Some(tf);
        active_cfg.fitness_metric = FitnessPrompt::prompt_selection(active_cfg.fitness_metric);

        Ok((symbol, tf, active_cfg))
    }
}

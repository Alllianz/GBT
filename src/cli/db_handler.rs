use rusqlite::Connection;
use crate::downloader::DownloaderService;
use crate::storage::KlineRepository;
use crate::types::Timeframe;
use super::input_reader::InputReader;
use super::menu_printer::MenuPrinter;

pub struct DbHandler;

impl DbHandler {
    pub async fn handle(conn: &mut Connection) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            MenuPrinter::print_db_menu();
            let choice = InputReader::read_line();

            match choice.as_str() {
                "1" => {
                    Self::show_inventory(conn)?;
                }
                "2" => {
                    Self::sync_single_timeframe(conn).await?;
                }
                "3" => {
                    Self::sync_all_timeframes(conn).await?;
                }
                "0" => break,
                _ => println!("Opción no reconocida."),
            }
        }
        Ok(())
    }

    fn show_inventory(conn: &Connection) -> Result<(), rusqlite::Error> {
        let all_tfs = [
            Timeframe::M1, Timeframe::M3, Timeframe::M5, Timeframe::M15, Timeframe::M30,
            Timeframe::H1, Timeframe::H2, Timeframe::H4, Timeframe::H6, Timeframe::H8, Timeframe::H12,
            Timeframe::D1, Timeframe::D3, Timeframe::W1,
        ];

        let available_symbols = KlineRepository::get_distinct_symbols(conn)?;
        let default_sym = available_symbols.first().cloned().unwrap_or_else(|| "BTCUSDT".to_string());
        let prompt = if available_symbols.is_empty() {
            "Símbolo de Binance a consultar (ej: BTCUSDT)".to_string()
        } else {
            format!("Símbolo a consultar (Detectados en DB: {})", available_symbols.join(", "))
        };
        let symbol = InputReader::read_with_default(&prompt, default_sym);

        println!("\n--- INVENTARIO DE VELAS EN BASE DE DATOS [{}] ---", symbol);
        for tf in all_tfs {
            let count = KlineRepository::count_klines(conn, &symbol, tf)?;
            let latest = KlineRepository::get_latest_timestamp(conn, &symbol, tf)?;

            let date_str = match latest {
                Some(ts) => {
                    use chrono::{TimeZone, Utc};
                    Utc.timestamp_millis_opt(ts)
                        .single()
                        .map(|d| d.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "N/A".to_string())
                }
                None => "Sin datos".to_string(),
            };

            println!("  [{:>3}] -> {:>7} velas | Última: {}", tf.as_db_str(), count, date_str);
        }
        println!("--------------------------------------------");
        Ok(())
    }

    async fn sync_single_timeframe(conn: &mut Connection) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let tf_input = InputReader::read_with_default("Timeframe (1m, 5m, 15m, 1h, 4h, 1d)", "1h".to_string());

        let tf: Timeframe = match tf_input.parse() {
            Ok(t) => t,
            Err(_) => {
                println!("Timeframe no válido: {}", tf_input);
                return Ok(());
            }
        };

        println!("Iniciando sincronización para {} en timeframe [{}]...", symbol, tf);
        let service = DownloaderService::new(&symbol)?;
        let downloaded = service.sync_timeframe(conn, tf, None).await?;
        println!("✅ Sincronización completada. Nuevas velas descargadas e insertadas: {}", downloaded);
        Ok(())
    }

    async fn sync_all_timeframes(conn: &mut Connection) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let symbol = InputReader::read_with_default("Símbolo de Binance (ej: BTCUSDT)", "BTCUSDT".to_string());
        let target_tfs = [Timeframe::M15, Timeframe::H1, Timeframe::H4, Timeframe::D1];

        println!("Sincronizando timeframes estándar ({:?}) para {}...", target_tfs, symbol);
        let service = DownloaderService::new(&symbol)?;
        let total = service.sync_all_timeframes(conn, &target_tfs).await?;
        println!("✅ Sincronización masiva finalizada. Total de velas: {}", total);
        Ok(())
    }
}

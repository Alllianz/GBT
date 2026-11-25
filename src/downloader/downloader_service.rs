use std::time::Duration;
use chrono::{TimeZone, Utc};
use rusqlite::Connection;
use crate::storage::KlineRepository;
use crate::types::Timeframe;
use super::binance_client::BinanceClient;

pub struct DownloaderService {
    client: BinanceClient,
    symbol: String,
}

impl DownloaderService {
    pub fn new(symbol: impl Into<String>) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: BinanceClient::new()?,
            symbol: symbol.into(),
        })
    }

    pub async fn sync_timeframe(
        &self,
        conn: &mut Connection,
        tf: Timeframe,
        default_start_time: Option<i64>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.sync_timeframe_range(conn, tf, default_start_time, None).await
    }

    pub async fn sync_timeframe_range(
        &self,
        conn: &mut Connection,
        tf: Timeframe,
        default_start_time: Option<i64>,
        target_end_time: Option<i64>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut total_downloaded = 0;
        let default_ts = default_start_time.unwrap_or_else(|| {
            Utc.with_ymd_and_hms(2017, 8, 17, 0, 0, 0)
                .single()
                .map(|dt| dt.timestamp_millis())
                .unwrap_or(1_502_928_000_000)
        });

        let now_ms = Utc::now().timestamp_millis();
        let effective_end_time = target_end_time.unwrap_or_else(|| now_ms.saturating_sub(1500));
        let tf_duration = tf.duration_millis();

        let mut consecutive_empty_batches = 0;
        const MAX_EMPTY_RETRIES: usize = 5;

        let last_ts = KlineRepository::get_latest_timestamp(conn, &self.symbol, tf)?;
        let mut cursor_time = match last_ts {
            Some(ts) => ts + 1,
            None => {
                println!(
                    "  [17-Ago-2017] Base de datos vacía para {} en {}. Iniciando histórico desde 2017-08-17 00:00:00 UTC...",
                    self.symbol,
                    tf.as_db_str()
                );
                default_ts
            }
        };

        loop {
            if cursor_time >= effective_end_time {
                break;
            }

            let batch = self
                .client
                .download_klines(&self.symbol, tf, Some(cursor_time), 1000)
                .await?;

            if batch.klines.is_empty() {
                if cursor_time + (tf_duration * 1000) < effective_end_time {
                    consecutive_empty_batches += 1;
                    if consecutive_empty_batches > MAX_EMPTY_RETRIES {
                        println!(
                            "  ⚠️ [{}] {} lotes vacíos consecutivos antes de alcanzar el tiempo final. Concluyendo sincronización.",
                            tf.as_db_str(),
                            consecutive_empty_batches
                        );
                        break;
                    }
                    cursor_time += tf_duration * 1000;
                    println!(
                        "  [{}] Hueco temporal en {}. Avanzando cursor 1000 velas (reintento {}/{})",
                        tf.as_db_str(),
                        cursor_time,
                        consecutive_empty_batches,
                        MAX_EMPTY_RETRIES
                    );
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                } else {
                    break;
                }
            }

            consecutive_empty_batches = 0;
            let count = batch.klines.len();
            let inserted = KlineRepository::insert_batch(conn, &self.symbol, tf, &batch.klines)?;
            total_downloaded += inserted;

            if let Some(last) = batch.klines.last() {
                cursor_time = if last.timestamp + 1 > cursor_time {
                    last.timestamp + 1
                } else {
                    cursor_time + tf_duration
                };

                if let Some(date) = Utc.timestamp_millis_opt(last.timestamp).single() {
                    println!(
                        "  [{}] Guardadas {} velas (total lote: {}, crudas: {}). Última fecha: {}",
                        tf.as_db_str(),
                        inserted,
                        count,
                        batch.raw_count,
                        date.format("%Y-%m-%d %H:%M:%S")
                    );
                }

                if last.timestamp + tf_duration >= effective_end_time {
                    break;
                }

                if batch.raw_count < 1000 && last.timestamp + (tf_duration * 2) >= effective_end_time {
                    break;
                }
            } else {
                cursor_time += tf_duration;
            }

            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        Ok(total_downloaded)
    }

    pub async fn sync_all_timeframes(
        &self,
        conn: &mut Connection,
        timeframes: &[Timeframe],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut total = 0;
        for &tf in timeframes {
            println!("Sincronizando velas para {} ({})", self.symbol, tf.as_db_str());
            let downloaded = self.sync_timeframe(conn, tf, None).await?;
            total += downloaded;
            println!("-> Total velas sincronizadas en {}: {}", tf.as_db_str(), downloaded);
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use crate::downloader::KlineBatch;
    use crate::types::Kline;

    #[test]
    fn test_bug05_kline_batch_preserves_raw_count_independent_of_filtered_klines() {
        let dummy = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 10.0,
        };
        let batch = KlineBatch {
            klines: vec![dummy; 995],
            raw_count: 1000,
        };

        assert_eq!(batch.klines.len(), 995);
        assert_eq!(batch.raw_count, 1000);
        assert!(batch.raw_count >= 1000, "Debe indicar que Binance tenía un bloque completo disponible");
    }
}

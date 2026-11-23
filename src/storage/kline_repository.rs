use crate::types::{Kline, Timeframe};
use rusqlite::{params, Connection, Result};
use super::gap_detector::{GapDetector, SeriesGap};

pub struct KlineRepository;

impl KlineRepository {
    pub fn get_latest_timestamp(conn: &Connection, symbol: &str, tf: Timeframe) -> Result<Option<i64>> {
        let mut stmt = conn.prepare(
            "SELECT timestamp FROM klines WHERE symbol = ?1 AND timeframe = ?2 ORDER BY timestamp DESC LIMIT 1",
        )?;
        let mut rows = stmt.query(params![symbol, tf.as_db_str()])?;

        if let Some(row) = rows.next()? {
            let ts: i64 = row.get(0)?;
            Ok(Some(ts))
        } else {
            Ok(None)
        }
    }

    pub fn insert_batch(conn: &mut Connection, symbol: &str, tf: Timeframe, klines: &[Kline]) -> Result<usize> {
        if klines.is_empty() {
            return Ok(0);
        }

        // Validación previa de dominio y monotonía temporal estricta de las velas a insertar
        let mut prev_ts: Option<i64> = None;
        for (idx, k) in klines.iter().enumerate() {
            Kline::new(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    idx,
                    rusqlite::types::Type::Null,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
                ))?;

            if let Some(prev) = prev_ts {
                if k.timestamp == prev {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        idx,
                        rusqlite::types::Type::Integer,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Timestamp duplicado al insertar en índice {}: {}", idx, k.timestamp),
                        )),
                    ));
                } else if k.timestamp < prev {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        idx,
                        rusqlite::types::Type::Integer,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Timestamps desordenados al insertar en índices {} y {}: {} > {}", idx - 1, idx, prev, k.timestamp),
                        )),
                    ));
                }
            }
            prev_ts = Some(k.timestamp);
        }

        let tx = conn.transaction()?;
        let mut count = 0;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO klines (symbol, timeframe, timestamp, open, high, low, close, volume)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;

            for k in klines {
                stmt.execute(params![
                    symbol,
                    tf.as_db_str(),
                    k.timestamp,
                    k.open,
                    k.high,
                    k.low,
                    k.close,
                    k.volume,
                ])?;
                count += 1;
            }
        }
        tx.commit()?;
        Ok(count)
    }

    /// Valida que la secuencia cargada preserve orden estrictamente monótono y sin duplicados
    fn validate_loaded_klines(klines: &[Kline], symbol: &str, tf: Timeframe) -> Result<()> {
        for i in 1..klines.len() {
            let prev = klines[i - 1].timestamp;
            let curr = klines[i].timestamp;
            if curr == prev {
                return Err(rusqlite::Error::FromSqlConversionFailure(
                    i,
                    rusqlite::types::Type::Integer,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Timestamp duplicado detectado en {} [{}] en índice {}: {}", symbol, tf, i, curr),
                    )),
                ));
            } else if curr < prev {
                return Err(rusqlite::Error::FromSqlConversionFailure(
                    i,
                    rusqlite::types::Type::Integer,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Timestamps desordenados detectados en {} [{}] en índices {} y {}: {} > {}", symbol, tf, i - 1, i, prev, curr),
                    )),
                ));
            }
        }
        Ok(())
    }

    pub fn load_all(conn: &Connection, symbol: &str, tf: Timeframe) -> Result<Vec<Kline>> {
        let mut stmt = conn.prepare(
            "SELECT timestamp, open, high, low, close, volume FROM klines 
             WHERE symbol = ?1 AND timeframe = ?2 
             ORDER BY timestamp ASC",
        )?;

        let rows = stmt.query_map(params![symbol, tf.as_db_str()], |row| {
            let ts: i64 = row.get(0)?;
            let open: f64 = row.get(1)?;
            let high: f64 = row.get(2)?;
            let low: f64 = row.get(3)?;
            let close: f64 = row.get(4)?;
            let volume: f64 = row.get(5)?;

            Kline::new(ts, open, high, low, close, volume)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Null,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
                ))
        })?;

        let mut result = Vec::new();
        for r in rows {
            result.push(r?);
        }

        Self::validate_loaded_klines(&result, symbol, tf)?;

        let gaps = GapDetector::find_gaps(&result, tf);
        if !gaps.is_empty() {
            eprintln!(
                "[WARN] KlineRepository: Se detectaron {} gap(s) temporales en {} [{}].",
                gaps.len(),
                symbol,
                tf
            );
        }

        Ok(result)
    }

    pub fn load_range(
        conn: &Connection,
        symbol: &str,
        tf: Timeframe,
        start_ts: i64,
        end_ts: i64,
    ) -> Result<Vec<Kline>> {
        let mut stmt = conn.prepare(
            "SELECT timestamp, open, high, low, close, volume FROM klines 
             WHERE symbol = ?1 AND timeframe = ?2 AND timestamp >= ?3 AND timestamp <= ?4 
             ORDER BY timestamp ASC",
        )?;

        let rows = stmt.query_map(params![symbol, tf.as_db_str(), start_ts, end_ts], |row| {
            let ts: i64 = row.get(0)?;
            let open: f64 = row.get(1)?;
            let high: f64 = row.get(2)?;
            let low: f64 = row.get(3)?;
            let close: f64 = row.get(4)?;
            let volume: f64 = row.get(5)?;

            Kline::new(ts, open, high, low, close, volume)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Null,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
                ))
        })?;

        let mut result = Vec::new();
        for r in rows {
            result.push(r?);
        }

        Self::validate_loaded_klines(&result, symbol, tf)?;

        let gaps = GapDetector::find_gaps(&result, tf);
        if !gaps.is_empty() {
            eprintln!(
                "[WARN] KlineRepository: Se detectaron {} gap(s) temporales en {} [{}].",
                gaps.len(),
                symbol,
                tf
            );
        }

        Ok(result)
    }

    /// Detecta gaps temporales superiores al timeframe esperado en un slice de klines
    pub fn detect_gaps(klines: &[Kline], tf: Timeframe) -> Vec<SeriesGap> {
        GapDetector::find_gaps(klines, tf)
    }

    /// Carga todas las velas y retorna también la lista de gaps detectados sin inventar velas
    pub fn load_all_with_gaps(conn: &Connection, symbol: &str, tf: Timeframe) -> Result<(Vec<Kline>, Vec<SeriesGap>)> {
        let klines = Self::load_all(conn, symbol, tf)?;
        let gaps = GapDetector::find_gaps(&klines, tf);
        Ok((klines, gaps))
    }

    /// Carga el rango de velas y retorna también la lista de gaps detectados sin inventar velas
    pub fn load_range_with_gaps(
        conn: &Connection,
        symbol: &str,
        tf: Timeframe,
        start_ts: i64,
        end_ts: i64,
    ) -> Result<(Vec<Kline>, Vec<SeriesGap>)> {
        let klines = Self::load_range(conn, symbol, tf, start_ts, end_ts)?;
        let gaps = GapDetector::find_gaps(&klines, tf);
        Ok((klines, gaps))
    }

    pub fn count_klines(conn: &Connection, symbol: &str, tf: Timeframe) -> Result<usize> {
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM klines WHERE symbol = ?1 AND timeframe = ?2")?;
        let count: i64 = stmt.query_row(params![symbol, tf.as_db_str()], |r| r.get(0))?;
        Ok(count as usize)
    }

    pub fn get_distinct_symbols(conn: &Connection) -> Result<Vec<String>> {
        let mut stmt = conn.prepare("SELECT DISTINCT symbol FROM klines ORDER BY symbol ASC")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut result = Vec::new();
        for r in rows {
            result.push(r?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{DbConfig, DbConnection};

    fn setup_memory_db() -> Connection {
        let cfg = DbConfig {
            db_path: std::path::PathBuf::from(":memory:"),
            wal_mode: false,
            busy_timeout: std::time::Duration::from_secs(5),
        };
        DbConnection::open(&cfg).expect("Error al inicializar DB en memoria")
    }

    #[test]
    fn test_bug_f08_multi_symbol_persistence_isolation() {
        let mut conn = setup_memory_db();
        let btc_candle = Kline {
            timestamp: 1_700_000_000,
            open: 65000.0,
            high: 66000.0,
            low: 64000.0,
            close: 65500.0,
            volume: 10.0,
        };
        let eth_candle = Kline {
            timestamp: 1_700_000_000, // Mismo timestamp exacto
            open: 3500.0,
            high: 3600.0,
            low: 3400.0,
            close: 3550.0,
            volume: 100.0,
        };

        // Insertar BTC
        let ins_btc = KlineRepository::insert_batch(&mut conn, "BTCUSDT", Timeframe::H1, &[btc_candle.clone()]).unwrap();
        assert_eq!(ins_btc, 1);

        // Insertar ETH con el mismo timestamp
        let ins_eth = KlineRepository::insert_batch(&mut conn, "ETHUSDT", Timeframe::H1, &[eth_candle.clone()]).unwrap();
        assert_eq!(ins_eth, 1);

        // Verificar que no se sobreescribieron
        let btc_loaded = KlineRepository::load_all(&conn, "BTCUSDT", Timeframe::H1).unwrap();
        let eth_loaded = KlineRepository::load_all(&conn, "ETHUSDT", Timeframe::H1).unwrap();

        assert_eq!(btc_loaded.len(), 1);
        assert_eq!(eth_loaded.len(), 1);
        assert_eq!(btc_loaded[0].close, 65500.0);
        assert_eq!(eth_loaded[0].close, 3550.0);

        let symbols = KlineRepository::get_distinct_symbols(&conn).unwrap();
        assert_eq!(symbols, vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()]);
    }

    #[test]
    fn test_c16_domain_validation_on_load() {
        let conn = setup_memory_db();
        // Insertar directamente en SQLite una vela con geometría OHLC anómala (High < Open)
        conn.execute(
            "INSERT INTO klines (symbol, timeframe, timestamp, open, high, low, close, volume)
             VALUES (?1, ?2, 1000, 105.0, 100.0, 90.0, 95.0, 10.0)",
            params!["BTCUSDT", Timeframe::H1.as_db_str()],
        ).unwrap();

        let load_res = KlineRepository::load_all(&conn, "BTCUSDT", Timeframe::H1);
        assert!(load_res.is_err());
    }

    #[test]
    fn test_c16_duplicate_timestamps_rejected_on_insert_and_load() {
        let mut conn = setup_memory_db();
        let k1 = Kline::new(1000, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();
        let k2 = Kline::new(1000, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();

        // Rechazado en insert_batch
        let ins_res = KlineRepository::insert_batch(&mut conn, "BTCUSDT", Timeframe::H1, &[k1, k2]);
        assert!(ins_res.is_err());
    }

    #[test]
    fn test_c16_disordered_timestamps_rejected() {
        let mut conn = setup_memory_db();
        let k1 = Kline::new(2000, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();
        let k2 = Kline::new(1000, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();

        let ins_res = KlineRepository::insert_batch(&mut conn, "BTCUSDT", Timeframe::H1, &[k1, k2]);
        assert!(ins_res.is_err());
    }

    #[test]
    fn test_c16_gap_detection_without_synthetic_candles() {
        let mut conn = setup_memory_db();
        let step = Timeframe::H1.duration_millis();
        let k1 = Kline::new(1_000_000, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();
        // Salto de 4 horas (faltan 3 velas)
        let k2 = Kline::new(1_000_000 + 4 * step, 100.0, 105.0, 95.0, 100.0, 10.0).unwrap();

        KlineRepository::insert_batch(&mut conn, "BTCUSDT", Timeframe::H1, &[k1.clone(), k2.clone()]).unwrap();

        let (loaded, gaps) = KlineRepository::load_all_with_gaps(&conn, "BTCUSDT", Timeframe::H1).unwrap();
        // NO rellena velas inventadas: tamaño exacto 2
        assert_eq!(loaded.len(), 2);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].missing_bars, 3);
        assert_eq!(gaps[0].start_timestamp, 1_000_000);
        assert_eq!(gaps[0].end_timestamp, 1_000_000 + 4 * step);
    }
}

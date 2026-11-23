use super::db_config::DbConfig;
use rusqlite::{Connection, Result};

pub struct DbConnection;

impl DbConnection {
    pub fn open(config: &DbConfig) -> Result<Connection> {
        let conn = Connection::open(&config.db_path)?;
        conn.busy_timeout(config.busy_timeout)?;

        if config.wal_mode {
            let _: String = conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
        }

        Self::init_tables(&conn)?;

        Ok(conn)
    }

    fn init_tables(conn: &Connection) -> Result<()> {
        // BUG-F08: Soporte multi-símbolo con clave primaria (symbol, timeframe, timestamp)
        // Verificar si la tabla klines ya existe y si contiene la columna 'symbol'
        let has_table: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='klines'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c > 0)
            .unwrap_or(false);

        if has_table {
            let mut stmt = conn.prepare("PRAGMA table_info(klines)")?;
            let mut rows = stmt.query([])?;
            let mut has_symbol = false;
            while let Some(row) = rows.next()? {
                let col_name: String = row.get(1)?;
                if col_name.eq_ignore_ascii_case("symbol") {
                    has_symbol = true;
                    break;
                }
            }

            if !has_symbol {
                // Migración segura: transferir datos preexistentes asignándoles 'BTCUSDT' por defecto
                conn.execute_batch(
                    "BEGIN TRANSACTION;
                     CREATE TABLE klines_migration (
                         symbol TEXT NOT NULL,
                         timeframe TEXT NOT NULL,
                         timestamp INTEGER NOT NULL,
                         open REAL NOT NULL,
                         high REAL NOT NULL,
                         low REAL NOT NULL,
                         close REAL NOT NULL,
                         volume REAL NOT NULL,
                         PRIMARY KEY (symbol, timeframe, timestamp)
                     );
                     INSERT OR IGNORE INTO klines_migration (symbol, timeframe, timestamp, open, high, low, close, volume)
                     SELECT 'BTCUSDT', timeframe, timestamp, open, high, low, close, volume FROM klines;
                     DROP TABLE klines;
                     ALTER TABLE klines_migration RENAME TO klines;
                     COMMIT;"
                )?;
            }
        } else {
            conn.execute(
                "CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )",
                [],
            )?;
        }

        // Eliminar índices obsoletos si existieran
        conn.execute("DROP INDEX IF EXISTS idx_klines_tf_ts", [])?;

        Ok(())
    }
}

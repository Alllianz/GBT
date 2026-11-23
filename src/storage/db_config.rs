use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct DbConfig {
    pub db_path: PathBuf,
    pub busy_timeout: Duration,
    pub wal_mode: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("candles.db"),
            busy_timeout: Duration::from_secs(30),
            wal_mode: true,
        }
    }
}

impl DbConfig {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            db_path: path.into(),
            ..Default::default()
        }
    }
}

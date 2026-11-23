use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EndpointPool {
    endpoints: Vec<&'static str>,
    current_index: Arc<AtomicUsize>,
}

impl Default for EndpointPool {
    fn default() -> Self {
        Self {
            endpoints: vec![
                "https://api.binance.com",
                "https://api1.binance.com",
                "https://api2.binance.com",
                "https://api3.binance.com",
                "https://api-gcp.binance.com",
            ],
            current_index: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl EndpointPool {
    pub fn new(endpoints: Vec<&'static str>) -> Self {
        Self {
            endpoints,
            current_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn current(&self) -> &'static str {
        if self.endpoints.is_empty() {
            return "";
        }
        let idx = self.current_index.load(Ordering::Relaxed) % self.endpoints.len();
        self.endpoints[idx]
    }

    /// BUG-20: Avance atómico y concurrente del cursor de endpoints
    pub fn next(&self) -> &'static str {
        if self.endpoints.is_empty() {
            return "";
        }
        let idx = self.current_index.fetch_add(1, Ordering::Relaxed) % self.endpoints.len();
        self.endpoints[idx]
    }

    pub fn all(&self) -> &[&'static str] {
        &self.endpoints
    }

    /// Retorna la lista de endpoints rotada a partir del índice atómico actual
    pub fn rotated_endpoints(&self) -> Vec<&'static str> {
        let n = self.endpoints.len();
        if n == 0 {
            return Vec::new();
        }
        let start = self.current_index.fetch_add(1, Ordering::Relaxed) % n;
        (0..n).map(|i| self.endpoints[(start + i) % n]).collect()
    }
}

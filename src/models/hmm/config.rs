use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HmmConfig {
    pub max_em_iters: usize,
    pub tol: f64,
    pub min_variance: f64,
    pub lookback_window: usize,
    pub refit_interval: usize,
}

impl Default for HmmConfig {
    fn default() -> Self {
        Self {
            max_em_iters: 50,
            tol: 1e-6,
            min_variance: 1e-6,
            lookback_window: 252,
            refit_interval: 20,
        }
    }
}

impl HmmConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_em_iters == 0 {
            return Err("max_em_iters must be > 0".to_string());
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err("tol must be finite and > 0".to_string());
        }
        if !self.min_variance.is_finite() || self.min_variance <= 0.0 {
            return Err("min_variance must be finite and > 0".to_string());
        }
        if self.lookback_window < 10 {
            return Err("lookback_window must be >= 10".to_string());
        }
        if self.refit_interval == 0 {
            return Err("refit_interval must be >= 1".to_string());
        }
        Ok(())
    }
}

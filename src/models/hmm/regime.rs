use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HmmRegime {
    Bear = 0,
    Sideways = 1,
    Bull = 2,
}

impl HmmRegime {
    #[inline]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Bear,
            1 => Self::Sideways,
            _ => Self::Bull,
        }
    }

    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimePosterior {
    pub probs: [f64; 3],
    pub dominant: HmmRegime,
    pub confidence: f64,
}

impl RegimePosterior {
    pub fn from_probs(probs: [f64; 3]) -> Self {
        let mut clean = [0.0; 3];
        let mut sum = 0.0;
        for i in 0..3 {
            let p = if probs[i].is_finite() && probs[i] > 0.0 {
                probs[i]
            } else {
                0.0
            };
            clean[i] = p;
            sum += p;
        }
        if sum > 0.0 {
            for p in &mut clean {
                *p /= sum;
            }
        } else {
            return Self::neutral();
        }

        let mut best_idx = 0;
        let mut best_prob = clean[0];
        for i in 1..3 {
            if clean[i] > best_prob {
                best_prob = clean[i];
                best_idx = i;
            }
        }

        Self {
            probs: clean,
            dominant: HmmRegime::from_index(best_idx),
            confidence: best_prob,
        }
    }

    pub fn neutral() -> Self {
        let third = 1.0 / 3.0;
        Self {
            probs: [third, third, third],
            dominant: HmmRegime::Sideways,
            confidence: third,
        }
    }
}

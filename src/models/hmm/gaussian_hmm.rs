use std::f64::consts::PI;

use super::config::HmmConfig;
use super::regime::RegimePosterior;

const N_STATES: usize = 3;
const N_DIMS: usize = 2;

#[derive(Debug, Clone)]
pub struct GaussianHmm {
    config: HmmConfig,
    pi: [f64; N_STATES],
    trans: [[f64; N_STATES]; N_STATES],
    means: [[f64; N_DIMS]; N_STATES],
    vars: [[f64; N_DIMS]; N_STATES],
    log_alpha: [f64; N_STATES],
    filter_initialized: bool,
    is_fitted: bool,
}

impl GaussianHmm {
    pub fn new(config: HmmConfig) -> Self {
        let min_var = config.min_variance.max(1e-12);
        Self {
            config,
            pi: [1.0 / 3.0; N_STATES],
            trans: [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            means: [
                [-0.01, 0.02],
                [0.0, 0.01],
                [0.01, 0.02],
            ],
            vars: [[min_var.max(1e-4); N_DIMS]; N_STATES],
            log_alpha: [0.0; N_STATES],
            filter_initialized: false,
            is_fitted: false,
        }
    }

    pub fn config(&self) -> HmmConfig {
        self.config
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    pub fn pi(&self) -> [f64; N_STATES] {
        self.pi
    }

    pub fn trans(&self) -> [[f64; N_STATES]; N_STATES] {
        self.trans
    }

    pub fn means(&self) -> [[f64; N_DIMS]; N_STATES] {
        self.means
    }

    pub fn vars(&self) -> [[f64; N_DIMS]; N_STATES] {
        self.vars
    }

    pub fn reset_filter(&mut self) {
        self.log_alpha = [0.0; N_STATES];
        self.filter_initialized = false;
    }

    pub fn fit_causal(&mut self, obs: &[[f64; N_DIMS]]) -> Result<(), String> {
        self.config.validate()?;
        let t_len = obs.len();
        if t_len < 6 {
            return Err("At least 6 observations required to fit 3-state GaussianHmm".to_string());
        }
        for (idx, o) in obs.iter().enumerate() {
            if !o[0].is_finite() || !o[1].is_finite() {
                return Err(format!("Non-finite observation at index {}", idx));
            }
        }

        self.init_deterministic(obs);

        let mut log_emit = vec![[0.0; N_STATES]; t_len];
        let mut log_alpha = vec![[0.0; N_STATES]; t_len];
        let mut log_beta = vec![[0.0; N_STATES]; t_len];
        let mut gamma = vec![[0.0; N_STATES]; t_len];
        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..self.config.max_em_iters {
            let mut log_pi = [0.0; N_STATES];
            let mut log_trans = [[0.0; N_STATES]; N_STATES];
            for i in 0..N_STATES {
                log_pi[i] = self.pi[i].max(1e-300).ln();
                for j in 0..N_STATES {
                    log_trans[i][j] = self.trans[i][j].max(1e-300).ln();
                }
            }

            for t in 0..t_len {
                for s in 0..N_STATES {
                    log_emit[t][s] = log_gaussian_diag(
                        obs[t],
                        self.means[s],
                        self.vars[s],
                        self.config.min_variance,
                    );
                }
            }

            for s in 0..N_STATES {
                log_alpha[0][s] = log_pi[s] + log_emit[0][s];
            }
            for t in 1..t_len {
                for j in 0..N_STATES {
                    let vals = [
                        log_alpha[t - 1][0] + log_trans[0][j],
                        log_alpha[t - 1][1] + log_trans[1][j],
                        log_alpha[t - 1][2] + log_trans[2][j],
                    ];
                    log_alpha[t][j] = log_sum_exp_3(vals) + log_emit[t][j];
                }
            }

            let log_likelihood = log_sum_exp_3(log_alpha[t_len - 1]);
            if !log_likelihood.is_finite() {
                return Err("Log-likelihood became non-finite during Baum-Welch".to_string());
            }

            log_beta[t_len - 1] = [0.0; N_STATES];
            for t in (0..t_len - 1).rev() {
                for i in 0..N_STATES {
                    let vals = [
                        log_trans[i][0] + log_emit[t + 1][0] + log_beta[t + 1][0],
                        log_trans[i][1] + log_emit[t + 1][1] + log_beta[t + 1][1],
                        log_trans[i][2] + log_emit[t + 1][2] + log_beta[t + 1][2],
                    ];
                    log_beta[t][i] = log_sum_exp_3(vals);
                }
            }

            for t in 0..t_len {
                let raw = [
                    log_alpha[t][0] + log_beta[t][0],
                    log_alpha[t][1] + log_beta[t][1],
                    log_alpha[t][2] + log_beta[t][2],
                ];
                let norm = log_sum_exp_3(raw);
                for s in 0..N_STATES {
                    gamma[t][s] = (raw[s] - norm).exp();
                }
            }

            let mut xi_sum = [[0.0; N_STATES]; N_STATES];
            for t in 0..t_len - 1 {
                let mut raw_xi = [0.0; 9];
                let mut idx = 0;
                for i in 0..N_STATES {
                    for j in 0..N_STATES {
                        raw_xi[idx] =
                            log_alpha[t][i] + log_trans[i][j] + log_emit[t + 1][j] + log_beta[t + 1][j];
                        idx += 1;
                    }
                }
                let norm_xi = log_sum_exp_9(raw_xi);
                let mut k = 0;
                for i in 0..N_STATES {
                    for j in 0..N_STATES {
                        xi_sum[i][j] += (raw_xi[k] - norm_xi).exp();
                        k += 1;
                    }
                }
            }

            let eps = 1e-12;
            let pi_sum: f64 = gamma[0].iter().sum::<f64>() + (N_STATES as f64) * eps;
            for i in 0..N_STATES {
                self.pi[i] = (gamma[0][i] + eps) / pi_sum;
            }

            for i in 0..N_STATES {
                let row_sum: f64 = xi_sum[i].iter().sum::<f64>() + (N_STATES as f64) * eps;
                for j in 0..N_STATES {
                    self.trans[i][j] = (xi_sum[i][j] + eps) / row_sum;
                }
            }

            for s in 0..N_STATES {
                let mut denom = 0.0;
                for t in 0..t_len {
                    denom += gamma[t][s];
                }
                if denom > 1e-12 {
                    for d in 0..N_DIMS {
                        let mut num_mean = 0.0;
                        for t in 0..t_len {
                            num_mean += gamma[t][s] * obs[t][d];
                        }
                        let new_mean = num_mean / denom;
                        let mut num_var = 0.0;
                        for t in 0..t_len {
                            let diff = obs[t][d] - new_mean;
                            num_var += gamma[t][s] * diff * diff;
                        }
                        self.means[s][d] = new_mean;
                        self.vars[s][d] = (num_var / denom).max(self.config.min_variance);
                    }
                }
            }

            if iter > 0 && (log_likelihood - prev_ll).abs() < self.config.tol {
                break;
            }
            prev_ll = log_likelihood;
        }

        self.sort_states_by_return();
        self.is_fitted = true;
        self.reset_filter();
        Ok(())
    }

    pub fn forward_step(&mut self, obs: [f64; N_DIMS]) -> RegimePosterior {
        if !self.is_fitted || !obs[0].is_finite() || !obs[1].is_finite() {
            return RegimePosterior::neutral();
        }

        let mut log_emit = [0.0; N_STATES];
        for j in 0..N_STATES {
            log_emit[j] = log_gaussian_diag(obs, self.means[j], self.vars[j], self.config.min_variance);
        }

        let mut raw = [0.0; N_STATES];
        if !self.filter_initialized {
            for j in 0..N_STATES {
                raw[j] = self.pi[j].max(1e-300).ln() + log_emit[j];
            }
            self.filter_initialized = true;
        } else {
            for j in 0..N_STATES {
                let pred_vals = [
                    self.log_alpha[0] + self.trans[0][j].max(1e-300).ln(),
                    self.log_alpha[1] + self.trans[1][j].max(1e-300).ln(),
                    self.log_alpha[2] + self.trans[2][j].max(1e-300).ln(),
                ];
                raw[j] = log_sum_exp_3(pred_vals) + log_emit[j];
            }
        }

        let log_norm = log_sum_exp_3(raw);
        if !log_norm.is_finite() {
            return RegimePosterior::neutral();
        }

        let mut probs = [0.0; N_STATES];
        for j in 0..N_STATES {
            self.log_alpha[j] = raw[j] - log_norm;
            probs[j] = self.log_alpha[j].exp();
        }

        RegimePosterior::from_probs(probs)
    }

    fn init_deterministic(&mut self, obs: &[[f64; N_DIMS]]) {
        let mut sorted: Vec<[f64; N_DIMS]> = obs.to_vec();
        sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));

        let n = sorted.len();
        let b1 = n / 3;
        let b2 = (2 * n) / 3;
        let bounds = [(0, b1.max(1)), (b1.max(1), b2.max(b1.max(1) + 1).min(n - 1)), (b2.max(b1.max(1) + 1).min(n - 1), n)];

        for (s, &(start, end)) in bounds.iter().enumerate() {
            let slice = &sorted[start..end];
            let count = slice.len() as f64;
            for d in 0..N_DIMS {
                let mean = slice.iter().map(|x| x[d]).sum::<f64>() / count;
                let var = slice
                    .iter()
                    .map(|x| {
                        let diff = x[d] - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / count;
                self.means[s][d] = mean;
                self.vars[s][d] = var.max(self.config.min_variance);
            }
        }

        self.pi = [1.0 / 3.0; N_STATES];
        self.trans = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ];
    }

    fn sort_states_by_return(&mut self) {
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| self.means[a][0].total_cmp(&self.means[b][0]).then(a.cmp(&b)));

        let old_pi = self.pi;
        let old_trans = self.trans;
        let old_means = self.means;
        let old_vars = self.vars;

        for new_i in 0..N_STATES {
            let old_i = order[new_i];
            self.pi[new_i] = old_pi[old_i];
            self.means[new_i] = old_means[old_i];
            self.vars[new_i] = old_vars[old_i];
            for new_j in 0..N_STATES {
                let old_j = order[new_j];
                self.trans[new_i][new_j] = old_trans[old_i][old_j];
            }
        }
    }
}

#[inline]
fn log_gaussian_diag(
    x: [f64; N_DIMS],
    mean: [f64; N_DIMS],
    var: [f64; N_DIMS],
    min_variance: f64,
) -> f64 {
    let log_2pi = (2.0 * PI).ln();
    let mut sum = 0.0;
    for d in 0..N_DIMS {
        let v = var[d].max(min_variance);
        let diff = x[d] - mean[d];
        sum += -0.5 * (log_2pi + v.ln() + (diff * diff) / v);
    }
    sum
}

#[inline]
fn log_sum_exp_3(vals: [f64; 3]) -> f64 {
    let m = vals[0].max(vals[1]).max(vals[2]);
    if !m.is_finite() {
        return m;
    }
    m + ((vals[0] - m).exp() + (vals[1] - m).exp() + (vals[2] - m).exp()).ln()
}

#[inline]
fn log_sum_exp_9(vals: [f64; 9]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut sum = 0.0;
    for &v in &vals {
        sum += (v - m).exp();
    }
    m + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hmm::HmmRegime;

    fn synthetic_regime_series() -> Vec<[f64; 2]> {
        let mut obs = Vec::with_capacity(90);
        for i in 0..30 {
            let jitter = ((i % 5) as f64 - 2.0) * 0.001;
            obs.push([-0.02 + jitter, 0.035 + (i % 3) as f64 * 0.002]);
        }
        for i in 0..30 {
            let jitter = ((i % 5) as f64 - 2.0) * 0.0005;
            obs.push([0.0005 + jitter, 0.008 + (i % 3) as f64 * 0.001]);
        }
        for i in 0..30 {
            let jitter = ((i % 5) as f64 - 2.0) * 0.001;
            obs.push([0.022 + jitter, 0.015 + (i % 3) as f64 * 0.0015]);
        }
        obs
    }

    fn assert_valid_simplex(post: &RegimePosterior) {
        let mut sum = 0.0;
        for &p in &post.probs {
            assert!(p.is_finite());
            assert!(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        assert!((sum - 1.0).abs() < 1e-12, "sum was {}", sum);
        assert!((post.confidence - post.probs[post.dominant.as_index()]).abs() < 1e-12);
    }

    #[test]
    fn test_causality_forward_step_no_lookahead() {
        let full_obs = synthetic_regime_series();
        let split_t = 55;

        let mut hmm_prefix = GaussianHmm::new(HmmConfig::default());
        hmm_prefix.fit_causal(&full_obs[..=split_t]).unwrap();

        let mut hmm_future_mutated = GaussianHmm::new(HmmConfig::default());
        let mut mutated = full_obs.clone();
        for item in &mut mutated[split_t + 1..] {
            *item = [0.99, 0.99];
        }
        hmm_future_mutated.fit_causal(&mutated[..=split_t]).unwrap();

        for t in 0..=split_t {
            let p1 = hmm_prefix.forward_step(full_obs[t]);
            let p2 = hmm_future_mutated.forward_step(mutated[t]);
            assert_valid_simplex(&p1);
            assert_valid_simplex(&p2);
            for s in 0..3 {
                assert!((p1.probs[s] - p2.probs[s]).abs() < 1e-12);
            }
            assert_eq!(p1.dominant, p2.dominant);
        }
    }

    #[test]
    fn test_state_ordering_and_regime_identification() {
        let obs = synthetic_regime_series();
        let mut hmm = GaussianHmm::new(HmmConfig::default());
        hmm.fit_causal(&obs).unwrap();

        let means = hmm.means();
        assert!(means[0][0] <= means[1][0]);
        assert!(means[1][0] <= means[2][0]);

        let bear_post = hmm.forward_step([-0.025, 0.036]);
        assert_valid_simplex(&bear_post);
        assert_eq!(bear_post.dominant, HmmRegime::Bear);

        hmm.reset_filter();
        let bull_post = hmm.forward_step([0.025, 0.015]);
        assert_valid_simplex(&bull_post);
        assert_eq!(bull_post.dominant, HmmRegime::Bull);
    }

    #[test]
    fn test_reset_filter_reproducibility() {
        let obs = synthetic_regime_series();
        let mut hmm = GaussianHmm::new(HmmConfig::default());
        hmm.fit_causal(&obs).unwrap();

        let mut pass1 = Vec::with_capacity(obs.len());
        for &o in &obs {
            let p = hmm.forward_step(o);
            assert_valid_simplex(&p);
            pass1.push(p);
        }

        hmm.reset_filter();
        for (idx, &o) in obs.iter().enumerate() {
            let p = hmm.forward_step(o);
            assert_valid_simplex(&p);
            for s in 0..3 {
                assert!((p.probs[s] - pass1[idx].probs[s]).abs() < 1e-12);
            }
        }
    }
}

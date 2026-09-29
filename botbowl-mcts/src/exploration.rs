//! Plan 048: exploration in self-play (plan 032 #4) — root Dirichlet noise and visit-proportional
//! sampling of the played move. Generation only; eval never sets either.
//!
//! **Root noise** mixes a Dirichlet draw into the root's priors:
//! `p'_i = (1 − ε)·p_i + ε·S·η_i`, `η ~ Dir(α/n, …, α/n)` over the `n` legal actions, where
//! `S = Σ p_i` keeps the total prior mass unchanged (NN priors are softmax × n, heuristic priors
//! are unnormalised — both stay on their own scale). α is the *total* concentration, so each root
//! gets `α/n` (KataGo style): plan 031 D4 found the root fan bimodal (median 6, p90 73), and no
//! single per-action α fits both modes.
//!
//! Purity: the noise is keyed to one state, the search's root, and drawn from a seed fixed for
//! that search — a per-search constant like the horizon anchor, so `available_actions` stays a
//! pure function of `(state, search)`. The root is the only node equal to its own state in an
//! acyclic DAG, so no other path can see a noisy prior. It takes effect only when a *fresh* tree
//! expands its root: a reused tree was built with an earlier search's dynamics and expanded this
//! node as a non-root long ago. [`RootNoise::applied`] says which happened.
//!
//! **Training targets see the clean priors.** The cq target is `softmax(ln prior + q/τ)`, so a
//! noisy prior recorded into the corpus would teach the net the noise. The root expansion keeps
//! the pre-noise priors and `MctsBot::get_action_explore` writes those into the sample.

use std::sync::OnceLock;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Action as EngineAction;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};

/// One search's root-noise parameters. `seed` is drawn by the caller per decision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootNoiseSpec {
    /// Mixing weight ε in `[0, 1]`.
    pub epsilon: f32,
    /// Total Dirichlet concentration α; each of the `n` root actions gets `α/n`.
    pub alpha: f32,
    pub seed: u64,
}

/// What the caller asks of one decision. `Default` is the plain greedy search.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExploreStep {
    pub noise: Option<RootNoiseSpec>,
    /// Play a root child with probability ∝ `visits^(1/temperature)` instead of the best-Q
    /// child. `u` is a uniform draw in `[0, 1)` supplied by the caller.
    pub sample: Option<SampleSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSpec {
    pub temperature: f32,
    pub u: f64,
}

/// The per-search noise, held by `BloodBowlDynamics`.
#[derive(Debug)]
pub struct RootNoise {
    root: GameState,
    spec: RootNoiseSpec,
    /// The root's pre-noise priors, set when (and only when) the root is expanded under this noise.
    clean: OnceLock<Vec<(EngineAction, f32)>>,
}

impl RootNoise {
    pub fn new(root: GameState, spec: RootNoiseSpec) -> Self {
        RootNoise {
            root,
            spec,
            clean: OnceLock::new(),
        }
    }

    /// Mix the noise into `priors` if `state` is this search's root; otherwise leave them alone.
    pub fn apply(&self, state: &GameState, actions: &[EngineAction], priors: &mut [f32]) {
        debug_assert_eq!(actions.len(), priors.len());
        if priors.is_empty() || *state != self.root {
            return;
        }
        let clean: Vec<(EngineAction, f32)> = actions.iter().copied().zip(priors.iter().copied()).collect();
        let eta = dirichlet(priors.len(), self.spec.alpha, self.spec.seed);
        let mass: f32 = priors.iter().sum();
        let eps = self.spec.epsilon.clamp(0.0, 1.0);
        for (p, e) in priors.iter_mut().zip(eta) {
            *p = (1.0 - eps) * *p + eps * mass * e;
        }
        // Concurrent expansion of the same root computes the same values; first write wins.
        let _ = self.clean.set(clean);
    }

    /// `true` once the root was expanded under this noise.
    pub fn applied(&self) -> bool {
        self.clean.get().is_some()
    }

    /// The pre-noise prior of a root action, if the noise was applied.
    pub fn clean_prior(&self, action: &EngineAction) -> Option<f32> {
        self.clean.get()?.iter().find(|(a, _)| a == action).map(|&(_, p)| p)
    }
}

/// A `Dir(α/n, …)` draw of length `n`, deterministic in `seed`.
pub fn dirichlet(n: usize, alpha_total: f32, seed: u64) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let shape = (alpha_total as f64 / n as f64).max(1e-3);
    let gamma = Gamma::new(shape, 1.0).expect("positive shape");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let draws: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = draws.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // Every draw underflowed (tiny shape): put the mass on one action, as the limit does.
        let mut v = vec![0.0; n];
        v[(seed % n as u64) as usize] = 1.0;
        return v;
    }
    draws.iter().map(|d| (d / sum) as f32).collect()
}

/// Index picked with probability ∝ `w_i^(1/temperature)`, or `None` when every weight is zero.
pub fn sample_index(weights: &[u32], temperature: f32, u: f64) -> Option<usize> {
    let inv_t = 1.0 / (temperature.max(1e-3) as f64);
    let scaled: Vec<f64> = weights.iter().map(|&w| (w as f64).powf(inv_t)).collect();
    let total: f64 = scaled.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let mut target = u.clamp(0.0, 1.0 - f64::EPSILON) * total;
    for (i, s) in scaled.iter().enumerate() {
        if *s > 0.0 && target < *s {
            return Some(i);
        }
        target -= s;
    }
    scaled.iter().rposition(|s| *s > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_is_a_distribution_and_deterministic() {
        for n in [1, 2, 6, 73] {
            let a = dirichlet(n, 10.0, 7);
            assert_eq!(a.len(), n);
            assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-4, "{n}: {a:?}");
            assert!(a.iter().all(|x| *x >= 0.0));
            assert_eq!(a, dirichlet(n, 10.0, 7));
        }
        assert_ne!(dirichlet(6, 10.0, 7), dirichlet(6, 10.0, 8));
    }

    #[test]
    fn small_alpha_concentrates_the_noise() {
        // α/n = 0.1 on a 100-wide root: most of the mass lands on a handful of actions.
        let a = dirichlet(100, 10.0, 3);
        let mut s = a.clone();
        s.sort_by(|x, y| y.total_cmp(x));
        assert!(s[..10].iter().sum::<f32>() > 0.6, "{:?}", &s[..10]);
    }

    #[test]
    fn sample_index_follows_the_weights() {
        let w = [0, 3, 1];
        assert_eq!(sample_index(&w, 1.0, 0.0), Some(1));
        assert_eq!(sample_index(&w, 1.0, 0.74), Some(1));
        assert_eq!(sample_index(&w, 1.0, 0.76), Some(2));
        assert_eq!(sample_index(&w, 1.0, 0.999), Some(2));
        assert_eq!(sample_index(&[0, 0], 1.0, 0.5), None);
        // A low temperature sharpens towards the most-visited child.
        assert_eq!(sample_index(&w, 0.1, 0.9), Some(1));
    }
}

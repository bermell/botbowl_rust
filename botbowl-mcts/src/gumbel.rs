//! Plan 053: Gumbel root search — sequential halving over the best few root moves.
//!
//! PUCT spends a small budget thinly over a wide fan: on 16x9 mid-turn roots (~98 legal moves) the
//! root decision did not converge below 4000 descents (exp057). Sequential halving (Danihelka et
//! al. 2022) takes the top `m` moves by `g + logit` (`g` Gumbel noise, 0 for deterministic play),
//! splits the budget evenly over them in `ceil(log2 m)` phases, keeps the better half after each by
//! `g + logit + σ(q̂)`, and plays the best survivor. Below the root the search stays PUCT.
//!
//! The search loop (`MctsBot::run_search`) runs the schedule: before each `tree.step()` it puts the
//! scheduled root move into a [`ForcedRoot`], and `select_node` takes it on the first player-node
//! selection of the descent, which is always the root. [`Halving`] is the schedule itself, a pure
//! function of the root children's stats, so it is tested here without a tree.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use botbowl_engine::core::model::Action as EngineAction;

/// σ(q̂) = (C_VISIT + max_b N(b)) · C_SCALE · q̂: the mctx defaults, and the same constants as
/// `prepare --policy-target gumbel`.
pub const C_VISIT: f32 = 50.0;
pub const C_SCALE: f32 = 0.1;

/// The root move the next descent must take. Set by the search loop before a step and taken by
/// `select_node` at the root; cleared after the step whether or not it was used (a fresh root's
/// first step expands it and selects nothing).
#[derive(Debug, Default)]
pub struct ForcedRoot {
    next: Mutex<Option<EngineAction>>,
    /// The named move was not on offer at the root (solved since, so hidden from selection).
    missed: AtomicBool,
}

impl ForcedRoot {
    pub fn set(&self, action: Option<EngineAction>) {
        *self.next.lock().unwrap() = action;
    }

    pub fn take(&self) -> Option<EngineAction> {
        self.next.lock().unwrap().take()
    }

    pub fn note_missed(&self) {
        self.missed.store(true, Ordering::Relaxed);
    }

    /// Whether the last named move was missed, clearing the flag.
    pub fn take_missed(&self) -> bool {
        self.missed.swap(false, Ordering::Relaxed)
    }
}

/// One root child as the halving sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct RootChild {
    pub action: EngineAction,
    /// `ln prior` (priors are softmax × n, so this is the logit up to a constant, which cancels).
    pub logit: f32,
    /// Q in the mover's frame (±1000 = a touchdown), `None` while unscored.
    pub q: Option<f32>,
    pub visits: u32,
    pub solved: bool,
}

/// The sequential-halving schedule for one search.
#[derive(Clone, Debug)]
pub struct Halving {
    /// Every considered move with its `g + logit`, best first.
    considered: Vec<(EngineAction, f32)>,
    /// The moves still in, a prefix-free subset of `considered`.
    survivors: Vec<EngineAction>,
    phases: usize,
    budget: usize,
    /// Smallest Q range (in Q points) the normalisation divides by; see [`Halving::new`].
    q_floor: f32,
}

impl Halving {
    /// The top `m` of `children` by `g + logit`, with `g` drawn from `seed` at `gumbel_scale`
    /// (0 = no noise: the top `m` by prior). `budget` is the descents the schedule may spend.
    ///
    /// `q_floor` is the smallest Q range the min-max normalisation of q̂ divides by. The paper's
    /// rule (0) maps any Q gap among the survivors, however small, onto the whole [0, 1], worth
    /// σ ≈ 10-40 against a prior gap of a few logits: halving then follows noise in Q. exp060
    /// measured exactly that (the reference's move, almost always the prior's favourite, was
    /// searched and then dropped in 40-46% of roots). A floor lets near-equal moves fall back on
    /// the prior.
    pub fn new(children: &[RootChild], m: usize, gumbel_scale: f32, seed: u64, budget: usize, q_floor: f32) -> Self {
        let mut rng = SplitMix64(seed);
        let mut scored: Vec<(EngineAction, f32)> = children
            .iter()
            .map(|c| {
                let g = if gumbel_scale > 0.0 {
                    gumbel_scale * rng.gumbel()
                } else {
                    0.0
                };
                (c.action, g + c.logit)
            })
            .collect();
        // Stable on ties, so equal priors keep the tree's child order.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(m.max(1));
        let survivors = scored.iter().map(|(a, _)| *a).collect::<Vec<_>>();
        let phases = (usize::BITS - (survivors.len().max(2) - 1).leading_zeros()) as usize; // ceil(log2)
        Halving {
            considered: scored,
            survivors,
            phases,
            budget,
            q_floor: q_floor.max(0.0),
        }
    }

    pub fn survivors(&self) -> &[EngineAction] {
        &self.survivors
    }

    /// Descents each survivor gets in the current phase.
    pub fn per_action(&self) -> usize {
        (self.budget / (self.phases * self.survivors.len().max(1))).max(1)
    }

    /// Keep the better half of the survivors (rounded up), while more than two remain.
    pub fn halve(&mut self, children: &[RootChild], root_q: f32) {
        if self.survivors.len() <= 2 {
            return;
        }
        let keep = self.survivors.len().div_ceil(2);
        let scores = self.scores(children, root_q);
        let mut ranked: Vec<(EngineAction, f32)> = self.survivors.iter().copied().zip(scores).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.survivors = ranked.into_iter().take(keep).map(|(a, _)| a).collect();
    }

    /// The move to play: the best survivor by `g + logit + σ(q̂)`.
    pub fn pick(&self, children: &[RootChild], root_q: f32) -> EngineAction {
        let scores = self.scores(children, root_q);
        self.survivors
            .iter()
            .zip(scores)
            .fold(None::<(EngineAction, f32)>, |best, (a, s)| match best {
                Some((_, bs)) if bs >= s => best,
                _ => Some((*a, s)),
            })
            .map(|(a, _)| a)
            .expect("a halving always has a survivor")
    }

    /// `g + logit + σ(q̂)` for each survivor, in survivor order. q̂ is the mover-frame Q min-max
    /// normalised over the survivors; an unscored child takes the root's Q, the same fill FPU uses.
    fn scores(&self, children: &[RootChild], root_q: f32) -> Vec<f32> {
        let find = |a: &EngineAction| children.iter().find(|c| c.action == *a);
        let q: Vec<f32> = self
            .survivors
            .iter()
            .map(|a| find(a).and_then(|c| c.q).unwrap_or(root_q))
            .collect();
        let lo = q.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_n = children.iter().map(|c| c.visits).max().unwrap_or(0) as f32;
        let sigma = (C_VISIT + max_n) * C_SCALE;
        self.survivors
            .iter()
            .zip(q)
            .map(|(a, q)| {
                let base = self
                    .considered
                    .iter()
                    .find(|(c, _)| c == a)
                    .map(|(_, s)| *s)
                    .unwrap_or(0.0);
                let range = (hi - lo).max(self.q_floor);
                let q_hat = if range > 1e-6 { (q - lo) / range } else { 0.5 };
                base + sigma * q_hat
            })
            .collect()
    }
}

/// Small, dependency-free generator for the Gumbel draws (the crate has no `rand`).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A standard Gumbel draw, `-ln(-ln u)` with `u` strictly inside (0, 1).
    fn gumbel(&mut self) -> f32 {
        let u = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-(-u.ln()).ln()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::model::Position;
    use botbowl_engine::core::table::PosAT;

    fn act(x: i32) -> EngineAction {
        EngineAction::Positional(PosAT::Move, Position::new((x as _, 1)))
    }

    fn child(x: i32, prior: f32, q: Option<f32>, visits: u32) -> RootChild {
        RootChild {
            action: act(x),
            logit: prior.ln(),
            q,
            visits,
            solved: false,
        }
    }

    #[test]
    fn without_noise_the_top_m_by_prior_are_considered() {
        let kids: Vec<_> = (0..10).map(|i| child(i, 1.0 + i as f32, None, 0)).collect();
        let h = Halving::new(&kids, 4, 0.0, 7, 400, 0.0);
        assert_eq!(h.survivors(), &[act(9), act(8), act(7), act(6)]);
        // Two phases for four moves: 400 / (2 · 4).
        assert_eq!(h.per_action(), 50);
    }

    #[test]
    fn halving_keeps_the_better_half_by_q_once_the_search_has_spoken() {
        let mut kids: Vec<_> = (0..4).map(|i| child(i, 1.0, None, 0)).collect();
        let mut h = Halving::new(&kids, 4, 0.0, 7, 400, 0.0);
        for (i, k) in kids.iter_mut().enumerate() {
            k.q = Some([100.0, -200.0, 600.0, 50.0][i]);
            k.visits = 50;
        }
        h.halve(&kids, 0.0);
        let mut s = h.survivors().to_vec();
        s.sort_by_key(|a| format!("{a:?}"));
        assert_eq!(s, vec![act(0), act(2)]);
        assert_eq!(h.pick(&kids, 0.0), act(2));
        // Two left: no further halving.
        h.halve(&kids, 0.0);
        assert_eq!(h.survivors().len(), 2);
    }

    #[test]
    fn a_strong_prior_needs_a_real_q_gap_to_lose() {
        // σ at 50 visits is (50 + 50) · 0.1 = 10 per unit of q̂, so a prior ratio of e^3 holds
        // against a q̂ gap of 0.25 (2.5) but not against the full range (10).
        let kids = vec![
            child(0, 20.0, Some(0.0), 50),
            child(1, 1.0, Some(100.0), 50),
            child(2, 1.0, Some(400.0), 50),
        ];
        let h = Halving::new(&kids, 3, 0.0, 7, 300, 0.0);
        let narrow: Vec<_> = vec![
            child(0, 20.0, Some(300.0), 50),
            child(1, 1.0, Some(0.0), 50),
            child(2, 1.0, Some(400.0), 50),
        ];
        assert_eq!(h.pick(&narrow, 0.0), act(0));
        assert_eq!(h.pick(&kids, 0.0), act(2));
    }

    #[test]
    fn a_q_floor_keeps_a_near_tie_on_the_prior() {
        // Q 524 vs 526: pure min-max calls that the whole range and the weaker prior wins it.
        let kids = vec![child(0, 20.0, Some(524.0), 50), child(1, 1.0, Some(526.0), 50)];
        let plain = Halving::new(&kids, 2, 0.0, 7, 100, 0.0);
        assert_eq!(plain.pick(&kids, 0.0), act(1));
        let floored = Halving::new(&kids, 2, 0.0, 7, 100, 100.0);
        assert_eq!(floored.pick(&kids, 0.0), act(0));
    }

    #[test]
    fn noise_is_seeded_and_changes_the_considered_set() {
        let kids: Vec<_> = (0..40).map(|i| child(i, 1.0, None, 0)).collect();
        let a = Halving::new(&kids, 8, 1.0, 1, 400, 0.0);
        let b = Halving::new(&kids, 8, 1.0, 1, 400, 0.0);
        let c = Halving::new(&kids, 8, 1.0, 2, 400, 0.0);
        assert_eq!(a.survivors(), b.survivors());
        assert_ne!(a.survivors(), c.survivors());
    }
}

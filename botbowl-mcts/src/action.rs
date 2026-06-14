use std::hash::{Hash, Hasher};

use botbowl_engine::core::model::Action as EngineAction;

/// The "player" each node belongs to from MCTS's perspective. The third
/// variant — `Chance` — is the node type whose children are stochastic
/// outcomes (modelled on the 2048 reference's `WaitingForRandom` state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BbPlayer {
    Home,
    Away,
    Chance,
}

/// A concrete roll outcome the dynamics can apply against a pending
/// `RequestedRoll`.
///
/// Pass/Fail covers the two-outcome rolls we model probabilistically
/// (D6PassFail, Sum2D6PassFail). `Advance` is the single-outcome
/// catch-all for rolls we don't enumerate (D8, Deviate, Scatter,
/// BlockDice, ThrowIn, ...): MCTS sees one chance child per such roll,
/// and `apply_action` resolves it by stepping the engine, which uses
/// the configured `DicePolicy` (or the RNG when no policy applies).
/// This keeps the tree branching bounded while still letting MCTS see
/// the post-roll state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChanceOutcome {
    /// The pending pass/fail roll passes. Applied by queuing a D6 of 6
    /// (or sum-2D6 of 12) — high enough to clear any target.
    Pass,
    /// The pending pass/fail roll fails. Applied by queuing a D6 of 1
    /// (or sum-2D6 of 2) — low enough to miss any non-trivial target.
    Fail,
    /// "Just let the engine resolve it." No dice fix is queued; the
    /// engine consumes the `pending_roll` via its dice policy / RNG on
    /// the next `micro_step(None)`. Used for non-pass/fail rolls.
    Advance,
}

/// MCTS-level action: either a game choice from the engine's
/// available_actions, or a chance outcome resolving a pending roll.
///
/// `prior_bits` on the `Player` variant caches the domain-knowledge
/// prior (`priors::prior_for`) at expansion time so `select_node`'s
/// PUCT descent doesn't recompute it per visit. The cache is excluded
/// from `Hash` / `Eq`: the prior is a pure function of the parent
/// state plus the engine action, so two `BbAction::Player`s with the
/// same engine action are interchangeable from MCTS's point of view
/// even if one carries a stale prior. (Sibling actions under the
/// same parent always have matching priors in practice — equality
/// only ever compares actions within one action list — but excluding
/// `prior_bits` keeps action identity meaning what it says.)
///
/// `prob_bits` on `Chance` is left in `Hash` / `Eq` so chance
/// outcomes with different probabilities (e.g. variant catches in
/// future scenarios) compare distinct. Tighten that to match the
/// `Player` policy if it ever bites.
#[derive(Debug, Clone)]
pub enum BbAction {
    Player { action: EngineAction, prior_bits: u32 },
    Chance { outcome: ChanceOutcome, prob_bits: u32 },
}

impl PartialEq for BbAction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BbAction::Player { action: a, .. }, BbAction::Player { action: b, .. }) => a == b,
            (
                BbAction::Chance {
                    outcome: a,
                    prob_bits: pa,
                },
                BbAction::Chance {
                    outcome: b,
                    prob_bits: pb,
                },
            ) => a == b && pa == pb,
            _ => false,
        }
    }
}

impl Eq for BbAction {}

impl Hash for BbAction {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self {
            BbAction::Player { action, .. } => {
                0u8.hash(h);
                action.hash(h);
            }
            BbAction::Chance { outcome, prob_bits } => {
                1u8.hash(h);
                outcome.hash(h);
                prob_bits.hash(h);
            }
        }
    }
}

impl BbAction {
    /// Constructor that stores the probability as IEEE-754 bits so the
    /// enum stays Hash + Eq (f32 is not Hash). Use `prob_f32()` to read.
    pub fn chance(outcome: ChanceOutcome, prob: f32) -> Self {
        BbAction::Chance {
            outcome,
            prob_bits: prob.to_bits(),
        }
    }

    /// Constructor for player actions; `prior` is cached as IEEE-754
    /// bits so the enum stays Hash + Eq. Use `prior_f32()` to read.
    /// Pass the value returned by `priors::prior_for` at expansion
    /// time — `select_node` reads it back without re-querying.
    pub fn player(action: EngineAction, prior: f32) -> Self {
        BbAction::Player {
            action,
            prior_bits: prior.to_bits(),
        }
    }

    pub fn prob_f32(&self) -> Option<f32> {
        match self {
            BbAction::Chance { prob_bits, .. } => Some(f32::from_bits(*prob_bits)),
            BbAction::Player { .. } => None,
        }
    }

    pub fn prior_f32(&self) -> Option<f32> {
        match self {
            BbAction::Player { prior_bits, .. } => Some(f32::from_bits(*prior_bits)),
            BbAction::Chance { .. } => None,
        }
    }
}

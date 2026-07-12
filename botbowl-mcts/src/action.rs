use std::hash::{Hash, Hasher};

use botbowl_engine::core::dices::RollResult;
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

/// MCTS-level action: either a game choice from the engine's
/// available_actions, or a chance outcome resolving a pending roll.
///
/// The `Chance` variant carries the concrete engine `RollResult` that
/// `apply_action` feeds straight into `SomeProcInput::Roll` — no
/// intermediate abstraction. `roll_outcomes::enumerate` decides which
/// results a given `RequestedRoll` fans out into (pass/fail rolls
/// branch into two weighted `RollResult::Pass`/`Fail` children; every
/// other roll type collapses to a single deterministic child), so the
/// stored result is a pure function of the parent's pending roll plus
/// the chosen branch — which is what keeps DAG recombination sound.
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
    Chance { result: RollResult, prob_bits: u32 },
}

impl PartialEq for BbAction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BbAction::Player { action: a, .. }, BbAction::Player { action: b, .. }) => a == b,
            (
                BbAction::Chance {
                    result: a,
                    prob_bits: pa,
                },
                BbAction::Chance {
                    result: b,
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
            BbAction::Chance { result, prob_bits } => {
                1u8.hash(h);
                result.hash(h);
                prob_bits.hash(h);
            }
        }
    }
}

impl BbAction {
    /// Constructor that stores the probability as IEEE-754 bits so the
    /// enum stays Hash + Eq (f32 is not Hash). Use `prob_f32()` to read.
    pub fn chance(result: RollResult, prob: f32) -> Self {
        BbAction::Chance {
            result,
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

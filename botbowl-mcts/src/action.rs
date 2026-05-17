use botbowl_engine::core::dices::RequestedRoll;
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
/// The MVP only needs Pass/Fail for D6PassFail and Sum2D6PassFail rolls,
/// which cover the Score TD Easy and Get-the-ball Easy/Medium scenarios.
/// Block dice / three-outcome / foul / scatter rolls will be added when
/// a lecture demands them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChanceOutcome {
    /// The pending pass/fail roll passes. Applied by queuing a D6 of 6
    /// (or sum-2D6 of 12) — high enough to clear any target.
    Pass,
    /// The pending pass/fail roll fails. Applied by queuing a D6 of 1
    /// (or sum-2D6 of 2) — low enough to miss any non-trivial target.
    Fail,
}

/// MCTS-level action: either a game choice from the engine's
/// available_actions, or a chance outcome resolving a pending roll.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BbAction {
    Player(EngineAction),
    Chance { outcome: ChanceOutcome, prob_bits: u32 },
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

    pub fn prob_f32(&self) -> Option<f32> {
        match self {
            BbAction::Chance { prob_bits, .. } => Some(f32::from_bits(*prob_bits)),
            BbAction::Player(_) => None,
        }
    }
}

/// Which roll kinds the MVP knows how to enumerate and apply. Anything
/// else triggers a `todo!` so we notice immediately if a lecture wanders
/// into uncharted territory.
pub fn is_supported(req: &RequestedRoll) -> bool {
    matches!(
        req,
        RequestedRoll::D6PassFail(_) | RequestedRoll::Sum2D6PassFail(_)
    )
}

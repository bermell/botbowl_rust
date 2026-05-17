use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::TeamType;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LectureStatus {
    InProgress,
    Success,
    Failure,
}

/// Snapshot of starting-state info that lectures need to recognise
/// "we have come back around to my next turn" without having to store it
/// in the lecture struct (lectures are shared across many trials).
#[derive(Debug, Clone, Copy)]
pub struct LectureContext {
    pub initial_home_turn: u8,
    pub initial_away_turn: u8,
    pub initial_half: u8,
}

impl LectureContext {
    pub fn from_state(state: &GameState) -> Self {
        Self {
            initial_home_turn: state.info.home_turn,
            initial_away_turn: state.info.away_turn,
            initial_half: state.info.half,
        }
    }
}

/// A curriculum scenario at a fixed difficulty.
///
/// Implementors describe how to spawn a starting `GameState`, which team
/// the agent under test controls, and how to read the running state to
/// decide whether the lecture has resolved.
pub trait Lecture {
    fn name(&self) -> &'static str;
    fn difficulty(&self) -> Difficulty;

    /// Build a fresh `GameState` for one trial. `rng` is the only source
    /// of randomness — using it consistently keeps trials reproducible
    /// from a seed.
    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState;

    /// Which team the agent under test plays. The runner routes the
    /// other team's actions to a default opponent bot.
    fn agent_team(&self) -> TeamType;

    /// Decide whether the lecture has resolved. Called after every
    /// `micro_step`. Returning `Success` or `Failure` ends the trial.
    /// `context` is the snapshot of starting-state info captured by the
    /// runner before any actions were taken; lectures use it to detect
    /// turn-cycle transitions (e.g. "the opponent's reply turn is over").
    fn evaluate(&self, state: &GameState, context: &LectureContext) -> LectureStatus;
}

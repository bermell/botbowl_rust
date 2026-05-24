use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Position, TeamType};
use botbowl_engine::core::table::Skill;
use rand::{Rng, RngCore};
use rand_chacha::ChaCha8Rng;

use crate::lecture::{Difficulty, Lecture, LectureContext, LectureStatus};

/// "Score TD" — Easy.
///
/// Single home lineman carrying the ball with no opponents on the pitch.
/// Per the grand plan: "start with ball, free path to end zone." Carrier x
/// is in [6..=9] so the run requires the full MA plus up to two GFIs to
/// reach the home endzone at x=1 in a single Move.
///
/// Random-agent baseline: ~5–10% success across 10k trials. The grand
/// plan's aspirational 1% target is *not* hit by this minimal "free path"
/// setup — the random agent's single decisive Move is too direct. Driving
/// the rate down further is a Medium-lecture concern (add an opponent to
/// dilute the action space).
pub struct ScoreTdEasy;

impl ScoreTdEasy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ScoreTdEasy {
    fn default() -> Self {
        Self::new()
    }
}

impl Lecture for ScoreTdEasy {
    fn name(&self) -> &'static str {
        "Score TD"
    }

    fn difficulty(&self) -> Difficulty {
        Difficulty::Easy
    }

    fn agent_team(&self) -> TeamType {
        TeamType::Home
    }

    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState {
        let carrier_x = rng.gen_range(6..=9);
        let carrier_y = rng.gen_range(3..=13);
        let carrier_pos = Position::new((carrier_x, carrier_y));

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .build();

        let carrier_id = state
            .get_player_id_at(carrier_pos)
            .expect("carrier should be at the seeded position after build");

        let bonus_skill = match rng.gen_range(0..3) {
            0 => None,
            1 => Some(Skill::SureHands),
            _ => Some(Skill::SureFeet),
        };
        if let Some(skill) = bonus_skill {
            state
                .get_mut_player(carrier_id)
                .expect("carrier id is valid")
                .stats
                .give_skill(skill);
        }

        // Enable in-engine RNG for any rolls the lecture doesn't pre-fix
        // (e.g. a GFI the agent decides to attempt). Seed it from the
        // lecture's own RNG so trials remain reproducible.
        state.set_seed(rng.next_u64());
        state.set_dice_mode(DiceMode::RollDice);

        state
    }

    fn evaluate(&self, state: &GameState, _context: &LectureContext) -> LectureStatus {
        evaluate_single_turn_td(state)
    }
}

/// "Score TD" — Medium.
///
/// Home carrier with one Away player standing between them and the endzone.
/// The carrier can still slip past with a dodge, or take the longer detour,
/// but a random agent's options are now diluted enough that the success
/// rate should drop sharply. The scripted bot is expected to plan a path
/// (going around or taking a single dodge) and score consistently.
pub struct ScoreTdMedium;

impl ScoreTdMedium {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ScoreTdMedium {
    fn default() -> Self {
        Self::new()
    }
}

impl Lecture for ScoreTdMedium {
    fn name(&self) -> &'static str {
        "Score TD"
    }

    fn difficulty(&self) -> Difficulty {
        Difficulty::Medium
    }

    fn agent_team(&self) -> TeamType {
        TeamType::Home
    }

    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState {
        // Same carrier distance range as Easy so the random baseline
        // comparison is apples-to-apples; the Medium difficulty comes
        // from the defensive presence, not from a longer run.
        let carrier_x = rng.gen_range(6..=9);
        let carrier_y = rng.gen_range(6..=10);
        let carrier_pos = Position::new((carrier_x, carrier_y));

        // Two adjacent Away defenders at x=3 form a partial wall in front
        // of the carrier. Their tackle zones overlap, so any path slipping
        // between them costs the agent at least one dodge.
        let blocker_y_top = carrier_y + rng.gen_range(-1..=0);
        let blocker_top = Position::new((3, blocker_y_top));
        let blocker_bottom = Position::new((3, blocker_y_top + 1));

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(carrier_pos)
            .add_away_player(blocker_top)
            .add_away_player(blocker_bottom)
            .add_ball_pos(carrier_pos)
            .build();

        state.set_seed(rng.next_u64());
        state.set_dice_mode(DiceMode::RollDice);

        state
    }

    fn evaluate(&self, state: &GameState, _context: &LectureContext) -> LectureStatus {
        evaluate_single_turn_td(state)
    }
}

/// Shared success/failure logic for "Score TD" lectures: the agent must
/// score on this turn or the trial counts as a failure.
fn evaluate_single_turn_td(state: &GameState) -> LectureStatus {
    if state.home.score > 0 {
        return LectureStatus::Success;
    }
    if state.info.game_over {
        return LectureStatus::Failure;
    }
    if state.info.team_turn != TeamType::Home {
        return LectureStatus::Failure;
    }
    LectureStatus::InProgress
}

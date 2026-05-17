use botbowl_engine::core::dices::{BlockDicePolicy, D6Target, DicePolicy, Sum2D6Target};
use botbowl_engine::core::gamestate::{BuilderState, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BallState, Position, TeamType};
use rand::{Rng, RngCore};
use rand_chacha::ChaCha8Rng;

use crate::lecture::{Difficulty, Lecture, LectureContext, LectureStatus};

/// Pickup AG3 baseline is 3+, so the "3+ succeeds, 4+ fails" policy from
/// the grand plan maps cleanly onto this lecture: an unmarked pickup
/// succeeds automatically; any modifier (e.g. a marking opponent's
/// tackle zone) pushes the target to 4+ and the policy fails it. This
/// is the whole point of stochasticity control here — the lecture
/// evaluates *strategy* (clear the marker first) rather than dice luck.
fn three_plus_policy() -> DicePolicy {
    DicePolicy::SucceedAtOrEasier {
        d6: D6Target::ThreePlus,
        sum2d6: Sum2D6Target::SevenPlus,
        block_dice: BlockDicePolicy::Default,
    }
}

/// Same pickup/dodge thresholds as `three_plus_policy`, plus block-dice
/// determinism: a 2+ attacker-dice block returns all Pow so the attacker
/// chooses a knockdown. Use this in lectures that hinge on a 2-dice
/// block landing reliably.
fn three_plus_with_knockdown_policy() -> DicePolicy {
    DicePolicy::SucceedAtOrEasier {
        d6: D6Target::ThreePlus,
        sum2d6: Sum2D6Target::SevenPlus,
        block_dice: BlockDicePolicy::KnockdownAtAdvantage,
    }
}

/// "Get the ball" — Easy.
///
/// Ball on the ground in midfield, no marking opponent, one distant Away
/// player so the opponent's reply turn isn't degenerate. The agent's
/// task: pick up the ball and still hold it after the opponent's turn.
pub struct GetTheBallEasy;

impl GetTheBallEasy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetTheBallEasy {
    fn default() -> Self {
        Self::new()
    }
}

impl Lecture for GetTheBallEasy {
    fn name(&self) -> &'static str {
        "Get the ball"
    }

    fn difficulty(&self) -> Difficulty {
        Difficulty::Easy
    }

    fn agent_team(&self) -> TeamType {
        TeamType::Home
    }

    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState {
        let carrier_y = rng.gen_range(5..=11);
        let carrier_pos = Position::new((15, carrier_y));
        let ball_pos = Position::new((13, carrier_y));
        // Lone Away player parked deep on their own half. They exist so
        // the engine has someone to ask for actions during the reply turn;
        // they can't reach the carrier in one turn.
        let away_pos = Position::new((25, 8));

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(carrier_pos)
            .add_away_player(away_pos)
            .add_ball_pos(ball_pos)
            .build();

        state.dice_policy = three_plus_policy();
        state.set_seed(rng.next_u64());
        state.rng_enabled = true;

        state
    }

    fn evaluate(&self, state: &GameState, context: &LectureContext) -> LectureStatus {
        evaluate_ball_acquisition(state, context, TeamType::Home)
    }
}

/// "Get the ball" — Medium.
///
/// Same midfield setup as Easy, but the ball is marked by an adjacent
/// Away defender. With the "3+ succeeds, 4+ fails" policy the pickup
/// auto-fails while the marker stands — the agent must displace them
/// (a block / blitz) before attempting the pickup.
pub struct GetTheBallMedium;

impl GetTheBallMedium {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetTheBallMedium {
    fn default() -> Self {
        Self::new()
    }
}

impl Lecture for GetTheBallMedium {
    fn name(&self) -> &'static str {
        "Get the ball"
    }

    fn difficulty(&self) -> Difficulty {
        Difficulty::Medium
    }

    fn agent_team(&self) -> TeamType {
        TeamType::Home
    }

    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState {
        // The geometry must support a 2-dice block on the marker so the
        // scripted bot's "safe block" planner picks it up. Layout:
        //
        //          col 9   10   11
        //   row y          ball picker
        //   row y+1 assist mark blocker
        //
        // The blocker at (11, y+1) attacking the marker at (10, y+1) gets
        // assists from the assistant at (9, y+1) and the picker at (11, y)
        // (both adjacent to the marker, neither marked by anyone other
        // than the marker themselves). With 3-str home vs 3-str away and
        // two assists, the block resolves at 2 dice in the attacker's
        // favour.
        let center_y = rng.gen_range(6..=10);
        let ball_pos = Position::new((10, center_y));
        let marker_pos = Position::new((10, center_y + 1));
        let picker_pos = Position::new((11, center_y));
        let blocker_pos = Position::new((11, center_y + 1));
        let assistant_pos = Position::new((9, center_y + 1));
        let distant_away = Position::new((25, 8));

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(picker_pos)
            .add_home_player(blocker_pos)
            .add_home_player(assistant_pos)
            .add_away_player(marker_pos)
            .add_away_player(distant_away)
            .add_ball_pos(ball_pos)
            .build();

        state.dice_policy = three_plus_policy();
        state.set_seed(rng.next_u64());
        state.rng_enabled = true;

        state
    }

    fn evaluate(&self, state: &GameState, context: &LectureContext) -> LectureStatus {
        evaluate_ball_acquisition(state, context, TeamType::Home)
    }
}

/// "Get the ball" — Hard.
///
/// The Away team carries the ball. The agent must:
///   1. Land a 2-dice block on the carrier (the policy turns 2+ attacker
///      dice into a knockdown).
///   2. Pick up the ball after it bounces off the downed carrier.
///   3. Survive the opponent's reply turn without losing the ball again.
///
/// Geometry: blocker + assistant adjacent to the carrier (for the 2-die
/// block), picker one square off so they aren't burned by the block and
/// stay fresh for the pickup.
pub struct GetTheBallHard;

impl GetTheBallHard {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GetTheBallHard {
    fn default() -> Self {
        Self::new()
    }
}

impl Lecture for GetTheBallHard {
    fn name(&self) -> &'static str {
        "Get the ball"
    }

    fn difficulty(&self) -> Difficulty {
        Difficulty::Hard
    }

    fn agent_team(&self) -> TeamType {
        TeamType::Home
    }

    fn setup(&self, rng: &mut ChaCha8Rng) -> GameState {
        let center_y = rng.gen_range(7..=9);
        let carrier_pos = Position::new((10, center_y));
        // Blocker is adjacent but ALONE — a straight block is 3-vs-3, only
        // 1 die. The agent must engineer a 2-die block by blitzing in a
        // second attacker (picker) to a square adjacent to the carrier,
        // using the blocker as an assist.
        let blocker_pos = Position::new((11, center_y));
        let picker_pos = Position::new((8, center_y));
        let distant_away = Position::new((25, 8));

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(blocker_pos)
            .add_home_player(picker_pos)
            .add_away_player(carrier_pos)
            .add_away_player(distant_away)
            .add_ball_pos(carrier_pos)
            .build();

        state.dice_policy = three_plus_with_knockdown_policy();
        state.set_seed(rng.next_u64());
        state.rng_enabled = true;

        state
    }

    fn evaluate(&self, state: &GameState, context: &LectureContext) -> LectureStatus {
        evaluate_ball_acquisition(state, context, TeamType::Home)
    }
}

/// Shared evaluator: the lecture resolves once we've cycled past the
/// agent's starting turn into their *next* turn (i.e. the opponent has
/// had their reply turn). At that point, success = the agent's team
/// holds the ball; failure otherwise.
///
/// We also treat `game_over` as a terminator — protects against weird
/// edge cases like a touchback OOB or the half ending.
fn evaluate_ball_acquisition(
    state: &GameState,
    context: &LectureContext,
    agent: TeamType,
) -> LectureStatus {
    let cycled =
        state.info.home_turn > context.initial_home_turn && state.info.half == context.initial_half;
    if cycled || state.info.game_over {
        return if home_holds_ball(state, agent) {
            LectureStatus::Success
        } else {
            LectureStatus::Failure
        };
    }
    LectureStatus::InProgress
}

fn home_holds_ball(state: &GameState, team: TeamType) -> bool {
    match state.ball {
        BallState::Carried(id) => state
            .get_player(id)
            .map(|p| p.stats.team == team)
            .unwrap_or(false),
        _ => false,
    }
}

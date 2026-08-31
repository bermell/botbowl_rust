//! A search whose whole reachable space fits in the budget must mark the
//! tree solved and return early — not idle out the rest of its wall-clock
//! budget re-descending exhausted subtrees. Uses a runtime-shrunk board so
//! the single-carrier turn tree is small enough to exhaust in debug builds.

use std::time::{Duration, Instant};

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameStateBuilder;
use botbowl_engine::core::model::{Action, BoardDims, Position, TEAM_SIZE};
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::{MctsBot, PuctMode, SearchBudget};

#[test]
fn solved_search_returns_well_before_the_time_budget() {
    run_solved_early_stop(PuctMode::raw());
}

/// Same scenario under the normalised selection rule (plan 026). A cheap,
/// fast canary that the new mode still converges an exhaustible tree and still
/// terminates — run this before spending the benchmark suite's wall clock on a
/// `c` sweep. A `c` far too high shows up here first.
#[test]
fn solved_search_returns_early_under_normalised_puct() {
    run_solved_early_stop(PuctMode::normalised(1.0));
}

fn run_solved_early_stop(puct: PuctMode) {
    // 16x9 engine board (14x7 playable), same small tier as the engine's
    // runtime-dims tests. Skip if compiled capacity is smaller.
    const W: i8 = 16;
    const H: i8 = 9;
    const PLAYERS: usize = 3;
    if (botbowl_engine::core::model::WIDTH as i8) < W
        || (botbowl_engine::core::model::HEIGHT as i8) < H
        || TEAM_SIZE < PLAYERS
    {
        return;
    }

    // Single home carrier two squares from the endzone (x=1), empty
    // pitch: every line ends inside this turn (TD, turnover, or end of
    // turn), so the whole tree is exhaustible.
    let carrier_pos = Position::new((3, 4));
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W, H, PLAYERS))
        .add_home_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .build();
    state.set_seed(0);
    state.set_dice_mode(botbowl_engine::core::gamestate::DiceMode::RollDice);

    const BUDGET: Duration = Duration::from_secs(5);
    let mut bot = MctsBot::new(SearchBudget::Time(BUDGET)).with_workers(1).with_puct(puct);

    // Drive the turn: activation first, then the move. Every search over
    // this position exhausts the tree, so each get_action must come back
    // in a fraction of the budget — without the early stop each call is
    // pinned at ~5 s. Generous ceiling (60%) for slow machines.
    let mut actions_taken = Vec::new();
    for _ in 0..4 {
        if state.home.score > 0 {
            break;
        }
        let t0 = Instant::now();
        let action = bot.get_action(&state);
        let elapsed = t0.elapsed();
        assert!(
            elapsed < BUDGET.mul_f32(0.6),
            "get_action took {elapsed:?} — solved search failed to stop early (budget {BUDGET:?})"
        );
        actions_taken.push(action);
        state.step(action).unwrap();
    }

    // The exhaustive answer is a touchdown: activate the carrier, then
    // Move onto the endzone column (Q-based pick over a solved tree =
    // exact max).
    assert_eq!(
        state.home.score, 1,
        "the solved search must convert the touchdown; actions: {actions_taken:?}"
    );
    assert!(
        actions_taken
            .iter()
            .any(|a| matches!(a, Action::Positional(PosAT::Move, pos) if pos.x == 1)),
        "expected a Move onto the endzone column, got {actions_taken:?}"
    );
}

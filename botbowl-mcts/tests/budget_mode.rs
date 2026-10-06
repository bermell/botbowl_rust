//! `BudgetMode`: what an `Iterations(n)` budget counts.
//!
//! `Iterations` runs `n` fresh descents every decision, on top of whatever a reused tree already
//! holds, so a root that inherited a big subtree ends up searched far beyond `n`. `Visits` grows
//! the root to `n` visits instead (KataGo's `maxVisits` vs `maxPlayouts`): only the shortfall is
//! searched, so every decision ends with a tree of about the same size.

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, Position, TEAM_SIZE};
use botbowl_mcts::{BudgetMode, MctsBot, ReuseOutcome, SearchBudget};

const W: i8 = 16;
const H: i8 = 9;
const PLAYERS: usize = 3;
const N: usize = 300;

/// Three a side and a carrier in midfield: wide enough that a few hundred iterations never solve
/// the turn, so every decision spends its whole budget.
fn open_state() -> Option<GameState> {
    if (botbowl_engine::core::model::WIDTH as i8) < W
        || (botbowl_engine::core::model::HEIGHT as i8) < H
        || TEAM_SIZE < PLAYERS
    {
        return None;
    }
    let carrier = Position::new((5, 4));
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W, H, PLAYERS))
        .add_home_player(carrier)
        .add_home_player(Position::new((4, 2)))
        .add_home_player(Position::new((4, 6)))
        .add_away_player(Position::new((11, 3)))
        .add_away_player(Position::new((11, 5)))
        .add_away_player(Position::new((13, 4)))
        .add_ball_pos(carrier)
        .build();
    state.set_seed(7);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    Some(state)
}

struct Decision {
    reuse: ReuseOutcome,
    iterations: u64,
    root_visits: u32,
    solved: bool,
}

fn drive(mode: BudgetMode, moves: usize) -> Vec<Decision> {
    let Some(mut state) = open_state() else {
        return Vec::new();
    };
    let mut bot = MctsBot::new(SearchBudget::Iterations(N))
        .with_workers(1)
        .with_budget_mode(mode);
    let mut out = Vec::new();
    for _ in 0..moves {
        if state.info.game_over {
            break;
        }
        let before = bot.telemetry().iterations;
        let action = bot.get_action(&state);
        // A kickoff setup after a score is answered from a formation, without a search (plan
        // 047); only searched decisions are the subject here.
        let Some(s) = bot.last_search() else {
            state.step(action).unwrap();
            continue;
        };
        out.push(Decision {
            reuse: s.reuse.outcome,
            iterations: bot.telemetry().iterations - before,
            root_visits: s.root.visits,
            solved: s.root.solved,
        });
        state.step(action).unwrap();
    }
    out
}

#[test]
fn iterations_mode_runs_the_full_budget_every_decision() {
    let ds = drive(BudgetMode::Iterations, 6);
    if ds.is_empty() {
        return; // board too small for this build
    }
    for d in ds.iter().filter(|d| !d.solved) {
        assert_eq!(d.iterations, N as u64);
    }
}

#[test]
fn visits_mode_stops_once_the_root_has_enough_visits() {
    let ds = drive(BudgetMode::Visits, 6);
    if ds.is_empty() {
        return;
    }
    assert!(
        ds.iter().any(|d| d.reuse == ReuseOutcome::Reused),
        "the scenario must re-root at least once to test anything"
    );
    for d in ds.iter().filter(|d| !d.solved) {
        assert!(d.iterations <= N as u64, "never more than the budget: {}", d.iterations);
        assert!(
            d.root_visits as usize >= N || d.iterations == N as u64,
            "stopped early without reaching the target: {} visits after {} iterations",
            d.root_visits,
            d.iterations
        );
        if d.reuse != ReuseOutcome::Reused {
            // A descent does not always land a visit on the root of the recombining DAG, so a
            // fresh tree never reaches N visits within N descents: it costs what Iterations does.
            assert_eq!(d.iterations, N as u64, "a fresh tree runs the whole budget");
        }
    }
    let total: u64 = ds.iter().map(|d| d.iterations).sum();
    assert!(
        total < (N * ds.len()) as u64,
        "a reused subtree's visits must count toward the target: {total} iterations over {} decisions",
        ds.len()
    );
}

/// The default must not move: `Iterations` is what every corpus and benchmark so far was made with.
#[test]
fn the_shipped_config_counts_iterations() {
    assert_eq!(botbowl_mcts::MctsConfig::new().budget_mode, BudgetMode::Iterations);
}

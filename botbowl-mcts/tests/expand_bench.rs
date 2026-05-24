//! Microbench for MCTS leaf-expansion cost (plan 011 step 1).
//!
//! Three sub-measurements timed against two start states:
//!   (a) `MctsBot::get_action` wall-clock — full search incl. select/score/expand.
//!   (b) `GameState::clone` raw cost — the per-child clone in `make_branch`.
//!   (c) `BloodBowlDynamics::apply_action(StartMove)` — clone + micro_step,
//!       including `PathFinder::player_paths` for the activated player.
//!
//! Both states are seeded with `0xCAFE_1234` (same seed as
//! `score_td_easy.rs` / `parallel_bench.rs`) so the numbers are
//! reproducible and comparable across the bench files.
//!
//! Run:
//!   cargo test --release -p botbowl-mcts --test expand_bench \
//!       -- --ignored --nocapture

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::Lecture;
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::{BbAction, BloodBowlDynamics, MctsBot};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use recon_mcts::GameDynamics;
use std::time::{Duration, Instant};

const SEED: u64 = 0xCAFE_1234;

fn build_score_td_easy() -> GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    ScoreTdEasy::new().setup(&mut rng)
}

fn build_full_teams_turn_start() -> GameState {
    // 11 home + 11 away in a plausible post-kickoff formation, ball with
    // a home carrier at (13, 8). `GameStateBuilder::build` runs the
    // kickoff sequence then drops these players onto the pitch and
    // hands turn 1 to Home. Goal is the 30+-legal-actions fan-out plan
    // 011 targets — every fielded home player is a StartMove candidate,
    // plus StartBlitz / StartHandoff / StartPass / StartFoul / EndTurn.
    let home_players = [
        (13, 8),
        (13, 7),
        (13, 9),
        (13, 5),
        (13, 11),
        (13, 3),
        (13, 13),
        (10, 8),
        (10, 6),
        (10, 10),
        (7, 8),
    ];
    let away_players = [
        (14, 8),
        (14, 7),
        (14, 9),
        (14, 5),
        (14, 11),
        (14, 3),
        (14, 13),
        (17, 8),
        (17, 6),
        (17, 10),
        (20, 8),
    ];
    let mut state = GameStateBuilder::new()
        .set_state(BuilderState::Turn { turn: 1 })
        .add_home_players(&home_players)
        .add_away_players(&away_players)
        .add_ball((13, 8))
        .build();
    state.set_seed(SEED);
    state.set_dice_mode(DiceMode::RollDice);
    state
}

fn prep_for_search(state: &mut GameState) {
    // Mirror what `MctsBot::get_action` does up front (dynamics.rs:499)
    // so we don't over-attribute to log-Vec cloning during the bench.
    state.set_logging_state(false);
    state.clear_log();
}

fn first_start_move(state: &GameState) -> Option<EngineAction> {
    state
        .available_actions
        .get_all()
        .into_iter()
        .find(|a| matches!(a, EngineAction::Positional(PosAT::StartMove, _)))
}

fn bench_get_action(label: &str, state: &GameState, trials: usize, iters: usize) {
    let mut total = Duration::ZERO;
    let mut sink: u64 = 0;
    for _ in 0..trials {
        let mut bot = MctsBot::new(iters);
        let t0 = Instant::now();
        let action = bot.get_action(state);
        total += t0.elapsed();
        sink = sink.wrapping_add(format!("{action:?}").len() as u64);
    }
    let per_trial = total.as_secs_f64() / trials as f64;
    let per_iter_us = per_trial * 1e6 / iters as f64;
    eprintln!(
        "EXPAND_BENCH {label}/get_action_ms={:.2} trials={trials} iters={iters} \
         per_iter_us={:.2} sink={sink}",
        per_trial * 1e3,
        per_iter_us,
    );
}

fn bench_clone(label: &str, state: &GameState, iters: usize) {
    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..iters {
        let c = state.clone();
        sink = sink.wrapping_add(c.info.home_turn as u64);
        std::hint::black_box(c);
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "EXPAND_BENCH {label}/clone_ns={:.0} iters={iters} sink={sink}",
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn bench_apply_start_move(label: &str, state: &GameState, action: EngineAction, iters: usize) {
    // Match `MctsBot::get_action`'s dice mode so we exercise the same
    // code path the tree actually traverses.
    let mut base = state.clone();
    base.set_dice_mode(DiceMode::RegisterRolls);

    let dynamics = BloodBowlDynamics::default();
    let bb_action = BbAction::Player(action);
    let t0 = Instant::now();
    let mut applied: u64 = 0;
    for _ in 0..iters {
        let s = base.clone();
        let out = dynamics.apply_action(s, &bb_action);
        if out.is_some() {
            applied += 1;
        }
        std::hint::black_box(out);
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "EXPAND_BENCH {label}/apply_start_move_ns={:.0} iters={iters} applied={applied}",
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn run_scenario(label: &str, mut state: GameState) {
    prep_for_search(&mut state);
    let legal = state.available_actions.get_all().len();
    let team = state.available_actions.team;
    eprintln!(
        "EXPAND_BENCH {label}/legal_actions={legal} team={:?}",
        team
    );

    bench_get_action(label, &state, 20, 1000);
    bench_clone(label, &state, 100_000);
    match first_start_move(&state) {
        Some(a) => bench_apply_start_move(label, &state, a, 10_000),
        None => eprintln!("EXPAND_BENCH {label}/apply_start_move=SKIPPED (no StartMove legal)"),
    }
}

#[test]
#[ignore = "manual wall-clock bench — run with --ignored"]
fn expand_bench_main() {
    eprintln!("EXPAND_BENCH seed={SEED:#x}");
    run_scenario("score_td_easy", build_score_td_easy());
    run_scenario("full_teams", build_full_teams_turn_start());
}

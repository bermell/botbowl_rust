//! A decision with exactly one action left after pruning is played without a search.
//!
//! The search root's legal set is the engine's actions minus `pruning::should_prune` (all of them
//! when pruning would leave none). When that set has one action, every `MctsBot` entry point —
//! `get_action`, `get_action_with_record`, `get_action_explore`, under PUCT and under the Gumbel
//! root — returns it at once: no tree, no descent, no network forward. The training record keeps
//! one sample per decision (replay rebuilds a trajectory from every `chosen_action`), with the one
//! surviving action as its only child, so `prepare` can tell it apart and skip it.

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action as EA, Position};
use botbowl_engine::core::table::{PosAT, SimpleAT};
use botbowl_mcts::pruning::should_prune;
use botbowl_mcts::{ExploreStep, MctsBot, MctsConfig, RootNoiseSpec, SampleSpec, SearchBudget};
use botbowl_nn::eval::{profile_counters, NnEvaluator};

/// `profile_counters()` is process-global; the tests in this binary that read it run one at a time.
static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx")
}

/// A Home player activated with `StartMove` who has made his first move. The engine still offers
/// every reachable square plus `EndPlayerTurn`; pruning (P8: a settled mover's further moves)
/// leaves `EndPlayerTurn` alone.
fn forced_state() -> GameState {
    let pos = Position::new((5, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(pos)
        .add_home_player(Position::new((3, 3)))
        .add_away_player(Position::new((20, 10)))
        .build();
    state.step(EA::Positional(PosAT::StartMove, pos)).unwrap();
    state.step(EA::Positional(PosAT::Move, Position::new((6, 5)))).unwrap();
    let legal = state.get_all_actions();
    let surviving: Vec<EA> = legal.iter().copied().filter(|a| !should_prune(&state, a)).collect();
    assert!(legal.len() > 2, "several engine-legal actions: {legal:?}");
    assert_eq!(
        surviving,
        vec![EA::Simple(SimpleAT::EndPlayerTurn)],
        "one survives pruning"
    );
    state
}

/// The same team at the start of its turn: a real choice (start any of two players, end turn).
fn open_state() -> GameState {
    let state = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_home_player(Position::new((3, 3)))
        .add_away_player(Position::new((20, 10)))
        .build();
    let surviving = state
        .get_all_actions()
        .into_iter()
        .filter(|a| !should_prune(&state, a))
        .count();
    assert!(surviving >= 2);
    state
}

fn config(gumbel_m: u16) -> MctsConfig {
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.gumbel_m = gumbel_m;
    cfg.gumbel_scale = if gumbel_m > 0 { 1.0 } else { 0.0 };
    cfg
}

fn heuristic_bot(gumbel_m: u16) -> MctsBot {
    MctsBot::with_budget_and_config(SearchBudget::Iterations(64), config(gumbel_m))
}

fn nn_bot(gumbel_m: u16) -> MctsBot {
    let nn = Arc::new(NnEvaluator::from_path(fixture()).expect("load tiny.onnx"));
    heuristic_bot(gumbel_m).with_evaluator(nn)
}

fn explore_step() -> ExploreStep {
    ExploreStep {
        noise: Some(RootNoiseSpec {
            epsilon: 0.5,
            alpha: 10.0,
            seed: 3,
        }),
        sample: Some(SampleSpec {
            temperature: 1.0,
            u: 0.7,
        }),
    }
}

/// Nothing was searched: no decision in the telemetry, no descent, no tree, no summary.
fn assert_unsearched(bot: &MctsBot, what: &str) {
    assert_eq!(bot.telemetry().searches, 0, "{what}: no search recorded");
    assert_eq!(bot.telemetry().iterations, 0, "{what}: no descents");
    assert!(!bot.has_cached_tree(), "{what}: no tree built");
    assert!(bot.last_search().is_none(), "{what}: no search summary");
}

fn assert_forced_record(sample: &botbowl_data::Sample, state: &GameState, what: &str) {
    let forced = EA::Simple(SimpleAT::EndPlayerTurn);
    assert_eq!(sample.chosen_action, forced, "{what}");
    assert_eq!(
        sample.children.len(),
        1,
        "{what}: the one post-pruning action is the only child"
    );
    let child = &sample.children[0];
    assert_eq!(child.action, forced, "{what}");
    assert_eq!(child.q, None, "{what}: unsearched, so no Q");
    assert_eq!(sample.root_value, None, "{what}: unsearched, so no root value");
    assert!(sample.scripted, "{what}: an unsearched decision is a scripted one");
    assert!(!sample.root_solved, "{what}");
    assert_eq!(sample.outcome_value, None, "{what}: backfilled later");
    assert!(sample.state == *state, "{what}: the decision state itself is recorded");
}

#[test]
fn every_entry_point_plays_a_forced_move_without_searching() {
    let state = forced_state();
    let forced = EA::Simple(SimpleAT::EndPlayerTurn);
    for gumbel_m in [0u16, 16] {
        let what = format!("gumbel_m={gumbel_m}");

        let mut bot = heuristic_bot(gumbel_m);
        assert_eq!(bot.get_action(&state), forced, "{what}: get_action");
        assert_unsearched(&bot, &format!("{what}: get_action"));

        let mut bot = heuristic_bot(gumbel_m);
        let (a, sample) = bot.get_action_with_record(&state);
        assert_eq!(a, forced);
        assert_unsearched(&bot, &format!("{what}: get_action_with_record"));
        assert_forced_record(&sample, &state, &format!("{what}: get_action_with_record"));

        let mut bot = heuristic_bot(gumbel_m);
        let (a, sample, outcome) = bot.get_action_explore(&state, explore_step());
        assert_eq!(a, forced);
        assert_unsearched(&bot, &format!("{what}: get_action_explore"));
        assert_forced_record(&sample, &state, &format!("{what}: get_action_explore"));
        assert!(
            !outcome.noised && !outcome.sampled && !outcome.deviated,
            "{what}: {outcome:?}"
        );
    }
}

#[test]
fn a_forced_move_costs_no_network_forward() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let state = forced_state();
    for gumbel_m in [0u16, 16] {
        let mut bot = nn_bot(gumbel_m);
        let (before, _) = profile_counters();
        let _ = bot.get_action(&state);
        let _ = bot.get_action_with_record(&state);
        let _ = bot.get_action_explore(&state, explore_step());
        let (after, _) = profile_counters();
        assert_eq!(
            after - before,
            0,
            "gumbel_m={gumbel_m}: a forced move must not reach the net"
        );
        assert_unsearched(&bot, &format!("nn gumbel_m={gumbel_m}"));
    }
}

/// The control: a real choice is still searched, under both roots, with the net.
#[test]
fn a_real_choice_is_still_searched() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let state = open_state();
    for gumbel_m in [0u16, 16] {
        let mut bot = nn_bot(gumbel_m);
        let (before, _) = profile_counters();
        let (_, sample) = bot.get_action_with_record(&state);
        let (after, _) = profile_counters();
        assert!(after > before, "gumbel_m={gumbel_m}: the search runs the net");
        assert_eq!(bot.telemetry().searches, 1);
        assert!(bot.telemetry().iterations > 0);
        assert!(sample.children.len() >= 2);
        assert!(!sample.scripted);
    }
}

/// A forced move leaves the previous decision's tree alone, so the next real decision can still
/// reuse it.
#[test]
fn a_forced_move_keeps_the_cached_tree() {
    let mut state = open_state();
    let mut bot = heuristic_bot(0);
    // Activate a player for a move: the search reaches the first move from here.
    let start = EA::Positional(PosAT::StartMove, Position::new((5, 5)));
    assert!(state.is_legal_action(&start));
    state.step(start).unwrap();
    let _ = bot.get_action(&state);
    assert!(bot.has_cached_tree());
    let searches = bot.telemetry().searches;

    state.step(EA::Positional(PosAT::Move, Position::new((6, 5)))).unwrap();
    assert_eq!(bot.get_action(&state), EA::Simple(SimpleAT::EndPlayerTurn));
    assert!(bot.has_cached_tree(), "the forced move must not drop the tree");
    assert_eq!(bot.telemetry().searches, searches, "and must not count as a search");
}

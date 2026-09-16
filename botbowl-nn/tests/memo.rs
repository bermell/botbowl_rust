//! The per-thread evaluation memo (`LAST_FORWARD` in `eval.rs`) is keyed on
//! the `GameState` itself, so the priors call that follows a value call on the
//! same node costs neither an encode nor a forward — and a hit is
//! indistinguishable from a fresh evaluation. Uses the committed `tiny.onnx`
//! and the process-wide forward counter, so this file holds exactly one test:
//! a second one in the same binary would race the counter.

use std::path::PathBuf;

use botbowl_engine::core::gamestate::{GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, Position};
use botbowl_engine::core::table::PosAT;
use botbowl_nn::eval::{profile_counters, NnEvaluator};

fn state_with_paths(ball_on_carrier: bool) -> GameState {
    let carrier = Position::new((5, 5));
    let mut builder = GameStateBuilder::new();
    builder.add_home_player(carrier).add_away_player(Position::new((8, 5)));
    if ball_on_carrier {
        builder.add_ball_pos(carrier);
    }
    let mut state = builder.build();
    state.set_logging_state(false);
    state
        .step(Action::Positional(PosAT::StartMove, carrier))
        .expect("activate the player");
    assert!(state.available_actions.has_paths(), "fixture has no path offerings");
    state
}

#[test]
fn a_state_is_encoded_and_forwarded_once_for_both_heads() {
    let onnx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.onnx");
    let eval = NnEvaluator::from_path(&onnx).expect("load tiny.onnx");
    let forwards = || profile_counters().0;

    let a = state_with_paths(true);
    let actions = a.get_all_actions();
    assert!(actions.len() > 5, "fixture offers {} actions", actions.len());

    // score_leaf, then available_actions, on one node: one forward.
    let n0 = forwards();
    let value_a = eval.value_home_i64_prefetch_policy(&a);
    assert_eq!(forwards(), n0 + 1, "the first call forwards");
    let priors_hit = eval.priors(&a, &actions);
    assert_eq!(forwards(), n0 + 1, "priors on the same state is a memo hit");

    // Equality by value, not by address: a clone is the same key.
    let a2 = a.clone();
    assert_eq!(eval.value_home_i64(&a2), value_a);
    assert_eq!(forwards(), n0 + 1, "a clone of the state is still a hit");

    // A different state evicts the slot ...
    let b = state_with_paths(false);
    assert_ne!(a, b);
    let _ = eval.value_home_i64(&b);
    assert_eq!(forwards(), n0 + 2, "a different state misses");

    // ... and a fresh evaluation of `a` reproduces the hit bit for bit, which
    // is what makes the memo invisible to the search.
    let priors_fresh = eval.priors(&a, &actions);
    assert_eq!(forwards(), n0 + 3, "evicted, so this one forwards again");
    assert_eq!(priors_hit, priors_fresh, "a memo hit must equal a fresh forward");
    assert_eq!(eval.value_home_i64(&a), value_a);
    assert_eq!(forwards(), n0 + 3);
}

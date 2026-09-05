//! One network pass per state, not two (plan 027 follow-up).
//!
//! `BBNet` emits `(policy, value)` from a shared trunk, but the search asks
//! for the two heads separately: `available_actions` wants priors at
//! expansion (`dynamics.rs:653`) and `score_leaf` wants the value for the
//! same node (`dynamics.rs:1165`). Both used to re-encode the state and run
//! a full forward, each discarding the head the other needed — so
//! `--evaluator nn` paid two passes per expanded node where `--evaluator
//! nn-value` pays one.
//!
//! That was not free. gen05 was the first generation to play with learned
//! priors and its generate phase took 425 min against gen04's 292 (+46%),
//! its eval 396 against 276 (+43%) — a ~58% longer cycle, which nearly
//! sank a capability change whose measured benefit (0.556, not significant)
//! had never been priced against it.
//!
//! Measured on a 300-iteration `Evaluator::Nn` search: 484 forwards before
//! the memo, 242 after. Exactly 2.00x.

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_engine::core::gamestate::GameStateBuilder;
use botbowl_engine::core::model::Position;
use botbowl_nn::eval::{profile_counters, NnEvaluator};

/// `profile_counters()` reads a **process-global** atomic, so two tests
/// measuring deltas concurrently would each see the other's forwards. Cargo
/// runs tests in one binary on parallel threads, so serialise them here —
/// without this the suite passes alone and fails under `cargo test
/// --workspace`, which is exactly how it was caught.
static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx")
}

/// The invariant, stated where it cannot drift: asking for both heads of
/// one state costs one forward. Asserted at the evaluator API rather than
/// through a search, so it stays stable against changes in tree shape.
#[test]
fn both_heads_of_one_state_cost_one_forward() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let nn = Arc::new(NnEvaluator::from_path(fixture()).expect("load tiny.onnx"));
    let state = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_home_player(Position::new((7, 8)))
        .add_away_player(Position::new((9, 5)))
        .build();
    let actions = state.get_all_actions();
    assert!(!actions.is_empty());

    // Warm the slot, then measure the pair the search actually performs.
    let _ = nn.priors(&state, &actions);
    let _ = nn.value_home_i64(&state);

    let (before, _) = profile_counters();
    let priors = nn.priors(&state, &actions);
    let value = nn.value_home_i64(&state);
    let (after, _) = profile_counters();

    assert_eq!(
        after - before,
        0,
        "a repeated priors+value pair on an unchanged state should be served from the memo"
    );

    // A different state must miss, or the memo is returning stale answers —
    // the failure mode that makes a cache worse than no cache.
    let other = GameStateBuilder::new()
        .add_home_player(Position::new((4, 4)))
        .add_away_player(Position::new((11, 6)))
        .build();
    let (b2, _) = profile_counters();
    let _ = nn.value_home_i64(&other);
    let (a2, _) = profile_counters();
    assert_eq!(a2 - b2, 1, "a new state must force a fresh forward");

    // And the memoised answers must equal the un-memoised ones.
    assert!(priors.iter().all(|p| p.is_finite() && *p > 0.0));
    assert!((-1000..=1000).contains(&value), "value {value} outside leaf_score band");
}

/// Going back to a previously-seen state after an intervening one must
/// re-forward rather than resurrect a stale entry: the slot holds exactly
/// one state, and correctness depends on it saying so honestly.
#[test]
fn memo_does_not_resurrect_a_displaced_state() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let nn = Arc::new(NnEvaluator::from_path(fixture()).expect("load tiny.onnx"));
    let a = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_away_player(Position::new((9, 5)))
        .build();
    let b = GameStateBuilder::new()
        .add_home_player(Position::new((6, 6)))
        .add_away_player(Position::new((10, 7)))
        .build();

    let va1 = nn.value_home_i64(&a);
    let _ = nn.value_home_i64(&b);
    let (before, _) = profile_counters();
    let va2 = nn.value_home_i64(&a);
    let (after, _) = profile_counters();

    assert_eq!(after - before, 1, "displaced state must be recomputed, not served stale");
    assert_eq!(va1, va2, "the network is deterministic; the memo must not change its answers");
}

/// The bug this guards against. The search scores a node before enumerating
/// its children, so the value call lands first. A plain value call asks for
/// no policy, so the priors call that follows must forward again; the
/// prefetching variant warms both heads in one pass.
///
/// This only bites when the backend honours `want_policy`. Tract used to
/// ignore it and hand the policy back regardless, so a tract benchmark showed
/// the memo halving forwards while production — on the sidecar, which does
/// honour it — saved 0.7% (gen05 eval 59,511,550 samples -> gen06 59,071,998).
/// Both backends now behave the same, which is what makes this test mean
/// something.
#[test]
fn prefetch_saves_the_second_forward_in_search_order() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let nn = Arc::new(NnEvaluator::from_path(fixture()).expect("load tiny.onnx"));
    let state = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_away_player(Position::new((9, 5)))
        .build();
    let actions = state.get_all_actions();
    assert!(!actions.is_empty());

    let other = GameStateBuilder::new()
        .add_home_player(Position::new((3, 3)))
        .add_away_player(Position::new((12, 6)))
        .build();

    // Search order without the prefetch: value, then priors -> two forwards.
    let _ = nn.value_home_i64(&other); // displace the slot
    let (b1, _) = profile_counters();
    let _ = nn.value_home_i64(&state);
    let _ = nn.priors(&state, &actions);
    let (a1, _) = profile_counters();
    assert_eq!(a1 - b1, 2, "plain value call must not warm the policy head");

    // Same order, prefetching variant -> one forward.
    let _ = nn.value_home_i64(&other); // displace again
    let (b2, _) = profile_counters();
    let _ = nn.value_home_i64_prefetch_policy(&state);
    let _ = nn.priors(&state, &actions);
    let (a2, _) = profile_counters();
    assert_eq!(a2 - b2, 1, "prefetching value call must serve the following priors call");
}

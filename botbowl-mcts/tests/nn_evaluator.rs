//! Unit checks on the NN evaluator wiring, using the committed tiny.onnx
//! fixture (an untrained seeded net — arithmetic only, not quality).
//!
//! Asserts the two calibration bridges plan 017 introduces:
//! - priors are a softmax **rescaled × K** (mean ≈ 1.0), matching the
//!   un-normalised `BASE = 1.0` scale `PUCT_C` expects;
//! - the leaf value lands in `leaf_score`'s `±1000` band.

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_engine::core::gamestate::GameStateBuilder;
use botbowl_engine::core::model::Position;
use botbowl_nn::eval::NnEvaluator;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx")
}

#[test]
fn nn_priors_are_rescaled_softmax_and_value_in_band() {
    let onnx = fixture();
    assert!(
        onnx.exists(),
        "missing fixture {} — run `uv run python -m bbnn.fixture`",
        onnx.display()
    );
    let nn = Arc::new(NnEvaluator::from_path(&onnx).expect("load tiny.onnx"));

    // A concrete Home-to-move decision on the default 28x17 board.
    let state = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_home_player(Position::new((7, 8)))
        .build();
    let actions = state.get_all_actions();
    assert!(!actions.is_empty(), "expected legal actions at a Home turn");

    let priors = nn.priors(&state, &actions);
    assert_eq!(priors.len(), actions.len());
    assert!(priors.iter().all(|&p| p > 0.0), "priors must be positive");
    let sum: f32 = priors.iter().sum();
    // Rescaled softmax → sum ≈ K (mean ≈ 1.0).
    assert!(
        (sum - actions.len() as f32).abs() < 1e-2,
        "priors should sum to K={}, got {sum}",
        actions.len()
    );

    let v = nn.value_home_i64(&state);
    assert!((-1000..=1000).contains(&v), "value {v} outside leaf_score's ±1000 band");
}

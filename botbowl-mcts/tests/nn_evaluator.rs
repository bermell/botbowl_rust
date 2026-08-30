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

/// Known-outcome leaves must get the exact drive outcome, not an NN
/// estimate: a score change since the anchor (or game over) is a
/// gold-standard value, and the net never trains on post-TD kickoff
/// states — its output there is noise that would make TDs invisible to
/// the search (the score_td_easy failure mode all over again).
#[test]
fn nn_score_leaf_uses_exact_value_for_known_outcomes() {
    use botbowl_engine::core::dices::RequestedRoll;
    use botbowl_engine::core::model::TeamType;
    use botbowl_mcts::dynamics::HorizonAnchor;
    use botbowl_mcts::{BbPlayer, BloodBowlDynamics, Evaluator};
    use recon_mcts::GameDynamics;

    let nn = Arc::new(NnEvaluator::from_path(fixture()).expect("load tiny.onnx"));
    let base = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
    let anchor = HorizonAnchor::capture(&base, TeamType::Home);
    let dynamics = BloodBowlDynamics {
        horizon: Some(anchor),
        virtual_loss: 0,
        evaluator: Evaluator::Nn(nn),
        ..Default::default()
    };

    // Home TD since the anchor, engine paused on the kickoff roll — the
    // state every in-search touchdown lands on. Exact +1000, not NN.
    let mut td = base.clone();
    td.home.score += 1;
    td.pending_roll = Some(RequestedRoll::D8);
    let score = dynamics
        .score_leaf(None, &BbPlayer::Chance, &td)
        .expect("score-changed leaf must be scored");
    assert_eq!(score.score, 1000, "home TD must be scored exactly");

    // Away TD since the anchor → exact -1000.
    let mut away_td = base.clone();
    away_td.away.score += 1;
    let score = dynamics.score_leaf(None, &BbPlayer::Chance, &away_td).unwrap();
    assert_eq!(score.score, -1000, "away TD must be scored exactly");

    // Game over without a score change since the anchor → exact 0 (the
    // drive ended scoreless), regardless of the absolute scoreline.
    let mut over = base.clone();
    over.info.game_over = true;
    let score = dynamics.score_leaf(None, &BbPlayer::Chance, &over).unwrap();
    assert_eq!(score.score, 0, "scoreless game-over leaf must be exactly 0");
}

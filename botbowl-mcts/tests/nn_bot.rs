//! End-to-end smoke: an NN-backed MctsBot plays real Blood Bowl through
//! the full search. Gated on `BLOOD_NN_MODEL` (path to an exported ONNX)
//! and `#[ignore]`d — the smoke model isn't expected to be *good*, so
//! there is **no success-rate assertion**; we only assert the trials
//! complete legally and print the NN vs heuristic rate for eyeballing.
//!
//! Run with e.g.:
//!   BLOOD_NN_MODEL=models/score_td.onnx \
//!     cargo test -p botbowl-mcts --test nn_bot -- --ignored --nocapture

use std::sync::Arc;
use std::time::Duration;

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::{run_trials_cfg, TrialConfig};
use botbowl_mcts::{MctsBot, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

const SEARCH_TIME: Duration = Duration::from_millis(150);
const N_TRIALS: u32 = 10;

fn trial_config() -> TrialConfig {
    TrialConfig {
        n_trials: N_TRIALS,
        seed: 0xCAFE_1234,
        max_steps_per_trial: 400,
        max_agent_actions: Some(25),
    }
}

#[test]
#[ignore = "NN smoke — needs BLOOD_NN_MODEL, run with --ignored"]
fn nn_backed_bot_plays_legal_blood_bowl() {
    let model_path = match std::env::var("BLOOD_NN_MODEL") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("BLOOD_NN_MODEL unset — skipping NN smoke test (export one via train/ first)");
            return;
        }
    };
    let nn = Arc::new(NnEvaluator::from_path(&model_path).expect("load BLOOD_NN_MODEL"));

    // NN-backed bot.
    let lecture = ScoreTdEasy::new();
    let mut nn_agent = MctsBot::new(SearchBudget::Time(SEARCH_TIME))
        .with_workers(1)
        .with_evaluator(nn);
    let nn_stats = run_trials_cfg(&lecture, &mut nn_agent, trial_config());
    assert_eq!(nn_stats.trials, N_TRIALS, "NN bot did not complete all trials");

    // Heuristic baseline for side-by-side comparison (same budget/seed).
    let lecture_h = ScoreTdEasy::new();
    let mut heur_agent = MctsBot::new(SearchBudget::Time(SEARCH_TIME)).with_workers(1);
    let heur_stats = run_trials_cfg(&lecture_h, &mut heur_agent, trial_config());

    eprintln!(
        "ScoreTdEasy smoke ({SEARCH_TIME:?}/move, {N_TRIALS} trials):\n  \
         NN model {model_path}: successes={} failures={} timeouts={} rate={:.3}\n  \
         Heuristic baseline:      successes={} failures={} timeouts={} rate={:.3}",
        nn_stats.successes,
        nn_stats.failures,
        nn_stats.timeouts,
        nn_stats.success_rate(),
        heur_stats.successes,
        heur_stats.failures,
        heur_stats.timeouts,
        heur_stats.success_rate(),
    );
    // No rate assertion — the smoke model is a plumbing artifact, not a
    // trained policy. Completing legally is the bar here.
}

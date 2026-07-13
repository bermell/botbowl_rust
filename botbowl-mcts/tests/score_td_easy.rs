use std::time::Duration;

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::{run_trials_cfg, TrialConfig};
use botbowl_mcts::{MctsBot, SearchBudget};

// TODO(mattias): tune. Placeholder ≈ the cost of the old 1000
// iters/move (0.15–0.7 s/search measured on GetTheBallMedium).
const SEARCH_TIME: Duration = Duration::from_millis(300);
const MAX_AGENT_ACTIONS: u32 = 25;

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn mcts_lifts_random_baseline() {
    let lecture = ScoreTdEasy::new();
    let mut agent = MctsBot::new(SearchBudget::Time(SEARCH_TIME)).with_workers(1);
    let stats = run_trials_cfg(
        &lecture,
        &mut agent,
        TrialConfig {
            n_trials: 50,
            seed: 0xCAFE_1234,
            max_steps_per_trial: 400,
            max_agent_actions: Some(MAX_AGENT_ACTIONS),
        },
    );

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy MCTS (PUCT + priors, {SEARCH_TIME:?}/move, ≤{MAX_AGENT_ACTIONS} agent actions): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline ~9%. Threshold history:
    // - Initial PUCT + priors: ~0.86 measured, threshold 0.80.
    // - Plan 010 (score_leaf FF through mid-procedure states): drops
    //   to ~0.74 mean (range 0.70–0.86), because Away's leaves now
    //   resolve through full Move actions too — the opponent plays
    //   sharper defense and Home's TD-scoring paths are evaluated
    //   against a tighter reply. Net positive trade (GetTheBallEasy
    //   went 0.00 → 1.00, Medium 0.00 → ~0.35), but the dip is real.
    //   Threshold lowered to 0.65 to absorb concurrent-search variance.
    // - Plan 018 (8d81444) silently collapsed this to ~0.04: a post-TD
    //   state is pending-roll (next kickoff's Deviate) AND past the
    //   horizon, so it was neither scored nor expandable — TDs became
    //   invisible to backprop. Fixed 2026-07-13 (score_leaf carve-out
    //   for past-horizon pending-roll states, terminal re-descent
    //   backprop in recon_mcts, FPU for unexplored children, Q-based
    //   root pick, StartFoul pruning): ~0.96 measured.
    assert!(
        rate >= 0.65,
        "MCTS bot success rate {:.4} below 0.65 — priors/pruning regression",
        rate
    );
}

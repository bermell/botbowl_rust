use std::time::Duration;

use botbowl_curriculum::lectures::score_td::ScoreTdMedium;
use botbowl_curriculum::{run_trials_cfg, TrialConfig};
use botbowl_mcts::{MctsBot, SearchBudget};

// Tuned 2026-07-13: 150 ms/move measured 0.90 vs 0.96 at 300 ms —
// half the suite wall-clock for a within-noise rate cost.
const SEARCH_TIME: Duration = Duration::from_millis(150);
const MAX_AGENT_ACTIONS: u32 = 25;

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn mcts_solves_score_td_medium() {
    let lecture = ScoreTdMedium::new();
    let mut agent = MctsBot::new(SearchBudget::Time(SEARCH_TIME)).with_workers(1);
    let stats = run_trials_cfg(
        &lecture,
        &mut agent,
        TrialConfig {
            n_trials: 50,
            seed: 0xC0DE_5678,
            max_steps_per_trial: 400,
            max_agent_actions: Some(MAX_AGENT_ACTIONS),
        },
    );

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdMedium MCTS (PUCT + priors, {SEARCH_TIME:?}/move, ≤{MAX_AGENT_ACTIONS} agent actions): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline ~6%. Scripted bot target ≥70%. Threshold history:
    // - v1 target 0.50 (never measured green: the throw-in ⇄ failed-catch
    //   state cycle crashed the search via the recon_mcts cycle guard).
    // - 2026-07-13 search fixes (TD-visible score_leaf, terminal-redescent
    //   backprop, FPU, Q-based pick, catch-square engine fix, chance
    //   completeness gate): 0.96 at 300 ms; 0.90 and 0.82 across two
    //   150 ms runs. Threshold 0.70 ≈ 3σ below the 150 ms mean
    //   (50 trials, σ≈0.05).
    assert!(
        rate >= 0.70,
        "MCTS bot success rate {:.4} below 0.70 on ScoreTdMedium",
        rate
    );
}

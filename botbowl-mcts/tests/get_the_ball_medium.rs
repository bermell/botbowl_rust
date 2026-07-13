use std::time::Duration;

use botbowl_curriculum::lectures::get_the_ball::GetTheBallMedium;
use botbowl_curriculum::{run_trials_cfg, TrialConfig};
use botbowl_mcts::{MctsBot, SearchBudget};

// Tuned 2026-07-13: 150 ms/move measured 0.98 vs 1.00 at 300 ms —
// half the suite wall-clock for a within-noise rate cost. (Successful
// trials use 14–20 agent actions — each get_action call counts, incl.
// Move-per-square, SelectPush, FollowUp, EndTurn — hence the cap.)
const SEARCH_TIME: Duration = Duration::from_millis(150);
const MAX_AGENT_ACTIONS: u32 = 25;

/// Rate history: plan 010 (Track A.alt) lifted this from 0.00 to ≈0.35
/// (threshold 0.30). The 2026-07-13 search fixes lifted it further:
/// 0.78–0.82 after the TD-visibility / terminal-backprop / FPU / Q-pick
/// batch, then 1.00 (300 ms) / 0.98 (150 ms) once the chance-node
/// completeness gate landed — the gate removes the conditioned-on-
/// success over-valuation of the marked-ball pickup, which was exactly
/// this lecture's failure mode. Threshold 0.80 ≈ 3σ below the 150 ms
/// measurement (50 trials).
#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn mcts_solves_get_the_ball_medium() {
    let lecture = GetTheBallMedium::new();
    let mut agent = MctsBot::new(SearchBudget::Time(SEARCH_TIME)).with_workers(1);
    let stats = run_trials_cfg(
        &lecture,
        &mut agent,
        TrialConfig {
            n_trials: 50,
            seed: 0xBEEF_3456,
            max_steps_per_trial: 400,
            max_agent_actions: Some(MAX_AGENT_ACTIONS),
        },
    );

    let rate = stats.success_rate();
    eprintln!(
        "GetTheBallMedium MCTS (PUCT + priors, {SEARCH_TIME:?}/move, ≤{MAX_AGENT_ACTIONS} agent actions): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    assert!(
        rate >= 0.80,
        "MCTS bot success rate {:.4} below 0.80 on GetTheBallMedium",
        rate
    );
}

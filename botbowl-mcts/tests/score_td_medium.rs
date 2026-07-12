use std::time::Duration;

use botbowl_curriculum::lectures::score_td::ScoreTdMedium;
use botbowl_curriculum::{run_trials_cfg, TrialConfig};
use botbowl_mcts::{MctsBot, SearchBudget};

// TODO(mattias): tune. Placeholder ≈ the cost of the old 1000
// iters/move (0.15–0.7 s/search measured on GetTheBallMedium).
const SEARCH_TIME: Duration = Duration::from_millis(300);
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

    // Random baseline ~6%. Scripted bot target ≥70%. The carrier-toward-
    // endzone prior should help substantially; v1 target is ≥0.50.
    assert!(
        rate >= 0.50,
        "MCTS bot success rate {:.4} below 0.50 on ScoreTdMedium",
        rate
    );
}

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

#[test]
fn mcts_lifts_random_baseline() {
    let lecture = ScoreTdEasy::new();
    let mut agent = MctsBot::new(1000);
    let stats = run_trials(&lecture, &mut agent, 50, 0xCAFE_1234, 400);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy MCTS (PUCT + priors, 1000 iters/move): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline ~9%. With PUCT + the carrier-toward-endzone and
    // end-turn priors, the bot should clear 80% comfortably (86% measured
    // during initial implementation). Threshold left at 0.80 — bump if
    // future tuning lifts the rate further.
    assert!(
        rate >= 0.80,
        "MCTS bot success rate {:.4} below 0.80 — priors/pruning regression",
        rate
    );
}

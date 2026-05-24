use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

#[test]
fn mcts_lifts_random_baseline() {
    let lecture = ScoreTdEasy::new();
    let mut agent = MctsBot::new(1000).with_workers(1);
    let stats = run_trials(&lecture, &mut agent, 50, 0xCAFE_1234, 400);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy MCTS (PUCT + priors, 1000 iters/move): \
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
    assert!(
        rate >= 0.65,
        "MCTS bot success rate {:.4} below 0.65 — priors/pruning regression",
        rate
    );
}

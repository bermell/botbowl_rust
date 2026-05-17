use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

#[test]
fn mcts_lifts_random_baseline() {
    let lecture = ScoreTdEasy::new();
    let mut agent = MctsBot::new(1000);
    // Fewer trials than for random/scripted — each trial runs 1000
    // search iterations per move, so this test takes meaningful time.
    let stats = run_trials(&lecture, &mut agent, 50, 0xCAFE_1234, 400);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy MCTS (pure UCT, 1000 iters/move): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline on this lecture is ~9%. Demand a clear lift to
    // confirm tree search is doing real work. 60% is the plan target;
    // we leave some headroom for now (50%) because the search has zero
    // domain heuristic for action *selection* — only the leaf ladder.
    assert!(
        rate >= 0.50,
        "MCTS bot success rate {:.4} below 0.50 — search isn't navigating effectively",
        rate
    );
}

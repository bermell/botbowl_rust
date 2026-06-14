use botbowl_curriculum::lectures::score_td::ScoreTdMedium;
use botbowl_curriculum::run_trials;
use botbowl_mcts::{MctsBot, SearchBudget};

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn mcts_solves_score_td_medium() {
    let lecture = ScoreTdMedium::new();
    let mut agent = MctsBot::new(SearchBudget::Iterations(1000)).with_workers(1);
    let stats = run_trials(&lecture, &mut agent, 50, 0xC0DE_5678, 400);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdMedium MCTS (PUCT + priors, 1000 iters/move): \
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

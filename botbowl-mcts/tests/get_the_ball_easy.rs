use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

/// Plan 010 (Track A.alt) lifts this from 0.00 to ≈1.00: `score_leaf`
/// now forward-simulates through mid-procedure engine work (Move
/// walking one square per `micro_step`) and through pending rolls
/// (pickup, dodge, GFI), so leaves between squares actually score
/// the post-pickup board rather than an intermediate position. The
/// chance child still does not enter the tree — see plan 010 for
/// the deferred Track A work if a future lecture needs that.
#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn mcts_solves_get_the_ball_easy() {
    let lecture = GetTheBallEasy::new();
    let mut agent = MctsBot::new(1000).with_workers(1);
    let stats = run_trials(&lecture, &mut agent, 50, 0xF00D_9012, 400);

    let rate = stats.success_rate();
    eprintln!(
        "GetTheBallEasy MCTS (PUCT + priors, 1000 iters/move): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline is in the 0.5–50% band; the ×10 pickup prior
    // *should* make this near-trivial once the v2 fast-forward lands.
    assert!(
        rate >= 0.70,
        "MCTS bot success rate {:.4} below 0.70 on GetTheBallEasy",
        rate
    );
}

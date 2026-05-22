use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

/// Still parked. v2 added `ChanceOutcome::Advance` (no more panics on
/// unsupported roll types) and scripted block-die selection, but
/// reaching the pickup chance node also needs fast-forwarding through
/// mid-Move procedure transitions inside `apply_action`. Naively
/// adding that loop blew the search tree out by ~1000× — even tiny
/// 50-iter trials wouldn't finish inside the cargo 60s slow-test
/// budget. v3 needs a different chance-modelling approach (smarter
/// state-equivalence hashing, or modelling each Move as a single
/// atomic action that resolves pickup inside `apply_action`).
#[ignore = "v3: needs chance-node modelling rework (FF blows up the tree)"]
#[test]
fn mcts_solves_get_the_ball_easy() {
    let lecture = GetTheBallEasy::new();
    let mut agent = MctsBot::new(1000);
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

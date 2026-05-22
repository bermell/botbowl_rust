use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

/// Pickup-driven lectures need the engine's mid-path procedure
/// transitions to be fast-forwarded inside `apply_action` so that the
/// pickup chance node actually surfaces — otherwise `score_leaf` sees
/// a mid-Move-procedure terminal state, scores it 0, and the ×10
/// pickup prior loses to the Q=200 adjacent-to-ball moves. v1 ships
/// without that fast-forward (each attempt at it either timed out the
/// search or hit unimplemented roll types like `Deviate`), so this
/// lecture is parked as a v2 target. See `dynamics::apply_action`
/// and `plans/003-idea--mcts-action-pruning.md`.
#[ignore = "v2: needs apply_action fast-forward + broader roll_outcomes coverage"]
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

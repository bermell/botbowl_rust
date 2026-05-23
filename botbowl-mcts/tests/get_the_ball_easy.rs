use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

/// Still parked. v3 added deterministic dice fixes for non-pass/fail
/// rolls (closing the RNG-non-determinism gap from v2's `Advance`) and
/// an optimistic chance-state `score_leaf`, but the pickup chance node
/// still doesn't surface because the engine processes a Move(target)
/// path one square per `micro_step` and we don't fast-forward
/// (FF+chance produced both deep-tree per-iter slowdowns of ~10000×
/// and reconstruction panics during `Tree` drop — `apply_action`
/// returns `None` for some recombined edge during `recon_mcts`
/// `get_state` walks, source unknown). v4 candidates: a non-recursive
/// `recon_mcts` `Drop`, or moving the FF inside `score_leaf` only
/// (forward-looking value without reifying the chance state into the
/// tree).
#[ignore = "v5: adversarial backprop landed but rate stays 0% — bottleneck is the unaddressed FF/chance-node blocker, not opponent modelling"]
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

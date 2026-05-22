use botbowl_curriculum::lectures::get_the_ball::GetTheBallMedium;
use botbowl_curriculum::run_trials;
use botbowl_mcts::MctsBot;

/// Same v2 blocker as `get_the_ball_easy`: needs fast-forward through
/// mid-Move-procedure states so the pickup chance node surfaces, plus
/// broader `roll_outcomes` coverage for block dice / deviate rolls
/// that follow the displacement of the marker.
#[ignore = "v2: needs apply_action fast-forward + broader roll_outcomes coverage"]
#[test]
fn mcts_solves_get_the_ball_medium() {
    let lecture = GetTheBallMedium::new();
    let mut agent = MctsBot::new(1000);
    let stats = run_trials(&lecture, &mut agent, 50, 0xBEEF_3456, 400);

    let rate = stats.success_rate();
    eprintln!(
        "GetTheBallMedium MCTS (PUCT + priors, 1000 iters/move): \
         trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Random baseline ~0% (marked pickup auto-fails). Scripted bot
    // target ≥50%. v1 priors don't include a blocking heuristic, so we
    // ask for a clear lift over random rather than scripted parity:
    // ≥0.40 is the target.
    assert!(
        rate >= 0.40,
        "MCTS bot success rate {:.4} below 0.40 on GetTheBallMedium",
        rate
    );
}

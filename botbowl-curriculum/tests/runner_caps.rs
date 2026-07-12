use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::{run_trials_cfg, LectureSession, TrialConfig};
use botbowl_engine::bots::RandomBot;

/// The agent-action cap ends a session as unfinished-in-progress
/// (a timeout in `run_trials` terms) without counting opponent or
/// engine micro-steps against it.
#[test]
fn session_stops_at_max_agent_actions() {
    let lecture = GetTheBallEasy::new();
    let mut agent = RandomBot::new();
    let mut session =
        LectureSession::new(&lecture, 0xABCD, 400, &mut agent).with_max_agent_actions(Some(3));

    while !session.is_finished() {
        session.step(&mut agent);
    }

    assert!(session.agent_actions_taken() <= 3);
}

/// `max_agent_actions: None` preserves the old behaviour exactly —
/// same stats as the positional `run_trials`.
#[test]
fn cfg_without_cap_matches_run_trials() {
    let lecture = GetTheBallEasy::new();

    let mut agent = RandomBot::new();
    let uncapped = run_trials_cfg(
        &lecture,
        &mut agent,
        TrialConfig {
            n_trials: 20,
            seed: 0xABCD,
            max_steps_per_trial: 400,
            max_agent_actions: None,
        },
    );

    let mut agent = RandomBot::new();
    let legacy = botbowl_curriculum::run_trials(&lecture, &mut agent, 20, 0xABCD, 400);

    assert_eq!(uncapped, legacy);
}

use botbowl_curriculum::lectures::get_the_ball::{GetTheBallEasy, GetTheBallHard, GetTheBallMedium};
use botbowl_curriculum::{run_trials, Lecture, LectureContext, LectureStatus};
use botbowl_engine::bots::RandomBot;
use botbowl_engine::core::model::{BallState, TeamType};
use botbowl_engine::scripted_bot::ScriptedBot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn easy_setup_places_ball_on_ground_and_home_nearby() {
    let lecture = GetTheBallEasy::new();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let state = lecture.setup(&mut rng);

    assert!(matches!(state.ball, BallState::OnGround(_)));
    let home_count = state.get_players_on_pitch_in_team(TeamType::Home).count();
    let away_count = state.get_players_on_pitch_in_team(TeamType::Away).count();
    assert_eq!(home_count, 1);
    assert_eq!(away_count, 1);

    let ctx = LectureContext::from_state(&state);
    assert_eq!(lecture.evaluate(&state, &ctx), LectureStatus::InProgress);
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn easy_scripted_picks_up_the_ball() {
    let lecture = GetTheBallEasy::new();
    let mut agent = ScriptedBot::new();
    let stats = run_trials(&lecture, &mut agent, 500, 0xC0FFEE, 400);

    let rate = stats.success_rate();
    eprintln!(
        "GetTheBallEasy scripted: trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Unmarked pickup with the "3+ succeeds" policy is deterministic — the
    // scripted bot should almost always pick up and hang onto the ball
    // through the lone Away player's distant reply turn.
    assert!(
        rate >= 0.90,
        "scripted GetTheBallEasy success rate {:.4} below 0.90 — the plan-then-execute loop is regressed",
        rate
    );
    assert_eq!(stats.timeouts, 0, "scripted bot trials must terminate");
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn easy_random_baseline_is_meaningful() {
    let lecture = GetTheBallEasy::new();
    let mut agent = RandomBot::new();
    let stats = run_trials(&lecture, &mut agent, 2_000, 0xC0FFEE, 400);

    let rate = stats.success_rate();
    eprintln!(
        "GetTheBallEasy random: trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // The lecture is solvable (random will sometimes blunder into a pickup
    // and end the turn) but should be much harder than for the scripted
    // bot. A loose 0.5%-50% band is enough to catch lecture regressions.
    assert!(
        (0.005..0.50).contains(&rate),
        "random GetTheBallEasy rate {:.4} outside [0.005, 0.50]",
        rate
    );
}

#[test]
fn medium_setup_places_marker_adjacent_to_ball() {
    let lecture = GetTheBallMedium::new();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let state = lecture.setup(&mut rng);

    let ball_pos = match state.ball {
        BallState::OnGround(p) => p,
        _ => panic!("expected ball on ground"),
    };
    let any_marker = state
        .get_players_on_pitch_in_team(TeamType::Away)
        .any(|p| p.position.distance_to(&ball_pos) == 1);
    assert!(any_marker, "expected at least one Away player adjacent to the ball");

    let home_count = state.get_players_on_pitch_in_team(TeamType::Home).count();
    assert_eq!(home_count, 3, "Medium needs picker + blocker + assistant");
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn medium_scripted_dominates_random() {
    let lecture = GetTheBallMedium::new();

    let mut random = RandomBot::new();
    let random_stats = run_trials(&lecture, &mut random, 2_000, 0xFEED_BABE, 400);
    let random_rate = random_stats.success_rate();

    let mut scripted = ScriptedBot::new();
    let scripted_stats = run_trials(&lecture, &mut scripted, 500, 0xFEED_BABE, 400);
    let scripted_rate = scripted_stats.success_rate();

    eprintln!(
        "GetTheBallMedium random:   trials={} successes={} rate={:.4}",
        random_stats.trials, random_stats.successes, random_rate
    );
    eprintln!(
        "GetTheBallMedium scripted: trials={} successes={} rate={:.4}",
        scripted_stats.trials, scripted_stats.successes, scripted_rate
    );

    // The pickup auto-fails until the marker is displaced. The scripted
    // bot should plan a block-then-pickup sequence; random will rarely
    // stumble into it.
    assert!(
        scripted_rate >= 0.50,
        "scripted GetTheBallMedium success rate {:.4} below 0.50 — improve the planner to clear the marker first",
        scripted_rate
    );
    assert!(
        scripted_rate - random_rate >= 0.30,
        "scripted-vs-random gap {:.4} too small (scripted={:.4} random={:.4})",
        scripted_rate - random_rate,
        scripted_rate,
        random_rate
    );
}

#[test]
fn hard_setup_places_carrier_with_ball() {
    let lecture = GetTheBallHard::new();
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let state = lecture.setup(&mut rng);

    // Ball must be carried by an Away player.
    let carrier_id = match state.ball {
        BallState::Carried(id) => id,
        other => panic!("expected ball to be carried by the opponent, got {:?}", other),
    };
    let carrier = state.get_player(carrier_id).unwrap();
    assert_eq!(carrier.stats.team, TeamType::Away);

    // Only 2 home players — no built-in 2-die block. The bot must blitz.
    let home_count = state.get_players_on_pitch_in_team(TeamType::Home).count();
    assert_eq!(home_count, 2);
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn hard_scripted_dominates_random() {
    let lecture = GetTheBallHard::new();

    let mut random = RandomBot::new();
    let random_stats = run_trials(&lecture, &mut random, 2_000, 0xC0DE_F00D, 400);
    let random_rate = random_stats.success_rate();

    let mut scripted = ScriptedBot::new();
    let scripted_stats = run_trials(&lecture, &mut scripted, 500, 0xC0DE_F00D, 400);
    let scripted_rate = scripted_stats.success_rate();

    eprintln!(
        "GetTheBallHard random:   trials={} successes={} rate={:.4}",
        random_stats.trials, random_stats.successes, random_rate
    );
    eprintln!(
        "GetTheBallHard scripted: trials={} successes={} rate={:.4}",
        scripted_stats.trials, scripted_stats.successes, scripted_rate
    );

    assert!(
        scripted_rate >= 0.50,
        "scripted GetTheBallHard success rate {:.4} below 0.50 — extend the planner to tackle then pickup",
        scripted_rate
    );
    assert!(
        scripted_rate - random_rate >= 0.30,
        "scripted-vs-random gap {:.4} too small (scripted={:.4} random={:.4})",
        scripted_rate - random_rate,
        scripted_rate,
        random_rate
    );
}

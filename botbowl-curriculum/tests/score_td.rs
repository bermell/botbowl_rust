use botbowl_curriculum::lectures::score_td::{ScoreTdEasy, ScoreTdMedium};
use botbowl_curriculum::{run_trials, Lecture, LectureStatus};
use botbowl_engine::bots::RandomBot;
use botbowl_engine::scripted_bot::ScriptedBot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn setup_yields_a_home_carrier_with_the_ball() {
    let lecture = ScoreTdEasy::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let state = lecture.setup(&mut rng);

    let home_on_pitch = state
        .get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Home)
        .count();
    let away_on_pitch = state
        .get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Away)
        .count();
    assert_eq!(
        home_on_pitch, 1,
        "expected exactly one home player on pitch"
    );
    assert_eq!(away_on_pitch, 0, "expected an empty away side");

    let carrier = state
        .get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Home)
        .next()
        .unwrap();
    // Ball position should match the carrier's position.
    assert_eq!(state.get_ball_position(), Some(carrier.position));
    assert!(carrier.position.x >= 6 && carrier.position.x <= 9);
    assert!(carrier.position.y >= 3 && carrier.position.y <= 13);

    // Lecture is in progress at setup (no score, game not over, Home to act).
    let ctx = botbowl_curriculum::LectureContext::from_state(&state);
    assert_eq!(lecture.evaluate(&state, &ctx), LectureStatus::InProgress);
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn random_agent_baseline_success_rate() {
    let lecture = ScoreTdEasy::new();
    let mut agent = RandomBot::new();
    let stats = run_trials(&lecture, &mut agent, 10_000, 0xC0DE_BEEF, 200);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy random baseline: trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // Measured baseline with free-path setup is ~5-10%. The lecture is
    // sound as long as the rate stays well below 50% (random isn't
    // trivially solving it) and above 1% (the lecture is actually
    // solvable). Future Medium/Hard difficulties will push the rate
    // toward the grand plan's 1% aspiration.
    assert!(
        (0.01..0.20).contains(&rate),
        "random success rate {:.4} outside expected band [0.01, 0.20]; \
         tune carrier distance from end zone if drifted",
        rate
    );
    // Timeouts indicate the lecture didn't terminate — they should be rare.
    assert!(
        stats.timeouts < stats.trials / 20,
        "too many timeouts: {}/{}",
        stats.timeouts,
        stats.trials
    );
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn scripted_agent_dominates_easy() {
    let lecture = ScoreTdEasy::new();
    let mut agent = ScriptedBot::new();
    let stats = run_trials(&lecture, &mut agent, 1_000, 0xDEAD_BEEF, 200);

    let rate = stats.success_rate();
    eprintln!(
        "ScoreTdEasy scripted baseline: trials={} successes={} failures={} timeouts={} rate={:.4}",
        stats.trials, stats.successes, stats.failures, stats.timeouts, rate
    );

    // The scripted bot's safest-path-to-endzone planner should pretty much
    // always succeed when there are no opponents. Demand >= 90% to catch
    // regressions in the scripted bot's plan-then-execute loop.
    assert!(
        rate >= 0.90,
        "scripted bot success rate {:.4} below 0.90 — the planner is broken or the lecture regressed",
        rate
    );
    assert_eq!(stats.timeouts, 0, "scripted bot trials must terminate");
}

#[test]
fn medium_setup_places_blocker_in_path() {
    let lecture = ScoreTdMedium::new();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let state = lecture.setup(&mut rng);

    let home_on_pitch = state
        .get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Home)
        .count();
    let away_on_pitch = state
        .get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Away)
        .count();
    assert_eq!(home_on_pitch, 1);
    assert_eq!(away_on_pitch, 2);

    for blocker in state.get_players_on_pitch_in_team(botbowl_engine::core::model::TeamType::Away) {
        assert_eq!(
            blocker.position.x, 3,
            "blockers should sit between carrier and endzone"
        );
    }

    let ctx = botbowl_curriculum::LectureContext::from_state(&state);
    assert_eq!(lecture.evaluate(&state, &ctx), LectureStatus::InProgress);
}

#[test]
#[ignore = "bot benchmark — run with --ignored"]
fn medium_scripted_dominates_random() {
    let lecture = ScoreTdMedium::new();

    let mut random_agent = RandomBot::new();
    let random_stats = run_trials(&lecture, &mut random_agent, 5_000, 0xBEEF_F00D, 300);
    let random_rate = random_stats.success_rate();

    let mut scripted_agent = ScriptedBot::new();
    let scripted_stats = run_trials(&lecture, &mut scripted_agent, 1_000, 0xFACE_F00D, 300);
    let scripted_rate = scripted_stats.success_rate();

    eprintln!(
        "ScoreTdMedium random:   trials={} successes={} rate={:.4}",
        random_stats.trials, random_stats.successes, random_rate
    );
    eprintln!(
        "ScoreTdMedium scripted: trials={} successes={} rate={:.4}",
        scripted_stats.trials, scripted_stats.successes, scripted_rate
    );

    // The pitch is wide enough that a random agent occasionally routes
    // around the wall by picking a distant endzone destination, so the
    // absolute random rate hovers near the Easy lecture's value. What
    // demonstrates the lecture's value is the *gap*: scripted must
    // dramatically outperform random.
    assert!(
        scripted_rate >= 0.70,
        "scripted bot success rate {:.4} on Medium below 0.70 — improve the planner",
        scripted_rate
    );
    assert!(
        scripted_rate - random_rate >= 0.50,
        "scripted-vs-random gap {:.4} too small (scripted={:.4} random={:.4}) — \
         the scripted bot isn't meaningfully better than random on this lecture",
        scripted_rate - random_rate,
        scripted_rate,
        random_rate
    );
    assert_eq!(
        scripted_stats.timeouts, 0,
        "scripted bot trials must terminate"
    );
}

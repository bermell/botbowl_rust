//! Headless training-data generator (grand-plan steps 6–7).
//!
//! Drives the MCTS bot through full self-play games or curriculum lecture
//! trials, harvesting a [`botbowl_data::Sample`] at every agent decision
//! (state + raw search distribution + root value), then backfills the
//! drive/game outcome and writes one [`botbowl_data::Trajectory`] per
//! game/trial as a JSONL line — pinned to the git commit and board config
//! that produced it.
//!
//! Usage:
//! ```text
//! # self-play (both teams MCTS)
//! botbowl-ui dataset --mode self-play --games 4 --mcts-time-ms 150 --out data.jsonl
//! # curriculum (MCTS agent vs RandomBot)
//! botbowl-ui dataset --mode curriculum --lecture "Score TD" --difficulty easy \
//!     --games 20 --mcts-iters 400 --out score_td.jsonl
//! ```

use std::io;
use std::time::Duration;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use botbowl_curriculum::{
    available_lectures, generate_random_start, make_lecture, Difficulty, LectureContext, LectureStatus,
};
use botbowl_data::{DatasetWriter, Outcome, Sample, Trajectory, TrajectoryMeta};
use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::TeamType;
use botbowl_mcts::{MctsBot, SearchBudget};

use crate::cli::{CliDifficulty, CliEvaluator, DatasetArgs, DatasetMode};

// Mirror the seed-mixing constants in `botbowl-curriculum`'s runner so a
// curriculum dataset trial is reproducible against `run_trials` for the
// same seed.
const OPPONENT_SEED_MIX: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const AGENT_SEED_MIX: u64 = 0x5A5A_5A5A_5A5A_5A5A;

pub fn run(args: DatasetArgs) -> io::Result<()> {
    let mut writer = if args.truncate {
        DatasetWriter::create(&args.out)?
    } else {
        DatasetWriter::append(&args.out)?
    };

    let mut total_samples = 0usize;
    let mut written = 0u32;

    for g in 0..args.games {
        let seed = args.seed.wrapping_add(g as u64);
        let traj = match args.mode {
            DatasetMode::SelfPlay => Some(self_play_trajectory(&args, seed)),
            DatasetMode::RandomStart => Some(random_start_trajectory(&args, seed)),
            DatasetMode::Curriculum => match curriculum_trajectory(&args, seed) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{e}");
                    return Ok(());
                }
            },
        };
        if let Some(traj) = traj {
            total_samples += traj.samples.len();
            written += 1;
            println!(
                "[{}/{}] seed={seed} samples={} z_home={:+} score={}-{}",
                g + 1,
                args.games,
                traj.samples.len(),
                traj.outcome.z_home,
                traj.outcome.home_score,
                traj.outcome.away_score,
            );
            writer.write(&traj)?;
            writer.flush()?;
        }
    }

    println!(
        "wrote {written} trajectories / {total_samples} samples to {} (commit {}{})",
        args.out,
        botbowl_data::git_commit(),
        if botbowl_data::git_dirty() { "-dirty" } else { "" },
    );
    Ok(())
}

fn make_mcts(args: &DatasetArgs) -> MctsBot {
    let budget = match args.mcts_time_ms {
        Some(ms) => SearchBudget::Time(Duration::from_millis(ms)),
        None => SearchBudget::Iterations(args.mcts_iters),
    };
    let bot = MctsBot::new(budget).with_workers(args.mcts_workers);
    match args.evaluator {
        CliEvaluator::Heuristic => bot,
        CliEvaluator::PureTd => bot.with_pure_td(),
    }
}

fn budget_label(args: &DatasetArgs) -> String {
    let eval = match args.evaluator {
        CliEvaluator::Heuristic => "heuristic",
        CliEvaluator::PureTd => "pure-td",
    };
    match args.mcts_time_ms {
        Some(ms) => format!("mcts(time={ms}ms,workers={},eval={eval})", args.mcts_workers),
        None => format!("mcts(iters={},workers={},eval={eval})", args.mcts_iters, args.mcts_workers),
    }
}

/// Play `state` to completion with MctsBot on both teams, sampling both
/// teams' decisions.
fn mcts_vs_mcts_samples(state: &mut GameState, args: &DatasetArgs, seed: u64) -> Vec<Sample> {
    let mut home = make_mcts(args);
    let mut away = make_mcts(args);
    home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
    away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));

    let mut samples: Vec<Sample> = Vec::new();
    let mut steps = 0u32;

    while !state.info.game_over && steps < args.max_steps {
        let action = match state.available_actions.team {
            Some(TeamType::Home) => {
                let (a, s) = home.get_action_with_record(state);
                samples.push(s);
                a
            }
            Some(TeamType::Away) => {
                let (a, s) = away.get_action_with_record(state);
                samples.push(s);
                a
            }
            // Under RollDice, `step` auto-resolves chance internally, so a
            // running game always presents a team to act until game-over.
            None => break,
        };
        state.step(action).expect("engine step failed during self-play");
        steps += 1;
    }
    samples
}

/// One full MctsBot-vs-MctsBot game, from kickoff.
fn self_play_trajectory(args: &DatasetArgs, seed: u64) -> Trajectory {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let samples = mcts_vs_mcts_samples(&mut state, args, seed);

    let label = budget_label(args);
    let meta = TrajectoryMeta::new("self-play", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "self-play")
        .with_extra("max_steps", args.max_steps.to_string());
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One MctsBot-vs-MctsBot game from a randomized mid-game state (plan 019).
fn random_start_trajectory(args: &DatasetArgs, seed: u64) -> Trajectory {
    let cfg = args.bias.to_config();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = generate_random_start(&cfg, &mut rng);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let (start_half, start_home_turn, start_away_turn) =
        (state.info.half, state.info.home_turn, state.info.away_turn);
    let start_score = format!("{}-{}", state.home.score, state.away.score);
    let samples = mcts_vs_mcts_samples(&mut state, args, seed);

    let label = budget_label(args);
    let bias = &args.bias;
    let meta = TrajectoryMeta::new("random-start", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "random-start")
        .with_extra("max_steps", args.max_steps.to_string())
        .with_extra("ball_distance", bias.ball_distance.to_string())
        .with_extra("front_line", bias.front_line.to_string())
        .with_extra("mark_teammate", bias.mark_teammate.to_string())
        .with_extra("mark_opponent", bias.mark_opponent.to_string())
        .with_extra("own_side", bias.own_side.to_string())
        .with_extra("temperature", bias.temperature.to_string())
        .with_extra("carried_prob", bias.carried_prob.to_string())
        .with_extra("line_fraction", bias.line_fraction.to_string())
        .with_extra("pocket_fraction", bias.pocket_fraction.to_string())
        .with_extra("start_half", start_half.to_string())
        .with_extra("start_home_turn", start_home_turn.to_string())
        .with_extra("start_away_turn", start_away_turn.to_string())
        .with_extra("start_score", start_score);
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One curriculum lecture trial: MctsBot agent vs RandomBot opponent.
fn curriculum_trajectory(args: &DatasetArgs, seed: u64) -> Result<Option<Trajectory>, String> {
    let name = args
        .lecture
        .as_deref()
        .ok_or_else(|| "curriculum mode requires --lecture NAME".to_string())?;
    let difficulty = match args.difficulty {
        CliDifficulty::Easy => Difficulty::Easy,
        CliDifficulty::Medium => Difficulty::Medium,
        CliDifficulty::Hard => Difficulty::Hard,
    };
    let lecture = make_lecture(name, difficulty).ok_or_else(|| {
        let mut msg = format!("unknown lecture: {name:?} ({difficulty:?}).\nAvailable lectures:");
        for (n, d) in available_lectures() {
            msg.push_str(&format!("\n  --lecture {n:?} --difficulty {d:?}"));
        }
        msg
    })?;

    let agent_team = lecture.agent_team();
    let opponent_team = match agent_team {
        TeamType::Home => TeamType::Away,
        TeamType::Away => TeamType::Home,
    };

    let mut setup_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = lecture.setup(&mut setup_rng);
    state.set_logging_state(false);
    let context = LectureContext::from_state(&state);

    let mut opponent = RandomBot::new();
    opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ OPPONENT_SEED_MIX));
    let mut agent = make_mcts(args);
    agent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ AGENT_SEED_MIX));

    let board_dims = state.board_dims;
    let mut samples: Vec<Sample> = Vec::new();
    let mut status = lecture.evaluate(&state, &context);
    let mut steps = 0u32;

    while status == LectureStatus::InProgress && steps < args.max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == agent_team => {
                let (a, s) = agent.get_action_with_record(&state);
                samples.push(s);
                Some(a)
            }
            Some(t) if t == opponent_team => Some(opponent.get_action(&state)),
            Some(_) | None => None,
        };
        let Some(action) = action else { break };
        state.step(action).expect("engine step failed during lecture");
        steps += 1;
        status = lecture.evaluate(&state, &context);
    }

    let meta = TrajectoryMeta::new(lecture.name(), board_dims)
        .with_bots(budget_label(args), "random")
        .with_seed(seed)
        .with_extra("mode", "curriculum")
        .with_extra("difficulty", format!("{difficulty:?}"))
        .with_extra("lecture_status", format!("{status:?}"));
    let outcome = Outcome::from_state(&state, Some(format!("{status:?}")));
    Ok(Some(Trajectory::new(meta, samples, outcome)))
}

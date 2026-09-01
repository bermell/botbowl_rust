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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
use botbowl_nn::eval::NnEvaluator;

use crate::cli::{CliDifficulty, CliEvaluator, DatasetArgs, DatasetMode};

// Mirror the seed-mixing constants in `botbowl-curriculum`'s runner so a
// curriculum dataset trial is reproducible against `run_trials` for the
// same seed.
const OPPONENT_SEED_MIX: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const AGENT_SEED_MIX: u64 = 0x5A5A_5A5A_5A5A_5A5A;

/// Game workers only orchestrate — the search runs on `MctsBot`'s own
/// threads — but they do build a `GameState` on the stack, so match the
/// engine's generous convention rather than the 2 MB default.
const GAME_STACK_SIZE: usize = 16 * 1024 * 1024;

/// What the parallel game workers share.
struct RunState {
    /// One JSONL line per trajectory. `serde_json` writes straight into
    /// the `BufWriter`, so concurrent writes would interleave *within* a
    /// line, not merely reorder lines — this lock is load-bearing.
    writer: Mutex<DatasetWriter>,
    /// Work is handed out one game at a time rather than in static
    /// chunks: random-start games differ several-fold in length, so a
    /// fixed split would leave workers idle at the tail.
    next_game: AtomicU32,
    total_samples: AtomicUsize,
    written: AtomicU32,
    /// Set when a worker hits a fatal configuration error (a bad lecture
    /// name), so its peers stop instead of repeating the same failure
    /// once per remaining game.
    stop: AtomicBool,
    per_game_profile: bool,
}

/// Pull games off `state.next_game` until they run out.
///
/// Line order in the output is no longer game order once `parallel > 1`.
/// That is safe: every consumer is line-oriented (`prepare` streams the
/// JSONL) and each trajectory carries its own seed in `TrajectoryMeta`,
/// so a run is still fully identifiable — but it does mean two runs of
/// the same command produce the same *set* of lines in a different order.
fn run_games(
    args: &DatasetArgs,
    nn: Option<&Arc<NnEvaluator>>,
    state: &RunState,
    t_start: Instant,
    fw0: u64,
    ns0: u64,
) -> io::Result<()> {
    loop {
        if state.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let g = state.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= args.games {
            return Ok(());
        }
        let seed = args.seed.wrapping_add(g as u64);
        let traj = match args.mode {
            DatasetMode::SelfPlay => Some(self_play_trajectory(args, nn, seed)),
            DatasetMode::RandomStart => Some(random_start_trajectory(args, nn, seed)),
            DatasetMode::Curriculum => match curriculum_trajectory(args, nn, seed) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{e}");
                    state.stop.store(true, Ordering::Relaxed);
                    return Ok(());
                }
            },
        };
        let Some(traj) = traj else { continue };
        state.total_samples.fetch_add(traj.samples.len(), Ordering::Relaxed);
        let done = state.written.fetch_add(1, Ordering::Relaxed) + 1;
        println!(
            "[{}/{}] seed={seed} samples={} z_home={:+} score={}-{}",
            done,
            args.games,
            traj.samples.len(),
            traj.outcome.z_home,
            traj.outcome.home_score,
            traj.outcome.away_score,
        );
        {
            let mut w = state.writer.lock().expect("writer mutex");
            w.write(&traj)?;
            w.flush()?;
        }
        if state.per_game_profile {
            let (fw, ns) = botbowl_nn::eval::profile_counters();
            println!(
                "    NN_PROFILE game forwards={} inference_ms={} elapsed_ms={}",
                fw - fw0,
                (ns - ns0) / 1_000_000,
                t_start.elapsed().as_millis()
            );
        }
    }
}

pub fn run(args: DatasetArgs) -> io::Result<()> {
    // Load the ONNX evaluator once; every bot in every game shares the
    // Arc (the net is frozen — pure function of state).
    let nn: Option<Arc<NnEvaluator>> = match args.evaluator {
        CliEvaluator::Nn | CliEvaluator::NnValue => {
            let path = args.model.as_deref().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--evaluator nn/nn-value requires --model PATH")
            })?;
            let server = crate::cli::nn_server_path(args.nn_server.as_deref());
            let eval = NnEvaluator::from_path_with_server(path, server.as_deref())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("failed to load {path}: {e}")))?;
            Some(Arc::new(eval))
        }
        _ => None,
    };

    let writer = if args.truncate {
        DatasetWriter::create(&args.out)?
    } else {
        DatasetWriter::append(&args.out)?
    };

    // Plan 024 Stage 0: wall clock of the whole generation loop, against
    // which the NN forward counters give the inference share directly
    // (no cross-arm subtraction needed).
    let t_start = Instant::now();
    let (fw0, ns0) = botbowl_nn::eval::profile_counters();

    // Plan 024 Stage 4: games are embarrassingly parallel — each one
    // builds its own `GameState`, its own bots and its own RNG from its
    // own seed — so running several at once costs nothing in search
    // fidelity and is the only way a *single* process (the eval phase, or
    // a RAM-bound generate phase) can offer the sidecar more than one
    // concurrent request. See `run_games`.
    let parallel = args.parallel_games.clamp(1, args.games.max(1)) as usize;
    let state = RunState {
        writer: Mutex::new(writer),
        next_game: AtomicU32::new(0),
        total_samples: AtomicUsize::new(0),
        written: AtomicU32::new(0),
        stop: AtomicBool::new(false),
        // The forward counters are process-global, so a per-game delta is
        // only meaningful while one game runs at a time.
        per_game_profile: botbowl_nn::eval::profile_enabled() && parallel == 1,
    };

    if parallel == 1 {
        run_games(&args, nn.as_ref(), &state, t_start, fw0, ns0)?;
    } else {
        println!("running {parallel} games in parallel ({} concurrent inference streams)", parallel);
        let mut errs: Vec<io::Error> = Vec::new();
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for i in 0..parallel {
                let st = &state;
                let args = &args;
                let nn = nn.as_ref();
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("game-{i}"))
                        .stack_size(GAME_STACK_SIZE)
                        .spawn_scoped(s, move || run_games(args, nn, st, t_start, fw0, ns0))
                        .expect("spawn game worker"),
                );
            }
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => errs.push(e),
                    Err(_) => errs.push(io::Error::other("a game worker panicked")),
                }
            }
        });
        if let Some(e) = errs.into_iter().next() {
            return Err(e);
        }
    }

    let total_samples = state.total_samples.load(Ordering::Relaxed);
    let written = state.written.load(Ordering::Relaxed);
    // Every game already flushed under the lock; drop it here anyway so
    // the file is closed before the summary line claims it was written.
    drop(state.writer);

    if botbowl_nn::eval::profile_enabled() {
        let (fw, ns) = botbowl_nn::eval::profile_counters();
        let (forwards, nanos) = (fw - fw0, ns - ns0);
        let wall_ms = t_start.elapsed().as_secs_f64() * 1e3;
        let total_ms = nanos as f64 / 1e6;
        let mean_us = if forwards > 0 { nanos as f64 / forwards as f64 / 1e3 } else { 0.0 };
        println!(
            "NN_PROFILE forwards={forwards} total_ms={total_ms:.0} mean_us={mean_us:.0} \
wall_ms={wall_ms:.0} share={:.3} games={written} forwards_per_game={:.0} \
forwards_per_decision={:.1}",
            if wall_ms > 0.0 { total_ms / wall_ms } else { 0.0 },
            if written > 0 { forwards as f64 / written as f64 } else { 0.0 },
            if total_samples > 0 { forwards as f64 / total_samples as f64 } else { 0.0 },
        );
    }

    if let Some((served, fell_back)) = nn.as_ref().and_then(|n| n.remote_stats()) {
        println!(
            "NN_SERVER served={served} fell_back_to_tract={fell_back}{}",
            if fell_back > 0 { "  <-- the server was unreachable for some forwards" } else { "" }
        );
    }
    println!(
        "wrote {written} trajectories / {total_samples} samples to {} (commit {}{})",
        args.out,
        botbowl_data::git_commit(),
        if botbowl_data::git_dirty() { "-dirty" } else { "" },
    );
    Ok(())
}

fn make_mcts(args: &DatasetArgs, nn: Option<&Arc<NnEvaluator>>) -> MctsBot {
    let budget = match args.mcts_time_ms {
        Some(ms) => SearchBudget::Time(Duration::from_millis(ms)),
        None => SearchBudget::Iterations(args.mcts_iters),
    };
    let bot = MctsBot::new(budget).with_workers(args.mcts_workers);
    match args.evaluator {
        CliEvaluator::Heuristic => bot,
        CliEvaluator::PureTd => bot.with_pure_td(),
        CliEvaluator::Nn => bot.with_evaluator(Arc::clone(nn.expect("nn evaluator loaded in run()"))),
        CliEvaluator::NnValue => bot.with_nn_value(Arc::clone(nn.expect("nn evaluator loaded in run()"))),
    }
}

fn budget_label(args: &DatasetArgs) -> String {
    let eval = match args.evaluator {
        CliEvaluator::Heuristic => "heuristic".to_string(),
        CliEvaluator::PureTd => "pure-td".to_string(),
        CliEvaluator::Nn => format!("nn:{}", args.model.as_deref().unwrap_or("?")),
        CliEvaluator::NnValue => format!("nn-value:{}", args.model.as_deref().unwrap_or("?")),
    };
    match args.mcts_time_ms {
        Some(ms) => format!("mcts(time={ms}ms,workers={},eval={eval})", args.mcts_workers),
        None => format!("mcts(iters={},workers={},eval={eval})", args.mcts_iters, args.mcts_workers),
    }
}

/// Play `state` with MctsBot on both teams, sampling both teams'
/// decisions, until game over, the step cap, or `stop(state)` — the
/// latter lets random-start mode end the trajectory at the end of the
/// current drive instead of playing the game out.
fn mcts_vs_mcts_samples(
    state: &mut GameState,
    args: &DatasetArgs,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
    stop: impl Fn(&GameState) -> bool,
) -> Vec<Sample> {
    let mut home = make_mcts(args, nn);
    let mut away = make_mcts(args, nn);
    home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
    away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));

    let mut samples: Vec<Sample> = Vec::new();
    let mut steps = 0u32;

    while !state.info.game_over && !stop(state) && steps < args.max_steps {
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
fn self_play_trajectory(args: &DatasetArgs, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Trajectory {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let samples = mcts_vs_mcts_samples(&mut state, args, nn, seed, |_| false);

    let label = budget_label(args);
    let meta = TrajectoryMeta::new("self-play", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "self-play")
        .with_extra("max_steps", args.max_steps.to_string());
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One MctsBot-vs-MctsBot **drive** from a randomized mid-game state
/// (plan 019): the trajectory ends when either team scores, the half
/// ends, or the game ends — never plays into the next drive. Everything
/// after the drive resolves would be downstream of self-play (correlated
/// states, bot-chosen kickoff formations) instead of the diverse random
/// placement this mode exists to provide (plan 020).
fn random_start_trajectory(args: &DatasetArgs, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Trajectory {
    let mut cfg = args.bias.to_config();
    // Alternate the placement temperature per game so the corpus mixes
    // sharp (clustered) and flat (scattered) player distributions.
    if seed % 2 == 1 {
        cfg.temperature = args.bias.temperature2;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = generate_random_start(&cfg, &mut rng);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let (start_half, start_home_turn, start_away_turn) =
        (state.info.half, state.info.home_turn, state.info.away_turn);
    let (start_home_score, start_away_score) = (state.home.score, state.away.score);
    let start_score = format!("{start_home_score}-{start_away_score}");
    let samples = mcts_vs_mcts_samples(&mut state, args, nn, seed, |s| {
        s.home.score != start_home_score || s.away.score != start_away_score || s.info.half != start_half
    });

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
        .with_extra("temperature", cfg.temperature.to_string())
        .with_extra("temperature2", bias.temperature2.to_string())
        .with_extra("drive_bounded", "true".to_string())
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
fn curriculum_trajectory(
    args: &DatasetArgs,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
) -> Result<Option<Trajectory>, String> {
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
    let mut agent = make_mcts(args, nn);
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

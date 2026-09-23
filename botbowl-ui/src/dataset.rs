//! Headless training-data generator (grand-plan steps 6–7).
//!
//! The game-playing core lives in `botbowl_play::generate` (plan 041 phase
//! 0); this module is the single-process shell around it: CLI flags to a
//! `GenerateConfig`, a shared JSONL writer, N game workers pulling seeds off
//! one counter, and the profile/provenance lines the loop scripts grep for.
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

use botbowl_data::DatasetWriter;
use botbowl_mcts::SearchBudget;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::bots::{load_nn, SearchConfig};
use botbowl_play::generate::{play_trajectory, GenerateConfig};
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::DatasetArgs;

fn config_of(args: &DatasetArgs) -> io::Result<GenerateConfig> {
    let board_sizes = args
        .sizes
        .to_dist()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let budget = match args.mcts_time_ms {
        Some(ms) => SearchBudget::Time(Duration::from_millis(ms)),
        None => SearchBudget::Iterations(args.mcts_iters),
    };
    // Plan 043: a named preset replaces the bot's whole configuration, environment included.
    // Unset leaves the historical behaviour untouched.
    let preset = args
        .bot_config
        .as_deref()
        .map(botbowl_play::bots::load_mcts_config)
        .transpose()?;
    Ok(GenerateConfig {
        mode: args.mode.into(),
        search: SearchConfig {
            budget,
            workers: args.mcts_workers,
            // `dataset` has never exposed these; the bot's own defaults
            // (env-driven) apply, exactly as before the extraction.
            puct: None,
            horizon_turns: None,
            backup: None,
            fpu_reduction: None,
            config: preset.as_ref().map(|p| p.config),
        },
        config_name: preset.map(|p| p.name),
        evaluator: args.evaluator.into(),
        model: args.model.clone(),
        max_steps: args.max_steps,
        lecture: args.lecture.clone(),
        difficulty: args.difficulty.into(),
        bias: args.bias.to_bias(),
        board_sizes,
    })
}

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
    cfg: &GenerateConfig,
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
        let traj = match play_trajectory(cfg, nn, seed) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                state.stop.store(true, Ordering::Relaxed);
                return Ok(());
            }
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
    let cfg = config_of(&args)?;
    if let Some(d) = &cfg.board_sizes {
        println!(
            "board sizes: {} -> {}",
            d.label,
            d.table()
                .iter()
                .map(|(b, p)| format!("{b} {:.1}%", p * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // Load the ONNX evaluator once; every bot in every game shares the
    // Arc (the net is frozen — pure function of state).
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        cfg.evaluator,
        args.model.as_deref(),
        "--evaluator nn/nn-value requires --model PATH",
        server.as_deref(),
    )?;

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
        run_games(&args, &cfg, nn.as_ref(), &state, t_start, fw0, ns0)?;
    } else {
        println!(
            "running {parallel} games in parallel ({} concurrent inference streams)",
            parallel
        );
        let mut errs: Vec<io::Error> = Vec::new();
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for i in 0..parallel {
                let st = &state;
                let args = &args;
                let cfg = &cfg;
                let nn = nn.as_ref();
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("game-{i}"))
                        .stack_size(GAME_STACK_SIZE)
                        .spawn_scoped(s, move || run_games(args, cfg, nn, st, t_start, fw0, ns0))
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
        let mean_us = if forwards > 0 {
            nanos as f64 / forwards as f64 / 1e3
        } else {
            0.0
        };
        println!(
            "NN_PROFILE forwards={forwards} total_ms={total_ms:.0} mean_us={mean_us:.0} \
wall_ms={wall_ms:.0} share={:.3} games={written} forwards_per_game={:.0} \
forwards_per_decision={:.1}",
            if wall_ms > 0.0 { total_ms / wall_ms } else { 0.0 },
            if written > 0 {
                forwards as f64 / written as f64
            } else {
                0.0
            },
            if total_samples > 0 {
                forwards as f64 / total_samples as f64
            } else {
                0.0
            },
        );
    }

    if let Some((served, fell_back)) = nn.as_ref().and_then(|n| n.remote_stats()) {
        println!(
            "NN_SERVER served={served} fell_back_to_tract={fell_back}{}",
            if fell_back > 0 {
                "  <-- the server was unreachable for some forwards"
            } else {
                ""
            }
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

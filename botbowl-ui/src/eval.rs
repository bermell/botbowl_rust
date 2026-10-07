//! Evaluation report card for a candidate bot (plan 020).
//!
//! Two low-variance instruments, replacing TDs/game in stochastic
//! random-start self-play (±0.5 noise at 12 games made generations
//! indistinguishable):
//!
//! 1. **Lecture battery** — every `botbowl-curriculum` lecture × difficulty,
//!    N trials each, success rate as the metric. Short episodes, binary
//!    outcomes, tight confidence intervals, and diagnostic: failing
//!    "score TD medium" while passing "easy" says *what* the bot can't do.
//! 2. **Opponent ladder** — full games from kickoff against fixed opponents
//!    (RandomBot floor, ScriptedBot, heuristic-MCTS bar), alternating
//!    Home/Away on a fixed seed set so candidates are compared on
//!    identical situations. Win rate + TDs for/against.
//!
//! The per-game core and the report types live in `botbowl_play::eval`
//! (plan 041 phase 0); this module is the single-process shell: CLI flags
//! to configs, rung workers, the per-game JSONL file, the printed table.
//!
//! Output: a printed table and (optionally) a JSON report for tracking
//! across generations.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use botbowl_curriculum::{available_lectures, make_lecture, run_trials, TrialStats};
use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::model::BoardDims;
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::PuctMode;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::board_sizes::board_label;
use botbowl_play::bots::{
    candidate_label, evaluator_label, load_mcts_config, load_nn, make_candidate_bot, make_mcts, resolve_puct,
    CandidateBot, Evaluator, NamedConfig, SearchConfig,
};
use botbowl_play::drives::{
    drive_assignment, drive_rung_name, play_drive_game, position_state, DriveRung, PositionSet,
};
use botbowl_play::eval::{ladder_assignment, play_ladder_game, rung_name, LadderRow, LectureRow, Report};
use botbowl_play::stats::Sprt;
use botbowl_play::trace::ReuseTraceWriter;
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::EvalArgs;

/// Max micro-steps per lecture trial (mirrors the curriculum CLI default).
const LECTURE_MAX_STEPS: u32 = 2000;

/// Refuse to start rather than run the wrong arm of a multi-hour head-to-head.
fn puct_of(mode: Option<&str>, c: Option<f32>) -> Option<PuctMode> {
    resolve_puct(mode, c).unwrap_or_else(|e| panic!("--puct-mode: {e}"))
}

/// The candidate's search knobs. `horizon_turns` and `fpu_reduction` are always `Some` (the CLI
/// defaults stand in for the bot's); `puct` is `None` unless `--puct-mode`/`--puct-c` is given,
/// so `BLOOD_MCTS_PUCT_*` applies here exactly as it does in `dataset`.
///
/// Plan 043: `--bot-config` replaces all of them with a named preset. The per-knob flags are
/// `conflicts_with` it in clap, so the two can never be mixed — a run is described entirely by a
/// preset or entirely by flags.
fn candidate_search(args: &EvalArgs, preset: Option<&NamedConfig>) -> SearchConfig {
    SearchConfig {
        budget: botbowl_mcts::SearchBudget::Iterations(args.mcts_iters),
        workers: args.mcts_workers,
        puct: preset
            .is_none()
            .then(|| puct_of(args.puct_mode.as_deref(), args.puct_c))
            .flatten(),
        horizon_turns: preset.is_none().then_some(args.horizon_turns),
        fpu_reduction: preset.is_none().then_some(args.fpu_reduction),
        config: preset.map(|p| p.config),
    }
}

/// The opponent's search knobs. Unset `--vs-*` means "match the candidate",
/// so existing invocations are unchanged and setting one flag alone makes
/// it a head-to-head on that knob.
///
/// `--vs-config` follows the same rule: unset, the opponent inherits the candidate's preset, so
/// `--bot-config` alone configures both sides and setting `--vs-config` alone is a
/// configuration head-to-head — the same net under two configurations.
fn opponent_search(args: &EvalArgs, preset: Option<&NamedConfig>) -> SearchConfig {
    SearchConfig {
        budget: botbowl_mcts::SearchBudget::Iterations(args.opponent_iters.unwrap_or(args.mcts_iters)),
        workers: args.mcts_workers,
        puct: preset
            .is_none()
            .then(|| {
                puct_of(
                    args.vs_puct_mode.as_deref().or(args.puct_mode.as_deref()),
                    args.vs_puct_c.or(args.puct_c),
                )
            })
            .flatten(),
        horizon_turns: preset
            .is_none()
            .then(|| args.vs_horizon_turns.unwrap_or(args.horizon_turns)),
        fpu_reduction: preset
            .is_none()
            .then(|| args.vs_fpu_reduction.unwrap_or(args.fpu_reduction)),
        config: preset.map(|p| p.config),
    }
}

/// Resolve `--bot-config` and `--vs-config` once, up front, so a bad path or a typo'd knob fails
/// before hours of games rather than after.
fn presets(args: &EvalArgs) -> io::Result<(Option<NamedConfig>, Option<NamedConfig>)> {
    let candidate = args.bot_config.as_deref().map(load_mcts_config).transpose()?;
    // Unset `--vs-config` inherits the candidate's, matching every other `--vs-` flag.
    let opponent = match args.vs_config.as_deref() {
        Some(p) => Some(load_mcts_config(p)?),
        None => candidate.clone(),
    };
    Ok((candidate, opponent))
}

fn candidate_bot(args: &EvalArgs, preset: Option<&NamedConfig>, nn: Option<&Arc<NnEvaluator>>) -> Box<dyn Bot> {
    make_candidate_bot(
        CandidateBot::from(args.candidate_bot),
        &candidate_search(args, preset),
        Evaluator::from(args.evaluator),
        nn,
    )
}

/// What the parallel rung workers share. Every `LadderRow` field is a
/// commutative counter, so one `Mutex` around the whole row is both
/// correct and cheap — a rung game takes seconds, the lock is held for
/// microseconds.
struct RungState {
    row: Mutex<LadderRow>,
    /// Handed out one game at a time. Full games vary several-fold in
    /// length, so a static split would idle workers at the tail.
    next_game: AtomicU32,
    /// Plan 051: the rung's SPRT has a verdict, so no more games are handed out. Games already
    /// in flight finish and are recorded; overshoot does not bias a sequential test.
    decided: AtomicBool,
    /// `writeln!` of a whole JSONL line must be atomic against its peers.
    per_game: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// Plan 043 `--trace-reuse`: `None` unless the flag was given. Shared across the rung's
    /// workers, and its own mutex keeps a row atomic.
    reuse_trace: Option<ReuseTraceWriter>,
}

/// Where a rung's games are played (plan 051): full games from kickoff on a board (`None` = the
/// env board), or paired drives from a frozen position set on the set's board.
#[derive(Clone, Copy)]
enum Venue<'a> {
    Games(Option<BoardDims>),
    Drives(BoardDims, &'a DriveRung),
}

impl Venue<'_> {
    fn board(&self) -> Option<BoardDims> {
        match self {
            Venue::Games(b) => *b,
            Venue::Drives(b, _) => Some(*b),
        }
    }

    /// `opponent@board` for games (plan 042), `opponent drives(set)@board` for drives.
    fn rung_name(&self, opponent: &str) -> String {
        match self {
            Venue::Games(b) => rung_name(opponent, *b),
            Venue::Drives(b, d) => drive_rung_name(opponent, &d.set, *b),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ladder_rung(
    args: &EvalArgs,
    preset: Option<&NamedConfig>,
    nn: Option<&Arc<NnEvaluator>>,
    opponent: &str,
    venue: Venue<'_>,
    games: u32,
    sprt: Option<Sprt>,
    make_opponent: impl Fn() -> Box<dyn Bot> + Sync,
) -> LadderRow {
    // Plan 042: on a multi-size ladder the rung is `opponent@board`, so the
    // per-game lines and the report rows group by board without a new
    // file format; on an env-board ladder it is the bare opponent name.
    let name = &venue.rung_name(opponent);
    // Per-game side-relative record (plan 023 deferred item 5): the pooled
    // report line cannot distinguish a scoring-rate bias from a
    // win-conversion one, nor see who received the opening kickoff.
    let per_game = args.per_game_out.as_ref().map(|path| {
        std::io::BufWriter::new(
            std::fs::File::options()
                .create(true)
                .append(true)
                .open(path)
                .expect("--per-game-out: cannot open"),
        )
    });
    let mut row = LadderRow::on_board(opponent, venue.board()).with_sprt(sprt);
    row.opponent = name.clone();
    let state = RungState {
        row: Mutex::new(row),
        decided: AtomicBool::new(false),
        next_game: AtomicU32::new(0),
        per_game: Mutex::new(per_game),
        reuse_trace: args
            .trace_reuse
            .as_deref()
            .map(|path| ReuseTraceWriter::create(path).expect("--trace-reuse: cannot open")),
    };

    // Plan 024 Stage 4b. Eval was the loop's one wholly serial phase, and
    // once generation got ~4x faster it became the dominant one (plan 022
    // measured 117-690 min against a generate phase now near 200). A rung
    // game is independent of its siblings — own `GameState`, own bots, own
    // seed derived from `g` — so this changes nothing about a result, only
    // how many run at once. It is also what lets the eval phase use
    // `--nn-server` at all: at one stream a batching server is *slower*
    // than tract.
    let parallel = args.parallel_games.clamp(1, games.max(1)) as usize;
    if parallel == 1 {
        run_rung_games(args, preset, nn, name, venue, games, &make_opponent, &state);
    } else {
        eprintln!("  vs {name}: {parallel} games in parallel");
        std::thread::scope(|s| {
            for i in 0..parallel {
                let st = &state;
                let mk = &make_opponent;
                std::thread::Builder::new()
                    .name(format!("rung-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(s, move || run_rung_games(args, preset, nn, name, venue, games, mk, st))
                    .expect("spawn rung worker");
            }
        });
    }
    eprintln!();

    state.row.into_inner().expect("row mutex").finish()
}

/// Play rung games off the shared counter until they run out.
///
/// Bots are built **here**, per worker, and never leave this thread —
/// which is what makes this sound despite `dyn Bot` having no `Send`
/// bound. It also matches the sequential behaviour: bots were already
/// reused across the games of a rung, and `MctsBot`'s cached tree is
/// discarded anyway when the horizon anchor fails to match at a new
/// game's kickoff.
#[allow(clippy::too_many_arguments)]
fn run_rung_games(
    args: &EvalArgs,
    preset: Option<&NamedConfig>,
    nn: Option<&Arc<NnEvaluator>>,
    name: &str,
    venue: Venue<'_>,
    games: u32,
    make_opponent: &(impl Fn() -> Box<dyn Bot> + Sync),
    state: &RungState,
) {
    let mut candidate = candidate_bot(args, preset, nn);
    let mut opponent = make_opponent();
    loop {
        if state.decided.load(Ordering::Relaxed) {
            return;
        }
        let g = state.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= games {
            return;
        }
        let line = match venue {
            Venue::Games(board) => {
                let (candidate_team, seed) = ladder_assignment(args.seed, g);
                play_ladder_game(
                    &mut *candidate,
                    &mut *opponent,
                    name,
                    g,
                    candidate_team,
                    seed,
                    args.max_steps,
                    board,
                    state.reuse_trace.as_ref(),
                )
            }
            Venue::Drives(board, set) => {
                let (i, attacks, dice) = drive_assignment(set.positions.len(), args.seed, g);
                let seed = set.positions[i];
                play_drive_game(
                    &mut *candidate,
                    &mut *opponent,
                    name,
                    g,
                    seed,
                    position_state(&set.bias, board, seed),
                    attacks,
                    dice,
                    args.max_steps,
                )
            }
        };

        if let Some(w) = state.per_game.lock().expect("per-game mutex").as_mut() {
            use std::io::Write;
            serde_json::to_writer(&mut *w, &line).expect("per-game log write failed");
            writeln!(w).expect("per-game log write failed");
        }

        let mut row = state.row.lock().expect("row mutex");
        row.record(&line);
        if row.decided() {
            state.decided.store(true, Ordering::Relaxed);
        }
        eprint!(
            "\r  vs {name}: {}/{} (W{} D{} L{}){}",
            row.games,
            games,
            row.wins,
            row.draws,
            row.losses,
            row.sprt
                .map(|s| format!(" LLR {:.2} [{:.2}, {:.2}] {:?}", s.llr, s.lower, s.upper, s.verdict))
                .unwrap_or_default()
        );
    }
}

pub fn run(args: EvalArgs) -> io::Result<()> {
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    // Plan 043: resolve the bot presets before anything else, so a bad path or a misspelled knob
    // fails in the first second rather than after the lecture battery.
    let (cand_preset, opp_preset) = presets(&args)?;
    let evaluator = Evaluator::from(args.evaluator);
    let nn = load_nn(
        evaluator,
        args.model.as_deref(),
        "--evaluator nn/nn-value requires --model PATH",
        server.as_deref(),
    )?;
    // Load the --vs-evaluator opponent's net up front so a bad path fails
    // before hours of fixed-rung games.
    let vs_evaluator = args.vs_evaluator.map(Evaluator::from);
    let vs_nn = match vs_evaluator {
        Some(vs) => load_nn(
            vs,
            args.vs_model.as_deref(),
            "--vs-evaluator nn/nn-value requires --vs-model PATH",
            server.as_deref(),
        )?,
        None => None,
    };

    let mut lectures: Vec<LectureRow> = Vec::new();
    if !args.skip_lectures {
        eprintln!("== lecture battery ({} trials per cell) ==", args.trials);
        for &(name, difficulty) in available_lectures() {
            let lecture = make_lecture(name, difficulty).expect("available_lectures entry must construct");
            let mut agent = candidate_bot(&args, cand_preset.as_ref(), nn.as_ref());
            // Lectures place players at hard-coded full-pitch coordinates;
            // on smaller compiled boards a cell can panic mid-setup. Run
            // the whole cell under catch_unwind (quiet panic hook) and
            // report it skipped rather than aborting the report card.
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_trials(&*lecture, &mut *agent, args.trials, args.seed, LECTURE_MAX_STEPS)
            }));
            std::panic::set_hook(hook);
            match result {
                Ok(stats @ TrialStats { .. }) => {
                    eprintln!(
                        "  {name:20} {difficulty:?}: {:.2} ({}/{} ok, {} fail, {} timeout)",
                        stats.success_rate(),
                        stats.successes,
                        stats.trials,
                        stats.failures,
                        stats.timeouts,
                    );
                    lectures.push(LectureRow {
                        lecture: name.to_string(),
                        difficulty: format!("{difficulty:?}"),
                        trials: stats.trials,
                        successes: stats.successes,
                        failures: stats.failures,
                        timeouts: stats.timeouts,
                        success_rate: stats.success_rate(),
                        skipped_board_too_small: false,
                    });
                }
                Err(_) => {
                    eprintln!("  {name:20} {difficulty:?}: skipped (board too small for lecture setup)");
                    lectures.push(LectureRow {
                        lecture: name.to_string(),
                        difficulty: format!("{difficulty:?}"),
                        trials: 0,
                        successes: 0,
                        failures: 0,
                        timeouts: 0,
                        success_rate: 0.0,
                        skipped_board_too_small: true,
                    });
                }
            }
        }
    }

    // Plan 042: every rung runs once per board; `[None]` is the env board.
    let boards = args
        .sizes
        .boards()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    // Plan 051: `--positions` replaces the boards with drive rungs, one per position set.
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidInput, e);
    let drive_sets: Vec<(BoardDims, DriveRung)> = match args.positions.as_deref() {
        Some(list) => list
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|path| {
                let set = PositionSet::load(path)?;
                Ok((set.board_dims()?, set.rung()?))
            })
            .collect::<Result<_, String>>()
            .map_err(invalid)?,
        None => Vec::new(),
    };
    let venues: Vec<Venue> = if drive_sets.is_empty() {
        boards.iter().map(|&b| Venue::Games(b)).collect()
    } else {
        drive_sets.iter().map(|(b, d)| Venue::Drives(*b, d)).collect()
    };
    let mut ladder: Vec<LadderRow> = Vec::new();
    if !args.skip_ladder {
        eprintln!(
            "== opponent ladder ({} {} per rung, {} on the vs rung{}) ==",
            args.games,
            if drive_sets.is_empty() { "games" } else { "drives" },
            args.vs_games.unwrap_or(args.games),
            if venues[0].board().is_some() {
                format!(
                    ", boards {}",
                    venues
                        .iter()
                        .filter_map(|v| v.board())
                        .map(board_label)
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            } else {
                String::new()
            }
        );
        let cand = candidate_search(&args, cand_preset.as_ref());
        let opp = opponent_search(&args, opp_preset.as_ref());
        if !args.skip_fixed_rungs {
            let wanted: Vec<&str> = args.rungs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
            for name in &wanted {
                if !matches!(*name, "random" | "scripted" | "mcts-heuristic") {
                    panic!("--rungs: expected `random`, `scripted` or `mcts-heuristic`, got `{name}`");
                }
            }
            for &venue in &venues {
                if wanted.contains(&"random") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "random",
                        venue,
                        args.games,
                        args.sprt,
                        || Box::new(RandomBot::new()),
                    ));
                }
                if wanted.contains(&"scripted") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "scripted",
                        venue,
                        args.games,
                        args.sprt,
                        || Box::new(ScriptedBot::new()),
                    ));
                }
                if wanted.contains(&"mcts-heuristic") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "mcts-heuristic",
                        venue,
                        args.games,
                        args.sprt,
                        || Box::new(make_mcts(&opp, Evaluator::Heuristic, None)),
                    ));
                }
            }
        }
        if let Some(vs) = vs_evaluator {
            // The label says how the opponent differs from the candidate. Under a preset the
            // per-knob fields are deliberately `None` — the configuration name is the difference,
            // and it is the thing you can look up in `cfgs/`.
            let label = if let (Some(o), Some(c)) = (&opp_preset, &cand_preset) {
                let base = evaluator_label(vs, args.vs_model.as_deref());
                if o.name == c.name {
                    format!("vs:{base} [{}]", o.name)
                } else {
                    format!("vs:{base} [{} v {}]", o.name, c.name)
                }
            } else {
                let (opp_puct, opp_horizon, opp_fpu) = (
                    opp.effective_puct(),
                    opp.horizon_turns.expect("set when no preset is named"),
                    opp.fpu_reduction.expect("set when no preset is named"),
                );
                let (cand_horizon, cand_fpu) = (
                    cand.horizon_turns.expect("set when no preset is named"),
                    cand.fpu_reduction.expect("set when no preset is named"),
                );
                format!(
                    "vs:{} [{}{}{}]",
                    evaluator_label(vs, args.vs_model.as_deref()),
                    opp_puct.label(),
                    if opp_horizon != cand_horizon {
                        format!(" horizon={opp_horizon}v{cand_horizon}")
                    } else {
                        String::new()
                    },
                    if opp_fpu != cand_fpu {
                        format!(" fpu_k={opp_fpu}v{cand_fpu}")
                    } else {
                        String::new()
                    }
                )
            };
            // The gating rung: `--vs-games` if given, else `--games`.
            let vs_games = args.vs_games.unwrap_or(args.games);
            for &venue in &venues {
                ladder.push(run_ladder_rung(
                    &args,
                    cand_preset.as_ref(),
                    nn.as_ref(),
                    &label,
                    venue,
                    vs_games,
                    args.sprt,
                    || Box::new(make_mcts(&opp, vs, vs_nn.as_ref())),
                ));
            }
        }
    }

    let report = Report {
        candidate: candidate_label(
            CandidateBot::from(args.candidate_bot),
            &candidate_search(&args, cand_preset.as_ref()),
            evaluator,
            args.model.as_deref(),
            cand_preset.as_ref().map(|p| p.name.as_str()),
        ),
        candidate_config: cand_preset.as_ref().map(|p| p.name.clone()),
        opponent_config: opp_preset.as_ref().map(|p| p.name.clone()),
        telemetry: Report::telemetry_of(&ladder),
        mcts_iters: args.mcts_iters,
        seed: args.seed,
        board_env: if venues[0].board().is_some() {
            venues
                .iter()
                .filter_map(|v| v.board())
                .map(board_label)
                .collect::<Vec<_>>()
                .join(",")
        } else {
            format!("{:?}", BoardDims::from_env())
        },
        git_commit: botbowl_data::git_commit().to_string(),
        git_dirty: botbowl_data::git_dirty(),
        lectures,
        ladder,
    };

    println!("\n== report card: {} ==", report.candidate);
    for l in &report.lectures {
        if l.skipped_board_too_small {
            println!(
                "  lecture {:24} {:8} skipped (board too small)",
                l.lecture, l.difficulty
            );
        } else {
            println!("  lecture {:24} {:8} {:.2}", l.lecture, l.difficulty, l.success_rate);
        }
    }
    for r in &report.ladder {
        println!("{}", r.report_line());
    }

    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
        println!("wrote {out}");
    }
    Ok(())
}

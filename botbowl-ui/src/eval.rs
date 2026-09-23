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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use botbowl_curriculum::{available_lectures, make_lecture, run_trials, TrialStats};
use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::model::BoardDims;
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::{BackupMode, PuctMode};
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::board_sizes::board_label;
use botbowl_play::bots::{
    candidate_label, evaluator_label, load_mcts_config, load_nn, make_candidate_bot, make_mcts, parse_backup,
    parse_puct, CandidateBot, Evaluator, NamedConfig, SearchConfig,
};
use botbowl_play::eval::{ladder_assignment, play_ladder_game, rung_name, LadderRow, LectureRow, Report};
use botbowl_play::trace::ReuseTraceWriter;
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::EvalArgs;

/// Max micro-steps per lecture trial (mirrors the curriculum CLI default).
const LECTURE_MAX_STEPS: u32 = 2000;

/// Refuse to start rather than run the wrong arm of a multi-hour head-to-head.
fn puct_of(mode: &str, c: Option<f32>) -> PuctMode {
    parse_puct(mode, c).unwrap_or_else(|e| panic!("--puct-mode: {e}"))
}

fn backup_of(s: &str) -> BackupMode {
    parse_backup(s).unwrap_or_else(|e| panic!("--backup: {e}"))
}

/// The candidate's search knobs. Every one is `Some`: `eval` has always
/// set them explicitly (the CLI defaults stand in for the bot's), so the
/// environment never reaches the candidate here.
///
/// Plan 043: `--bot-config` replaces all of them with a named preset. The per-knob flags are
/// `conflicts_with` it in clap, so the two can never be mixed — a run is described entirely by a
/// preset or entirely by flags.
fn candidate_search(args: &EvalArgs, preset: Option<&NamedConfig>) -> SearchConfig {
    SearchConfig {
        budget: botbowl_mcts::SearchBudget::Iterations(args.mcts_iters),
        workers: args.mcts_workers,
        puct: preset.is_none().then(|| puct_of(&args.puct_mode, args.puct_c)),
        horizon_turns: preset.is_none().then_some(args.horizon_turns),
        backup: preset.is_none().then(|| backup_of(&args.backup)),
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
        puct: preset.is_none().then(|| {
            puct_of(
                args.vs_puct_mode.as_deref().unwrap_or(&args.puct_mode),
                args.vs_puct_c.or(args.puct_c),
            )
        }),
        horizon_turns: preset
            .is_none()
            .then(|| args.vs_horizon_turns.unwrap_or(args.horizon_turns)),
        backup: preset
            .is_none()
            .then(|| backup_of(args.vs_backup.as_deref().unwrap_or(&args.backup))),
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
    /// `writeln!` of a whole JSONL line must be atomic against its peers.
    per_game: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// Plan 043 `--trace-reuse`: `None` unless the flag was given. Shared across the rung's
    /// workers, and its own mutex keeps a row atomic.
    reuse_trace: Option<ReuseTraceWriter>,
}

#[allow(clippy::too_many_arguments)]
fn run_ladder_rung(
    args: &EvalArgs,
    preset: Option<&NamedConfig>,
    nn: Option<&Arc<NnEvaluator>>,
    opponent: &str,
    board: Option<BoardDims>,
    games: u32,
    make_opponent: impl Fn() -> Box<dyn Bot> + Sync,
) -> LadderRow {
    // Plan 042: on a multi-size ladder the rung is `opponent@board`, so the
    // per-game lines and the report rows group by board without a new
    // file format; on an env-board ladder it is the bare opponent name.
    let name = &rung_name(opponent, board);
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
    let state = RungState {
        row: Mutex::new(LadderRow::on_board(opponent, board)),
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
        run_rung_games(args, preset, nn, name, board, games, &make_opponent, &state);
    } else {
        eprintln!("  vs {name}: {parallel} games in parallel");
        std::thread::scope(|s| {
            for i in 0..parallel {
                let st = &state;
                let mk = &make_opponent;
                std::thread::Builder::new()
                    .name(format!("rung-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(s, move || run_rung_games(args, preset, nn, name, board, games, mk, st))
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
    board: Option<BoardDims>,
    games: u32,
    make_opponent: &(impl Fn() -> Box<dyn Bot> + Sync),
    state: &RungState,
) {
    let mut candidate = candidate_bot(args, preset, nn);
    let mut opponent = make_opponent();
    loop {
        let g = state.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= games {
            return;
        }
        let (candidate_team, seed) = ladder_assignment(args.seed, g);
        let line = play_ladder_game(
            &mut *candidate,
            &mut *opponent,
            name,
            g,
            candidate_team,
            seed,
            args.max_steps,
            board,
            state.reuse_trace.as_ref(),
        );

        if let Some(w) = state.per_game.lock().expect("per-game mutex").as_mut() {
            use std::io::Write;
            serde_json::to_writer(&mut *w, &line).expect("per-game log write failed");
            writeln!(w).expect("per-game log write failed");
        }

        let mut row = state.row.lock().expect("row mutex");
        row.record(&line);
        eprint!(
            "\r  vs {name}: {}/{} (W{} D{} L{})",
            row.games, games, row.wins, row.draws, row.losses
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
    let mut ladder: Vec<LadderRow> = Vec::new();
    if !args.skip_ladder {
        eprintln!(
            "== opponent ladder ({} games per rung, {} on the vs rung{}) ==",
            args.games,
            args.vs_games.unwrap_or(args.games),
            if boards[0].is_some() {
                format!(
                    ", boards {}",
                    boards
                        .iter()
                        .flatten()
                        .map(|d| board_label(*d))
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
            for &board in &boards {
                if wanted.contains(&"random") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "random",
                        board,
                        args.games,
                        || Box::new(RandomBot::new()),
                    ));
                }
                if wanted.contains(&"scripted") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "scripted",
                        board,
                        args.games,
                        || Box::new(ScriptedBot::new()),
                    ));
                }
                if wanted.contains(&"mcts-heuristic") {
                    ladder.push(run_ladder_rung(
                        &args,
                        cand_preset.as_ref(),
                        nn.as_ref(),
                        "mcts-heuristic",
                        board,
                        args.games,
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
                let (opp_puct, opp_horizon, opp_backup, opp_fpu) = (
                    opp.puct.expect("set when no preset is named"),
                    opp.horizon_turns.expect("set when no preset is named"),
                    opp.backup.expect("set when no preset is named"),
                    opp.fpu_reduction.expect("set when no preset is named"),
                );
                let (cand_horizon, cand_backup, cand_fpu) = (
                    cand.horizon_turns.expect("set when no preset is named"),
                    cand.backup.expect("set when no preset is named"),
                    cand.fpu_reduction.expect("set when no preset is named"),
                );
                format!(
                    "vs:{} [{}{}{}{}]",
                    evaluator_label(vs, args.vs_model.as_deref()),
                    opp_puct.label(),
                    if opp_horizon != cand_horizon {
                        format!(" horizon={opp_horizon}v{cand_horizon}")
                    } else {
                        String::new()
                    },
                    if opp_backup != cand_backup {
                        format!(" {}v{}", opp_backup.label(), cand_backup.label())
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
            for &board in &boards {
                ladder.push(run_ladder_rung(
                    &args,
                    cand_preset.as_ref(),
                    nn.as_ref(),
                    &label,
                    board,
                    vs_games,
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
        board_env: if boards[0].is_some() {
            boards
                .iter()
                .flatten()
                .map(|d| board_label(*d))
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
        println!(
            "  ladder  vs {:16} win_rate {:.2}  (W{} D{} L{})  [home {}-{} away {}-{}]  TD {}:{}  [side TD H{} A{}]{}",
            r.opponent,
            r.win_rate,
            r.wins,
            r.draws,
            r.losses,
            r.wins_as_home,
            r.losses_as_home,
            r.wins_as_away,
            r.losses_as_away,
            r.tds_for,
            r.tds_against,
            r.tds_by_home,
            r.tds_by_away,
            if r.unfinished > 0 {
                format!("  [{} unfinished]", r.unfinished)
            } else {
                String::new()
            },
        );
    }

    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
        println!("wrote {out}");
    }
    Ok(())
}

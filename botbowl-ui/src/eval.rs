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
//! Output: a printed table and (optionally) a JSON report for tracking
//! across generations.

use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_curriculum::{available_lectures, make_lecture, run_trials, TrialStats};
use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::TeamType;
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::{MctsBot, PuctMode, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

use crate::cli::{CliCandidateBot, CliEvaluator, EvalArgs};
use crate::dataset::GAME_STACK_SIZE;

/// Max micro-steps per lecture trial (mirrors the curriculum CLI default).
const LECTURE_MAX_STEPS: u32 = 2000;

const OPPONENT_SEED_MIX: u64 = 0xC3C3_C3C3_C3C3_C3C3;
const CANDIDATE_SEED_MIX: u64 = 0x3C3C_3C3C_3C3C_3C3C;

#[derive(Serialize, Debug)]
struct LectureRow {
    lecture: String,
    difficulty: String,
    trials: u32,
    successes: u32,
    failures: u32,
    timeouts: u32,
    success_rate: f64,
    /// The lecture's hard-coded full-pitch coordinates don't fit the
    /// compiled board — cell skipped (see plan 020 next-next steps:
    /// board-relative lecture setups).
    skipped_board_too_small: bool,
}

#[derive(Serialize, Debug, Default)]
struct LadderRow {
    opponent: String,
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
    tds_for: u32,
    tds_against: u32,
    unfinished: u32,
    win_rate: f64,
    /// Per-side split (games alternate Home/Away): a Home/Away asymmetry
    /// cancels out of `win_rate` but shows up here (plan 021 open issue 5,
    /// the 0.40 mirror anomaly).
    wins_as_home: u32,
    losses_as_home: u32,
    wins_as_away: u32,
    losses_as_away: u32,
    /// Side-relative TD totals (not candidate-relative): closes the
    /// instrument gap noted in plan 023 — `tds_for/against` are pooled over
    /// both sides and so are balanced by construction in a mirror.
    tds_by_home: u32,
    tds_by_away: u32,
}

#[derive(Serialize, Debug)]
struct Report {
    candidate: String,
    mcts_iters: usize,
    seed: u64,
    board_env: String,
    git_commit: String,
    git_dirty: bool,
    lectures: Vec<LectureRow>,
    ladder: Vec<LadderRow>,
}

/// Resolve a `(mode, c)` pair into a `PuctMode`. Panics on an unknown mode
/// rather than silently running the wrong arm of a multi-hour head-to-head.
fn puct_of(mode: &str, c: Option<f32>) -> PuctMode {
    match mode {
        "raw" => match c {
            Some(c) => PuctMode::Raw { c },
            None => PuctMode::raw(),
        },
        "normalised" | "normalized" | "norm" => PuctMode::normalised(c.unwrap_or(1.0)),
        other => panic!("--puct-mode: expected `raw` or `normalised`, got `{other}`"),
    }
}

fn make_bot(
    evaluator: CliEvaluator,
    nn: Option<&Arc<NnEvaluator>>,
    iters: usize,
    workers: usize,
    puct: PuctMode,
) -> MctsBot {
    let bot = MctsBot::new(SearchBudget::Iterations(iters))
        .with_workers(workers)
        .with_puct(puct);
    match evaluator {
        CliEvaluator::Heuristic => bot,
        CliEvaluator::PureTd => bot.with_pure_td(),
        CliEvaluator::Nn => bot.with_evaluator(Arc::clone(nn.expect("nn loaded"))),
        CliEvaluator::NnValue => bot.with_nn_value(Arc::clone(nn.expect("nn loaded"))),
    }
}

fn make_candidate(args: &EvalArgs, nn: Option<&Arc<NnEvaluator>>) -> MctsBot {
    make_bot(
        args.evaluator,
        nn,
        args.mcts_iters,
        args.mcts_workers,
        puct_of(&args.puct_mode, args.puct_c),
    )
}

/// The bot in the candidate seat. `Mcts` is the report card's normal
/// candidate; the other two exist to take search out of the picture.
fn make_candidate_bot(args: &EvalArgs, nn: Option<&Arc<NnEvaluator>>) -> Box<dyn Bot> {
    match args.candidate_bot {
        CliCandidateBot::Mcts => Box::new(make_candidate(args, nn)),
        CliCandidateBot::Scripted => Box::new(ScriptedBot::new()),
        CliCandidateBot::Random => Box::new(RandomBot::new()),
    }
}

fn evaluator_label(evaluator: CliEvaluator, model: Option<&str>) -> String {
    match evaluator {
        CliEvaluator::Heuristic => "mcts(heuristic)".to_string(),
        CliEvaluator::PureTd => "mcts(pure-td)".to_string(),
        CliEvaluator::Nn => format!("mcts(nn:{})", model.unwrap_or("?")),
        CliEvaluator::NnValue => format!("mcts(nn-value:{})", model.unwrap_or("?")),
    }
}

fn candidate_label(args: &EvalArgs) -> String {
    match args.candidate_bot {
        CliCandidateBot::Mcts => evaluator_label(args.evaluator, args.model.as_deref()),
        CliCandidateBot::Scripted => "scripted".to_string(),
        CliCandidateBot::Random => "random".to_string(),
    }
}

/// Load the ONNX evaluator an nn/nn-value bot needs; `None` otherwise.
/// `missing_msg` is the error when the model path flag wasn't given.
fn load_nn(
    evaluator: CliEvaluator,
    model: Option<&str>,
    missing_msg: &str,
    server: Option<&std::path::Path>,
) -> io::Result<Option<Arc<NnEvaluator>>> {
    match evaluator {
        CliEvaluator::Nn | CliEvaluator::NnValue => {
            let path = model
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, missing_msg.to_string()))?;
            // Each evaluator names its own model at handshake and gets its
            // own canary, so the candidate and the champion can share one
            // socket with no chance of being cross-wired.
            let eval = NnEvaluator::from_path_with_server(path, server)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("failed to load {path}: {e}")))?;
            Ok(Some(Arc::new(eval)))
        }
        _ => Ok(None),
    }
}

/// One full game from kickoff. Returns `(candidate_score, opponent_score,
/// finished)`.
fn play_game(
    candidate: &mut dyn Bot,
    opponent: &mut dyn Bot,
    candidate_team: TeamType,
    seed: u64,
    max_steps: u32,
) -> (u8, u8, bool, TeamType) {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    candidate.set_seed(ChaCha8Rng::seed_from_u64(seed ^ CANDIDATE_SEED_MIX));
    opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ OPPONENT_SEED_MIX));

    let mut steps = 0u32;
    while !state.info.game_over && steps < max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == candidate_team => candidate.get_action(&state),
            Some(_) => opponent.get_action(&state),
            None => break,
        };
        state.step(action).expect("engine step failed during eval game");
        steps += 1;
    }

    let (cand, opp) = score_for(&state, candidate_team);
    (cand, opp, state.info.game_over, state.info.kicking_first_half)
}

fn score_for(state: &GameState, team: TeamType) -> (u8, u8) {
    match team {
        TeamType::Home => (state.home.score, state.away.score),
        TeamType::Away => (state.away.score, state.home.score),
    }
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
}

fn run_ladder_rung(
    args: &EvalArgs,
    nn: Option<&Arc<NnEvaluator>>,
    name: &str,
    games: u32,
    make_opponent: impl Fn() -> Box<dyn Bot> + Sync,
) -> LadderRow {
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
        row: Mutex::new(LadderRow { opponent: name.to_string(), ..Default::default() }),
        next_game: AtomicU32::new(0),
        per_game: Mutex::new(per_game),
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
        run_rung_games(args, nn, name, games, &make_opponent, &state);
    } else {
        eprintln!("  vs {name}: {parallel} games in parallel");
        std::thread::scope(|s| {
            for i in 0..parallel {
                let st = &state;
                let mk = &make_opponent;
                std::thread::Builder::new()
                    .name(format!("rung-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(s, move || run_rung_games(args, nn, name, games, mk, st))
                    .expect("spawn rung worker");
            }
        });
    }
    eprintln!();

    let mut row = state.row.into_inner().expect("row mutex");
    row.win_rate = if row.games > 0 {
        row.wins as f64 / row.games as f64
    } else {
        0.0
    };
    row
}

/// Play rung games off the shared counter until they run out.
///
/// Bots are built **here**, per worker, and never leave this thread —
/// which is what makes this sound despite `dyn Bot` having no `Send`
/// bound. It also matches the sequential behaviour: bots were already
/// reused across the games of a rung, and `MctsBot`'s cached tree is
/// discarded anyway when the horizon anchor fails to match at a new
/// game's kickoff.
fn run_rung_games(
    args: &EvalArgs,
    nn: Option<&Arc<NnEvaluator>>,
    name: &str,
    games: u32,
    make_opponent: &(impl Fn() -> Box<dyn Bot> + Sync),
    state: &RungState,
) {
    let mut candidate = make_candidate_bot(args, nn);
    let mut opponent = make_opponent();
    loop {
        let g = state.next_game.fetch_add(1, Ordering::Relaxed);
        if g >= games {
            return;
        }
        // Alternate sides; the seed is shared by the mirrored game `g±1`,
        // so every candidate faces the same situations from both sides.
        // Both are pure functions of `g`, so which worker picks up which
        // game cannot change the pairing.
        let candidate_team = if g % 2 == 0 { TeamType::Home } else { TeamType::Away };
        let seed = args.seed.wrapping_add((g / 2) as u64);
        let (cand, opp, finished, kicking_first_half) =
            play_game(&mut *candidate, &mut *opponent, candidate_team, seed, args.max_steps);

        let (h, a) = match candidate_team {
            TeamType::Home => (cand, opp),
            TeamType::Away => (opp, cand),
        };
        if let Some(w) = state.per_game.lock().expect("per-game mutex").as_mut() {
            use std::io::Write;
            writeln!(
                w,
                r#"{{"rung":"{name}","game":{g},"seed":{seed},"candidate_team":"{candidate_team:?}","home_score":{h},"away_score":{a},"kicking_first_half":"{kicking_first_half:?}","finished":{finished}}}"#
            )
            .expect("per-game log write failed");
        }

        let mut row = state.row.lock().expect("row mutex");
        row.games += 1;
        row.tds_for += cand as u32;
        row.tds_against += opp as u32;
        row.tds_by_home += h as u32;
        row.tds_by_away += a as u32;
        if !finished {
            row.unfinished += 1;
        }
        let home = candidate_team == TeamType::Home;
        match cand.cmp(&opp) {
            std::cmp::Ordering::Greater => {
                row.wins += 1;
                if home { row.wins_as_home += 1 } else { row.wins_as_away += 1 }
            }
            std::cmp::Ordering::Equal => row.draws += 1,
            std::cmp::Ordering::Less => {
                row.losses += 1;
                if home { row.losses_as_home += 1 } else { row.losses_as_away += 1 }
            }
        }
        eprint!("\r  vs {name}: {}/{} (W{} D{} L{})", row.games, games, row.wins, row.draws, row.losses);
    }
}

pub fn run(args: EvalArgs) -> io::Result<()> {
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        args.evaluator,
        args.model.as_deref(),
        "--evaluator nn/nn-value requires --model PATH",
        server.as_deref(),
    )?;
    // Load the --vs-evaluator opponent's net up front so a bad path fails
    // before hours of fixed-rung games.
    let vs_nn = match args.vs_evaluator {
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
            let mut agent = make_candidate_bot(&args, nn.as_ref());
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

    let mut ladder: Vec<LadderRow> = Vec::new();
    if !args.skip_ladder {
        eprintln!(
            "== opponent ladder ({} games per rung, {} on the vs rung) ==",
            args.games,
            args.vs_games.unwrap_or(args.games)
        );
        let opp_iters = args.opponent_iters.unwrap_or(args.mcts_iters);
        // Opponent defaults to the candidate's rule, so existing invocations
        // are unchanged; setting either --vs-puct-* makes it a rule head-to-head.
        let opp_puct = puct_of(
            args.vs_puct_mode.as_deref().unwrap_or(&args.puct_mode),
            args.vs_puct_c.or(args.puct_c),
        );
        if !args.skip_fixed_rungs {
            let wanted: Vec<&str> = args.rungs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
            for name in &wanted {
                if !matches!(*name, "random" | "scripted" | "mcts-heuristic") {
                    panic!("--rungs: expected `random`, `scripted` or `mcts-heuristic`, got `{name}`");
                }
            }
            if wanted.contains(&"random") {
                ladder.push(run_ladder_rung(&args, nn.as_ref(), "random", args.games, || {
                    Box::new(RandomBot::new())
                }));
            }
            if wanted.contains(&"scripted") {
                ladder.push(run_ladder_rung(&args, nn.as_ref(), "scripted", args.games, || {
                    Box::new(ScriptedBot::new())
                }));
            }
            if wanted.contains(&"mcts-heuristic") {
                ladder.push(run_ladder_rung(&args, nn.as_ref(), "mcts-heuristic", args.games, || {
                    Box::new(
                        MctsBot::new(SearchBudget::Iterations(opp_iters))
                            .with_workers(args.mcts_workers)
                            .with_puct(opp_puct),
                    )
                }));
            }
        }
        if let Some(vs) = args.vs_evaluator {
            let label = format!(
                "vs:{} [{}]",
                evaluator_label(vs, args.vs_model.as_deref()),
                opp_puct.label()
            );
            // The gating rung: `--vs-games` if given, else `--games`.
            let vs_games = args.vs_games.unwrap_or(args.games);
            ladder.push(run_ladder_rung(&args, nn.as_ref(), &label, vs_games, || {
                Box::new(make_bot(vs, vs_nn.as_ref(), opp_iters, args.mcts_workers, opp_puct))
            }));
        }
    }

    let report = Report {
        candidate: candidate_label(&args),
        mcts_iters: args.mcts_iters,
        seed: args.seed,
        board_env: format!(
            "{:?}",
            botbowl_engine::core::model::BoardDims::from_env()
        ),
        git_commit: botbowl_data::git_commit().to_string(),
        git_dirty: botbowl_data::git_dirty(),
        lectures,
        ladder,
    };

    println!("\n== report card: {} ==", report.candidate);
    for l in &report.lectures {
        if l.skipped_board_too_small {
            println!("  lecture {:24} {:8} skipped (board too small)", l.lecture, l.difficulty);
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

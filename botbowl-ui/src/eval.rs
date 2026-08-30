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
use std::sync::Arc;

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

use crate::cli::{CliEvaluator, EvalArgs};

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

fn evaluator_label(evaluator: CliEvaluator, model: Option<&str>) -> String {
    match evaluator {
        CliEvaluator::Heuristic => "mcts(heuristic)".to_string(),
        CliEvaluator::PureTd => "mcts(pure-td)".to_string(),
        CliEvaluator::Nn => format!("mcts(nn:{})", model.unwrap_or("?")),
        CliEvaluator::NnValue => format!("mcts(nn-value:{})", model.unwrap_or("?")),
    }
}

fn candidate_label(args: &EvalArgs) -> String {
    evaluator_label(args.evaluator, args.model.as_deref())
}

/// Load the ONNX evaluator an nn/nn-value bot needs; `None` otherwise.
/// `missing_msg` is the error when the model path flag wasn't given.
fn load_nn(evaluator: CliEvaluator, model: Option<&str>, missing_msg: &str) -> io::Result<Option<Arc<NnEvaluator>>> {
    match evaluator {
        CliEvaluator::Nn | CliEvaluator::NnValue => {
            let path = model
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, missing_msg.to_string()))?;
            let eval = NnEvaluator::from_path(path)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("failed to load {path}: {e}")))?;
            Ok(Some(Arc::new(eval)))
        }
        _ => Ok(None),
    }
}

/// One full game from kickoff. Returns `(candidate_score, opponent_score,
/// finished)`.
fn play_game(
    candidate: &mut MctsBot,
    opponent: &mut dyn Bot,
    candidate_team: TeamType,
    seed: u64,
    max_steps: u32,
) -> (u8, u8, bool) {
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
    (cand, opp, state.info.game_over)
}

fn score_for(state: &GameState, team: TeamType) -> (u8, u8) {
    match team {
        TeamType::Home => (state.home.score, state.away.score),
        TeamType::Away => (state.away.score, state.home.score),
    }
}

fn run_ladder_rung(
    args: &EvalArgs,
    nn: Option<&Arc<NnEvaluator>>,
    name: &str,
    mut make_opponent: impl FnMut() -> Box<dyn Bot>,
) -> LadderRow {
    let mut row = LadderRow {
        opponent: name.to_string(),
        ..Default::default()
    };
    let mut candidate = make_candidate(args, nn);
    let mut opponent = make_opponent();
    for g in 0..args.games {
        // Alternate sides; the seed is shared by the mirrored game `g±1`,
        // so every candidate faces the same situations from both sides.
        let candidate_team = if g % 2 == 0 { TeamType::Home } else { TeamType::Away };
        let seed = args.seed.wrapping_add((g / 2) as u64);
        let (cand, opp, finished) = play_game(&mut candidate, &mut *opponent, candidate_team, seed, args.max_steps);
        row.games += 1;
        row.tds_for += cand as u32;
        row.tds_against += opp as u32;
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
        eprint!("\r  vs {name}: {}/{} (W{} D{} L{})", g + 1, args.games, row.wins, row.draws, row.losses);
    }
    eprintln!();
    row.win_rate = if row.games > 0 {
        row.wins as f64 / row.games as f64
    } else {
        0.0
    };
    row
}

pub fn run(args: EvalArgs) -> io::Result<()> {
    let nn = load_nn(
        args.evaluator,
        args.model.as_deref(),
        "--evaluator nn/nn-value requires --model PATH",
    )?;
    // Load the --vs-evaluator opponent's net up front so a bad path fails
    // before hours of fixed-rung games.
    let vs_nn = match args.vs_evaluator {
        Some(vs) => load_nn(
            vs,
            args.vs_model.as_deref(),
            "--vs-evaluator nn/nn-value requires --vs-model PATH",
        )?,
        None => None,
    };

    let mut lectures: Vec<LectureRow> = Vec::new();
    if !args.skip_lectures {
        eprintln!("== lecture battery ({} trials per cell) ==", args.trials);
        for &(name, difficulty) in available_lectures() {
            let lecture = make_lecture(name, difficulty).expect("available_lectures entry must construct");
            let mut agent = make_candidate(&args, nn.as_ref());
            // Lectures place players at hard-coded full-pitch coordinates;
            // on smaller compiled boards a cell can panic mid-setup. Run
            // the whole cell under catch_unwind (quiet panic hook) and
            // report it skipped rather than aborting the report card.
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_trials(&*lecture, &mut agent, args.trials, args.seed, LECTURE_MAX_STEPS)
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
        eprintln!("== opponent ladder ({} games per rung) ==", args.games);
        let opp_iters = args.opponent_iters.unwrap_or(args.mcts_iters);
        // Opponent defaults to the candidate's rule, so existing invocations
        // are unchanged; setting either --vs-puct-* makes it a rule head-to-head.
        let opp_puct = puct_of(
            args.vs_puct_mode.as_deref().unwrap_or(&args.puct_mode),
            args.vs_puct_c.or(args.puct_c),
        );
        if !args.skip_fixed_rungs {
            ladder.push(run_ladder_rung(&args, nn.as_ref(), "random", || {
                Box::new(RandomBot::new())
            }));
            ladder.push(run_ladder_rung(&args, nn.as_ref(), "scripted", || {
                Box::new(ScriptedBot::new())
            }));
            ladder.push(run_ladder_rung(&args, nn.as_ref(), "mcts-heuristic", || {
                Box::new(
                    MctsBot::new(SearchBudget::Iterations(opp_iters))
                        .with_workers(args.mcts_workers)
                        .with_puct(opp_puct),
                )
            }));
        }
        if let Some(vs) = args.vs_evaluator {
            let label = format!(
                "vs:{} [{}]",
                evaluator_label(vs, args.vs_model.as_deref()),
                opp_puct.label()
            );
            ladder.push(run_ladder_rung(&args, nn.as_ref(), &label, || {
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
            "  ladder  vs {:16} win_rate {:.2}  (W{} D{} L{})  [home {}-{} away {}-{}]  TD {}:{}{}",
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

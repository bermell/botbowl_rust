//! Search-budget convergence probe (plan 025).
//!
//! Answers "how many MCTS iterations does the training data actually need?"
//! by re-searching the *same* state at a ladder of budgets and measuring how
//! far each budget's output sits from a long-reference search.
//!
//! Two properties make the result interpretable:
//!
//! 1. **Distance to a reference, not to the previous checkpoint.** Successive
//!    differences look converged at every step for a slowly drifting
//!    distribution. The largest budget is the reference.
//! 2. **A noise floor from repeats.** `MctsBot` is not reproducible from seeds
//!    (`recon_mcts` randomises `HashMap` tie-break order per process, plan
//!    020), so two searches of the same state at the same budget differ. That
//!    is the instrument here: repeat every (state, budget) cell and compare the
//!    budget effect against the run-to-run spread. Convergence is "the budget
//!    stops mattering more than the search's own nondeterminism".
//!
//! This writes raw per-child stats, not distances — every metric in
//! `scripts/convergence_summary.py` is recomputable offline from the output, so
//! a second question does not mean re-running the search.
//!
//! Usage:
//! ```text
//! botbowl-ui convergence --states 50 --repeats 3 \
//!     --budgets 100,200,500,1000,2000,4000,8000,16000 \
//!     --evaluator nn-value --model models/bbnet_14x7_gen01.onnx \
//!     --out runs/convergence/nn_value.jsonl
//! ```

use std::io::{self, Write};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_curriculum::generate_random_start;
use botbowl_data::{ChildStat, Team};
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_mcts::{MctsBot, PuctMode, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

use botbowl_engine::core::model::BoardDims;
use botbowl_play::board_sizes::{board_label, parse_board, DEFAULT_CELLS_PER_PLAYER};
use botbowl_play::bots::{load_mcts_config, load_nn, Evaluator};
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::{CliEvaluator, ConvergenceArgs};

/// One (state, repeat, budget) cell. Deliberately excludes the `GameState`:
/// it is identical across every cell of a state and would dominate the file.
#[derive(Serialize)]
struct Row<'a> {
    state_idx: u32,
    /// Seed the state was generated from — regenerates it exactly.
    state_seed: u64,
    repeat: u32,
    budget: usize,
    /// Which selection rule produced this row. Rows are pooled by
    /// `state_seed`, so without an arm key two arms sharing a seed base merge
    /// silently and the sweep measures nothing.
    puct: &'a str,
    /// The playable board (`16x9/6`) and how many decisions the probe advanced past the random
    /// start, so files from several runs can be pooled and split again.
    board: &'a str,
    advance: u32,
    /// Stratification keys (constant per state, repeated for convenience).
    n_legal_actions: usize,
    half: u8,
    home_turn: u8,
    away_turn: u8,
    to_move: Team,
    /// Search output.
    chosen_action: &'a EngineAction,
    children: &'a [ChildStat],
    root_value: Option<i64>,
    root_visits: u32,
    root_solved: bool,
    /// Wall time for this single search, to sanity-check the cost model.
    elapsed_ms: u64,
}

/// Resolve `--puct-mode` / `--puct-c` / `--puct-range-floor` into a `PuctMode`.
/// Panics on an unknown mode rather than silently sweeping the wrong arm.
fn puct_from_args(args: &ConvergenceArgs) -> PuctMode {
    match args.puct_mode.as_str() {
        "raw" => match args.puct_c {
            Some(c) => PuctMode::Raw { c },
            None => PuctMode::raw(),
        },
        "normalised" | "normalized" | "norm" => {
            let base = PuctMode::normalised(args.puct_c.unwrap_or(1.0));
            match (base, args.puct_range_floor) {
                (PuctMode::NormalisedQ { c, .. }, Some(f)) => PuctMode::NormalisedQ { c, range_floor: f },
                (m, _) => m,
            }
        }
        other => panic!("--puct-mode: expected `raw` or `normalised`, got `{other}`"),
    }
}

fn make_bot(args: &ConvergenceArgs, nn: Option<&Arc<NnEvaluator>>, budget: usize) -> MctsBot {
    let bot = match &args.bot_config {
        Some(path) => {
            let preset = load_mcts_config(path).unwrap_or_else(|e| panic!("{e}"));
            MctsBot::with_budget_and_config(SearchBudget::Iterations(budget), preset.config)
                .with_workers(args.mcts_workers)
        }
        None => MctsBot::from_env(SearchBudget::Iterations(budget))
            .with_workers(args.mcts_workers)
            .with_puct(puct_from_args(args)),
    };
    match args.evaluator {
        CliEvaluator::Heuristic => bot,
        CliEvaluator::PureTd => bot.with_pure_td(),
        CliEvaluator::Nn => bot.with_evaluator(Arc::clone(nn.expect("nn evaluator required"))),
        CliEvaluator::NnValue => bot.with_nn_value(Arc::clone(nn.expect("nn evaluator required"))),
    }
}

pub fn run(args: ConvergenceArgs) -> io::Result<()> {
    let budgets: Vec<usize> = args
        .budgets
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("--budgets: '{s}' is not a positive integer"))
        })
        .collect();
    assert!(!budgets.is_empty(), "--budgets must list at least one budget");
    assert!(
        budgets.windows(2).all(|w| w[0] < w[1]),
        "--budgets must be strictly increasing (the largest is the reference)"
    );

    let evaluator = match args.evaluator {
        CliEvaluator::Heuristic => Evaluator::Heuristic,
        CliEvaluator::PureTd => Evaluator::PureTd,
        CliEvaluator::Nn => Evaluator::Nn,
        CliEvaluator::NnValue => Evaluator::NnValue,
    };
    // `--nn-server` batches every probe thread's forwards on the GPU sidecar; without it each
    // thread runs tract on the CPU, about 12x the cost per forward.
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        evaluator,
        args.model.as_deref(),
        "--evaluator nn/nn-value requires --model PATH",
        server.as_deref(),
    )?;
    let board: Option<BoardDims> = args
        .board
        .as_deref()
        .map(|b| parse_board(b, DEFAULT_CELLS_PER_PLAYER))
        .transpose()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let board_name = board
        .map(board_label)
        .unwrap_or_else(|| board_label(BoardDims::from_env()));

    let puct_label = match &args.bot_config {
        Some(path) => {
            let preset = load_mcts_config(path)?;
            format!("{}@{}", preset.config.puct.label(), preset.name)
        }
        None => puct_from_args(&args).label(),
    };
    eprintln!(
        "selection rule: {puct_label}, board {board_name}, {} probe thread(s)",
        args.parallel.max(1)
    );
    let out = Mutex::new(std::fs::File::create(&args.out)?);
    let started = Instant::now();
    let total_cells = args.states as u64 * args.repeats as u64 * budgets.len() as u64;
    let done = AtomicU64::new(0);
    let next_state = AtomicU32::new(0);

    let worker = || -> io::Result<()> {
        loop {
            let state_idx = next_state.fetch_add(1, Ordering::Relaxed);
            if state_idx >= args.states {
                return Ok(());
            }
            let lines = probe_state(&args, nn.as_ref(), &budgets, board, &board_name, &puct_label, state_idx)?;
            let n = lines.len() as u64;
            {
                let mut f = out.lock().expect("output mutex");
                for l in &lines {
                    f.write_all(l.as_bytes())?;
                    f.write_all(b"\n")?;
                }
                f.flush()?;
            }
            let d = done.fetch_add(n, Ordering::Relaxed) + n;
            let elapsed = started.elapsed().as_secs_f64();
            let frac = d as f64 / total_cells as f64;
            eprintln!(
                "[{d}/{total_cells} cells] state {state_idx} — {elapsed:.0}s elapsed, ~{:.0}s remaining",
                if frac > 0.0 { elapsed / frac - elapsed } else { 0.0 }
            );
        }
    };
    let parallel = args.parallel.max(1);
    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = (0..parallel)
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("probe-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(scope, &worker)
                    .expect("spawn probe thread")
            })
            .collect();
        for h in handles {
            h.join().expect("probe thread panicked")?;
        }
        Ok(())
    })?;

    eprintln!(
        "wrote {} rows to {} in {:.0}s",
        done.load(Ordering::Relaxed),
        args.out,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Every (repeat, budget) cell of one state, as JSON lines. Empty when the state is skipped.
fn probe_state(
    args: &ConvergenceArgs,
    nn: Option<&Arc<NnEvaluator>>,
    budgets: &[usize],
    board: Option<BoardDims>,
    board_name: &str,
    puct_label: &str,
    state_idx: u32,
) -> io::Result<Vec<String>> {
    // Disjoint from every corpus seed: the loop uses 10_000_000 + G*1e6 +
    // K*1e5, so a base far above that cannot collide.
    let state_seed = args.seed + state_idx as u64 * 1_000;
    let mut cfg = args.bias.to_config();
    if state_seed % 2 == 1 {
        cfg.temperature = args.bias.temperature2;
    }
    cfg.board_dims = board;
    let mut rng = ChaCha8Rng::seed_from_u64(state_seed);
    let mut state: GameState = generate_random_start(&cfg, &mut rng);
    state.set_logging_state(false);
    // Production generation searches under real dice; anything else would
    // measure convergence of a different search.
    state.set_dice_mode(DiceMode::RollDice);

    if state.available_actions.team.is_none() {
        eprintln!("[{state_idx}] seed={state_seed} has no team to act — skipped");
        return Ok(Vec::new());
    }

    // Plan 032 #7: step into the turn with a production-budget bot so
    // the probed root is a mid-turn (wide-fan) decision. Each decision
    // gets a fresh bot: no tree reuse leaks into the probe. Always the plain
    // PUCT bot, never `--bot-config`'s: a preset must be probed on the states
    // a plain run probes. Even so, a search is not reproducible across
    // processes, so an advanced state only matches *within* one run — never
    // score one run's `--advance > 0` rows against another run's reference
    // (exp060 did, and its mid-turn numbers compared different positions).
    for _ in 0..args.advance {
        if state.info.game_over || state.available_actions.team.is_none() {
            break;
        }
        let plain = ConvergenceArgs {
            bot_config: None,
            ..args.clone()
        };
        let mut bot = make_bot(&plain, nn, 1000);
        let action = bot.get_action(&state);
        state
            .step(action)
            .expect("engine step failed while advancing a probe state");
    }
    if state.info.game_over || state.available_actions.team.is_none() {
        eprintln!("[{state_idx}] seed={state_seed} left the decision loop while advancing — skipped");
        return Ok(Vec::new());
    }
    if args.min_legal > 0 {
        // The fan the search sees is the *pruned* one, so measure it the
        // way the search does: a root-expansion-only search.
        let mut probe = make_bot(args, nn, 2);
        let n = probe.get_action_with_record(&state).1.children.len();
        if n < args.min_legal {
            eprintln!(
                "[{state_idx}] seed={state_seed} fan {n} < --min-legal {} — skipped",
                args.min_legal
            );
            return Ok(Vec::new());
        }
    }

    let mut lines = Vec::with_capacity(args.repeats as usize * budgets.len());
    for repeat in 0..args.repeats {
        for &budget in budgets {
            let mut bot = make_bot(args, nn, budget);
            // `MctsBot` has no RNG of its own (`Bot::set_seed` is the trait's
            // no-op default for it), so this seed does nothing today. Repeats
            // are independent because each cell builds a fresh bot (no tree
            // reuse) and `recon_mcts`'s HashMap tie-break order is randomised
            // per process (plan 020). Kept so a future seeded bot is covered.
            bot.set_seed(ChaCha8Rng::seed_from_u64(
                state_seed ^ ((repeat as u64) << 32) ^ (budget as u64),
            ));

            let t0 = Instant::now();
            let (_action, sample) = bot.get_action_with_record(&state);
            let elapsed_ms = t0.elapsed().as_millis() as u64;

            let row = Row {
                state_idx,
                state_seed,
                repeat,
                budget,
                puct: puct_label,
                board: board_name,
                advance: args.advance,
                n_legal_actions: sample.children.len(),
                half: state.info.half,
                home_turn: state.info.home_turn,
                away_turn: state.info.away_turn,
                to_move: sample.to_move,
                chosen_action: &sample.chosen_action,
                children: &sample.children,
                root_value: sample.root_value,
                root_visits: sample.root_visits,
                root_solved: sample.root_solved,
                elapsed_ms,
            };
            lines.push(serde_json::to_string(&row)?);
        }
    }
    Ok(lines)
}

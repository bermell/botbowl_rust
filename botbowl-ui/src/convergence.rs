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
use std::sync::Arc;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_curriculum::generate_random_start;
use botbowl_data::{ChildStat, Team};
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_mcts::{MctsBot, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

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

fn make_bot(args: &ConvergenceArgs, nn: Option<&Arc<NnEvaluator>>, budget: usize) -> MctsBot {
    let bot = MctsBot::new(SearchBudget::Iterations(budget)).with_workers(args.mcts_workers);
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

    let nn = match args.evaluator {
        CliEvaluator::Nn | CliEvaluator::NnValue => {
            let path = args
                .model
                .as_deref()
                .expect("--evaluator nn/nn-value requires --model PATH");
            Some(Arc::new(NnEvaluator::from_path(path).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("loading {path}: {e}"))
            })?))
        }
        _ => None,
    };

    let mut out = std::fs::File::create(&args.out)?;
    let started = Instant::now();
    let total_cells = args.states as u64 * args.repeats as u64 * budgets.len() as u64;
    let mut done = 0u64;

    let mut n_legal_seen: usize = 0;
    for state_idx in 0..args.states {
        // Disjoint from every corpus seed: the loop uses 10_000_000 + G*1e6 +
        // K*1e5, so a base far above that cannot collide.
        let state_seed = args.seed + state_idx as u64 * 1_000;
        let mut cfg = args.bias.to_config();
        if state_seed % 2 == 1 {
            cfg.temperature = args.bias.temperature2;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(state_seed);
        let mut state: GameState = generate_random_start(&cfg, &mut rng);
        state.set_logging_state(false);
        // Production generation searches under real dice; anything else would
        // measure convergence of a different search.
        state.set_dice_mode(DiceMode::RollDice);

        let Some(_team) = state.available_actions.team else {
            eprintln!("[{state_idx}] seed={state_seed} has no team to act — skipped");
            continue;
        };

        for repeat in 0..args.repeats {
            for &budget in &budgets {
                let mut bot = make_bot(&args, nn.as_ref(), budget);
                // Distinct per (state, repeat) so repeats are independent
                // draws. Note the dominant nondeterminism is HashMap tie-break
                // order, which no seed controls (plan 020) — this only keeps
                // the bot's own RNG from being identical.
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
                serde_json::to_writer(&mut out, &row)?;
                out.write_all(b"\n")?;

                n_legal_seen = sample.children.len();
                done += 1;
            }
        }
        out.flush()?;
        let elapsed = started.elapsed().as_secs_f64();
        let frac = done as f64 / total_cells as f64;
        eprintln!(
            "[{}/{}] state seed={state_seed} actions={} — {:.0}s elapsed, ~{:.0}s remaining",
            state_idx + 1,
            args.states,
            n_legal_seen,
            elapsed,
            if frac > 0.0 { elapsed / frac - elapsed } else { 0.0 }
        );
    }

    eprintln!(
        "wrote {done} rows to {} in {:.0}s",
        args.out,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

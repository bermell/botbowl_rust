//! `drive-agree` (plan 068 §1): do two bots play the same moves on a drive benchmark's positions?
//!
//! For each position of a drive set (the pair's own dice: `seed + index`, as
//! `drive_assignment` gives pair `index` on the first pass), the two bots are run in lockstep
//! (`botbowl_play::drives::lockstep_drive`): four instances, one per (bot, role), both asked at
//! every decision of the shared drive until they first choose differently. That is exactly the
//! normal pair interleaved, so "never diverges" here is "the pair's two games are one game".
//! Optional prescreens run the same lockstep with cheap bots first (`--prescreen-iters`, `0` is
//! the bare policy), so their divergence can be checked against the full search's. Each decision
//! of the full run also records whether the two nets' bare-policy argmaxes agree there.
//!
//! One JSON row per position in `--out`; `scripts/drive_agree_summary.py` makes the tables.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use botbowl_engine::bots::Bot;
use botbowl_mcts::SearchBudget;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::bots::{load_mcts_config, load_nn, make_mcts, Evaluator, SearchConfig};
use botbowl_play::drives::{lockstep_drive, position_state, LockstepBots, LockstepOutcome, PositionSet};
use botbowl_play::policy::{search_legal, PolicyBot};
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::DriveAgreeArgs;

#[derive(Serialize)]
struct Run {
    /// Search budget; 0 = the bare policy (`PolicyBot`).
    iters: usize,
    #[serde(flatten)]
    outcome: LockstepOutcome,
    secs: f64,
}

#[derive(Serialize)]
struct Row {
    set: String,
    board: String,
    index: usize,
    position: u64,
    dice: u64,
    candidate: String,
    opponent: String,
    full: Option<Run>,
    prescreen: Vec<Run>,
    /// The full run's decisions: `[step, attacker_to_move, fan, search_agree, policy_agree,
    /// cand_is_own_argmax, opp_is_own_argmax, after_divergence]` (booleans as 0/1). Policy fields
    /// are 1 on a forced decision (fan 1).
    decisions: Vec<[u32; 8]>,
}

fn bots(iters: usize, preset: &botbowl_mcts::MctsConfig, nn: &Arc<NnEvaluator>) -> Box<dyn Bot> {
    if iters == 0 {
        return Box::new(PolicyBot::new(Arc::clone(nn)));
    }
    let search = SearchConfig {
        budget: SearchBudget::Iterations(iters),
        workers: 1,
        puct: None,
        horizon_turns: None,
        fpu_reduction: None,
        config: Some(*preset),
    };
    Box::new(make_mcts(&search, Evaluator::Nn, Some(nn)))
}

#[allow(clippy::too_many_arguments)]
fn run_lockstep(
    iters: usize,
    preset: &botbowl_mcts::MctsConfig,
    a: &Arc<NnEvaluator>,
    b: &Arc<NnEvaluator>,
    state: botbowl_engine::core::gamestate::GameState,
    dice: u64,
    max_steps: u32,
    follow: bool,
    probe: impl FnMut(&botbowl_play::drives::LockstepDecision<'_>),
) -> Run {
    let (mut ca, mut cd) = (bots(iters, preset, a), bots(iters, preset, a));
    let (mut oa, mut od) = (bots(iters, preset, b), bots(iters, preset, b));
    let t0 = Instant::now();
    let outcome = lockstep_drive(
        LockstepBots {
            cand_att: &mut *ca,
            cand_def: &mut *cd,
            opp_att: &mut *oa,
            opp_def: &mut *od,
        },
        state,
        dice,
        max_steps,
        false,
        follow,
        probe,
    );
    Run {
        iters,
        outcome,
        secs: t0.elapsed().as_secs_f64(),
    }
}

pub fn run(args: DriveAgreeArgs) -> io::Result<()> {
    args.backend.apply()?;
    let preset = load_mcts_config(&args.bot_config)?;
    let set = PositionSet::load(&args.positions).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let board = set
        .board_dims()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let positions = set.positions();
    let a = load_nn(Evaluator::Nn, Some(&args.model), "--model", None)?.expect("nn");
    let b = load_nn(Evaluator::Nn, Some(&args.vs_model), "--vs-model", None)?.expect("nn");
    let picked: Vec<usize> = (args.offset..positions.len()).take(args.count).collect();
    eprintln!(
        "drive-agree: {} positions of {} ({}), {} vs {}, {} iters, prescreens {:?}, {} threads",
        picked.len(),
        set.name,
        set.board,
        args.model,
        args.vs_model,
        args.mcts_iters,
        args.prescreen_iters,
        args.parallel
    );
    let out = Mutex::new(BufWriter::new(File::create(&args.out)?));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for w in 0..args.parallel.max(1) {
            let (set, positions, picked, preset, a, b, out, next, done, args) =
                (&set, &positions, &picked, &preset, &a, &b, &out, &next, &done, &args);
            std::thread::Builder::new()
                .name(format!("agree-{w}"))
                .stack_size(GAME_STACK_SIZE)
                .spawn_scoped(s, move || loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = picked.get(k) else { return };
                    let seed = positions[i];
                    let dice = args.seed.wrapping_add(i as u64);
                    let state = || position_state(&set.bias, board, seed);
                    let prescreen: Vec<Run> = args
                        .prescreen_iters
                        .iter()
                        .map(|&it| run_lockstep(it, &preset.config, a, b, state(), dice, args.max_steps, false, |_| {}))
                        .collect();
                    let mut decisions = Vec::new();
                    let full = (!args.skip_full).then(|| {
                        let (pa, pb) = (PolicyBot::new(Arc::clone(a)), PolicyBot::new(Arc::clone(b)));
                        run_lockstep(
                            args.mcts_iters,
                            &preset.config,
                            a,
                            b,
                            state(),
                            dice,
                            args.max_steps,
                            args.follow,
                            |d| {
                                let fan = search_legal(d.state).len() as u32;
                                let (p_agree, a_own, b_own) = if fan > 1 {
                                    let (la, _, ia) = pa.priors(d.state);
                                    let (lb, _, ib) = pb.priors(d.state);
                                    (la[ia] == lb[ib], d.cand == la[ia], d.opp == lb[ib])
                                } else {
                                    (true, true, true)
                                };
                                decisions.push([
                                    d.step,
                                    d.attacker_to_move as u32,
                                    fan,
                                    (d.cand == d.opp) as u32,
                                    p_agree as u32,
                                    a_own as u32,
                                    b_own as u32,
                                    d.after_divergence as u32,
                                ]);
                            },
                        )
                    });
                    let row = Row {
                        set: set.name.clone(),
                        board: set.board.clone(),
                        index: i,
                        position: seed,
                        dice,
                        candidate: args.model.clone(),
                        opponent: args.vs_model.clone(),
                        full,
                        prescreen,
                        decisions,
                    };
                    let mut w = out.lock().expect("out mutex");
                    serde_json::to_writer(&mut *w, &row).expect("write row");
                    writeln!(w).expect("write row");
                    w.flush().expect("flush");
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "  {n}/{} done ({:.0} s): position {i} full {:?}",
                        picked.len(),
                        t0.elapsed().as_secs_f64(),
                        row.full
                            .as_ref()
                            .map(|r| (r.outcome.divergence, r.outcome.prefix_steps, r.outcome.ended))
                    );
                })
                .expect("spawn");
        }
    });
    Ok(())
}

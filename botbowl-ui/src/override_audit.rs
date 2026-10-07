//! The override audit (plan 055 §3 phase 2): does the search's overruling of its own policy win
//! real drives, judged by Monte Carlo playouts that do not depend on the value head?
//!
//! For a random sample of corpus decisions `s` (mover `m`):
//!
//! 1. `a_p`, the **policy move**: the prior's argmax over the legal set the search's root sees,
//!    which is what `cfgs/policy_only.toml` plays ([`PolicyBot`]; a test pins the two agree).
//! 2. `a_s`, the **search move**: a fresh search of `s` under `--search-config` at
//!    `--search-iters`. The corpus's own pick is not used — it was searched with Gumbel noise.
//! 3. Overrides (`a_s != a_p`) are kept, plus `--control-frac` of the rest as a control (or every
//!    decision under `--all`). Each row carries its `keep_prob`, so `1 / keep_prob` weights the
//!    rows back to the population of decisions.
//! 4. Each kept move is applied to `s` and the drive is played out `--playouts` times with the
//!    policy bot on both sides and real dice. Playout `i` of `a_p` and of `a_s` share one dice
//!    seed (common random numbers), so their difference is paired. A control row plays `a_p` only:
//!    its `a_s` is the same move, so the second arm would replay the first exactly.
//! 5. A drive ends where the corpus's and the drive benchmark's do ([`DriveStart`]), and scores
//!    in the value target's units, in `m`'s frame: +1 `m` scored, -1 the opponent did, 0 neither.
//!
//! Policy-only from `s` *is* `a_p` then policy-only, so `policy.mc_*` is also `MC(s)`, and against
//! `v_state` it gives the value head's per-state error with the dice averaged out (H1) — for
//! every row, override or control.
//!
//! **Corpus states are rebuilt by replay, not read.** A deserialised `GameState` has no path
//! buffer (it is `#[serde(skip)]`), so a mid-activation state's move offerings are gone: the legal
//! set loses every path action and stepping one panics. The engine needs no help to rebuild it:
//! [`position_state`] regenerates the trajectory's start from its seed (it is how every corpus
//! drive was drawn), and replaying the recorded moves under the state's own seeded dice reproduces
//! each decision exactly. Every replayed state is checked against the recorded one, and a
//! trajectory that diverges (a corpus from an engine that has since changed) is skipped.
//!
//! Usage:
//! ```text
//! BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
//! cargo run --release -p botbowl-ui -- override-audit \
//!     --corpus runs/loopmix16x9g/gen06/shard4.jsonl runs/loopmix16x9g/gen06/shard7.jsonl \
//!     --model models/az_v7/bbnet_mix16x9g_gen05.onnx \
//!     --search-config cfgs/gumbel16_f1000.toml --search-iters 1000 \
//!     --decisions 1000 --playouts 64 --parallel 8 --out runs/exp065/audit_g_gen05.jsonl
//! scripts/override_audit_summary.py runs/exp065/audit_g_gen05.jsonl
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_data::{BoardCapacity, Team, Trajectory, TrajectoryMeta};
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::{Action as EngineAction, BoardDims, TeamType};
use botbowl_mcts::pruning::search_actions;
use botbowl_mcts::SearchBudget;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::board_sizes::board_label;
use botbowl_play::bots::{load_mcts_config, load_nn, make_mcts, Evaluator, SearchConfig};
use botbowl_play::drives::{position_state, DriveStart};
use botbowl_play::generate::RandomStartBias;
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::OverrideAuditArgs;

// ---------------------------------------------------------------------------------------------
// The policy bot
// ---------------------------------------------------------------------------------------------

/// The legal set a search root offers: the engine's actions minus `should_prune`, or all of them
/// when pruning would leave none, in the engine's sorted order. The search's own definition
/// (`botbowl_mcts::pruning::search_actions`), not a copy of it.
pub fn search_legal(state: &GameState) -> Vec<EngineAction> {
    search_actions(state)
}

/// Index of the first maximum. The Gumbel root sorts its candidates stably by `ln prior`, so on a
/// tie the policy-only preset plays the earliest action too.
fn first_argmax(priors: &[f32]) -> usize {
    let mut best = 0;
    for (i, p) in priors.iter().enumerate() {
        if *p > priors[best] {
            best = i;
        }
    }
    best
}

/// The bare policy, without a search: what `cfgs/policy_only.toml` (`gumbel_m = 1`, no noise)
/// plays, at one forward per decision instead of the preset's root expansion plus its descents.
/// A decision with one legal move costs no forward at all.
pub struct PolicyBot {
    nn: Arc<NnEvaluator>,
}

impl PolicyBot {
    pub fn new(nn: Arc<NnEvaluator>) -> Self {
        PolicyBot { nn }
    }

    /// The legal set, its priors (softmax × len, as the search sees them) and the argmax's index.
    pub fn priors(&self, state: &GameState) -> (Vec<EngineAction>, Vec<f32>, usize) {
        let legal = search_legal(state);
        let priors = if legal.len() > 1 {
            self.nn.priors(state, &legal)
        } else {
            vec![1.0; legal.len()]
        };
        let best = first_argmax(&priors);
        (legal, priors, best)
    }
}

impl Bot for PolicyBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
        let (legal, _, best) = self.priors(state);
        legal[best]
    }
}

// ---------------------------------------------------------------------------------------------
// Playouts
// ---------------------------------------------------------------------------------------------

/// One playout of the rest of a drive.
#[derive(Clone, Debug, PartialEq)]
pub struct Playout {
    /// In `mover`'s frame: +1 it scored, -1 the opponent did, 0 neither (half, game end, cap).
    pub outcome: f32,
    /// The net's value at the first decision after `first`, in `mover`'s frame; the exact outcome
    /// when `first` ended the drive. `None` without a net.
    pub v_after: Option<f32>,
    /// False only when `max_steps` ran out before the drive ended.
    pub finished: bool,
    pub steps: u32,
}

/// The net's value at `state` in `team`'s frame, in TD units.
pub fn value_for(nn: &NnEvaluator, state: &GameState, team: TeamType) -> f32 {
    in_frame(team, nn.value_home_i64(state) as f32 / 1000.0)
}

/// A Home-centric number in `team`'s frame (`0.0 - x`, so a draw never prints as `-0`).
fn in_frame(team: TeamType, home_centric: f32) -> f32 {
    match team {
        TeamType::Home => home_centric,
        TeamType::Away => 0.0 - home_centric,
    }
}

/// Play `first` (if any) from `state`, then the rest of the drive with `home` / `away` choosing,
/// under real dice from `dice_seed`. The state is cloned: the same `(state, first, dice_seed)`
/// and deterministic bots always give the same playout, which is what pairs two moves' playouts.
#[allow(clippy::too_many_arguments)]
pub fn play_out(
    state: &GameState,
    first: Option<EngineAction>,
    mover: TeamType,
    home: &mut dyn Bot,
    away: &mut dyn Bot,
    nn: Option<&NnEvaluator>,
    dice_seed: u64,
    max_steps: u32,
) -> (Playout, GameState) {
    let mut st = state.clone();
    st.set_seed(dice_seed);
    st.set_dice_mode(DiceMode::RollDice);
    st.set_logging_state(false);
    let drive = DriveStart::of(&st);
    let mut steps = 0u32;
    if let Some(a) = first {
        st.step(a).expect("engine step failed on the audited move");
        steps += 1;
    }
    // The value after the move: read at the first decision, after the bot has asked for its
    // priors, so on the policy bot it is the same forward (the evaluator's memo).
    let mut v_after = match (first, nn) {
        (Some(_), Some(_)) if drive.over(&st) => Some(drive.outcome_for(&st, mover)),
        _ => None,
    };
    let mut want_v = first.is_some() && nn.is_some() && v_after.is_none();
    while !drive.over(&st) && steps < max_steps {
        let action = match st.available_actions.team {
            Some(TeamType::Home) => home.get_action(&st),
            Some(TeamType::Away) => away.get_action(&st),
            None => break,
        };
        if want_v {
            v_after = Some(value_for(nn.expect("want_v implies a net"), &st, mover));
            want_v = false;
        }
        st.step(action).expect("engine step failed during an audit playout");
        steps += 1;
    }
    let playout = Playout {
        outcome: drive.outcome_for(&st, mover),
        v_after,
        finished: drive.over(&st),
        steps,
    };
    (playout, st)
}

/// Mean and standard error of the mean.
fn mean_se(xs: &[f32]) -> (f32, f32) {
    let n = xs.len();
    if n == 0 {
        return (f32::NAN, f32::NAN);
    }
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    if n < 2 {
        return (mean as f32, f32::NAN);
    }
    let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean as f32, (var / n as f64).sqrt() as f32)
}

/// One move's playouts, summarised.
#[derive(Clone, Debug, Serialize)]
struct Arm {
    action: EngineAction,
    /// The engine action type (`Move`, `StartBlitz`, `EndTurn`, ...).
    kind: String,
    positional: bool,
    /// The policy's probability for the move (the prior is softmax × fan), and its rank (0 = the
    /// argmax).
    prior: f32,
    prior_rank: usize,
    /// The search's Q and visits for the move, Q in `m`'s frame and TD units. `None` if unscored.
    q: Option<f32>,
    visits: u32,
    v_mean: f32,
    v_se: f32,
    mc_mean: f32,
    mc_se: f32,
    td_for: u32,
    td_against: u32,
    unfinished: u32,
    #[serde(skip)]
    outcomes: Vec<f32>,
}

#[derive(Serialize)]
struct Row<'a> {
    /// `file:line:sample`, stable across runs on the same inputs.
    decision: String,
    corpus: &'a str,
    traj_seed: Option<u64>,
    sample: usize,
    board: String,
    #[serde(rename = "override")]
    is_override: bool,
    /// The chance this decision was written: 1 for an override (and under `--all`),
    /// `--control-frac` for a non-override. Weight by `1 / keep_prob` for population figures.
    keep_prob: f64,
    mover: Team,
    half: u8,
    /// The mover's turn number in the half.
    turn: u8,
    /// No player of the mover's has acted yet this turn.
    turn_start: bool,
    /// A player is activated (choosing a square, a block target, a die, ...).
    mid_activation: bool,
    proc: Option<&'a str>,
    player_action_type: Option<String>,
    fan: usize,
    /// The net's value at `s`, `m`'s frame.
    v_state: f32,
    root_value: Option<f32>,
    root_visits: u32,
    root_solved: bool,
    corpus_action: EngineAction,
    /// The corpus's one realised drive outcome from `s`, `m`'s frame.
    corpus_outcome: Option<f32>,
    policy: &'a Arm,
    search: &'a Arm,
    /// Mean and SE of the paired `MC(a_s) - MC(a_p)` (0 for a control row).
    diff_mean: f32,
    diff_se: f32,
    playouts: u32,
    search_config: &'a str,
    search_iters: usize,
    model: &'a str,
    /// Wall time of the search alone, and of the whole row (replay, search, playouts).
    search_ms: u64,
    elapsed_ms: u64,
}

fn action_kind(a: &EngineAction) -> (String, bool) {
    match a {
        EngineAction::Positional(at, _) => (format!("{at:?}"), true),
        EngineAction::Simple(at) => (format!("{at:?}"), false),
    }
}

// ---------------------------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------------------------

/// The random-start placement a trajectory was drawn with, from its provenance. `extra.temperature`
/// is the one this seed used (already alternated), so it stands in for both temperatures.
pub fn bias_of(meta: &TrajectoryMeta) -> Result<RandomStartBias, String> {
    if meta.extra.get("mode").map(String::as_str) != Some("random-start") {
        return Err("not a random-start trajectory".into());
    }
    let f = |k: &str| -> Result<f32, String> {
        meta.extra
            .get(k)
            .ok_or_else(|| format!("meta.extra has no {k}"))?
            .parse::<f32>()
            .map_err(|e| format!("meta.extra.{k}: {e}"))
    };
    let temperature = f("temperature")?;
    Ok(RandomStartBias {
        ball_distance: f("ball_distance")?,
        front_line: f("front_line")?,
        mark_teammate: f("mark_teammate")?,
        mark_opponent: f("mark_opponent")?,
        own_side: f("own_side")?,
        temperature,
        temperature2: temperature,
        carried_prob: f("carried_prob")?,
        line_fraction: f("line_fraction")?,
        pocket_fraction: f("pocket_fraction")?,
    })
}

/// Sample `upto` of `traj` as a playable state: the start regenerated from the trajectory's seed,
/// then the recorded moves replayed under its own dice. `Err` names the first state that does not
/// match the recorded one.
pub fn replay_to(traj: &Trajectory, upto: usize) -> Result<GameState, String> {
    let seed = traj.meta.seed.ok_or("trajectory has no seed")?;
    let mut state = position_state(&bias_of(&traj.meta)?, traj.meta.board_dims, seed);
    for (k, sample) in traj.samples.iter().enumerate().take(upto + 1) {
        if state != sample.state {
            return Err(format!("replay diverged from the recorded state at sample {k}"));
        }
        if k == upto {
            return Ok(state);
        }
        state
            .step(sample.chosen_action)
            .map_err(|e| format!("replaying sample {k}: {e:?}"))?;
    }
    Err(format!("trajectory has no sample {upto}"))
}

/// What indexing needs from a trajectory, without building a `GameState` per sample.
#[derive(Deserialize)]
struct LiteTraj {
    meta: TrajectoryMeta,
    samples: Vec<LiteSample>,
}

#[derive(Deserialize)]
struct LiteSample {
    children: Vec<serde::de::IgnoredAny>,
}

/// One decision the audit may sample.
#[derive(Clone, Debug)]
struct Candidate {
    file: usize,
    line: usize,
    offset: u64,
    sample: usize,
}

fn board_matches(filters: &[String], dims: BoardDims) -> bool {
    if filters.is_empty() {
        return true;
    }
    let label = board_label(dims);
    let wh = label.split('/').next().unwrap_or("");
    filters.iter().any(|f| f.trim() == label || f.trim() == wh)
}

/// Every eligible decision in the shards: random-start trajectories on a wanted board, decisions
/// whose recorded root had at least `min_fan` children.
fn index_corpus(args: &OverrideAuditArgs) -> io::Result<Vec<Candidate>> {
    let mut out = Vec::new();
    let capacity = BoardCapacity::current();
    for (fi, path) in args.corpus.iter().enumerate() {
        let mut reader =
            BufReader::new(File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?);
        let mut offset = 0u64;
        let mut line = String::new();
        let mut li = 0usize;
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            let here = offset;
            offset += n as u64;
            li += 1;
            if line.trim().is_empty() {
                continue;
            }
            let t: LiteTraj = serde_json::from_str(&line)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}:{li}: {e}")))?;
            if t.meta.board_capacity != capacity {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{path}: corpus built at capacity {:?}, this binary is {capacity:?}; rebuild with the matching \
                         BOARD_SIZE_W/H/PLAYERS (playable = engine - 2)",
                        t.meta.board_capacity
                    ),
                ));
            }
            if t.meta.extra.get("mode").map(String::as_str) != Some("random-start")
                || !board_matches(&args.boards, t.meta.board_dims)
            {
                continue;
            }
            for (si, s) in t.samples.iter().enumerate() {
                if s.children.len() >= args.min_fan {
                    out.push(Candidate {
                        file: fi,
                        line: li,
                        offset: here,
                        sample: si,
                    });
                }
            }
        }
    }
    Ok(out)
}

pub fn load_trajectory(path: &str, offset: u64) -> io::Result<Trajectory> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut line = String::new();
    BufReader::new(f).read_line(&mut line)?;
    serde_json::from_str(&line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------------------------
// The audit
// ---------------------------------------------------------------------------------------------

struct Ctx<'a> {
    args: &'a OverrideAuditArgs,
    nn: Arc<NnEvaluator>,
    search: SearchConfig,
    config_name: String,
}

/// The decision's seed: playout dice and the control draw come from it, not from the sampling
/// order, so a decision reads the same in any run over the same inputs.
fn decision_seed(base: u64, c: &Candidate) -> u64 {
    let id = ((c.file as u64) << 48) ^ ((c.line as u64) << 16) ^ c.sample as u64;
    base ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// All playouts of one move from `state`.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    ctx: &Ctx,
    state: &GameState,
    mover: TeamType,
    action: EngineAction,
    seeds: &[u64],
    legal: &[EngineAction],
    priors: &[f32],
    children: &HashMap<EngineAction, (Option<i64>, u32)>,
) -> Arm {
    let mut home = PolicyBot::new(Arc::clone(&ctx.nn));
    let mut away = PolicyBot::new(Arc::clone(&ctx.nn));
    let mut outcomes = Vec::with_capacity(seeds.len());
    let mut vs = Vec::with_capacity(seeds.len());
    let (mut td_for, mut td_against, mut unfinished) = (0, 0, 0);
    for &seed in seeds {
        let (p, _) = play_out(
            state,
            Some(action),
            mover,
            &mut home,
            &mut away,
            Some(&ctx.nn),
            seed,
            ctx.args.max_steps,
        );
        outcomes.push(p.outcome);
        if let Some(v) = p.v_after {
            vs.push(v);
        }
        td_for += (p.outcome > 0.0) as u32;
        td_against += (p.outcome < 0.0) as u32;
        unfinished += (!p.finished) as u32;
    }
    let (mc_mean, mc_se) = mean_se(&outcomes);
    let (v_mean, v_se) = mean_se(&vs);
    let idx = legal.iter().position(|a| *a == action);
    let fan = legal.len() as f32;
    let prior = idx.map_or(f32::NAN, |i| priors[i] / fan);
    let prior_rank = idx.map_or(usize::MAX, |i| priors.iter().filter(|p| **p > priors[i]).count());
    let (q, visits) = children.get(&action).copied().unwrap_or((None, 0));
    let (kind, positional) = action_kind(&action);
    Arm {
        action,
        kind,
        positional,
        prior,
        prior_rank,
        q: q.map(|q| in_frame(mover, q as f32 / 1000.0)),
        visits,
        v_mean,
        v_se,
        mc_mean,
        mc_se,
        td_for,
        td_against,
        unfinished,
        outcomes,
    }
}

enum Audited {
    Row { line: String, is_override: bool },
    Dropped,
    Skipped(String),
}

fn audit_one(ctx: &Ctx, c: &Candidate) -> io::Result<Audited> {
    let t0 = Instant::now();
    let args = ctx.args;
    let path = &args.corpus[c.file];
    let traj = load_trajectory(path, c.offset)?;
    let state = match replay_to(&traj, c.sample) {
        Ok(s) => s,
        Err(e) => return Ok(Audited::Skipped(e)),
    };
    let Some(mover) = state.available_actions.team else {
        return Ok(Audited::Skipped("no team to move".into()));
    };
    let policy = PolicyBot::new(Arc::clone(&ctx.nn));
    let (legal, priors, best) = policy.priors(&state);
    if legal.len() < args.min_fan {
        return Ok(Audited::Skipped(format!("fan {} < --min-fan", legal.len())));
    }
    let a_p = legal[best];
    let v_state = value_for(&ctx.nn, &state, mover);

    let t_search = Instant::now();
    let mut bot = make_mcts(&ctx.search, Evaluator::Nn, Some(&ctx.nn));
    let (a_s, rec) = bot.get_action_with_record(&state);
    drop(bot);
    let search_ms = t_search.elapsed().as_millis() as u64;
    let children: HashMap<EngineAction, (Option<i64>, u32)> =
        rec.children.iter().map(|ch| (ch.action, (ch.q, ch.visits))).collect();

    let seed = decision_seed(args.seed, c);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let is_override = a_s != a_p;
    let keep_prob = if is_override || args.all {
        1.0
    } else {
        args.control_frac
    };
    if !is_override && !args.all && rng.gen::<f64>() >= args.control_frac {
        return Ok(Audited::Dropped);
    }
    let seeds: Vec<u64> = (0..args.playouts).map(|_| rng.next_u64()).collect();
    let arm_p = run_arm(ctx, &state, mover, a_p, &seeds, &legal, &priors, &children);
    let arm_s = if is_override {
        run_arm(ctx, &state, mover, a_s, &seeds, &legal, &priors, &children)
    } else {
        arm_p.clone()
    };
    let diffs: Vec<f32> = arm_s.outcomes.iter().zip(&arm_p.outcomes).map(|(s, p)| s - p).collect();
    let (diff_mean, diff_se) = if is_override { mean_se(&diffs) } else { (0.0, 0.0) };

    let sample = &traj.samples[c.sample];
    let turn = match mover {
        TeamType::Home => state.info.home_turn,
        TeamType::Away => state.info.away_turn,
    };
    let row = Row {
        decision: format!("{}:{}:{}", c.file, c.line, c.sample),
        corpus: path,
        traj_seed: traj.meta.seed,
        sample: c.sample,
        board: board_label(state.board_dims),
        is_override,
        keep_prob,
        mover: mover.into(),
        half: state.info.half,
        turn,
        turn_start: state.info.active_player.is_none() && !state.get_players_on_pitch_in_team(mover).any(|p| p.used),
        mid_activation: state.info.active_player.is_some(),
        proc: state.proc_stack_top(),
        player_action_type: state.info.player_action_type.map(|a| format!("{a:?}")),
        fan: legal.len(),
        v_state,
        root_value: rec.root_value.map(|q| in_frame(mover, q as f32 / 1000.0)),
        root_visits: rec.root_visits,
        root_solved: rec.root_solved,
        corpus_action: sample.chosen_action,
        corpus_outcome: sample.outcome_value.map(|z| in_frame(mover, z)),
        policy: &arm_p,
        search: &arm_s,
        diff_mean,
        diff_se,
        playouts: args.playouts,
        search_config: &ctx.config_name,
        search_iters: args.search_iters,
        model: &args.model,
        search_ms,
        elapsed_ms: t0.elapsed().as_millis() as u64,
    };
    Ok(Audited::Row {
        line: serde_json::to_string(&row)?,
        is_override,
    })
}

pub fn run(args: OverrideAuditArgs) -> io::Result<()> {
    let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    if !(0.0..=1.0).contains(&args.control_frac) {
        return Err(invalid("--control-frac must be in [0, 1]".into()));
    }
    let preset = load_mcts_config(&args.search_config)?;
    if preset.config.gumbel_scale > 0.0 {
        eprintln!(
            "warning: {} draws Gumbel noise (gumbel_scale {}); the audit means the deterministic eval search",
            preset.name, preset.config.gumbel_scale
        );
    }
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        Evaluator::Nn,
        Some(&args.model),
        "--model is required",
        server.as_deref(),
    )?
    .expect("Evaluator::Nn always loads a net");
    let search = SearchConfig {
        budget: SearchBudget::Iterations(args.search_iters),
        workers: 1,
        puct: None,
        horizon_turns: None,
        fpu_reduction: None,
        config: Some(preset.config),
    };
    let ctx = Ctx {
        args: &args,
        nn,
        search,
        config_name: preset.name.clone(),
    };

    let started = Instant::now();
    let mut candidates = index_corpus(&args)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    candidates.shuffle(&mut rng);
    eprintln!(
        "override-audit: {} eligible decisions in {} shard(s) ({:.0}s to index); search {}@{}, {} playouts, target {} {}",
        candidates.len(),
        args.corpus.len(),
        started.elapsed().as_secs_f64(),
        preset.name,
        args.search_iters,
        args.playouts,
        args.decisions,
        if args.all { "decisions" } else { "overrides" },
    );

    let out = Mutex::new(File::create(&args.out)?);
    let next = AtomicUsize::new(0);
    let kept = AtomicU32::new(0);
    let overrides = AtomicU32::new(0);
    let searched = AtomicU32::new(0);
    let skipped = AtomicU32::new(0);
    let done = || {
        let n = if args.all {
            kept.load(Ordering::Relaxed)
        } else {
            overrides.load(Ordering::Relaxed)
        };
        n >= args.decisions
    };
    let worker = || -> io::Result<()> {
        while !done() {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(c) = candidates.get(i) else { return Ok(()) };
            searched.fetch_add(1, Ordering::Relaxed);
            match audit_one(&ctx, c)? {
                Audited::Row { line, is_override } => {
                    {
                        let mut f = out.lock().expect("output mutex");
                        f.write_all(line.as_bytes())?;
                        f.write_all(b"\n")?;
                        f.flush()?;
                    }
                    let k = kept.fetch_add(1, Ordering::Relaxed) + 1;
                    let o = overrides.fetch_add(is_override as u32, Ordering::Relaxed) + is_override as u32;
                    eprintln!(
                        "[{k} rows, {o} overrides / {} searched] {:.0}s",
                        searched.load(Ordering::Relaxed),
                        started.elapsed().as_secs_f64()
                    );
                }
                Audited::Dropped => {}
                Audited::Skipped(why) => {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    eprintln!("skipped decision {}:{}:{}: {why}", c.file, c.line, c.sample);
                }
            }
        }
        Ok(())
    };
    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = (0..args.parallel.max(1))
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("audit-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(scope, &worker)
                    .expect("spawn audit thread")
            })
            .collect();
        for h in handles {
            h.join().expect("audit thread panicked")?;
        }
        Ok(())
    })?;

    let searched = searched.load(Ordering::Relaxed);
    let o = overrides.load(Ordering::Relaxed);
    let (forwards, nanos) = botbowl_nn::eval::profile_counters();
    eprintln!(
        "wrote {} rows ({o} overrides) to {} from {searched} searched decisions ({} skipped, override rate {:.3}) in {:.0}s; \
         {forwards} forwards, {:.2} ms each",
        kept.load(Ordering::Relaxed),
        args.out,
        skipped.load(Ordering::Relaxed),
        o as f64 / (searched.max(1) as f64),
        started.elapsed().as_secs_f64(),
        nanos as f64 / 1e6 / forwards.max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_data::{Outcome, Sample};
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::Position;
    use botbowl_engine::core::table::PosAT;
    use botbowl_engine::scripted_bot::ScriptedBot;
    use botbowl_mcts::MctsBot;
    use botbowl_play::drives::{attacker_of, play_drive_game};
    use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig};

    fn tiny_net() -> Arc<NnEvaluator> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");
        Arc::new(NnEvaluator::from_path(path).expect("load tiny.onnx"))
    }

    /// The boards a test plays on: the 14x7/4 tier and the compiled capacity, where they fit.
    fn boards() -> Vec<BoardDims> {
        let mut b: Vec<BoardDims> = BoardDims::try_new(16, 9, 4).ok().into_iter().collect();
        b.push(BoardDims::from_env());
        b
    }

    /// (a) The policy bot plays exactly what the `policy_only` preset plays, along whole drives
    /// (turn starts, mid-activation move fans, block targets and dice), on a net whose priors are
    /// arbitrary but fixed.
    #[test]
    fn the_policy_bot_plays_what_the_policy_only_preset_plays() {
        let nn = tiny_net();
        let preset = load_mcts_config(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cfgs/policy_only.toml"
        )))
        .expect("cfgs/policy_only.toml");
        let mut policy = PolicyBot::new(Arc::clone(&nn));
        let mut compared = 0;
        let mut mid_activation = 0;
        for board in boards() {
            for seed in 0..4u64 {
                let mut state = position_state(&RandomStartBias::default(), board, 500 + seed);
                state.set_seed(seed);
                let drive = DriveStart::of(&state);
                for _ in 0..60 {
                    if drive.over(&state) || state.available_actions.team.is_none() {
                        break;
                    }
                    let mut preset_bot = MctsBot::with_budget_and_config(SearchBudget::Iterations(8), preset.config)
                        .with_evaluator(Arc::clone(&nn));
                    let want = preset_bot.get_action(&state);
                    let got = policy.get_action(&state);
                    assert_eq!(
                        got, want,
                        "board {board:?} seed {seed}: policy bot and policy_only preset disagree"
                    );
                    compared += 1;
                    mid_activation += state.info.active_player.is_some() as u32;
                    state.step(got).unwrap();
                }
            }
        }
        assert!(
            compared > 20 && mid_activation > 5,
            "too few decisions compared: {compared} ({mid_activation} mid-activation)"
        );
    }

    /// (b) Paired playouts are reproducible: the same state, move and dice seed give the same
    /// drive, step for step, so `a_p`'s and `a_s`'s playout `i` differ only by the move.
    #[test]
    fn a_playout_is_a_function_of_state_move_and_seed() {
        let nn = tiny_net();
        let board = boards()[0];
        // A position whose drive reaches the dice: under the v8 action layout (plan 047) the
        // fixture net's argmax is often `EndTurn`, and many positions just run out the half.
        let state = position_state(&RandomStartBias::default(), board, 81);
        let mover = attacker_of(&state);
        let (legal, _, best) = PolicyBot::new(Arc::clone(&nn)).priors(&state);
        let mut finals = Vec::new();
        for seed in [1u64, 2, 3] {
            let run = || {
                let (mut h, mut a) = (PolicyBot::new(Arc::clone(&nn)), PolicyBot::new(Arc::clone(&nn)));
                play_out(
                    &state,
                    Some(legal[best]),
                    mover,
                    &mut h,
                    &mut a,
                    Some(&nn),
                    seed,
                    100_000,
                )
            };
            let (p1, s1) = run();
            let (p2, s2) = run();
            assert_eq!(p1, p2, "seed {seed}");
            assert!(s1 == s2, "seed {seed}: final states differ");
            assert!(p1.finished && p1.v_after.is_some());
            finals.push(s1);
        }
        assert!(
            !(finals[0] == finals[1] && finals[1] == finals[2]),
            "three dice seeds played the identical drive: the seed is not reaching the dice"
        );
    }

    /// (c) A playout ends and scores the way the drive benchmark does: on random positions it
    /// reproduces `play_drive_game`'s touchdowns exactly, and on a constructed touchdown it scores
    /// +1 for the scorer and -1 for the other side, which is the corpus's value target.
    #[test]
    fn a_playout_ends_and_scores_like_the_drive_benchmark() {
        // `botbowl_play::eval`'s seed mixes, which `play_drive_game` applies to its two bots.
        const CANDIDATE_SEED_MIX: u64 = 0x3C3C_3C3C_3C3C_3C3C;
        const OPPONENT_SEED_MIX: u64 = 0xC3C3_C3C3_C3C3_C3C3;
        let board = boards()[0];
        for seed in 0..6u64 {
            let state = position_state(&RandomStartBias::default(), board, 300 + seed);
            let attacker = attacker_of(&state);
            let dice = 9_000 + seed;
            let (mut c, mut o) = (ScriptedBot::new(), ScriptedBot::new());
            let line = play_drive_game(&mut c, &mut o, "t", 0, seed, state.clone(), true, dice, 20_000);
            let (mut c, mut o) = (ScriptedBot::new(), ScriptedBot::new());
            c.set_seed(ChaCha8Rng::seed_from_u64(dice ^ CANDIDATE_SEED_MIX));
            o.set_seed(ChaCha8Rng::seed_from_u64(dice ^ OPPONENT_SEED_MIX));
            let (home, away): (&mut dyn Bot, &mut dyn Bot) = match attacker {
                TeamType::Home => (&mut c, &mut o),
                TeamType::Away => (&mut o, &mut c),
            };
            let (p, end) = play_out(&state, None, attacker, home, away, None, dice, 20_000);
            let (h, a) = DriveStart::of(&state).scored(&end);
            assert_eq!((h, a), (line.home_score, line.away_score), "seed {seed}");
            assert_eq!(p.finished, line.finished, "seed {seed}");
            let attacker_td = match attacker {
                TeamType::Home => h as f32 - a as f32,
                TeamType::Away => a as f32 - h as f32,
            };
            assert_eq!(p.outcome, attacker_td.clamp(-1.0, 1.0), "seed {seed}");
        }

        // Home carries the ball to its endzone at x = 1: no dice, a certain touchdown.
        let start = Position::new((2, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start)
            .add_away_player(Position::new((5, 5)))
            .add_ball_pos(start)
            .build();
        state.step_positional(PosAT::StartMove, start);
        let td = EngineAction::Positional(PosAT::Move, Position::new((1, 5)));
        let (mut h, mut a) = (ScriptedBot::new(), ScriptedBot::new());
        let (home_view, end) = play_out(&state, Some(td), TeamType::Home, &mut h, &mut a, None, 5, 100);
        assert_eq!(home_view.outcome, 1.0);
        assert!(home_view.finished);
        assert_eq!(home_view.steps, 1, "the drive ends at the touchdown");
        assert_eq!(DriveStart::of(&state).outcome_for(&end, TeamType::Away), -1.0);
        // The same convention as the value head's target: the corpus labels this decision +1.
        let sample = Sample {
            state: state.clone(),
            to_move: Team::Home,
            chosen_action: td,
            children: Vec::new(),
            root_value: None,
            root_visits: 0,
            root_solved: false,
            outcome_value: None,
            scripted: false,
        };
        let traj = Trajectory::new(
            TrajectoryMeta::new("random-start", state.board_dims),
            vec![sample],
            Outcome::from_state(&end, None),
        );
        assert_eq!(
            botbowl_nn::targets::value_target(&traj.samples[0]),
            Some(home_view.outcome)
        );
    }

    /// Replay rebuilds every corpus decision, path offerings included, which a deserialised state
    /// does not have.
    #[test]
    fn replay_rebuilds_corpus_states_with_their_path_offerings() {
        let board = boards()[0];
        let cfg = GenerateConfig {
            mode: GenMode::RandomStart,
            search: SearchConfig::iterations(4),
            evaluator: Evaluator::Heuristic,
            model: None,
            max_steps: 80,
            lecture: None,
            difficulty: botbowl_curriculum::Difficulty::Easy,
            bias: RandomStartBias::default(),
            board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
            config_name: None,
            exploration: None,
            next_drive: false,
        };
        let original = play_trajectory(&cfg, None, 4242).unwrap().remove(0);
        let traj: Trajectory = serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
        let mut with_paths = 0;
        let mut forced = 0;
        for (k, s) in original.samples.iter().enumerate() {
            // A forced decision (one action after pruning) is played unsearched but still recorded,
            // with that action as its only child: replay needs every `chosen_action`.
            if s.children.len() == 1 {
                assert!(s.scripted && s.root_value.is_none(), "sample {k}: forced record");
                assert_eq!(search_legal(&s.state), vec![s.chosen_action], "sample {k}");
                forced += 1;
            }
            let rebuilt = replay_to(&traj, k).expect("replay");
            assert!(rebuilt == s.state);
            assert_eq!(rebuilt.get_all_actions(), s.state.get_all_actions(), "sample {k}");
            // What the deserialised state lost: the offerings in its path buffer, when it had any.
            let read = traj.samples[k].state.get_all_actions();
            assert!(read.iter().all(|a| rebuilt.get_all_actions().contains(a)), "sample {k}");
            with_paths += (read.len() < rebuilt.get_all_actions().len()) as u32;
        }
        assert!(
            with_paths > 0,
            "no decision lost its path offerings to serde: either the test trajectory has no move fan, or the \
             buffer is serialised now and replay is not needed"
        );
        assert!(
            forced > 0,
            "the trajectory should hold a forced decision for replay to step through"
        );
    }
}

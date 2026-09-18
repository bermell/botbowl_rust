//! Bot construction shared by generation and eval.
//!
//! Every knob is explicit here; the CLI layer maps its flags onto these
//! types, and the plan-041 wire protocol will carry them verbatim. The
//! `Option` search knobs mean "leave `MctsBot`'s default (which may come
//! from the environment)": `dataset` never set them, `eval` always did,
//! and both behaviours must survive the extraction unchanged.

use std::io;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::{BackupMode, MctsBot, PuctMode, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

/// Which leaf evaluator the MCTS bot uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Evaluator {
    #[default]
    Heuristic,
    PureTd,
    /// NN value + NN policy priors.
    Nn,
    /// NN value only, heuristic priors.
    NnValue,
}

impl Evaluator {
    pub fn needs_model(self) -> bool {
        matches!(self, Evaluator::Nn | Evaluator::NnValue)
    }
}

/// Search knobs for one `MctsBot`. `None` keeps the bot's own default.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub budget: SearchBudget,
    pub workers: usize,
    pub puct: Option<PuctMode>,
    pub horizon_turns: Option<u8>,
    pub backup: Option<BackupMode>,
    pub fpu_reduction: Option<f32>,
}

impl SearchConfig {
    /// A plain iteration budget with every other knob at its default.
    pub fn iterations(iters: usize) -> Self {
        SearchConfig {
            budget: SearchBudget::Iterations(iters),
            workers: 1,
            puct: None,
            horizon_turns: None,
            backup: None,
            fpu_reduction: None,
        }
    }
}

/// Build an `MctsBot` from a search config and an evaluator. Panics if the
/// evaluator needs a net and none was passed — callers load it up front
/// with [`load_nn`] so a bad path fails before hours of games.
pub fn make_mcts(search: &SearchConfig, evaluator: Evaluator, nn: Option<&Arc<NnEvaluator>>) -> MctsBot {
    let mut bot = MctsBot::new(search.budget).with_workers(search.workers);
    if let Some(p) = search.puct {
        bot = bot.with_puct(p);
    }
    if let Some(h) = search.horizon_turns {
        bot = bot.with_horizon_turns(h);
    }
    if let Some(b) = search.backup {
        bot = bot.with_backup(b);
    }
    if let Some(k) = search.fpu_reduction {
        bot = bot.with_fpu_reduction(k);
    }
    match evaluator {
        Evaluator::Heuristic => bot,
        Evaluator::PureTd => bot.with_pure_td(),
        Evaluator::Nn => bot.with_evaluator(Arc::clone(nn.expect("nn evaluator loaded"))),
        Evaluator::NnValue => bot.with_nn_value(Arc::clone(nn.expect("nn evaluator loaded"))),
    }
}

/// Load the ONNX evaluator an nn/nn-value bot needs; `None` otherwise.
/// `missing_msg` is the error when the model path wasn't given.
pub fn load_nn(
    evaluator: Evaluator,
    model: Option<&str>,
    missing_msg: &str,
    server: Option<&Path>,
) -> io::Result<Option<Arc<NnEvaluator>>> {
    if !evaluator.needs_model() {
        return Ok(None);
    }
    let path = model.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, missing_msg.to_string()))?;
    // Each evaluator names its own model at handshake and gets its own
    // canary, so two nets can share one sidecar socket with no chance of
    // being cross-wired.
    let eval = NnEvaluator::from_path_with_server(path, server)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("failed to load {path}: {e}")))?;
    Ok(Some(Arc::new(eval)))
}

/// Report-card label for an MCTS bot's evaluator: `mcts(heuristic)`,
/// `mcts(nn:<model>)`, ...
pub fn evaluator_label(evaluator: Evaluator, model: Option<&str>) -> String {
    match evaluator {
        Evaluator::Heuristic => "mcts(heuristic)".to_string(),
        Evaluator::PureTd => "mcts(pure-td)".to_string(),
        Evaluator::Nn => format!("mcts(nn:{})", model.unwrap_or("?")),
        Evaluator::NnValue => format!("mcts(nn-value:{})", model.unwrap_or("?")),
    }
}

/// Resolve a `(mode, c)` pair into a `PuctMode`. `Err` on an unknown mode
/// so a multi-hour head-to-head refuses to start rather than running the
/// wrong arm.
pub fn parse_puct(mode: &str, c: Option<f32>) -> Result<PuctMode, String> {
    match mode {
        "raw" => Ok(match c {
            Some(c) => PuctMode::Raw { c },
            None => PuctMode::raw(),
        }),
        "normalised" | "normalized" | "norm" => Ok(PuctMode::normalised(c.unwrap_or(1.0))),
        other => Err(format!("expected `raw` or `normalised`, got `{other}`")),
    }
}

/// Same contract as [`parse_puct`].
pub fn parse_backup(s: &str) -> Result<BackupMode, String> {
    BackupMode::parse(s).ok_or_else(|| format!("expected `minimax` or `mean`, got `{s}`"))
}

/// The bot in eval's candidate seat. `Mcts` is the normal candidate; the
/// other two exist to take search out of the picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CandidateBot {
    #[default]
    Mcts,
    Scripted,
    Random,
}

pub fn make_candidate_bot(
    kind: CandidateBot,
    search: &SearchConfig,
    evaluator: Evaluator,
    nn: Option<&Arc<NnEvaluator>>,
) -> Box<dyn Bot> {
    match kind {
        CandidateBot::Mcts => Box::new(make_mcts(search, evaluator, nn)),
        CandidateBot::Scripted => Box::new(ScriptedBot::new()),
        CandidateBot::Random => Box::new(RandomBot::new()),
    }
}

/// Self-describing candidate label for a report: non-default search knobs
/// go in so plan-032 arms that differ only in them are distinguishable.
pub fn candidate_label(kind: CandidateBot, search: &SearchConfig, evaluator: Evaluator, model: Option<&str>) -> String {
    match kind {
        CandidateBot::Mcts => {
            let base = evaluator_label(evaluator, model);
            let mut knobs: Vec<String> = Vec::new();
            if let Some(b @ BackupMode::Mean) = search.backup {
                knobs.push(b.label().to_string());
            }
            if let Some(k) = search.fpu_reduction.filter(|k| *k > 0.0) {
                knobs.push(format!("fpu_k={k}"));
            }
            if knobs.is_empty() {
                base
            } else {
                format!("{base} [{}]", knobs.join(" "))
            }
        }
        CandidateBot::Scripted => "scripted".to_string(),
        CandidateBot::Random => "random".to_string(),
    }
}

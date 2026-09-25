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
use botbowl_mcts::{BackupMode, MctsBot, MctsConfig, PuctMode, SearchBudget};
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
///
/// Two ways to configure a bot live here, and they are deliberately exclusive. The `Option` knobs
/// are the historical per-flag overrides on top of `MctsBot`'s env-driven default. `config` is a
/// whole [`MctsConfig`] loaded from a named preset (plan 043); when it is `Some` it replaces that
/// default outright — environment included — so an A/B run is reproducible and the preset's name
/// says exactly what played.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub budget: SearchBudget,
    pub workers: usize,
    pub puct: Option<PuctMode>,
    pub horizon_turns: Option<u8>,
    pub backup: Option<BackupMode>,
    pub fpu_reduction: Option<f32>,
    /// A named preset, wholesale. Serde-defaulted so a worker built before plan 043 is the only
    /// thing that changes shape on the wire, not every existing caller.
    #[serde(default)]
    pub config: Option<MctsConfig>,
}

impl SearchConfig {
    /// Fill `config` from **this** process's environment when no preset named one, so the whole
    /// configuration travels with the job instead of being re-derived wherever the game lands.
    ///
    /// Plan 041's rule is that a job describes its own games. `config: None` quietly broke it:
    /// it means "whatever `MctsConfig::from_env()` says on the machine that plays this", and more
    /// than half of `MctsConfig` — `tree_reuse`, `virtual_loss`, `tie_break`, `memory_mode`,
    /// `horizon`, the PUCT range floor — has no per-knob field here at all. A stray
    /// `BLOOD_MCTS_TREE_REUSE=off` on one helper box therefore changed that box's search, showed
    /// up nowhere in `report.json`, and was indistinguishable from the arm under test.
    ///
    /// The hub calls this at submit time, on both seats. Single-process callers do not need it:
    /// there, "the environment" and "the submitter" are the same machine.
    pub fn pinned_to_env(mut self) -> Self {
        if self.config.is_none() {
            let mut cfg = MctsConfig::from_env();
            // Diagnostics are a property of the terminal someone is watching, not of the bot —
            // and `stats` walks the whole DAG after every search. Never push them onto a fleet.
            cfg.stats = false;
            cfg.leaf_stats = false;
            cfg.debug_root = false;
            self.config = Some(cfg);
        }
        self
    }

    /// A plain iteration budget with every other knob at its default.
    pub fn iterations(iters: usize) -> Self {
        SearchConfig {
            budget: SearchBudget::Iterations(iters),
            workers: 1,
            puct: None,
            horizon_turns: None,
            backup: None,
            fpu_reduction: None,
            config: None,
        }
    }
}

/// A [`MctsConfig`] together with the name it is known by.
///
/// The name is what reaches `report.json`, a rung label and a trajectory's provenance, so a result
/// can be traced back to the configuration that produced it. Kept beside the config rather than in
/// it because `SearchConfig` must stay `Copy` to cross the hub protocol unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedConfig {
    pub name: String,
    pub config: MctsConfig,
}

/// Read a bot preset from a TOML file.
///
/// The file names only the knobs it overrides; everything else comes from [`MctsConfig::new`], the
/// shipped defaults, **not** from the environment. The config's name is the file stem, so
/// `cfgs/aggressive.toml` plays as `aggressive`.
///
/// ```toml
/// backup = "mean"
/// fpu_reduction = 0.25
/// puct = { mode = "normalised_q", c = 1.4, range_floor = 0.1 }
/// ```
pub fn load_mcts_config(path: &Path) -> io::Result<NamedConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| io::Error::new(e.kind(), format!("failed to read bot config {}: {e}", path.display())))?;
    let config: MctsConfig = toml::from_str(&text).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse bot config {}: {e}", path.display()),
        )
    })?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("config")
        .to_string();
    Ok(NamedConfig { name, config })
}

/// Build an `MctsBot` from a search config and an evaluator. Panics if the
/// evaluator needs a net and none was passed — callers load it up front
/// with [`load_nn`] so a bad path fails before hours of games.
pub fn make_mcts(search: &SearchConfig, evaluator: Evaluator, nn: Option<&Arc<NnEvaluator>>) -> MctsBot {
    // A preset wins wholesale: `with_budget_and_config` is the env-free constructor, so nothing the
    // environment says can reach a named configuration. `workers` stays a CLI concern either way —
    // it is a property of the machine, not of the bot being compared.
    let mut bot = match search.config {
        Some(config) => MctsBot::with_budget_and_config(search.budget, config).with_workers(search.workers),
        None => MctsBot::new(search.budget).with_workers(search.workers),
    };
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
///
/// `config_name` is the plan-043 preset, if one was named. It appends as `@name`, because with a
/// preset the individual knobs no longer describe the bot — the name does, and it is the thing
/// that can be looked up in `cfgs/`.
pub fn candidate_label(
    kind: CandidateBot,
    search: &SearchConfig,
    evaluator: Evaluator,
    model: Option<&str>,
    config_name: Option<&str>,
) -> String {
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
            let base = match config_name {
                Some(name) => format!("{base}@{name}"),
                None => base,
            };
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

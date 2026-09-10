//! `BotSpec -> a bot the session can drive`, and the model catalogue the
//! lobby offers.
//!
//! Two things are deliberate here:
//!
//! 1. **A closed enum, not `Box<dyn Bot>`.** The session wants
//!    `last_search()` / `explore()` off the MCTS bot for the inspector, and
//!    downcasting a trait object to get them is worse than three variants.
//! 2. **No environment variables.** `MctsConfig::new()` (not `from_env`) is
//!    the base, so a `BLOOD_MCTS_*` left in the shell cannot silently move
//!    the bot the lobby said it was building — and two differently-configured
//!    bots can coexist in one process. That is what the `MctsConfig` refactor
//!    in `botbowl-mcts` was for.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::dynamics::MemoryMode;
use botbowl_mcts::report::{NodeView, SearchSummary};
use botbowl_mcts::{BackupMode, BbAction, MctsBot, MctsConfig, PuctMode, SearchBudget, TieBreak};
use botbowl_nn::eval::NnEvaluator;
use botbowl_web_proto::msg::{BackupSpec, BotSpec, Budget, EvaluatorSpec, MctsSpec, ModelInfo, PuctSpec, TieBreakSpec};

/// The bot playing one side of one game.
pub enum SessionBot {
    Random(RandomBot),
    Scripted(ScriptedBot),
    Mcts(MctsBot),
}

impl SessionBot {
    pub fn get_action(&mut self, state: &GameState) -> EngineAction {
        match self {
            SessionBot::Random(b) => b.get_action(state),
            SessionBot::Scripted(b) => b.get_action(state),
            SessionBot::Mcts(b) => b.get_action(state),
        }
    }

    /// Only the MCTS bot has a search to report on.
    pub fn last_search(&self) -> Option<&SearchSummary> {
        match self {
            SessionBot::Mcts(b) => b.last_search(),
            _ => None,
        }
    }

    pub fn principal_variation(&self, max_depth: usize) -> Vec<botbowl_mcts::report::Edge> {
        match self {
            SessionBot::Mcts(b) => b.principal_variation(max_depth),
            _ => Vec::new(),
        }
    }

    pub fn explore(&self, path: &[BbAction], with_state: bool) -> Option<NodeView> {
        match self {
            SessionBot::Mcts(b) => b.explore(path, with_state),
            _ => None,
        }
    }

    pub fn seed(&mut self, rng: rand_chacha::ChaCha8Rng) {
        match self {
            SessionBot::Random(b) => b.set_seed(rng),
            SessionBot::Scripted(b) => b.set_seed(rng),
            // `MctsBot` has no RNG of its own — its nondeterminism comes from
            // worker interleaving and HashMap order, which a seed cannot pin.
            SessionBot::Mcts(_) => {}
        }
    }
}

/// Loaded nets, keyed by the path the lobby offered. Loading an ONNX model
/// through tract takes long enough that reloading it per game is noticeable,
/// and `NnEvaluator` is `Send + Sync` by design.
#[derive(Default)]
pub struct ModelCache {
    loaded: Mutex<HashMap<PathBuf, Arc<NnEvaluator>>>,
}

impl ModelCache {
    pub fn get(&self, path: &Path) -> Result<Arc<NnEvaluator>, String> {
        let mut loaded = self.loaded.lock().unwrap();
        if let Some(nn) = loaded.get(path) {
            return Ok(Arc::clone(nn));
        }
        let nn = Arc::new(NnEvaluator::from_path(path).map_err(|e| format!("could not load {}: {e}", path.display()))?);
        loaded.insert(path.to_path_buf(), Arc::clone(&nn));
        Ok(nn)
    }
}

/// List the `.onnx` files under `dir`, newest first, with the `_WxH_` tag
/// parsed out of the filename.
///
/// The tag is the *only* guard against a board/model mismatch, which panics
/// inside `NnEvaluator` rather than erroring — nothing in Rust parsed these
/// filenames before, so the convention becomes load-bearing here.
pub fn list_models(dir: &Path) -> Vec<ModelInfo> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut models: Vec<(std::time::SystemTime, ModelInfo)> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "onnx"))
        .map(|e| {
            let path = e.path();
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let modified = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            (
                modified,
                ModelInfo {
                    path: path.to_string_lossy().to_string(),
                    board_tag: board_tag_of(&name),
                    name,
                },
            )
        })
        .collect();
    models.sort_by(|a, b| b.0.cmp(&a.0));
    models.into_iter().map(|(_, m)| m).collect()
}

/// Pull `14x7` out of `bbnet_14x7_gen0c.onnx`. Returns `None` for a name with
/// no `_<w>x<h>_` segment — such a model is offered for every board, because
/// we have no way to know what it was trained on.
pub fn board_tag_of(name: &str) -> Option<String> {
    name.split('_')
        .find(|seg| {
            let mut parts = seg.splitn(2, 'x');
            matches!(
                (parts.next(), parts.next()),
                (Some(w), Some(h))
                    if !w.is_empty()
                        && !h.is_empty()
                        && w.bytes().all(|b| b.is_ascii_digit())
                        && h.bytes().all(|b| b.is_ascii_digit())
            )
        })
        .map(str::to_string)
}

fn puct_of(spec: PuctSpec) -> PuctMode {
    match spec {
        PuctSpec::Raw { c } => PuctMode::Raw { c },
        PuctSpec::NormalisedQ { c, range_floor } => PuctMode::NormalisedQ { c, range_floor },
    }
}

fn tie_break_of(spec: TieBreakSpec) -> TieBreak {
    match spec {
        TieBreakSpec::Hash => TieBreak::Hash,
        TieBreakSpec::Asc => TieBreak::Asc,
        TieBreakSpec::Desc => TieBreak::Desc,
        TieBreakSpec::Mover => TieBreak::Mover,
    }
}

fn backup_of(spec: BackupSpec) -> BackupMode {
    match spec {
        BackupSpec::Minimax => BackupMode::Minimax,
        BackupSpec::Mean => BackupMode::Mean,
    }
}

/// Translate the lobby's knobs into an `MctsConfig`. Starts from
/// `MctsConfig::new()` — the shipped defaults with **no** env lookups.
pub fn mcts_config(spec: &MctsSpec) -> MctsConfig {
    let mut cfg = MctsConfig::new();
    if let Some(workers) = spec.workers {
        cfg.workers = workers.max(1);
    }
    cfg.memory_mode = MemoryMode::StoreState;
    cfg.tree_reuse = spec.tree_reuse;
    cfg.virtual_loss = spec.virtual_loss;
    cfg.puct = puct_of(spec.puct);
    cfg.tie_break = tie_break_of(spec.tie_break);
    cfg.backup = backup_of(spec.backup);
    cfg.fpu_reduction = spec.fpu_reduction.max(0.0);
    cfg.horizon_turns = spec.horizon_turns.max(1);
    cfg.horizon = spec.horizon;
    cfg
}

/// Build the bot the lobby asked for.
///
/// `allowed_models` is the catalogue the lobby was shown; a model outside it
/// is rejected rather than opened, so a websocket message cannot name an
/// arbitrary path on the host.
pub fn build(spec: &BotSpec, models: &ModelCache, allowed_models: &[ModelInfo]) -> Result<SessionBot, String> {
    match spec {
        BotSpec::Random => Ok(SessionBot::Random(RandomBot::new())),
        BotSpec::Scripted => Ok(SessionBot::Scripted(ScriptedBot::new())),
        BotSpec::Mcts(m) => {
            let budget = match m.budget {
                Budget::Iterations(n) => SearchBudget::Iterations(n.max(1)),
                Budget::Millis(ms) => SearchBudget::Time(Duration::from_millis(ms.max(1))),
            };
            let bot = MctsBot::with_budget_and_config(budget, mcts_config(m));
            let bot = match &m.evaluator {
                EvaluatorSpec::Heuristic => bot,
                EvaluatorSpec::PureTd => bot.with_pure_td(),
                EvaluatorSpec::Nn { model } => bot.with_evaluator(resolve_model(model, models, allowed_models)?),
                EvaluatorSpec::NnValue { model } => bot.with_nn_value(resolve_model(model, models, allowed_models)?),
            };
            Ok(SessionBot::Mcts(bot))
        }
    }
}

fn resolve_model(requested: &str, models: &ModelCache, allowed: &[ModelInfo]) -> Result<Arc<NnEvaluator>, String> {
    let hit = allowed
        .iter()
        .find(|m| m.path == requested || m.name == requested)
        .ok_or_else(|| format!("{requested:?} is not one of this server's models"))?;
    models.get(Path::new(&hit.path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_tags_come_from_the_filename_convention() {
        assert_eq!(board_tag_of("bbnet_14x7_gen0c.onnx"), Some("14x7".into()));
        assert_eq!(board_tag_of("bbnet_8x3_gen0.onnx"), Some("8x3".into()));
        assert_eq!(board_tag_of("bbnet_26x15_db.onnx"), Some("26x15".into()));
        // No tag: offered for every board, because we cannot know better.
        assert_eq!(board_tag_of("score_td.onnx"), None);
    }

    #[test]
    fn a_model_outside_the_catalogue_is_refused() {
        let cache = ModelCache::default();
        let spec = BotSpec::Mcts(MctsSpec {
            evaluator: EvaluatorSpec::Nn {
                model: "/etc/passwd".into(),
            },
            ..Default::default()
        });
        let err = match build(&spec, &cache, &[]) {
            Err(e) => e,
            Ok(_) => panic!("an off-catalogue model must not be loaded"),
        };
        assert!(err.contains("not one of this server's models"), "{err}");
    }

    #[test]
    fn the_lobby_knobs_survive_the_trip_into_mcts_config() {
        let spec = MctsSpec {
            workers: Some(3),
            backup: BackupSpec::Mean,
            puct: PuctSpec::NormalisedQ {
                c: 1.5,
                range_floor: 20.0,
            },
            fpu_reduction: 0.25,
            horizon_turns: 2,
            horizon: false,
            tree_reuse: false,
            virtual_loss: 7,
            tie_break: TieBreakSpec::Mover,
            ..Default::default()
        };
        let cfg = mcts_config(&spec);
        assert_eq!(cfg.workers, 3);
        assert_eq!(cfg.backup, BackupMode::Mean);
        assert!(matches!(cfg.puct, PuctMode::NormalisedQ { c, range_floor } if c == 1.5 && range_floor == 20.0));
        assert_eq!(cfg.fpu_reduction, 0.25);
        assert_eq!(cfg.horizon_turns, 2);
        assert!(!cfg.horizon);
        assert!(!cfg.tree_reuse);
        assert_eq!(cfg.virtual_loss, 7);
        assert_eq!(cfg.tie_break, TieBreak::Mover);
    }
}

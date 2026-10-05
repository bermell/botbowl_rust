//! `BotSpec -> a bot the session can drive`, and the model catalogue the
//! lobby offers.
//!
//! Three things are deliberate here:
//!
//! 1. **A closed enum, not `Box<dyn Bot>`.** The session wants
//!    `last_search()` / `explore()` off the MCTS bot for the inspector, and
//!    downcasting a trait object to get them is worse than two variants.
//! 2. **No environment variables.** `MctsConfig::new()` (not `from_env`) is
//!    the base, so a `BLOOD_MCTS_*` left in the shell cannot silently move
//!    the bot the lobby said it was building — and two differently-configured
//!    bots can coexist in one process. That is what the `MctsConfig` refactor
//!    in `botbowl-mcts` was for.
//! 3. **The MCTS bot always searches on a net, for both value and priors**
//!    (`Evaluator::Nn`). The heuristic, pure-TD and value-only evaluators are
//!    CLI diagnostics; the web app exists to debug the net-guided search we
//!    actually train. The session keeps its own handle on the `NnEvaluator`
//!    so it can read the net out on positions the search never scored — a
//!    human's decision, or a root child before the search moved its value.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType as EngineTeam};
use botbowl_mcts::dynamics::MemoryMode;
use botbowl_mcts::report::{NodeView, SearchSummary, Q_SCALE};
use botbowl_mcts::{BbAction, BbPlayer, MctsBot, MctsConfig, PuctMode, SearchBudget, TieBreak};
use botbowl_nn::eval::NnEvaluator;
use botbowl_web_proto::decision::{ActionPrior, NetReadout};
use botbowl_web_proto::msg::{BoardSpec, BotSpec, Budget, MctsSpec, ModelInfo, PuctSpec, TieBreakSpec};
use botbowl_web_proto::search as ps;

use crate::{mirror, report};

/// A loaded net and the name the read-outs call it by.
#[derive(Clone)]
pub struct Net {
    pub name: String,
    pub eval: Arc<NnEvaluator>,
}

impl Net {
    /// One forward pass over `state`: the value, and the policy over every
    /// legal action of `mover`. The priors are empty when nobody is being
    /// asked for an action (a pending roll, game over).
    ///
    /// A frozen net is a pure function of the state, so reading it out cannot
    /// perturb the game or a search.
    pub fn readout(&self, state: &GameState, mover: EngineTeam) -> NetReadout {
        let value_home = self.eval.value_home_i64(state) as f32 / Q_SCALE;
        let actions: Vec<EngineAction> = if state.pending_roll.is_none() && !state.info.game_over {
            state.get_all_actions()
        } else {
            Vec::new()
        };
        // `priors` is `softmax × len` (mean ≈ 1, the PUCT convention);
        // dividing by the sum turns it back into a distribution.
        let raw = self.eval.priors(state, &actions);
        let total: f32 = raw.iter().sum();
        let mut priors: Vec<ActionPrior> = actions
            .iter()
            .zip(raw)
            .map(|(a, p)| ActionPrior {
                action: mirror::action_to_proto(*a),
                prob: if total > 0.0 { p / total } else { 0.0 },
            })
            .collect();
        priors.sort_by(|a, b| b.prob.total_cmp(&a.prob));
        NetReadout {
            model: self.name.clone(),
            value_home,
            mover: mirror::team_to_proto(mover),
            priors,
        }
    }
}

/// The bot playing one side of one game. Two per session, so the variants' size
/// gap is not worth a box.
#[allow(clippy::large_enum_variant)]
pub enum SessionBot {
    Random(RandomBot),
    Mcts {
        bot: Box<MctsBot>,
        net: Net,
        /// [`config_label`] of the bot's `MctsConfig`, for the report.
        config: String,
    },
}

impl SessionBot {
    pub fn get_action(&mut self, state: &GameState) -> EngineAction {
        match self {
            SessionBot::Random(b) => b.get_action(state),
            SessionBot::Mcts { bot, .. } => bot.get_action(state),
        }
    }

    pub fn is_mcts(&self) -> bool {
        matches!(self, SessionBot::Mcts { .. })
    }

    /// The net this bot searches with.
    pub fn net(&self) -> Option<&Net> {
        match self {
            SessionBot::Mcts { net, .. } => Some(net),
            SessionBot::Random(_) => None,
        }
    }

    /// Only the MCTS bot has a search to report on.
    pub fn last_search(&self) -> Option<&SearchSummary> {
        match self {
            SessionBot::Mcts { bot, .. } => bot.last_search(),
            SessionBot::Random(_) => None,
        }
    }

    /// The wire report of the bot's most recent search, every root child
    /// annotated with the net's own value of the position it leads to.
    pub fn report(&self, search_id: u64, pv_depth: usize) -> Option<ps::SearchReport> {
        let SessionBot::Mcts { bot, net, config } = self else {
            return None;
        };
        let summary = bot.last_search()?;
        let pv = bot.principal_variation(pv_depth);
        let sign = match summary.agent {
            EngineTeam::Home => 1.0,
            EngineTeam::Away => -1.0,
        };
        // Only a decision node is a position the value head was trained on:
        // a chance child is mid-roll, and a never-descended one has no state.
        let child_value = |action: &BbAction| -> Option<f32> {
            let node = bot.explore(std::slice::from_ref(action), true)?;
            if !matches!(node.stats.player, Some(BbPlayer::Home | BbPlayer::Away)) {
                return None;
            }
            let state = node.state?;
            Some(sign * net.eval.value_home_i64(&state) as f32 / Q_SCALE)
        };
        Some(report::summary_to_proto(
            search_id,
            summary,
            &pv,
            summary.root.solved,
            config,
            child_value,
        ))
    }

    pub fn explore(&self, path: &[BbAction], with_state: bool) -> Option<NodeView> {
        match self {
            SessionBot::Mcts { bot, .. } => bot.explore(path, with_state),
            SessionBot::Random(_) => None,
        }
    }

    pub fn seed(&mut self, rng: rand_chacha::ChaCha8Rng) {
        match self {
            SessionBot::Random(b) => b.set_seed(rng),
            // `MctsBot` has no RNG of its own — its nondeterminism comes from
            // worker interleaving and HashMap order, which a seed cannot pin.
            SessionBot::Mcts { .. } => {}
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
///
/// Walks subdirectories too (a few levels): the live nets sit in per-run
/// folders like `models/az_v7/`, and the top level is mostly older nets. A
/// model's `name` is its path relative to `dir`, so two `gen00`s in different
/// folders stay distinguishable.
pub fn list_models(dir: &Path) -> Vec<ModelInfo> {
    let mut models: Vec<(std::time::SystemTime, ModelInfo)> = Vec::new();
    collect_models(dir, dir, 3, &mut models);
    models.sort_by(|a, b| b.0.cmp(&a.0));
    models.into_iter().map(|(_, m)| m).collect()
}

fn collect_models(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(std::time::SystemTime, ModelInfo)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            if depth > 0 {
                collect_models(root, &path, depth - 1, out);
            }
            continue;
        }
        if path.extension().is_none_or(|x| x != "onnx") {
            continue;
        }
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let name = path
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| file_name.clone());
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        out.push((
            modified,
            ModelInfo {
                path: path.to_string_lossy().to_string(),
                board_tag: board_tag_of(&file_name),
                name,
            },
        ));
    }
}

/// One forward pass on an empty board of the game's size, so a net built for
/// another encoder (a different global-feature count, say) is refused with a
/// message instead of panicking the session on the bot's first move. tract
/// only builds its plan — and so only discovers the mismatch — on the first
/// forward for a board size, inside an `expect`.
fn probe(eval: &NnEvaluator, name: &str, board: BoardSpec) -> Result<(), String> {
    let (w, h, team_size) = board.engine_dims();
    let state = botbowl_engine::core::gamestate::GameStateBuilder::new()
        .with_board_dims(botbowl_engine::core::model::BoardDims::new(w, h, team_size))
        .set_state(botbowl_engine::core::gamestate::BuilderState::CoinToss)
        .build();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval.value_home_i64(&state))).map_err(|_| {
        format!(
            "{name} does not load on this build — it was probably exported for an older encoder \
             (the server's stderr has tract's reason). Pick another model."
        )
    })?;
    Ok(())
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
    cfg.fpu_reduction = spec.fpu_reduction.max(0.0);
    cfg.horizon_turns = spec.horizon_turns.max(1);
    cfg.horizon = spec.horizon;
    cfg
}

/// One line naming every knob that shapes the search, for the inspector —
/// so two bots in one game can be told apart by more than their net.
pub fn config_label(cfg: &MctsConfig) -> String {
    let horizon = if cfg.horizon {
        format!("horizon {} turn(s)", cfg.horizon_turns)
    } else {
        "no horizon".to_string()
    };
    format!(
        "{} · fpu {} · {horizon} · vl {} · reuse {} · {} worker(s)",
        cfg.puct.label(),
        cfg.fpu_reduction,
        cfg.virtual_loss,
        if cfg.tree_reuse { "on" } else { "off" },
        cfg.workers,
    )
}

/// Build the bot the lobby asked for.
///
/// `allowed_models` is the catalogue the lobby was shown; a model outside it
/// is rejected rather than opened, so a websocket message cannot name an
/// arbitrary path on the host.
pub fn build(
    spec: &BotSpec,
    board: BoardSpec,
    models: &ModelCache,
    allowed_models: &[ModelInfo],
) -> Result<SessionBot, String> {
    match spec {
        BotSpec::Random => Ok(SessionBot::Random(RandomBot::new())),
        BotSpec::Mcts(m) => {
            if m.model.is_empty() {
                return Err("the MCTS bot needs a model — this server found none for this board".into());
            }
            let budget = match m.budget {
                Budget::Iterations(n) => SearchBudget::Iterations(n.max(1)),
                Budget::Millis(ms) => SearchBudget::Time(Duration::from_millis(ms.max(1))),
            };
            let cfg = mcts_config(m);
            let config = config_label(&cfg);
            let (info, eval) = resolve_model(&m.model, models, allowed_models)?;
            probe(&eval, &info.name, board)?;
            let bot = MctsBot::with_budget_and_config(budget, cfg).with_evaluator(Arc::clone(&eval));
            Ok(SessionBot::Mcts {
                bot: Box::new(bot),
                net: Net {
                    name: info.name.clone(),
                    eval,
                },
                config,
            })
        }
    }
}

/// A net trained on another board size panics *inside* `NnEvaluator` rather
/// than erroring, so the `_WxH_` filename tag is checked before anything is
/// loaded.
pub fn model_tag_mismatch(spec: &BotSpec, board: BoardSpec, models: &[ModelInfo]) -> Option<String> {
    let requested = spec.model()?;
    let info = models.iter().find(|m| m.path == requested || m.name == requested)?;
    (!info.fits(board)).then(|| {
        format!(
            "{} was trained on a {} board, but this game is {} — pick a matching model",
            info.name,
            info.board_tag.as_deref().unwrap_or("?"),
            board.tag()
        )
    })
}

fn resolve_model<'a>(
    requested: &str,
    models: &ModelCache,
    allowed: &'a [ModelInfo],
) -> Result<(&'a ModelInfo, Arc<NnEvaluator>), String> {
    let hit = allowed
        .iter()
        .find(|m| m.path == requested || m.name == requested)
        .ok_or_else(|| format!("{requested:?} is not one of this server's models"))?;
    Ok((hit, models.get(Path::new(&hit.path))?))
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
            model: "/etc/passwd".into(),
            ..Default::default()
        });
        let err = match build(&spec, BoardSpec::new(14, 7, 4), &cache, &[]) {
            Err(e) => e,
            Ok(_) => panic!("an off-catalogue model must not be loaded"),
        };
        assert!(err.contains("not one of this server's models"), "{err}");
    }

    #[test]
    fn an_mcts_bot_without_a_model_is_refused() {
        let spec = BotSpec::Mcts(MctsSpec::default());
        let err = match build(&spec, BoardSpec::new(14, 7, 4), &ModelCache::default(), &[]) {
            Err(e) => e,
            Ok(_) => panic!("there is no heuristic fallback any more"),
        };
        assert!(err.contains("needs a model"), "{err}");
    }

    #[test]
    fn the_lobby_knobs_survive_the_trip_into_mcts_config() {
        let spec = MctsSpec {
            workers: Some(3),
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
        assert!(matches!(cfg.puct, PuctMode::NormalisedQ { c, range_floor } if c == 1.5 && range_floor == 20.0));
        assert_eq!(cfg.fpu_reduction, 0.25);
        assert_eq!(cfg.horizon_turns, 2);
        assert!(!cfg.horizon);
        assert!(!cfg.tree_reuse);
        assert_eq!(cfg.virtual_loss, 7);
        assert_eq!(cfg.tie_break, TieBreak::Mover);
        let label = config_label(&cfg);
        assert!(label.contains("no horizon") && label.contains("3 worker(s)"), "{label}");
    }

    #[test]
    fn a_tagged_model_is_refused_on_another_board() {
        let models = vec![ModelInfo {
            path: "m/bbnet_14x7_gen1.onnx".into(),
            name: "bbnet_14x7_gen1.onnx".into(),
            board_tag: Some("14x7".into()),
        }];
        let spec = BotSpec::Mcts(MctsSpec {
            model: "m/bbnet_14x7_gen1.onnx".into(),
            ..Default::default()
        });
        assert!(model_tag_mismatch(&spec, BoardSpec::new(14, 7, 4), &models).is_none());
        let err = model_tag_mismatch(&spec, BoardSpec::new(16, 9, 6), &models).expect("a mismatch");
        assert!(err.contains("14x7") && err.contains("16x9"), "{err}");
        assert!(model_tag_mismatch(&BotSpec::Random, BoardSpec::new(16, 9, 6), &models).is_none());
    }
}

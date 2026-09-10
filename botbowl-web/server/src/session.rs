//! One game, one websocket, one thread.
//!
//! The engine and the bots are synchronous and the MCTS bot spawns its own
//! `std::thread::scope` workers inside `get_action`, so none of this belongs
//! on an async runtime. The websocket handler therefore hands the socket's
//! traffic to a `spawn_blocking` thread that owns the `GameState` outright and
//! talks in channels — no locks, no interleaving, and a long search blocks
//! nothing but its own session.
//!
//! The session drives the engine in `DiceMode::RegisterRolls` (decision 4 of
//! plan 034) and rolls with its own `ChaCha8Rng`. That is what makes every
//! die visible to the UI — `RollDice` mode resolves rolls inside the engine
//! where a UI can never see them — and it is what makes "pin the next roll"
//! fall out for free.

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_engine::core::dices as ed;
use botbowl_engine::core::game_runner::Recording;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model as em;
use botbowl_engine::core::model::{Action as EngineAction, BoardDims, SomeProcInput};
use botbowl_web_proto::msg::{ClientMsg, GameSpec, LobbyInfo, ServerMsg, StartFrom};
use botbowl_web_proto::search as ps;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tokio::sync::mpsc;

use crate::bots::{self, SessionBot};
use crate::mirror;
use crate::report;
use crate::view::{self, DeriveCtx};
use crate::{dice, AppState};

/// How deep a principal variation to report.
const PV_DEPTH: usize = 8;
/// How many session log lines to keep and ship.
const LOG_TAIL: usize = 60;

/// One undo point: the board, the dice stream, and where the recording was.
/// The bot is deliberately *not* snapshotted — `MctsBot`'s cached tree keys on
/// a horizon anchor and discards itself on a mismatch, so an undo just costs a
/// wasted reuse, and `ScriptedBot`'s queued actions are re-validated for
/// legality before being played.
struct Snapshot {
    state: GameState,
    rng: ChaCha8Rng,
    steps: usize,
    log: usize,
}

pub struct GameSession {
    spec: GameSpec,
    state: GameState,
    rng: ChaCha8Rng,
    human: em::TeamType,
    bot: SessionBot,
    history: Vec<Snapshot>,
    /// A roll the debug control pinned for the next request.
    pinned_roll: Option<ed::RollResult>,
    seq: u64,
    /// Micro-step snapshots, so a web game opens in `botbowl-ui replay`.
    steps: Vec<GameState>,
    /// Player-facing event log. The engine's own logging is left off: it
    /// `println!`s a rules trace on every micro-step.
    log: Vec<String>,
    /// Identifies the bot's current search tree. Bumped on every search,
    /// because each one re-roots (or rebuilds) the single cached tree —
    /// an inspector opened on an earlier move cannot be walked any more.
    search_id: u64,
}

impl GameSession {
    fn new(spec: GameSpec, bot: SessionBot) -> Result<Self, String> {
        let (w, h, team_size) = spec.board.engine_dims();
        let dims = BoardDims::new(w, h, team_size);
        let mut state = match &spec.start {
            StartFrom::CoinToss => GameStateBuilder::new()
                .with_board_dims(dims)
                .set_state(BuilderState::CoinToss)
                .build(),
            StartFrom::Recording { path, step } => load_recording(path, *step)?,
        };
        // Off deliberately — see the `log` field.
        state.set_logging_state(false);
        state.set_dice_mode(DiceMode::RegisterRolls);

        let seed = spec.seed.unwrap_or_else(rand::random);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let mut bot = bot;
        // Give the stateless bots their own stream, derived from the session
        // seed so a seeded game is reproducible end to end.
        bot.seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB0B));

        let human = mirror::team_from_proto(spec.human);
        let steps = vec![state.clone()];
        Ok(GameSession {
            spec,
            state,
            rng,
            human,
            bot,
            history: Vec::new(),
            pinned_roll: None,
            seq: 0,
            steps,
            log: vec![format!("new game, seed {seed}")],
            search_id: 0,
        })
    }

    fn note(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > 4 * LOG_TAIL {
            self.log.drain(..self.log.len() - LOG_TAIL);
        }
    }

    fn log_tail(&self) -> Vec<String> {
        self.log[self.log.len().saturating_sub(LOG_TAIL)..].to_vec()
    }

    /// Who must supply the next action. `available_actions.team` is the
    /// authority (it is not always `team_turn` — an uphill block's dice are
    /// picked by the *defender*), with `team_turn` as the fallback the engine
    /// and `MctsBot` both use for mid-procedure states.
    fn actor(&self) -> em::TeamType {
        self.state.get_active_teamtype().unwrap_or(self.state.info.team_turn)
    }

    fn view(&mut self, out: &Out, bot_thinking: bool) {
        self.seq += 1;
        let ctx = DeriveCtx {
            human: self.human,
            seq: self.seq,
            can_undo: !self.history.is_empty(),
            bot_thinking,
            log_tail: self.log_tail(),
        };
        out.send(ServerMsg::View(Box::new(view::derive(&self.state, &ctx))));
    }

    /// Resolve one requested roll: a pinned value if it fits, else the
    /// session RNG.
    fn resolve(&mut self, requested: ed::RequestedRoll, out: &Out) -> (ed::RollResult, bool) {
        if let Some(pinned) = self.pinned_roll.take() {
            if requested.is_compatible(pinned) {
                return (pinned, true);
            }
            out.send(ServerMsg::Error(format!(
                "pinned roll {:?} does not fit the engine's request {:?} — rolling instead",
                mirror::roll_result_to_proto(pinned),
                mirror::requested_roll_to_proto(requested),
            )));
            out.send(ServerMsg::RollPinned(None));
        }
        (ed::resolve_with_rng(requested, &mut self.rng), false)
    }

    fn step(&mut self, input: SomeProcInput) {
        self.state.step_with_roll_or_action(input);
        self.steps.push(self.state.clone());
    }

    /// Drive the engine until the human has something to decide, or the game
    /// is over: rolling every die, and letting the bot answer its own prompts.
    fn advance(&mut self, out: &Out) {
        let mut before = (self.state.home.score, self.state.away.score);
        loop {
            if self.state.info.game_over {
                self.note(format!(
                    "game over: {} - {}",
                    self.state.home.score, self.state.away.score
                ));
                self.view(out, false);
                out.send(ServerMsg::GameOver {
                    winner: self.state.info.winner.map(mirror::team_to_proto),
                    home_score: self.state.home.score,
                    away_score: self.state.away.score,
                });
                return;
            }

            if let Some(requested) = self.state.pending_roll {
                let (result, fixed) = self.resolve(requested, out);
                let event = dice::event(requested, result, fixed);
                self.note(event.text.clone());
                out.send(ServerMsg::Dice(event));
                self.step(SomeProcInput::Roll(result));
                let after = (self.state.home.score, self.state.away.score);
                if after != before {
                    self.note(format!("TOUCHDOWN — {} - {}", after.0, after.1));
                    before = after;
                }
                continue;
            }

            if self.actor() == self.human {
                self.view(out, false);
                return;
            }

            // The bot's turn. Show the board with the spinner *before* the
            // search starts, or the human stares at a stale position for the
            // whole think time.
            self.view(out, true);
            let budget = self.spec.bot.label();
            out.send(ServerMsg::BotThinking {
                team: mirror::team_to_proto(self.actor()),
                budget,
            });

            let action = self.bot.get_action(&self.state);
            self.search_id += 1;
            let search_id = self.search_id;
            let report = self.bot.last_search().map(|summary| {
                let pv = self.bot.principal_variation(PV_DEPTH);
                Box::new(report::summary_to_proto(search_id, summary, &pv, summary.root.solved))
            });
            self.note(format!("bot: {:?}", action));
            out.send(ServerMsg::BotMoved {
                action: mirror::action_to_proto(action),
                report,
            });

            if !self.state.is_legal_action(&action) {
                // A bot returning an illegal action is a bug in the bot, but
                // stepping it would panic the session thread and take the
                // socket with it. Surface it instead.
                out.send(ServerMsg::Error(format!(
                    "bot proposed an illegal action {action:?} at {:?}",
                    self.state.proc_stack_top()
                )));
                self.view(out, false);
                return;
            }
            self.step(SomeProcInput::Action(action));
            let after = (self.state.home.score, self.state.away.score);
            if after != before {
                self.note(format!("TOUCHDOWN — {} - {}", after.0, after.1));
                before = after;
            }
        }
    }

    fn act(&mut self, action: EngineAction, out: &Out) {
        if self.state.info.game_over {
            out.send(ServerMsg::Error("the game is over".into()));
            return;
        }
        if self.state.pending_roll.is_some() {
            out.send(ServerMsg::Error(
                "the engine is waiting on a roll, not an action".into(),
            ));
            return;
        }
        if self.actor() != self.human {
            out.send(ServerMsg::Error("not your decision".into()));
            return;
        }
        if !self.state.is_legal_action(&action) {
            out.send(ServerMsg::Error(format!("{action:?} is not legal here")));
            self.view(out, false);
            return;
        }
        // Undo point: the state as the human was asked, before their answer.
        self.history.push(Snapshot {
            state: self.state.clone(),
            rng: self.rng.clone(),
            steps: self.steps.len(),
            log: self.log.len(),
        });
        self.note(format!("you: {action:?}"));
        self.step(SomeProcInput::Action(action));
        self.advance(out);
    }

    fn undo(&mut self, out: &Out) {
        match self.history.pop() {
            None => out.send(ServerMsg::Error("nothing to undo".into())),
            Some(snapshot) => {
                self.state = snapshot.state;
                self.rng = snapshot.rng;
                self.steps.truncate(snapshot.steps);
                self.log.truncate(snapshot.log);
                self.pinned_roll = None;
                self.note("undo".into());
                // Rewinding lands on a human decision point by construction,
                // so there is nothing to advance through — but going through
                // `advance` keeps the "who acts next" logic in one place.
                self.advance(out);
            }
        }
    }

    fn expand(&mut self, search_id: u64, path: Vec<ps::SearchEdge>, with_view: bool, out: &Out) {
        if search_id != self.search_id {
            return out.send(ServerMsg::Error(format!(
                "search {search_id} has been superseded by search {} — the bot keeps only its \
                 most recent tree, so earlier moves cannot be explored",
                self.search_id
            )));
        }
        let edges: Result<Vec<_>, String> = path.iter().map(report::edge_from_proto).collect();
        let edges = match edges {
            Ok(e) => e,
            Err(e) => return out.send(ServerMsg::Error(e)),
        };
        match self.bot.explore(&edges, with_view) {
            None => out.send(ServerMsg::Error(
                "that node is not in the cached search tree (it may have been re-rooted)".into(),
            )),
            Some(node) => {
                let view = node.state.as_ref().filter(|_| with_view).map(|state| {
                    let ctx = DeriveCtx {
                        human: self.human,
                        seq: 0,
                        can_undo: false,
                        bot_thinking: false,
                        log_tail: Vec::new(),
                    };
                    Box::new(view::derive(state, &ctx))
                });
                out.send(ServerMsg::Node(Box::new(report::node_to_proto(
                    search_id, path, &node, view,
                ))));
            }
        }
    }

    fn pin(&mut self, roll: Option<botbowl_web_proto::dice::RollResult>, out: &Out) {
        match roll {
            None => {
                self.pinned_roll = None;
                out.send(ServerMsg::RollPinned(None));
            }
            Some(wire) => match mirror::roll_result_from_proto(&wire) {
                Err(e) => out.send(ServerMsg::Error(e)),
                Ok(result) => {
                    self.pinned_roll = Some(result);
                    out.send(ServerMsg::RollPinned(Some(wire)));
                }
            },
        }
    }

    fn save(&self, name: &str, dir: &std::path::Path, out: &Out) {
        // Only a bare filename: this is a local POC, but a websocket message
        // still must not choose where on the host a file lands.
        if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
            return out.send(ServerMsg::Error("recording name must be a bare filename".into()));
        }
        if let Err(e) = std::fs::create_dir_all(dir) {
            return out.send(ServerMsg::Error(format!("could not create {}: {e}", dir.display())));
        }
        let path = dir.join(name);
        // `Recording::to_file` unwraps on IO errors, so check writability by
        // doing the serialisation ourselves would duplicate the format; catch
        // the panic instead and report it rather than killing the session.
        let recording = Recording::new(self.steps.clone());
        let display = path.to_string_lossy().to_string();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recording.to_file(&display))) {
            Ok(()) => out.send(ServerMsg::Saved { path: display }),
            Err(_) => out.send(ServerMsg::Error(format!("could not write {display}"))),
        }
    }
}

/// Resume a `botbowl-ui`-compatible recording at one micro-step.
fn load_recording(path: &str, step: usize) -> Result<GameState, String> {
    let recording = std::panic::catch_unwind(|| Recording::from_file(path))
        .map_err(|_| format!("could not read a recording from {path}"))?;
    let total = recording.total_steps();
    if step >= total {
        return Err(format!("step {step} is past the end of {path} ({total} steps)"));
    }
    let mut recording = recording;
    for _ in 0..step {
        botbowl_engine::core::game_runner::GameRunner::step(&mut recording);
    }
    Ok(botbowl_engine::core::game_runner::GameRunner::get_state(&recording).clone())
}

/// The outbound half of the socket. Sending is best-effort: a closed socket
/// just means the session is about to shut down.
pub struct Out(mpsc::Sender<ServerMsg>);

impl Out {
    pub fn send(&self, msg: ServerMsg) {
        let _ = self.0.blocking_send(msg);
    }
}

/// Own one socket's game for as long as the socket lives.
///
/// Runs on a blocking worker: every call in here (engine stepping, bot
/// searches) is synchronous and some of it takes seconds.
pub fn run(app: Arc<AppState>, mut input: mpsc::Receiver<ClientMsg>, output: mpsc::Sender<ServerMsg>) {
    let out = Out(output);
    let models = bots::list_models(&app.models_dir);
    out.send(ServerMsg::Lobby(Box::new(LobbyInfo {
        capacity: app.capacity,
        boards: app.board_presets(),
        models: models.clone(),
        defaults: GameSpec::default_for(app.capacity),
        server: app.server.clone(),
    })));

    let mut session: Option<GameSession> = None;

    while let Some(msg) = input.blocking_recv() {
        match msg {
            ClientMsg::NewGame(spec) => {
                if let Err(e) = spec.board.validate(app.capacity) {
                    out.send(ServerMsg::Error(e));
                    continue;
                }
                if let Some(model) = spec.bot.model_tag_mismatch(spec.board, &models) {
                    out.send(ServerMsg::Error(model));
                    continue;
                }
                match bots::build(&spec.bot, &app.model_cache, &models) {
                    Err(e) => out.send(ServerMsg::Error(e)),
                    Ok(bot) => match GameSession::new(spec, bot) {
                        Err(e) => out.send(ServerMsg::Error(e)),
                        Ok(mut new_session) => {
                            new_session.advance(&out);
                            session = Some(new_session);
                        }
                    },
                }
            }
            other => match session.as_mut() {
                None => out.send(ServerMsg::Error("no game yet — send NewGame first".into())),
                Some(s) => match other {
                    ClientMsg::NewGame(_) => unreachable!("handled above"),
                    ClientMsg::Act(action) => s.act(mirror::action_from_proto(action), &out),
                    ClientMsg::Undo => s.undo(&out),
                    ClientMsg::ExpandNode {
                        search_id,
                        path,
                        with_view,
                    } => s.expand(search_id, path, with_view, &out),
                    ClientMsg::FixNextRoll(roll) => s.pin(roll, &out),
                    ClientMsg::SaveRecording { path } => s.save(&path, &app.recordings_dir, &out),
                },
            },
        }
    }
}

/// Extra validation on a `BotSpec` that needs to know the board.
trait ModelTagCheck {
    fn model_tag_mismatch(
        &self,
        board: botbowl_web_proto::msg::BoardSpec,
        models: &[botbowl_web_proto::msg::ModelInfo],
    ) -> Option<String>;
}

impl ModelTagCheck for botbowl_web_proto::msg::BotSpec {
    /// A net trained on another board size panics *inside* `NnEvaluator`
    /// rather than erroring, so the `_WxH_` filename tag is checked here
    /// before anything is loaded.
    fn model_tag_mismatch(
        &self,
        board: botbowl_web_proto::msg::BoardSpec,
        models: &[botbowl_web_proto::msg::ModelInfo],
    ) -> Option<String> {
        let requested = match self {
            botbowl_web_proto::msg::BotSpec::Mcts(m) => m.evaluator.model()?,
            _ => return None,
        };
        let info = models.iter().find(|m| m.path == requested || m.name == requested)?;
        let tag = info.board_tag.as_deref()?;
        (tag != board.tag()).then(|| {
            format!(
                "{} was trained on a {tag} board, but this game is {} — pick a matching model",
                info.name,
                board.tag()
            )
        })
    }
}

/// Where recordings go by default.
pub fn default_recordings_dir() -> PathBuf {
    PathBuf::from("data/web-games")
}

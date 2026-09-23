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
use std::time::{Duration, Instant};

use botbowl_engine::core::dices as ed;
use botbowl_engine::core::game_runner::Recording;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model as em;
use botbowl_engine::core::model::{Action as EngineAction, BoardDims, SomeProcInput};
use botbowl_web_proto::msg::{ClientMsg, GameSpec, LobbyInfo, ServerMsg, StartFrom, StepMode};
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
/// How long the run loop sleeps between polls while it is waiting out an
/// `Auto` step delay. Short enough that a mode change or an undo does not feel
/// stuck behind the delay, long enough not to spin.
const POLL: Duration = Duration::from_millis(5);

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
    /// How fast the session may run through steps the human does not answer.
    step_mode: StepMode,
    /// True while `advance` has stopped *before* a step it could take. The
    /// board the client last saw is the result of the previous step, and the
    /// session is back in the run loop, so undo, mode changes and tree
    /// inspection all still work while it holds.
    paused: bool,
    /// When an `Auto` hold expires. `None` under `Manual`, which waits for
    /// [`ClientMsg::StepOnce`] instead of a clock.
    resume_at: Option<Instant>,
}

/// Whether the session may take the step it is standing in front of.
enum Hold {
    /// Take it now.
    Go,
    /// Stop and hand control back to the run loop; resume at this instant, or
    /// on an explicit `StepOnce` when there is none.
    Wait(Option<Instant>),
}

impl GameSession {
    fn new(spec: GameSpec, bot: SessionBot, step_mode: StepMode) -> Result<Self, String> {
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
            step_mode,
            paused: false,
            resume_at: None,
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
            step_mode: self.step_mode,
            paused: self.paused,
        };
        out.send(ServerMsg::View(Box::new(view::derive(&self.state, &ctx))));
        // Plan 043: one forward pass per board change, so the debug drawer answers "who does the
        // net think scores next" during the human's turn too — `SearchReport.evaluator_value`
        // only ever appears after a *bot* move. Negligible next to a search, and it cannot
        // perturb the game: a frozen net is a pure function of the state.
        if let Some(value_home) = self.bot.evaluate_home(&self.state) {
            out.send(ServerMsg::Valuation { value_home });
        }
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

    /// Whether the step the session is standing in front of may be taken now.
    ///
    /// Asked *before* the step rather than after it, so that the board the
    /// client sees while the session holds is the finished result of the
    /// previous step — never a half-applied one — and so the hold never
    /// stands between the human and their own next decision.
    fn hold(&self) -> Hold {
        match self.step_mode {
            StepMode::Run => Hold::Go,
            StepMode::Manual => Hold::Wait(None),
            StepMode::Auto { ms } => Hold::Wait(Some(Instant::now() + Duration::from_millis(ms))),
        }
    }

    /// Drive the engine until the human has something to decide, the game is
    /// over, or the step mode says to hold: rolling every die, and letting the
    /// bot answer its own prompts.
    fn advance(&mut self, out: &Out) {
        let mut before = (self.state.home.score, self.state.away.score);
        loop {
            if self.state.info.game_over {
                self.paused = false;
                self.resume_at = None;
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

            if self.state.pending_roll.is_none() && self.actor() == self.human {
                self.paused = false;
                self.resume_at = None;
                self.view(out, false);
                return;
            }

            // From here there is a step to take that the human does not
            // answer — a die or a bot move — so this is where a hold belongs.
            if let Hold::Wait(resume_at) = self.hold() {
                self.paused = true;
                self.resume_at = resume_at;
                self.view(out, false);
                return;
            }

            if !self.take_one(out, &mut before) {
                return;
            }
        }
    }

    /// One engine step: resolve the pending roll, or let the bot move.
    /// Returns `false` when the session must stop rather than loop on (a bot
    /// that proposed an illegal action).
    fn take_one(&mut self, out: &Out, before: &mut (u8, u8)) -> bool {
        if let Some(requested) = self.state.pending_roll {
            let (result, fixed) = self.resolve(requested, out);
            let event = dice::event(requested, result, fixed);
            self.note(event.text.clone());
            out.send(ServerMsg::Dice(event));
            self.step(SomeProcInput::Roll(result));
            self.note_score(before);
            return true;
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
            self.paused = false;
            self.resume_at = None;
            self.view(out, false);
            return false;
        }
        self.step(SomeProcInput::Action(action));
        self.note_score(before);
        true
    }

    fn note_score(&mut self, before: &mut (u8, u8)) {
        let after = (self.state.home.score, self.state.away.score);
        if after != *before {
            self.note(format!("TOUCHDOWN — {} - {}", after.0, after.1));
            *before = after;
        }
    }

    /// When the run loop should come back and step the session on its own.
    /// `None` means it may block on the socket: either nothing is held, or
    /// the hold is waiting for an explicit `StepOnce`.
    fn wake_at(&self) -> Option<Instant> {
        self.paused.then_some(self.resume_at).flatten()
    }

    /// Take the held step, then carry on under the current mode.
    fn step_once(&mut self, out: &Out) {
        if !self.paused {
            // Not an error: the human can hit Step on a board that is already
            // waiting for *them*, and a queued Step can arrive after a mode
            // change released the hold.
            return;
        }
        self.paused = false;
        self.resume_at = None;
        let mut before = (self.state.home.score, self.state.away.score);
        if self.take_one(out, &mut before) {
            self.advance(out);
        }
    }

    /// Re-pace the session. Switching to `Run` while it holds releases it;
    /// switching between the holding modes just re-arms the hold, without
    /// taking a step, so changing the speed never skips anything.
    fn set_step_mode(&mut self, mode: StepMode, out: &Out) {
        self.step_mode = mode;
        self.note(format!("step mode: {}", mode.label()));
        if self.paused {
            self.paused = false;
            self.resume_at = None;
            self.advance(out);
        } else {
            self.view(out, false);
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
                    // A node's board is hypothetical: it belongs to no
                    // session moment, so none of the session flags apply.
                    let ctx = DeriveCtx {
                        human: self.human,
                        ..DeriveCtx::default()
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
    // The pacing outlives any one game: the client sets it on the game screen
    // and expects "New game" to keep it, and a `SetStepMode` that arrives
    // before the first `NewGame` must not be dropped on the floor.
    let mut step_mode = StepMode::default();

    loop {
        let msg = match session.as_ref().and_then(GameSession::wake_at) {
            // An `Auto` hold is running: come back when it expires, but stay
            // responsive to anything the client sends in the meantime.
            Some(deadline) => match wait_until(&mut input, deadline) {
                Wait::Msg(msg) => msg,
                Wait::Closed => break,
                Wait::Elapsed => {
                    session.as_mut().expect("a session, to have a deadline").step_once(&out);
                    continue;
                }
            },
            None => match input.blocking_recv() {
                Some(msg) => msg,
                None => break,
            },
        };

        match msg {
            ClientMsg::SetStepMode(mode) => {
                step_mode = mode;
                if let Some(s) = session.as_mut() {
                    s.set_step_mode(mode, &out);
                }
            }
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
                    Ok(bot) => match GameSession::new(spec, bot, step_mode) {
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
                    ClientMsg::NewGame(_) | ClientMsg::SetStepMode(_) => unreachable!("handled above"),
                    ClientMsg::Act(action) => s.act(mirror::action_from_proto(action), &out),
                    ClientMsg::Undo => s.undo(&out),
                    ClientMsg::StepOnce => s.step_once(&out),
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

enum Wait {
    Msg(ClientMsg),
    /// The deadline passed with nothing to read.
    Elapsed,
    /// The socket went away.
    Closed,
}

/// Block for a client message, but no later than `deadline`.
///
/// Polling rather than `tokio::time::timeout`: this runs on a `spawn_blocking`
/// thread that must not touch the async runtime, and a 5 ms poll is far below
/// the shortest step delay the UI offers.
fn wait_until(input: &mut mpsc::Receiver<ClientMsg>, deadline: Instant) -> Wait {
    loop {
        match input.try_recv() {
            Ok(msg) => return Wait::Msg(msg),
            Err(mpsc::error::TryRecvError::Disconnected) => return Wait::Closed,
            Err(mpsc::error::TryRecvError::Empty) => {
                let now = Instant::now();
                if now >= deadline {
                    return Wait::Elapsed;
                }
                std::thread::sleep(POLL.min(deadline - now));
            }
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

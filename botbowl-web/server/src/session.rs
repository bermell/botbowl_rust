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
//!
//! Either seat may be a human or a bot. Every decision either side takes is
//! logged as a [`DecisionRecord`] and streamed to the client — the net's read
//! of the position always, and the whole search root for an MCTS move.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use botbowl_engine::core::dices as ed;
use botbowl_engine::core::game_runner::Recording;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model as em;
use botbowl_engine::core::model::{Action as EngineAction, BoardDims, SomeProcInput};
use botbowl_play::drives::{self, DriveStart};
use botbowl_play::generate::RandomStartBias;
use botbowl_web_proto::decision::{Decider, DecisionRecord, NetReadout};
use botbowl_web_proto::msg::{BotSpec, ClientMsg, GameSpec, LobbyInfo, Seat, ServerMsg, StartFrom, StepMode};
use botbowl_web_proto::search as ps;
use botbowl_web_proto::team::TeamDef;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tokio::sync::mpsc;

use crate::bots::{self, Net, SessionBot};
use crate::mirror;
use crate::report;
use crate::teams::{self, Looks};
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
/// Under `Run`, the session yields to the run loop after every bot move so the
/// socket stays responsive (a bot-vs-bot game would otherwise play to the end
/// inside one call). A yield only re-sends the board when this long has passed
/// since the last one — two random bots make thousands of moves a second.
const RUN_VIEW_EVERY: Duration = Duration::from_millis(100);

/// One undo point: the board, the dice stream, and where the recording and the
/// decision log were. The bots are deliberately *not* snapshotted — `MctsBot`'s
/// cached tree keys on a horizon anchor and discards itself on a mismatch, so
/// an undo just costs a wasted reuse.
struct Snapshot {
    state: GameState,
    rng: ChaCha8Rng,
    steps: usize,
    log: usize,
    decisions: u64,
}

/// `[home, away]` indexing.
fn side(team: em::TeamType) -> usize {
    match team {
        em::TeamType::Home => 0,
        em::TeamType::Away => 1,
    }
}

pub struct GameSession {
    spec: GameSpec,
    state: GameState,
    rng: ChaCha8Rng,
    /// `[home, away]`; `None` is a seat played from the browser.
    bots: [Option<SessionBot>; 2],
    history: Vec<Snapshot>,
    /// A roll the debug control pinned for the next request.
    pinned_roll: Option<ed::RollResult>,
    seq: u64,
    /// Micro-step snapshots, so a web game opens in `botbowl-ui replay`.
    steps: Vec<GameState>,
    /// Player-facing event log. The engine's own logging is left off: it
    /// `println!`s a rules trace on every micro-step.
    log: Vec<String>,
    /// Bumped on every search by either bot, so a search id names one search.
    search_id: u64,
    /// `[home, away]`: each bot's latest search id. A bot keeps exactly one
    /// tree — its most recent search's — so these are the only two searches
    /// that can still be walked below the root.
    latest_search: [u64; 2],
    /// Decisions logged so far; the next record's index.
    decisions: u64,
    /// For each logged decision, the index into `steps` of the state it was
    /// taken in — what [`ClientMsg::ShowDecision`] renders.
    decision_steps: Vec<usize>,
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
    /// `Run` handed control back to the run loop between two bot moves and
    /// wants to carry on as soon as the socket has been drained. Not a hold:
    /// the client is not told, and nothing waits.
    yielded: bool,
    /// When a board was last sent, for [`RUN_VIEW_EVERY`].
    last_view: Option<Instant>,
    /// A random-start drive: where it started and who attacks. The session stops when
    /// [`DriveStart::over`] says so, exactly where the corpus's trajectories stop.
    drive: Option<(DriveStart, em::TeamType)>,
    /// Each side's team, for the players' pictures.
    looks: Arc<Looks>,
}

/// Whether the session may take the step it is standing in front of.
enum Hold {
    /// Take it now.
    Go,
    /// Stop and hand control back to the run loop; resume at this instant, or
    /// on an explicit `StepOnce` when there is none.
    Wait(Option<Instant>),
}

/// What [`GameSession::take_one`] did.
enum Took {
    Roll,
    Move,
    /// The session must stop rather than loop on (a bot proposed an illegal
    /// action).
    Stop,
}

impl GameSession {
    fn new(
        spec: GameSpec,
        bots: [Option<SessionBot>; 2],
        step_mode: StepMode,
        sides: [TeamDef; 2],
    ) -> Result<Self, String> {
        let (w, h, team_size) = spec.board.engine_dims();
        let dims = BoardDims::new(w, h, team_size);
        let mut start_note = None;
        let mut state = match &spec.start {
            StartFrom::CoinToss => {
                let mut state = GameStateBuilder::new()
                    .with_board_dims(dims)
                    .set_state(BuilderState::CoinToss)
                    .build();
                teams::apply(&mut state, em::TeamType::Home, &sides[0])?;
                teams::apply(&mut state, em::TeamType::Away, &sides[1])?;
                state
            }
            StartFrom::Recording { path, step } => load_recording(path, *step)?,
            // The generated players stay as drawn — that is the training distribution — and the
            // teams only lend their pictures.
            StartFrom::RandomDrive { seed } => {
                let seed = seed.unwrap_or_else(rand::random);
                start_note = Some(format!("random-start drive, position seed {seed}"));
                drives::position_state(&RandomStartBias::default(), dims, seed)
            }
        };
        let drive = spec
            .start
            .is_drive()
            .then(|| (DriveStart::of(&state), drives::attacker_of(&state)));
        // Off deliberately — see the `log` field.
        state.set_logging_state(false);
        state.set_dice_mode(DiceMode::RegisterRolls);

        let seed = spec.seed.unwrap_or_else(rand::random);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let mut bots = bots;
        // Give the stateless bots their own streams, derived from the session
        // seed so a seeded game is reproducible end to end.
        for (i, bot) in bots.iter_mut().enumerate() {
            if let Some(bot) = bot {
                bot.seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB0B ^ i as u64));
            }
        }

        let steps = vec![state.clone()];
        let mut log = vec![
            format!("new game, seed {seed}"),
            format!(
                "home: {} ({}) · away: {} ({})",
                spec.home.label(),
                sides[0].name,
                spec.away.label(),
                sides[1].name
            ),
        ];
        log.extend(start_note);
        if let Some((_, attacker)) = drive {
            log.push(format!("{attacker:?} attacks"));
        }
        let [home_def, away_def] = sides;
        let looks = Arc::new(Looks {
            teams: [Some(home_def), Some(away_def)],
        });
        Ok(GameSession {
            spec,
            state,
            rng,
            bots,
            history: Vec::new(),
            pinned_roll: None,
            seq: 0,
            steps,
            log,
            search_id: 0,
            latest_search: [0; 2],
            decisions: 0,
            decision_steps: Vec::new(),
            step_mode,
            paused: false,
            resume_at: None,
            yielded: false,
            last_view: None,
            drive,
            looks,
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

    fn is_human(&self, team: em::TeamType) -> bool {
        self.bots[side(team)].is_none()
    }

    /// The net that reads out a decision by `team`: that side's own net when
    /// it is an MCTS bot, otherwise the first net seated in the game (Home's
    /// before Away's). `None` when neither seat has one.
    fn net_for(&self, team: em::TeamType) -> Option<&Net> {
        self.bots[side(team)]
            .as_ref()
            .and_then(SessionBot::net)
            .or_else(|| self.bots.iter().flatten().find_map(SessionBot::net))
    }

    fn readout(&self, team: em::TeamType) -> Option<NetReadout> {
        self.net_for(team).map(|net| net.readout(&self.state, team))
    }

    fn view(&mut self, out: &Out, bot_thinking: bool) {
        self.seq += 1;
        self.last_view = Some(Instant::now());
        let ctx = DeriveCtx {
            humans: self.spec.humans(),
            seq: self.seq,
            can_undo: !self.history.is_empty(),
            bot_thinking,
            log_tail: self.log_tail(),
            step_mode: self.step_mode,
            paused: self.paused,
            looks: self.looks.clone(),
        };
        out.send(ServerMsg::View(Box::new(view::derive(&self.state, &ctx))));
        // Plan 043: one forward pass per board change, so the debug drawer answers "who does the
        // net think scores next, and what would it play" during the human's turn too — a search
        // report only ever appears after a *bot* move. Negligible next to a search, and it cannot
        // perturb the game: a frozen net is a pure function of the state.
        if let Some(readout) = self.readout(self.actor()) {
            out.send(ServerMsg::Net(Box::new(readout)));
        }
    }

    /// Log one decision about to be applied to the current state, and send it.
    fn record(
        &mut self,
        team: em::TeamType,
        action: EngineAction,
        net: Option<NetReadout>,
        search: Option<ps::SearchReport>,
        out: &Out,
    ) {
        let by = match &self.bots[side(team)] {
            None => Decider::Human,
            Some(_) => Decider::Bot {
                label: self.spec.seat(mirror::team_to_proto(team)).label(),
            },
        };
        let record = DecisionRecord {
            index: self.decisions,
            team: mirror::team_to_proto(team),
            by,
            action: mirror::action_to_proto(action),
            proc: self.state.proc_stack_top().unwrap_or("-").to_string(),
            half: self.state.info.half,
            turn: match team {
                em::TeamType::Home => self.state.info.home_turn,
                em::TeamType::Away => self.state.info.away_turn,
            },
            n_legal: self.state.get_all_actions().len(),
            net,
            search: search.map(Box::new),
        };
        self.decisions += 1;
        // `steps` always ends with the current state.
        self.decision_steps.push(self.steps.len() - 1);
        out.send(ServerMsg::Decision(Box::new(record)));
    }

    fn show_decision(&self, index: u64, out: &Out) {
        let Some(state) = self
            .decision_steps
            .get(index as usize)
            .and_then(|&step| self.steps.get(step))
        else {
            return out.send(ServerMsg::Error(format!("no decision {index} in this game")));
        };
        // A past board is hypothetical: none of the session flags apply.
        let ctx = DeriveCtx {
            humans: self.spec.humans(),
            looks: self.looks.clone(),
            ..DeriveCtx::default()
        };
        out.send(ServerMsg::DecisionBoard {
            index,
            view: Box::new(view::derive(state, &ctx)),
        });
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

    /// Drive the engine until a human has something to decide, the game is
    /// over, or the step mode says to hold: rolling every die, and letting the
    /// bots answer their own prompts.
    fn advance(&mut self, out: &Out) {
        self.yielded = false;
        let mut moved = false;
        let mut before = (self.state.home.score, self.state.away.score);
        loop {
            // Before the game-over check: a drive that runs out the second half is a drive that
            // ended, and the corpus scores it as one.
            if let Some((start, attacker)) = self.drive {
                if start.over(&self.state) {
                    self.paused = false;
                    self.resume_at = None;
                    let (home, away) = start.scored(&self.state);
                    let scored = match (home > 0, away > 0) {
                        (true, _) => Some(em::TeamType::Home),
                        (_, true) => Some(em::TeamType::Away),
                        _ => None,
                    };
                    self.note(match scored {
                        Some(t) => format!("drive over: {t:?} scored"),
                        None => "drive over: no score".into(),
                    });
                    self.view(out, false);
                    out.send(ServerMsg::DriveOver {
                        attacker: mirror::team_to_proto(attacker),
                        scored: scored.map(mirror::team_to_proto),
                        home_score: self.state.home.score,
                        away_score: self.state.away.score,
                    });
                    return;
                }
            }

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

            if self.state.pending_roll.is_none() && self.is_human(self.actor()) {
                self.paused = false;
                self.resume_at = None;
                self.view(out, false);
                return;
            }

            // From here there is a step to take that no human answers — a die
            // or a bot move — so this is where a hold belongs.
            if let Hold::Wait(resume_at) = self.hold() {
                self.paused = true;
                self.resume_at = resume_at;
                self.view(out, false);
                return;
            }

            // `Run` lets the run loop drain the socket between two bot moves,
            // so "Step" or "New game" land mid-reply — and mid-game, when two
            // bots play. In front of the *next* move rather than behind the
            // last, so a yield never lands on a human's decision.
            if moved && self.state.pending_roll.is_none() {
                self.yielded = true;
                if self.last_view.is_none_or(|t| t.elapsed() >= RUN_VIEW_EVERY) {
                    self.view(out, false);
                }
                return;
            }

            match self.take_one(out, &mut before) {
                Took::Stop => return,
                Took::Roll => {}
                Took::Move => moved = true,
            }
        }
    }

    /// One engine step: resolve the pending roll, or let the side to act's
    /// bot move.
    fn take_one(&mut self, out: &Out, before: &mut (u8, u8)) -> Took {
        if let Some(requested) = self.state.pending_roll {
            let (result, fixed) = self.resolve(requested, out);
            let event = dice::event(requested, result, fixed);
            self.note(event.text.clone());
            out.send(ServerMsg::Dice(event));
            self.step(SomeProcInput::Roll(result));
            self.note_score(before);
            return Took::Roll;
        }

        let team = self.actor();
        let i = side(team);
        let searches = self.bots[i].as_ref().is_some_and(SessionBot::is_mcts);
        if searches {
            // Show the board with the spinner *before* the search starts, or
            // the human stares at a stale position for the whole think time.
            self.view(out, true);
            out.send(ServerMsg::BotThinking {
                team: mirror::team_to_proto(team),
                budget: self.spec.seat(mirror::team_to_proto(team)).label(),
            });
        }
        // Read the net out *before* the move: it is about this position.
        let net = self.readout(team);

        let bot = self.bots[i].as_mut().expect("advance only steps a bot's decision");
        let action = bot.get_action(&self.state);
        let search = if searches {
            self.search_id += 1;
            self.latest_search[i] = self.search_id;
            self.bots[i].as_ref().and_then(|b| b.report(self.search_id, PV_DEPTH))
        } else {
            None
        };
        self.note(format!("{team:?} bot: {}", mirror::action_to_proto(action).describe()));
        self.record(team, action, net, search, out);

        if !self.state.is_legal_action(&action) {
            // A bot returning an illegal action is a bug in the bot, but
            // stepping it would panic the session thread and take the
            // socket with it. Surface it instead.
            out.send(ServerMsg::Error(format!(
                "{team:?} bot proposed an illegal action {action:?} at {:?}",
                self.state.proc_stack_top()
            )));
            self.paused = false;
            self.resume_at = None;
            self.view(out, false);
            return Took::Stop;
        }
        self.step(SomeProcInput::Action(action));
        self.note_score(before);
        Took::Move
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
        if self.yielded {
            return Some(Instant::now());
        }
        self.paused.then_some(self.resume_at).flatten()
    }

    /// The run loop's wake-up: carry on after a yield, or take the held step.
    fn resume(&mut self, out: &Out) {
        if self.yielded {
            self.advance(out);
        } else {
            self.step_once(out);
        }
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
        match self.take_one(out, &mut before) {
            Took::Stop => {}
            Took::Roll | Took::Move => self.advance(out),
        }
    }

    /// Re-pace the session. Switching to `Run` while it holds releases it;
    /// switching between the holding modes just re-arms the hold, without
    /// taking a step, so changing the speed never skips anything.
    fn set_step_mode(&mut self, mode: StepMode, out: &Out) {
        self.step_mode = mode;
        self.note(format!("step mode: {}", mode.label()));
        if self.paused || self.yielded {
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
        if self.drive.is_some_and(|(start, _)| start.over(&self.state)) {
            out.send(ServerMsg::Error("the drive is over — start the next one".into()));
            return;
        }
        if self.state.pending_roll.is_some() {
            out.send(ServerMsg::Error(
                "the engine is waiting on a roll, not an action".into(),
            ));
            return;
        }
        let team = self.actor();
        if !self.is_human(team) {
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
            decisions: self.decisions,
        });
        self.note(format!(
            "{team:?} (you): {}",
            mirror::action_to_proto(action).describe()
        ));
        let net = self.readout(team);
        self.record(team, action, net, None, out);
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
                self.decisions = snapshot.decisions;
                self.decision_steps.truncate(snapshot.decisions as usize);
                out.send(ServerMsg::DecisionsTruncated { keep: self.decisions });
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
        let Some(i) = (0..2).find(|&i| search_id != 0 && self.latest_search[i] == search_id) else {
            return out.send(ServerMsg::Error(format!(
                "search {search_id} has been superseded by that bot's search {} — each bot keeps \
                 only its most recent tree, so earlier moves can be read but not walked",
                self.latest_search.iter().max().copied().unwrap_or(0)
            )));
        };
        let edges: Result<Vec<_>, String> = path.iter().map(report::edge_from_proto).collect();
        let edges = match edges {
            Ok(e) => e,
            Err(e) => return out.send(ServerMsg::Error(e)),
        };
        let Some(bot) = self.bots[i].as_ref() else {
            return out.send(ServerMsg::Error("that seat has no bot".into()));
        };
        match bot.explore(&edges, with_view) {
            None => out.send(ServerMsg::Error(
                "that node is not in the cached search tree (it may have been re-rooted)".into(),
            )),
            Some(node) => {
                let view = node.state.as_ref().filter(|_| with_view).map(|state| {
                    // A node's board is hypothetical: it belongs to no
                    // session moment, so none of the session flags apply.
                    let ctx = DeriveCtx {
                        humans: self.spec.humans(),
                        looks: self.looks.clone(),
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

/// Build both seats' bots, or say why not.
fn build_seats(
    spec: &GameSpec,
    app: &AppState,
    models: &[botbowl_web_proto::msg::ModelInfo],
) -> Result<[Option<SessionBot>; 2], String> {
    let seat = |seat: &Seat| -> Result<Option<SessionBot>, String> {
        let Some(bot) = seat.bot() else { return Ok(None) };
        if let Some(e) = bots::model_tag_mismatch(bot, spec.board, models) {
            return Err(e);
        }
        bots::build(bot, spec.board, &app.model_cache, models).map(Some)
    };
    Ok([seat(&spec.home)?, seat(&spec.away)?])
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
    let store = &app.opts.teams;
    out.send(ServerMsg::Lobby(Box::new(LobbyInfo {
        capacity: app.capacity,
        boards: app.board_presets(),
        models: models.clone(),
        defaults: GameSpec::default_for(app.capacity, &models),
        server: app.server.clone(),
        teams: store.list(),
        skills: teams::skills(),
        pictures: store.pictures(app.opts.assets_dir.as_deref()),
        can_save_teams: store.dir.is_some(),
        step_mode: app.opts.initial_step_mode,
        can_resume: app.opts.allow_recording_paths,
    })));

    let mut session: Option<GameSession> = None;
    // The pacing outlives any one game: the client sets it on the game screen
    // and expects "New game" to keep it, and a `SetStepMode` that arrives
    // before the first `NewGame` must not be dropped on the floor.
    let mut step_mode = app.opts.initial_step_mode;

    loop {
        let msg = match session.as_ref().and_then(GameSession::wake_at) {
            // A hold or a yield is running: come back when it expires, but
            // stay responsive to anything the client sends in the meantime.
            Some(deadline) => match wait_until(&mut input, deadline) {
                Wait::Msg(msg) => msg,
                Wait::Closed => break,
                Wait::Elapsed => {
                    session.as_mut().expect("a session, to have a deadline").resume(&out);
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
            ClientMsg::SaveTeam(def) => match store.save(def) {
                Ok(()) => out.send(ServerMsg::Teams(store.list())),
                Err(e) => out.send(ServerMsg::Error(e)),
            },
            ClientMsg::DeleteTeam { name } => match store.delete(&name) {
                Ok(()) => out.send(ServerMsg::Teams(store.list())),
                Err(e) => out.send(ServerMsg::Error(e)),
            },
            ClientMsg::UploadPicture { data_url } => match store.save_picture(&data_url) {
                Ok(picture) => out.send(ServerMsg::PictureSaved {
                    picture,
                    pictures: store.pictures(app.opts.assets_dir.as_deref()),
                }),
                Err(e) => out.send(ServerMsg::Error(e)),
            },
            ClientMsg::NewGame(mut spec) => {
                if let Err(e) = spec.board.validate(app.capacity) {
                    out.send(ServerMsg::Error(e));
                    continue;
                }
                if matches!(spec.start, StartFrom::Recording { .. }) && !app.opts.allow_recording_paths {
                    out.send(ServerMsg::Error(
                        "this server does not open recordings by path (it listens on the network)".into(),
                    ));
                    continue;
                }
                cap_workers(&mut spec, app.opts.max_workers);
                let sides = match (store.find(&spec.home_team), store.find(&spec.away_team)) {
                    (Some(h), Some(a)) => [h, a],
                    (h, _) => {
                        let missing = if h.is_none() { &spec.home_team } else { &spec.away_team };
                        out.send(ServerMsg::Error(format!("no team called {missing:?}")));
                        continue;
                    }
                };
                match build_seats(&spec, &app, &models) {
                    Err(e) => out.send(ServerMsg::Error(e)),
                    Ok(bots) => match GameSession::new(spec, bots, step_mode, sides) {
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
                    ClientMsg::NewGame(_)
                    | ClientMsg::SetStepMode(_)
                    | ClientMsg::SaveTeam(_)
                    | ClientMsg::DeleteTeam { .. }
                    | ClientMsg::UploadPicture { .. } => unreachable!("handled above"),
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
                    ClientMsg::ShowDecision { index } => s.show_decision(index, &out),
                },
            },
        }
    }
}

/// Hold every MCTS seat to the server's thread cap. `workers: None` ("all cores") becomes the
/// cap itself.
fn cap_workers(spec: &mut GameSpec, cap: Option<usize>) {
    let Some(cap) = cap else { return };
    for seat in [&mut spec.home, &mut spec.away] {
        if let Seat::Bot(BotSpec::Mcts(m)) = seat {
            m.workers = Some(m.workers.unwrap_or(cap).clamp(1, cap.max(1)));
        }
    }
}

/// One per received message, like `ClientMsg` itself — not worth a box.
#[allow(clippy::large_enum_variant)]
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

/// Where recordings go by default.
pub fn default_recordings_dir() -> PathBuf {
    PathBuf::from("data/web-games")
}

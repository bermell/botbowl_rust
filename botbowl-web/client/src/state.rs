//! The client's whole state: a handful of signals, all of them fed by the
//! server.
//!
//! The client is a renderer (decision 3 of plan 034) — it owns no game logic,
//! so there is nothing here but "the last thing the server said" plus local
//! view preferences (which overlay is on, which square's menu is open).

use botbowl_web_proto::decision::{DecisionRecord, NetReadout};
use botbowl_web_proto::dice::RollResult;
use botbowl_web_proto::log::LogEntry;
use botbowl_web_proto::msg::{GameSpec, LobbyInfo, StepMode};
use botbowl_web_proto::search::{NodeExpansion, SearchReport};
use botbowl_web_proto::team::TeamDef;
use botbowl_web_proto::view::ViewState;
use botbowl_web_proto::{Action, Position, TeamType};
use leptos::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Connection {
    Connecting,
    Open,
    Closed,
}

/// Which per-square overlay is painted. One at a time: they all colour the
/// same squares and stacking them is unreadable.
/// What the page shows when no game is running.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Screen {
    #[default]
    Lobby,
    Teams,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Overlay {
    /// Legal moves only.
    #[default]
    Moves,
    /// Path success probability, green → red.
    Risk,
    /// Where the inspected search spent its visits.
    BotVisits,
    /// What the net's policy wanted: for the position on screen while
    /// following the game live, or for the inspected decision.
    NetPriors,
    /// Nothing — just the board.
    None,
}

impl Overlay {
    pub const ALL: [Overlay; 5] = [
        Overlay::Moves,
        Overlay::Risk,
        Overlay::BotVisits,
        Overlay::NetPriors,
        Overlay::None,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Overlay::Moves => "Moves",
            Overlay::Risk => "Risk",
            Overlay::BotVisits => "Search visits",
            Overlay::NetPriors => "Net priors",
            Overlay::None => "Plain",
        }
    }

    /// `1`..`5`, the keyboard shortcut that selects it.
    pub fn key(self) -> char {
        match self {
            Overlay::Moves => '1',
            Overlay::Risk => '2',
            Overlay::BotVisits => '3',
            Overlay::NetPriors => '4',
            Overlay::None => '5',
        }
    }
}

/// The tackle-zone layer, separate from the tints above because it coexists
/// with any of them. `Auto` paints the mover's opponent's zones whenever a
/// move is being chosen (`ViewState::threat_team`) and nothing otherwise;
/// `Always` paints them through kickoffs, dice prompts and the bot's think
/// too; `Off` never does — with two bots playing, the layer flips sides every
/// turn and some people would rather read the board bare.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TzMode {
    #[default]
    Auto,
    Always,
    Off,
}

impl TzMode {
    pub fn next(self) -> Self {
        match self {
            TzMode::Auto => TzMode::Always,
            TzMode::Always => TzMode::Off,
            TzMode::Off => TzMode::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TzMode::Auto => "auto",
            TzMode::Always => "always",
            TzMode::Off => "off",
        }
    }
}

/// A square whose action menu is open, because it offers more than one.
#[derive(Clone, PartialEq, Debug)]
pub struct Menu {
    pub pos: Position,
    pub actions: Vec<botbowl_web_proto::PosAT>,
}

/// How a random-start drive ended: attacker, who scored, home score, away score.
pub type DriveOutcome = (TeamType, Option<TeamType>, u8, u8);

#[derive(Clone, Copy)]
pub struct App {
    pub connection: RwSignal<Connection>,
    pub lobby: RwSignal<Option<LobbyInfo>>,
    /// The game spec being assembled in the lobby.
    pub spec: RwSignal<Option<GameSpec>>,
    pub view: RwSignal<Option<ViewState>>,
    /// The game log, oldest first, as the server streamed it: text, dice and
    /// decisions in one. Clicking a line rewinds the game to its step.
    pub log: RwSignal<Vec<LogEntry>>,
    /// Every decision of this game, human and bot, in order.
    pub decisions: RwSignal<Vec<DecisionRecord>>,
    /// The decision the inspector is open on. `None` follows the game: the
    /// newest decision is always the one shown.
    pub selected: RwSignal<Option<u64>>,
    /// The last `ExpandNode` answer, for the tree explorer.
    pub node: RwSignal<Option<NodeExpansion>>,
    /// Where the explorer currently is, as an edge path from the root.
    pub node_path: RwSignal<Vec<botbowl_web_proto::search::SearchEdge>>,
    pub errors: RwSignal<Vec<String>>,
    pub thinking: RwSignal<Option<String>>,
    pub game_over: RwSignal<Option<(Option<TeamType>, u8, u8)>>,
    /// A random-start drive ended: attacker, who scored, the score.
    pub drive_over: RwSignal<Option<DriveOutcome>>,
    /// Built-in and saved teams, as the server last listed them.
    pub teams: RwSignal<Vec<TeamDef>>,
    /// Pictures the team editor can offer.
    pub pictures: RwSignal<Vec<String>>,
    /// The path the last upload was stored under, for the editor to pick up.
    pub uploaded: RwSignal<Option<String>>,
    /// A short confirmation for the team editor ("saved").
    pub notice: RwSignal<Option<String>>,
    pub screen: RwSignal<Screen>,
    pub pinned: RwSignal<Option<RollResult>>,
    pub saved: RwSignal<Option<String>>,
    /// Plan 043: the net's read of the *current* position — value and policy. Updated on every
    /// board change, so it stays live during a human's turn. `None` when no seat has a net.
    pub net_now: RwSignal<Option<NetReadout>>,
    /// How the server is pacing the bot. Set from the step control and then
    /// echoed back on every view, which is the authority; kept here because
    /// it outlives one game — "New game" keeps the pacing you chose.
    pub step_mode: RwSignal<StepMode>,
    /// The `Auto` delay the speed slider last showed, remembered while
    /// another mode is selected so switching back does not reset it.
    pub step_ms: RwSignal<u64>,

    // ---- local view state, never sent anywhere
    pub overlay: RwSignal<Overlay>,
    pub tz: RwSignal<TzMode>,
    pub menu: RwSignal<Option<Menu>>,
    /// Square the pointer is over, for the route preview.
    pub hover: RwSignal<Option<Position>>,
    /// A board that is not the live one — a tree node's, or a past
    /// decision's — drawn on the pitch in the live board's place.
    pub hypothetical: RwSignal<Option<ViewState>>,
    /// Which past decision `hypothetical` is, when it is one.
    pub board_of: RwSignal<Option<u64>>,
    pub inspector_open: RwSignal<bool>,
}

/// The log is a transcript, but a bot-vs-bot game left running overnight must
/// not grow the DOM without bound. The server keeps the whole thing, so a
/// rewind to anything older is still a `RewindTo` away via the decision log.
pub const LOG_CAP: usize = 3000;

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        App {
            connection: RwSignal::new(Connection::Connecting),
            lobby: RwSignal::new(None),
            spec: RwSignal::new(None),
            view: RwSignal::new(None),
            log: RwSignal::new(Vec::new()),
            decisions: RwSignal::new(Vec::new()),
            selected: RwSignal::new(None),
            node: RwSignal::new(None),
            node_path: RwSignal::new(Vec::new()),
            errors: RwSignal::new(Vec::new()),
            thinking: RwSignal::new(None),
            game_over: RwSignal::new(None),
            drive_over: RwSignal::new(None),
            teams: RwSignal::new(Vec::new()),
            pictures: RwSignal::new(Vec::new()),
            uploaded: RwSignal::new(None),
            notice: RwSignal::new(None),
            screen: RwSignal::new(Screen::Lobby),
            pinned: RwSignal::new(None),
            saved: RwSignal::new(None),
            net_now: RwSignal::new(None),
            // The server's `LobbyInfo.step_mode` replaces this as soon as the socket opens.
            step_mode: RwSignal::new(StepMode::Auto { ms: 600 }),
            step_ms: RwSignal::new(600),
            overlay: RwSignal::new(Overlay::default()),
            tz: RwSignal::new(TzMode::default()),
            menu: RwSignal::new(None),
            hover: RwSignal::new(None),
            hypothetical: RwSignal::new(None),
            board_of: RwSignal::new(None),
            inspector_open: RwSignal::new(true),
        }
    }

    /// True while a human seat is the one being asked something.
    ///
    /// A held step is not one of those moments even when `to_act` still names
    /// the human: the engine is standing on a roll it has not made, and the
    /// server would reject an action. Stepping made those boards visible for
    /// the first time, so the check has to be here.
    pub fn my_turn(&self) -> bool {
        self.view.get().is_some_and(|v| {
            v.human_to_act() && !v.scoreboard.game_over && !v.bot_thinking && !v.paused && v.pending_roll.is_none()
        })
    }

    /// The decision the inspector shows: the selected one, or the newest.
    pub fn inspected(&self) -> Option<DecisionRecord> {
        let selected = self.selected.get();
        self.decisions.with(|d| match selected {
            Some(i) => d.get(i as usize).cloned(),
            None => d.last().cloned(),
        })
    }

    /// The search the board's visit overlay paints: the inspected decision's,
    /// or — while following — the newest search, so the overlay does not go
    /// blank on every human move.
    pub fn report(&self) -> Option<SearchReport> {
        let selected = self.selected.get();
        self.decisions.with(|d| match selected {
            Some(i) => d.get(i as usize).and_then(|r| r.search.as_deref().cloned()),
            None => d.iter().rev().find_map(|r| r.search.as_deref().cloned()),
        })
    }

    /// The policy the net-priors overlay paints: the live position's while
    /// following, the inspected decision's otherwise.
    pub fn priors_shown(&self) -> Option<NetReadout> {
        match self.selected.get() {
            None => self.net_now.get(),
            Some(i) => self.decisions.with(|d| d.get(i as usize).and_then(|r| r.net.clone())),
        }
    }

    /// Whether `search_id` is still the newest search of the bot that ran it —
    /// the only tree a bot keeps, so the only one that can be walked.
    pub fn walkable(&self, team: TeamType, search_id: u64) -> bool {
        self.decisions.with(|d| {
            d.iter()
                .rev()
                .filter(|r| r.team == team)
                .find_map(|r| r.search.as_ref().map(|s| s.search_id))
                == Some(search_id)
        })
    }

    /// Put the live board back on the pitch.
    pub fn back_to_live(&self) {
        self.hypothetical.set(None);
        self.board_of.set(None);
        self.node.set(None);
        self.node_path.set(Vec::new());
    }

    /// True while the server is holding before a step it could take.
    pub fn paused(&self) -> bool {
        self.view.get().is_some_and(|v| v.paused)
    }

    pub fn error(&self, message: String) {
        self.errors.update(|e| {
            e.push(message);
            if e.len() > 8 {
                e.remove(0);
            }
        });
    }

    /// Clear everything that belongs to one game, keeping the connection.
    pub fn reset_game(&self) {
        self.view.set(None);
        self.log.set(Vec::new());
        self.decisions.set(Vec::new());
        self.selected.set(None);
        self.net_now.set(None);
        self.board_of.set(None);
        self.node.set(None);
        self.node_path.set(Vec::new());
        self.game_over.set(None);
        self.drive_over.set(None);
        self.thinking.set(None);
        self.menu.set(None);
        self.hypothetical.set(None);
        self.pinned.set(None);
        self.saved.set(None);
    }

    /// Positional actions the human may take on `pos`, if any.
    pub fn actions_at(&self, pos: Position) -> Vec<botbowl_web_proto::PosAT> {
        if !self.my_turn() {
            return Vec::new();
        }
        self.view
            .get()
            .and_then(|v| v.square(pos).map(|s| s.actions.clone()))
            .unwrap_or_default()
    }
}

/// Where a click on a square should go.
pub enum Click {
    Nothing,
    Send(Action),
    OpenMenu(Menu),
}

pub fn click_target(app: &App, pos: Position) -> Click {
    let actions = app.actions_at(pos);
    match actions.len() {
        0 => Click::Nothing,
        1 => Click::Send(Action::Positional(actions[0], pos)),
        _ => Click::OpenMenu(Menu { pos, actions }),
    }
}

//! The client's whole state: a handful of signals, all of them fed by the
//! server.
//!
//! The client is a renderer (decision 3 of plan 034) — it owns no game logic,
//! so there is nothing here but "the last thing the server said" plus local
//! view preferences (which overlay is on, which square's menu is open).

use botbowl_web_proto::dice::{DiceEvent, RollResult};
use botbowl_web_proto::msg::{GameSpec, LobbyInfo};
use botbowl_web_proto::search::{NodeExpansion, SearchReport};
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Overlay {
    /// Legal moves only.
    #[default]
    Moves,
    /// Path success probability, green → red.
    Risk,
    /// Opposing tackle zones.
    TackleZones,
    /// Where the bot's search spent its visits.
    BotVisits,
    /// What the bot's priors wanted before it searched.
    BotPriors,
    /// Nothing — just the board.
    None,
}

impl Overlay {
    pub const ALL: [Overlay; 6] = [
        Overlay::Moves,
        Overlay::Risk,
        Overlay::TackleZones,
        Overlay::BotVisits,
        Overlay::BotPriors,
        Overlay::None,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Overlay::Moves => "Moves",
            Overlay::Risk => "Risk",
            Overlay::TackleZones => "Tackle zones",
            Overlay::BotVisits => "Bot visits",
            Overlay::BotPriors => "Bot priors",
            Overlay::None => "Plain",
        }
    }

    /// `1`..`6`, the keyboard shortcut that selects it.
    pub fn key(self) -> char {
        match self {
            Overlay::Moves => '1',
            Overlay::Risk => '2',
            Overlay::TackleZones => '3',
            Overlay::BotVisits => '4',
            Overlay::BotPriors => '5',
            Overlay::None => '6',
        }
    }
}

/// A square whose action menu is open, because it offers more than one.
#[derive(Clone, PartialEq, Debug)]
pub struct Menu {
    pub pos: Position,
    pub actions: Vec<botbowl_web_proto::PosAT>,
}

#[derive(Clone, Copy)]
pub struct App {
    pub connection: RwSignal<Connection>,
    pub lobby: RwSignal<Option<LobbyInfo>>,
    /// The game spec being assembled in the lobby.
    pub spec: RwSignal<Option<GameSpec>>,
    pub view: RwSignal<Option<ViewState>>,
    /// Newest first, capped.
    pub dice: RwSignal<Vec<DiceEvent>>,
    /// The most recent bot search, and whether it is still explorable.
    pub report: RwSignal<Option<SearchReport>>,
    /// The last `ExpandNode` answer, for the tree explorer.
    pub node: RwSignal<Option<NodeExpansion>>,
    /// Where the explorer currently is, as an edge path from the root.
    pub node_path: RwSignal<Vec<botbowl_web_proto::search::SearchEdge>>,
    pub errors: RwSignal<Vec<String>>,
    pub thinking: RwSignal<Option<String>>,
    pub game_over: RwSignal<Option<(Option<TeamType>, u8, u8)>>,
    pub pinned: RwSignal<Option<RollResult>>,
    pub saved: RwSignal<Option<String>>,

    // ---- local view state, never sent anywhere
    pub overlay: RwSignal<Overlay>,
    pub menu: RwSignal<Option<Menu>>,
    /// Square the pointer is over, for the route preview.
    pub hover: RwSignal<Option<Position>>,
    /// Board shown in the inspector's "step into the PV" panel.
    pub hypothetical: RwSignal<Option<ViewState>>,
    pub inspector_open: RwSignal<bool>,
}

/// Keep the ticker bounded — it is a running commentary, not a transcript.
pub const DICE_TICKER: usize = 40;

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
            dice: RwSignal::new(Vec::new()),
            report: RwSignal::new(None),
            node: RwSignal::new(None),
            node_path: RwSignal::new(Vec::new()),
            errors: RwSignal::new(Vec::new()),
            thinking: RwSignal::new(None),
            game_over: RwSignal::new(None),
            pinned: RwSignal::new(None),
            saved: RwSignal::new(None),
            overlay: RwSignal::new(Overlay::default()),
            menu: RwSignal::new(None),
            hover: RwSignal::new(None),
            hypothetical: RwSignal::new(None),
            inspector_open: RwSignal::new(true),
        }
    }

    /// True while the human is the one being asked something.
    pub fn my_turn(&self) -> bool {
        self.view
            .get()
            .is_some_and(|v| v.to_act == Some(v.human) && !v.scoreboard.game_over && !v.bot_thinking)
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
        self.dice.set(Vec::new());
        self.report.set(None);
        self.node.set(None);
        self.node_path.set(Vec::new());
        self.game_over.set(None);
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

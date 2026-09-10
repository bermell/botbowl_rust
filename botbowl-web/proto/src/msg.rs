//! The websocket protocol: one socket per game, JSON messages, full
//! [`ViewState`] on every change (decision 7 — no deltas for a POC).

use serde::{Deserialize, Serialize};

use crate::action::{Action, TeamType};
use crate::dice::{DiceEvent, RollResult};
use crate::search::{NodeExpansion, SearchEdge, SearchReport};
use crate::view::ViewState;

/// A board size in the terms the lobby and the model filenames use: the
/// **playable** rectangle. The engine board is this plus a 2-cell border.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardSpec {
    pub width: i8,
    pub height: i8,
    pub team_size: usize,
}

impl BoardSpec {
    pub fn new(width: i8, height: i8, team_size: usize) -> Self {
        Self {
            width,
            height,
            team_size,
        }
    }

    /// The `_WxH_` tag a model filename carries.
    pub fn tag(self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Engine dims — playable plus the out-of-bounds border.
    pub fn engine_dims(self) -> (i8, i8, usize) {
        (self.width + 2, self.height + 2, self.team_size)
    }

    /// The engine's own constraints, checked before the server builds a state
    /// so a bad lobby choice is an error message rather than a panic.
    pub fn validate(self, capacity: BoardSpec) -> Result<(), String> {
        if self.width < 8 || self.width % 2 != 0 {
            return Err(format!("width {} must be even and >= 8", self.width));
        }
        if self.height < 3 || self.height % 2 == 0 {
            return Err(format!("height {} must be odd and >= 3", self.height));
        }
        if self.team_size < 1 {
            return Err("team size must be >= 1".into());
        }
        if self.width > capacity.width || self.height > capacity.height || self.team_size > capacity.team_size {
            return Err(format!(
                "{}x{}/{} exceeds this binary's compiled capacity {}x{}/{} — rebuild with \
                 BOARD_SIZE_W/BOARD_SIZE_H/BOARD_PLAYERS",
                self.width, self.height, self.team_size, capacity.width, capacity.height, capacity.team_size
            ));
        }
        Ok(())
    }
}

/// How much search one bot move gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Budget {
    Iterations(usize),
    Millis(u64),
}

impl Budget {
    pub fn label(self) -> String {
        match self {
            Budget::Iterations(n) => format!("{n} iterations"),
            Budget::Millis(ms) => format!("{ms} ms"),
        }
    }
}

/// Mirror of `botbowl_mcts::Evaluator`, with the net named by path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluatorSpec {
    Heuristic,
    PureTd,
    /// NN value **and** NN priors.
    Nn {
        model: String,
    },
    /// NN value, scripted priors.
    NnValue {
        model: String,
    },
}

impl EvaluatorSpec {
    pub fn model(&self) -> Option<&str> {
        match self {
            EvaluatorSpec::Nn { model } | EvaluatorSpec::NnValue { model } => Some(model),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            EvaluatorSpec::Heuristic => "heuristic".into(),
            EvaluatorSpec::PureTd => "pure-td".into(),
            EvaluatorSpec::Nn { model } => format!("nn:{model}"),
            EvaluatorSpec::NnValue { model } => format!("nn-value:{model}"),
        }
    }
}

/// Mirror of `botbowl_mcts::BackupMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BackupSpec {
    #[default]
    Minimax,
    Mean,
}

/// Mirror of `botbowl_mcts::PuctMode`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PuctSpec {
    Raw { c: f32 },
    NormalisedQ { c: f32, range_floor: f32 },
}

/// Mirror of `botbowl_mcts::TieBreak`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TieBreakSpec {
    #[default]
    Hash,
    Asc,
    Desc,
    Mover,
}

/// Every MCTS knob the lobby exposes. Defaults match `MctsBot::new` with no
/// `BLOOD_MCTS_*` set, so a default `MctsSpec` is the shipped configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MctsSpec {
    pub budget: Budget,
    pub evaluator: EvaluatorSpec,
    /// `None` = `available_parallelism()`.
    pub workers: Option<usize>,
    pub backup: BackupSpec,
    pub puct: PuctSpec,
    pub fpu_reduction: f32,
    pub horizon_turns: u8,
    /// `false` disables the horizon entirely (search to game over).
    pub horizon: bool,
    pub tree_reuse: bool,
    pub virtual_loss: i32,
    pub tie_break: TieBreakSpec,
}

impl Default for MctsSpec {
    fn default() -> Self {
        MctsSpec {
            budget: Budget::Iterations(2000),
            evaluator: EvaluatorSpec::Heuristic,
            workers: None,
            backup: BackupSpec::Minimax,
            puct: PuctSpec::Raw { c: 10.0 },
            fpu_reduction: 0.0,
            horizon_turns: 1,
            horizon: true,
            tree_reuse: true,
            virtual_loss: 30,
            tie_break: TieBreakSpec::Hash,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BotSpec {
    Random,
    Scripted,
    Mcts(MctsSpec),
}

impl BotSpec {
    pub fn label(&self) -> String {
        match self {
            BotSpec::Random => "random".into(),
            BotSpec::Scripted => "scripted".into(),
            BotSpec::Mcts(m) => format!("mcts[{}, {}]", m.budget.label(), m.evaluator.label()),
        }
    }

    /// Only the MCTS bot has a search to report on.
    pub fn has_search_report(&self) -> bool {
        matches!(self, BotSpec::Mcts(_))
    }
}

/// Where a game starts. `NewGame` is the coin toss; the others are the phase-3
/// debugging entry points.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StartFrom {
    #[default]
    CoinToss,
    /// A `botbowl-ui`-compatible `Recording` file, resumed at one micro-step.
    Recording { path: String, step: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameSpec {
    pub board: BoardSpec,
    /// The side the browser plays.
    pub human: TeamType,
    pub bot: BotSpec,
    /// Seeds the server's own dice RNG. `None` = from entropy.
    pub seed: Option<u64>,
    pub start: StartFrom,
}

impl GameSpec {
    /// The 14x7 default the plan targets first.
    pub fn default_for(capacity: BoardSpec) -> Self {
        let board = if BoardSpec::new(14, 7, 4).validate(capacity).is_ok() {
            BoardSpec::new(14, 7, 4)
        } else {
            capacity
        };
        GameSpec {
            board,
            human: TeamType::Home,
            bot: BotSpec::Scripted,
            seed: None,
            start: StartFrom::CoinToss,
        }
    }
}

/// One model file the server found under its `--models-dir`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Path as the server will open it.
    pub path: String,
    /// Basename, for the dropdown.
    pub name: String,
    /// The `_WxH_` tag parsed out of the filename, when it has one. The lobby
    /// only offers models whose tag matches the chosen board (a mismatch
    /// panics inside `NnEvaluator`).
    pub board_tag: Option<String>,
}

/// Sent once when a socket opens, before any game exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LobbyInfo {
    /// The board this binary was compiled for — the ceiling on `GameSpec`.
    pub capacity: BoardSpec,
    /// Board presets that fit the capacity.
    pub boards: Vec<BoardSpec>,
    pub models: Vec<ModelInfo>,
    pub defaults: GameSpec,
    /// Server version line for the footer.
    pub server: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    NewGame(GameSpec),
    Act(Action),
    Undo,
    /// Walk into the cached search DAG. The path is the edge sequence from
    /// the root of the search identified by `search_id` — which must be the
    /// *most recent* one, because that is the only tree the bot keeps.
    ExpandNode {
        search_id: u64,
        path: Vec<SearchEdge>,
        /// Also render the node's board (needs `MemoryMode::StoreState`).
        with_view: bool,
    },
    /// Pin the next roll the engine asks for; `None` clears a pending pin.
    FixNextRoll(Option<RollResult>),
    SaveRecording {
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServerMsg {
    Lobby(Box<LobbyInfo>),
    /// The authoritative board. Sent on every change.
    View(Box<ViewState>),
    Dice(DiceEvent),
    BotThinking {
        team: TeamType,
        budget: String,
    },
    BotMoved {
        action: Action,
        report: Option<Box<SearchReport>>,
    },
    Node(Box<NodeExpansion>),
    /// A pinned roll is queued (or was cleared).
    RollPinned(Option<RollResult>),
    Saved {
        path: String,
    },
    GameOver {
        winner: Option<TeamType>,
        home_score: u8,
        away_score: u8,
    },
    Error(String),
}

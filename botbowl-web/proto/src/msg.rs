//! The websocket protocol: one socket per game, JSON messages, full
//! [`ViewState`] on every change (decision 7 — no deltas for a POC).

use serde::{Deserialize, Serialize};

use crate::action::{Action, TeamType};
use crate::decision::{DecisionRecord, NetReadout};
use crate::dice::{DiceEvent, RollResult};
use crate::search::{NodeExpansion, SearchEdge};
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
        if self.height < 3 {
            return Err(format!("height {} must be >= 3", self.height));
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
/// `BLOOD_MCTS_*` set, so a default `MctsSpec` is the shipped configuration —
/// except the evaluator: the web bot **always** takes both its leaf value and
/// its priors from `model` (`Evaluator::Nn`). The heuristic, pure-TD and
/// value-only evaluators are CLI diagnostics and are not offered here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MctsSpec {
    pub budget: Budget,
    /// The net, by the path (or basename) the lobby's `ModelInfo` offered.
    pub model: String,
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
            model: String::new(),
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
    /// MCTS with the net's value and priors.
    Mcts(MctsSpec),
}

impl BotSpec {
    pub fn label(&self) -> String {
        match self {
            BotSpec::Random => "random".into(),
            BotSpec::Mcts(m) => format!("mcts[{}, {}]", m.budget.label(), model_short(&m.model)),
        }
    }

    /// The net this bot loads, if any.
    pub fn model(&self) -> Option<&str> {
        match self {
            BotSpec::Mcts(m) => Some(&m.model),
            BotSpec::Random => None,
        }
    }

    /// Only the MCTS bot has a search to report on.
    pub fn has_search_report(&self) -> bool {
        matches!(self, BotSpec::Mcts(_))
    }
}

/// `az_v7/bbnet_14x7_gen19.onnx` out of a full path — the file and its folder,
/// which is what tells two runs' `gen00`s apart in a label.
pub fn model_short(model: &str) -> &str {
    let mut cuts = model.rmatch_indices(['/', '\\']).map(|(i, _)| i);
    match (cuts.next(), cuts.next()) {
        (Some(_), Some(i)) => &model[i + 1..],
        _ => model,
    }
}

#[cfg(test)]
mod tests {
    use super::model_short;

    #[test]
    fn a_model_label_keeps_its_folder() {
        assert_eq!(model_short("/r/models/az_v7/gen13.onnx"), "az_v7/gen13.onnx");
        assert_eq!(model_short("az_v7/gen13.onnx"), "az_v7/gen13.onnx");
        assert_eq!(model_short("gen13.onnx"), "gen13.onnx");
    }
}

/// Who plays one side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Seat {
    /// This browser.
    Human,
    Bot(BotSpec),
}

impl Seat {
    pub fn is_human(&self) -> bool {
        matches!(self, Seat::Human)
    }

    pub fn bot(&self) -> Option<&BotSpec> {
        match self {
            Seat::Human => None,
            Seat::Bot(b) => Some(b),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Seat::Human => "you".into(),
            Seat::Bot(b) => b.label(),
        }
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
    /// Either side may be a human or a bot: bot-vs-bot is how you watch two
    /// configurations (or two nets) play each other.
    pub home: Seat,
    pub away: Seat,
    /// Seeds the server's own dice RNG. `None` = from entropy.
    pub seed: Option<u64>,
    pub start: StartFrom,
}

impl GameSpec {
    /// The 14x7 default the plan targets first: the browser plays Home against
    /// an MCTS bot on the newest net that fits the board, or against the random
    /// bot when the server has no such net.
    pub fn default_for(capacity: BoardSpec, models: &[ModelInfo]) -> Self {
        let board = if BoardSpec::new(14, 7, 4).validate(capacity).is_ok() {
            BoardSpec::new(14, 7, 4)
        } else {
            capacity
        };
        let away = match models.iter().find(|m| m.fits(board)) {
            Some(m) => BotSpec::Mcts(MctsSpec {
                model: m.path.clone(),
                ..Default::default()
            }),
            None => BotSpec::Random,
        };
        GameSpec {
            board,
            home: Seat::Human,
            away: Seat::Bot(away),
            seed: None,
            start: StartFrom::CoinToss,
        }
    }

    pub fn seat(&self, team: TeamType) -> &Seat {
        match team {
            TeamType::Home => &self.home,
            TeamType::Away => &self.away,
        }
    }

    /// The sides this browser plays — none for bot-vs-bot, both for hot-seat.
    pub fn humans(&self) -> Vec<TeamType> {
        [TeamType::Home, TeamType::Away]
            .into_iter()
            .filter(|t| self.seat(*t).is_human())
            .collect()
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

impl ModelInfo {
    /// A tagged model fits only its own board; an untagged one is offered
    /// everywhere, because we cannot know what it was trained on.
    pub fn fits(&self, board: BoardSpec) -> bool {
        self.board_tag.as_deref().is_none_or(|t| t == board.tag())
    }
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

/// How fast the session is allowed to run through the steps the human does
/// not answer — the bot's moves and the engine's own dice.
///
/// `Run` is the original behaviour: one click plays the bot's whole reply.
/// The other two exist because that is unwatchable — the board jumps from
/// your move to the bot's finished turn with no way to see the order things
/// happened in, or to open the inspector on the search that produced them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StepMode {
    /// Play on until the human has something to decide.
    #[default]
    Run,
    /// Hold before every step and wait for [`ClientMsg::StepOnce`].
    Manual,
    /// Hold `ms` before every step, then take it.
    Auto { ms: u64 },
}

impl StepMode {
    pub fn label(self) -> String {
        match self {
            StepMode::Run => "run".into(),
            StepMode::Manual => "step".into(),
            StepMode::Auto { ms } => format!("auto {ms} ms"),
        }
    }

    /// The delay an `Auto` mode waits, for a speed control to read back.
    pub fn millis(self) -> Option<u64> {
        match self {
            StepMode::Auto { ms } => Some(ms),
            _ => None,
        }
    }
}

/// `NewGame` dwarfs the rest, but it is sent once per game — not worth a box.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    NewGame(GameSpec),
    Act(Action),
    Undo,
    /// Change how the session paces itself. Takes effect immediately, even
    /// while it is already holding.
    SetStepMode(StepMode),
    /// Take one held step. Ignored when the session is not holding.
    StepOnce,
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
    /// Render the board as it stood when decision `index` was taken, so a
    /// logged search's heatmap can be read over the position it was about.
    ShowDecision {
        index: u64,
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
    /// One decision was taken — by a human or a bot. Every decision is
    /// logged, so the decision log can show the net's read of a human's move
    /// next to the search behind a bot's.
    Decision(Box<DecisionRecord>),
    /// An undo rewound the game: keep only the first `keep` decisions.
    DecisionsTruncated {
        keep: u64,
    },
    Node(Box<NodeExpansion>),
    /// Answer to [`ClientMsg::ShowDecision`]: a hypothetical board, like a
    /// tree node's, never the live one.
    DecisionBoard {
        index: u64,
        view: Box<ViewState>,
    },
    /// Plan 043: the net's read of the **current** position — its value, Home-centric in
    /// `[-1, 1]` (`+1` means it expects Home to score next), and its policy over the legal
    /// actions of whoever is to act.
    ///
    /// Sent with every board change, not only after a bot move, so the debug read-out stays live
    /// during the human's turn. Absent when no seat has a net, which is why this is a message of
    /// its own rather than a field on `ViewState`: the view is re-sent in full on every step and
    /// should not carry a value most sessions do not have.
    Net(Box<NetReadout>),
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

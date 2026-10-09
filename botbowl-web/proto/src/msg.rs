//! The websocket protocol: one socket per game, JSON messages, full
//! [`ViewState`] on every change (decision 7 — no deltas for a POC).

use serde::{Deserialize, Serialize};

use crate::action::{Action, TeamType};
use crate::decision::{DecisionRecord, NetReadout};
use crate::dice::RollResult;
use crate::log::LogEntry;
use crate::search::{NodeExpansion, SearchEdge};
use crate::team::{SkillInfo, TeamDef, DEFAULT_TEAM};
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
    /// Search threads. `None` = `available_parallelism()`, capped by the server's
    /// `--play-max-workers` when it has one.
    pub workers: Option<usize>,
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
            // One thread unless asked: a browser tab left open on bot-vs-bot must not take the
            // whole machine.
            workers: Some(1),
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
    /// One drive from a random-start position — the same draw the training corpus makes for this
    /// seed (`botbowl_play::drives::position_state` with the default bias). It ends when either
    /// side scores, the half changes or the game ends ([`ServerMsg::DriveOver`]). `None` = a
    /// fresh seed each time.
    RandomDrive { seed: Option<u64> },
}

impl StartFrom {
    pub fn is_drive(&self) -> bool {
        matches!(self, StartFrom::RandomDrive { .. })
    }
}

fn default_team() -> String {
    DEFAULT_TEAM.to_string()
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
    /// [`TeamDef::name`]s. A random-start drive keeps its generated players and only takes the
    /// team's pictures, so the position stays the training distribution.
    #[serde(default = "default_team")]
    pub home_team: String,
    #[serde(default = "default_team")]
    pub away_team: String,
    /// No natural one-turn: every player's MA is capped one short of the line-of-scrimmage
    /// to end-zone distance, so nobody walks the ball in from the line on MA alone. On by
    /// default — on the small boards the stock MA turns a handoff into a free touchdown.
    #[serde(default = "yes")]
    pub no_natural_one_turn: bool,
}

fn yes() -> bool {
    true
}

impl BoardSpec {
    /// The most MA a player may have under [`GameSpec::no_natural_one_turn`]: one less than
    /// the squares from their own line of scrimmage to the opponent's end zone. The engine
    /// board is this one plus a two-square border, and that distance is `engine_width / 2 - 1`.
    pub fn no_one_turn_ma(self) -> u8 {
        let engine_width = self.width as i16 + 2;
        (engine_width / 2 - 2).max(1) as u8
    }
}

impl GameSpec {
    /// The 14x7 default: two MCTS bots on the newest net that fits the board, one search thread
    /// each, or the random bot when the server has no such net. Bot-vs-bot rather than you-vs-bot
    /// so that opening the page shows a game, and one thread so that it costs little.
    pub fn default_for(capacity: BoardSpec, models: &[ModelInfo]) -> Self {
        let board = if BoardSpec::new(14, 7, 4).validate(capacity).is_ok() {
            BoardSpec::new(14, 7, 4)
        } else {
            capacity
        };
        let bot = match models.iter().find(|m| m.fits(board)) {
            Some(m) => BotSpec::Mcts(MctsSpec {
                model: m.path.clone(),
                ..Default::default()
            }),
            None => BotSpec::Random,
        };
        GameSpec {
            board,
            home: Seat::Bot(bot.clone()),
            away: Seat::Bot(bot),
            seed: None,
            start: StartFrom::CoinToss,
            home_team: default_team(),
            away_team: default_team(),
            no_natural_one_turn: true,
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
    /// Built-in teams first, then the saved ones.
    pub teams: Vec<TeamDef>,
    /// Every skill the engine has, for the team editor.
    pub skills: Vec<SkillInfo>,
    /// Pictures the editor can offer: sprite stems under `img/iconssmall/`, then uploaded
    /// `custom/<file>`s.
    pub pictures: Vec<String>,
    /// The server can save teams (it has a config directory).
    pub can_save_teams: bool,
    /// The pacing a new connection starts with.
    pub step_mode: StepMode,
    /// [`StartFrom::Recording`] is accepted (off when the server listens on the network).
    pub can_resume: bool,
}

/// How fast the session is allowed to run through the bots' moves.
///
/// `Run` is the original behaviour: one click plays the bot's whole reply.
/// The other two exist because that is unwatchable — the board jumps from
/// your move to the bot's finished turn with no way to see the order things
/// happened in, or to open the inspector on the search that produced them.
/// The hold sits *between* a bot's search and its move: the board on screen
/// is the position the search was about, the report beside it is that
/// search, and the next step plays the move it chose
/// ([`crate::view::ViewState::pending_action`]). Dice are never held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StepMode {
    /// Play on until the human has something to decide.
    #[default]
    Run,
    /// Hold after every bot search and wait for [`ClientMsg::StepOnce`].
    Manual,
    /// Hold `ms` after every bot search, then play the move.
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
    /// Several actions as one decision — a declaration and its first target
    /// ([`crate::view::IntentView`]). One undo point; dice in between are
    /// rolled, and the chain stops at the first action no longer legal.
    ActChain(Vec<Action>),
    /// Preview a player of the side to act: answered with
    /// [`ServerMsg::Selection`]. Changes nothing.
    Select {
        pos: crate::action::Position,
    },
    /// Play out the rest of the human's setup with the named formation (one
    /// of `SetupView::formations`). Ignored outside the human's setup.
    AutoSetup(String),
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
    /// Rewind the game to micro-step `step` — the one a [`LogEntry`] or a
    /// [`crate::decision::DecisionRecord`] names — and hold there. Everything
    /// after it is dropped: the log and the decision log are truncated
    /// ([`ServerMsg::LogTruncated`], [`ServerMsg::DecisionsTruncated`]) and
    /// play continues from that position when stepped. Under `Run` the
    /// session switches to `Manual` first, or the rewind would be undone by
    /// the next tick.
    RewindTo {
        step: usize,
    },
    /// Create or overwrite a saved team (by name). Answered with [`ServerMsg::Teams`].
    SaveTeam(TeamDef),
    DeleteTeam {
        name: String,
    },
    /// An image for a position, as a `data:image/...;base64,` URL. Answered with
    /// [`ServerMsg::PictureSaved`].
    UploadPicture {
        data_url: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServerMsg {
    Lobby(Box<LobbyInfo>),
    /// The authoritative board. Sent on every change.
    View(Box<ViewState>),
    /// One line of the game log — text, a die, a decision, a score.
    Log(LogEntry),
    /// A rewind or an undo cut the log: keep only the first `keep` lines.
    LogTruncated {
        keep: u64,
    },
    BotThinking {
        team: TeamType,
        budget: String,
    },
    /// Answer to [`ClientMsg::Select`].
    Selection(Box<crate::view::SelectionView>),
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
    /// A random-start drive ended: `scored` is the side that scored, `None` when the half (or
    /// the game) ran out first.
    DriveOver {
        attacker: TeamType,
        scored: Option<TeamType>,
        home_score: u8,
        away_score: u8,
    },
    /// The team list after a save or delete.
    Teams(Vec<TeamDef>),
    /// The picture path an upload was stored under, plus the refreshed picture list.
    PictureSaved {
        picture: String,
        pictures: Vec<String>,
    },
    Error(String),
}

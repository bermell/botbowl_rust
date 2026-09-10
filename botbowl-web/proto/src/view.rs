//! The fully derived board view.
//!
//! Decision 3 of plan 034: the client depends on neither the engine nor its
//! geometry. Every square arrives pre-annotated — square kind, sprite path,
//! legal positional actions, path success probability, block dice, tackle-zone
//! count — so the client is a pure renderer and all the game logic sits in one
//! testable function, `botbowl_web_server::view::derive`.

use serde::{Deserialize, Serialize};

use crate::action::{PosAT, Position, SimpleAT, TeamType};
use crate::dice::{NumBlockDices, RequestedRoll};

/// The runtime board. `width`/`height` are the **engine** dims, i.e. playable
/// plus the 2-cell out-of-bounds border, so a `Position` indexes the grid
/// directly. The playable rect is `x in 1..=width-2`, `y in 1..=height-2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dims {
    pub width: i8,
    pub height: i8,
    pub team_size: usize,
}

impl Dims {
    pub fn playable_width(self) -> i8 {
        self.width - 2
    }
    pub fn playable_height(self) -> i8 {
        self.height - 2
    }
    /// Row-major index of a position in [`ViewState::squares`].
    pub fn index(self, pos: Position) -> usize {
        pos.y as usize * self.width as usize + pos.x as usize
    }
    /// How the lobby writes this board: `14x7`.
    pub fn tag(self) -> String {
        format!("{}x{}", self.playable_width(), self.playable_height())
    }
}

/// Static colouring of a square. Derived server-side from `BoardDims` so the
/// client carries no geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SquareKind {
    /// The border ring — never occupied, only a scatter/throw-in waypoint.
    OutOfBounds,
    /// `x == 1`: the end zone Home attacks.
    EndzoneHome,
    /// `x == width - 2`: the end zone Away attacks.
    EndzoneAway,
    /// A line-of-scrimmage square on either side of halfway.
    Scrimmage,
    /// Wide zone north of the scrimmage rows.
    WingNorth,
    /// Wide zone south of the scrimmage rows.
    WingSouth,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayerRole {
    Lineman,
    Blitzer,
    Thrower,
    Catcher,
}

impl PlayerRole {
    pub fn sprite_stem(self) -> &'static str {
        match self {
            PlayerRole::Lineman => "hlineman1",
            PlayerRole::Blitzer => "hblitzer1",
            PlayerRole::Thrower => "hthrower1",
            PlayerRole::Catcher => "hcatcher1",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PlayerRole::Lineman => "Lineman",
            PlayerRole::Blitzer => "Blitzer",
            PlayerRole::Thrower => "Thrower",
            PlayerRole::Catcher => "Catcher",
        }
    }

    /// `iconssmall/h<role>1[b][an].gif` — `b` is the home colourway, `an`
    /// means "has not acted yet" (the engine's `used == false`).
    pub fn sprite(self, team: TeamType, used: bool) -> String {
        let home = if team == TeamType::Home { "b" } else { "" };
        let not_acted = if used { "" } else { "an" };
        format!("iconssmall/{}{home}{not_acted}.gif", self.sprite_stem())
    }
}

/// Mirror of `model::PlayerStatus`. "Prone" is `Down`; "has acted" is the
/// separate [`PlayerView::used`] flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayerStatus {
    Up,
    Down,
    Stunned,
}

impl PlayerStatus {
    /// Overlay sprite under `/img/`, when the status has one.
    pub fn overlay(self) -> Option<&'static str> {
        match self {
            PlayerStatus::Up => None,
            PlayerStatus::Down => Some("player_status/prone.gif"),
            PlayerStatus::Stunned => Some("player_status/stunned.gif"),
        }
    }
}

/// Mirror of `model::DugoutPlace` (the engine spells the casualty box
/// `Injuried`; the wire uses the corrected spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DugoutPlace {
    Reserves,
    Heated,
    KnockOut,
    Injured,
    Ejected,
}

impl DugoutPlace {
    pub fn label(self) -> &'static str {
        match self {
            DugoutPlace::Reserves => "Reserves",
            DugoutPlace::Heated => "Heated",
            DugoutPlace::KnockOut => "KO",
            DugoutPlace::Injured => "Casualty",
            DugoutPlace::Ejected => "Sent off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Weather {
    Nice,
    Sunny,
    Rain,
    Blizzard,
    Sweltering,
}

impl Weather {
    pub fn label(self) -> &'static str {
        match self {
            Weather::Nice => "Nice",
            Weather::Sunny => "Very sunny",
            Weather::Rain => "Pouring rain",
            Weather::Blizzard => "Blizzard",
            Weather::Sweltering => "Sweltering heat",
        }
    }
}

/// A player on the pitch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerView {
    /// Engine `PlayerID` — the fielded-players slot index. Stable within a
    /// drive but **reused** after a player leaves the pitch, so never treat it
    /// as a persistent identity.
    pub id: usize,
    pub team: TeamType,
    pub role: PlayerRole,
    pub status: PlayerStatus,
    /// Has already acted this turn.
    pub used: bool,
    pub sprite: String,
    pub st: u8,
    pub ma: u8,
    pub ag: u8,
    pub av: u8,
    /// Movement (including GFIs) still available this activation.
    pub movement_left: u8,
    pub has_ball: bool,
    pub skills: Vec<String>,
    /// Currently the activated player.
    pub active: bool,
}

/// Where the ball is, as far as this square is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BallView {
    OnGround,
    InAir,
    Carried,
}

/// A pathfinder route to this square, for the hover overlay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteView {
    /// Squares from the first step to this one (the origin is excluded).
    pub steps: Vec<Position>,
    /// Rolls the route needs, in order — `"Dodge 3+"`, `"GFI"`, `"Pickup 4+"`.
    pub rolls: Vec<String>,
}

/// One board square, fully annotated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquareView {
    pub pos: Position,
    pub kind: SquareKind,
    pub player: Option<PlayerView>,
    pub ball: Option<BallView>,
    /// Standing Home players adjacent to this square — half the tackle-zone
    /// overlay. (Counted for empty squares too, which is why this is not
    /// `get_tz_on`, which needs a player id.)
    pub tz_home: u8,
    /// Standing Away players adjacent to this square.
    pub tz_away: u8,
    /// Legal positional actions that target this square, sorted.
    pub actions: Vec<PosAT>,
    /// Probability the pathfinder route to this square succeeds, when the
    /// square is offered as a path action.
    pub move_prob: Option<f32>,
    /// Block dice a `Block`/`Blitz` into this square would get.
    pub block_dice: Option<NumBlockDices>,
    pub route: Option<RouteView>,
}

impl SquareView {
    /// Tackle zones exerted on this square by `team`.
    pub fn tz(&self, team: TeamType) -> u8 {
        match team {
            TeamType::Home => self.tz_home,
            TeamType::Away => self.tz_away,
        }
    }

    pub fn empty(pos: Position, kind: SquareKind) -> Self {
        SquareView {
            pos,
            kind,
            player: None,
            ball: None,
            tz_home: 0,
            tz_away: 0,
            actions: Vec::new(),
            move_prob: None,
            block_dice: None,
            route: None,
        }
    }
}

/// One team's bench, grouped by box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DugoutView {
    pub team: TeamType,
    pub players: Vec<DugoutPlayerView>,
}

impl DugoutView {
    pub fn count(&self, place: DugoutPlace) -> usize {
        self.players.iter().filter(|p| p.place == place).count()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DugoutPlayerView {
    pub id: usize,
    pub team: TeamType,
    pub role: PlayerRole,
    pub place: DugoutPlace,
    pub sprite: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scoreboard {
    /// `0` before the first half starts.
    pub half: u8,
    pub home_turn: u8,
    pub away_turn: u8,
    pub home_score: u8,
    pub away_score: u8,
    pub home_rerolls: u8,
    pub away_rerolls: u8,
    /// A team may use at most one reroll per turn.
    pub home_can_reroll: bool,
    pub away_can_reroll: bool,
    pub weather: Weather,
    /// Whose *turn* it is (not necessarily who must answer the next prompt).
    pub team_turn: TeamType,
    pub kicking_this_drive: TeamType,
    pub game_over: bool,
    pub winner: Option<TeamType>,
}

/// A non-positional action offered to the player, ready to render as a button.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimpleActionView {
    pub at: SimpleAT,
    pub label: String,
    /// Sprite for the button (block-die faces), relative to `/img/`.
    pub img: Option<String>,
}

/// Everything the client needs to draw one moment of the game.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewState {
    /// Monotonic per-session counter — the client ignores stale views.
    pub seq: u64,
    pub dims: Dims,
    pub scoreboard: Scoreboard,
    /// Row-major over the **full** grid including the border ring, so
    /// `squares[dims.index(pos)]` is `pos`.
    pub squares: Vec<SquareView>,
    /// `[home, away]`.
    pub dugouts: Vec<DugoutView>,
    pub simple_actions: Vec<SimpleActionView>,
    /// Which side must supply the next action. `None` while the engine is
    /// mid-procedure or the game is over.
    pub to_act: Option<TeamType>,
    /// The side this browser plays.
    pub human: TeamType,
    /// `proc_stack_top()` — the rules subsystem currently asking.
    pub proc: String,
    pub active_player: Option<usize>,
    /// Set while the engine is paused on a roll the server has not yet made
    /// (only observable with a pinned roll queued).
    pub pending_roll: Option<RequestedRoll>,
    pub log_tail: Vec<String>,
    pub can_undo: bool,
    /// During setup: whether the current placement is legal enough to end on.
    pub setup_legal: Option<bool>,
    /// True while a bot search is running.
    pub bot_thinking: bool,
}

impl ViewState {
    pub fn square(&self, pos: Position) -> Option<&SquareView> {
        self.squares.get(self.dims.index(pos))
    }

    /// Every square that offers at least one positional action.
    pub fn actionable(&self) -> impl Iterator<Item = &SquareView> {
        self.squares.iter().filter(|s| !s.actions.is_empty())
    }
}

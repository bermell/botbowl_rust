use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{error, fmt};

use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use super::dices::{D6Target, RequestedRoll, RollResult, Sum2D6Target};
use super::gamestate::GameState;
use super::procedures::AnyProc;
use super::table::{NumBlockDices, PlayerRole, PosAT, SimpleAT, Skill};
use crate::core::table;

pub type PlayerID = usize;
pub type DugoutPlayerID = usize;
pub type Coord = i8;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FullPitch<T> {
    data: [[T; HEIGHT]; WIDTH],
}
impl<T> Index<Position> for FullPitch<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Position) -> &Self::Output {
        // Hot path: pitch indexing happens in every pathfinding step.
        // `Position::to_usize` returned `Result` via `?`, which the optimizer
        // wasn't fully eliding. Direct casts skip Result construction entirely;
        // out-of-bounds positions are programming errors caught in debug builds.
        debug_assert!(
            index.x >= 0 && index.y >= 0 && (index.x as usize) < WIDTH && (index.y as usize) < HEIGHT,
            "FullPitch index out of bounds: {:?}",
            index,
        );
        &self.data[index.x as usize][index.y as usize]
    }
}
impl<T> IndexMut<Position> for FullPitch<T> {
    #[inline]
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        debug_assert!(
            index.x >= 0 && index.y >= 0 && (index.x as usize) < WIDTH && (index.y as usize) < HEIGHT,
            "FullPitch index_mut out of bounds: {:?}",
            index,
        );
        &mut self.data[index.x as usize][index.y as usize]
    }
}

impl<T> FullPitch<T> {
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.data[x][y]
    }
    #[inline]
    pub fn get_pos(&self, pos: Position) -> &T {
        &self[pos]
    }
    #[inline]
    pub fn get_pos_mut(&mut self, pos: Position) -> &mut T {
        &mut self[pos]
    }
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[x][y]
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|r| r.iter())
    }
    pub fn iter_position(&self) -> impl Iterator<Item = (Position, &T)> {
        Position::all_positions().map(|p| (p, &self[p]))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().flat_map(|r| r.iter_mut())
    }
}
impl<T: Default> Default for FullPitch<T> {
    fn default() -> Self {
        FullPitch {
            data: Default::default(),
        }
    }
}
impl<T> FullPitch<Option<T>> {
    pub fn clear(&mut self) {
        *self = Default::default();
    }
    pub fn take_pos(&mut self, pos: Position) -> Option<T> {
        self.data[pos.x as usize][pos.y as usize].take()
    }
}

// Board dimensions and team size are build-time configurable (plan 017). The
// constants below — WIDTH/HEIGHT/WIDTH_/HEIGHT_, TEAM_SIZE/ROSTER_PER_TEAM, and
// the LOS / wing geometry — are emitted by `build.rs` into `$OUT_DIR`. The
// default build reproduces the historical 28x17 / 11-player values exactly.
// `Coord` (defined above) is in scope for the generated file.
include!(concat!(env!("OUT_DIR"), "/board_config.rs"));

// Change the alias to `Box<error::Error>`.
pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Direction {
    pub dx: Coord,
    pub dy: Coord,
}
impl From<(Coord, Coord)> for Direction {
    fn from(xy: (Coord, Coord)) -> Self {
        let (dx, dy) = xy;
        Direction { dx, dy }
    }
}
const ALL_DIRECTIONS: [Direction; 8] = [
    Direction { dx: 1, dy: 1 },
    Direction { dx: 0, dy: 1 },
    Direction { dx: -1, dy: 1 },
    Direction { dx: 1, dy: 0 },
    Direction { dx: -1, dy: 0 },
    Direction { dx: 1, dy: -1 },
    Direction { dx: 0, dy: -1 },
    Direction { dx: -1, dy: -1 },
];
impl Direction {
    pub fn all_directions_iter() -> impl Iterator<Item = &'static Direction> {
        ALL_DIRECTIONS.iter()
    }
    pub fn all_directions_as_array() -> [Direction; 8] {
        ALL_DIRECTIONS
    }
    pub fn distance(&self) -> Coord {
        max(self.dx.abs(), self.dy.abs())
    }
    pub fn up() -> Direction {
        Direction { dx: 0, dy: -1 }
    }
    pub fn upleft() -> Direction {
        Direction { dx: -1, dy: -1 }
    }
    pub fn upright() -> Direction {
        Direction { dx: 1, dy: -1 }
    }
    pub fn left() -> Direction {
        Direction { dx: -1, dy: 0 }
    }
    pub fn right() -> Direction {
        Direction { dx: 1, dy: 0 }
    }
    pub fn down() -> Direction {
        Direction { dx: 0, dy: 1 }
    }
    pub fn downleft() -> Direction {
        Direction { dx: -1, dy: 1 }
    }
    pub fn downright() -> Direction {
        Direction { dx: 1, dy: 1 }
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Position {
    pub x: Coord,
    pub y: Coord,
}
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Position {
    pub fn new(xy: (Coord, Coord)) -> Position {
        let (x, y) = xy;
        Position { x, y }
    }
    pub fn all_positions() -> impl Iterator<Item = Position> {
        (0..WIDTH_).cartesian_product(0..HEIGHT_).map(Position::new)
    }
    pub fn to_usize(&self) -> Result<(usize, usize)> {
        let x: usize = usize::try_from(self.x)?;
        let y: usize = usize::try_from(self.y)?;
        Ok((x, y))
    }
    pub fn from_usize(x: usize, y: usize) -> Result<Position> {
        let x_: Coord = Coord::try_from(x)?;
        let y_: Coord = Coord::try_from(y)?;
        Ok(Position::new((x_, y_)))
    }
    pub fn distance_to(&self, other: &Position) -> Coord {
        (*self - *other).distance()
    }
    pub fn is_out(&self) -> bool {
        self.x <= 0 || self.x >= WIDTH_ - 1 || self.y <= 0 || self.y >= HEIGHT_ - 1
    }
    pub fn is_on_team_side(&self, team: TeamType) -> bool {
        match team {
            TeamType::Home => self.x >= WIDTH_ / 2,
            TeamType::Away => self.x < WIDTH_ / 2,
        }
    }
}
impl From<(usize, usize)> for Position {
    fn from(xy: (usize, usize)) -> Self {
        Position {
            x: Coord::try_from(xy.0).unwrap(),
            y: Coord::try_from(xy.1).unwrap(),
        }
    }
}
impl From<Position> for (usize, usize) {
    fn from(p: Position) -> Self {
        debug_assert!(!p.is_out());
        (usize::try_from(p.x).unwrap(), usize::try_from(p.y).unwrap())
    }
}
impl From<Position> for (u16, u16) {
    fn from(p: Position) -> Self {
        (u16::try_from(p.x).unwrap(), u16::try_from(p.y).unwrap())
    }
}

impl Add<Direction> for Position {
    type Output = Position;

    fn add(self, rhs: Direction) -> Self::Output {
        Position::new((self.x + rhs.dx, self.y + rhs.dy))
    }
}
impl Add<(Coord, Coord)> for Position {
    type Output = Position;

    fn add(self, rhs: (Coord, Coord)) -> Self::Output {
        self + Direction::from(rhs)
    }
}
impl AddAssign<(Coord, Coord)> for Position {
    fn add_assign(&mut self, rhs: (Coord, Coord)) {
        *self = *self + Direction::from(rhs);
    }
}
impl SubAssign<Direction> for Position {
    fn sub_assign(&mut self, rhs: Direction) {
        self.x -= rhs.dx;
        self.y -= rhs.dy;
    }
}

impl Sub<Position> for Position {
    type Output = Direction;

    fn sub(self, rhs: Position) -> Self::Output {
        Direction {
            dx: self.x - rhs.x,
            dy: self.y - rhs.y,
        }
    }
}

impl Sub<Direction> for Position {
    type Output = Position;

    fn sub(self, rhs: Direction) -> Self::Output {
        Position::new((self.x - rhs.dx, self.y - rhs.dy))
    }
}

impl Mul<i8> for Direction {
    type Output = Direction;

    fn mul(self, rhs: i8) -> Self::Output {
        Direction {
            dx: self.dx * rhs,
            dy: self.dy * rhs,
        }
    }
}
impl Mul<Direction> for i8 {
    type Output = Direction;

    fn mul(self, rhs: Direction) -> Self::Output {
        rhs * self
    }
}
impl Mul<i8> for Position {
    type Output = Position;

    fn mul(self, rhs: i8) -> Self::Output {
        Position {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Copy, Clone, Serialize, Deserialize)]
pub enum Action {
    Positional(table::PosAT, Position),
    Simple(table::SimpleAT),
}
impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Positional(at, pos) => write!(f, "{:?} ({:?},{:?})", at, pos.x, pos.y),
            Action::Simple(at) => write!(f, "{:?}", at),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionChoice {
    Positional(Vec<Position>),
    Simple,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum PlayerStatus {
    Up,
    Down,
    Stunned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerStats {
    pub str_: u8,
    pub ma: u8,
    pub ag: u8,
    pass: D6Target,
    pub av: u8,
    pub team: TeamType,
    skills: HashSet<Skill>,
    pub role: PlayerRole,
    //skills: [Option<table::Skill>; 3],
    //injuries
    //spp
}
impl PlayerStats {
    pub fn new_lineman(team: TeamType) -> PlayerStats {
        PlayerStats {
            str_: 3,
            ma: 6,
            ag: 3,
            av: 8,
            team,
            skills: HashSet::new(),
            role: PlayerRole::Lineman,
            pass: D6Target::FourPlus,
        }
    }
    pub fn new_blitzer(team: TeamType) -> PlayerStats {
        PlayerStats {
            str_: 3,
            ma: 7,
            ag: 3,
            av: 9,
            team,
            skills: HashSet::from_iter([Skill::Block]),
            role: PlayerRole::Blitzer,
            pass: D6Target::FourPlus,
        }
    }
    pub fn new_catcher(team: TeamType) -> PlayerStats {
        PlayerStats {
            str_: 2,
            ma: 8,
            ag: 3,
            av: 8,
            team,
            skills: HashSet::from_iter([Skill::Dodge, Skill::Catch]),
            role: PlayerRole::Catcher,
            pass: D6Target::FivePlus,
        }
    }
    pub fn new_thrower(team: TeamType) -> PlayerStats {
        PlayerStats {
            str_: 3,
            ma: 6,
            ag: 3,
            av: 8,
            team,
            skills: HashSet::from_iter([Skill::SureHands, Skill::Throw]),
            role: PlayerRole::Thrower,
            pass: D6Target::TwoPlus,
        }
    }
    pub fn give_skill(&mut self, skill: Skill) {
        self.skills.insert(skill);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DugoutPlace {
    Reserves,
    Heated,
    KnockOut,
    Injuried,
    Ejected,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct DugoutPlayer {
    pub stats: PlayerStats,
    pub place: DugoutPlace,
    pub id: DugoutPlayerID,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct FieldedPlayer {
    pub id: PlayerID,
    pub stats: PlayerStats,
    pub position: Position,
    pub status: PlayerStatus,
    pub used: bool,
    pub moves: u8,
    pub used_skills: HashSet<Skill>,
}
impl FieldedPlayer {
    pub fn armor_target(&self) -> Sum2D6Target {
        Sum2D6Target::try_from(self.stats.av + 1).unwrap()
    }

    pub fn ag_target(&self) -> D6Target {
        D6Target::try_from(7 - self.stats.ag).unwrap()
    }

    pub fn pass_target(&self) -> D6Target {
        self.stats.pass
    }

    pub fn can_catch(&self) -> bool {
        match self.status {
            PlayerStatus::Up => true,
            PlayerStatus::Down => false,
            PlayerStatus::Stunned => false,
        }
    }
    pub fn has_tackle_zone(&self) -> bool {
        match self.status {
            PlayerStatus::Up => true,
            PlayerStatus::Down => false,
            PlayerStatus::Stunned => false,
        }
    }
    /// Returns how many normal moves the player has left. Before activating the player this is
    /// equal to MA (movement allowence)
    pub fn moves_left(&self) -> u8 {
        self.stats.ma.saturating_sub(self.moves)
    }
    /// Returns how many gfis the player has left. Before exhausting the normal moves,
    /// it's equal to 2
    pub fn gfis_left(&self) -> u8 {
        if self.moves <= self.stats.ma {
            2
        } else {
            2 + self.stats.ma - self.moves
        }
    }
    /// Ruturns the total number of mover the player has left, normal moves + gfis. Before
    /// activating the player, it's equal to MA + 2
    pub fn total_movement_left(&self) -> u8 {
        debug_assert!(self.moves <= self.stats.ma + 2);
        self.stats.ma + 2 - self.moves
    }
    pub fn add_move(&mut self, num_moves: u8) {
        assert!(self.total_movement_left() >= num_moves);
        self.moves += num_moves;
    }
    pub fn can_use_skill(&self, skill: Skill) -> bool {
        self.has_skill(skill) && !self.used_skills.contains(&skill)
    }
    pub fn has_skill(&self, skill: Skill) -> bool {
        self.stats.skills.contains(&skill)
    }
    pub fn use_skill(&mut self, skill: Skill) {
        let not_present_before = self.used_skills.insert(skill);
        debug_assert!(not_present_before);
    }
    pub fn reset_skills_and_moves(&mut self) {
        self.moves = 0;
        self.used = false;
        self.used_skills.clear();
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TeamState {
    pub bribes: u8,
    //babes: u8,
    //apothecaries: u8,
    //wizard_available: bool,
    //masterchef: bool,
    pub score: u8,
    //turn: u8,
    //rerolls_start: u8,
    pub rerolls: u8,
    //ass_coaches: u8,
    //cheerleaders: u8,
    //fame: u8,
    reroll_used: bool,
    //time_violation: u8,
}
impl TeamState {
    #[allow(clippy::new_without_default)]
    pub fn new() -> TeamState {
        TeamState {
            rerolls: 3,
            reroll_used: false,
            score: 0,
            bribes: 0,
        }
        //TeamState { bribes: 0, score: 0, turn: 0, rerolls_start: 3, rerolls: 3, fame: 3, reroll_used: false }
    }
    pub fn can_use_reroll(&self) -> bool {
        !self.reroll_used && self.rerolls > 0
    }
    pub fn use_reroll(&mut self) {
        assert!(self.can_use_reroll());
        self.reroll_used = true;
        self.rerolls -= 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TeamType {
    Home,
    Away,
}

pub fn other_team(team: TeamType) -> TeamType {
    match team {
        TeamType::Home => TeamType::Away,
        TeamType::Away => TeamType::Home,
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum BallState {
    OffPitch,
    OnGround(Position),
    Carried(PlayerID),
    InAir(Position),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Weather {
    Nice,
    Sunny,
    Rain,
    Blizzard,
    Sweltering,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SomeProcInput {
    Action(Action),
    Roll(RollResult),
}
impl From<Action> for SomeProcInput {
    fn from(action: Action) -> Self {
        SomeProcInput::Action(action)
    }
}
impl From<RollResult> for SomeProcInput {
    fn from(roll: RollResult) -> Self {
        SomeProcInput::Roll(roll)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcInput {
    Nothing,
    Action(Action),
    Roll(RollResult),
}
impl From<SomeProcInput> for ProcInput {
    fn from(input: SomeProcInput) -> Self {
        match input {
            SomeProcInput::Action(action) => ProcInput::Action(action),
            SomeProcInput::Roll(roll) => ProcInput::Roll(roll),
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub enum ProcState {
    DoneNewProcs(Vec<AnyProc>),
    NotDoneNewProcs(Vec<AnyProc>),
    NotDoneNew(AnyProc),
    DoneNew(AnyProc),
    Done,
    NotDone,
    NeedRoll(RequestedRoll),
    NeedAction(Box<AvailableActions>),
    /// Producer has already populated `game_state.available_actions` (and
    /// `path_buffer` if relevant) in-place during `step`. `micro_step`
    /// should transition to NeedAction without moving any data. Used by
    /// path-producing procedures (`MoveAction`, `BlockAction`) to avoid
    /// per-frame `Box<AvailableActions>` allocation.
    NeedActionInPlace,
}

//rename to something more descriptive
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MicroStepState {
    RunAgain,
    NeedAction,
    NeedRoll,
    GameOver,
}

pub trait Procedure: std::fmt::Debug {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState;
}
use smallvec::SmallVec;

pub type SmallVecPosAT = SmallVec<[PosAT; 4]>;

#[derive(Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AvailableActions {
    pub team: Option<TeamType>,
    simple: HashSet<SimpleAT>,
    positional: Option<FullPitch<SmallVecPosAT>>,
    // Whether `GameState::path_buffer` currently holds a valid set of path
    // offerings for this decision point. The actual `FullPitch` lives on
    // `GameState` so we can reuse the 4KB buffer across MoveAction frames
    // and keep clones cheap when no pathing is in flight. Cleared (with the
    // buffer's Arc payload) on every non-NeedAction transition in
    // `GameState::micro_step`.
    #[serde(default)]
    pub(crate) has_paths: bool,
}

impl std::fmt::Debug for AvailableActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut info = f.debug_struct("AvailableActions");
        if let Some(team) = self.team {
            info.field("team", &team);
        }
        if !self.simple.is_empty() {
            info.field("simple", &self.simple);
        }
        let mut pos_at_count: HashMap<PosAT, u16> = HashMap::new();
        if let Some(positional) = &self.positional {
            for pos_at in positional.iter().flat_map(|pos_ats| pos_ats.iter()) {
                pos_at_count
                    .entry(*pos_at)
                    .and_modify(|counter| *counter += 1)
                    .or_insert(1);
            }
        }
        if self.has_paths {
            info.field("has_paths", &true);
        }
        for (pos_at, count) in pos_at_count {
            let field_name = format!("{:?}", pos_at);
            info.field(&field_name, &count);
        }

        info.finish()
    }
}
impl AvailableActions {
    pub fn get_simple(&self) -> &HashSet<SimpleAT> {
        &self.simple
    }
    pub fn get_positional(&self) -> &Option<FullPitch<SmallVecPosAT>> {
        &self.positional
    }
    pub fn has_paths(&self) -> bool {
        self.has_paths
    }
    pub fn new_empty() -> Box<Self> {
        Box::default()
    }
    pub fn new(team: TeamType) -> Box<Self> {
        let mut aa = AvailableActions::new_empty();
        aa.team = Some(team);
        aa
    }
    pub fn is_empty(&self) -> bool {
        self.simple.is_empty() && !self.has_paths && self.positional.is_none()
    }
    /// Collects every simple action and every positional action backed by
    /// `self.positional`. Path-style actions are NOT included — call
    /// `GameState::get_all_actions` (the wrapper) to pick those up from
    /// `path_buffer`. The split exists because paths live on `GameState`
    /// now to allow per-game buffer reuse. No sort: every consumer is
    /// either index-pick (RandomBot), `.contains()` (ScriptedBot) or filter
    /// (MCTS), so deterministic ordering isn't needed.
    pub fn collect_non_path_actions(&self, out: &mut Vec<Action>) {
        if let Some(positional) = self.positional.as_ref() {
            for (pos, sv) in positional.iter_position() {
                for at in sv.iter() {
                    out.push(Action::Positional(*at, pos));
                }
            }
        }
        for at in self.simple.iter() {
            out.push(Action::Simple(*at));
        }
    }
    pub fn insert_simple(&mut self, action_type: SimpleAT) {
        assert!(self.team.is_some());
        self.simple.insert(action_type);
    }
    pub fn insert_positional(&mut self, action_type: PosAT, positions: Vec<Position>) {
        assert!(self.team.is_some());
        if positions.is_empty() {
            return;
        }
        if self.positional.is_none() {
            self.positional = Some(Default::default());
        }

        let self_positional = self.positional.as_mut().unwrap();
        positions.into_iter().for_each(|pos| {
            self_positional[pos].push(action_type);
        })
    }

    /// Returns true iff `action` is in `simple`/`positional`. Path-style
    /// positional actions are checked separately by `GameState::is_legal_action`
    /// using `path_buffer`.
    pub fn is_legal_non_path_action(&self, action: Action) -> bool {
        match action {
            Action::Simple(at) => self.simple.contains(&at),
            Action::Positional(at, pos) => {
                if let Some(allowed_at) = self.positional.as_ref().map(|positions| &positions[pos]) {
                    allowed_at.contains(&at)
                } else {
                    false
                }
            }
        }
    }
    pub fn get_team(&self) -> Option<TeamType> {
        self.team
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockActionChoice {
    // This will have all things needed in the Block procedure. Might as well merge them. Slightly funny code but it's ok!
    pub num_dices: NumBlockDices,
    pub position: Position,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InjuryOutcome {
    Stunned,
    KO,
    Casualty,
}

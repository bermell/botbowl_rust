use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::error;
use std::ops::{Add, Index, IndexMut, Mul, Sub, SubAssign};

use super::dices::{D6Target, Sum2D6Target};
use super::gamestate::GameState;
use super::pathing::Path;
use super::table::{NumBlockDices, PosAT, SimpleAT, Skill};
use crate::core::table;

pub type PlayerID = usize;
pub type Coord = i8;

pub struct FullPitch<T> {
    data: [[T; HEIGHT]; WIDTH],
}
impl<T> Index<Position> for FullPitch<T> {
    type Output = T;

    fn index(&self, index: Position) -> &Self::Output {
        let (x, y) = index.to_usize().unwrap();
        &self.data[x][y]
    }
}
impl<T> IndexMut<Position> for FullPitch<T> {
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        let (x, y) = index.to_usize().unwrap();
        &mut self.data[x][y]
    }
}

impl<T> FullPitch<T> {
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.data[x][y]
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|r| r.iter())
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
}

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 17;
pub const WIDTH_: Coord = WIDTH as Coord;
pub const HEIGHT_: Coord = HEIGHT as Coord;
pub const LINE_OF_SCRIMMAGE_HOME_X: Coord = 14;
pub const LINE_OF_SCRIMMAGE_AWAY_X: Coord = 13;
pub const LINE_OF_SCRIMMAGE_Y_RANGE: std::ops::RangeInclusive<Coord> = 5..=11;
pub const NORTH_WING_Y_RANGE: std::ops::RangeInclusive<Coord> = 1..=4;
pub const SOUTH_WING_Y_RANGE: std::ops::RangeInclusive<Coord> = 12..=15;

// Change the alias to `Box<error::Error>`.
pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Position {
    pub x: Coord,
    pub y: Coord,
}

impl Position {
    pub fn new(xy: (Coord, Coord)) -> Position {
        let (x, y) = xy;
        Position { x, y }
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Action {
    Positional(table::PosAT, Position),
    Simple(table::SimpleAT),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ActionChoice {
    Positional(Vec<Position>),
    Simple,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PlayerStatus {
    Up,
    Down,
    Stunned,
}

#[derive(Debug, Clone)]
pub struct PlayerStats {
    pub str_: u8,
    pub ma: u8,
    pub ag: u8,
    pub av: u8,
    pub team: TeamType,
    skills: HashSet<Skill>,
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
        }
    }
    pub fn give_skill(&mut self, skill: Skill) {
        self.skills.insert(skill);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DugoutPlace {
    Reserves,
    Heated,
    KnockOut,
    Injuried,
    Ejected,
}

pub struct DugoutPlayer {
    pub stats: PlayerStats,
    pub place: DugoutPlace,
    pub id: PlayerID,
}

#[derive(Debug)]
pub struct FieldedPlayer {
    pub id: PlayerID,
    pub stats: PlayerStats,
    pub position: Position,
    pub status: PlayerStatus,
    pub used: bool,
    pub moves: u8,
    pub used_skills: HashSet<Skill>,
    //bone_headed: bool
    //hypnotized: bool
    //really_stupid: bool
    //wild_animal: bool
    //taken_root: bool
    //blood_lust: bool
    //picked_up: bool
    //used_skills: Set[Skill]
    //used_dodge: bool,
    //used_catch: bool,
    //squares_moved: List['Square'] //might need this
    //has_blocked: bool
}
impl FieldedPlayer {
    pub fn armor_target(&self) -> Sum2D6Target {
        Sum2D6Target::try_from(self.stats.av + 1).unwrap()
    }

    pub fn ag_target(&self) -> D6Target {
        D6Target::try_from(7 - self.stats.ag).unwrap()
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
    pub fn moves_left(&self) -> u8 {
        if self.moves <= self.stats.ma {
            self.stats.ma - self.moves
        } else {
            0
        }
    }
    pub fn gfis_left(&self) -> u8 {
        if self.moves <= self.stats.ma {
            2
        } else {
            2 + self.stats.ma - self.moves
        }
    }
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

#[derive(Debug, Clone, Copy)]
pub struct TeamState {
    //bribes: u8,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BallState {
    OffPitch,
    OnGround(Position),
    Carried(PlayerID),
    InAir(Position),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weather {
    Nice,
    Sunny,
    Rain,
    Blizzard,
    Sweltering,
}

#[derive(Debug)]
pub enum ProcState {
    DoneNewProcs(Vec<Box<dyn Procedure>>),
    NotDoneNewProcs(Vec<Box<dyn Procedure>>),
    NotDoneNew(Box<dyn Procedure>),
    DoneNew(Box<dyn Procedure>),
    Done,
    NotDone,
    NeedAction(AvailableActions),
}

#[allow(unused_variables)]
pub trait Procedure: std::fmt::Debug {
    //fn start(&self, game_state: &GameState) {}
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState;
    //fn end(&self, game_state: &mut GameState) {}
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        AvailableActions::new_empty()
    }
}

#[derive(PartialEq, Eq)]
pub struct AvailableActions {
    pub team: Option<TeamType>,
    pub simple: HashSet<SimpleAT>,
    pub positional: HashMap<PosAT, Vec<Position>>,
    pub blocks: Vec<BlockActionChoice>,
}
impl std::fmt::Debug for AvailableActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut info = f.debug_struct("Point");
        if let Some(team) = self.team {
            info.field("team:", &team);
        }
        if !self.simple.is_empty() {
            info.field("simple", &self.simple);
        }
        if !self.positional.is_empty() {
            info.field("positional", &self.positional.keys());
        }
        if !self.blocks.is_empty() {
            info.field("blocks", &self.blocks);
        }

        info.finish()
    }
}
impl AvailableActions {
    pub fn new_empty() -> Self {
        AvailableActions {
            team: None,
            simple: HashSet::new(),
            positional: HashMap::new(),
            blocks: Vec::new(),
        }
    }
    pub fn new(team: TeamType) -> Self {
        AvailableActions {
            team: Some(team),
            simple: HashSet::new(),
            positional: HashMap::new(),
            blocks: Vec::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.simple.is_empty()
            && !self
                .positional
                .iter()
                .any(|(_, positions)| !positions.is_empty())
    }
    pub fn insert_simple(&mut self, action_type: SimpleAT) {
        assert!(self.team.is_some());
        self.simple.insert(action_type);
    }
    pub fn insert_path(&mut self, path: &Path) {
        assert!(self.team.is_some());
        if let Some(num_dices) = path.block_dice {
            self.blocks.push(BlockActionChoice {
                num_dices,
                position: path.target,
            })
        } else {
            self.insert_single_positional(path.action_type, path.target);
        }
    }

    pub fn insert_positional(&mut self, action_type: PosAT, positions: Vec<Position>) {
        assert!(self.team.is_some());
        assert!(!self.positional.contains_key(&action_type));
        self.positional.insert(action_type, positions);
    }
    pub fn insert_single_positional(&mut self, action_type: PosAT, position: Position) {
        match self.positional.entry(action_type) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(vec![position]);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.get_mut().push(position);
            }
        };
    }
    pub fn insert_block(&mut self, ac: Vec<BlockActionChoice>) {
        if self.blocks.is_empty() {
            self.blocks = ac;
        } else {
            self.blocks.extend(ac);
        }
    }

    pub fn is_legal_action(&self, action: Action) -> bool {
        match action {
            Action::Positional(PosAT::Block, position) => {
                self.blocks.iter().any(|ac| ac.position == position)
            }
            Action::Positional(at, pos) => match self.positional.get(&at) {
                Some(legal_positions) => legal_positions.contains(&pos),
                None => false,
            },
            Action::Simple(at) => self.simple.contains(&at),
        }
    }
    pub fn get_team(&self) -> Option<TeamType> {
        self.team
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockActionChoice {
    // This will have all things needed in the Block procedure. Might as well merge them. Slightly funny code but it's ok!
    pub num_dices: NumBlockDices,
    pub position: Position,
}

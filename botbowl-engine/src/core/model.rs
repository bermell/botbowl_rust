use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{error, fmt};

use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, RangeInclusive, Sub, SubAssign};

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

/// Runtime-active board dimensions (plan 017 → runtime tiers). The compile-time
/// consts above (`WIDTH`/`HEIGHT`/`TEAM_SIZE`/`ROSTER_PER_TEAM`) are the physical
/// **capacity** — the largest board this binary can run, and the fixed size of
/// `FullPitch` / the roster arrays. `BoardDims` is the smaller **logical** board
/// actually in play, inset into the low corner of the full-size arrays; cells
/// beyond it are permanent no-man's-land (always out of bounds, never occupied).
///
/// All gameplay geometry (out-of-bounds, team side, end zones, line-of-scrimmage
/// and wing ranges, kickoff aim, throw-in/deviate caps, kickoff-table gating) is
/// derived here at runtime — the formulas mirror `build.rs`. Physical/array-bound
/// checks keep using the capacity consts (`Position::is_out`, `FullPitch` index).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoardDims {
    /// Engine width incl. the 1-cell out-of-bounds border; `<= WIDTH`.
    pub width: Coord,
    /// Engine height incl. the 1-cell out-of-bounds border; `<= HEIGHT`.
    pub height: Coord,
    /// Players fielded per team; `<= TEAM_SIZE`.
    pub team_size: usize,
}

impl Default for BoardDims {
    /// The full compiled board — makes a default build byte-identical to the
    /// pre-runtime engine (existing tests unaffected when no env vars are set).
    fn default() -> Self {
        BoardDims {
            width: WIDTH_,
            height: HEIGHT_,
            team_size: TEAM_SIZE,
        }
    }
}

impl BoardDims {
    /// `width`/`height` are engine dimensions (playable + 2 border); validates
    /// the same rules as `build.rs` and that nothing exceeds the compiled
    /// capacity. Panics on violation.
    ///
    /// **Width must stay even.** The whole codebase mirrors a state with
    /// `x -> width - 1 - x` (`GameState::mirrored`, `AvailableActions::mirrored`,
    /// the NN's perspective canonicalisation). That map only exchanges the two
    /// LOS columns — and therefore the two halves — when the width is even; on
    /// an odd width `los_home_x` is a fixed point inside Home's own half, so a
    /// mirrored state puts Away players on Home's side and the halves differ by
    /// a column. Height carries no such constraint: `bands()` keeps the wings
    /// equal at any height, so odd and even both mirror cleanly about the
    /// centre row.
    pub fn new(width: Coord, height: Coord, team_size: usize) -> BoardDims {
        match BoardDims::try_new(width, height, team_size) {
            Ok(d) => d,
            Err(e) => panic!("{e}"),
        }
    }

    /// [`BoardDims::new`] without the panic: the same rules, as an `Err`
    /// message, for callers that enumerate or parse candidate boards (the
    /// plan-042 size sampler) and want to reject a bad one before it costs a
    /// game.
    pub fn try_new(width: Coord, height: Coord, team_size: usize) -> std::result::Result<BoardDims, String> {
        let pw = width - 2; // playable width
        let ph = height - 2; // playable height
        if !(pw >= 8 && pw % 2 == 0) {
            return Err(format!("playable width must be even and >= 8, got {pw}"));
        }
        if ph < 3 {
            return Err(format!("playable height must be >= 3, got {ph}"));
        }
        if team_size < 1 {
            return Err(format!("team_size must be >= 1, got {team_size}"));
        }
        if !((width as usize) <= WIDTH && (height as usize) <= HEIGHT && team_size <= TEAM_SIZE) {
            return Err(format!(
                "board {width}x{height}/{team_size} exceeds compiled capacity {WIDTH}x{HEIGHT}/{TEAM_SIZE} \
                 — recompile with larger BOARD_SIZE_W/BOARD_SIZE_H/BOARD_PLAYERS"
            ));
        }
        let dims = BoardDims {
            width,
            height,
            team_size,
        };
        // The bands must tile the playable rows exactly, with equal wings. A
        // wing may legitimately be empty on a short board; the LOS may not.
        let (los, wing) = dims.bands();
        assert!(
            los >= 1 && wing >= 0 && los + 2 * wing == ph,
            "bands {los}+2x{wing} != {ph}"
        );
        assert_eq!(*dims.los_y_range().start(), wing + 1);
        assert_eq!(*dims.los_y_range().end(), height - 2 - wing);
        assert_eq!(dims.north_wing_y_range().count(), wing as usize);
        assert_eq!(dims.south_wing_y_range().count(), wing as usize);
        Ok(dims)
    }

    /// Read the active board from `BOARD_SIZE_W`/`BOARD_SIZE_H`/`BOARD_PLAYERS`
    /// (env carries *playable* dims; we add the 2-cell border). Any var left
    /// unset keeps the compiled default for that axis.
    pub fn from_env() -> BoardDims {
        fn parse(name: &str) -> Option<usize> {
            match std::env::var(name) {
                Ok(v) => Some(
                    v.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name}='{v}' is not a usize")),
                ),
                Err(_) => None,
            }
        }
        let d = BoardDims::default();
        let width = parse("BOARD_SIZE_W").map_or(d.width, |pw| (pw + 2) as Coord);
        let height = parse("BOARD_SIZE_H").map_or(d.height, |ph| (ph + 2) as Coord);
        let team_size = parse("BOARD_PLAYERS").unwrap_or(d.team_size);
        BoardDims::new(width, height, team_size)
    }

    /// Split the playable rows into `wing | LOS | wing`, wings always equal so
    /// the board stays mirror-symmetric about its centre row. Returns
    /// `(los_rows, wing_rows)`; `los_rows + 2 * wing_rows == playable height`.
    ///
    /// As the board gets shorter the bands shrink in a fixed priority order —
    /// wings 4→1 first, then the LOS 7→5, then wings →0 — so the contact line
    /// (which has to hold three players) survives longest. The LOS is capped at
    /// 7 rows, matched to the playable height's parity so the wings divide
    /// evenly; on an even height the cap is 6. Reproduces the historical
    /// 28x17 bands exactly (LOS 7, wings 4).
    fn bands(&self) -> (Coord, Coord) {
        let ph = self.height - 2; // playable rows
        let cap = if ph % 2 == 1 { 7 } else { 6 };
        let mut wing = ((ph - cap) / 2).max(0);
        // A one-row wing is worth more than the last two LOS rows, but only
        // while the LOS can still stay at 5.
        if wing == 0 && ph - 2 >= 5 {
            wing = 1;
        }
        (ph - 2 * wing, wing)
    }
    /// Rows in the line-of-scrimmage band. Always >= 1.
    pub fn los_rows(&self) -> Coord {
        self.bands().0
    }
    /// Rows in *each* wing; 0 on boards too short to spare any.
    pub fn wing_rows(&self) -> Coord {
        self.bands().1
    }
    /// Players a team may set up in one wing: half the wing's rows, rounded up
    /// (2 on the full pitch's 4-row wings). 0 when there are no wing rows.
    pub fn max_players_per_wing(&self) -> usize {
        ((self.wing_rows() + 1) / 2) as usize
    }
    /// Players a team must put on the line of scrimmage — three, or the whole
    /// band when it is narrower than that.
    pub fn min_players_on_los(&self) -> usize {
        (self.los_rows().min(3)) as usize
    }
    pub fn los_home_x(&self) -> Coord {
        self.width / 2
    }
    pub fn los_away_x(&self) -> Coord {
        self.width / 2 - 1
    }
    pub fn los_x(&self, team: TeamType) -> Coord {
        match team {
            TeamType::Home => self.los_home_x(),
            TeamType::Away => self.los_away_x(),
        }
    }
    pub fn los_y_range(&self) -> RangeInclusive<Coord> {
        let (los, wing) = self.bands();
        (wing + 1)..=(wing + los)
    }
    /// Empty when the board is too short for wings — callers must treat an
    /// empty range as "this board has no wing", not as row 1.
    pub fn north_wing_y_range(&self) -> RangeInclusive<Coord> {
        1..=self.wing_rows()
    }
    /// Empty when the board is too short for wings (see `north_wing_y_range`).
    pub fn south_wing_y_range(&self) -> RangeInclusive<Coord> {
        let (los, wing) = self.bands();
        (wing + los + 1)..=(self.height - 2)
    }
    pub fn endzone_x(&self, team: TeamType) -> Coord {
        match team {
            TeamType::Home => 1,
            TeamType::Away => self.width - 2,
        }
    }
    /// LOS-to-endzone distance is `width/2 - 1` for either team (the pitch is
    /// symmetric), so this is team-independent.
    pub fn los_to_endzone_distance(&self) -> Coord {
        self.width / 2 - 1
    }
    /// The greatest MA a player can have and still be unable to reach the
    /// opponent's endzone from a standing start on their own LOS in one turn
    /// — even with the two GFI squares this engine allows beyond MA
    /// (`FieldedPlayer::total_movement_left` is `ma + 2`). Not applied to the
    /// stock roster by the engine itself; callers that want it opt in (eval
    /// games, to force a multi-turn advance instead of a reliable one-turn
    /// score on a narrow board — see `botbowl-play::eval`). No-op ceiling on
    /// the full pitch, where it already exceeds every stock role's MA.
    pub fn ma_cap(&self) -> Coord {
        (self.los_to_endzone_distance() - 3).max(0)
    }
    /// Kickoff scatter/deviate & throw-in distances are capped here so the ball
    /// can't be flung clear across a narrow board.
    pub fn max_scatter(&self) -> Coord {
        self.width / 2
    }
    /// Divides the raw kickoff-deviate (D6) and throw-in (2D6) roll down on a
    /// narrow board, so a kickoff aimed at the middle — or a throw-in back
    /// onto the pitch — rarely scatters out of bounds. The dice themselves
    /// (D6/D8 for deviate, 2D6/D3 for throw-in) are unchanged; only how far
    /// the roll carries the ball is scaled down. No-op (divisor 1) once the
    /// narrower playable axis (excluding the 2-cell OOB border) is at least
    /// as wide as the largest roll it scales, 2D6 = 12.
    pub fn scatter_divisor(&self) -> Coord {
        let axis = (self.width.min(self.height) - 2).max(1);
        (12 + axis - 1) / axis
    }
    pub fn kickoff_table_enabled(&self) -> bool {
        self.team_size >= 7
    }
    pub fn roster_per_team(&self) -> usize {
        self.team_size + 1
    }
    /// Out of bounds of the *logical* board (not the physical array).
    pub fn is_out(&self, pos: Position) -> bool {
        pos.x <= 0 || pos.x >= self.width - 1 || pos.y <= 0 || pos.y >= self.height - 1
    }
    pub fn is_on_team_side(&self, pos: Position, team: TeamType) -> bool {
        match team {
            TeamType::Home => pos.x >= self.width / 2,
            TeamType::Away => pos.x < self.width / 2,
        }
    }
}

/// Early-return from a `#[test]` (returning `()`) when the build's board is
/// smaller than the given playable-ish engine dimensions. Used to skip tests
/// that pin full-pitch behavior (specific LOS/wing rules, long pathing lanes,
/// hand-tuned kickoff scatter) when the engine is compiled for a smaller tier.
#[macro_export]
macro_rules! skip_if_board_smaller_than {
    ($w:expr, $h:expr) => {
        if ($crate::core::model::WIDTH as usize) < $w || ($crate::core::model::HEIGHT as usize) < $h {
            eprintln!(
                "skipping {}: board {}x{} smaller than required {}x{}",
                module_path!(),
                $crate::core::model::WIDTH,
                $crate::core::model::HEIGHT,
                $w,
                $h
            );
            return;
        }
    };
}

// Change the alias to `Box<error::Error>`.
pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
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
/// `ALL_DIRECTIONS` reflected in x. Used by the pathfinder for players
/// attacking the low-x endzone — see [`Direction::all_directions_toward`].
const ALL_DIRECTIONS_MIRRORED: [Direction; 8] = [
    Direction { dx: -1, dy: 1 },
    Direction { dx: 0, dy: 1 },
    Direction { dx: 1, dy: 1 },
    Direction { dx: -1, dy: 0 },
    Direction { dx: 1, dy: 0 },
    Direction { dx: -1, dy: -1 },
    Direction { dx: 0, dy: -1 },
    Direction { dx: 1, dy: -1 },
];

impl Direction {
    pub fn all_directions_iter() -> impl Iterator<Item = &'static Direction> {
        ALL_DIRECTIONS.iter()
    }

    /// The eight directions, ordered so that steps toward `attacking_dx`
    /// come first. `ALL_DIRECTIONS` hard-codes a preference for `dx = +1`
    /// (every `+1` entry precedes its `-1` partner), and reflecting the list
    /// in x gives a *permutation* of itself rather than itself. Anywhere
    /// that order breaks a tie — the pathfinder's route choice is the live
    /// case, see `pathing.rs::expand_node` — that preference is an absolute
    /// board-coordinate bias, which means opposite treatment for the two
    /// teams, since Home attacks x=1 and Away attacks x=width-2. Selecting
    /// the order by the mover's own attacking direction makes such a
    /// tie-break mirror-covariant instead.
    pub fn all_directions_toward(attacking_dx: Coord) -> &'static [Direction; 8] {
        if attacking_dx < 0 {
            &ALL_DIRECTIONS_MIRRORED
        } else {
            &ALL_DIRECTIONS
        }
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
    /// Out of bounds of the *physical* pitch array (the compiled capacity, not
    /// the runtime-active board). Use this only for array-bound sanity checks;
    /// gameplay out-of-bounds goes through `BoardDims::is_out` / `GameState::is_out`.
    pub fn is_out(&self) -> bool {
        self.x <= 0 || self.x >= WIDTH_ - 1 || self.y <= 0 || self.y >= HEIGHT_ - 1
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

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ActionChoice {
    Positional(Vec<Position>),
    Simple,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
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
    pub pass: D6Target,
    pub av: u8,
    pub team: TeamType,
    pub skills: HashSet<Skill>,
    pub role: PlayerRole,
    //skills: [Option<table::Skill>; 3],
    //injuries
    //spp
}
impl PlayerStats {
    /// Inclusive caps on the characteristics. They bound what any roster or
    /// generator may produce, and the NN encoder divides by them so every
    /// per-player feature plane lands in `[0, 1]` — which is what makes the
    /// side-factored encoding exactly recoverable (`botbowl-nn/src/encode.rs`).
    /// Raise one and the encoder's normalisers follow automatically.
    pub const MAX_ST: u8 = 8;
    pub const MAX_MA: u8 = 10;
    /// AG is a D6 target number, so 6 is the worst possible and also the cap.
    pub const MAX_AG: u8 = 6;
    pub const MAX_AV: u8 = 12;
    /// `FieldedPlayer::total_movement_left` is `ma + 2` before the player
    /// moves — the two Go-For-It steps are movement the plane has to hold.
    pub const MAX_MOVEMENT: u8 = Self::MAX_MA + 2;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum DugoutPlace {
    Reserves,
    Heated,
    KnockOut,
    Injuried,
    Ejected,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Hash)]
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

/// Order-independent hash of a set.
///
/// `HashSet` has no `Hash` impl because its iteration order is not stable, but set *equality* is
/// order-independent — so the hash has to be too, or two equal states could hash differently and
/// split the MCTS DAG. Summing per-element hashes is commutative, which is exactly the property
/// needed; the length is mixed in so `{}` and a set of hash-zero elements stay distinguishable.
pub(crate) fn hash_set_unordered<T: std::hash::Hash, H: std::hash::Hasher>(set: &HashSet<T>, h: &mut H) {
    use std::hash::Hash as _;
    let mut acc: u64 = 0;
    for item in set {
        let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
        item.hash(&mut item_hasher);
        acc = acc.wrapping_add(std::hash::Hasher::finish(&item_hasher));
    }
    set.len().hash(h);
    acc.hash(h);
}

impl std::hash::Hash for PlayerStats {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.str_.hash(h);
        self.ma.hash(h);
        self.ag.hash(h);
        self.pass.hash(h);
        self.av.hash(h);
        self.team.hash(h);
        hash_set_unordered(&self.skills, h);
        self.role.hash(h);
    }
}

impl std::hash::Hash for FieldedPlayer {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.id.hash(h);
        self.stats.hash(h);
        self.position.hash(h);
        self.status.hash(h);
        self.used.hash(h);
        self.moves.hash(h);
        // A Dodge or Block skill already spent this activation is a different situation from one
        // still in hand, and `PartialEq` agrees — so it has to move the hash.
        hash_set_unordered(&self.used_skills, h);
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
    pub fn reset_reroll_used(&mut self) {
        self.reroll_used = false;
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
pub enum BallState {
    OffPitch,
    OnGround(Position),
    Carried(PlayerID),
    InAir(Position),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub enum Weather {
    Nice,
    Sunny,
    Rain,
    Blizzard,
    Sweltering,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
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
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
impl std::hash::Hash for AvailableActions {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.team.hash(h);
        hash_set_unordered(&self.simple, h);
        self.positional.hash(h);
        self.has_paths.hash(h);
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
    /// Companion to [`crate::core::gamestate::GameState::mirrored`]:
    /// reflect every positional offering about `x -> width-1-x` and hand
    /// the decision to the other team. Simple actions carry no
    /// coordinates, so they survive unchanged.
    ///
    /// Path offerings are dropped (`has_paths` cleared) — their `Node`
    /// chains live on `GameState` and carry their own positions. See the
    /// scope note on `GameState::mirrored`.
    pub fn mirrored(&self, width: Coord) -> AvailableActions {
        let positional = self.positional.as_ref().map(|src| {
            let mut dst: FullPitch<SmallVecPosAT> = Default::default();
            for (pos, sv) in src.iter_position() {
                if sv.is_empty() {
                    continue;
                }
                dst[Position::new((width - 1 - pos.x, pos.y))] = sv.clone();
            }
            dst
        });
        AvailableActions {
            team: self.team.map(other_team),
            simple: self.simple.clone(),
            positional,
            has_paths: false,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct BlockActionChoice {
    // This will have all things needed in the Block procedure. Might as well merge them. Slightly funny code but it's ok!
    pub num_dices: NumBlockDices,
    pub position: Position,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InjuryOutcome {
    Stunned,
    KO,
    Casualty,
}

#[cfg(test)]
mod board_dims_tests {
    use super::{BoardDims, Coord, HEIGHT, WIDTH};

    /// `(playable height, LOS rows, wing rows)`. The bands shrink in a fixed
    /// priority order as the board gets shorter — wings 4→1, then the LOS 7→5,
    /// then wings →0 — so the contact line outlives the flanks.
    const DESCENT: [(Coord, Coord, Coord); 13] = [
        (15, 7, 4), // the historical full pitch
        (14, 6, 4),
        (13, 7, 3),
        (12, 6, 3),
        (11, 7, 2),
        (10, 6, 2),
        (9, 7, 1),
        (8, 6, 1),
        (7, 5, 1),
        (6, 6, 0),
        (5, 5, 0),
        (4, 4, 0),
        (3, 3, 0),
    ];

    fn dims_for(ph: Coord) -> Option<BoardDims> {
        let (w, h) = (10, ph + 2);
        ((w as usize) <= WIDTH && (h as usize) <= HEIGHT).then(|| BoardDims::new(w, h, 1))
    }

    #[test]
    fn bands_follow_the_descent_order_and_always_tile_the_pitch() {
        for (ph, los, wing) in DESCENT {
            let Some(dims) = dims_for(ph) else { continue };
            assert_eq!(
                (dims.los_rows(), dims.wing_rows()),
                (los, wing),
                "playable height {ph} should split {los} + 2x{wing}"
            );
            // Equal wings, and the three bands tile the playable rows exactly.
            assert_eq!(los + 2 * wing, ph);
            assert_eq!(dims.north_wing_y_range().count(), wing as usize);
            assert_eq!(dims.south_wing_y_range().count(), wing as usize);
            assert_eq!(dims.los_y_range().count(), los as usize);
            let covered: Vec<Coord> = dims
                .north_wing_y_range()
                .chain(dims.los_y_range())
                .chain(dims.south_wing_y_range())
                .collect();
            assert_eq!(covered, (1..=ph).collect::<Vec<_>>(), "bands must tile rows 1..={ph}");
        }
    }

    /// The bands are symmetric about the centre row, so a state mirrored in y
    /// still has its LOS band and wings where the rules put them.
    #[test]
    fn bands_are_symmetric_in_y_at_every_height() {
        for (ph, ..) in DESCENT {
            let Some(dims) = dims_for(ph) else { continue };
            let flip = |y: Coord| dims.height - 1 - y;
            assert_eq!(
                flip(*dims.los_y_range().end()),
                *dims.los_y_range().start(),
                "LOS band not y-symmetric at playable height {ph}"
            );
            let north: Vec<Coord> = dims.north_wing_y_range().map(flip).collect();
            let south: Vec<Coord> = dims.south_wing_y_range().rev().collect();
            assert_eq!(north, south, "wings not y-mirrors at playable height {ph}");
        }
    }

    /// Setup caps derive from the band sizes, and reproduce the full pitch's
    /// historical "three on the line, two per wing".
    #[test]
    fn setup_caps_scale_with_the_bands() {
        for (ph, los, wing) in DESCENT {
            let Some(dims) = dims_for(ph) else { continue };
            assert_eq!(dims.max_players_per_wing(), ((wing + 1) / 2) as usize);
            assert_eq!(dims.min_players_on_los(), los.min(3) as usize);
            // A board can always satisfy its own line requirement.
            assert!(dims.min_players_on_los() <= los as usize);
        }
    }

    #[test]
    #[should_panic(expected = "must be even")]
    fn odd_width_is_rejected() {
        BoardDims::new(11, 9, 2);
    }

    /// No-op on the full pitch; on a narrow board it shrinks the roll enough
    /// that the largest kickoff-deviate/throw-in roll (2D6 = 12) fits inside
    /// the narrower playable axis from a centred aim.
    #[test]
    fn scatter_divisor_is_a_noop_on_full_pitch_and_shrinks_narrow_boards() {
        assert_eq!(BoardDims::default().scatter_divisor(), 1, "full pitch must be a no-op");

        // 16x9 engine (14x7 playable, plan 042's small tier): narrow axis 7.
        if let Ok(dims) = BoardDims::try_new(16, 9, 3) {
            assert_eq!(dims.scatter_divisor(), 2);
            assert!(12 / dims.scatter_divisor() <= 7);
        }
        // 14x7 engine (12x5 playable): narrow axis 5.
        if let Ok(dims) = BoardDims::try_new(14, 7, 3) {
            assert_eq!(dims.scatter_divisor(), 3);
            assert!(12 / dims.scatter_divisor() <= 5);
        }
    }

    /// `ma_cap` must guarantee a player can't reach the endzone from a
    /// standing LOS start in one turn, on every board the compiled capacity
    /// supports: `ma_cap() + 2` (the engine's GFI ceiling) always falls
    /// short of `los_to_endzone_distance()`.
    #[test]
    fn ma_cap_always_falls_short_of_the_endzone() {
        assert_eq!(BoardDims::default().ma_cap(), 10, "full pitch must be a no-op (exceeds every stock MA)");

        for (w, h, players) in [(16, 9, 3), (14, 7, 3), (12, 5, 1), (10, 5, 1), (8, 5, 1)] {
            let Ok(dims) = BoardDims::try_new(w, h, players) else { continue };
            assert!(
                dims.ma_cap() + 2 < dims.los_to_endzone_distance(),
                "{w}x{h}: ma_cap {} + 2 GFI must fall short of the {}-square LOS-to-endzone distance",
                dims.ma_cap(),
                dims.los_to_endzone_distance(),
            );
        }
    }

    /// `try_new` is `new` as a `Result`: the same rules, the same message,
    /// no unwinding — so a size sampler can enumerate candidates.
    #[test]
    fn try_new_reports_every_rule_new_panics_on() {
        assert!(BoardDims::try_new(11, 9, 2).unwrap_err().contains("must be even"));
        assert!(BoardDims::try_new(8, 9, 2).unwrap_err().contains(">= 8"));
        assert!(BoardDims::try_new(10, 4, 2).unwrap_err().contains(">= 3"));
        assert!(BoardDims::try_new(10, 5, 0).unwrap_err().contains("team_size"));
        assert!(BoardDims::try_new(WIDTH as Coord + 2, 5, 1)
            .unwrap_err()
            .contains("exceeds compiled capacity"));
        assert_eq!(BoardDims::try_new(10, 5, 1).unwrap(), BoardDims::new(10, 5, 1));
    }
}

#[cfg(test)]
mod player_stats_tests {
    use super::PlayerStats;
    use crate::core::model::TeamType;

    /// The stock roster must fit under the caps the NN encoder normalises by;
    /// a roster over a cap would encode as a feature plane above 1.0 and
    /// silently break the side-factored encoding's exact recovery.
    #[test]
    fn stock_rosters_respect_the_characteristic_caps() {
        for team in [TeamType::Home, TeamType::Away] {
            for stats in [
                PlayerStats::new_lineman(team),
                PlayerStats::new_blitzer(team),
                PlayerStats::new_catcher(team),
                PlayerStats::new_thrower(team),
            ] {
                assert!(stats.str_ <= PlayerStats::MAX_ST, "{:?} ST {}", stats.role, stats.str_);
                assert!(stats.ma <= PlayerStats::MAX_MA, "{:?} MA {}", stats.role, stats.ma);
                assert!(stats.ag <= PlayerStats::MAX_AG, "{:?} AG {}", stats.role, stats.ag);
                assert!(stats.av <= PlayerStats::MAX_AV, "{:?} AV {}", stats.role, stats.av);
            }
        }
    }
}

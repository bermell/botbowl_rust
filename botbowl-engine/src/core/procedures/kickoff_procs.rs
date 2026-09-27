use crate::core::model::ProcInput;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, BoardDims, Coord, Direction, DugoutPlace, DugoutPlayerID,
    PlayerID, Position, ProcState, Procedure, TeamType, Weather,
};
use crate::core::procedures::ball_procs;
use crate::core::table::*;

use crate::core::gamestate::GameState;

use super::AnyProc;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Kickoff {
    aim: Position,
}
impl Kickoff {
    pub fn new() -> AnyProc {
        AnyProc::Kickoff(Kickoff {
            aim: Position::new((0, 0)),
        })
    }
}
impl Procedure for Kickoff {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let (len_roll, dir_roll) = match input {
            ProcInput::Nothing => {
                let mut aa = AvailableActions::new(game_state.info.kicking_this_drive);
                aa.insert_simple(SimpleAT::KickoffAimMiddle);
                return ProcState::NeedAction(aa);
            }
            ProcInput::Action(Action::Simple(SimpleAT::KickoffAimMiddle)) => {
                self.aim = game_state.get_best_kickoff_aim_for(game_state.info.kicking_this_drive);
                return ProcState::NeedRoll(RequestedRoll::Deviate);
            }
            ProcInput::Roll(RollResult::Deviate(len_roll, dir_roll)) => (len_roll, dir_roll),
            _ => panic!("Unexpected input {:?}", input),
        };

        // Scale the roll down on narrow tiers (no-op on the full pitch) so an
        // aim-middle kickoff rarely deviates out of bounds, then cap at half
        // the board width as a final safety net.
        let dims = game_state.board_dims;
        let len = ((len_roll as Coord) / dims.scatter_divisor()).min(dims.max_scatter());
        let ball_pos = self.aim + Direction::from(dir_roll) * len;
        game_state.set_ball(BallState::InAir(ball_pos));
        if game_state.board_dims.kickoff_table_enabled() {
            ProcState::DoneNew(KickoffTable::new())
        } else {
            // Degenerate kickoff for small tiers: ball just lands, no event roll.
            ProcState::DoneNew(LandKickoff::new())
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct KickoffTable {}
impl KickoffTable {
    pub fn new() -> AnyProc {
        AnyProc::KickoffTable(KickoffTable {})
    }
}
impl Procedure for KickoffTable {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let kickoff_roll = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::Sum2D6);
            }
            ProcInput::Roll(RollResult::Sum2D6(kickoff_roll)) => kickoff_roll,
            _ => panic!("Unexpected input {:?}", input),
        };
        let mut procs: Vec<AnyProc> = vec![LandKickoff::new()]; //TODO: this should be added
                                                                //by the kickoff procedure
        match kickoff_roll {
            Sum2D6::Two => {
                //get the ref
                game_state.home.bribes += 1;
                game_state.away.bribes += 1;
            }
            Sum2D6::Three => {
                //Timeout
                if game_state.info.home_turn <= 5 {
                    game_state.info.away_turn += 1;
                    game_state.info.home_turn += 1;
                } else {
                    game_state.info.away_turn -= 1;
                    game_state.info.home_turn -= 1;
                }
            }
            Sum2D6::Four => {
                //solid defense
            }
            Sum2D6::Five => {
                //High Kick
            }
            Sum2D6::Six => {
                //Cheering fans
            }
            Sum2D6::Seven => {
                //Brilliant coaching
            }
            Sum2D6::Eight => {
                procs.push(ChangingWeather::new());
            }
            Sum2D6::Nine => {
                //Quick snap
            }
            Sum2D6::Ten => {
                //Blitz!
            }
            Sum2D6::Eleven => {
                //Officious ref
            }
            Sum2D6::Twelve => {
                //Pitch invasion
            }
        }

        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ChangingWeather {}
impl ChangingWeather {
    pub fn new() -> AnyProc {
        AnyProc::ChangingWeather(ChangingWeather {})
    }
}
impl Procedure for ChangingWeather {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::Sum2D6),
            ProcInput::Roll(RollResult::Sum2D6(roll)) => {
                game_state.info.weather = Weather::from(roll);
                let ball_pos = game_state.get_ball_position().unwrap();
                if game_state.info.weather == Weather::Nice && !game_state.is_out(ball_pos) {
                    ProcState::NeedRoll(RequestedRoll::D8)
                } else {
                    ProcState::Done
                }
            }
            ProcInput::Roll(RollResult::D8(d8)) => {
                let scattered = game_state.get_ball_position().unwrap() + Direction::from(d8);
                game_state.set_ball(BallState::InAir(scattered));
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LandKickoff {}
impl LandKickoff {
    pub fn new() -> AnyProc {
        AnyProc::LandKickoff(LandKickoff {})
    }
}
impl Procedure for LandKickoff {
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
        let BallState::InAir(ball_position) = game_state.ball else {
            unreachable!()
        };

        if game_state.is_out(ball_position)
            || !game_state.is_on_team_side(ball_position, other_team(game_state.info.kicking_this_drive))
        {
            return ProcState::DoneNew(ball_procs::Touchback::new());
        }

        match game_state.get_player_id_at(ball_position) {
            Some(id) => ProcState::DoneNew(ball_procs::Catch::new_with_kick_arg(
                id,
                game_state.get_catch_target(id).unwrap(),
                true,
            )),
            None => ProcState::DoneNew(ball_procs::Bounce::new_with_kick_arg(true)),
        }
    }
}
/// Where a formation slot sits along the y axis. Resolved against the *active*
/// board, never against hard-coded offsets, so a formation means the same thing
/// on every board size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Row {
    /// `i`-th row of the line-of-scrimmage band, counted outwards from the
    /// centre (0 = centre, 1 = one south, 2 = one north, …). Yields nothing
    /// when the band is narrower than `i` — that slot is simply skipped, which
    /// is what keeps the front rank inside `los_y_range` on every board (and
    /// therefore `is_setup_legal`).
    Los(usize),
    /// `k` rows from the centre, clamped onto the pitch.
    Off(Coord),
    /// Outermost row of the north (-1) / south (+1) wing.
    Wing(Coord),
}

/// One fielding slot: the role we'd like there, how many squares back from our
/// own line of scrimmage, and which row.
type Slot = (PlayerRole, Coord, Row);

/// Pre-configured setups, used as *planners*: the engine's own setup is one
/// `PlacePlayer`/`BenchPlayer` decision per player (see [`Setup`]), and a
/// formation answers those decisions one at a time ([`Formation::next_action`])
/// for the scripted bot, the MCTS opponent model and test scaffolding
/// ([`auto_setup`]). Each is a pure function of `(BoardDims, TeamType)`; slots
/// are listed in fielding priority order, so a team smaller than the formation
/// fields the front of the list and the rest sit out. Every formation opens
/// with three line-of-scrimmage slots, so any of them is legal
/// (`GameState::is_setup_legal`) on any board whose LOS band is three rows
/// wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Formation {
    /// The historical auto-setup: everything on the line, catchers just behind
    /// it, throwers deep. Reproduces the pre-`Formation` layout exactly on the
    /// full pitch.
    Line,
    /// Wide defence: sentries on both wings, a thin line, deep safeties.
    Spread,
    /// Receiving wedge: a cage around a deep receiver, wide outlets.
    Wedge,
    /// Deep defensive screen: a second rank behind the line and a lone safety.
    Zone,
}

impl Formation {
    pub const ALL: [Formation; 4] = [Formation::Line, Formation::Spread, Formation::Wedge, Formation::Zone];

    /// Board requirements. The clamping below would keep an oversized formation
    /// on-pitch, but it would collapse it onto its neighbours and stop being
    /// the formation it claims to be — so the tighter shapes are simply not
    /// offered on boards that can't hold them. `Line` is always available.
    pub fn fits(self, dims: &BoardDims) -> bool {
        let depth = Self::max_back(dims);
        // Spread stacks two players on each outer wing row, so it needs a wing
        // that may legally hold two. On narrower boards the wing cap drops to
        // one (or to none at all) and the shape stops being Spread.
        let wide_wings = dims.max_players_per_wing() >= 2;
        match self {
            Formation::Line => true,
            Formation::Spread => dims.team_size >= 4 && depth >= 3 && wide_wings,
            Formation::Wedge => dims.team_size >= 4 && depth >= 5,
            Formation::Zone => dims.team_size >= 4 && depth >= 4,
        }
    }

    /// Formations legal on this board, in menu order.
    pub fn available(dims: &BoardDims) -> impl Iterator<Item = Formation> + '_ {
        Self::ALL.into_iter().filter(|f| f.fits(dims))
    }

    /// Deepest offset from our own line of scrimmage that still lands on the
    /// pitch and on our own half.
    fn max_back(dims: &BoardDims) -> Coord {
        dims.width / 2 - 2
    }

    fn slots(self) -> Vec<Slot> {
        use PlayerRole::*;
        match self {
            // Front rank centre-out (blitzers on the shoulders at Los(3)/Los(4)),
            // catchers a step behind it, throwers deep. On the full pitch this is
            // square-for-square the old hard-coded formation.
            Formation::Line => vec![
                (Lineman, 0, Row::Los(0)),
                (Lineman, 0, Row::Los(1)),
                (Lineman, 0, Row::Los(2)),
                (Blitzer, 0, Row::Los(3)),
                (Blitzer, 0, Row::Los(4)),
                (Lineman, 0, Row::Los(5)),
                (Lineman, 0, Row::Los(6)),
                (Catcher, 2, Row::Off(-2)),
                (Catcher, 2, Row::Off(2)),
                (Thrower, 6, Row::Off(-3)),
                (Thrower, 6, Row::Off(3)),
            ],
            // Two per wing is the cap in `is_setup_legal`, so the wing pairs sit
            // at different depths on the same outer row.
            Formation::Spread => vec![
                (Lineman, 0, Row::Los(0)),
                (Lineman, 0, Row::Los(1)),
                (Lineman, 0, Row::Los(2)),
                (Blitzer, 1, Row::Wing(-1)),
                (Blitzer, 1, Row::Wing(1)),
                (Catcher, 3, Row::Wing(-1)),
                (Catcher, 3, Row::Wing(1)),
                (Lineman, 0, Row::Los(3)),
                (Lineman, 0, Row::Los(4)),
                (Thrower, 5, Row::Off(0)),
                (Thrower, 5, Row::Off(1)),
            ],
            // A cage around a deep receiver: corners at back 3 and 5, wide
            // outlets for the hand-off.
            Formation::Wedge => vec![
                (Lineman, 0, Row::Los(0)),
                (Lineman, 0, Row::Los(1)),
                (Lineman, 0, Row::Los(2)),
                (Thrower, 4, Row::Off(0)),
                (Blitzer, 3, Row::Off(1)),
                (Blitzer, 3, Row::Off(-1)),
                (Lineman, 5, Row::Off(1)),
                (Lineman, 5, Row::Off(-1)),
                (Catcher, 2, Row::Off(3)),
                (Catcher, 2, Row::Off(-3)),
                (Lineman, 0, Row::Los(3)),
            ],
            // Nothing committed to the wings: a second rank two back, wide cover
            // further out, one safety on the deep centre.
            Formation::Zone => vec![
                (Lineman, 0, Row::Los(0)),
                (Lineman, 0, Row::Los(1)),
                (Lineman, 0, Row::Los(2)),
                (Blitzer, 2, Row::Off(2)),
                (Blitzer, 2, Row::Off(-2)),
                (Lineman, 2, Row::Off(0)),
                (Catcher, 4, Row::Off(3)),
                (Catcher, 4, Row::Off(-3)),
                (Thrower, 6, Row::Off(0)),
                (Lineman, 0, Row::Los(3)),
                (Lineman, 0, Row::Los(4)),
            ],
        }
    }

    /// The board's LOS rows ordered centre-out; `Row::Los(i)` indexes this.
    fn los_rows_center_out(dims: &BoardDims) -> Vec<Coord> {
        let band = dims.los_y_range();
        // Clamped into the band: on an even playable height the band has an
        // even number of rows, so `height / 2` is half a square off centre and
        // (on the shortest boards) could fall outside it entirely — which would
        // spin the loop below forever.
        let center = (dims.height / 2).clamp(*band.start(), *band.end());
        let mut rows = vec![center];
        let mut step = 1;
        while rows.len() < band.clone().count() {
            for y in [center + step, center - step] {
                if band.contains(&y) {
                    rows.push(y);
                }
            }
            step += 1;
        }
        rows
    }

    /// Resolve a slot to a square on `team`'s own half, or `None` when the
    /// board has no row for it.
    fn square(dims: &BoardDims, team: TeamType, slot: Slot) -> Option<Position> {
        let (_, back, row) = slot;
        let center = dims.height / 2;
        let y = match row {
            Row::Los(i) => *Self::los_rows_center_out(dims).get(i)?,
            Row::Off(k) => (center + k).clamp(1, dims.height - 2),
            // A short board has no wing rows at all; the slot is then skipped
            // rather than collapsing onto row 1, which belongs to the LOS band.
            Row::Wing(-1) => dims.north_wing_y_range().next()?,
            Row::Wing(_) => dims.south_wing_y_range().last()?,
        };
        let back = back.min(Self::max_back(dims));
        let x_delta_sign = if team == TeamType::Home { 1 } else { -1 };
        Some(Position::new((dims.los_x(team) + back * x_delta_sign, y)))
    }

    /// The squares this formation gives `team`'s players currently on the
    /// pitch — the ones [`Setup`] staged plus any already placed — laid out on
    /// an empty half. Players the formation has no room for (beyond
    /// `team_size`) are absent from the map: they sit the drive out.
    ///
    /// Recomputed from scratch at every placement, and stable across them:
    /// the roster is sorted by role rank then id (ids are handed out in that
    /// same order when the players are staged), and a player the greedy slot
    /// fill never picks does not influence which players it does pick, so
    /// benching one leaves the rest of the plan as it was. That is what lets
    /// a stateless caller follow one formation one placement at a time.
    pub fn plan(self, game_state: &GameState, team: TeamType) -> HashMap<PlayerID, Position> {
        let dims = game_state.board_dims;
        // Fixed role order, never dugout-slot order: the dugout is one shared
        // array refilled first-free-slot, so after a drive in which the other
        // team set up first our bench order is scrambled (plan 032 #11).
        let mut bench: Vec<(u8, PlayerID, PlayerRole)> = game_state
            .get_players_on_pitch_in_team(team)
            .map(|p| (role_rank(p.stats.role), p.id, p.stats.role))
            .collect();
        bench.sort_unstable_by_key(|(rank, id, _)| (*rank, *id));
        let mut occupied: HashSet<Position> = HashSet::new();
        let mut plan = HashMap::new();
        // Slots first, then whoever is left over: a formation can have fewer
        // usable slots than the board has players (LOS slots are skipped when
        // the band is narrow), and the team must still field `team_size`.
        let slots = self.slots().into_iter().map(Some).chain(std::iter::repeat(None));
        for slot in slots {
            if plan.len() >= dims.team_size || bench.is_empty() {
                break;
            }
            let position = match slot {
                // A slot the board has no row for is dropped, not relocated —
                // the leftover players are placed by the `None` arm below,
                // after every real slot has had its turn.
                Some(s) => match Formation::square(&dims, team, s) {
                    None => continue,
                    // Clamping `back` on a shallow board can still land two
                    // slots on the same square; that one falls back.
                    Some(pos) if dims.is_out(pos) || occupied.contains(&pos) => reserve_square(&dims, team, &occupied),
                    Some(pos) => pos,
                },
                None => reserve_square(&dims, team, &occupied),
            };
            // Take the wanted role if it's still on the bench, else the
            // lowest-ranked player left (linemen before positionals).
            let wanted = slot.map(|s| s.0);
            let idx = wanted
                .and_then(|want| bench.iter().position(|(_, _, role)| *role == want))
                .unwrap_or(0);
            let (_, id, _) = bench.remove(idx);
            occupied.insert(position);
            plan.insert(id, position);
        }
        plan
    }

    /// This formation's answer to the placement `team` is being asked for, or
    /// `None` when `team` is not setting up. Always a legal action: the
    /// planned square when the mask allows it, the bench when the plan has no
    /// square for this player and the team can spare it, else the first legal
    /// square (a hand-placed teammate may already sit on the planned one).
    pub fn next_action(self, game_state: &GameState, team: TeamType) -> Option<Action> {
        if game_state.setup_team()? != team {
            return None;
        }
        let head = game_state.info.active_player?;
        match self.plan(game_state, team).get(&head) {
            Some(&pos) if game_state.is_legal_action(&Action::Positional(PosAT::PlacePlayer, pos)) => {
                return Some(Action::Positional(PosAT::PlacePlayer, pos));
            }
            None if game_state.is_legal_action(&Action::Simple(SimpleAT::BenchPlayer)) => {
                return Some(Action::Simple(SimpleAT::BenchPlayer));
            }
            _ => {}
        }
        game_state
            .get_all_actions()
            .into_iter()
            .find(|a| matches!(a, Action::Positional(PosAT::PlacePlayer, _)))
    }
}

/// Play out the whole of the current team's setup with `formation`. Panics
/// when nobody is setting up.
pub fn auto_setup(game_state: &mut GameState, formation: Formation) {
    let team = game_state.setup_team().expect("auto_setup: no team is setting up");
    while let Some(action) = formation.next_action(game_state, team) {
        game_state.step(action).expect("formation placement was not legal");
    }
}

/// Setup placement order: linemen first, throwers last. Also the tie-break the
/// formations use when a wanted role is off the bench.
fn role_rank(role: PlayerRole) -> u8 {
    match role {
        PlayerRole::Lineman => 0,
        PlayerRole::Blitzer => 1,
        PlayerRole::Catcher => 2,
        PlayerRole::Thrower => 3,
    }
}

/// Fallback square for a player a formation has no room for: the first free
/// square behind our own line of scrimmage, shallow ranks first and centre
/// rows before wings (the wings are capped by `is_setup_legal`, so they are
/// filled only as a last resort).
fn reserve_square(dims: &BoardDims, team: TeamType, occupied: &HashSet<Position>) -> Position {
    let center = dims.height / 2;
    let x_delta_sign = if team == TeamType::Home { 1 } else { -1 };
    let los_x = dims.los_x(team);
    let (north, south) = (dims.north_wing_y_range(), dims.south_wing_y_range());
    let mut rows: Vec<Coord> = (1..=dims.height - 2).collect();
    rows.sort_by_key(|y| (y - center).abs());
    for wings_allowed in [false, true] {
        for back in 0..=(dims.width / 2 - 2) {
            for &y in &rows {
                if !wings_allowed && (north.contains(&y) || south.contains(&y)) {
                    continue;
                }
                let pos = Position::new((los_x + back * x_delta_sign, y));
                if !occupied.contains(&pos) {
                    return pos;
                }
            }
        }
    }
    panic!("no free square on own half for setup");
}

/// Squares to park the players waiting to be placed on: the team's own
/// endzone column, top to bottom, spilling one column forward at a time if
/// the roster outgrows it.
fn staging_squares(dims: BoardDims, team: TeamType) -> impl Iterator<Item = Position> {
    let x0 = dims.own_endzone_x(team);
    let inward = if team == TeamType::Home { -1 } else { 1 };
    (0..dims.width / 2 - 1)
        .flat_map(move |back| (1..=dims.height - 2).map(move |y| Position::new((x0 + back * inward, y))))
}

/// One team's kickoff setup, one player at a time.
///
/// Every player available for the drive (the reserves — `KOWakeUp` has run
/// by now) is *staged* onto the pitch first, in placement order, parked in
/// the team's own endzone and flagged `used` until placed. That deliberately
/// puts more players on the pitch than the rules allow, and never reaches a
/// kickoff that way: the procedure walks its queue, asking about one player
/// at a time (it is `info.active_player` while asked about), and every
/// waiting player is benched the moment `team_size` are placed. Staging is
/// what lets a network that only sees the pitch see who is still to come
/// and whom it is placing, with no bench representation at all.
///
/// The offer for each player is the set of own-half squares the rules still
/// allow (`GameState::is_setup_legal` is guaranteed at the end, never
/// checked mid-way):
///
/// - a square occupied by a *placed* teammate is off; one a *waiting*
///   teammate is parked on is fine — the two swap;
/// - a wing closes once it holds its cap of placed players (the LOS column
///   inside a wing row is a wing square — the bands partition the rows);
/// - when the line of scrimmage still needs as many players as there are
///   placements left, only LOS squares are offered;
/// - `BenchPlayer` is offered only while the team could still field its
///   minimum (`min(team_size, available)`) and the LOS without this player.
///
/// The queue order is fixed (role rank, then id) so a setup is a *tree* of
/// depth "players available" rather than a choice of whom to place next;
/// two orders of the same placements would otherwise field the same players
/// under different ids and never recombine in the MCTS DAG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Setup {
    team: TeamType,
    /// Players still to be asked about, next first. Filled when the
    /// procedure first runs.
    queue: Vec<PlayerID>,
    placed: u8,
    started: bool,
}
impl Setup {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::Setup(Setup {
            team,
            queue: Vec::new(),
            placed: 0,
            started: false,
        })
    }
    pub fn team(&self) -> TeamType {
        self.team
    }
    /// Players still waiting to be placed, the one being asked about first.
    pub fn queue(&self) -> &[PlayerID] {
        &self.queue
    }
    pub fn placed(&self) -> usize {
        self.placed as usize
    }

    /// Bring every reserve onto the pitch in placement order.
    fn stage(&mut self, game_state: &mut GameState) {
        let mut bench: Vec<(u8, DugoutPlayerID)> = game_state
            .get_dugout()
            .filter(|p| p.stats.team == self.team && p.place == DugoutPlace::Reserves)
            .map(|p| (role_rank(p.stats.role), p.id))
            .collect();
        bench.sort_unstable();
        let mut squares = staging_squares(game_state.board_dims, self.team);
        for (_, dugout_id) in bench {
            let pos = squares
                .find(|p| game_state.get_player_id_at(*p).is_none())
                .expect("no free square to stage a player on");
            game_state.field_dugout_player(dugout_id, pos);
            let id = game_state.get_player_id_at(pos).unwrap();
            game_state.get_mut_player_unsafe(id).used = true;
            self.queue.push(id);
        }
    }

    /// The squares the next player may go to, and whether it may be benched.
    fn legal_placements(&self, game_state: &GameState) -> (Vec<Position>, bool) {
        let dims = game_state.board_dims;
        let team = self.team;
        let team_size = dims.team_size;
        let placed = self.placed as usize;
        let queued = self.queue.len();
        let reserves = game_state
            .get_dugout()
            .filter(|p| p.stats.team == team && p.place == DugoutPlace::Reserves)
            .count();
        // Everyone who could play this drive, benched or not — the same count
        // `is_setup_legal` measures the minimums against.
        let available = placed + queued + reserves;
        let min_on_pitch = team_size.min(available);
        let min_los = dims.min_players_on_los().min(available).min(team_size);

        let los_x = dims.los_x(team);
        let los_band = dims.los_y_range();
        let (north, south) = (dims.north_wing_y_range(), dims.south_wing_y_range());
        let max_wing = dims.max_players_per_wing();
        let on_los = |pos: Position| pos.x == los_x && los_band.contains(&pos.y);

        let (mut los_placed, mut north_placed, mut south_placed) = (0usize, 0usize, 0usize);
        for p in game_state.get_players_on_pitch_in_team(team) {
            if self.queue.contains(&p.id) {
                continue;
            }
            if on_los(p.position) {
                los_placed += 1;
            } else if north.contains(&p.position.y) {
                north_placed += 1;
            } else if south.contains(&p.position.y) {
                south_placed += 1;
            }
        }
        let los_needed = min_los.saturating_sub(los_placed);
        let slots_left = (team_size - placed).min(queued);
        let force_los = los_needed >= slots_left;
        let bench_ok = placed + queued > min_on_pitch && los_needed <= (team_size - placed).min(queued - 1);

        let mut squares = Vec::new();
        for x in 1..=dims.width - 2 {
            for y in 1..=dims.height - 2 {
                let pos = Position::new((x, y));
                if !dims.is_on_team_side(pos, team) {
                    continue;
                }
                if let Some(id) = game_state.get_player_id_at(pos) {
                    if !self.queue.contains(&id) {
                        continue;
                    }
                }
                if force_los && !on_los(pos) {
                    continue;
                }
                if !on_los(pos)
                    && ((north.contains(&y) && north_placed >= max_wing)
                        || (south.contains(&y) && south_placed >= max_wing))
                {
                    continue;
                }
                squares.push(pos);
            }
        }
        (squares, bench_ok)
    }

    fn offer(&self, game_state: &mut GameState) -> ProcState {
        let head = self.queue[0];
        game_state.info.active_player = Some(head);
        let (squares, bench_ok) = self.legal_placements(game_state);
        debug_assert!(!squares.is_empty() || bench_ok, "setup offers nothing for {head}");
        let mut aa = AvailableActions::new(self.team);
        aa.insert_positional(PosAT::PlacePlayer, squares);
        if bench_ok {
            aa.insert_simple(SimpleAT::BenchPlayer);
        }
        ProcState::NeedAction(aa)
    }

    fn advance(&mut self, game_state: &mut GameState) -> ProcState {
        if self.placed as usize >= game_state.board_dims.team_size || self.queue.is_empty() {
            return self.finish(game_state);
        }
        self.offer(game_state)
    }

    /// Bench whoever is still waiting and hand over.
    fn finish(&mut self, game_state: &mut GameState) -> ProcState {
        for id in self.queue.drain(..) {
            game_state.unfield_player(id, DugoutPlace::Reserves).unwrap();
        }
        game_state.info.active_player = None;
        debug_assert!(
            game_state.is_setup_legal(self.team),
            "setup procedure produced an illegal setup for {:?}",
            self.team
        );
        ProcState::Done
    }
}
impl Procedure for Setup {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => {
                if !self.started {
                    self.started = true;
                    self.stage(game_state);
                }
                self.advance(game_state)
            }
            ProcInput::Action(Action::Positional(PosAT::PlacePlayer, pos)) => {
                let head = self.queue.remove(0);
                match game_state.get_player_id_at(pos) {
                    Some(other) if other == head => {}
                    Some(other) => game_state.swap_players(head, other).unwrap(),
                    None => game_state.move_player(head, pos).unwrap(),
                }
                game_state.get_mut_player_unsafe(head).used = false;
                crate::game_log!(game_state, "placing {:?} {} at {:?}", self.team, head, pos);
                self.placed += 1;
                self.advance(game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::BenchPlayer)) => {
                let head = self.queue.remove(0);
                crate::game_log!(game_state, "benching {:?} {}", self.team, head);
                game_state.unfield_player(head, DugoutPlace::Reserves).unwrap();
                self.advance(game_state)
            }
            _ => unreachable!("unexpected setup input {input:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{auto_setup, Formation, Kickoff};
    use crate::core::dices::{RollResult, D6, D8};
    use crate::core::gamestate::{BuilderState, GameState, GameStateBuilder};
    use crate::core::model::*;
    use crate::core::table::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::iter::zip;

    #[test]
    fn test_setup_preconfigured_formations() {
        // The hard-coded formation offsets only fit (un-clamped) and field the
        // full 11-player line on the default pitch.
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state: GameState = GameStateBuilder::new_at_setup();
        //away as defense
        auto_setup(&mut state, Formation::Line);
        //home as offense
        auto_setup(&mut state, Formation::Line);

        for team in [TeamType::Home, TeamType::Away] {
            let middle_x = state.get_line_of_scrimage_x(team);
            let middle_y = state.board_dims.height / 2;

            let linemen_pos = vec![(0, 0), (0, -1), (0, 1), (0, -3), (0, 3)];
            let blitzer_pos = vec![(0, -2), (0, 2)];
            let catcher_pos = vec![(2, 2), (2, -2)];
            let thrower_pos = vec![(6, 3), (6, -3)];
            let stats_types = vec![
                PlayerStats::new_lineman(team),
                PlayerStats::new_blitzer(team),
                PlayerStats::new_catcher(team),
                PlayerStats::new_thrower(team),
            ];
            let stats_positions = vec![linemen_pos, blitzer_pos, catcher_pos, thrower_pos];

            let expected_count = stats_positions.iter().map(|x| x.len()).sum::<usize>();
            let actual_count = state.get_players_on_pitch_in_team(team).count();
            assert_eq!(
                actual_count, expected_count,
                "Team {:?} has {:?} players,",
                team, actual_count
            );

            let x_delta_sign = if team == TeamType::Home { 1 } else { -1 };

            for (stats, positions) in zip(stats_types, stats_positions) {
                for (dx, dy) in positions {
                    let (x, y) = (middle_x + dx * x_delta_sign, middle_y + dy);
                    match state.get_player_at_coord(x, y) {
                        Some(correct_player) if correct_player.stats == stats => (),
                        Some(wrong_player) => panic!(
                            "Wrong player at ({:?}, {:?}), found a {:?} ({:?}) but expected a {:?} ({:?})",
                            x, y, wrong_player.stats.role, wrong_player.stats.team, stats.role, stats.team
                        ),
                        None => panic!(
                            "No player at ({:?}, {:?}), expected a {:?} ({:?})",
                            x, y, stats.role, stats.team
                        ),
                    }
                }
            }
        }
    }

    /// Board sizes to exercise the formations on: the compiled capacity plus a
    /// couple of smaller tiers (skipped when the build is too small for them).
    fn test_boards() -> Vec<BoardDims> {
        let capacity = BoardDims::default();
        let mut boards = vec![capacity];
        for (w, h, players) in [
            (28, 17, 11),
            (22, 11, 8),
            (18, 11, 6),
            (16, 9, 4),
            // Even heights (no longer rejected) and the short boards where the
            // wings empty out and the LOS band is the whole pitch.
            (16, 10, 4),
            (16, 8, 4),
            (16, 7, 3),
            (18, 6, 3),
            (12, 5, 2),
            (10, 5, 2),
        ] {
            if (w, h, players) != (capacity.width, capacity.height, capacity.team_size)
                && w <= capacity.width
                && h <= capacity.height
                && players <= capacity.team_size
            {
                boards.push(BoardDims::new(w, h, players));
            }
        }
        boards
    }

    /// Away wins the toss and kicks, so Away (the kicker) is asked to set up
    /// first.
    fn at_setup(dims: BoardDims) -> GameState {
        let mut state = GameStateBuilder::new()
            .with_board_dims(dims)
            .set_state(BuilderState::CoinToss)
            .build();
        state.fix_coin(crate::core::dices::Coin::Heads);
        state.step_simple(SimpleAT::Heads);
        state.step_simple(SimpleAT::Kick);
        state
    }

    fn squares(state: &GameState, team: TeamType) -> Vec<Position> {
        let mut v: Vec<Position> = state.get_players_on_pitch_in_team(team).map(|p| p.position).collect();
        v.sort_by_key(|p| (p.x, p.y));
        v
    }

    #[test]
    fn every_offered_formation_is_legal_on_every_board() {
        for dims in test_boards() {
            for formation in Formation::available(&dims) {
                let mut state = at_setup(dims);
                let team = state.setup_team().unwrap();
                auto_setup(&mut state, formation);
                assert_eq!(
                    state.get_players_on_pitch_in_team(team).count(),
                    dims.team_size,
                    "{formation:?} on {}x{}/{} should field the whole team",
                    dims.width,
                    dims.height,
                    dims.team_size
                );
                assert!(
                    state.is_setup_legal(team),
                    "{formation:?} is an illegal setup on {}x{}/{}: {:?}",
                    dims.width,
                    dims.height,
                    dims.team_size,
                    squares(&state, team)
                );
            }
        }
    }

    #[test]
    fn offered_formations_are_distinct() {
        for dims in test_boards() {
            let mut seen: Vec<(Formation, Vec<Position>)> = Vec::new();
            for formation in Formation::available(&dims) {
                let mut state = at_setup(dims);
                let team = state.setup_team().unwrap();
                auto_setup(&mut state, formation);
                let occupied = squares(&state, team);
                if let Some((other, _)) = seen.iter().find(|(_, other)| *other == occupied) {
                    panic!(
                        "{formation:?} and {other:?} field identical squares on {}x{}/{} — \
                         one of them should not be offered there",
                        dims.width, dims.height, dims.team_size
                    );
                }
                seen.push((formation, occupied));
            }
            assert!(
                seen.len() >= 2 || dims.team_size < 4,
                "{}x{}/{} offers only one formation",
                dims.width,
                dims.height,
                dims.team_size
            );
        }
    }

    /// Both teams get the same shape, mirrored across the halfway line.
    #[test]
    fn formations_are_mirrored_between_teams() {
        let dims = BoardDims::default();
        for formation in Formation::available(&dims) {
            let mut state = at_setup(dims);
            auto_setup(&mut state, formation); // Away (kicking)
            auto_setup(&mut state, formation); // Home (receiving)
            let home = squares(&state, TeamType::Home);
            let mut away: Vec<Position> = squares(&state, TeamType::Away)
                .iter()
                .map(|p| Position::new((dims.width - 1 - p.x, p.y)))
                .collect();
            away.sort_by_key(|p| (p.x, p.y));
            assert_eq!(home, away, "{formation:?} is not mirror-symmetric");
        }
    }

    // ---- per-player setup -------------------------------------------------

    fn placements(state: &GameState) -> Vec<Position> {
        state
            .get_all_actions()
            .into_iter()
            .filter_map(|a| match a {
                Action::Positional(PosAT::PlacePlayer, pos) => Some(pos),
                _ => None,
            })
            .collect()
    }

    fn bench_offered(state: &GameState) -> bool {
        state.is_legal_action(&Action::Simple(SimpleAT::BenchPlayer))
    }

    fn head(state: &GameState) -> PlayerID {
        state.info.active_player.expect("setup asks about one player")
    }

    #[test]
    fn setup_stages_every_reserve_in_the_own_endzone_and_asks_about_one() {
        for dims in test_boards() {
            let state = at_setup(dims);
            let team = state.setup_team().expect("the kicker is setting up");
            assert_eq!(team, TeamType::Away);
            let roster = dims.roster_per_team();
            let staged: Vec<&FieldedPlayer> = state.get_players_on_pitch_in_team(team).collect();
            assert_eq!(staged.len(), roster, "every reserve is on the pitch while waiting");
            assert_eq!(
                state.get_dugout().filter(|p| p.stats.team == team).count(),
                0,
                "nobody is left in the dugout"
            );
            for p in &staged {
                assert!(p.used, "a waiting player is flagged used");
                assert!(dims.is_on_team_side(p.position, team));
            }
            if roster <= (dims.height - 2) as usize {
                assert!(
                    staged.iter().all(|p| p.position.x == dims.own_endzone_x(team)),
                    "the whole roster fits in the endzone column on {}x{}",
                    dims.width,
                    dims.height
                );
            }
            // The first player asked about has the lowest role rank on the roster
            // (linemen first, throwers last; the smallest rosters carry no linemen).
            let h = state.get_player_unsafe(head(&state));
            let lowest = staged.iter().map(|p| super::role_rank(p.stats.role)).min().unwrap();
            assert_eq!(super::role_rank(h.stats.role), lowest);

            // It may go anywhere on its own half — including the squares its
            // waiting teammates are parked on — and, with a spare player on
            // the roster, it may sit out.
            let offered = placements(&state);
            let own_half: Vec<Position> = (1..=dims.width - 2)
                .flat_map(|x| (1..=dims.height - 2).map(move |y| Position::new((x, y))))
                .filter(|p| dims.is_on_team_side(*p, team))
                .filter(|p| {
                    let wing = dims.north_wing_y_range().contains(&p.y) || dims.south_wing_y_range().contains(&p.y);
                    !(wing && dims.max_players_per_wing() == 0)
                })
                .collect();
            if dims.team_size > dims.min_players_on_los() {
                assert_eq!(offered, own_half, "{}x{}/{}", dims.width, dims.height, dims.team_size);
            } else {
                // A team no bigger than the LOS minimum has no choice at all.
                let los_x = dims.los_x(team);
                let band = dims.los_y_range();
                assert!(offered.iter().all(|p| p.x == los_x && band.contains(&p.y)));
            }
            assert_eq!(bench_offered(&state), roster > dims.team_size);
        }
    }

    #[test]
    fn placing_walks_the_queue_and_benches_the_rest_at_team_size() {
        for dims in test_boards() {
            let mut state = at_setup(dims);
            let team = state.setup_team().unwrap();
            let roster = dims.roster_per_team();
            let mut asked = Vec::new();
            while state.setup_team() == Some(team) {
                asked.push(head(&state));
                let pos = placements(&state)[0];
                state.step_positional(PosAT::PlacePlayer, pos);
            }
            assert_eq!(
                asked.len(),
                dims.team_size.min(roster),
                "one question per fielded player"
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(team).count(),
                dims.team_size.min(roster)
            );
            assert!(state.get_players_on_pitch_in_team(team).all(|p| !p.used));
            assert_eq!(
                state
                    .get_dugout()
                    .filter(|p| p.stats.team == team && p.place == DugoutPlace::Reserves)
                    .count(),
                roster.saturating_sub(dims.team_size),
                "whoever was still waiting went back to the reserves"
            );
            assert!(state.is_setup_legal(team));
            assert_eq!(state.setup_team(), Some(other_team(team)), "the receiver sets up next");
        }
    }

    #[test]
    fn line_of_scrimmage_is_forced_when_the_remaining_placements_need_it() {
        let dims = BoardDims::default();
        let mut state = at_setup(dims);
        let team = state.setup_team().unwrap();
        let los_x = dims.los_x(team);
        let band = dims.los_y_range();
        let on_los = |p: &Position| p.x == los_x && band.contains(&p.y);
        let need = dims.min_players_on_los().min(dims.team_size);
        // Keep everyone off the line until only `need` placements are left.
        let mut placed = 0;
        while dims.team_size - placed > need {
            let offered = placements(&state);
            assert!(offered.iter().any(|p| !on_los(p)), "off-line squares are still open");
            let pos = *offered.iter().find(|p| !on_los(p)).unwrap();
            state.step_positional(PosAT::PlacePlayer, pos);
            placed += 1;
        }
        // From here on only the line is offered. The one spare may still sit
        // out (three others remain for the line); after that nobody may.
        assert!(bench_offered(&state), "the spare can still sit out");
        state.step_simple(SimpleAT::BenchPlayer);
        for _ in 0..need {
            let offered = placements(&state);
            assert!(!offered.is_empty());
            assert!(
                offered.iter().all(on_los),
                "only LOS squares may be offered now, got {offered:?}"
            );
            assert!(!bench_offered(&state), "benching would leave the line short");
            state.step_positional(PosAT::PlacePlayer, offered[0]);
        }
        assert_ne!(state.setup_team(), Some(team));
        assert!(state.is_setup_legal(team));
    }

    #[test]
    fn a_wing_closes_once_it_holds_its_cap() {
        let dims = BoardDims::default();
        let cap = dims.max_players_per_wing();
        assert!(cap >= 1, "the default board has wings");
        let mut state = at_setup(dims);
        let north = dims.north_wing_y_range();
        for _ in 0..cap {
            let pos = *placements(&state)
                .iter()
                .find(|p| north.contains(&p.y))
                .expect("a north wing square is open");
            state.step_positional(PosAT::PlacePlayer, pos);
        }
        let offered = placements(&state);
        assert!(!offered.is_empty());
        assert!(
            offered.iter().all(|p| !north.contains(&p.y)),
            "the north wing is full, got {offered:?}"
        );
        assert!(
            offered.iter().any(|p| dims.south_wing_y_range().contains(&p.y)),
            "the south wing is still open"
        );
    }

    #[test]
    fn bench_is_offered_only_while_the_team_can_spare_the_player() {
        let dims = BoardDims::default();
        let spare = dims.roster_per_team() - dims.team_size;
        assert_eq!(spare, 1, "the stock roster carries one spare");
        let mut state = at_setup(dims);
        let team = state.setup_team().unwrap();
        assert!(bench_offered(&state));
        let benched = head(&state);
        state.step_simple(SimpleAT::BenchPlayer);
        assert!(
            state
                .get_dugout()
                .any(|p| p.place == DugoutPlace::Reserves && p.stats.team == team),
            "the benched player is back in the reserves"
        );
        assert!(state.get_player(benched).is_err(), "and off the pitch");
        while state.setup_team() == Some(team) {
            assert!(!bench_offered(&state), "nobody else may sit out");
            state.step_positional(PosAT::PlacePlayer, placements(&state)[0]);
        }
        assert_eq!(state.get_players_on_pitch_in_team(team).count(), dims.team_size);
        assert!(state.is_setup_legal(team));
    }

    #[test]
    fn placing_onto_a_waiting_teammates_square_swaps_them() {
        let dims = BoardDims::default();
        let mut state = at_setup(dims);
        let mover = head(&state);
        let from = state.get_player_unsafe(mover).position;
        let other = state
            .get_players_on_pitch_in_team(state.setup_team().unwrap())
            .find(|p| p.id != mover)
            .unwrap();
        let (other_id, to) = (other.id, other.position);
        assert!(placements(&state).contains(&to));
        state.step_positional(PosAT::PlacePlayer, to);
        assert_eq!(state.get_player_unsafe(mover).position, to);
        assert_eq!(state.get_player_unsafe(other_id).position, from);
        assert!(!state.get_player_unsafe(mover).used);
        assert!(
            state.get_player_unsafe(other_id).used,
            "the displaced player is still waiting"
        );
        // Staying put is a placement too.
        let next = head(&state);
        let here = state.get_player_unsafe(next).position;
        assert!(placements(&state).contains(&here));
        state.step_positional(PosAT::PlacePlayer, here);
        assert_eq!(state.get_player_unsafe(next).position, here);
    }

    /// Every sequence of offered actions ends in a legal setup for both
    /// teams and reaches the kickoff — the mask, not a check at the end, is
    /// what makes a setup legal.
    #[test]
    fn any_sequence_of_offered_actions_ends_in_a_legal_setup() {
        for dims in test_boards() {
            for seed in 0..20u64 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut state = at_setup(dims);
                let mut guard = 0;
                while let Some(team) = state.setup_team() {
                    let actions = state.get_all_actions();
                    assert!(!actions.is_empty(), "setup offered nothing on {dims:?}");
                    let action = actions[rng.gen_range(0..actions.len())];
                    state.step(action).unwrap();
                    if state.setup_team() != Some(team) {
                        assert!(
                            state.is_setup_legal(team),
                            "seed {seed} on {}x{}/{} left {team:?} with an illegal setup: {:?}",
                            dims.width,
                            dims.height,
                            dims.team_size,
                            squares(&state, team)
                        );
                    }
                    guard += 1;
                    assert!(guard < 200, "setup did not terminate");
                }
                assert!(state.is_legal_action(&Action::Simple(SimpleAT::KickoffAimMiddle)));
                assert_eq!(
                    state.info.active_player, None,
                    "no player is being asked about any more"
                );
                for team in [TeamType::Home, TeamType::Away] {
                    assert_eq!(
                        state.get_players_on_pitch_in_team(team).count(),
                        dims.team_size.min(dims.roster_per_team())
                    );
                }
            }
        }
    }

    /// A formation followed one placement at a time lands exactly where the
    /// whole-team fielding used to, so the plan is stable across the
    /// placements it drives.
    #[test]
    fn a_formation_followed_stepwise_matches_its_plan() {
        for dims in test_boards() {
            for formation in Formation::available(&dims) {
                let mut state = at_setup(dims);
                let team = state.setup_team().unwrap();
                let plan = formation.plan(&state, team);
                assert_eq!(plan.len(), dims.team_size.min(dims.roster_per_team()));
                auto_setup(&mut state, formation);
                for (id, pos) in plan {
                    assert_eq!(
                        state.get_player(id).map(|p| p.position).ok(),
                        Some(pos),
                        "{formation:?} on {}x{}/{}",
                        dims.width,
                        dims.height,
                        dims.team_size
                    );
                }
            }
        }
    }

    #[test]
    fn kickoff_get_the_ref() {
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fix_d8_direction(Direction::up()); // scatter direction
        state.fix_d6(5); // scatter length

        // kickoff event fix
        state.fix_d6(1);
        state.fix_d6(1);

        state.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert_eq!(state.home.bribes, 1);
        assert_eq!(state.away.bribes, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 0);

        // todo: this assertion should be a in more general test
        //assert_eq!(state.info.home_turn, 1, "home turn counter should be 1");
        assert!(state.home_to_act());
        assert_eq!(
            (state.info.home_turn, state.info.away_turn),
            (1, 0),
            "turn counter (home, away) is wrong!"
        );
    }
    #[test]
    fn kickoff_timeout_step_clock_forward() {
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fix_d8_direction(Direction::up()); // scatter direction
        state.fix_d6(5); // scatter length

        // kickoff event fix
        state.fix_d6(1);
        state.fix_d6(2);
        state.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert!(state.home_to_act());
        assert_eq!(state.info.home_turn, 2);
        assert_eq!(state.info.away_turn, 1);
    }

    #[test]
    fn kickoff_timeout_step_clock_backwards() {
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state: GameState = GameStateBuilder::new()
            .set_state(BuilderState::Kickoff { turn: 7 })
            .build();
        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 6);
        // ball fixes
        state.fix_d8_direction(Direction::up()); // scatter direction
        state.fix_d6(5); // scatter length

        // kickoff event fix
        state.fix_d6(1);
        state.fix_d6(2);
        state.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);
        assert!(state.home_to_act());

        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 5);
    }

    /// A kickoff aimed at the middle scales the D6 deviate roll down on a
    /// narrow board, so it can't fling the ball as far out as an unscaled
    /// roll would — the dice rolled (D6 length, D8 direction) are unchanged,
    /// only how far the length carries the ball.
    #[test]
    fn kickoff_deviate_distance_is_scaled_down_on_a_narrow_board() {
        let dims = BoardDims::new(16, 9, 3); // 14x7 playable
        assert_eq!(dims.scatter_divisor(), 2, "test assumes a divisor of 2");
        let mut state = GameStateBuilder::new()
            .with_board_dims(dims)
            .set_state(BuilderState::CoinToss)
            .build();
        let aim = Position::new((8, 4));
        let mut kickoff = Kickoff { aim };

        kickoff.step(
            &mut state,
            ProcInput::Roll(RollResult::Deviate(D6::Six, D8::from(Direction::right()))),
        );

        let BallState::InAir(pos) = state.ball else {
            panic!("ball should be airborne right after the deviate roll")
        };
        assert_eq!(
            pos,
            aim + Direction::right() * 3,
            "raw roll 6 / scatter_divisor 2 == 3, not the unscaled 6"
        );
    }
    // #[test]
    // fn kickoff_solid_defence() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fix_d6(1);
    //     state.fix_d6(3);
    //
    //     state.fix_d6(6); //fix number of re-arranged players (d3+3)
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    //
    //     // TODO: haven't implemented the setup yet
    // }
    //
    // #[test]
    // fn kickoff_high_kick() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fix_d6(1);
    //     state.fix_d6(4);
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    //
    //     let ball_pos = state.get_ball_position().unwrap();
    //     assert!(matches!(state.ball, BallState::InAir(_)));
    //
    //     assert!(state.home_to_act());
    //     let legal_positions = [(2, 9), (7, 9)]; //Open players
    //     for pos in legal_positions {
    //         let action = Action::Positional(PosAT::SelectPosition, Position::new(pos));
    //         assert!(state.available_actions.is_legal_action(action));
    //     }
    //
    //     let catcher_start_pos = Position::new(legal_positions[0]);
    //     let catcher_id = state.get_player_id_at(catcher_start_pos).unwrap();
    //
    //     state.fix_d6(6); // fix the roll for the catch
    //     state.step_positional(PosAT::SelectPosition, Position::new(legal_positions[0]));
    //
    //     assert_eq!(state.get_player_id_at(ball_pos).unwrap(), catcher_id);
    //     assert_eq!(state.get_player_id_at(catcher_start_pos), None);
    //
    //     match state.ball {
    //         BallState::Carried(id) => {
    //             assert_eq!(id, catcher_id);
    //         }
    //         _ => panic!("ball should be carried"),
    //     }
    //
    //     assert!(state.home_to_act());
    // }
    //
    // #[test]
    // fn kickoff_cheering_fans() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fix_d6(1);
    //     state.fix_d6(5);
    //     // TODO: Implement prayers to nuffle...
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    // }
    //
    // #[test]
    // fn kickoff_brilliant_coaching() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fix_d6(1);
    //     state.fix_d6(1);
    //
    //     state.fix_d6(5); //fix home brilliant coaching roll
    //     state.fix_d6(6); //fix away brilliant coaching roll
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    //
    //     assert_eq!(state.away.rerolls, 4);
    //     assert_eq!(state.home.rerolls, 3);
    // }
    // #[test]
    // fn kickoff_changing_weather() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fix_d6(1);
    //     state.fix_d6(1);
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    // }
    // #[test]
    // fn kickoff_after_td() {
    //     let start_pos = Position::new((2, 5));
    //     let mut state = GameStateBuilder::new()
    //         .add_home_player(start_pos)
    //         .add_ball_pos(start_pos)
    //         .build();
    //
    //     state.step_positional(PosAT::StartMove, start_pos);
    //     state.step_positional(PosAT::Move, Position::new((1, 5)));
    //
    //     assert_eq!(state.home.score, 1);
    //     assert_eq!(state.away.score, 0);
    //
    //     assert!(state.home_to_act());
    //     state.step_simple(SimpleAT::SetupLine);
    //     state.step_simple(SimpleAT::EndSetup);
    //
    //     assert!(state.away_to_act());
    //     state.step_simple(SimpleAT::SetupLine);
    //     state.step_simple(SimpleAT::EndSetup);
    // }
}

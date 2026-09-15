use crate::core::model::ProcInput;
use std::ops::RangeInclusive;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, BoardDims, Coord, Direction, DugoutPlace, PlayerID, Position,
    ProcState, Procedure, Result, TeamType, Weather,
};
use crate::core::procedures::ball_procs;
use crate::core::table::*;

use crate::core::gamestate::GameState;

use super::AnyProc;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

        // Cap deviate distance at half the board width so the kick can't be
        // flung out of bounds on narrow tiers (no-op on the full pitch).
        let len = (len_roll as Coord).min(game_state.board_dims.max_scatter());
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Pre-configured setups. Each is a pure function of `(BoardDims, TeamType)`;
/// slots are listed in fielding priority order, so a team smaller than the
/// formation fields the front of the list and the rest sit out. Every
/// formation opens with three line-of-scrimmage slots, so any of them is legal
/// (`GameState::is_setup_legal`) on any board whose LOS band is three rows
/// wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    pub fn action(self) -> SimpleAT {
        match self {
            Formation::Line => SimpleAT::SetupLine,
            Formation::Spread => SimpleAT::SetupSpread,
            Formation::Wedge => SimpleAT::SetupWedge,
            Formation::Zone => SimpleAT::SetupZone,
        }
    }

    pub fn from_action(at: SimpleAT) -> Option<Formation> {
        match at {
            SimpleAT::SetupLine => Some(Formation::Line),
            SimpleAT::SetupSpread => Some(Formation::Spread),
            SimpleAT::SetupWedge => Some(Formation::Wedge),
            SimpleAT::SetupZone => Some(Formation::Zone),
            _ => None,
        }
    }

    /// Board requirements. The clamping below would keep an oversized formation
    /// on-pitch, but it would collapse it onto its neighbours and stop being
    /// the formation it claims to be — so the tighter shapes are simply not
    /// offered on boards that can't hold them. `Line` is always available.
    pub fn fits(self, dims: &BoardDims) -> bool {
        let depth = Self::max_back(dims);
        match self {
            Formation::Line => true,
            Formation::Spread => dims.team_size >= 4 && depth >= 3,
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
        let center = dims.height / 2;
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
            Row::Wing(-1) => *dims.north_wing_y_range().start(),
            Row::Wing(_) => *dims.south_wing_y_range().end(),
        };
        let back = back.min(Self::max_back(dims));
        let x_delta_sign = if team == TeamType::Home { 1 } else { -1 };
        Some(Position::new((dims.los_x(team) + back * x_delta_sign, y)))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Setup {
    team: TeamType,
}
impl Setup {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::Setup(Setup { team })
    }
    fn get_empty_pos_in_box(
        game_state: &mut GameState,
        x_range: RangeInclusive<Coord>,
        y_range: RangeInclusive<Coord>,
    ) -> Position {
        loop {
            let x = game_state.rng.gen_range(x_range.clone());
            let y = game_state.rng.gen_range(y_range.clone());
            if game_state.get_player_id_at_coord(x, y).is_none() {
                return Position { x, y };
            }
        }
    }
    pub fn random_setup(&self, game_state: &mut GameState) {
        #[allow(clippy::needless_collect)]
        let players: Vec<PlayerID> = game_state
            .get_dugout()
            .take(game_state.board_dims.team_size)
            .filter(|dplayer| dplayer.stats.team == self.team)
            .map(|p| p.id)
            .collect();

        let mut ids = players.into_iter();
        let los_x = game_state.get_line_of_scrimage_x(self.team);
        let los_y_range = game_state.board_dims.los_y_range();
        let los_x_range = los_x..=los_x;
        let x_range = match self.team {
            TeamType::Home => los_x..=game_state.board_dims.width - 2,
            TeamType::Away => 1..=los_x,
        };
        for _ in 0..3 {
            if let Some(id) = ids.next() {
                let p = Setup::get_empty_pos_in_box(game_state, los_x_range.clone(), los_y_range.clone());
                game_state.field_dugout_player(id, p);
            }
        }
        for id in ids {
            let p = Setup::get_empty_pos_in_box(game_state, x_range.clone(), los_y_range.clone());
            game_state.field_dugout_player(id, p);
        }
    }
    fn setup_formation(&self, game_state: &mut GameState, formation: Formation) -> Result<()> {
        //unfield all players
        let player_ids = game_state
            .get_players_on_pitch_in_team(self.team)
            .map(|p| p.id)
            .collect::<Vec<_>>();
        for id in player_ids {
            game_state.unfield_player(id, DugoutPlace::Reserves)?;
        }
        // Field in a fixed role order (the roster's), not dugout-slot order:
        // `unfield_player` refills the *shared* dugout array first-free-slot,
        // so after a drive in which the other team set up first our players
        // sit behind theirs and our bench order is scrambled. On boards where
        // `team_size` is below the formation's role slots that used to swap
        // which role sits out — Home played a Thrower for its Catcher from the
        // second drive on whenever Away received (plan 032 #11).
        let role_rank = |role: PlayerRole| match role {
            PlayerRole::Lineman => 0,
            PlayerRole::Blitzer => 1,
            PlayerRole::Catcher => 2,
            PlayerRole::Thrower => 3,
        };
        let mut bench: Vec<(u8, PlayerID, PlayerRole)> = game_state
            .get_dugout()
            .filter(|dplayer| dplayer.stats.team == self.team)
            .filter(|dplayer| dplayer.place == DugoutPlace::Reserves)
            .map(|p| (role_rank(p.stats.role), p.id, p.stats.role))
            .collect();
        bench.sort_unstable_by_key(|(rank, id, _)| (*rank, *id));
        let dims = game_state.board_dims;
        let mut fielded = 0usize;
        // Slots first, then whoever is left over: a formation can have fewer
        // usable slots than the board has players (LOS slots are skipped when
        // the band is narrow), and the team must still field `team_size`.
        let slots = formation.slots().into_iter().map(Some).chain(std::iter::repeat(None));
        for slot in slots {
            if fielded >= dims.team_size || bench.is_empty() {
                break;
            }
            let position = match slot {
                // A slot the board has no row for is dropped, not relocated —
                // the leftover players are placed by the `None` arm below,
                // after every real slot has had its turn.
                Some(s) => match Formation::square(&dims, self.team, s) {
                    None => continue,
                    // Clamping `back` on a shallow board can still land two
                    // slots on the same square; that one falls back.
                    Some(pos) if game_state.is_out(pos) || game_state.get_player_id_at(pos).is_some() => {
                        Self::reserve_square(game_state, self.team)
                    }
                    Some(pos) => pos,
                },
                None => Self::reserve_square(game_state, self.team),
            };
            // Take the wanted role if it's still on the bench, else the
            // lowest-ranked player left (linemen before positionals).
            let wanted = slot.map(|s| s.0);
            let idx = wanted
                .and_then(|want| bench.iter().position(|(_, _, role)| *role == want))
                .unwrap_or(0);
            let (_, id, role) = bench.remove(idx);
            fielded += 1;
            crate::game_log!(game_state, "fielding {:?} {:?} at {:?}", role, self.team, position);
            game_state.field_dugout_player(id, position)
        }
        Ok(())
    }
    /// Fallback square for a player the formation has no room for: the first
    /// free square behind our own line of scrimmage, shallow ranks first and
    /// centre rows before wings (the wings are capped at two players by
    /// `is_setup_legal`, so they are filled only as a last resort).
    fn reserve_square(game_state: &GameState, team: TeamType) -> Position {
        let dims = game_state.board_dims;
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
                    let x = los_x + back * x_delta_sign;
                    if game_state.get_player_id_at_coord(x, y).is_none() {
                        return Position { x, y };
                    }
                }
            }
        }
        panic!("no free square on own half for setup");
    }
}
impl Procedure for Setup {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        if input == ProcInput::Nothing {
            for formation in Formation::available(&game_state.board_dims) {
                aa.insert_simple(formation.action());
            }
            return ProcState::NeedAction(aa);
        }

        match input {
            ProcInput::Action(Action::Simple(at)) if Formation::from_action(at).is_some() => {
                self.setup_formation(game_state, Formation::from_action(at).unwrap())
                    .unwrap();
                aa.insert_simple(SimpleAT::EndSetup);
                ProcState::NeedAction(aa)
            }

            ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => ProcState::Done,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Formation;
    use crate::core::gamestate::{BuilderState, GameState, GameStateBuilder};
    use crate::core::model::*;
    use crate::core::table::*;
    use std::iter::zip;

    #[test]
    fn test_setup_preconfigured_formations() {
        // The hard-coded formation offsets only fit (un-clamped) and field the
        // full 11-player line on the default pitch.
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state: GameState = GameStateBuilder::new_at_setup();
        //away as defense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);
        //home as offense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);

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
        for (w, h, players) in [(28, 17, 11), (22, 11, 8), (18, 11, 6), (16, 9, 4)] {
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

    /// Away wins the toss and kicks, so Home (the receiver) is asked to set up.
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
                let team = state.get_active_teamtype().unwrap();
                state.step_simple(formation.action());
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
                state.step_simple(SimpleAT::EndSetup);
            }
        }
    }

    #[test]
    fn offered_formations_are_distinct() {
        for dims in test_boards() {
            let mut seen: Vec<(Formation, Vec<Position>)> = Vec::new();
            for formation in Formation::available(&dims) {
                let mut state = at_setup(dims);
                let team = state.get_active_teamtype().unwrap();
                state.step_simple(formation.action());
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
            state.step_simple(formation.action()); // Home (receiving)
            state.step_simple(SimpleAT::EndSetup);
            state.step_simple(formation.action()); // Away (kicking)
            state.step_simple(SimpleAT::EndSetup);
            let home = squares(&state, TeamType::Home);
            let mut away: Vec<Position> = squares(&state, TeamType::Away)
                .iter()
                .map(|p| Position::new((dims.width - 1 - p.x, p.y)))
                .collect();
            away.sort_by_key(|p| (p.x, p.y));
            assert_eq!(home, away, "{formation:?} is not mirror-symmetric");
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

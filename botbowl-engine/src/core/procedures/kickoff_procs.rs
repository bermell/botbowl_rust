use crate::core::model::ProcInput;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction,
    DugoutPlace, PlayerID, PlayerStatus, Position, ProcState, Procedure,
    TeamType, Weather, LINE_OF_SCRIMMAGE_Y_RANGE, NORTH_WING_Y_RANGE, SOUTH_WING_Y_RANGE
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

        let ball_pos = self.aim + Direction::from(dir_roll) * (len_roll as Coord);
        game_state.ball = BallState::InAir(ball_pos);
        ProcState::DoneNewProcs(vec![LandKickoff::new(), KickoffTable::new()])
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
        let mut procs: Vec<AnyProc> = Vec::new();
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
                procs.push(SolidDefence::new());
            }
            Sum2D6::Five => {
                procs.push(HighKick::new());
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
                if game_state.info.weather == Weather::Nice && !ball_pos.is_out() {
                    ProcState::NeedRoll(RequestedRoll::D8)
                } else {
                    ProcState::Done
                }
            }
            ProcInput::Roll(RollResult::D8(d8)) => {
                game_state.ball =
                    BallState::InAir(game_state.get_ball_position().unwrap() + Direction::from(d8));
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HighKick {}
impl HighKick {
    pub fn new() -> AnyProc {
        AnyProc::HighKick(HighKick {})
    }
}
impl Procedure for HighKick {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let BallState::InAir(ball_position) = game_state.ball else {
            return ProcState::Done;
        };
        let receiving_team = other_team(game_state.info.kicking_this_drive);

        if ball_position.is_out()
            || !ball_position.is_on_team_side(receiving_team)
            || game_state.get_player_id_at(ball_position).is_some()
        {
            return ProcState::Done;
        }

        match input {
            ProcInput::Nothing => {
                let positions: Vec<Position> = game_state
                    .get_players_on_pitch_in_team(receiving_team)
                    .filter(|p| p.status == PlayerStatus::Up)
                    .filter(|p| game_state.get_tz_on(p.id) == 0)
                    .map(|p| p.position)
                    .collect();

                if positions.is_empty() {
                    return ProcState::Done;
                }

                let mut aa = AvailableActions::new(receiving_team);
                aa.insert_positional(PosAT::SelectPosition, positions);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let player_id = game_state.get_player_id_at(pos).unwrap();
                game_state.move_player(player_id, ball_position).unwrap();
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum SolidDefenceState {
    Init,
    SelectPlayers,
    RearrangePlayers,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SolidDefence {
    state: SolidDefenceState,
    team: TeamType,
    max_rearrange: usize,
    target_on_pitch: usize,
    selected_fielded_ids: Vec<PlayerID>,
    selected_reserve_ids: Vec<usize>,
    selected_fielded_player: Option<PlayerID>,
    controlled_fielded_ids: HashSet<PlayerID>,
}

#[derive(Clone, Copy)]
struct SetupCounts {
    on_pitch: usize,
    los: usize,
    north: usize,
    south: usize,
}


impl SolidDefence {
    pub fn new() -> AnyProc {
        AnyProc::SolidDefence(SolidDefence {
            state: SolidDefenceState::Init,
            team: TeamType::Away,
            max_rearrange: 0,
            target_on_pitch: 0,
            selected_fielded_ids: Vec::new(),
            selected_reserve_ids: Vec::new(),
            selected_fielded_player: None,
            controlled_fielded_ids: HashSet::new(),
        })
    }

    fn open_player_ids(&self, game_state: &GameState) -> Vec<PlayerID> {
        game_state
            .get_players_on_pitch_in_team(self.team)
            .filter(|player| player.status == PlayerStatus::Up)
            .filter(|player| game_state.get_tz_on(player.id) == 0)
            .map(|player| player.id)
            .collect()
    }

    fn open_selectable_positions(&self, game_state: &GameState) -> Vec<Position> {
        self.open_player_ids(game_state)
            .into_iter()
            .filter(|id| !self.selected_fielded_ids.contains(id))
            .map(|id| game_state.get_player_unsafe(id).position)
            .collect()
    }

    fn build_selection_actions(&self, game_state: &GameState) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        aa.insert_simple(SimpleAT::EndSetup);
        aa.insert_positional(PosAT::SelectPosition, self.open_selectable_positions(game_state));
        ProcState::NeedAction(aa)
    }

    fn count_setup_positions(
        &self,
        game_state: &GameState,
        removed_pos: Option<Position>,
    ) -> SetupCounts {
        let line_x = game_state.get_line_of_scrimage_x(self.team);
        let mut counts = SetupCounts {
            on_pitch: 0,
            los: 0,
            north: 0,
            south: 0,
        };

        for pos in game_state
            .get_players_on_pitch_in_team(self.team)
            .map(|player| player.position)
        {
            counts.on_pitch += 1;
            if pos.is_los_position(line_x) {
                counts.los += 1;
            } else if pos.is_north_wing_position() {
                counts.north += 1;
            } else if pos.is_south_wing_position() {
                counts.south += 1;
            }
        }
        if let Some(pos) = removed_pos {
            counts.on_pitch = counts.on_pitch.saturating_sub(1);
            if pos.is_los_position(line_x) {
                counts.los = counts.los.saturating_sub(1);
            } else if pos.is_north_wing_position() {
                counts.north = counts.north.saturating_sub(1);
            } else if pos.is_south_wing_position() {
                counts.south = counts.south.saturating_sub(1);
            }
        }
        counts
    }

    fn legal_rearrange_positions(
        &self,
        game_state: &GameState,
        removed_pos: Option<Position>,
    ) -> Vec<Position> {
        let line_x = game_state.get_line_of_scrimage_x(self.team);
        let counts = self.count_setup_positions(game_state, removed_pos);
        let remaining_to_place = self.target_on_pitch.saturating_sub(counts.on_pitch);
        if remaining_to_place == 0 {
            return Vec::new();
        }

        let min_los = 3.min(self.target_on_pitch);
        let empty_los_squares = LINE_OF_SCRIMMAGE_Y_RANGE
            .clone()
            .filter(|&y| {
                let pos = Position::new((line_x, y));
                !pos.is_out() && (game_state.get_player_id_at(pos).is_none() || removed_pos == Some(pos))
            })
            .count();

        let mut candidates = Vec::new();
        for pos in Position::all_positions() {
            if pos.is_out()
                || !pos.is_on_team_side(self.team)
                || (game_state.get_player_id_at(pos).is_some() && removed_pos != Some(pos))
            {
                continue;
            }

            let is_los = pos.is_los_position(line_x);
            let new_los = counts.los + usize::from(is_los);
            let new_north = counts.north + usize::from(pos.is_north_wing_position());
            let new_south = counts.south + usize::from(pos.is_south_wing_position());
            if new_north > 2 || new_south > 2 {
                continue;
            }

            let remaining_players_after = remaining_to_place.saturating_sub(1);
            let remaining_los_needed = min_los.saturating_sub(new_los);
            let remaining_los_squares = if is_los {
                empty_los_squares.saturating_sub(1)
            } else {
                empty_los_squares
            };
            if remaining_players_after < remaining_los_needed
                || remaining_los_squares < remaining_los_needed
            {
                continue;
            }
            candidates.push(pos);
        }
        candidates
    }

    fn build_rearrange_actions(&self, game_state: &GameState) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        if let Some(source_id) = self.selected_fielded_player {
            let source_pos = game_state.get_player_unsafe(source_id).position;
            let mut positions = self.legal_rearrange_positions(game_state, Some(source_pos));
            positions.extend(
                self.controlled_fielded_ids
                    .iter()
                    .copied()
                    .filter(|id| *id != source_id)
                    .map(|id| game_state.get_player_unsafe(id).position),
            );
            positions.sort_unstable_by_key(|pos| (pos.x, pos.y));
            positions.dedup();
            aa.insert_positional(PosAT::SelectPosition, positions);
            return ProcState::NeedAction(aa);
        }

        if !self.selected_reserve_ids.is_empty() {
            let positions: Vec<Position> = self
                .legal_rearrange_positions(game_state, None)
                .into_iter()
                .filter(|pos| game_state.get_player_id_at(*pos).is_none())
                .collect();
            aa.insert_positional(PosAT::SelectPosition, positions);
            return ProcState::NeedAction(aa);
        }

        let positions: Vec<Position> = self
            .controlled_fielded_ids
            .iter()
            .copied()
            .map(|id| game_state.get_player_unsafe(id).position)
            .collect();
        aa.insert_positional(PosAT::SelectPosition, positions);

        if game_state.get_players_on_pitch_in_team(self.team).count() == self.target_on_pitch
            && game_state.is_setup_legal(self.team)
        {
            aa.insert_simple(SimpleAT::EndSetup);
        }
        ProcState::NeedAction(aa)
    }

    fn start_rearrange_phase(&mut self, game_state: &mut GameState) {
        self.target_on_pitch = game_state.get_players_on_pitch_in_team(self.team).count();
        let reserves_before: HashSet<usize> = game_state
            .get_dugout()
            .filter(|player| player.stats.team == self.team && player.place == DugoutPlace::Reserves)
            .map(|player| player.id)
            .collect();
        for id in self.selected_fielded_ids.iter().copied() {
            game_state.unfield_player(id, DugoutPlace::Reserves).unwrap();
        }
        self.selected_reserve_ids = game_state
            .get_dugout()
            .filter(|player| player.stats.team == self.team && player.place == DugoutPlace::Reserves)
            .map(|player| player.id)
            .filter(|id| !reserves_before.contains(id))
            .collect();
        self.selected_fielded_player = None;
        self.controlled_fielded_ids.clear();
        self.state = SolidDefenceState::RearrangePlayers;
    }

    fn max_rearrange_from_d6(roll: u8) -> usize {
        usize::from((roll + 1) / 2 + 3)
    }
}
impl Procedure for SolidDefence {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            SolidDefenceState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
                ProcInput::Roll(RollResult::D6(roll)) => {
                    self.team = game_state.info.kicking_this_drive;
                    self.max_rearrange = Self::max_rearrange_from_d6(roll as u8);
                    self.selected_fielded_ids.clear();
                    self.selected_reserve_ids.clear();
                    self.selected_fielded_player = None;
                    self.controlled_fielded_ids.clear();
                    self.state = SolidDefenceState::SelectPlayers;
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            SolidDefenceState::SelectPlayers => match input {
                ProcInput::Nothing => self.build_selection_actions(game_state),
                ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => {
                    self.start_rearrange_phase(game_state);
                    self.build_rearrange_actions(game_state)
                }
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                    let id = game_state.get_player_id_at(pos).unwrap();
                    assert_eq!(game_state.get_player_unsafe(id).stats.team, self.team);
                    assert_eq!(game_state.get_player_unsafe(id).status, PlayerStatus::Up);
                    assert_eq!(game_state.get_tz_on(id), 0);
                    if !self.selected_fielded_ids.contains(&id) {
                        self.selected_fielded_ids.push(id);
                    }
                    if self.selected_fielded_ids.len() >= self.max_rearrange {
                        self.start_rearrange_phase(game_state);
                        return self.build_rearrange_actions(game_state);
                    }
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            SolidDefenceState::RearrangePlayers => match input {
                ProcInput::Nothing => self.build_rearrange_actions(game_state),
                ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => ProcState::Done,
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                    if let Some(source_id) = self.selected_fielded_player.take() {
                        if let Some(target_id) = game_state.get_player_id_at(pos) {
                            assert!(self.controlled_fielded_ids.contains(&target_id));
                            if source_id != target_id {
                                game_state.swap_players_positions(source_id, target_id).unwrap();
                            }
                        } else {
                            game_state.move_player(source_id, pos).unwrap();
                        }
                    } else if !self.selected_reserve_ids.is_empty() {
                        assert!(game_state.get_player_id_at(pos).is_none());
                        let reserve_id = self.selected_reserve_ids.pop().unwrap();
                        game_state.field_dugout_player(reserve_id, pos);
                        self.controlled_fielded_ids
                            .insert(game_state.get_player_id_at(pos).unwrap());
                    } else if let Some(id) = game_state.get_player_id_at(pos) {
                        assert!(self.controlled_fielded_ids.contains(&id));
                        self.selected_fielded_player = Some(id);
                    }
                    self.build_rearrange_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
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

        if ball_position.is_out()
            || !ball_position.is_on_team_side(other_team(game_state.info.kicking_this_drive))
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

#[cfg(test)]
mod tests {
    use crate::core::gamestate::{BuilderState, GameState, GameStateBuilder};
    use crate::core::model::*;
    use crate::core::table::*;
    use std::collections::HashSet;

    #[test]
    fn kickoff_get_the_ref() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);

        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

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
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(2);
        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert!(state.home_to_act());
        assert_eq!(state.info.home_turn, 2);
        assert_eq!(state.info.away_turn, 1);
    }

    #[test]
    fn kickoff_timeout_step_clock_backwards() {
        let mut state: GameState = GameStateBuilder::new()
            .set_state(BuilderState::Kickoff { turn: 7 })
            .build();
        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 6);
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(2);
        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);
        assert!(state.home_to_act());

        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 5);
    }

    #[test]
    fn kickoff_changing_weather_lands_after_gust() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        state.fixes.fix_d6(1); // scatter length
        state.fixes.fix_d8_direction(Direction::down()); // scatter direction

        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4); // kickoff table: changing weather

        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4); // weather: nice

        state.fixes.fix_d8_direction(Direction::right()); // gust of wind
        state.fixes.fix_d8_direction(Direction::right()); // bounce

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert_eq!(state.ball, BallState::OnGround(Position::new((23, 8))));
    }

    mod kickoff_solid_defence {
        use super::*;

        fn positions_for_action(aa: &AvailableActions, action: PosAT) -> Vec<Position> {
            let Some(positions) = aa.get_positional() else {
                return Vec::new();
            };
            Position::all_positions()
                .filter(|pos| positions[*pos].contains(&action))
                .collect()
        }

        fn reserve_count(state: &GameState, team: TeamType) -> usize {
            state
                .get_dugout()
                .filter(|player| player.stats.team == team && player.place == DugoutPlace::Reserves)
                .count()
        }

        #[test]
        fn kickoff_solid_defence() {
            // Scenario A: cap can be reached and no marked player is selectable.
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);
            // Remove all opposing tackle zones so kicking team has many open players (> 4 cap).
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }
            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d6(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = positions_for_action(&state.available_actions, PosAT::SelectPosition);
            assert!(selectable.len() > 4);
            let marked_positions: Vec<Position> = state
                .get_players_on_pitch_in_team(kicking_team)
                .filter(|player| player.status == PlayerStatus::Up)
                .filter(|player| state.get_tz_on(player.id) > 0)
                .map(|player| player.position)
                .collect();
            for pos in marked_positions {
                assert!(!selectable.contains(&pos));
            }

            let reserves_before_cap = reserve_count(&state, kicking_team);
            let pitch_before_cap = state.get_players_on_pitch_in_team(kicking_team).count();
            let selected_for_cap: Vec<Position> = selectable.iter().copied().take(4).collect();
            let unselected_open: Vec<Position> = selectable.iter().copied().skip(4).collect();
            for pos in selected_for_cap {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            assert_eq!(
                reserve_count(&state, kicking_team),
                reserves_before_cap + 4,
                "at most fixed number of re-arranged players can be chosen"
            );
            assert_eq!(state.get_players_on_pitch_in_team(kicking_team).count(), pitch_before_cap - 4);
            for pos in unselected_open {
                assert!(
                    !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, pos)),
                    "cannot keep choosing more than the fixed number"
                );
            }

            // Scenario B: fewer than the fixed number, reserve-first, legal setup squares, and swapping.
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let down_id = state
                .get_players_on_pitch_in_team(kicking_team)
                .filter(|player| player.status == PlayerStatus::Up)
                .filter(|player| state.get_tz_on(player.id) == 0)
                .map(|player| player.id)
                .next()
                .unwrap();
            let down_pos = state.get_player_unsafe(down_id).position;
            state.get_mut_player_unsafe(down_id).status = PlayerStatus::Down;

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d6(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = positions_for_action(&state.available_actions, PosAT::SelectPosition);
            assert!(
                !selectable.contains(&down_pos),
                "open player must be standing and not marked"
            );

            let selected: Vec<Position> = selectable.into_iter().take(2).collect();
            let reserves_before = reserve_count(&state, kicking_team);
            let pitch_before = state.get_players_on_pitch_in_team(kicking_team).count();
            for pos in selected {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            assert!(
                state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "should be possible to choose fewer players than the fixed number"
            );
            state.step_simple(SimpleAT::EndSetup);

            assert_eq!(
                reserve_count(&state, kicking_team),
                reserves_before + 2,
                "all selected players must be in reserves before any placement"
            );
            assert_eq!(state.get_players_on_pitch_in_team(kicking_team).count(), pitch_before - 2);

            let anchored_positions: HashSet<Position> = state
                .get_players_on_pitch_in_team(kicking_team)
                .map(|player| player.position)
                .collect();
            let legal_placements = positions_for_action(&state.available_actions, PosAT::SelectPosition);
            assert!(!legal_placements.is_empty());
            for pos in &anchored_positions {
                assert!(
                    !legal_placements.contains(pos),
                    "anchored players should make their occupied squares illegal setup positions"
                );
            }
            for pos in &legal_placements {
                assert!(state.get_player_id_at(*pos).is_none());
            }

            let first_pos = legal_placements[0];
            state.step_positional(PosAT::SelectPosition, first_pos);
            let second_pos = positions_for_action(&state.available_actions, PosAT::SelectPosition)
                .into_iter()
                .find(|pos| *pos != first_pos && state.get_player_id_at(*pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, second_pos);

            let first_id = state.get_player_id_at(first_pos).unwrap();
            let second_id = state.get_player_id_at(second_pos).unwrap();
            state.step_positional(PosAT::SelectPosition, first_pos);

            let anchored_after_fielding = state
                .get_players_on_pitch_in_team(kicking_team)
                .map(|player| player.position)
                .find(|pos| *pos != first_pos && *pos != second_pos)
                .unwrap();
            assert!(
                !state.is_legal_action(&Action::Positional(
                    PosAT::SelectPosition,
                    anchored_after_fielding
                )),
                "swapping should not include non-selected players"
            );
            assert!(state.is_legal_action(&Action::Positional(PosAT::SelectPosition, second_pos)));
            state.step_positional(PosAT::SelectPosition, second_pos);

            assert_eq!(state.get_player_id_at(first_pos), Some(second_id));
            assert_eq!(state.get_player_id_at(second_pos), Some(first_id));
        }
}
    
    #[test]
    fn kickoff_high_kick() {
         let mut state: GameState = GameStateBuilder::new_at_kickoff();
         // ball fixes
         state.fixes.fix_d8_direction(Direction::up()); // scatter direction
         state.fixes.fix_d6(5); // scatter length
    
         // kickoff event fix
         state.fixes.fix_d6(1);
         state.fixes.fix_d6(4);
    
         state.step_simple(SimpleAT::KickoffAimMiddle);
    
         let ball_pos = state.get_ball_position().unwrap();
         assert!(matches!(state.ball, BallState::InAir(_)));
    
         assert!(state.home_to_act());
        let receiving_team = other_team(state.info.kicking_this_drive);
        let legal_positions: Vec<Position> = state
            .get_players_on_pitch_in_team(receiving_team)
            .filter(|p| p.status == PlayerStatus::Up)
            .filter(|p| state.get_tz_on(p.id) == 0)
            .map(|p| p.position)
            .collect();
        assert!(!legal_positions.is_empty());
        for pos in &legal_positions {
            let action = Action::Positional(PosAT::SelectPosition, *pos);
            assert!(state.available_actions.is_legal_action(action));
        }

        let catcher_start_pos = legal_positions[0];
         let catcher_id = state.get_player_id_at(catcher_start_pos).unwrap();
    
         state.fixes.fix_d6(6); // fix the roll for the catch
        state.step_positional(PosAT::SelectPosition, legal_positions[0]);
    
         assert_eq!(state.get_player_id_at(ball_pos).unwrap(), catcher_id);
         assert_eq!(state.get_player_id_at(catcher_start_pos), None);
    
         match state.ball {
             BallState::Carried(id) => {
                 assert_eq!(id, catcher_id);
             }
             _ => panic!("ball should be carried"),
         }
    
         assert!(state.home_to_act());
    }
    //
    // #[test]
    // fn kickoff_cheering_fans() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(5);
    //     // TODO: Implement prayers to nuffle...
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    // }
    //
    // #[test]
    // fn kickoff_brilliant_coaching() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(1);
    //
    //     state.fixes.fix_d6(5); //fix home brilliant coaching roll
    //     state.fixes.fix_d6(6); //fix away brilliant coaching roll
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
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(1);
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
//}

}

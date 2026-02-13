use crate::core::model::ProcInput;
use crate::core::model::{
    Action, AvailableActions, Coord, DugoutPlace, PlayerID,
    Position, ProcState, Procedure, Result, TeamType, HEIGHT_,
    LINE_OF_SCRIMMAGE_Y_RANGE, NORTH_WING_Y_RANGE, SOUTH_WING_Y_RANGE
};
use crate::core::table::{SimpleAT, PlayerRole, PosAT};
use crate::core::gamestate::GameState;
use super::AnyProc;

use serde::{Deserialize, Serialize};

use rand::Rng;
use std::ops::RangeInclusive;

struct SetupCounts {
    on_pitch: usize,
    los: usize,
    north: usize,
    south: usize,
}

fn is_los_position(pos: Position, line_x: Coord) -> bool {
    pos.x == line_x && LINE_OF_SCRIMMAGE_Y_RANGE.contains(&pos.y)
}

fn is_north_wing_position(pos: Position) -> bool {
    NORTH_WING_Y_RANGE.contains(&pos.y)
}

fn is_south_wing_position(pos: Position) -> bool {
    SOUTH_WING_Y_RANGE.contains(&pos.y)
}

fn is_on_team_half(pos: Position, team: TeamType, line_x: Coord) -> bool {
    match team {
        TeamType::Home => pos.x >= line_x,
        TeamType::Away => pos.x <= line_x,
    }
}

fn count_setup_positions(state: &GameState, team: TeamType, line_x: Coord) -> SetupCounts {
    let mut counts = SetupCounts {
        on_pitch: 0,
        los: 0,
        north: 0,
        south: 0,
    };

    for pos in state.get_players_on_pitch_in_team(team).map(|p| p.position) {
        counts.on_pitch += 1;
        if is_los_position(pos, line_x) {
            counts.los += 1;
        } else if is_north_wing_position(pos) {
            counts.north += 1;
        } else if is_south_wing_position(pos) {
            counts.south += 1;
        }
    }

    counts
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
        game_state: &GameState,
        x_range: RangeInclusive<Coord>,
        y_range: RangeInclusive<Coord>,
    ) -> Position {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(x_range.clone());
            let y = rng.gen_range(y_range.clone());
            if game_state.get_player_id_at_coord(x, y).is_none() {
                return Position { x, y };
            }
        }
    }
    
    fn get_legal_setup_positions(&self, game_state: &GameState) -> Vec<Position> {
        let line_x = game_state.get_line_of_scrimage_x(self.team);
        let counts = count_setup_positions(game_state, self.team, line_x);
        let num_players_on_bench = game_state
            .get_dugout()
            .filter(|player| player.stats.team == self.team && player.place == DugoutPlace::Reserves)
            .count();

        let num_available_players = counts.on_pitch + num_players_on_bench;
        let min_on_pitch = 11.min(num_available_players);
        let min_los = 3.min(num_available_players);
        let remaining_to_place = min_on_pitch.saturating_sub(counts.on_pitch);

        if remaining_to_place == 0 {
            return Vec::new();
        }

        let empty_los_squares = LINE_OF_SCRIMMAGE_Y_RANGE
            .clone()
            .filter(|&y| {
                let pos = Position::new((line_x, y));
                !pos.is_out() && game_state.get_player_id_at(pos).is_none()
            })
            .count();

        let mut candidates = Vec::new();
        for pos in Position::all_positions() {
            if pos.is_out()
                || !is_on_team_half(pos, self.team, line_x)
                || game_state.get_player_id_at(pos).is_some()
            {
                continue;
            }

            let is_los = is_los_position(pos, line_x);
            let new_los = counts.los + usize::from(is_los);
            let new_north = counts.north + usize::from(is_north_wing_position(pos));
            let new_south = counts.south + usize::from(is_south_wing_position(pos));

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

    fn build_available_actions(&self, game_state: &GameState) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        let legal_positions = self.get_legal_setup_positions(game_state);
        if !legal_positions.is_empty() {
            aa.insert_positional(PosAT::SelectPosition, legal_positions);
        }
        if game_state.is_setup_legal(self.team) {
            aa.insert_simple(SimpleAT::EndSetup);
        }
        if game_state.get_players_on_pitch_in_team(self.team).next().is_none() {
            aa.insert_simple(SimpleAT::SetupLine);
        }
        ProcState::NeedAction(aa)
    }

    pub fn random_setup(&self, game_state: &mut GameState) {

        #[allow(clippy::needless_collect)]
        let players: Vec<PlayerID> = game_state
            .get_dugout()
            .take(11)
            .filter(|dplayer| dplayer.stats.team == self.team)
            .map(|p| p.id)
            .collect();

        let mut ids = players.into_iter();
        let los_x = game_state.get_line_of_scrimage_x(self.team);
        let los_x_range = los_x..=los_x;
        let x_range = match self.team {
            TeamType::Home => los_x..=crate::core::model::WIDTH_ - 2,
            TeamType::Away => 1..=los_x,
        };
        for _ in 0..3 {
            if let Some(id) = ids.next() {
                let p = Setup::get_empty_pos_in_box(
                    game_state,
                    los_x_range.clone(),
                    LINE_OF_SCRIMMAGE_Y_RANGE.clone(),
                );
                game_state.field_dugout_player(id, p);
            }
        }
        for id in ids {
            let p = Setup::get_empty_pos_in_box(
                game_state,
                x_range.clone(),
                LINE_OF_SCRIMMAGE_Y_RANGE.clone(),
            );
            game_state.field_dugout_player(id, p);
        }
    }

    fn setup_line(&self, game_state: &mut GameState) -> Result<()> {
        //unfield all players
        let player_ids = game_state
            .get_players_on_pitch_in_team(self.team)
            .map(|p| p.id)
            .collect::<Vec<_>>();
        for id in player_ids {
            game_state.unfield_player(id, DugoutPlace::Reserves)?;
        }
        let mut linemen_pos = vec![(0, 0), (0, -1), (0, 1), (0, -3), (0, 3)];
        let mut blitzer_pos = vec![(0, -2), (0, 2)];
        let mut catcher_pos = vec![(2, 2), (2, -2)];
        let mut thrower_pos = vec![(6, 3), (6, -3)];
        #[allow(clippy::needless_collect)]
        let players: Vec<PlayerID> = game_state
            .get_dugout()
            .filter(|dplayer| dplayer.stats.team == self.team)
            .filter(|dplayer| dplayer.place == DugoutPlace::Reserves)
            .map(|p| p.id)
            .collect();
        let x_delta_sign = if self.team == TeamType::Home { 1 } else { -1 };
        let middle_x = game_state.get_line_of_scrimage_x(self.team);
        let middle_y = HEIGHT_ / 2;
        for id in players {
            let player = game_state.get_dugout_player(id).unwrap();
            let (dx, dy) = {
                match player.stats.role {
                    PlayerRole::Blitzer if !blitzer_pos.is_empty() => blitzer_pos.pop().unwrap(),
                    PlayerRole::Thrower if !thrower_pos.is_empty() => thrower_pos.pop().unwrap(),
                    PlayerRole::Catcher if !catcher_pos.is_empty() => catcher_pos.pop().unwrap(),
                    PlayerRole::Lineman if !linemen_pos.is_empty() => linemen_pos.pop().unwrap(),
                    _ => continue,
                }
            };
            let position = Position::new((middle_x + dx * x_delta_sign, middle_y + dy));
            game_state.log(format!(
                "fielding {:?} {:?} at {:?}",
                player.stats.role, player.stats.team, position
            ));
            game_state.field_dugout_player(id, position)
        }
        Ok(())
    }
}

impl Procedure for Setup {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        if input == ProcInput::Nothing {
            return self.build_available_actions(game_state);
        }

        match input {
            ProcInput::Action(Action::Simple(SimpleAT::SetupLine)) => {
                self.setup_line(game_state).unwrap();
                self.build_available_actions(game_state)
            }

            ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => ProcState::Done,

            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let player_id = game_state
                    .get_dugout()
                    .filter(|player| {
                        player.stats.team == self.team && player.place == DugoutPlace::Reserves
                    })
                    .map(|player| player.id)
                    .next();
                if let Some(id) = player_id {
                    game_state.field_dugout_player(id, pos);
                }
                self.build_available_actions(game_state)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::zip, panic::{catch_unwind, AssertUnwindSafe}};
    use super::*;
    use crate::core::{gamestate::GameStateBuilder, model::{DugoutPlayer, PlayerStats}};

    fn add_reserves(state: &mut GameState, team: TeamType, count: usize) {
        for _ in 0..count {
            state.dugout_add_new_player(PlayerStats::new_lineman(team), DugoutPlace::Reserves);
        }
    }

    fn positions_for_action(aa: &AvailableActions, action: PosAT) -> Vec<Position> {
        let Some(positions) = aa.get_positional() else {
            return Vec::new();
        };
        Position::all_positions()
            .filter(|pos| positions[*pos].contains(&action))
            .collect()
    }

    #[test]
    fn half_constraint_for_both_teams() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        add_reserves(&mut state, TeamType::Away, 11);

        for team in [TeamType::Home, TeamType::Away] {
            let setup = Setup{team};
            let positions = setup.get_legal_setup_positions(&state);
            assert!(!positions.is_empty());
            let line_x = state.get_line_of_scrimage_x(team);
            for pos in positions {
                assert!(!pos.is_out());
                match team {
                    TeamType::Home => assert!(pos.x >= line_x),
                    TeamType::Away => assert!(pos.x <= line_x),
                }
            }
        }
    }

    #[test]
    fn setup_specific_player_from_dugout_reserve() {
        let mut state: GameState = GameStateBuilder::new_at_setup();
        let team: TeamType = TeamType::Away;
        let mut setup: Setup = Setup { team };

        let middle_x = state.get_line_of_scrimage_x(team);
        let middle_y = HEIGHT_ / 2;
        let pitch_pos: Position = Position { x: middle_x, y: middle_y };

        let player_id = 2;

        // build legal action squares
        let _ = setup.step(&mut state,ProcInput::Nothing);

        // should be able to select select_player to place at pitch_pos
        let _ = setup.step(
            &mut state, 
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pitch_pos))
        );

        assert!(player_id == state.get_player_at(pitch_pos).unwrap().id);
    }

    #[test]
    fn change_setup_square_for_already_fielded_player() {

    }

    #[test]
    fn during_setup_swap_position_of_dugout_reserve_player_with_fielded_player() {

    }

    #[test]
    fn during_setup_swap_positions_of_already_fielded_players() {

    }

    #[test]
    fn wing_cap_blocks_north_wing_positions() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let north_y = *NORTH_WING_Y_RANGE.start();

        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x + 1, north_y)),
            )
            .unwrap();
        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x + 1, north_y + 1)),
            )
            .unwrap();

        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();
        for offset in 0..3 {
            let offset = offset as Coord;
            state
                .add_new_player_to_field(
                    PlayerStats::new_lineman(TeamType::Home),
                    Position::new((line_x, los_y + offset)),
                )
                .unwrap();
        }

        let setup = Setup { team: TeamType::Home };
        let positions = setup.get_legal_setup_positions(&state);
        assert!(positions.iter().all(|pos| !NORTH_WING_Y_RANGE.contains(&pos.y)));
    }

    #[test]
    fn los_feasibility_requires_los_squares() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 2);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();

        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x, los_y)),
            )
            .unwrap();

        let setup = Setup { team: TeamType::Home };
        let positions = setup.get_legal_setup_positions(&state);
        assert!(!positions.is_empty());
        assert!(positions.iter().all(|pos| is_los_position(*pos, line_x)));
    }

    #[test]
    fn placed_player_who_change_position_during_setup_keeps_same_id() {
        let mut state: GameState = GameStateBuilder::new_at_setup();
        let team: TeamType = TeamType::Away;
        let mut setup: Setup = Setup { team };

        //place player
        let middle_x = state.get_line_of_scrimage_x(team);
        let middle_y = HEIGHT_ / 2;
        let pos: Position = Position { x: middle_x, y: middle_y };
        let player_id: PlayerID = state.add_new_player_to_field(PlayerStats::new_lineman(team), pos).unwrap();

        // setup step, choose fielded player to replace it, should keep same id
        let ProcState::NeedAction(_aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)),
        ) else {
            panic!("Expected NeedAction after building available actions.");
        };
        assert!(state.get_players_on_pitch_in_team(team).count() == 1);
        let should_be_same_id: Option<PlayerID> = state.get_player_id_at(pos);
        assert!(should_be_same_id.is_some());
        assert!(should_be_same_id.unwrap() == player_id);
    }

    #[test]
    fn los_requirement_satisfied_allows_non_los_positions() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 1);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();

        for offset in 0..3 {
            let offset = offset as Coord;
            state
                .add_new_player_to_field(
                    PlayerStats::new_lineman(TeamType::Home),
                    Position::new((line_x, los_y + offset)),
                )
                .unwrap();
        }

        let setup = Setup { team: TeamType::Home };
        let positions = setup.get_legal_setup_positions(&state);
        assert!(positions.iter().any(|pos| !is_los_position(*pos, line_x)));
    }

    #[test]
    fn occupied_positions_are_excluded() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let occupied_pos = Position::new((line_x + 1, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));

        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                occupied_pos,
            )
            .unwrap();

        let setup = Setup { team: TeamType::Home };
        let positions = setup.get_legal_setup_positions(&state);
        assert!(!positions.contains(&occupied_pos));
    }

    #[test]
    fn test_setup_preconfigured_formations() {
        let mut state: GameState = GameStateBuilder::new_at_setup();
        //away as defense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);
        //home as offense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);

        for team in [TeamType::Home, TeamType::Away] {
            let middle_x = state.get_line_of_scrimage_x(team);
            let middle_y = HEIGHT_ / 2;

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

    #[test]
    fn manual_setup_places_players_until_done() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 3);

        let mut setup = Setup { team: TeamType::Home };

        let ProcState::NeedAction(aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction on initial setup prompt.");
        };

        let positions = positions_for_action(&aa, PosAT::SelectPosition);
        assert!(!positions.is_empty());
        assert!(aa.get_simple().contains(&SimpleAT::SetupLine));
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, positions[0])),
        ) else {
            panic!("Expected NeedAction after placing first player.");
        };

        assert_eq!(state.get_players_on_pitch_in_team(TeamType::Home).count(), 1);
        assert!(!aa.get_simple().contains(&SimpleAT::SetupLine));
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let positions = positions_for_action(&aa, PosAT::SelectPosition);
        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, positions[0])),
        ) else {
            panic!("Expected NeedAction after placing second player.");
        };

        assert_eq!(state.get_players_on_pitch_in_team(TeamType::Home).count(), 2);
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let positions = positions_for_action(&aa, PosAT::SelectPosition);
        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, positions[0])),
        ) else {
            panic!("Expected NeedAction after placing third player.");
        };

        assert_eq!(state.get_players_on_pitch_in_team(TeamType::Home).count(), 3);
        assert!(aa.get_simple().contains(&SimpleAT::EndSetup));
        assert!(positions_for_action(&aa, PosAT::SelectPosition).is_empty());

        let reserves_left = state
            .get_dugout()
            .filter(|player| player.stats.team == TeamType::Home)
            .filter(|player| player.place == DugoutPlace::Reserves)
            .count();
        assert_eq!(reserves_left, 0);
    }
}

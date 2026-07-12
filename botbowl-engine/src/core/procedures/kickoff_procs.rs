use crate::core::model::ProcInput;
use std::ops::RangeInclusive;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction, DugoutPlace, PlayerID, Position, ProcState,
    Procedure, Result, TeamType, Weather,
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
        let dims = game_state.board_dims;
        let x_delta_sign = if self.team == TeamType::Home { 1 } else { -1 };
        let middle_x = game_state.get_line_of_scrimage_x(self.team);
        let middle_y = dims.height / 2;
        // Clamp formation offsets so they stay on-pitch and on our own half on
        // small boards. On the full pitch these bounds are wider than any offset,
        // so the historical formation is reproduced exactly.
        let max_dx = dims.width / 2 - 2;
        let max_dy = (middle_y - 1).min(dims.height - 2 - middle_y);
        let mut fielded = 0usize;
        for id in players {
            if fielded >= dims.team_size {
                break;
            }
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
            let (dx, dy) = (dx.min(max_dx), dy.clamp(-max_dy, max_dy));
            let mut position = Position::new((middle_x + dx * x_delta_sign, middle_y + dy));
            // Clamping can collide players onto the same square on tiny boards;
            // fall back to the nearest free square on our own half.
            if game_state.is_out(position) || game_state.get_player_id_at(position).is_some() {
                position = Self::first_free_own_half(game_state, middle_x, x_delta_sign);
            }
            fielded += 1;
            let player = game_state.get_dugout_player(id).unwrap();
            crate::game_log!(
                game_state,
                "fielding {:?} {:?} at {:?}",
                player.stats.role,
                player.stats.team,
                position
            );
            game_state.field_dugout_player(id, position)
        }
        Ok(())
    }
    /// First empty square on `team`'s own half, scanning from the line of
    /// scrimmage backwards toward the end zone. Used as a collision fallback
    /// when clamped formation offsets overlap on small boards.
    fn first_free_own_half(game_state: &GameState, los_x: Coord, x_delta_sign: Coord) -> Position {
        let xs: Vec<Coord> = if x_delta_sign > 0 {
            (los_x..=game_state.board_dims.width - 2).collect()
        } else {
            (1..=los_x).rev().collect()
        };
        for x in xs {
            for y in 1..=game_state.board_dims.height - 2 {
                if game_state.get_player_id_at_coord(x, y).is_none() {
                    return Position { x, y };
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
            aa.insert_simple(SimpleAT::SetupLine);
            return ProcState::NeedAction(aa);
        }

        match input {
            ProcInput::Action(Action::Simple(SimpleAT::SetupLine)) => {
                self.setup_line(game_state).unwrap();
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

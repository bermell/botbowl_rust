use crate::core::model::ProcInput;
use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction, DugoutPlace, PlayerID,
    PlayerStatus, Position, ProcState, Procedure, Weather
};
use crate::core::procedures::ball_procs;
use crate::core::procedures::setup_procs;
use crate::core::table::*;

use crate::core::gamestate::GameState;

use super::AnyProc;
use std::collections::HashSet;
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
pub struct SolidDefence {}
impl SolidDefence {
    pub fn new() -> AnyProc {
        AnyProc::SolidDefence(SolidDefence {})
    }
}
impl Procedure for SolidDefence {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
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
    #[test]
    fn kickoff_solid_defence() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        let kicking_team = state.info.kicking_this_drive;
        let open_before: Vec<PlayerID> = state
            .get_players_on_pitch_in_team(kicking_team)
            .filter(|p| p.status == PlayerStatus::Up)
            .filter(|p| state.get_tz_on(p.id) == 0)
            .map(|p| p.id)
            .collect();
        let initial_positions: HashSet<Position> = state
            .get_players_on_pitch_in_team(kicking_team)
            .map(|p| p.position)
            .collect();

        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length
    
         // kickoff event fix solid defence
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(3); 

        state.fixes.fix_d6(6); //fix number of re-arranged players (d3+3)
        state.step_simple(SimpleAT::KickoffAimMiddle);
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

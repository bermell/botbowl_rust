// extern crate rust_bb;
// use rust_bb::core::model::*;
// use rust_bb::core::table::*;
// use rust_bb::core::{
//     gamestate::{GameState, GameStateBuilder},
//     model::{Action, Position},
//     table::PosAT,
// };
//
// #[test]
// fn kickoff_get_the_ref() {
//     let mut state: GameState = GameStateBuilder::new_at_kickoff();
//     // ball fixes
//     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
//     state.fixes.fix_d6(5); // scatter length
//
//     // kickoff event fix
//     state.fixes.fix_d6(1);
//     state.fixes.fix_d6(1);
//     //
//     state.step_simple(SimpleAT::KickoffAimMiddle);
//
//     assert_eq!(state.home.bribes, 1);
//     assert_eq!(state.away.bribes, 1);
// }
// #[test]
// fn kickoff_timeout() {
//     // TODO: add test in turns 6 7 8, should gain a turn
//     let mut state: GameState = GameStateBuilder::new_at_kickoff();
//     // ball fixes
//     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
//     state.fixes.fix_d6(5); // scatter length
//
//     // kickoff event fix
//     state.fixes.fix_d6(1);
//     state.fixes.fix_d6(2);
//
//     state.step_simple(SimpleAT::KickoffAimMiddle);
//
//     assert_eq!(state.info.home_turn, 2);
//     assert_eq!(state.info.away_turn, 2);
// }
// #[test]
// fn kickoff_solid_defence() {
//     let mut state: GameState = GameStateBuilder::new_at_kickoff();
//     // ball fixes
//     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
//     state.fixes.fix_d6(5); // scatter length
//
//     // kickoff event fix
//     state.fixes.fix_d6(1);
//     state.fixes.fix_d6(3);
//
//     state.fixes.fix_d6(6); //fix number of re-arranged players (d3+3)
//     state.step_simple(SimpleAT::KickoffAimMiddle);
//
//     // TODO: haven't implemented the setup yet
// }
//
// #[test]
// fn kickoff_high_kick() {
//     let mut state: GameState = GameStateBuilder::new_at_kickoff();
//     // ball fixes
//     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
//     state.fixes.fix_d6(5); // scatter length
//
//     // kickoff event fix
//     state.fixes.fix_d6(1);
//     state.fixes.fix_d6(4);
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
//     state.fixes.fix_d6(6); // fix the roll for the catch
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
// }

use crate::core::gamestate::GameStateBuilder;

use crate::core::gamestate::GameState;

pub mod bots;
pub mod core;

pub fn standard_state() -> GameState {
    GameStateBuilder::new()
        .add_home_players(&[(1, 2), (2, 2), (3, 1)])
        .add_away_players(&[(5, 2), (5, 5), (2, 3)])
        .add_ball((3, 2))
        .build()
}

#[cfg(test)]
mod tests {

    use crate::core::dices::D6Target;
    use crate::core::dices::D6;
    use crate::core::model::*;
    use crate::core::pathing::CustomIntoIter;
    use crate::core::pathing::NodeIteratorItem;
    use crate::core::table::*;
    use crate::core::{
        gamestate::{GameState, GameStateBuilder},
        model::{Action, DugoutPlace, PlayerStats, Position, TeamType, HEIGHT_, WIDTH_},
        pathing::{PathFinder, PathingEvent},
        table::PosAT,
    };
    use crate::standard_state;
    use ansi_term::Colour::Red;
    use itertools::Either;
    use std::{
        collections::{HashMap, HashSet},
        iter::{repeat_with, zip},
    };

    #[test]
    fn draw_board() {
        let _state = standard_state();

        println!(
            "This is in red: {}",
            Red.strikethrough().paint("a red string")
        );
        // use unique greek letter for each player, color blue and red for home and away
        // use two letters for each position
        // use strikethrough for down
        // use darker shade for used.
        // use  ▒▒▒▒ for unoccupied positions
    }
    #[test]
    fn field_a_player() -> Result<()> {
        let mut state = standard_state();
        let player_stats = PlayerStats::new_lineman(TeamType::Home);
        let position = Position::new((10, 10));

        assert!(state.get_player_id_at(position).is_none());

        let id = state
            .add_new_player_to_field(player_stats, position)
            .unwrap();

        assert_eq!(state.get_player_id_at(position), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, position);

        state.unfield_player(id, DugoutPlace::Reserves)?;

        assert!(state.get_player_id_at(position).is_none());
        Ok(())
    }

    #[test]
    fn long_move_action() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let move_target = Position::new((2, 5));

        assert!(state.get_player_at(starting_pos).is_some());
        assert!(state.get_player_at(move_target).is_none());

        state.step_positional(PosAT::StartMove, starting_pos);

        state.fixes.fix_d6(6);
        state.fixes.fix_d6(6);
        state.fixes.fix_d6(6);
        state.step_positional(PosAT::Move, move_target);

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step_simple(SimpleAT::EndPlayerTurn);

        assert!(state.get_player_at(move_target).unwrap().used);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartMove, move_target)));

        Ok(())
    }

    #[test]
    fn start_move_action() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let move_target = Position::new((4, 1));

        assert!(state.get_player_at(starting_pos).is_some());
        assert!(state.get_player_at(move_target).is_none());

        state.step_positional(PosAT::StartMove, starting_pos);
        state.step_positional(PosAT::Move, move_target);

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step_simple(SimpleAT::EndPlayerTurn);

        assert!(state.get_player_at(move_target).unwrap().used);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartMove, move_target)));

        Ok(())
    }

    #[test]
    fn pathing() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let id = state.get_player_id_at(starting_pos).unwrap();
        state.step_positional(PosAT::StartMove, starting_pos);
        let paths = PathFinder::player_paths(&state, id)?;

        let mut errors = Vec::new();

        for x in 1..8 {
            for y in 1..8 {
                let pos = Position::new((x, y));
                match (state.get_player_id_at(pos), &paths[pos]) {
                    (Some(_), None) => (),
                    (None, Some(_)) => (),
                    (Some(_), Some(_)) => {
                        errors.push(format!("Found path already occupied square ({},{})", x, y))
                    }
                    (None, None) => errors.push(format!("Missing a path to ({},{})!", x, y)),
                }
            }
        }
        let no_errors: Vec<String> = Vec::new();
        assert_eq!(no_errors, errors);
        Ok(())
    }

    #[test]
    fn pathing_probs() -> Result<()> {
        let starting_pos = Position::new((3, 2));
        let state = GameStateBuilder::new()
            .add_home_player(starting_pos)
            .add_away_players(&[(1, 3), (3, 3), (4, 2)])
            .build();

        let id = state.get_player_id_at(starting_pos).unwrap();

        let paths = PathFinder::player_paths(&state, id)?;

        let mut pos_to_prob: HashMap<(usize, usize), Option<f32>> = HashMap::new();
        pos_to_prob.insert((1, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((1, 2), Some(2.0 / 3.0));
        pos_to_prob.insert((1, 3), None);
        pos_to_prob.insert((1, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((2, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((2, 2), Some(2.0 / 3.0));
        pos_to_prob.insert((2, 3), Some(1.0 / 3.0));
        pos_to_prob.insert((2, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((3, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((3, 2), None);
        pos_to_prob.insert((3, 3), None);
        pos_to_prob.insert((3, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((4, 1), Some(1.0 / 2.0));
        pos_to_prob.insert((4, 2), None);
        pos_to_prob.insert((4, 3), Some(1.0 / 3.0));
        pos_to_prob.insert((4, 4), Some(2.0 / 9.0));

        let mut errors = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for x in 1..5 {
            for y in 1..5 {
                match (pos_to_prob.get(&(x, y)).unwrap(), paths.get(x, y)) {
                    (Some(correct_prob), Some(path))
                        if (*correct_prob - path.prob).abs() > 0.001 =>
                    {
                        errors.push(format!(
                            "Path to ({}, {}) has wrong prob. \nExpected prob: {}\nGot prob: {}\n",
                            x, y, *correct_prob, path.prob
                        ))
                    }
                    (Some(correct_prob), Some(path))
                        if (*correct_prob - path.prob).abs() <= 0.001 => {}
                    (None, None) => (),
                    (Some(_), None) => errors.push(format!("No path to ({}, {})", x, y)),
                    (None, Some(path)) => errors.push(format!(
                        "There shouldn't be a path to ({}, {}). Found: {:?}",
                        x, y, path
                    )),
                    _ => (),
                }
            }
        }

        let no_errors: Vec<String> = Vec::new();
        assert_eq!(no_errors, errors);

        Ok(())
    }

    #[test]
    fn one_long_path() -> Result<()> {
        let starting_pos = Position::new((1, 1));
        let state = GameStateBuilder::new()
            .add_home_player(starting_pos)
            .add_away_players(&[(1, 2), (2, 3), (2, 4), (5, 3), (6, 4)])
            .add_ball((4, 6))
            .build();
        let id = state.get_player_id_at(starting_pos).unwrap();
        let paths = PathFinder::player_paths(&state, id)?;

        let expected_steps: Vec<NodeIteratorItem> = vec![
            Either::Left(Position::new((2, 1))),
            Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
            Either::Left(Position::new((3, 1))),
            Either::Right(PathingEvent::Dodge(D6Target::ThreePlus)),
            Either::Left(Position::new((3, 2))),
            Either::Left(Position::new((4, 3))),
            Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
            Either::Left(Position::new((4, 4))),
            Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
            Either::Left(Position::new((4, 5))),
            Either::Right(PathingEvent::Dodge(D6Target::ThreePlus)),
            Either::Left(Position::new((4, 6))),
            Either::Right(PathingEvent::GFI(D6Target::TwoPlus)),
            Either::Right(PathingEvent::Pickup(D6Target::ThreePlus)),
        ];

        let expected_prob = 0.03086;
        let path = paths.get(4, 6).clone().unwrap();

        for (i, (expected, actual)) in zip(expected_steps, path.iter()).enumerate() {
            if expected != actual {
                panic!("Step {}: {:?} != {:?}", i, expected, actual);
            }
        }

        assert!((expected_prob - path.prob).abs() < 0.0001);

        Ok(())
    }

    #[test]
    fn rng_seed_in_gamestate() -> Result<()> {
        let mut state = standard_state();
        state.rng_enabled = true;
        let seed = 5;
        state.set_seed(seed);

        fn get_random_rolls(state: &mut GameState) -> Vec<D6> {
            repeat_with(|| state.get_d6_roll()).take(200).collect()
        }

        let numbers: Vec<D6> = get_random_rolls(&mut state);
        let different_numbers = get_random_rolls(&mut state);
        assert_ne!(numbers, different_numbers);

        state.set_seed(seed);
        let same_numbers = get_random_rolls(&mut state);

        assert_eq!(numbers, same_numbers);

        Ok(())
    }

    #[test]
    fn fixed_rolls() {
        let mut state = standard_state();
        state.rng_enabled = true;
        let fixes = vec![1, 3, 5, 2, 4, 6];
        fixes.iter().for_each(|val| state.fixes.fix_d6(*val));

        let rolls: Vec<u8> = repeat_with(|| state.get_d6_roll() as u8)
            .take(fixes.len())
            .collect();
        assert_eq!(fixes, rolls);
    }

    #[test]
    fn movement() -> Result<()> {
        let mut state = standard_state();
        state.step_positional(PosAT::StartMove, Position::new((3, 1)));
        Ok(())
    }
}

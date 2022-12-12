use crate::core::gamestate::GameStateBuilder;

use crate::core::gamestate::GameState;

pub mod core;

// fn draw_board(game_state: &GameState) {
//     :
// }

pub fn standard_state() -> GameState {
    GameStateBuilder::new()
        .add_home_players(&[(1, 2), (2, 2), (3, 1)])
        .add_away_players(&[(5, 2), (5, 5), (2, 3)])
        .add_ball((3, 2))
        .build()
}

fn main() {
    let state = standard_state();
    println!("Hello world!");
    //draw_board(&state);
}

#[cfg(test)]
mod tests {

    use crate::core::dices::BlockDice;
    use crate::core::dices::D6Target;
    use crate::core::dices::D6;
    use crate::core::dices::D8;
    use crate::core::model::*;
    use crate::core::table::*;
    use crate::core::{
        gamestate::{GameState, GameStateBuilder},
        model::{Action, DugoutPlace, PlayerStats, Position, TeamType, HEIGHT_, WIDTH_},
        pathing::{PathFinder, Roll},
        table::PosAT,
    };
    use crate::standard_state;
    use ansi_term::Colour::Red;
    use std::{
        collections::{HashMap, HashSet},
        iter::{repeat_with, zip},
    };

    #[test]
    fn state_from_str() {
        let mut field = "".to_string();
        field += " aa\n";
        field += " Aa\n";
        field += "h  \n";
        let first_pos = Position::new((5, 1));
        let state = GameStateBuilder::new().add_str(first_pos, &field).build();
        assert_eq!(
            state
                .get_player_at(Position::new((5, 3)))
                .unwrap()
                .stats
                .team,
            TeamType::Home
        );

        assert_eq!(
            state
                .get_player_at(Position::new((6, 2)))
                .unwrap()
                .stats
                .team,
            TeamType::Away
        );

        let id = state.get_player_id_at_coord(6, 2).unwrap();
        assert_eq!(state.ball, BallState::Carried(id));
    }

    #[test]
    fn crowd_chain_push() {
        let mut field = "".to_string();
        field += " aa\n";
        field += " aa\n";
        field += "h  \n";
        let first_pos = Position::new((5, 1));
        let mut state = GameStateBuilder::new().add_str(first_pos, &field).build();

        state.step_positional(PosAT::StartBlock, Position::new((5, 3)));
        state.blockdice_fixes.push_back(BlockDice::Push);
        state.step_positional(PosAT::Block, Position::new((6, 2)));
        state.step_simple(SimpleAT::SelectPush);

        state.step_positional(PosAT::Push, Position::new((6, 1)));
        state.d6_fixes.push_back(D6::One);
        state.d6_fixes.push_back(D6::One);

        state.step_positional(PosAT::FollowUp, Position::new((6, 2)));

        state.step_simple(SimpleAT::EndTurn);

        assert!(matches!(
            state.dugout_players.pop(),
            Some(DugoutPlayer {
                place: DugoutPlace::Reserves,
                stats: PlayerStats {
                    team: TeamType::Away,
                    ..
                },
            })
        ));
    }

    #[test]
    fn crowd_surf_ball_carrier() {
        let carrier_pos = Position::new((5, 1));
        let blocker_pos = Position::new((5, 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(blocker_pos)
            .add_away_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .build();

        state
            .step(Action::Positional(PosAT::StartBlock, blocker_pos))
            .unwrap();

        state.blockdice_fixes.push_back(BlockDice::Pow);

        state
            .step(Action::Positional(PosAT::Block, carrier_pos))
            .unwrap();
        state.step(Action::Simple(SimpleAT::SelectPow)).unwrap();

        state.d6_fixes.push_back(D6::One); //armor
        state.d6_fixes.push_back(D6::One); //armor
        state.d6_fixes.push_back(D6::Three); //throw in direction down
        state.d6_fixes.push_back(D6::One); //throw in length
        state.d6_fixes.push_back(D6::One); //throw in length
        state.d8_fixes.push_back(D8::Two); //bounce direction down

        state
            .step(Action::Positional(PosAT::FollowUp, carrier_pos))
            .unwrap();

        assert_eq!(state.ball, BallState::OnGround(Position::new((5, 4))));

        assert!(matches!(
            state.dugout_players.pop(),
            Some(DugoutPlayer {
                place: DugoutPlace::Reserves,
                stats: PlayerStats {
                    team: TeamType::Away,
                    ..
                },
            })
        ));
    }

    #[test]
    fn path_with_two_failures() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let target_pos = Position::new((3, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((1, 2)))
            .build();

        state.step(Action::Positional(PosAT::StartMove, start_pos))?;

        state.d6_fixes.push_back(D6::One);

        state.step(Action::Positional(PosAT::Move, target_pos))?;

        state.d6_fixes.push_back(D6::Four); //succeed first reroll
        state.d6_fixes.push_back(D6::One); //fail next dodge
        state.d6_fixes.push_back(D6::One); //armor
        state.d6_fixes.push_back(D6::One); //armor

        state.step(Action::Simple(SimpleAT::UseReroll))?;

        assert_eq!(
            state.get_player_at(target_pos).unwrap().status,
            PlayerStatus::Down
        );

        assert!(state.get_player_at(target_pos).unwrap().used);

        Ok(())
    }

    #[test]
    fn turn_order() -> Result<()> {
        let mut state = standard_state();
        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 0);
        assert_eq!(state.info.team_turn, TeamType::Home);

        state.step(Action::Simple(SimpleAT::EndTurn))?;

        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 1);
        assert_eq!(state.info.team_turn, TeamType::Away);

        state.step(Action::Simple(SimpleAT::EndTurn))?;

        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 2);
        assert_eq!(state.info.away_turn, 1);
        assert_eq!(state.info.team_turn, TeamType::Home);

        Ok(())
    }

    #[test]
    fn test_block_2d_bothdown_casualty() -> Result<()> {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(Position::new((5, 6)))
            .add_away_player(away_pos)
            .build();

        state.step(Action::Positional(PosAT::StartBlock, home_pos))?;
        state.blockdice_fixes.push_back(BlockDice::Pow);
        state.blockdice_fixes.push_back(BlockDice::BothDown);
        state.step(Action::Positional(PosAT::Block, away_pos))?;
        state.d6_fixes.push_back(D6::One); //away armor
        state.d6_fixes.push_back(D6::One); //away armor
        state.d6_fixes.push_back(D6::Five); //home armor
        state.d6_fixes.push_back(D6::Six); //home armor
        state.d6_fixes.push_back(D6::Six); //home injury
        state.d6_fixes.push_back(D6::Six); //home injury
        state.step(Action::Simple(SimpleAT::SelectBothDown))?;

        assert!(state.get_player_at(home_pos).is_none());
        assert!(matches!(
            state.dugout_players.pop(),
            Some(DugoutPlayer {
                place: DugoutPlace::Injuried,
                stats: PlayerStats {
                    team: TeamType::Home,
                    ..
                },
            })
        ));
        assert_eq!(
            state.get_player_at(away_pos).unwrap().status,
            PlayerStatus::Down
        );

        assert!(state.fixed_dice_empty());
        Ok(())
    }

    #[test]
    fn single_dice_block() -> Result<()> {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let push_pos = Position::new((6, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();

        state.step(Action::Positional(PosAT::StartBlock, home_pos))?;
        state.blockdice_fixes.push_back(BlockDice::Pow);
        state.step(Action::Positional(PosAT::Block, away_pos))?;
        state.step(Action::Simple(SimpleAT::SelectPow))?;
        state.step(Action::Positional(PosAT::Push, push_pos))?;
        state.d6_fixes.push_back(D6::One);
        state.d6_fixes.push_back(D6::One);
        state.step(Action::Positional(PosAT::FollowUp, away_pos))?;

        assert_eq!(
            state.get_player_at(push_pos).unwrap().status,
            PlayerStatus::Down
        );
        assert!(state.fixed_dice_empty());

        Ok(())
    }

    #[test]
    fn bounce_on_knockdown() -> Result<()> {
        let start_pos = Position::new((2, 2));
        let move_to = Position::new((3, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((1, 1)))
            .add_ball_pos(start_pos)
            .build();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);
        let id = state.get_player_id_at(start_pos).unwrap();

        assert_eq!(state.ball, BallState::Carried(id));
        state.step(Action::Positional(PosAT::StartMove, start_pos))?;

        state.d6_fixes.push_back(D6::Two);

        state.step(Action::Positional(PosAT::Move, move_to))?;

        state.d6_fixes.push_back(D6::One); //armor
        state.d6_fixes.push_back(D6::Five); //armor
        state.d8_fixes.push_back(d8_fix);

        state.step(Action::Simple(SimpleAT::DontUseReroll))?;

        let player = state.get_player_unsafe(id);
        assert!(player.used);
        assert_eq!(state.ball, BallState::OnGround(move_to + direction));

        Ok(())
    }

    #[test]
    fn failed_dodge_ko() -> Result<()> {
        let mut state = standard_state();
        let id = state.get_player_id_at_coord(2, 2).unwrap();
        assert!(state.dugout_players.is_empty());

        state.step(Action::Positional(PosAT::StartMove, Position::new((2, 2))))?;

        state.d6_fixes.push_back(D6::Two);
        state.step(Action::Positional(PosAT::Move, Position::new((2, 1))))?;

        state.d6_fixes.push_back(D6::Four); //armor
        state.d6_fixes.push_back(D6::Five); //armor
        state.d6_fixes.push_back(D6::Four); //injury
        state.d6_fixes.push_back(D6::Five); //injury
        state.step(Action::Simple(SimpleAT::DontUseReroll))?;

        assert!(state.d6_fixes.is_empty());
        assert!(state.get_player_id_at_coord(2, 1).is_none());
        assert!(state.get_players_on_pitch().all(|player| player.id != id));

        assert!(matches!(
            state.dugout_players.pop(),
            Some(DugoutPlayer {
                place: DugoutPlace::KnockOut,
                ..
            })
        ));

        assert!(state.dugout_players.is_empty());
        Ok(())
    }

    #[test]
    fn pickup_fail_and_bounce() -> Result<()> {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);

        state.step(Action::Positional(PosAT::StartMove, start_pos))?;
        state.d6_fixes.push_back(D6::Two); //fail pickup (3+)
        state.d8_fixes.push_back(d8_fix);
        state.step(Action::Positional(PosAT::Move, ball_pos))?;
        state.step(Action::Simple(SimpleAT::DontUseReroll))?;

        let player = state.get_player(id).unwrap();
        assert!(player.used);
        assert!(matches!(state.ball, BallState::OnGround(pos) if pos == ball_pos + direction));

        Ok(())
    }

    #[test]
    fn pickup_success() -> Result<()> {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();

        assert_eq!(state.ball, BallState::OnGround(ball_pos));

        state
            .get_mut_player(id)
            .unwrap()
            .stats
            .skills
            .insert(Skill::SureHands);

        state.step(Action::Positional(PosAT::StartMove, Position::new((1, 1))))?;

        state.d6_fixes.push_back(D6::Two); //fail first (3+)
        state.d6_fixes.push_back(D6::Three); //succeed on reroll (3+)
        state.step(Action::Positional(PosAT::Move, Position::new((5, 5))))?;

        assert!(!state
            .get_player(id)
            .unwrap()
            .can_use_skill(Skill::SureHands));

        match state.ball {
            BallState::Carried(id_carrier) if id_carrier == id => (),
            _ => panic!("wrong ball carried"),
        }

        Ok(())
    }
    #[test]
    fn gfi_reroll() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

        let id = state.get_player_id_at(start_pos).unwrap();

        state.step(Action::Positional(PosAT::StartMove, Position::new((1, 1))))?;

        state.d6_fixes.push_back(D6::One); //fail first (2+)
        state.step(Action::Positional(PosAT::Move, Position::new((9, 1))))?;

        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge));

        state.d6_fixes.push_back(D6::Two); //succeed with team reroll
        state.d6_fixes.push_back(D6::Two); //succeed next gfi roll
        state.step(Action::Simple(SimpleAT::UseReroll))?;

        let state = state;
        let player = state.get_player(id).unwrap();
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Move, Position::new((9, 2)))));
        assert_eq!(state.get_player_id_at_coord(9, 1).unwrap(), id);
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll());
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2);
        assert_eq!(state.get_legal_positions(PosAT::Move).len(), 0);
        assert_eq!(player.total_movement_left(), 0);
        assert_eq!(player.gfis_left(), 0);
        assert_eq!(player.moves_left(), 0);

        Ok(())
    }

    #[test]
    fn dodge_reroll() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((2, 1)))
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();

        state.get_mut_player(id)?.stats.skills.insert(Skill::Dodge);
        assert!(state.get_player(id).unwrap().has_skill(Skill::Dodge));

        state.step(Action::Positional(PosAT::StartMove, Position::new((1, 1))))?;

        state.d6_fixes.push_back(D6::Three); //fail first (4+)
        state.d6_fixes.push_back(D6::Four); //Succeed on skill reroll
        state.d6_fixes.push_back(D6::Two); //fail second dodge  (3+)

        state.step(Action::Positional(PosAT::Move, Position::new((3, 3))))?;
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge));

        state.d6_fixes.push_back(D6::Three); //succeed with team reroll
        state.step(Action::Simple(SimpleAT::UseReroll))?;

        assert_eq!(state.get_player_id_at_coord(3, 3).unwrap(), id);
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll());
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2);
        assert_eq!(state.get_mut_player(id).unwrap().total_movement_left(), 6);
        assert_eq!(state.get_mut_player(id).unwrap().gfis_left(), 2);
        assert_eq!(state.get_mut_player(id).unwrap().moves_left(), 4);
        state.step(Action::Simple(SimpleAT::EndPlayerTurn))?;

        Ok(())
    }
    #[test]
    fn player_unique_id_and_correct_positions() {
        let state = standard_state();

        let mut ids = HashSet::new();
        for x in 0..WIDTH_ {
            for y in 0..HEIGHT_ {
                let pos = Position::new((x, y));
                if let Some(player) = state.get_player_at(pos) {
                    assert_eq!(player.position, pos);
                    assert!(ids.insert(player.id));
                }
            }
        }
        assert_eq!(0, ids.into_iter().filter(|id| *id >= 22).count());
    }

    #[test]
    fn adjescent() {
        let state = standard_state();
        let num_adj = state.get_adj_players(Position::new((2, 2))).count();
        assert_eq!(num_adj, 3);
    }

    #[test]
    fn mutate_player() {
        let mut state = standard_state();

        assert!(!(state.get_player(0).unwrap().used));
        state.get_mut_player(0).unwrap().used = true;
        assert!(state.get_player(0).unwrap().used);
    }

    #[test]
    fn move_player() -> Result<()> {
        let mut state = standard_state();
        let id = 1;
        let old_pos = Position::new((2, 2));
        let new_pos = Position::new((10, 10));

        assert_eq!(state.get_player_id_at(old_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, old_pos);
        assert!(state.get_player_id_at(new_pos).is_none());

        state.move_player(id, new_pos)?;

        assert!(state.get_player_id_at(old_pos).is_none());
        assert_eq!(state.get_player_id_at(new_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, new_pos);
        Ok(())
    }

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
        let player_stats = PlayerStats::new(TeamType::Home);
        let position = Position::new((10, 10));

        assert!(state.get_player_id_at(position).is_none());

        let id = state.field_player(player_stats, position).unwrap();

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
        state.d6_fixes.extend(&[D6::Six, D6::Six, D6::Six]);

        assert!(state.get_player_at(starting_pos).is_some());
        assert!(state.get_player_at(move_target).is_none());

        state.step(Action::Positional(PosAT::StartMove, starting_pos))?;
        state.step(Action::Positional(PosAT::Move, move_target))?;

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step(Action::Simple(SimpleAT::EndPlayerTurn))?;

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

        state.step(Action::Positional(PosAT::StartMove, starting_pos))?;
        state.step(Action::Positional(PosAT::Move, move_target))?;

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step(Action::Simple(SimpleAT::EndPlayerTurn))?;

        assert!(state.get_player_at(move_target).unwrap().used);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartMove, move_target)));

        Ok(())
    }

    #[test]
    fn pathing() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let id = state.get_player_id_at(starting_pos).unwrap();
        state.step(Action::Positional(PosAT::StartMove, starting_pos))?;
        let paths = PathFinder::player_paths(&state, id)?;

        let mut errors = Vec::new();

        for x in 1..8 {
            for y in 1..8 {
                let x_usize = usize::try_from(x).unwrap();
                let y_usize = usize::try_from(y).unwrap();
                match (state.get_player_id_at_coord(x, y), &paths[x_usize][y_usize]) {
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
                match (pos_to_prob.get(&(x, y)).unwrap(), &paths[x][y]) {
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

        let expected_steps = vec![
            (
                Position::new((4, 6)),
                vec![
                    Roll::GFI(D6Target::TwoPlus),
                    Roll::Pickup(D6Target::ThreePlus),
                ],
            ),
            (
                Position::new((4, 5)),
                vec![Roll::Dodge(D6Target::ThreePlus)],
            ),
            (Position::new((4, 4)), vec![Roll::Dodge(D6Target::FourPlus)]),
            (Position::new((4, 3)), vec![Roll::Dodge(D6Target::FourPlus)]),
            (Position::new((3, 2)), vec![]),
            (
                Position::new((3, 1)),
                vec![Roll::Dodge(D6Target::ThreePlus)],
            ),
            (Position::new((2, 1)), vec![Roll::Dodge(D6Target::FourPlus)]),
        ];
        let expected_prob = 0.03086;
        let path = paths[4][6].clone().unwrap();

        for (i, (expected, actual)) in zip(expected_steps, path.steps).enumerate() {
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
        let fixes = vec![D6::One, D6::Three, D6::Five, D6::Two, D6::Four, D6::Six];
        state.d6_fixes.extend(fixes.iter());

        let rolls: Vec<D6> = repeat_with(|| state.get_d6_roll())
            .take(fixes.len())
            .collect();
        assert_eq!(fixes, rolls);
    }

    #[test]
    fn movement() -> Result<()> {
        let mut state = standard_state();
        state.step(Action::Positional(PosAT::StartMove, Position::new((3, 1))))?;
        Ok(())
    }
}

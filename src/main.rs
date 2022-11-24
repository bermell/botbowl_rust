use crate::core::gamestate::GameStateBuilder;
use crate::core::model::*; 
pub mod core;

fn main() {
    let g = GameStateBuilder::new(&[(1, 2), (2, 2)], 
                                             &[(5,2), (5, 5)])
                                                    .add_ball((3, 2))
                                                    .build(); 
    let p = Position {x: 1, y:2}; 
    println!("I'm at {:?}!", p); 

    println!("testing gamestate {:?}", g.get_player_at(p)); 
}


#[cfg(test)]
mod tests {

    use std::{collections::{HashSet, HashMap}, iter::{zip, repeat_with}};
    use crate::core::{model::{Position, WIDTH_, HEIGHT_, PlayerStats, TeamType, DogoutPlace, Action}, table::{PosAT}, gamestate::{GameState, GameStateBuilder}, pathing::{PathFinder, Roll}}; 
    use ansi_term::Colour::Red;
    use crate::core::table::*; 
    use crate::core::model::*; 

    fn standard_state() -> GameState {
        GameStateBuilder::new(&[(1, 2), (2, 2), (3, 1)], 
                              &[(5,2), (5, 5), (2, 3)])
                                    .add_ball((3, 2))
                                    .build() 
    }

    #[test]
    fn gfi_reroll() -> Result<()> {
        let mut state = GameStateBuilder::new(&[(1, 1)], &[]).build(); 
        let id = state.get_player_id_at_coord(1, 1).unwrap(); 

        state.step(Action::Positional(PosAT::StartMove, Position { x: 1, y: 1 }))?; 
         
        state.d6_fixes.push_back(D6::One); //fail first (2+) 
        state.step(Action::Positional(PosAT::Move, Position { x: 9, y: 1 }))?; 
        
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll))); 
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge)); 
        
        state.d6_fixes.push_back(D6::Two); //succeed with team reroll  
        state.d6_fixes.push_back(D6::Two); //succeed next gfi roll
        state.step(Action::Simple(SimpleAT::UseReroll))?; 

        let state = state; 
        let player = state.get_player(id).unwrap(); 
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Move, Position { x: 9, y: 2 })));
        assert_eq!(state.get_player_id_at_coord(9, 1).unwrap(), id);  
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll()); 
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2); 
        assert_eq!(state.get_legal_positions(PosAT::Move).len(), 0); 
        assert_eq!(player.total_movement_left(), 0);
        assert_eq!(player.gfis_left() , 0);
        assert_eq!(player.moves_left(), 0);


        Ok(())
    }

    #[test] 
    fn dodge_reroll() -> Result<()> {
        let mut state = GameStateBuilder::new(&[(1, 1)], &[(2, 1)]).build(); 
        let id = state.get_player_id_at_coord(1, 1).unwrap(); 
        state.get_mut_player(id)?.stats.skills.insert(Skill::Dodge); 
        assert!(state.get_player(id).unwrap().has_skill(Skill::Dodge)); 

        state.step(Action::Positional(PosAT::StartMove, Position { x: 1, y: 1 }))?; 
         
        state.d6_fixes.push_back(D6::Three); //fail first (4+) 
        state.d6_fixes.push_back(D6::Four); //Succeed on skill reroll
        state.d6_fixes.push_back(D6::Two); //fail second dodge  (3+)
        
        state.step(Action::Positional(PosAT::Move, Position { x: 3, y: 3 }))?; 
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll))); 
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge)); 
        
        state.d6_fixes.push_back(D6::Three); //succeed with team reroll  
        state.step(Action::Simple(SimpleAT::UseReroll))?; 

        assert_eq!(state.get_player_id_at_coord(3, 3).unwrap(), id);  
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll()); 
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2); 
        assert_eq!(state.get_mut_player(id).unwrap().total_movement_left(), 6);
        assert_eq!(state.get_mut_player(id).unwrap().gfis_left() , 2);
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
                let pos = Position{x, y}; 
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
        let num_adj = state.get_adj_players(Position { x: 2, y: 2 }).count(); 
        assert_eq!(num_adj, 3); 
    }

    #[test]
    fn mutate_player(){
        let mut state = standard_state(); 

        assert!(!(state.get_player(0).unwrap().used)); 
        state.get_mut_player(0).unwrap().used = true; 
        assert!(state.get_player(0).unwrap().used); 
    }

    #[test]
    fn move_player() -> Result<()> {
        let mut state = standard_state(); 
        let id = 1;  
        let old_pos = Position{x: 2, y: 2}; 
        let new_pos = Position{x: 10, y: 10}; 

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
    fn draw_board(){
        let _state = standard_state(); 
        
        println!("This is in red: {}", Red.strikethrough().paint("a red string"));
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
        let position = Position{x: 10, y: 10}; 
        
        assert!(state.get_player_id_at(position).is_none()); 
        
        let id = state.field_player(player_stats, position).unwrap();  
       
        assert_eq!(state.get_player_id_at(position), Some(id)); 
        assert_eq!(state.get_player(id).unwrap().position, position); 
        
        state.unfield_player(id, DogoutPlace::Reserves)?; 
        
        assert!(state.get_player_id_at(position).is_none()); 
        Ok(())
    }
    
    #[test]
    fn long_move_action() -> Result<()> {
        let mut state = standard_state(); 
        let starting_pos = Position{x: 3, y: 1}; 
        let move_target = Position{x: 2, y: 5};  
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
        let starting_pos = Position{x: 3, y: 1}; 
        let move_target = Position{x: 4, y: 1};  

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
        let starting_pos = Position{x: 3, y: 1}; 
        let id = state.get_player_id_at(starting_pos).unwrap(); 
        state.step(Action::Positional(PosAT::StartMove, starting_pos))?; 
        let mut pf = PathFinder::new(&state); 
        let paths = pf.player_paths(id)?; 
      
        let mut errors = Vec::new(); 

        for x in 1..8 {
            for y in 1..8 {
                let x_usize = usize::try_from(x).unwrap(); 
                let y_usize = usize::try_from(y).unwrap(); 
                match (state.get_player_id_at_coord(x, y), &paths[x_usize][y_usize]) {
                    (Some(_), None) => (), 
                    (None, Some(_)) => (), 
                    (Some(_), Some(_)) => errors.push(format!("Found path already occupied square ({},{})", x, y)), 
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
        let state = GameStateBuilder::new(&[(3, 2)], &[(1, 3), (3, 3), (4, 2)]).build(); 
        let starting_pos = Position{x: 3, y: 2}; 
        let id = state.get_player_id_at(starting_pos).unwrap(); 
        
        let mut pf = PathFinder::new(&state); 
        let paths = pf.player_paths(id)?; 
        
        let mut pos_to_prob: HashMap<(usize, usize), Option<f32>> = HashMap::new();  
        pos_to_prob.insert((1, 1), Some(2.0/3.0)); 
        pos_to_prob.insert((1, 2), Some(2.0/3.0)); 
        pos_to_prob.insert((1, 3), None);  
        pos_to_prob.insert((1, 4), Some(2.0/9.0)); 
        pos_to_prob.insert((2, 1), Some(2.0/3.0)); 
        pos_to_prob.insert((2, 2), Some(2.0/3.0)); 
        pos_to_prob.insert((2, 3), Some(1.0/3.0)); 
        pos_to_prob.insert((2, 4), Some(2.0/9.0)); 
        pos_to_prob.insert((3, 1), Some(2.0/3.0)); 
        pos_to_prob.insert((3, 2), None);  
        pos_to_prob.insert((3, 3), None);  
        pos_to_prob.insert((3, 4), Some(2.0/9.0)); 
        pos_to_prob.insert((4, 1), Some(1.0/2.0)); 
        pos_to_prob.insert((4, 2), None);  
        pos_to_prob.insert((4, 3), Some(1.0/3.0)); 
        pos_to_prob.insert((4, 4), Some(2.0/9.0)); 
        
        let mut errors = Vec::new(); 

        #[allow(clippy::needless_range_loop)]
        for x in 1..5 {
            for y in 1..5 {
                match (pos_to_prob.get(&(x, y)).unwrap(), &paths[x][y]) {
                    (Some(correct_prob), Some(path)) if (*correct_prob - path.prob).abs() > 0.001 => errors.push(format!("Path to ({}, {}) has wrong prob. \nExpected prob: {}\nGot prob: {}\n", x, y, *correct_prob, path.prob)), 
                    (Some(correct_prob), Some(path)) if (*correct_prob - path.prob).abs() <= 0.001 => (), 
                    (None, None) => (), 
                    (Some(_), None) => errors.push(format!("No path to ({}, {})", x, y)), 
                    (None, Some(path)) => errors.push(format!("There shouldn't be a path to ({}, {}). Found: {:?}", x, y, path)), 
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
        let state = GameStateBuilder::new(
            &[(1, 1)], 
            &[(1, 2), (2, 3),(2, 4),  (5, 3), (6, 4)])
            .add_ball((4, 6))
            .build(); 
        let starting_pos = Position{x: 1, y: 1}; 
        let id = state.get_player_id_at(starting_pos).unwrap(); 
        let mut pf = PathFinder::new(&state); 
        let paths = pf.player_paths(id)?; 

        let expected_steps = vec![  (Position{x: 4, y: 6}, vec![Roll::GFI(2), Roll::Pickup(3)]), 
                                    (Position{x: 4, y: 5}, vec![Roll::Dodge(3)]), 
                                    (Position{x: 4, y: 4}, vec![Roll::Dodge(4)]), 
                                    (Position{x: 4, y: 3}, vec![Roll::Dodge(4)]), 
                                    (Position{x: 3, y: 2}, vec![]), 
                                    (Position{x: 3, y: 1}, vec![Roll::Dodge(3)]), 
                                    (Position{x: 2, y: 1}, vec![Roll::Dodge(4)]), ]; 
        let expected_prob = 0.03086; 
        let path = paths[4][6].clone().unwrap(); 

        for (i, (expected, actual)) in zip(expected_steps, path.steps).enumerate(){
            if expected != actual {
                panic!("Step {}: {:?} != {:?}",i, expected, actual ); 
            }
        }

        assert!((expected_prob-path.prob).abs() < 0.0001); 

        Ok(())
    }

    
    #[test]
    fn rng_seed_in_gamestate() -> Result<()> {
        let mut state = standard_state(); 
        state.rng_enabled = true; 
        let seed = 5; 
        state.set_seed(seed); 
        
        fn get_random_rolls(state: &mut GameState) -> Vec<D6> {
            repeat_with(|| state.get_roll()).take(200).collect()
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

        let rolls: Vec<D6> = repeat_with(|| state.get_roll()).take(fixes.len()).collect(); 
        assert_eq!(fixes, rolls); 
    }

    #[test]
    fn movement() -> Result<()>{
        let mut state = standard_state(); 
        state.step(Action::Positional(PosAT::StartMove, Position { x: 3, y: 1 }))?; 
        Ok(())
    }
}
 
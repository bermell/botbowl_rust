use crate::core::gamestate::GameStateBuilder;
use crate::core::model::*; 
use crate::core::table::*; 
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

    use std::collections::{HashSet, HashMap};
    use crate::core::{model::{Position, WIDTH_, HEIGHT_, PlayerStats, TeamType, DogoutPlace, ActionChoice, Action}, table::{AnyAT, PosAT}, gamestate::{GameState, GameStateBuilder}}; 
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
    fn move_player(){
        let mut state = standard_state(); 
        let id = 1;  
        let old_pos = Position{x: 2, y: 2}; 
        let new_pos = Position{x: 10, y: 10}; 

        assert_eq!(state.get_player_id_at(old_pos), Some(id)); 
        assert_eq!(state.get_player(id).unwrap().position, old_pos); 
        assert!(state.get_player_id_at(new_pos).is_none()); 

        state.move_player(id, new_pos); 

        assert!(state.get_player_id_at(old_pos).is_none()); 
        assert_eq!(state.get_player_id_at(new_pos), Some(id)); 
        assert_eq!(state.get_player(id).unwrap().position, new_pos); 
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
    fn field_a_player() { 
        let mut state = standard_state(); 
        let player_stats = PlayerStats::new(TeamType::Home); 
        let position = Position{x: 10, y: 10}; 
        
        assert!(state.get_player_id_at(position).is_none()); 
        
        let id = state.field_player(player_stats, position).unwrap();  
       
        assert_eq!(state.get_player_id_at(position), Some(id)); 
        assert_eq!(state.get_player(id).unwrap().position, position); 
        
        state.unfield_player(id, DogoutPlace::Reserves); 
        
        assert!(state.get_player_id_at(position).is_none()); 
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

        match state.get_available_actions().get(&AnyAT::from(PosAT::StartMove)) {
            Some(ActionChoice::Positional(positions)) => 
                assert!(!positions.iter().any(|p|*p==move_target)), 
            _ => panic!(), 
        }

        Ok(())
    }
}

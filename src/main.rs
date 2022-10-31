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

    use std::collections::HashSet;
    use crate::core::model::{Position, GameStateBuilder, GameState, WIDTH_, HEIGHT_, PlayerStats, TeamType, DogoutPlace}; 
    use ansi_term::Colour::Red;

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
                match state.get_player_at(pos) {
                    Some(player) => {assert_eq!(player.position, pos); 
                                     assert!(ids.insert(player.id)); 
                                    }  
                    _ => (), 
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
        assert_eq!(state.get_mut_player_unsafe(0).used, false); 
        state.get_mut_player_unsafe(0).used = true; 
        assert_eq!(state.get_mut_player_unsafe(0).used, true); 
    }

    #[test]
    fn move_player(){
        let mut state = standard_state(); 
        let id = 1;  
        let old_pos = Position{x: 2, y: 2}; 
        let new_pos = Position{x: 10, y: 10}; 

        assert_eq!(state.get_player_id_at(old_pos), Some(id)); 
        assert_eq!(state.get_player_unsafe(id).position, old_pos); 
        assert!(state.get_player_id_at(new_pos).is_none()); 

        state.move_player(id, new_pos); 

        assert!(state.get_player_id_at(old_pos).is_none()); 
        assert_eq!(state.get_player_id_at(new_pos), Some(id)); 
        assert_eq!(state.get_player_unsafe(id).position, new_pos); 
    }

    #[test]
    fn draw_board(){
        let state = standard_state(); 
        
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
        
        let id = state.field_player(player_stats, position); 
       
        assert_eq!(state.get_player_id_at(position), Some(id)); 
        assert_eq!(state.get_player_unsafe(id).position, position); 
        
        state.unfield_player(id, DogoutPlace::Reserves); 
        
        assert!(state.get_player_id_at(position).is_none()); 
        

    }
}

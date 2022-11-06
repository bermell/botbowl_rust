use crate::core::model; 
use model::*; 

use crate::core::table::*; 

use std::collections::HashMap;
use crate::core::table;

use super::{table::AnyAT, gamestate::GameState}; 

pub struct Turn{
    pub team: TeamType, 
}
impl Procedure for Turn {


    fn available_actions(&self, g: &mut GameState) -> HashMap<table::AnyAT, ActionChoice> {
        let positions = g.get_players_on_pitch_in_team(self.team)
            .filter(|p| !p.used)
            .map(|p| p.position);
        let mut aa = HashMap::new(); 
        aa.insert(AnyAT::from(PosAT::StartMove), ActionChoice::Positional(positions.collect()));
        aa.insert(AnyAT::from(SimpleAT::EndTurn), ActionChoice::Simple);
        aa
    }

    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::StartMove, position)) => {
                let move_action = MoveAction{player_id: g.get_player_id_at(position).unwrap()}; 
                g.push_proc(Box::new(move_action)); 
                false
            }
            Some(Action::Simple(SimpleAT::EndTurn)) => true, 
            _ => panic!("Action not allowed: {:?}", action), 
        }
    }
}

pub struct MoveAction{
    pub player_id: PlayerID, 
}
impl Procedure for MoveAction {


    fn available_actions(&self, g: &mut GameState) -> HashMap<table::AnyAT, ActionChoice> {
        let mut aa = HashMap::new(); 
        let pos = g.get_player(self.player_id).unwrap().position; 
        let free_adj_pos = g.get_adj_positions(pos).filter(|p| g.get_player_at(*p).is_none());
        aa.insert(AnyAT::from(PosAT::Move), ActionChoice::Positional(free_adj_pos.collect()));
        aa.insert(AnyAT::from(SimpleAT::EndPlayerTurn), ActionChoice::Simple); 
        aa
    }

    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::Move, position)) => {
                if g.get_player_at(position).is_some() {
                    panic!("Very wrong!");
                }
                if g.move_player(self.player_id, position).is_err(){
                    panic!("very wrong again!")
                } 
                false
            }
            Some(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                g.get_mut_player(self.player_id).unwrap().used = true; 
                true 
            }

            _ => panic!("very wrong!")
        }
    }
}

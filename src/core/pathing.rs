use std::rc::Rc;

use crate::core::model; 
use model::*;

use super::gamestate::GameState; 

type OptRcNode = Option<Rc<Node>>; 

pub enum Roll{ //Make more clever! 
    Dodge(u8), 
    GFI(u8),  
}

#[allow(dead_code)]
pub struct Node { 
    parent: OptRcNode, 
    position: Position, 
    moves_left: u8, 
    gfis_left: u8, 
    // foul_roll, handoff_roll, block_dice
    euclidiean_distance: f32, 
    prob: f32, 
    rolls: Vec<Roll>, 
}

#[allow(dead_code)]
pub struct Path {
    steps: Vec<(Position, Vec<Roll>)>, 
    prob: f32, 
}

#[allow(dead_code)]
pub struct PathFinder<'a> {
    pub game_state: &'a GameState, 
    nodes: FullPitch<OptRcNode>, 
    max_moves: u8, 
    max_gfis: u8, 
    node_queue: Vec<OptRcNode>, 
} 

impl<'a> PathFinder <'a>{
    pub fn new(game_state: &'a mut GameState) -> PathFinder<'a> {
        PathFinder { game_state, 
                     nodes: Default::default(),
                     max_moves: 0, 
                     max_gfis: 0, 
                     node_queue: Vec::new(), 
                    }
    }

    pub fn player_paths(&mut self, id: PlayerID) -> Result<FullPitch<Option<Path>>> {
        self.max_moves = self.game_state.get_player(id).unwrap().stats.ma; 
        self.max_gfis = 2; 
        
        
        
        
        Ok(Default::default())
    }
}

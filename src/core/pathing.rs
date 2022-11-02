use crate::core::model; 
use model::*; 

pub enum Roll{ //Make more clever! 
    Dodge(u8), 
    GFI(u8),  
}

pub struct Node<'a> { 
    parent: Option<&'a Node<'a>>, 
    position: Position, 
    moves_left: u8, 
    gfis_left: u8, 
    // foul_roll, handoff_roll, block_dice
    euclidiean_distance: f32, 
    prob: f32, 
    rolls: Vec<Roll>, 
}

pub struct Path {
    steps: Vec<(Position, Vec<Roll>)>, 
    prob: f32, 
}

pub struct PathFinder<'a> {
    game_state: &'a GameState, 

} 

impl<'a> PathFinder <'a>{
    pub fn active_player_paths() -> FullPitch<Option<Path>> {
        Default::default()
    }
}

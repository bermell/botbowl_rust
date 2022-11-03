use crate::core::model; 
use model::*; 

use std::collections::HashMap;
use crate::core::table; 

pub struct Turn; 
impl Procedure for Turn {
    fn start(&self, g: &GameState) {}

    fn end(&self, g: &mut GameState) {}

    fn available_actions(&self, g: &mut GameState) -> HashMap<table::AnyAT, ActionChoice> {HashMap::new()}

    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        true
    }
}

pub struct MoveAction; 
impl Procedure for MoveAction {
    fn start(&self, g: &GameState) {}

    fn end(&self, g: &mut GameState) {}

    fn available_actions(&self, g: &mut GameState) -> HashMap<table::AnyAT, ActionChoice> {HashMap::new()}

    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        true
    }
}

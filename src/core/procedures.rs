use crate::core::model; 
use model::*; 

struct Turn; 
impl Procedure for Turn {
    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        todo!()
    }
    fn available_actions(&self, g: &mut GameState) -> Vec<ActionChoice> {
        Vec::new()
    }
}

struct MoveAction; 
impl Procedure for MoveAction {
    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool {
        todo!()
    }
    fn available_actions(&self, g: &mut GameState) -> Vec<ActionChoice> {
        Vec::new()
    }
}

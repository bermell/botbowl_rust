use crate::core::model; 
use model::*; 

use crate::core::table::*; 

use std::{collections::HashMap, hash::Hash};
use crate::core::table;

use super::{table::AnyAT, gamestate::{GameState, self}, pathing::{PathFinder, Path, Roll}}; 

pub struct Turn{
    pub team: TeamType, 
}
impl Procedure for Turn {
    fn available_actions(&mut self, game_state: &GameState) -> HashMap<table::AnyAT, ActionChoice> {
        let positions = game_state.get_players_on_pitch_in_team(self.team)
            .filter(|p| !p.used)
            .map(|p| p.position);
        let mut aa = HashMap::new(); 
        aa.insert(AnyAT::from(PosAT::StartMove), ActionChoice::Positional(positions.collect()));
        aa.insert(AnyAT::from(SimpleAT::EndTurn), ActionChoice::Simple);
        aa
    }

    fn step(&mut self, g: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::StartMove, position)) => {
                let move_action = MoveAction::new(g.get_player_id_at(position).unwrap()); 
                g.push_proc(Box::new(move_action)); 
                false
            }
            Some(Action::Simple(SimpleAT::EndTurn)) => true, 
            _ => panic!("Action not allowed: {:?}", action), 
        }
    }
}

fn proc_from_roll(roll: Roll, move_action: &MoveAction) -> Box<dyn Procedure> {
    match roll {
        Roll::Dodge(target) => Box::new(DodgeProc::new(move_action.player_id)), 
        Roll::GFI(target) => todo!(), 
        Roll::Pickup(target) => todo!(), 
    }
}

pub struct MoveAction{
    player_id: PlayerID, 
    paths: FullPitch<Option<Path>>,
    active_path: Option<Path>, 
    rolls: Option<Vec<Roll>>, 
}
impl MoveAction{
    pub fn new(id: PlayerID) -> MoveAction {
        MoveAction { player_id: id, paths: Default::default(), active_path: None, rolls: None }
    }


    fn contine_active_path(&mut self, game_state: &mut GameState) -> bool {
        let roll_len_before = self.rolls.as_ref().map_or(0, |rolls| rolls.len());//debugging 

        //are the rolls left to handle?  
        if let Some(next_roll) = self.rolls.as_mut().map(|rolls| rolls.pop()).flatten() {
            let new_proc = proc_from_roll(next_roll, &self); 
            game_state.push_proc(new_proc); 
            
            let roll_len_after = self.rolls.as_ref().map_or(0, |rolls| rolls.len()); //debugging 
            assert_eq!(roll_len_before-1, roll_len_after); //debugging 
            
            return true; 
        }
        
        let path = self.active_path.as_mut().unwrap(); 
       
        // check if any rolls left to handle, if not then just move to end of path
        if path.steps.iter().any(|(_, rolls)| !rolls.is_empty()){
            game_state.move_player(self.player_id, path.target).unwrap(); 
            return false; 
        }

        while let Some((position, mut rolls)) = path.steps.pop(){
            
            game_state.move_player(self.player_id, position).unwrap(); 
            
            if let Some(next_roll) = rolls.pop() {
                let new_proc = proc_from_roll(next_roll, self); 
                game_state.push_proc(new_proc); 
                if !rolls.is_empty(){
                    self.rolls = Some(rolls); 
                }
                return true; 
            }
        }
        
        true
    } 

}
impl Procedure for MoveAction {
    fn available_actions(&mut self, g: &GameState) -> HashMap<table::AnyAT, ActionChoice> {
        let mut aa = HashMap::new(); 
        if self.active_path.is_some(){
            return aa; 
        }
        let player = g.get_player(self.player_id).unwrap(); 
        if player.used {
            return aa; 
        }
        if player.moves_left() > 0 {
            self.paths = PathFinder::new(g).player_paths(self.player_id).unwrap(); 
            let move_positions = gimmi_iter(&self.paths)
                    .flatten()
                    .map(|path| path.target).collect();
            
            aa.insert(AnyAT::from(PosAT::Move), ActionChoice::Positional(move_positions));
        }
        aa.insert(AnyAT::from(SimpleAT::EndPlayerTurn), ActionChoice::Simple); 
        aa
    }

    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::Move, position)) => {
                if game_state.get_player_at(position).is_some() {
                    panic!("Very wrong!");
                }
                let (x, y) = position.to_usize().unwrap(); 
                self.active_path = self.paths[x][y].clone(); 
                self.paths = Default::default();  
                self.contine_active_path(game_state) 
            }
            Some(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                game_state.get_mut_player(self.player_id).unwrap().used = true; 
                true 
            } 
            None => {
                if game_state.get_player(self.player_id).unwrap().used {
                    return true; 
                }
                
                self.contine_active_path(game_state) 
            }

            _ => panic!("very wrong!")
        }
    }
}

struct DodgeProc{
    target: D6,
    id: PlayerID, 
}
impl DodgeProc {
    fn new(id: PlayerID) -> MovementProc<DodgeProc> {
        MovementProc::new(DodgeProc { target: D6::Five, id }, id)
    }
}
impl SimpleProc for DodgeProc{
    fn d6_target(&self) -> D6 {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Dodge)
    }
}


trait SimpleProc {
    fn d6_target(&self) -> D6; //called immidiately before 
    fn reroll_skill(&self) -> Option<Skill>; 
    fn apply_success(&self, game_state: &mut GameState) {}
    fn apply_failure(&self, game_state: &mut GameState) {}
}

#[derive(Debug, PartialEq, Eq)]
enum RollProcState {
    Init, 
    WaitingForTeamReroll, 
    RerollUsed, 
    //WaitingForSkillReroll, 
}
struct MovementProc<T: SimpleProc> {
    proc: T, 
    state: RollProcState, 
    id: PlayerID
}
impl<T: SimpleProc> MovementProc<T> {
    pub fn new(proc: T, id: PlayerID) -> Self {
        MovementProc { proc, state: RollProcState::Init, id}
    }
}

impl<T> Procedure for MovementProc<T> 
where
    T: SimpleProc  
{
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        // if action is DON*T REROLL, apply failure, return true
        match action {
            Some(Action::Simple(SimpleAT::DontUseReroll)) => {
                self.proc.apply_failure(game_state);
                return true;
            } 
            Some(Action::Simple(SimpleAT::UseReroll)) => {
                game_state.get_active_team_mut().use_reroll(); 
                self.state = RollProcState::RerollUsed; 
            } 
            _ => (), 
        }
        
        loop{
            let roll = game_state.get_roll(); 
            if roll >= self.proc.d6_target() {
                self.proc.apply_success(game_state); 
                return true; 
            } 
            if self.state == RollProcState::RerollUsed {
                break; 
            }
            match self.proc.reroll_skill() {
                Some(skill) if game_state.get_player(self.id).unwrap().can_use_skill(skill) => {
                    game_state.get_mut_player(self.id).unwrap().use_skill(skill); 
                    self.state = RollProcState::RerollUsed; 
                    continue;
                }
                _ => (), 
            }
            if game_state.get_active_team().can_use_reroll(){
                self.state = RollProcState::WaitingForTeamReroll; 
                return false; 
            }
        }
        self.proc.apply_failure(game_state); 
        true
    }
    fn available_actions(&mut self, game_state: &GameState) -> HashMap<AnyAT, ActionChoice> {
        let mut aa = HashMap::new(); 
        
        match self.state {
            RollProcState::Init => (), 
            RollProcState::WaitingForTeamReroll => {
                aa.insert(AnyAT::from(SimpleAT::UseReroll), ActionChoice::Simple); 
                aa.insert(AnyAT::from(SimpleAT::DontUseReroll), ActionChoice::Simple); 
            }, 
            _ => panic!("Illegal state!"), 
        }
        aa
    }
}
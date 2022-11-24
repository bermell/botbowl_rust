use crate::core::model; 
use model::*; 

use crate::core::table::*; 

use super::{gamestate::{GameState}, pathing::{PathFinder, Path, Roll}}; 

pub struct Turn{
    pub team: TeamType, 
}
impl Procedure for Turn {
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let positions = game_state.get_players_on_pitch_in_team(self.team)
            .filter(|p| !p.used)
            .map(|p| p.position);
        let mut aa = AvailableActions::new(self.team); 
        aa.insert_positional(PosAT::StartMove, positions.collect());
        aa.insert_simple(SimpleAT::EndTurn); 
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
        Roll::Dodge(target) => Box::new(DodgeProc::new(move_action.player_id, D6::try_from(target).unwrap())), 
        Roll::GFI(_) => todo!(), 
        Roll::Pickup(_) => todo!(), 
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
    fn consolidate_active_path(&mut self) {
        if let Some(rolls) = &self.rolls {
            if !rolls.is_empty(){
                return; 
            }
        }
        if let Some(path) = &self.active_path {
            if !path.steps.is_empty() {
                return; 
            }  
        }
        self.rolls = None; 
        self.active_path = None; 
    }

    fn continue_active_path(&mut self, game_state: &mut GameState) {
        let debug_roll_len_before = self.rolls.as_ref().map_or(0, |rolls| rolls.len());

        //are the rolls left to handle?  
        if let Some(next_roll) = self.rolls.as_mut().and_then(|rolls| rolls.pop()) {
            let new_proc = proc_from_roll(next_roll, self); 
            game_state.push_proc(new_proc); 
            
            let debug_roll_len_after = self.rolls.as_ref().map_or(0, |rolls| rolls.len()); 
            assert_eq!(debug_roll_len_before-1, debug_roll_len_after); 
            
            return; 
        }
        
        let path = self.active_path.as_mut().unwrap(); 
       
        // check if any rolls left to handle, if not then just move to end of path
        if path.steps.iter().all(|(_, rolls)| rolls.is_empty()){
            
            //check if already there
            if let Some(id) = game_state.get_player_id_at(path.target) {
                debug_assert_eq!(id, self.player_id); 
            } else {
                game_state.move_player(self.player_id, path.target).unwrap(); 
            }
            path.steps.clear(); 
            return;
        }
        while let Some((position, mut rolls)) = path.steps.pop(){
            
            game_state.move_player(self.player_id, position).unwrap(); 
            
            if let Some(next_roll) = rolls.pop() {
                let new_proc = proc_from_roll(next_roll, self); 
                game_state.push_proc(new_proc); 
                if !rolls.is_empty(){
                    self.rolls = Some(rolls); 
                }
                return; 
            }
        }
        
    } 

}
impl Procedure for MoveAction {
    fn available_actions(&mut self, g: &GameState) -> AvailableActions {
        let mut aa = AvailableActions::new(g.get_player(self.player_id).unwrap().stats.team); 
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
            
            aa.insert_positional(PosAT::Move, move_positions);
        }
        aa.insert_simple(SimpleAT::EndPlayerTurn); 
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
                self.continue_active_path(game_state); 
                self.consolidate_active_path(); 
                false 
            }
            Some(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                game_state.get_mut_player(self.player_id).unwrap().used = true; 
                true 
            } 
            None => {
                if game_state.get_player(self.player_id).unwrap().used {
                    return true; 
                }
                
                self.continue_active_path(game_state); 
                self.consolidate_active_path();
                false
            }

            _ => panic!("very wrong!")
        }
    }
}

#[allow(dead_code)]
struct DodgeProc{
    target: D6,
    id: PlayerID, 
}
impl DodgeProc {
    fn new(id: PlayerID, target: D6) -> MovementProc<DodgeProc> {
        MovementProc::new(DodgeProc { target, id }, id)
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

#[allow(unused_variables)]
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
                game_state.get_active_team_mut().unwrap().use_reroll(); 
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
            
            if game_state.get_team_from_player(self.id).unwrap().can_use_reroll(){
                self.state = RollProcState::WaitingForTeamReroll; 
                return false; 
            }
        }
        self.proc.apply_failure(game_state); 
        true
    }
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        match self.state {
            RollProcState::Init => AvailableActions::new_empty(), 
            RollProcState::WaitingForTeamReroll => {
                let mut aa = AvailableActions::new(game_state.get_player(self.id).unwrap().stats.team);
                aa.insert_simple(SimpleAT::UseReroll); 
                aa.insert_simple(SimpleAT::DontUseReroll); 
                aa
            }, 
            _ => panic!("Illegal state!"), 
        }
    }
}
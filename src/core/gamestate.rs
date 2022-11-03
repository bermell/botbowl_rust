use core::panic;
use std::error;

use crate::core::model; 

use model::*; 

#[derive(Debug)]
pub struct Tomato{}

const DIRECTIONS: [(Coord, Coord); 8] = [(1, 1), (0, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (0, -1), (-1, -1)];  


impl GameState {
    pub fn get_player_id_at(&self, p: Position) -> Option<PlayerID> {
        self.get_player_id_at_coord(p.x, p.y)
    } 
    pub fn get_player_at(&self, p: Position) -> Option<&FieldedPlayer> {
        self.get_player_at_coord(p.x, p.y)
    }
    
    pub fn get_player_id_at_coord(&self, x: Coord, y: Coord) -> Option<PlayerID> {
        //unwrap is OK here because if you're requesting negative indicies, you want the program to crash!  
        let xx = usize::try_from(x).unwrap(); 
        let yy = usize::try_from(y).unwrap(); 
        self.board[xx][yy]
    } 
    pub fn get_player_at_coord(&self, x: Coord, y: Coord) -> Option<&FieldedPlayer> {
        match self.get_player_id_at_coord(x, y){
            None => None, 
            Some(id) => Some(self.get_player(id).unwrap()), 
            //above unwrap is safe for bad input. If it panics it's an interal logical error!  
        }
    }

    pub fn get_player(&self, id: PlayerID) -> Result<&FieldedPlayer> {
        match &self.fielded_players[id] {
            Some(player) => Ok(player), 
            None => Err(Box::new(InvalidPlayerId{id})), 
        }
    }

    pub fn get_adj_positions(&self, p: Position) -> impl Iterator<Item=Position> {
        match p {
            Position{x, ..} if x == 0 || x >= WIDTH_ => panic!(), 
            Position{y, ..} if y == 0 || y >= HEIGHT_ => panic!(), 
            Position{x, y} => DIRECTIONS.iter().map(move |(dx, dy)| Position{x: x+dx, y: y+dy}), 
        }
    } 

    pub fn get_adj_players(&self, p: Position) -> impl Iterator<Item=&FieldedPlayer> + '_ {
        self.get_adj_positions(p).filter_map(|adj_pos|self.get_player_at(adj_pos))
    } 
    
    pub fn get_mut_player(&mut self, id: PlayerID) -> Result<&mut FieldedPlayer> {
        match &mut self.fielded_players[id] {
            Some(player) => Ok(player), 
            None => Err(Box::new(InvalidPlayerId{id: 0})), 
        }
    }

    pub fn move_player(&mut self, id: PlayerID, new_pos: Position) -> Result<()>{
        let (old_x, old_y) = self.get_player(id)?.position.to_usize()?; 
        let (new_x, new_y) = new_pos.to_usize()?; 
        if self.board[new_x][new_y].is_some() {
            return Err(Box::new(InvalidPlayerId{id: 5} ))
        }
        self.board[old_x][old_y] = None; 
        self.get_mut_player(id)?.position = new_pos; 
        self.board[new_x][new_y] = Some(id); 
        Ok(())
    }

    pub fn field_player(&mut self, player_stats: PlayerStats, position: Position) -> Result<PlayerID> {

        let (new_x, new_y) = position.to_usize()?; 
        if self.board[new_x][new_y].is_some() {
            return Err(Box::new(InvalidPlayerId{id: 5} ))
        }
        
        let id = match self.fielded_players.iter().enumerate()
                                            .find(|(_, player)| player.is_none()) 
                                {
                                    Some((id, _)) => id, 
                                    None => return Err(Box::new(InvalidPlayerId{id: 5} )), //todo 
                                }; 

        self.board[new_x][new_y] = Some(id); 
        self.fielded_players[id] = Some(FieldedPlayer{ id, stats: player_stats, position, status: PlayerStatus::Up, used: false, moves: 0 });
        Ok(id) 
    }

    pub fn unfield_player(&mut self, id: PlayerID, place: DogoutPlace) -> Result<()> {

        let player = self.get_player(id)?; 
        let (x, y) = player.position.to_usize()?; 

        let dugout_player = DugoutPlayer{ stats: player.stats, place, }; 
        self.dugout_players.push(dugout_player); 

        self.board[x][y] = None; 
        self.fielded_players[id] = None; 
        Ok(())
    }

    pub fn push_proc(&mut self, proc: Box<dyn Procedure>) {
        self.new_procs.push_back(proc); 
    }
    
    pub fn step(&mut self, action: Action) -> Result<()> {
        //let mut top_proc = match self.proc_stack.pop() {
        //    Some(proc) => proc, 
        //    None => return Err(Box::new(InvalidPlayerId{id: 0})), 
        //};
        
        let mut top_proc = self.proc_stack.pop()
            .ok_or_else(|| Box::new(InvalidPlayerId{id: 0}))?;  
        
        let mut top_proc_is_finished = top_proc.step(self, Some(action)); 
        
        //todo: Check that action is allowed. 
        loop {
            if self.game_over {
                break;
            }
            top_proc = match (self.new_procs.pop_back(), top_proc_is_finished) {
                (Some(new_top_proc), false) => {self.proc_stack.push(top_proc); new_top_proc}
                (Some(new_top_proc), true) => new_top_proc,
                (None, false) => top_proc,
                (None, true) => {self.proc_stack.pop()
                                        .ok_or_else(|| Box::new(InvalidPlayerId{id: 1}))?}
            };
            
            while let Some(new_proc) = self.new_procs.pop_front() {
                self.proc_stack.push(new_proc); 
            }
            
            self.available_actions = top_proc.available_actions(self); 
            if !self.available_actions.is_empty() {
                break;
            }

            top_proc_is_finished = top_proc.step(self, None ); 
        }
        Ok(())

    }
} 
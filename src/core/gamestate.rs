use core::panic;

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
        let xx = usize::try_from(x).unwrap(); 
        let yy = usize::try_from(y).unwrap(); 
        self.board[xx][yy]
    } 
    pub fn get_player_at_coord(&self, x: Coord, y: Coord) -> Option<&FieldedPlayer> {
        match self.get_player_id_at_coord(x, y){
            None => None, 
            Some(id) => Some(self.get_player_unsafe(id)),
        }
    }

    pub fn get_player_unsafe(&self, id: PlayerID) -> &FieldedPlayer {
        match &self.fielded_players[id] {
            Some(player) => player, 
            None => panic!(), 
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
    
    pub fn get_mut_player_unsafe(&mut self, id: PlayerID) -> &mut FieldedPlayer {
        match &mut self.fielded_players[id] {
            Some(player) => player, 
            None => panic!(), 
        }
    }

    pub fn move_player(&mut self, id: PlayerID, new_pos: Position){
        let (old_x, old_y) = self.get_player_unsafe(id).position.to_usize(); 
        let (new_x, new_y) = new_pos.to_usize(); 
        if self.board[new_x][new_y].is_some() {panic!();}
        self.board[old_x][old_y] = None; 
        self.get_mut_player_unsafe(id).position = new_pos; 
        self.board[new_x][new_y] = Some(id); 
    }

    pub fn field_player(&mut self, player_stats: PlayerStats, position: Position) -> PlayerID {

        let (new_x, new_y) = position.to_usize(); 
        if self.board[new_x][new_y].is_some() {panic!();}
        
        let (id, _) = self.fielded_players.iter().enumerate()
                                .find(|(_, player)| player.is_none())
                                .unwrap(); 

        self.board[new_x][new_y] = Some(id); 
        self.fielded_players[id] = Some(FieldedPlayer{ id, stats: player_stats, position, status: PlayerStatus::Up, used: false, moves: 0 });
        id
    }

    pub fn unfield_player(&mut self, id: PlayerID, place: DogoutPlace) {

        let player = self.get_player_unsafe(id); 
        let (x, y) = player.position.to_usize(); 

        let dugout_player = DugoutPlayer{ stats: player.stats, place, }; 
        self.dugout_players.push(dugout_player); 

        self.board[x][y] = None; 
        self.fielded_players[id] = None; 
    }

    pub fn push_proc(&mut self, proc: Box<dyn Procedure>) {
        self.new_procs.push_back(proc); 
    }
    
    pub fn step(&mut self, action: Action) {
        let mut top_proc = self.proc_stack.pop().unwrap(); 
        let mut done = top_proc.step(self, Some(action)); 
        
        //todo: Check that action is allowed. 
        loop {
            if self.game_over {
                break;
            }
            top_proc = match (self.new_procs.pop_back(), done) {
                (Some(new_top_proc), true) => {new_top_proc}
                (Some(new_top_proc), false) => {self.proc_stack.push(top_proc); new_top_proc}
                (None, true) => {self.proc_stack.pop().unwrap()}
                (None, false) => {top_proc}
            };
            
            while let Some(new_proc) = self.new_procs.pop_front() {
                self.proc_stack.push(new_proc); 
            }
            
            self.available_actions = top_proc.available_actions(self); 
            if !self.available_actions.is_empty() {
                break;
            }

            done = top_proc.step(self, None ); 
        }


    }
} 
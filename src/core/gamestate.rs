use core::panic;
use std::{collections::{HashMap, VecDeque}};

use crate::core::model; 

use model::*;

use super::{table::AnyAT, procedures::Turn}; 

const DIRECTIONS: [(Coord, Coord); 8] = [(1, 1), (0, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (0, -1), (-1, -1)];  

pub struct GameStateBuilder {
    home_players: Vec<Position>, 
    away_players: Vec<Position>, 
    ball_pos: Option<Position>, 
}

impl GameStateBuilder {
    pub fn new(home_players: &[(Coord, Coord)], 
               away_players: &[(Coord, Coord)] ) -> GameStateBuilder {
        let mut builder = GameStateBuilder{
            home_players: Vec::new(), 
            away_players: Vec::new(), 
            ball_pos: None, 
        }; 

        for (x, y) in home_players {
            let p = Position{x: *x, y: *y}; 
            builder.home_players.push(p); 
        }
        for (x, y) in away_players {
            let p = Position{x: *x, y: *y}; 
            builder.away_players.push(p); 
        }
        builder
    }

    pub fn add_ball(&mut self, xy: (Coord, Coord)) -> &mut GameStateBuilder {
        self.ball_pos = Some(Position{x: xy.0, y: xy.1}); 
        self
    }

    pub fn build(&mut self) -> GameState {
                
        let mut state = GameState {
            fielded_players: Default::default(), 
            home: TeamState::new(), 
            away: TeamState::new(), 
            board: Default::default(), 
            ball: BallState::OffPitch,
            half: 1, 
            turn: 1,
            active_player: None, 
            game_over: false,
            dugout_players: Vec::new(), 
            proc_stack: Vec::new(), 
            new_procs: VecDeque::new(), 
            available_actions: HashMap::new(),
            paths: Default::default(), 
            }; 
            
        
        for position in self.home_players.iter() {
            let player_stats = PlayerStats::new(TeamType::Home); 
            _ = state.field_player(player_stats, *position)
        }

        for position in self.away_players.iter() {
            let player_stats = PlayerStats::new(TeamType::Away); 
            _ = state.field_player(player_stats, *position)
        }

        if let Some(pos) = self.ball_pos {
            state.ball = match state.get_player_at(pos) {
                None => BallState::OnGround(pos), 
                Some(p) if p.status == PlayerStatus::Up => BallState::Carried(p.id), 
                _ => panic!(),
            }
        }
        let proc = Turn{team: TeamType::Home}; 
        state.available_actions = proc.available_actions(&mut state); 
        state.proc_stack.push(Box::new(proc));
         
        state
    }

}

#[allow(dead_code)]
pub struct GameState {
    pub home: TeamState, 
    pub away: TeamState,
    fielded_players: [Option<FieldedPlayer>; 22],  
    dugout_players: Vec<DugoutPlayer>, 
    board: FullPitch<Option<PlayerID>>, 
    paths: FullPitch<Option<Path>>,
    pub ball: BallState, 
    pub half: u8, 
    pub turn: u8,
    pub active_player: Option<PlayerID>,  
    pub game_over: bool, 
    proc_stack: Vec<Box<dyn Procedure>>, //shouldn't be pub
    new_procs: VecDeque<Box<dyn Procedure>>, //shouldn't be pub
    available_actions: HashMap<AnyAT, ActionChoice>, 
    //rerolled_procs: ???? //TODO!!! 
}

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
    pub fn get_available_actions(&self) -> &HashMap<AnyAT, ActionChoice> {
        &self.available_actions
    }
    pub fn get_players_on_pitch(&self) -> impl Iterator<Item=&FieldedPlayer>{
        self.fielded_players.iter().filter_map(|x|x.as_ref())
    } 
    pub fn get_players_on_pitch_in_team(&self, team: TeamType) -> impl Iterator<Item=&FieldedPlayer>{
        self.get_players_on_pitch().filter(move|p| p.stats.team == team)
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
        if let BallState::Carried(carrier_id) = self.ball {
            if carrier_id == id {
                return Err(Box::new(InvalidPlayerId{id: 4}))
            }
        }

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
        
        assert!(self.is_legal_action(&action)); 

        let mut top_proc = self.proc_stack.pop()
            .ok_or_else(|| Box::new(InvalidPlayerId{id: 0}))?;  
        
        let mut top_proc_is_finished = top_proc.step(self, Some(action)); 
        
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
                self.proc_stack.push(top_proc); 
                break;
            }

            top_proc_is_finished = top_proc.step(self, None ); 
        }
        Ok(())

    }

    pub fn is_legal_action(&mut self, action: &Action) -> bool {
        
        let top_proc = self.proc_stack.pop().unwrap(); 
        debug_assert_eq!(top_proc.available_actions(self), self.available_actions); 
        self.proc_stack.push(top_proc); 
         
        action_in_aa(&self.available_actions, action)
    }
}

fn action_in_aa(available_actions: &HashMap<AnyAT, ActionChoice>, action: &Action) -> bool {
    match *action {
        Action::Simple(at) => available_actions.get(&AnyAT::from(at)).is_some(), 
        Action::Positional(at, position) => {
            match available_actions.get(&AnyAT::from(at)) {
                Some(ActionChoice::Positional(positions)) => positions.contains(&position),
                _ => false, 
            }
        }
    }
} 

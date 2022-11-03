use std::collections::HashMap;
use std::collections::VecDeque;

use crate::core::table; 

pub type PlayerID = usize; 
pub type Coord = i8; 
pub type FullPitch<T> = [[T; HEIGHT]; WIDTH]; 

pub const WIDTH: usize = 26; 
pub const WIDTH_: Coord = 26; 
pub const HEIGHT: usize = 17;
pub const HEIGHT_: Coord = 17; 

use std::error;
use std::fmt;

use super::table::AnyAT;
use super::table::PosAT;
use super::table::SimpleAT;

// Change the alias to `Box<error::Error>`.
pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

#[derive(Debug, Clone)]
pub struct InvalidPlayerId{
    pub id: PlayerID, 
}

impl error::Error for InvalidPlayerId {
    
}

impl fmt::Display for InvalidPlayerId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Not valid PlayerId: {}", self.id)
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Position {
    pub x: Coord, 
    pub y: Coord, 
}
impl Position{
    pub fn to_usize(&self) -> Result<(usize, usize)> {
        let x: usize = usize::try_from(self.x)?;
        let y: usize = usize::try_from(self.y)?; 
        Ok((x, y))
    }
    pub fn from_usize(x: usize, y: usize) -> Result<Position> {
        let x_: Coord = Coord::try_from(x)?; 
        let y_: Coord = Coord::try_from(y)?; 
        Ok(Position{x: x_, y: y_} )
    }
}


pub enum Action {
    Positional(table::PosAT, Position),
    Simple(table::SimpleAT), 
}

pub enum ActionChoice {
    Positional(Vec<Position>),
    Simple, 
}


#[derive(Debug, PartialEq, Eq)]
pub enum PlayerStatus{
    Up, 
    Down, 
    Stunned, 
}

#[derive(Debug, Copy, Clone)]
pub struct PlayerStats{
    pub str_: u8, 
    pub ma: u8, 
    pub ag: u8, 
    pub av: u8,
    pub team: TeamType,  
    //skills: [Option<table::Skill>; 3],  
    //injuries 
    //spp 
}
impl PlayerStats{
    pub fn new(team: TeamType) -> PlayerStats {
        PlayerStats { str_: 3, ma: 6, ag: 3, av: 8, team}
    }
}

#[derive(Debug)]
pub enum DogoutPlace {
    Reserves, 
    Heated,
    KnockOut, 
    Injuried, 
    Ejected, 
}

pub struct DugoutPlayer{
    pub stats: PlayerStats, 
    pub place: DogoutPlace, 
}

#[derive(Debug)]
pub struct FieldedPlayer{
    pub id: PlayerID, 
    pub stats: PlayerStats, 
    pub position: Position, 
    pub status: PlayerStatus, 
    pub used: bool, 
    pub moves: u16, 
    //bone_headed: bool
    //hypnotized: bool
    //really_stupid: bool
    //wild_animal: bool
    //taken_root: bool
    //blood_lust: bool
    //picked_up: bool
    //used_skills: Set[Skill]
    //used_dodge: bool, 
    //used_catch: bool, 
    //squares_moved: List['Square'] //might need this
    //has_blocked: bool
}

#[derive(Debug, Clone, Copy)]
pub struct TeamState {
    //bribes: u8, 
    //babes: u8, 
    //apothecaries: u8,
    //wizard_available: bool,
    //masterchef: bool,
    //score: u8,
    //turn: u8,
    //rerolls_start: u8,
    //rerolls: u8,
    //ass_coaches: u8,
    //cheerleaders: u8,
    //fame: u8,
    //reroll_used: bool,
    //time_violation: u8,
}
impl TeamState {
    fn new() -> TeamState {
        TeamState {  }       
        //TeamState { bribes: 0, score: 0, turn: 0, rerolls_start: 3, rerolls: 3, fame: 3, reroll_used: false }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TeamType{
    Home,
    Away,
}

pub enum BallState {
    OffPitch, 
    OnGround(Position), 
    Carried(PlayerID),
    InAir(Position),  
}

pub enum Weather{
    Nice, 
    Sunny,
    Rain, 
    Blizzard,
    Sweltering,
}

pub struct Path{
    steps: Vec<Position>, 
    prop: f32, 
}

pub struct GameState {
    pub home: TeamState, 
    pub away: TeamState,
    pub fielded_players: [Option<FieldedPlayer>; 22],  
    pub dugout_players: Vec<DugoutPlayer>, 
    pub board: FullPitch<Option<PlayerID>>, 
    pub paths: FullPitch<Option<Path>>,
    pub ball: BallState, 
    pub half: u8, 
    pub turn: u8,
    pub active_player: Option<PlayerID>,  
    pub game_over: bool, 
    pub proc_stack: Vec<Box<dyn Procedure>>, //shouldn't be pub
    pub new_procs: VecDeque<Box<dyn Procedure>>, //shouldn't be pub
    pub available_actions: HashMap<AnyAT, ActionChoice>, 
    //rerolled_procs: ???? //TODO!!! 

}

pub trait Procedure {
    fn start(&self, g: &GameState) {}
    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool; 
    fn end(&self, g: &mut GameState) {}
    fn available_actions(&self, g: &mut GameState) -> HashMap<AnyAT, ActionChoice> {HashMap::new()}
}

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
        
        state
    }

}

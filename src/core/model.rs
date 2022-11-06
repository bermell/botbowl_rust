use std::collections::HashMap;


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

use super::gamestate::GameState;

use super::table::AnyAT;



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


#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Action {
    Positional(table::PosAT, Position),
    Simple(table::SimpleAT), 
}

#[derive(Debug, PartialEq, Eq)]
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
    #[allow(clippy::new_without_default)]
    pub fn new() -> TeamState {
        TeamState {  }       
        //TeamState { bribes: 0, score: 0, turn: 0, rerolls_start: 3, rerolls: 3, fame: 3, reroll_used: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[allow(dead_code)]
pub struct Path{
    steps: Vec<Position>, 
    prop: f32, 
}

#[allow(unused_variables)]
pub trait Procedure {
    fn start(&self, g: &GameState) {}
    fn step(&self, g: &mut GameState, action: Option<Action>) -> bool; 
    fn end(&self, g: &mut GameState) {}
    fn available_actions(&self, g: &mut GameState) -> HashMap<AnyAT, ActionChoice> {HashMap::new()}
}


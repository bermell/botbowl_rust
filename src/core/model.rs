use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::error;
use std::ops::Add;

use crate::core::table; 
use super::gamestate::GameState;
use super::table::{Skill, SimpleAT, PosAT};

pub type PlayerID = usize; 
pub type Coord = i8; 
pub type FullPitch<T> = [[T; HEIGHT]; WIDTH]; 

pub const WIDTH: usize = 26; 
pub const WIDTH_: Coord = 26; 
pub const HEIGHT: usize = 17;
pub const HEIGHT_: Coord = 17; 

// Change the alias to `Box<error::Error>`.
pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

pub fn gimmi_iter<T>(pitch: &FullPitch<T>) -> impl Iterator<Item=&T> {
    pitch.iter().flat_map(|r| r.iter())
}

pub fn gimmi_mut_iter<T>(pitch: &mut FullPitch<T>) -> impl Iterator<Item=&mut T> {
    pitch.iter_mut().flat_map(|r| r.iter_mut())
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
    pub fn distance(&self, other: &Position) -> i8 {
        max((self.x - other.x).abs(), (self.y-other.y).abs())
    }
}
impl Add<(Coord, Coord)> for Position {
    type Output = Position;

    fn add(self, rhs: (Coord, Coord)) -> Self::Output {
        Position{ x: self.x + rhs.0, y: self.y + rhs.1}
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

#[derive(Debug, Clone)]
pub struct PlayerStats{
    pub str_: u8, 
    pub ma: u8, 
    pub ag: u8, 
    pub av: u8,
    pub team: TeamType,  
    pub skills: HashSet<Skill>, 
    //skills: [Option<table::Skill>; 3],  
    //injuries 
    //spp 
}
impl PlayerStats{
    pub fn new(team: TeamType) -> PlayerStats {
        PlayerStats { str_: 3, ma: 6, ag: 3, av: 8, team, skills: HashSet::new()}
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
    pub moves: u8, 
    pub used_skills: HashSet<Skill>, 
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
impl FieldedPlayer {
    pub fn moves_left(&self) -> u8 {
        if self.moves <= self.stats.ma {
            self.stats.ma - self.moves
        } else {
            0
        }
    }
    pub fn gfis_left(&self) -> u8 {
        if self.moves <= self.stats.ma {
            2
        } else {
            2 + self.stats.ma - self.moves 
        } 
    }
    pub fn total_movement_left(&self) -> u8 {
        debug_assert!(self.moves <= self.stats.ma + 2); 
        self.stats.ma + 2 - self.moves
    }
    pub fn add_move(&mut self, num_moves: u8) {
        assert!(self.total_movement_left() >= num_moves); 
        self.moves += num_moves; 
    }
    pub fn can_use_skill(&self, skill: Skill) -> bool {
        self.has_skill(skill) && !self.used_skills.contains(&skill)
    }
    pub fn has_skill(&self, skill: Skill) -> bool {
        self.stats.skills.contains(&skill)   
    }
    pub fn use_skill(&mut self, skill: Skill) {
        let not_present_before = self.used_skills.insert(skill);
        debug_assert!(not_present_before);
    }
    pub fn reset_skills_and_moves(&mut self) {
        self.moves = 0; 
        self.used = false; 
        self.used_skills.clear(); 
    }
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
    pub rerolls: u8,
    //ass_coaches: u8,
    //cheerleaders: u8,
    //fame: u8,
    reroll_used: bool,
    //time_violation: u8,
}
impl TeamState {
    #[allow(clippy::new_without_default)]
    pub fn new() -> TeamState {
        TeamState {rerolls: 3, reroll_used: false }       
        //TeamState { bribes: 0, score: 0, turn: 0, rerolls_start: 3, rerolls: 3, fame: 3, reroll_used: false }
    }
    pub fn can_use_reroll(&self) -> bool {
        !self.reroll_used && self.rerolls > 0
    }
    pub fn use_reroll(&mut self) {
        assert!(self.can_use_reroll()); 
        self.reroll_used = true; 
        self.rerolls -= 1; 
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Weather{
    Nice, 
    Sunny,
    Rain, 
    Blizzard,
    Sweltering,
}

#[allow(unused_variables)]
pub trait Procedure {
    fn start(&self, game_state: &GameState) {}
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool; 
    fn end(&self, game_state: &mut GameState) {}
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {AvailableActions::new_empty()}
}

#[derive(Debug, PartialEq, Eq)]
pub struct AvailableActions {
    pub team: Option<TeamType>, 
    pub simple: HashSet<SimpleAT>, 
    pub positional: HashMap<PosAT, Vec<Position>>, 
}
impl AvailableActions {
    pub fn new_empty() -> Self {
        AvailableActions { team: None, simple: HashSet::new(), positional: HashMap::new() }
    }
    pub fn new(team: TeamType) -> Self {
        AvailableActions { team: Some(team), simple: HashSet::new(), positional: HashMap::new() }
    }
    pub fn is_empty(&self) -> bool {
        self.simple.is_empty() && ! self.positional.iter().any(|(_, positions)| !positions.is_empty()) 
    }  
    pub fn insert_simple(&mut self, action_type: SimpleAT) {
        assert!(self.team.is_some()); 
        self.simple.insert(action_type); 

    }
    pub fn insert_positional(&mut self, action_type: PosAT, positions: Vec<Position>) {
        assert!(self.team.is_some()); 
        assert!(!self.positional.contains_key(&action_type)); 
        self.positional.insert(action_type, positions); 
    }
    pub fn is_legal_action(&self, action: Action) -> bool {
        match action {
            Action::Positional(at, pos) => match self.positional.get(&at) {
                                                Some(legal_positions) => legal_positions.contains(&pos),
                                                None => false,
            }
            Action::Simple(at) => self.simple.contains(&at),
        }
    }
    pub fn get_team(&self) -> Option<TeamType> {
        self.team
    }
}
use core::panic;
use std::{rc::Rc, collections::HashMap, hash, cmp::{max, Ordering}, iter::zip};

use crate::core::model; 
use model::*;

use super::gamestate::{GameState, DIRECTIONS}; 

type OptRcNode = Option<Rc<Node>>; 



#[derive(Debug, Clone, Copy )]
pub enum Roll{ //Make more clever! 
    Dodge(u8), 
    GFI(u8),  
}

pub struct Node { 
    parent: OptRcNode, 
    position: Position, 
    moves_left: i8, 
    gfis_left: i8, 
    // foul_roll, handoff_roll, block_dice
    //euclidiean_distance: f32, 
    prob: f32, 
    rolls: Vec<Roll>, 
} 
impl Node {
    fn is_dominant_over(&self, othr: &Node) -> bool {
        if self.prob > othr.prob && self.moves_left + self.gfis_left > othr.moves_left + othr.gfis_left {
            return true; 
        } 
        //if othr.prob > self.prob && othr.moves_left + othr.gfis_left > self.moves_left + self.gfis_left {
        //    return Some(a); 
        //} 
        false
    }
    fn apply_gfi(&mut self, target: u8) {}
    fn apply_dodge(&mut self, target: u8) {}
    fn apply_pickup(&mut self, target: u8) {}
}
impl std::cmp::PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        //self.parent == other.parent && self.position == other.position && self.moves_left == other.moves_left && self.gfis_left == other.gfis_left && self.prob == other.prob
        todo!()
    }
}

impl std::cmp::PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<std::cmp::Ordering> {
        //Helper function
        fn is_greater_or_less(o: Option<Ordering>) -> bool {
            match o {
                Some(Ordering::Equal) => false, 
                Some(Ordering::Greater) => true, 
                Some(Ordering::Less) => true, 
                None => panic!("Expected successful comparison!"),
            }
        }
        let tmp = f32::partial_cmp(&self.prob,
                                  &other.prob);
        if is_greater_or_less(tmp) {return tmp;}

        let tmp = i8::partial_cmp(&(self.moves_left+self.gfis_left),
                                 &(other.moves_left+other.gfis_left));
        if is_greater_or_less(tmp) {return tmp;}


        Some(Ordering::Equal)
        
    }
}

#[allow(dead_code)]
pub struct Path {
    steps: Vec<(Position, Vec<Roll>)>, 
    prob: f32, 
}
impl Path {
    fn new(final_node: &Node) -> Path {
        let mut node: &Node = final_node; 
        let mut steps = vec![(node.position, node.rolls.clone())]; 
        
        while let Some(parent) = &node.parent {
            steps.push((parent.position, parent.rolls.clone()));  
            node = parent; 
        }

        Path{steps, prob: final_node.prob}
    }
}

#[allow(dead_code)]
pub struct PathFinder<'a> {
    pub game_state: &'a GameState, 
    nodes: FullPitch<OptRcNode>, 
    locked_nodes: FullPitch<OptRcNode>, 
    tzones: FullPitch<u8>, 
    ball_pos: Option<Position>, 
    max_moves: i8, 
    max_gfis: i8, 
    open_set: Vec<Rc<Node>>, 
    start_pos: Position, 
    risky_sets: RiskySet, 
} 

impl<'a> PathFinder <'a>{
    pub fn new(game_state: &'a mut GameState) -> PathFinder<'a> {
        PathFinder { game_state, 
                     nodes: Default::default(),
                     locked_nodes: Default::default(),
                     tzones: Default::default(), 
                     max_moves: 0, 
                     ball_pos: match game_state.ball {
                        BallState::OnGround(position) => Some(position), 
                        _ => None, 
                     }, 
                     max_gfis: 0, 
                     open_set: Vec::new(), 
                     start_pos: Position { x: 0, y: 0 }, 
                     risky_sets: Default::default(), 
                    }
    }
    fn tackles_zones_at(&self, position: &Position) -> u8 {
        let (x, y) = position.to_usize().unwrap(); 
        self.tzones[x][y]
    }

    pub fn player_paths(&mut self, id: PlayerID) -> Result<FullPitch<Option<Path>>> {
        let player = self.game_state.get_player(id).unwrap(); 
        self.max_moves = i8::try_from(player.stats.ma).unwrap(); 
        self.max_gfis = 2; 
        self.start_pos = player.position;  
        
        let root_node = Rc::new( Node{ parent: None, position: self.start_pos, moves_left: self.max_moves, gfis_left: self.max_gfis, prob: 1.0, rolls: Vec::new() }); 

        self.open_set.push(root_node); 
        
        loop {
            //expansion 
            while let Some(node) = self.open_set.pop() {
                self.expand_node(node); 
            } 

            for (node, locked) in zip(gimmi_mut_iter(&mut self.nodes), gimmi_mut_iter(&mut self.locked_nodes)){
                match node {
                    Some(_) if node > locked => *locked = node.clone(), 
                    _ => (),
                }
            }

            self.nodes = Default::default(); 

            //prepare nodes 
            self.open_set = match self.risky_sets.get_next_batch() {
                                Some(new_open_set) => new_open_set, 
                                None => break, 
            }; 
        }

        let mut paths: FullPitch<Option<Path>> = Default::default(); 
        for (path, locked_node) in zip(gimmi_mut_iter(&mut paths), gimmi_iter(&self.locked_nodes)){
            if let Some(node) = locked_node {
                *path = Some(Path::new(node));
            }
        }
        Ok(paths)
    }

    fn expand_node(&mut self, node: Rc<Node>){
        let out_of_moves = node.moves_left + node.gfis_left == 0; 
        let mut parent = &node.parent; 
        
        if out_of_moves /*and can't handoff or foul*/ {
            return 
        }
        if let Some(parent_node) = parent {
            if parent_node.position == node.position {
                parent = &None; 
            }
        }
        // stop if block roll or handoff roll is set

        // stop if carries ball and node.position is in endzone

        // stop if not carries ball and node.position is ball
        
        for direction in DIRECTIONS {
            let to_square = node.position + direction;  
            if to_square.x == 0 || to_square.x == WIDTH_ || to_square.y == 0 || to_square.y == HEIGHT_ {
                // out of bounds
                continue;
            }
            if let Some( Node { position, ..  }) = parent.as_deref() {
                // filter bad paths early
                if position.distance(&to_square) < 2 && (self.tackles_zones_at(position) == 0 || self.tackles_zones_at(&to_square) == 0) {
                    continue;
                } 
            } 
            
            match self.expand_to(&node, to_square) {
                Some(risky_node) if risky_node.prob < node.prob  => self.risky_sets.insert_node(risky_node), 
                Some(better_node) => {
                    self.open_set.push(better_node.clone()); 
                    let (x, y) = better_node.position.to_usize().unwrap(); 
                    self.nodes[x][y] = Some(better_node); 
                }
                None => (), 
            }
        }
    }

    fn expand_to(&self, from_node: &Rc<Node>, to: Position) -> OptRcNode { 
        match self.game_state.get_player_at(to) {
            Some(_) => None, 
            None => self.expand_move_to(from_node, to), 
        }
    }
    fn expand_move_to(&self, from_node: &Rc<Node>, to: Position) -> OptRcNode { 
        let gfi = from_node.moves_left == 0; 
        let (x, y) = to.to_usize().unwrap(); 
        let moves_left_next = max(0, from_node.moves_left-1); 
        let gfis_left_next = from_node.gfis_left - i8::from(gfi);   

        if let Some(best_node) = &self.nodes[x][y] {
            if moves_left_next + gfis_left_next <= best_node.moves_left + best_node.gfis_left{
                return None; 
            }
        }

        let mut next_node =  Node{ parent: from_node.parent.clone(), 
                                           position: to, 
                                           moves_left: moves_left_next, 
                                           gfis_left: gfis_left_next, 
                                           prob: from_node.prob, 
                                           rolls: Vec::new()}; 
        if gfi {next_node.apply_gfi(2);}
        if self.tzones[x][y] > 0 {next_node.apply_dodge(3);}
        match self.ball_pos {
            Some(ball_pos) if ball_pos == to => next_node.apply_pickup(3), 
            _ => (),
        } 
        if let Some(best_before) = &self.locked_nodes[x][y]{
            if best_before.is_dominant_over(&next_node) {
                return None; 
            }
        } 
        Some(Rc::new(next_node))
    }
}
 

#[derive(Default)]
struct RiskySet{
    set: HashMap<HashableFloat, Vec<Rc<Node>>>, 
}
impl RiskySet {
    pub fn insert_node(&mut self, node: Rc<Node>) {
        assert!(0_f32 < node.prob && node.prob <= 1.0_f32); 
        let prob = HashableFloat(node.prob); 
        self.set.entry(prob).or_insert_with(Vec::new);
    }
    pub fn get_next_batch(&mut self) -> Option<Vec<Rc<Node>>> {
        match self.set.keys().map(|hf| hf.0).reduce(f32::max){
            Some(max_prob) => self.set.remove(&HashableFloat(max_prob)), 
            None => None, 
        }
    }
}

// Nasty workaround to get hashable floats
#[derive(Debug, Copy, Clone)]
struct HashableFloat(f32);

impl HashableFloat {
    fn key(&self) -> u32 {
        self.0.to_bits()
    }
}

impl hash::Hash for HashableFloat {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for HashableFloat {
    fn eq(&self, other: &HashableFloat) -> bool {
        self.key() == other.key()
    }
}

impl Eq for HashableFloat {}

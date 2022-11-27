use std::fmt::Debug;
use std::{collections::HashMap, hash, iter::zip, rc::Rc};

use crate::core::model;
use model::*;

use super::dices::{D6Target, RollTarget};
use super::gamestate::{GameState, DIRECTIONS};

type OptRcNode = Option<Rc<Node>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Roll {
    //Make more clever!
    Dodge(D6Target),
    GFI(D6Target),
    Pickup(D6Target),
    //StandUp,
}
#[derive(Debug)]
pub struct Node {
    parent: OptRcNode,
    position: Position,
    moves_left: u8,
    gfis_left: u8,
    // foul_roll, handoff_roll, block_dice
    //euclidiean_distance: f32,
    prob: f32,
    rolls: Vec<Roll>,
}
impl Node {
    fn apply_gfi(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.rolls.push(Roll::GFI(target));
    }
    fn apply_dodge(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.rolls.push(Roll::Dodge(target));
    }
    fn apply_pickup(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.rolls.push(Roll::Pickup(target));
    }

    fn is_dominant_over(&self, othr: &Node) -> bool {
        assert_eq!(self.position, othr.position);

        let self_moves_left = self.moves_left + self.gfis_left;
        let othr_moves_left = othr.moves_left + othr.gfis_left;

        if self.prob > othr.prob && self_moves_left > othr_moves_left {
            return true;
        }
        false
    }

    fn is_better_than(&self, othr: &Node) -> bool {
        assert_eq!(self.position, othr.position);

        if self.prob > othr.prob {
            return true;
        }
        if self.prob < othr.prob {
            return false;
        }

        let self_moves_left = self.moves_left + self.gfis_left;
        let othr_moves_left = othr.moves_left + othr.gfis_left;

        if self_moves_left > othr_moves_left {
            return true;
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub steps: Vec<(Position, Vec<Roll>)>,
    pub target: Position,
    pub prob: f32,
}
impl Path {
    fn new(final_node: &Node) -> Path {
        let mut path = Path {
            steps: vec![(final_node.position, final_node.rolls.clone())],
            prob: final_node.prob,
            target: final_node.position,
        };
        let mut node: &Node = final_node;
        while let Some(parent) = &node.parent {
            if parent.parent.is_none() {
                //We don't want the root node here
                break;
            }
            path.steps.push((parent.position, parent.rolls.clone()));
            node = parent;
        }
        path
    }
}

#[allow(dead_code)]
pub struct PathFinder<'a> {
    game_state: &'a GameState,
    nodes: FullPitch<OptRcNode>,
    locked_nodes: FullPitch<OptRcNode>,
    tzones: FullPitch<i8>,
    ball_pos: Option<Position>,
    ag: u8,
    open_set: Vec<Rc<Node>>,
    start_pos: Position,
    risky_sets: RiskySet,
    dodge_target: D6Target,
    gfi_target: D6Target,
    pickup_target: D6Target,
}

impl<'a> PathFinder<'a> {
    pub fn new(game_state: &'a GameState) -> PathFinder<'a> {
        PathFinder {
            game_state,
            nodes: Default::default(),
            locked_nodes: Default::default(),
            tzones: Default::default(),
            ball_pos: match game_state.ball {
                BallState::OnGround(position) => Some(position),
                _ => None,
            },
            ag: 0,
            open_set: Vec::new(),
            start_pos: Position::new((0, 0)),
            risky_sets: Default::default(),
            dodge_target: D6Target::SixPlus,
            gfi_target: D6Target::TwoPlus,
            pickup_target: D6Target::SixPlus,
        }
    }
    fn tackles_zones_at(&self, position: &Position) -> i8 {
        let (x, y) = position.to_usize().unwrap();
        self.tzones[x][y]
    }

    pub fn player_paths(&mut self, id: PlayerID) -> Result<FullPitch<Option<Path>>> {
        let player = self.game_state.get_player_unsafe(id);
        self.start_pos = player.position;
        self.ag = player.stats.ag;
        self.dodge_target = *player.ag_target().add_modifer(1);
        self.pickup_target = *player.ag_target().add_modifer(1);
        if let Weather::Blizzard = self.game_state.weather {
            self.gfi_target.add_modifer(-1);
        }
        if let Weather::Rain = self.game_state.weather {
            self.pickup_target.add_modifer(-1);
        }

        let root_node = Rc::new(Node {
            parent: None,
            position: self.start_pos,
            moves_left: player.moves_left(),
            gfis_left: player.gfis_left(),
            prob: 1.0,
            rolls: Vec::new(),
        });

        let team = player.stats.team;
        self.game_state
            .get_players_on_pitch()
            .filter(|player| player.stats.team != team)
            .filter(|player| player.has_tackle_zone())
            .flat_map(|player| self.game_state.get_adj_positions(player.position))
            .map(|position| position.to_usize().unwrap())
            .for_each(|(x, y)| self.tzones[x][y] += 1);

        self.open_set.push(root_node);
        loop {
            //expansion
            while let Some(node) = self.open_set.pop() {
                self.expand_node(node);
            }

            //clear
            for (node, locked) in zip(
                gimmi_mut_iter(&mut self.nodes),
                gimmi_mut_iter(&mut self.locked_nodes),
            ) {
                match (&node, &locked) {
                    (Some(n), Some(l)) if n.is_better_than(l) => *locked = node.clone(),
                    (Some(_), None) => *locked = node.clone(),
                    _ => (),
                }
            }
            self.nodes = Default::default();

            //prepare nodes
            match self.risky_sets.get_next_batch() {
                None => break,
                Some(new_open_set) => {
                    for new_node in new_open_set {
                        let (x, y) = new_node.position.to_usize().unwrap();
                        if let Some(best_before) = &self.locked_nodes[x][y] {
                            if !new_node.is_dominant_over(best_before) {
                                continue;
                            }
                        }
                        let current_best = &mut self.nodes[x][y];
                        if current_best.is_some()
                            && !new_node.is_better_than(current_best.as_ref().unwrap())
                        {
                            continue;
                        }
                        *current_best = Some(new_node.clone());
                        self.open_set.push(new_node);
                    }
                }
            };
            if self.open_set.is_empty() && self.risky_sets.is_empty() {
                break;
            }
        }

        let mut paths: FullPitch<Option<Path>> = Default::default();
        for (path, locked_node) in zip(gimmi_mut_iter(&mut paths), gimmi_iter(&self.locked_nodes)) {
            if let Some(node) = locked_node {
                *path = Some(Path::new(node));
            }
        }
        Ok(paths)
    }

    fn expand_node(&mut self, node: Rc<Node>) {
        let out_of_moves = node.moves_left + node.gfis_left == 0;
        let mut parent = &node.parent;

        if out_of_moves
        /*and can't handoff or foul*/
        {
            return;
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
            if to_square.is_out() {
                continue;
            }
            if let Some(Node { position, .. }) = parent.as_deref() {
                // filter bad paths early
                if position.distance(&to_square) < 2
                    && (self.tackles_zones_at(position) == 0
                        || self.tackles_zones_at(&to_square) == 0)
                {
                    continue;
                }
            }

            match self.expand_to(&node, to_square) {
                Some(risky_node) if risky_node.prob < node.prob => {
                    self.risky_sets.insert_node(risky_node)
                }
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
        let (to_x, to_y) = to.to_usize().unwrap();
        let moves_left_next = match gfi {
            true => 0,
            false => from_node.moves_left - 1,
        };
        let gfis_left_next = from_node.gfis_left - u8::from(gfi);

        if let Some(current_best) = &self.nodes[to_x][to_y] {
            if moves_left_next + gfis_left_next <= current_best.moves_left + current_best.gfis_left
            {
                return None;
            }
        }
        let mut next_node = Node {
            parent: Some(from_node.clone()),
            position: to,
            moves_left: moves_left_next,
            gfis_left: gfis_left_next,
            prob: from_node.prob,
            rolls: Vec::new(),
        };
        if gfi {
            next_node.apply_gfi(self.gfi_target);
        }
        if self.tackles_zones_at(&from_node.position) > 0 {
            next_node.apply_dodge(
                *self
                    .dodge_target
                    .clone()
                    .add_modifer(-self.tzones[to_x][to_y]),
            );
        }
        match self.ball_pos {
            Some(ball_pos) if ball_pos == to => next_node.apply_pickup(
                *self
                    .pickup_target
                    .clone()
                    .add_modifer(-self.tzones[to_x][to_y]),
            ),
            _ => (),
        }

        let next_node = next_node; //we're done mutating.

        if let Some(best_before) = &self.locked_nodes[to_x][to_y] {
            assert!(best_before.prob > next_node.prob);
            if !next_node.is_dominant_over(best_before) {
                return None;
            }
        }
        Some(Rc::new(next_node))
    }
}

#[derive(Default)]
struct RiskySet {
    set: HashMap<HashableFloat, Vec<Rc<Node>>>,
}
impl RiskySet {
    pub fn insert_node(&mut self, node: Rc<Node>) {
        assert!(0_f32 < node.prob && node.prob <= 1.0_f32);
        let prob = HashableFloat(node.prob);
        self.set.entry(prob).or_insert_with(Vec::new).push(node);
    }
    pub fn get_next_batch(&mut self) -> Option<Vec<Rc<Node>>> {
        match self.set.keys().map(|hf| hf.0).reduce(f32::max) {
            Some(max_prob) => self.set.remove(&HashableFloat(max_prob)),
            None => None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}
impl Debug for RiskySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RiskySet")
            .field("len", &self.set.len())
            .finish()
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

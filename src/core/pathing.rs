use std::fmt::Debug;
use std::{collections::HashMap, hash, iter::zip, rc::Rc};

use crate::core::model;
use model::*;

use super::dices::{D6Target, RollTarget};
use super::gamestate::GameState;
use super::table::NumBlockDices;

type OptRcNode = Option<Rc<Node>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Roll {
    Dodge(D6Target),
    GFI(D6Target),
    Pickup(D6Target),
    Block(PlayerID, NumBlockDices),
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
    fn remaining_movement(&self) -> u8 {
        self.moves_left + self.gfis_left
    }
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

        if self.prob > othr.prob && self.remaining_movement() > othr.remaining_movement() {
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

        if self.remaining_movement() > othr.remaining_movement() {
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
    nodes: FullPitch<OptRcNode>,
    locked_nodes: FullPitch<OptRcNode>,
    open_set: Vec<Rc<Node>>,
    risky_sets: RiskySet,
    info: GameInfo<'a>,
}

enum NodeType {
    Risky(Rc<Node>),
    ContinueExpanding(Rc<Node>),
    NoNode,
}
#[derive(Debug)]
enum PathingBallState {
    IsCarrier(Coord),
    OnGround(Position),
    NotRelevant,
}

//This struct gather all infomation needed about the board
struct GameInfo<'a> {
    game_state: &'a GameState,
    tzones: FullPitch<i8>,
    ball: PathingBallState,
    start_pos: Position,
    dodge_target: D6Target,
    gfi_target: D6Target,
    pickup_target: D6Target,
}
impl<'a> GameInfo<'a> {
    fn tackles_zones_at(&self, position: &Position) -> i8 {
        let (x, y) = position.to_usize().unwrap();
        self.tzones[x][y]
    }

    fn new(game_state: &'a GameState, player: &FieldedPlayer) -> GameInfo<'a> {
        let dodge_target = *player.ag_target().add_modifer(1);
        let mut gfi_target = D6Target::TwoPlus;
        let mut pickup_target = *player.ag_target().add_modifer(1);

        if game_state.info.weather == Weather::Blizzard {
            gfi_target.add_modifer(-1);
        }
        if game_state.info.weather == Weather::Rain {
            pickup_target.add_modifer(-1);
        }

        let team = player.stats.team;
        let mut tzones: FullPitch<i8> = Default::default();
        game_state
            .get_players_on_pitch()
            .filter(|player| player.stats.team != team)
            .filter(|player| player.has_tackle_zone())
            .flat_map(|player| game_state.get_adj_positions(player.position))
            .map(|position| position.to_usize().unwrap())
            .for_each(|(x, y)| tzones[x][y] += 1);

        GameInfo {
            tzones,
            ball: match game_state.ball {
                BallState::OnGround(position) => PathingBallState::OnGround(position),
                BallState::Carried(id) if id == player.id => {
                    PathingBallState::IsCarrier(game_state.get_endzone_x(player.stats.team))
                }
                _ => PathingBallState::NotRelevant,
            },
            start_pos: player.position,
            dodge_target,
            gfi_target,
            pickup_target,
            game_state,
        }
    }
    fn can_continue_expanding(&self, node: &Rc<Node>) -> bool {
        if node.remaining_movement() == 0 {
            //todo: and can't handoff here.
            return false;
        }
        // todo: stop if block roll or handoff roll is set

        match self.ball {
            PathingBallState::IsCarrier(endzone_x) if endzone_x == node.position.x => false,
            PathingBallState::OnGround(ball_pos) if ball_pos == node.position => false,
            _ => true,
        }
    }

    fn expand_to(
        &self,
        to: Position,
        parent_node: &Rc<Node>,
        prev: &mut OptRcNode,
        best: &OptRcNode,
    ) -> NodeType {
        debug_assert!(self.can_continue_expanding(parent_node));

        // expand to move_node, block_node, handoff_mode
        let new_node: Option<Node> = match self.game_state.get_player_id_at(to) {
            Some(_) => return NodeType::NoNode,
            None => self.expand_move_to(to, parent_node, prev),
        };

        let new_node: Rc<Node> = match new_node {
            Some(node) => Rc::new(node),
            None => return NodeType::NoNode,
        };

        if let Some(best_before) = &best {
            debug_assert!(best_before.prob > new_node.prob); //this is only here to remind us of this fact
            if !best_before.is_dominant_over(&new_node) {
                return NodeType::NoNode;
            }
        }

        if new_node.prob < parent_node.prob {
            return NodeType::Risky(new_node);
        }

        if let Some(previous) = prev {
            debug_assert!(new_node.is_better_than(previous)); //this should be the case!
        }

        *prev = Some(new_node.clone());

        if self.can_continue_expanding(&new_node) {
            NodeType::ContinueExpanding(new_node)
        } else {
            NodeType::NoNode
        }
    }

    fn expand_move_to(
        &self,
        to: Position,
        parent_node: &Rc<Node>,
        prev: &OptRcNode,
    ) -> Option<Node> {
        let gfi = parent_node.moves_left == 0;
        let (to_x, to_y) = to.to_usize().unwrap();

        if let Some(current_best) = &prev {
            if parent_node.remaining_movement() - 1 <= current_best.remaining_movement() {
                return None;
            }
        }
        let (moves_left, gfis_left) = match gfi {
            true if parent_node.gfis_left > 0 => (0, parent_node.gfis_left - 1),
            true => (0, 0),
            false => (parent_node.moves_left - 1, parent_node.gfis_left),
        };

        let mut next_node = Node {
            parent: Some(parent_node.clone()),
            position: to,
            moves_left,
            gfis_left,
            prob: parent_node.prob,
            rolls: Vec::new(),
        };
        if gfi {
            next_node.apply_gfi(self.gfi_target);
        }
        if self.tackles_zones_at(&parent_node.position) > 0 {
            next_node.apply_dodge(
                *self
                    .dodge_target
                    .clone()
                    .add_modifer(-self.tzones[to_x][to_y]),
            );
        }
        if matches!(self.ball, PathingBallState::OnGround(ball_pos) if ball_pos == to ) {
            next_node.apply_pickup(
                *self
                    .pickup_target
                    .clone()
                    .add_modifer(-self.tzones[to_x][to_y]),
            );
        }

        Some(next_node)
    }
}

impl<'a> PathFinder<'a> {
    pub fn player_paths(game_state: &GameState, id: PlayerID) -> Result<FullPitch<Option<Path>>> {
        let player = game_state.get_player_unsafe(id);
        let mut pf = PathFinder {
            nodes: Default::default(),
            locked_nodes: Default::default(),
            open_set: Default::default(),
            risky_sets: Default::default(),
            info: GameInfo::new(game_state, player),
        };

        let root_node = Rc::new(Node {
            parent: None,
            position: pf.info.start_pos,
            moves_left: player.moves_left(),
            gfis_left: player.gfis_left(),
            prob: 1.0,
            rolls: Vec::new(),
        });

        assert!(pf.info.can_continue_expanding(&root_node));
        pf.open_set.push(root_node);
        loop {
            //expansion
            while let Some(node) = pf.open_set.pop() {
                pf.expand_node(node);
            }

            //clear pf.nodes
            for (node, locked) in zip(
                gimmi_mut_iter(&mut pf.nodes),
                gimmi_mut_iter(&mut pf.locked_nodes),
            ) {
                match (&node, &locked) {
                    (Some(n), Some(l)) if n.is_better_than(l) => *locked = node.take(),
                    (Some(_), None) => *locked = node.take(),
                    (Some(_), _) => *node = None,
                    _ => (),
                }
            }

            //prepare nodes
            match pf.risky_sets.get_next_batch() {
                None => break,
                Some(new_open_set) => pf.prepare_nodes(new_open_set),
            };
        }

        let mut paths: FullPitch<Option<Path>> = Default::default();
        for (path, locked_node) in zip(gimmi_mut_iter(&mut paths), gimmi_iter(&pf.locked_nodes)) {
            if let Some(node) = locked_node {
                *path = Some(Path::new(node));
            }
        }
        Ok(paths)
    }

    fn prepare_nodes(&mut self, new_nodes: Vec<Rc<Node>>) {
        for new_node in new_nodes {
            let (x, y) = new_node.position.to_usize().unwrap();
            if self.locked_nodes[x][y]
                .as_ref()
                .map(|locked| locked.is_dominant_over(&new_node))
                .unwrap_or(false)
            {
                continue;
            }

            let best_in_batch = &mut self.nodes[x][y];
            if let Some(best_in_batch) = &best_in_batch {
                debug_assert!((best_in_batch.prob - new_node.prob).abs() < 0.001);
                if !new_node.is_better_than(best_in_batch) {
                    continue;
                }
            }
            *best_in_batch = Some(new_node.clone());

            if self.info.can_continue_expanding(&new_node) {
                self.open_set.push(new_node);
            }
        }
    }

    fn expand_node(&mut self, node: Rc<Node>) {
        debug_assert!(self.info.can_continue_expanding(&node));

        let parent_pos_and_in_tz: Option<(Position, bool)> = node
            .parent
            .as_ref()
            .filter(|parent| parent.position != node.position)
            .map(|parent| {
                (
                    parent.position,
                    self.info.tackles_zones_at(&parent.position) > 0,
                )
            });

        Direction::all_directions_iter()
            .map(|direction| node.position + *direction)
            .filter(|to_pos| !to_pos.is_out())
            .map(|to_pos| (to_pos, to_pos.to_usize().unwrap()))
            .filter(|(to_pos, (x, y))| {
                parent_pos_and_in_tz
                    .map(|(parent_pos, parent_in_tz)| {
                        parent_pos.distance_to(to_pos) == 2
                            || (parent_in_tz && 0 < self.info.tzones[*x][*y])
                    })
                    .unwrap_or(true)
            })
            .map(|(to_square, (x, y))| {
                self.info.expand_to(
                    to_square,
                    &node,
                    &mut self.nodes[x][y],
                    &self.locked_nodes[x][y],
                )
            })
            .for_each(|node_type| match node_type {
                NodeType::Risky(node) => self.risky_sets.insert_node(node),
                NodeType::ContinueExpanding(node) => {
                    debug_assert!(self.info.can_continue_expanding(&node));
                    self.open_set.push(node);
                }
                NodeType::NoNode => (),
            });
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

use std::fmt::Debug;
use std::{collections::HashMap, hash, iter::zip, rc::Rc};

use crate::core::model;
use itertools::Either;
use model::*;

use super::dices::{D6Target, RollTarget, Sum2D6Target};
use super::gamestate::GameState;
use super::table::{NumBlockDices, PosAT};

type OptRcNode = Option<Rc<Node>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathingEvent {
    Dodge(D6Target),
    GFI(D6Target),
    Pickup(D6Target),
    Block(PlayerID, NumBlockDices),
    Handoff(PlayerID, D6Target),
    Touchdown(PlayerID),
    Foul(PlayerID, Sum2D6Target),
    StandUp,
}

pub fn event_ends_player_action(event: &PathingEvent) -> bool {
    match event {
        PathingEvent::Handoff(_, _) => true,
        PathingEvent::Foul(_, _) => true,
        PathingEvent::Touchdown(_) => true,
        PathingEvent::Dodge(_) => false,
        PathingEvent::GFI(_) => false,
        PathingEvent::Pickup(_) => false,
        PathingEvent::Block(_, _) => false,
        PathingEvent::StandUp => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedQueue<T> {
    data: [Option<T>; 6],
}

impl<T> Default for FixedQueue<T> {
    fn default() -> Self {
        FixedQueue {
            data: Default::default(),
        }
    }
}
impl<T> FixedQueue<T> {
    pub fn len(&self) -> usize {
        self.data.iter().filter(|val| val.is_some()).count()
    }
    pub fn push_back(&mut self, val: T) {
        self.add(val)
    }
    pub fn add(&mut self, val: T) {
        assert!(!self.is_full());

        let next_entry = self.data.iter_mut().find(|val| val.is_none()).unwrap();
        *next_entry = Some(val);
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        self.data
            .iter_mut()
            .find(|val| val.is_some())
            .unwrap()
            .take()
    }
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|entry| entry.is_none())
    }
    pub fn is_full(&self) -> bool {
        self.data[5].is_some()
    }
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.data
                .iter()
                .rev()
                .filter(|val| val.is_some())
                .flatten()
                .next()
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|item| item.as_ref())
    }
    pub fn iter_rev(&self) -> impl Iterator<Item = &T> {
        self.data.iter().rev().filter_map(|item| item.as_ref())
    }
}
impl<T> From<Vec<T>> for FixedQueue<T> {
    fn from(vector: Vec<T>) -> Self {
        let mut q: Self = Default::default();
        vector.into_iter().for_each(|val| q.add(val));
        q
    }
}

#[derive(Debug)]
pub struct NodeIterator {
    stack: Vec<NodeIteratorItem>,
}

pub type NodeIteratorItem = Either<Position, PathingEvent>;

impl NodeIterator {
    fn new(node: &Rc<Node>) -> Self {
        let mut queue = Vec::new();
        let mut n = node.clone();

        //this will ensure we ignore the root node
        while let Some(parent) = &n.parent {
            n.add_iter_items(&mut queue);
            n = parent.clone();
        }
        n.add_iter_items(&mut queue); //root node

        Self { stack: queue }
    }
    pub fn len(&self) -> usize {
        self.stack.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

impl Iterator for NodeIterator {
    type Item = NodeIteratorItem;

    fn next(&mut self) -> Option<NodeIteratorItem> {
        self.stack.pop()
    }
}

pub trait CustomIntoIter {
    fn iter(&self) -> NodeIterator;
}
impl CustomIntoIter for Rc<Node> {
    fn iter(&self) -> NodeIterator {
        NodeIterator::new(self)
    }
}

#[derive(Debug)]
pub struct Node {
    parent: OptRcNode,
    pub position: Position,
    moves_left: u8,
    gfis_left: u8,
    block_dice: Option<NumBlockDices>,
    // foul_roll, handoff_roll, block_dice
    //euclidiean_distance: f32,
    pub prob: f32,
    events: FixedQueue<PathingEvent>,
}
impl Node {
    pub fn get_block_dice(&self) -> Option<NumBlockDices> {
        self.block_dice
    }
    fn add_iter_items(&self, items: &mut Vec<NodeIteratorItem>) {
        for event in self.events.iter_rev() {
            items.push(Either::Right(*event));
        }
        if self.move_to_position() {
            items.push(Either::Left(self.position));
        }
    }
    pub fn move_to_position(&self) -> bool {
        if self.parent.is_none() {
            return false;
        }
        if let Some(event) = self.events.last() {
            match event {
                PathingEvent::Block(_, _) => false,
                PathingEvent::Handoff(_, _) => false,
                PathingEvent::Foul(_, _) => false,
                PathingEvent::StandUp => false,
                PathingEvent::Dodge(_) => true,
                PathingEvent::GFI(_) => true,
                PathingEvent::Pickup(_) => true,
                PathingEvent::Touchdown(_) => true,
            }
        } else {
            true
        }
    }

    pub fn new_direct_block_node(block_dice: NumBlockDices, position: Position) -> Node {
        Node {
            parent: None,
            position,
            moves_left: 0,
            gfis_left: 0,
            block_dice: Some(block_dice),
            prob: 1.0,
            events: Default::default(),
        }
    }

    pub fn get_action_type(&self) -> PosAT {
        if self.block_dice.is_some() {
            PosAT::Block
        } else {
            match self.events.last() {
                Some(PathingEvent::Block(_, _)) => PosAT::Block,
                Some(PathingEvent::Handoff(_, _)) => PosAT::Handoff,
                Some(PathingEvent::Foul(_, _)) => PosAT::Foul,
                _ => PosAT::Move,
            }
        }
    }

    fn new(parent: OptRcNode, position: Position, moves_left: u8, gfis_left: u8) -> Node {
        Node {
            prob: parent.as_ref().map(|node| node.prob).unwrap_or(1.0),
            parent,
            position,
            moves_left,
            gfis_left,
            block_dice: None,
            events: Default::default(),
        }
    }
    fn remaining_movement(&self) -> u8 {
        self.moves_left + self.gfis_left
    }
    fn apply_gfi(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.events.push_back(PathingEvent::GFI(target));
    }
    fn apply_dodge(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.events.push_back(PathingEvent::Dodge(target));
    }
    fn apply_pickup(&mut self, target: D6Target) {
        self.prob *= target.success_prob();
        self.events.push_back(PathingEvent::Pickup(target));
    }

    fn apply_handoff(&mut self, id: PlayerID, target: D6Target) {
        self.prob *= target.success_prob();
        self.events.push_back(PathingEvent::Handoff(id, target));
    }
    fn apply_block(&mut self, vicitm_id: PlayerID, target: NumBlockDices) {
        self.block_dice = Some(target);
        self.events
            .push_back(PathingEvent::Block(vicitm_id, target));
    }
    fn apply_foul(&mut self, vicitm_id: PlayerID, target: Sum2D6Target) {
        self.events.push_back(PathingEvent::Foul(vicitm_id, target));
    }
    fn apply_touchdown(&mut self, id: PlayerID) {
        self.events.push_back(PathingEvent::Touchdown(id));
    }
    fn apply_standup(&mut self) {
        self.events.push_back(PathingEvent::StandUp);
        self.moves_left -= 3;
    }

    fn is_dominant_over(&self, othr: &Node) -> bool {
        assert_eq!(self.position, othr.position);

        if self.prob > othr.prob
            && self.remaining_movement() > othr.remaining_movement()
            && self.block_dice > othr.block_dice
        {
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

        match (self.block_dice, othr.block_dice) {
            (Some(s), Some(o)) if s != o => return s > o,
            (Some(_), None) => panic!("very wrong"), // casual debugging
            (None, Some(_)) => panic!("very wrong"), // casual debugging
            _ => (),
        }

        // best foul target, low is better
        match (self.events.last(), othr.events.last()) {
            (
                Some(PathingEvent::Foul(_, self_target)),
                Some(PathingEvent::Foul(_, othr_target)),
            ) if self_target != othr_target => return self_target < othr_target,
            _ => (),
        }

        if self.remaining_movement() > othr.remaining_movement() {
            return true;
        }
        false
    }
}

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
    player_action: PosAT,
    team: TeamType,
    tzones: FullPitch<i8>,
    ball: PathingBallState,
    start_pos: Position,
    dodge_target: D6Target,
    gfi_target: D6Target,
    pickup_target: D6Target,
    id: PlayerID,
}
impl<'a> GameInfo<'a> {
    fn tackles_zones_at(&self, position: Position) -> i8 {
        self.tzones[position]
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
            .for_each(|position| tzones[position] += 1);
        let ball = match game_state.ball {
            BallState::OnGround(position) => PathingBallState::OnGround(position),
            BallState::Carried(id) if id == player.id => {
                PathingBallState::IsCarrier(game_state.get_endzone_x(player.stats.team))
            }
            _ => PathingBallState::NotRelevant,
        };
        let mut player_action = game_state
            .info
            .player_action_type
            .unwrap_or(PosAT::StartMove);
        if player_action == PosAT::StartHandoff && !matches!(ball, PathingBallState::IsCarrier(_)) {
            // Can't handoff if not ball carrier
            player_action = PosAT::StartMove;
        }

        GameInfo {
            tzones,
            ball,
            start_pos: player.position,
            dodge_target,
            gfi_target,
            pickup_target,
            game_state,
            team: player.stats.team,
            player_action,
            id: player.id,
        }
    }
    fn can_continue_expanding(&self, node: &Rc<Node>) -> bool {
        if node.remaining_movement() == 0
            && !matches!(self.player_action, PosAT::StartFoul | PosAT::StartHandoff)
        {
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
        let new_node: Option<Node> = match self.game_state.get_player_at(to) {
            Some(player)
                if self.player_action == PosAT::StartHandoff
                    && player.stats.team == self.team
                    && player.can_catch() =>
            {
                self.expand_handoff_to(to, player.id, parent_node, prev)
            }
            Some(player)
                if self.player_action == PosAT::StartBlitz
                    && player.stats.team != self.team
                    && parent_node.remaining_movement() > 0
                    && player.status == PlayerStatus::Up =>
            {
                self.expand_block_to(to, player.id, parent_node, prev)
            }
            Some(player)
                if self.player_action == PosAT::StartFoul
                    && player.stats.team != self.team
                    && player.status != PlayerStatus::Up =>
            {
                self.expand_foul_to(to, player.id, parent_node, prev)
            }
            None if parent_node.remaining_movement() > 0 => {
                self.expand_move_to(to, parent_node, prev)
            }
            _ => return NodeType::NoNode,
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

    fn expand_foul_to(
        &self,
        to: Position,
        victim_id: PlayerID,
        parent_node: &Rc<Node>,
        prev: &OptRcNode,
    ) -> Option<Node> {
        let mut next_node = Node::new(Some(parent_node.clone()), to, 0, 0);
        let victim = self.game_state.get_player_unsafe(victim_id);
        let mut target = victim.armor_target();

        target.add_modifer(
            self.game_state
                .get_adj_players(victim.position)
                .filter(|adj_player| {
                    adj_player.id != self.id
                        && adj_player.stats.team == self.team
                        && self.game_state.get_tz_on(adj_player.id) == 0
                })
                .count() as i8,
        );
        target.add_modifer(
            -(self
                .game_state
                .get_adj_players(parent_node.position)
                .filter(|adj_player| {
                    adj_player.stats.team != self.team
                        && adj_player.has_tackle_zone()
                        && self
                            .game_state
                            .get_tz_on_except_from_id(adj_player.id, self.id)
                            == 0
                })
                .count() as i8),
        );

        next_node.apply_foul(victim_id, target);

        if let Some(current_best) = prev {
            if !next_node.is_better_than(current_best) {
                // todo: if there is a current_best, it will always have higher prob right?
                //       that's just how it works with the risky batches. Oh well, optimize later..
                return None;
            }
        }
        Some(next_node)
    }

    fn expand_block_to(
        &self,
        to: Position,
        victim_id: PlayerID,
        parent_node: &Rc<Node>,
        prev: &OptRcNode,
    ) -> Option<Node> {
        let mut next_node = Node::new(Some(parent_node.clone()), to, 0, 0);

        if parent_node.moves_left == 0 {
            next_node.apply_gfi(self.gfi_target);
        }

        next_node.apply_block(
            victim_id,
            self.game_state
                .get_blockdices_from(self.id, parent_node.position, victim_id),
        );
        if let Some(current_best) = prev {
            if !next_node.is_better_than(current_best) {
                // todo: if there is a current_best, it will always have higher prob right?
                //       that's just how it works with the risky batches. Oh well, optimize later..
                return None;
            }
        }
        Some(next_node)
    }

    fn expand_handoff_to(
        &self,
        to: Position,
        id: PlayerID,
        parent_node: &Rc<Node>,
        prev: &OptRcNode,
    ) -> Option<Node> {
        let mut next_node = Node::new(Some(parent_node.clone()), to, 0, 0);

        let target: D6Target = self.game_state.get_catch_target(id).unwrap();
        next_node.apply_handoff(id, target);
        // the Catch procedure will check fo touchdown

        if let Some(current_best) = prev {
            if current_best.is_better_than(&next_node) {
                // todo: if there is a current_best, it will always have higher prob right?
                //       that's just how it works with the risky batches. Oh well, optimize later..
                return None;
            }
        }
        Some(next_node)
    }

    fn expand_move_to(
        &self,
        to: Position,
        parent_node: &Rc<Node>,
        prev: &OptRcNode,
    ) -> Option<Node> {
        let gfi = parent_node.moves_left == 0;

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

        let mut next_node = Node::new(Some(parent_node.clone()), to, moves_left, gfis_left);

        if gfi {
            next_node.apply_gfi(self.gfi_target);
        }
        if self.tackles_zones_at(parent_node.position) > 0 {
            next_node.apply_dodge(*self.dodge_target.clone().add_modifer(-self.tzones[to]));
        }
        match self.ball {
            PathingBallState::OnGround(ball_pos) if ball_pos == to => {
                // touchdown by pickup is handled by the pickup procedure
                next_node.apply_pickup(*self.pickup_target.clone().add_modifer(-self.tzones[to]));
            }
            PathingBallState::IsCarrier(endzone_x) if to.x == endzone_x => {
                next_node.apply_touchdown(self.id);
            }
            _ => (),
        }

        Some(next_node)
    }
}

impl<'a> PathFinder<'a> {
    fn new(info: GameInfo) -> PathFinder {
        PathFinder {
            nodes: Default::default(),
            locked_nodes: Default::default(),
            open_set: Default::default(),
            risky_sets: Default::default(),
            info,
        }
    }
    pub fn player_paths(game_state: &GameState, id: PlayerID) -> Result<FullPitch<OptRcNode>> {
        let player = game_state.get_player_unsafe(id);
        let info = GameInfo::new(game_state, player);
        let mut root_node = Node::new(
            None,
            info.start_pos,
            player.moves_left(),
            player.gfis_left(),
        );
        if player.status != PlayerStatus::Up {
            assert!(player.moves_left() == player.stats.ma);
            root_node.apply_standup();
        }

        let root_node = Rc::new(root_node);

        if !info.can_continue_expanding(&root_node) {
            return Ok(Default::default());
        }

        let mut pf = PathFinder::new(info);

        pf.open_set.push(root_node);

        loop {
            //expansion
            while let Some(node) = pf.open_set.pop() {
                pf.expand_node(node);
            }

            //clear pf.nodes
            for (node, locked) in zip(pf.nodes.iter_mut(), pf.locked_nodes.iter_mut()) {
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

        Ok(pf.locked_nodes)
    }

    fn prepare_nodes(&mut self, new_nodes: Vec<Rc<Node>>) {
        for new_node in new_nodes {
            if self.locked_nodes[new_node.position]
                .as_ref()
                .map(|locked| locked.is_dominant_over(&new_node))
                .unwrap_or(false)
            {
                continue;
            }

            let best_in_batch = &mut self.nodes[new_node.position];
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
                    self.info.tackles_zones_at(parent.position) > 0,
                )
            });

        Direction::all_directions_iter()
            .map(|direction| node.position + *direction)
            .filter(|to_pos| !to_pos.is_out())
            .filter(|to_pos| {
                parent_pos_and_in_tz
                    .map(|(parent_pos, parent_in_tz)| {
                        parent_pos.distance_to(to_pos) == 2
                            || (parent_in_tz && 0 < self.info.tzones[*to_pos])
                    })
                    .unwrap_or(true)
            })
            .map(|to_pos| {
                self.info.expand_to(
                    to_pos,
                    &node,
                    &mut self.nodes[to_pos],
                    &self.locked_nodes[to_pos],
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
    // pub fn is_empty(&self) -> bool {
    //     self.set.is_empty()
    // }
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

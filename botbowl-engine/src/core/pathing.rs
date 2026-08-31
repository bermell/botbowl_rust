use std::fmt::Debug;
use std::{collections::HashMap, hash, iter::zip, sync::Arc};

use crate::core::model;
use model::*;
use serde::{Deserialize, Serialize};

use super::dices::{D6Target, RollTarget, Sum2D6Target};
use super::gamestate::GameState;
use super::table::{NumBlockDices, PosAT};

type OptRcNode = Option<Arc<Node>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathingEvent {
    Dodge(D6Target),
    GFI(D6Target),
    Pickup(D6Target),
    Block(PlayerID, NumBlockDices),
    Handoff(PlayerID, D6Target),
    Pass { to: Position, pass: D6Target, modifer: i8 },
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
        PathingEvent::Pass { .. } => true,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    // Invariant: Some entries form a contiguous prefix of `data`; everything
    // after the first None is also None. Every public mutator preserves this,
    // so reads can short-circuit instead of scanning the whole array.
    pub fn len(&self) -> usize {
        self.data.iter().take_while(|v| v.is_some()).count()
    }
    pub fn push_back(&mut self, val: T) {
        self.add(val)
    }
    pub fn add(&mut self, val: T) {
        let next = self.data.iter_mut().find(|v| v.is_none()).expect("FixedQueue full");
        *next = Some(val);
    }
    pub fn pop(&mut self) -> Option<T> {
        let first = self.data[0].take()?;
        // shift the rest left so the invariant holds
        for i in 0..5 {
            self.data[i] = self.data[i + 1].take();
        }
        Some(first)
    }
    pub fn is_empty(&self) -> bool {
        self.data[0].is_none()
    }
    pub fn is_full(&self) -> bool {
        self.data[5].is_some()
    }
    pub fn last(&self) -> Option<&T> {
        // Under the contiguity invariant, the last Some is the rightmost
        // non-None entry. Reverse iter + find_map stops as soon as we see one.
        self.data.iter().rev().find_map(|v| v.as_ref())
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // take_while terminates at the first None, then `unwrap` is safe by invariant.
        self.data
            .iter()
            .take_while(|v| v.is_some())
            .map(|v| v.as_ref().unwrap())
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionOrEvent {
    Position(Position),
    Event(PathingEvent),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeIterator {
    stack: Vec<PositionOrEvent>,
}

impl NodeIterator {
    fn new(node: &Arc<Node>) -> Self {
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
    type Item = PositionOrEvent;

    fn next(&mut self) -> Option<PositionOrEvent> {
        self.stack.pop()
    }
}

pub trait CustomIntoIter {
    fn iter(&self) -> NodeIterator;
}
impl CustomIntoIter for Arc<Node> {
    fn iter(&self) -> NodeIterator {
        NodeIterator::new(self)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    parent: Option<Arc<Node>>,
    pub position: Position,
    moves_left: u8,
    gfis_left: u8,
    block_dice: Option<NumBlockDices>,
    // foul_roll, handoff_roll, block_dice
    //euclidiean_distance: f32,
    pub prob: f32,
    events: FixedQueue<PathingEvent>,
    // Cumulative manhattan distance from the root; used as a tie-breaker in
    // `is_better_than`. Pre-computed so we don't walk the parent chain there.
    // `#[serde(default)]` keeps replay files written before this field was
    // added deserializable — they'll come back with 0, which only affects
    // tie-break ordering and not correctness.
    #[serde(default)]
    cum_dist: u16,
    // Cached result of `get_action_type` — `events.last()` plus the block_dice
    // override. Updated by every `apply_*` so reads in `can_continue_expanding`
    // skip the FixedQueue walk.
    #[serde(default = "default_action_type")]
    action_type: PosAT,
}

fn default_action_type() -> PosAT {
    PosAT::Move
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.parent == other.parent
    }
}
impl Eq for Node {}
impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("position", &self.position)
            .field("prob", &self.prob);
        if let Some(parent) = &self.parent {
            f.debug_struct("Node").field("parent_pos", &parent.position);
        }

        f.debug_struct("Node").finish()
    }
}
impl Node {
    pub fn get_block_dice(&self) -> Option<NumBlockDices> {
        self.block_dice
    }
    fn add_iter_items(&self, items: &mut Vec<PositionOrEvent>) {
        for event in self.events.iter_rev() {
            items.push(PositionOrEvent::Event(event.clone()));
        }
        if self.move_to_position() {
            items.push(PositionOrEvent::Position(self.position));
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
                PathingEvent::Pass { .. } => false,
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
            cum_dist: 0,
            action_type: PosAT::Block,
        }
    }

    pub fn get_action_type(&self) -> PosAT {
        self.action_type
    }

    fn new(parent: OptRcNode, position: Position, moves_left: u8, gfis_left: u8) -> Node {
        let (prob, cum_dist) = match parent.as_ref() {
            Some(p) => (p.prob, p.cum_dist + p.position.distance_to(&position) as u16),
            None => (1.0, 0),
        };
        Node {
            parent,
            position,
            moves_left,
            gfis_left,
            block_dice: None,
            prob,
            events: Default::default(),
            cum_dist,
            action_type: PosAT::Move,
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
        // TODO: concider catch (remember the intercep too!)
        self.prob *= target.success_prob();
        self.events.push_back(PathingEvent::Handoff(id, target));
        self.action_type = PosAT::Handoff;
    }
    fn apply_pass(
        &mut self,
        to: Position,
        catch_target: D6Target,
        pass_target: D6Target,
        intercept: Option<D6Target>,
        pass_modifer: i8,
    ) {
        // TODO: concider catch and pass skill (remember the intercep too!)
        self.prob *= catch_target.success_prob();
        self.prob *= pass_target.success_prob();
        if let Some(intercept_target) = intercept {
            // TODO: find the best intercept here..
            self.prob *= 1.0 - intercept_target.success_prob();
        }
        self.events.push_back(PathingEvent::Pass {
            to,
            pass: pass_target,
            modifer: pass_modifer,
        });
        self.action_type = PosAT::Pass;
    }
    fn apply_block(&mut self, vicitm_id: PlayerID, target: NumBlockDices) {
        self.block_dice = Some(target);
        self.events.push_back(PathingEvent::Block(vicitm_id, target));
        self.action_type = PosAT::Block;
    }
    fn apply_foul(&mut self, vicitm_id: PlayerID, target: Sum2D6Target) {
        self.events.push_back(PathingEvent::Foul(vicitm_id, target));
        self.action_type = PosAT::Foul;
    }
    fn apply_touchdown(&mut self, id: PlayerID) {
        self.events.push_back(PathingEvent::Touchdown(id));
        // Touchdown does not change action_type — pickup/move still drives expansion.
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
            (Some(PathingEvent::Foul(_, self_target)), Some(PathingEvent::Foul(_, othr_target)))
                if self_target != othr_target =>
            {
                return self_target < othr_target
            }
            _ => (),
        }

        if self.remaining_movement() > othr.remaining_movement() {
            return true;
        }
        if self.manhattan_distance() < othr.manhattan_distance() {
            return true;
        }
        false
    }

    fn manhattan_distance(&self) -> u16 {
        // Pre-computed during construction; avoids walking the parent chain in
        // the `is_better_than` tie-breaker.
        self.cum_dist
    }
}

pub struct PathFinder<'a> {
    nodes: FullPitch<OptRcNode>,
    locked_nodes: FullPitch<OptRcNode>,
    open_set: Vec<Arc<Node>>,
    risky_sets: RiskySet,
    info: GameInfo<'a>,
}

enum NodeType {
    Risky(Arc<Node>),
    ContinueExpanding(Arc<Node>),
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
    teammate_catch_mod: FullPitch<Option<D6Target>>,
    ball: PathingBallState,
    start_pos: Position,
    /// Expansion order for [`PathFinder::expand_node`], oriented by the
    /// moving team's attacking direction. Insertion order decides which of
    /// two otherwise-identical routes to a square survives
    /// (`Node::is_better_than` returns `false` on a full tie), so a fixed
    /// order would make the route choice a function of absolute x — and
    /// therefore treat the two teams' mirror-image situations differently.
    /// See `Direction::all_directions_toward`.
    dir_order: &'static [Direction; 8],
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
        let mut player_action = game_state.info.player_action_type.unwrap_or(PosAT::StartMove);
        let mut catch_mods: FullPitch<Option<D6Target>> = Default::default();
        if player_action == PosAT::StartHandoff || player_action == PosAT::StartPass {
            if matches!(ball, PathingBallState::IsCarrier(_)) {
                // player is ball carrier and can handoff or pass
                game_state
                    .get_players_on_pitch()
                    .filter(|p| p.stats.team == team)
                    .filter(|p| p.can_catch())
                    .for_each(|p| catch_mods[p.position] = Some(game_state.get_catch_target(p.id).unwrap()));
                catch_mods[player.position] = None; // can't handoff or pass to self
            } else {
                // If not ball carrier we don't care about handoff or pass
                player_action = PosAT::StartMove;
            }
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
            dir_order: Direction::all_directions_toward(
                game_state.get_endzone_x(player.stats.team)
                    - game_state.get_endzone_x(other_team(player.stats.team)),
            ),
            player_action,
            id: player.id,
            teammate_catch_mod: catch_mods,
        }
    }
    fn can_continue_expanding(&self, node: &Arc<Node>) -> bool {
        if node.remaining_movement() == 0 {
            let is_foul = self.player_action == PosAT::StartFoul;
            let can_do_ball_action = matches!(self.ball, PathingBallState::IsCarrier(_))
                && matches!(self.player_action, PosAT::StartHandoff | PosAT::StartPass);

            if is_foul || can_do_ball_action {
                // do nothing, continue be checks below
            } else {
                return false;
            }
        }
        match node.get_action_type() {
            PosAT::Handoff => return false,
            PosAT::Pass => return false,
            PosAT::Foul => return false,
            PosAT::Block => return false,
            PosAT::Move => (),
            _ => panic!("very wrong!"),
        }

        match self.ball {
            PathingBallState::IsCarrier(endzone_x) if endzone_x == node.position.x => false,
            PathingBallState::OnGround(ball_pos) if ball_pos == node.position => false,
            _ => true,
        }
    }

    fn expand_to(&self, to: Position, parent_node: &Arc<Node>, prev: &mut OptRcNode, best: &OptRcNode) -> NodeType {
        debug_assert!(self.can_continue_expanding(parent_node));

        // expand to move_node, block_node, handoff_mode
        let new_node: Option<Node> = match self.game_state.get_player_at(to) {
            Some(player) if self.teammate_catch_mod[to].is_some() => {
                // handoff or pass
                match self.player_action {
                    PosAT::StartPass => self.expand_pass_to(to, player.id, parent_node, prev),
                    PosAT::StartHandoff => self.expand_handoff_to(to, player.id, parent_node, prev),
                    _ => unreachable!("very wrong!"),
                }
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
            None if parent_node.remaining_movement() > 0 => self.expand_move_to(to, parent_node, prev),
            _ => return NodeType::NoNode,
        };

        let new_node = match new_node {
            Some(node) => node,
            None => return NodeType::NoNode,
        };

        // Dominance check before Arc allocation: every expansion to a position
        // already locked from a higher-prob batch gets discarded here (since
        // `is_dominant_over` is the typical outcome with matching block_dice),
        // and we'd otherwise be heap-allocating just to throw it away.
        if let Some(best_before) = &best {
            debug_assert!(best_before.prob > new_node.prob);
            if !best_before.is_dominant_over(&new_node) {
                return NodeType::NoNode;
            }
        }

        let new_node: Arc<Node> = Arc::new(new_node);

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
        parent_node: &Arc<Node>,
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
                        && self.game_state.get_tz_on_except_from_id(adj_player.id, self.id) == 0
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
        parent_node: &Arc<Node>,
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

    fn expand_handoff_to(&self, to: Position, id: PlayerID, parent_node: &Arc<Node>, prev: &OptRcNode) -> Option<Node> {
        let mut next_node = Node::new(Some(parent_node.clone()), to, 0, 0);

        next_node.apply_handoff(id, self.teammate_catch_mod[to].unwrap());
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

    fn expand_pass_to(&self, to: Position, id: PlayerID, parent_node: &Arc<Node>, prev: &OptRcNode) -> Option<Node> {
        let mut next_node = Node::new(Some(parent_node.clone()), to, 0, 0);
        if parent_node.position == to {
            println!(
                "very wrong.. {}, {:?}",
                parent_node.position,
                parent_node.get_action_type()
            );
            panic!(
                "very wrong.. {}, {:?}",
                parent_node.position,
                parent_node.get_action_type()
            );
        }

        let pass_target = self.game_state.get_pass_target(id, parent_node.position, to)?;

        let catch_target = self.teammate_catch_mod[to].unwrap();
        let best_intercept = self
            .game_state
            .get_intercepters(other_team(self.team), parent_node.position, to)
            .iter()
            .map(|(_, target)| *target)
            .max_by(|target_a, target_b| {
                target_a
                    .success_prob()
                    .partial_cmp(&target_b.success_prob())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let modifier = self.game_state.get_pass_modifier(id, parent_node.position, to).unwrap();
        next_node.apply_pass(to, catch_target, pass_target, best_intercept, modifier);
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
    fn expand_move_to(&self, to: Position, parent_node: &Arc<Node>, prev: &OptRcNode) -> Option<Node> {
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
        let mut out: FullPitch<OptRcNode> = Default::default();
        Self::fill_player_paths(game_state, id, &mut out)?;
        Ok(out)
    }

    /// Fill `out` with reachable paths for `id`. Clears `out` on entry, so
    /// callers can pass a buffer that was filled by a previous call. This is
    /// the variant used by `MoveAction`/`BlockAction` to write directly into
    /// `GameState::path_buffer` and skip a per-frame 4KB allocation.
    pub fn fill_player_paths(game_state: &GameState, id: PlayerID, out: &mut FullPitch<OptRcNode>) -> Result<()> {
        // Release any stale Arc payload from a previous fill — caller may
        // be reusing the same buffer.
        for slot in out.iter_mut() {
            *slot = None;
        }

        let player = game_state.get_player_unsafe(id);
        // Stunned players cannot move or stand up this turn — they go Prone at the next
        // TurnStunned. Returning empty paths matches that, and prevents callers from
        // spawning a StandUp procedure that would trip its own assertion.
        if player.status == PlayerStatus::Stunned {
            return Ok(());
        }
        let info = GameInfo::new(game_state, player);
        let mut root_node = Node::new(None, info.start_pos, player.moves_left(), player.gfis_left());
        if player.status != PlayerStatus::Up {
            assert!(player.moves_left() == player.stats.ma);
            debug_assert!(matches!(player.status, PlayerStatus::Down));
            root_node.apply_standup();
        }

        let root_node = Arc::new(root_node);

        if !info.can_continue_expanding(&root_node) {
            return Ok(());
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

        // Move the search results into the caller's buffer. `pf.locked_nodes`
        // was None-everywhere at the start (Default in PathFinder::new) so
        // the swap leaves us with an empty FullPitch to drop with pf.
        std::mem::swap(&mut pf.locked_nodes, out);
        Ok(())
    }

    /// Returns the best path from `id`'s current position to `target`, or None if no path exists.
    /// Safe to call for an unused fresh-turn player on the moving team — moves_left, gfis_left,
    /// and `player_action_type` default to sensible values (MA, 2, StartMove respectively).
    pub fn safest_path_to(game_state: &GameState, id: PlayerID, target: Position) -> Result<Option<Arc<Node>>> {
        let paths = PathFinder::player_paths(game_state, id)?;
        Ok(paths[target].clone())
    }

    /// Returns the highest-probability path that ends on the opponent endzone column.
    /// None if no such path exists.
    pub fn safest_path_to_endzone(game_state: &GameState, id: PlayerID) -> Result<Option<Arc<Node>>> {
        let paths = PathFinder::player_paths(game_state, id)?;
        let team = game_state.get_player_unsafe(id).stats.team;
        let endzone_x = game_state.get_endzone_x(team);
        let best = paths
            .iter_position()
            .filter(|(pos, _)| pos.x == endzone_x)
            .filter_map(|(_, n)| n.clone())
            .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal));
        Ok(best)
    }
}

impl<'a> PathFinder<'a> {
    fn prepare_nodes(&mut self, new_nodes: Vec<Arc<Node>>) {
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

    fn expand_node(&mut self, node: Arc<Node>) {
        debug_assert!(self.info.can_continue_expanding(&node));

        let parent_pos_and_in_tz: Option<(Position, bool)> = node
            .parent
            .as_ref()
            .filter(|parent| parent.position != node.position)
            .map(|parent| (parent.position, self.info.tackles_zones_at(parent.position) > 0));

        //handle moving
        self.info
            .dir_order
            .iter()
            .map(|direction| node.position + *direction)
            .filter(|to_pos| !self.info.game_state.is_out(*to_pos))
            .filter(|to_pos| {
                parent_pos_and_in_tz
                    .map(|(parent_pos, parent_in_tz)| {
                        parent_pos.distance_to(to_pos) == 2 || (parent_in_tz && 0 < self.info.tzones[*to_pos])
                    })
                    .unwrap_or(true)
            })
            .map(|to_pos| {
                self.info
                    .expand_to(to_pos, &node, &mut self.nodes[to_pos], &self.locked_nodes[to_pos])
            })
            .for_each(|node_type| match node_type {
                NodeType::Risky(node) => self.risky_sets.insert_node(node),
                NodeType::ContinueExpanding(node) => {
                    debug_assert!(self.info.can_continue_expanding(&node));
                    self.open_set.push(node);
                }
                NodeType::NoNode => (),
            });

        //handle passing
        if self.info.player_action == PosAT::StartPass && matches!(self.info.ball, PathingBallState::IsCarrier(_)) {
            self.info
                .game_state
                .get_players_on_pitch()
                .filter(|player| player.stats.team == self.info.team)
                .filter(|player| player.id != self.info.id)
                .filter(|player| player.can_catch())
                // TODO: check withing passing range
                .map(|player| player.position)
                .map(|to_pos| {
                    self.info
                        .expand_to(to_pos, &node, &mut self.nodes[to_pos], &self.locked_nodes[to_pos])
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
}

#[derive(Default)]
struct RiskySet {
    set: HashMap<HashableFloat, Vec<Arc<Node>>>,
}
impl RiskySet {
    pub fn insert_node(&mut self, node: Arc<Node>) {
        assert!(0_f32 < node.prob && node.prob <= 1.0_f32);
        let prob = HashableFloat(node.prob);
        self.set.entry(prob).or_default().push(node);
    }
    pub fn get_next_batch(&mut self) -> Option<Vec<Arc<Node>>> {
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
        f.debug_struct("RiskySet").field("len", &self.set.len()).finish()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gamestate::{BuilderState, GameStateBuilder};
    use crate::core::model::TeamType;
    use crate::core::table::SimpleAT;

    #[test]
    fn safest_path_to_endzone_clear_lane() {
        // Home ball carrier near its endzone (x=1) with no opponents in the way.
        let start = Position::new((3, crate::core::model::HEIGHT_ / 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(start)
            .add_ball((start.x, start.y))
            .set_state(BuilderState::Turn { turn: 1 })
            .build();
        // Make sure it's the home team's turn so a Home player's MoveAction is well-formed.
        if state.info.team_turn != TeamType::Home {
            state.step_simple(SimpleAT::EndTurn);
        }
        let id = state.get_player_id_at(start).unwrap();
        let path = PathFinder::safest_path_to_endzone(&state, id)
            .unwrap()
            .expect("expected a path to the endzone");
        assert!((path.prob - 1.0).abs() < 1e-6);
        assert_eq!(path.position.x, state.get_endzone_x(TeamType::Home));
    }

    #[test]
    fn safest_path_to_non_active_teammate() {
        // Two home players, no opponents. Path from teammate (not yet activated) to a target.
        let cx = crate::core::model::WIDTH_ / 2;
        let cy = crate::core::model::HEIGHT_ / 2;
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((cx - 1, cy)))
            .add_home_player(Position::new((cx + 1, cy)))
            .set_state(BuilderState::Turn { turn: 1 })
            .build();
        if state.info.team_turn != TeamType::Home {
            state.step_simple(SimpleAT::EndTurn);
        }
        let id = state.get_player_id_at(Position::new((cx - 1, cy))).unwrap();
        // No START_xxx action taken yet, so this player is not active.
        assert!(state.info.active_player.is_none());
        let target = Position::new((cx, cy));
        let path = PathFinder::safest_path_to(&state, id, target)
            .unwrap()
            .expect("expected a path to adjacent empty square");
        assert!((path.prob - 1.0).abs() < 1e-6);
        assert_eq!(path.position, target);
    }

    #[test]
    fn player_paths_returns_empty_for_stunned_player() {
        // Stunned players cannot act this turn (TurnStunned converts them to Down
        // at the start of *their next* team turn, not before). Pathing must reflect
        // that — otherwise we generate standup paths whose execution trips
        // StandUp::step's debug_assert.
        let start = Position::new((crate::core::model::WIDTH_ / 2, crate::core::model::HEIGHT_ / 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(start)
            .set_state(BuilderState::Turn { turn: 1 })
            .build();
        if state.info.team_turn != TeamType::Home {
            state.step_simple(SimpleAT::EndTurn);
        }
        let id = state.get_player_id_at(start).unwrap();
        state.get_mut_player_unsafe(id).status = PlayerStatus::Stunned;

        let paths = PathFinder::player_paths(&state, id).unwrap();
        let any_path = paths.iter().any(|n| n.is_some());
        assert!(!any_path, "expected no paths for a stunned player, found at least one");
    }

    #[test]
    fn safest_path_for_downed_teammate_includes_standup() {
        let start = Position::new((crate::core::model::WIDTH_ / 2, crate::core::model::HEIGHT_ / 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(start)
            .set_state(BuilderState::Turn { turn: 1 })
            .build();
        if state.info.team_turn != TeamType::Home {
            state.step_simple(SimpleAT::EndTurn);
        }
        let id = state.get_player_id_at(start).unwrap();
        state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
        // Adjacent square, reachable after 3-move standup + 1 move.
        let target = Position::new((start.x + 1, start.y));
        let path = PathFinder::safest_path_to(&state, id, target)
            .unwrap()
            .expect("expected a path that includes standup");
        // The first event in the chain (root) should be StandUp.
        let items: Vec<PositionOrEvent> = path.iter().collect();
        assert!(
            items
                .iter()
                .any(|item| matches!(item, PositionOrEvent::Event(PathingEvent::StandUp))),
            "path must contain a StandUp event, got {:?}",
            items
        );
    }
}

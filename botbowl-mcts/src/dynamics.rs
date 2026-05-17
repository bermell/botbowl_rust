//! `GameDynamics` implementation that plugs Blood Bowl into recon_mcts,
//! plus the MctsBot adapter that exposes the searcher as a `Bot`.
//!
//! Design mirrors the 2048 reference (`recon_mcts/tests/nim/test_mcts_2048.rs`):
//! - `Player` discriminator carries the kind of node (`Home`/`Away` action
//!   nodes vs. `Chance` outcome nodes).
//! - `Score` is a small struct with atomic visit counter, integer score,
//!   and the node-kind discriminator that `backprop_scores` switches on.
//! - `available_actions` returns engine actions for player nodes,
//!   `ChanceOutcome` choices for chance nodes (when `state.pending_roll`
//!   is `Some`).
//! - `apply_action` calls `state.micro_step(Some(a))` for player actions,
//!   queues a dice fix and calls `state.micro_step(None)` for chance
//!   actions.

use std::ops::Deref;
use std::sync::atomic::{AtomicU32, Ordering};

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType};
use recon_mcts::{GameDynamics, HashOnly, SearchTree, SelectNodeState, Tree};

use crate::action::{BbAction, BbPlayer};
use crate::roll_outcomes;
use crate::score::leaf_score;

const UCT_C: f32 = 1.4;

#[derive(Debug)]
pub struct BbScore {
    pub visits: AtomicU32,
    pub score: i64,
    pub node_kind: BbPlayer,
}

impl Clone for BbScore {
    fn clone(&self) -> Self {
        BbScore {
            visits: AtomicU32::new(self.visits.load(Ordering::Relaxed)),
            score: self.score,
            node_kind: self.node_kind,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BloodBowlDynamics;

/// Inspect a state and decide which "player" owns it from MCTS's
/// perspective. Chance nodes are detected by a pending roll; otherwise
/// the engine's `available_actions.team` tells us whose move it is.
fn player_for_state(state: &GameState) -> BbPlayer {
    if state.pending_roll.is_some() {
        return BbPlayer::Chance;
    }
    match state.available_actions.team {
        Some(TeamType::Home) => BbPlayer::Home,
        Some(TeamType::Away) => BbPlayer::Away,
        // Engine has no decision to expose — treat as a chance-like
        // "advance" node so callers know there's nothing to choose.
        None => BbPlayer::Chance,
    }
}

impl GameDynamics for BloodBowlDynamics {
    type Player = BbPlayer;
    type State = GameState;
    type Action = BbAction;
    type Score = BbScore;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(
        &self,
        _player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter> {
        if state.info.game_over {
            return None;
        }

        // Chance node: enumerate roll outcomes.
        if state.pending_roll.is_some() {
            let req = state.pending_roll.as_ref().unwrap();
            let outcomes = roll_outcomes::enumerate(req);
            return Some(
                outcomes
                    .into_iter()
                    .map(|a| (BbPlayer::Chance, a))
                    .collect(),
            );
        }

        // Player node: copy engine's available actions.
        let team = state.available_actions.team?;
        let mcts_player = match team {
            TeamType::Home => BbPlayer::Home,
            TeamType::Away => BbPlayer::Away,
        };
        let actions: Vec<(BbPlayer, BbAction)> = state
            .available_actions
            .get_all()
            .into_iter()
            .map(|a| (mcts_player, BbAction::Player(a)))
            .collect();
        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        let mut new_state = state;
        match action {
            BbAction::Player(engine_action) => {
                if new_state
                    .micro_step(Some(*engine_action))
                    .is_err()
                {
                    return None;
                }
            }
            BbAction::Chance { outcome, .. } => {
                if new_state.pending_roll.is_none() {
                    // Shouldn't happen if available_actions was honest;
                    // returning None lets the library treat the edge as
                    // invalid instead of panicking.
                    return None;
                }
                roll_outcomes::fix_for_outcome(&mut new_state, *outcome);
                if new_state.micro_step(None).is_err() {
                    return None;
                }
            }
        }
        Some(new_state)
    }

    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        parent_node_state: &Self::State,
        _purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        let parent_visits = parent_score
            .map(|s| s.visits.load(Ordering::Relaxed))
            .unwrap_or(1) as f32;

        // Chance node: pick the child with the fewest visits.
        if parent_node_state.pending_roll.is_some() {
            let pick = scores_and_actions
                .clone()
                .into_iter()
                .min_by(|a, b| {
                    let va = a
                        .0
                        .as_ref()
                        .as_ref()
                        .map(|s| s.visits.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    let vb = b
                        .0
                        .as_ref()
                        .as_ref()
                        .map(|s| s.visits.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    va.cmp(&vb)
                })
                .expect("chance node must have at least one outcome");
            if let Some(s) = pick.0.as_ref() {
                s.visits.fetch_add(1, Ordering::Relaxed);
            }
            return pick.1.deref().clone();
        }

        // Player node: standard UCT.
        let pick = scores_and_actions
            .clone()
            .into_iter()
            .max_by(|a, b| {
                let ua = uct_value(a.0.as_ref(), parent_visits);
                let ub = uct_value(b.0.as_ref(), parent_visits);
                ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("player node must have at least one action");
        if let Some(s) = pick.0.as_ref() {
            s.visits.fetch_add(1, Ordering::Relaxed);
        }
        pick.1.deref().clone()
    }

    fn backprop_scores<II, Q, A>(
        &self,
        _player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        // Chance node: probability-weighted average over visited children.
        if score_current
            .map(|s| s.node_kind == BbPlayer::Chance)
            .unwrap_or(false)
        {
            let mut total_visits: u32 = 0;
            let mut weighted_sum: f64 = 0.0;
            let mut total_prob: f64 = 0.0;
            for (q, a) in child_scores_and_actions.into_iter() {
                let v = q.visits.load(Ordering::Relaxed);
                if v == 0 {
                    continue;
                }
                let prob = a.prob_f32().unwrap_or(0.0) as f64;
                weighted_sum += prob * q.score as f64;
                total_prob += prob;
                total_visits += v;
            }
            if total_visits == 0 {
                return None;
            }
            // Normalise in case some outcomes haven't been visited yet.
            let avg = if total_prob > 0.0 {
                weighted_sum / total_prob
            } else {
                0.0
            };
            return Some(BbScore {
                visits: AtomicU32::new(total_visits),
                score: avg as i64,
                node_kind: BbPlayer::Chance,
            });
        }

        // Player node: max over children. For the MVP we treat Away the
        // same as Home (single-player optimisation view from Home's
        // perspective) — opponent modelling lives in step-N follow-ups.
        let mut best: Option<(u32, i64)> = None;
        for (q, _) in child_scores_and_actions.into_iter() {
            let s = (q.visits.load(Ordering::Relaxed), q.score);
            best = match best {
                None => Some(s),
                Some(b) if s.1 > b.1 => Some(s),
                Some(b) => Some(b),
            };
        }
        best.map(|(visits, score)| BbScore {
            visits: AtomicU32::new(visits),
            score,
            node_kind: BbPlayer::Home,
        })
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        Some(BbScore {
            visits: AtomicU32::new(1),
            score: leaf_score(state),
            node_kind: player_for_state(state),
        })
    }
}

fn uct_value(score: Option<&BbScore>, parent_visits: f32) -> f32 {
    match score {
        None => f32::INFINITY, // unexplored — always prefer
        Some(s) => {
            let v = s.visits.load(Ordering::Relaxed) as f32;
            if v == 0.0 {
                f32::INFINITY
            } else {
                s.score as f32 + UCT_C * (parent_visits.max(1.0).ln() / v).sqrt()
            }
        }
    }
}

/// `Bot` adapter that drives a fresh MCTS search per call.
pub struct MctsBot {
    pub iterations_per_move: usize,
}

impl MctsBot {
    pub fn new(iterations_per_move: usize) -> Self {
        Self { iterations_per_move }
    }
}

impl Bot for MctsBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
        // Clone the state and turn on roll-by-roll stepping for the
        // search. The bot's caller still owns the live state.
        let mut root_state = state.clone();
        root_state.expose_rolls = true;
        // The engine may have queued fixes for the caller's own use —
        // don't let them leak into MCTS rollouts.
        root_state.fixes = Default::default();
        root_state.rng_enabled = true;

        let root_player = player_for_state(&root_state);
        let tree = Tree::new(BloodBowlDynamics, HashOnly, root_player, root_state);

        for _ in 0..self.iterations_per_move {
            tree.step();
        }

        let move_info = tree
            .get_next_move_info()
            .expect("MCTS tree has no move info at root");
        let best = move_info
            .iter()
            .max_by_key(|(_, info)| {
                info.score
                    .as_ref()
                    .map(|s| s.visits.load(Ordering::Relaxed))
                    .unwrap_or(0)
            })
            .expect("root must offer at least one action");
        match &best.0 {
            BbAction::Player(a) => *a,
            BbAction::Chance { .. } => {
                panic!("root selected a chance action — root must be a player turn");
            }
        }
    }
}

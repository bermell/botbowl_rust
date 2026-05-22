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
use crate::priors::prior_for;
use crate::pruning::should_prune;
use crate::roll_outcomes;
use crate::score::leaf_score;

/// PUCT exploration constant. Sized so that the `c · P · √N(parent) /
/// (1 + N(a))` term is comparable to leaf-score magnitudes (game score
/// ±1000, ball control ±50, carrier-distance ±26 — see `score.rs`).
/// 10.0 keeps unexplored high-prior children competitive with explored
/// children of moderate `Q` for at least the first few hundred parent
/// visits — verified empirically against the existing UCT baseline test.
const PUCT_C: f32 = 10.0;

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
            .filter(|a| !should_prune(state, a))
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
        // NOTE: we deliberately do *not* fast-forward past intermediate
        // engine states here. The engine resolves "Move(target)" one
        // square per `micro_step`, so the returned state is often
        // mid-path with empty `available_actions` and no
        // `pending_roll`. MCTS treats those as terminal
        // (`Children::None`), which means lectures requiring the
        // pickup chance node to be reached (`Get the Ball *`) score
        // 0 on the pickup move and the search prefers an inferior
        // adjacent-to-ball move. v2 will add fast-forwarding plus
        // broader `roll_outcomes` coverage (Deviate, Block dice, etc.)
        // so chance nodes downstream of pickup are reachable.
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

        // Player node: PUCT with domain priors. We compute priors lazily
        // here rather than caching them on `BbScore` — the cost is one
        // `prior_for` call per child per descent, which is cheap (enum
        // match + a few position comparisons) and avoids threading
        // parent-state into `score_leaf`.
        let pick = scores_and_actions
            .clone()
            .into_iter()
            .max_by(|a, b| {
                let pa = prior_for(parent_node_state, &a.1);
                let pb = prior_for(parent_node_state, &b.1);
                let va = puct_value(a.0.as_ref(), parent_visits, pa);
                let vb = puct_value(b.0.as_ref(), parent_visits, pb);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
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
        // For chance states (engine has a `pending_roll`) the bare
        // `leaf_score` returns the *pre-roll* board, which is usually
        // misleading — e.g. a Move that lands on a free ball produces a
        // state where the player is on top of the ball but the ball is
        // still `OnGround`, so `ball_control_value` sees no adjacency
        // and returns 0. That makes the pickup look worse than just
        // standing next to the ball, even though the actual outcome
        // (after the pickup roll) is +500. To give MCTS a meaningful Q
        // before it descends into the chance node, score chance states
        // as the probability-weighted leaf score over their immediate
        // outcomes. The chance node's own `backprop_scores` will refine
        // this once both outcomes have been visited.
        let score = if state.pending_roll.is_some() {
            expected_leaf_score(state).unwrap_or_else(|| leaf_score(state))
        } else {
            leaf_score(state)
        };
        Some(BbScore {
            visits: AtomicU32::new(1),
            score,
            node_kind: player_for_state(state),
        })
    }
}

/// Score a chance state as the probability-weighted expected leaf score
/// over its immediate outcomes. Returns `None` if any outcome can't be
/// applied (e.g. roll type isn't yet supported by `roll_outcomes`).
///
/// This is a one-level lookahead, not a full rollout — if an outcome
/// state is *also* a chance state, we just score it with the bare
/// `leaf_score` (no further unrolling). The intent is to give MCTS a
/// reasonable Q for chance leaves so it doesn't ignore the pickup move
/// in favour of an inferior adjacent-to-ball move that scores higher on
/// the bare board.
fn expected_leaf_score(state: &GameState) -> Option<i64> {
    let req = state.pending_roll.as_ref()?;
    if !crate::action::is_supported(req) {
        return None;
    }
    let outcomes = roll_outcomes::enumerate(req);
    let mut weighted_sum: f64 = 0.0;
    let mut total_prob: f64 = 0.0;
    for outcome_action in &outcomes {
        let (outcome, prob) = match outcome_action {
            BbAction::Chance { outcome, .. } => (
                *outcome,
                outcome_action.prob_f32().unwrap_or(0.0) as f64,
            ),
            BbAction::Player(_) => continue,
        };
        if prob <= 0.0 {
            continue;
        }
        let mut sim = state.clone();
        roll_outcomes::fix_for_outcome(&mut sim, outcome);
        if sim.micro_step(None).is_err() {
            continue;
        }
        weighted_sum += prob * leaf_score(&sim) as f64;
        total_prob += prob;
    }
    if total_prob <= 0.0 {
        None
    } else {
        Some((weighted_sum / total_prob) as i64)
    }
}

/// PUCT(a) = Q(a) + c · P(a) · √N(parent) / (1 + N(a))
///
/// Unexplored children (`score == None`) have `Q = 0` and `N(a) = 0`, so
/// their value collapses to `c · P · √N(parent)` — ranked purely by
/// prior. This replaces the pure-UCT `f32::INFINITY` sentinel for
/// unexplored children; high-prior unexplored children are still
/// preferred, but low-prior unexplored children no longer crowd out
/// well-scored explored siblings.
fn puct_value(score: Option<&BbScore>, parent_visits: f32, prior: f32) -> f32 {
    let parent_term = parent_visits.max(1.0).sqrt();
    match score {
        None => PUCT_C * prior * parent_term,
        Some(s) => {
            let v = s.visits.load(Ordering::Relaxed) as f32;
            let q = s.score as f32;
            q + PUCT_C * prior * parent_term / (1.0 + v)
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
        // search. `expose_rolls = true` keeps `pending_roll` visible on
        // post-action states, letting `score_leaf` invoke
        // `expected_leaf_score` to give pickup / dodge / GFI chance
        // nodes a probability-weighted Q instead of the misleading
        // pre-roll value. (We tried `expose_rolls = false` to elide the
        // chance-node modelling, but the engine's internal roll
        // resolution per `micro_step` ran orders of magnitude slower
        // for the same search budget.)
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

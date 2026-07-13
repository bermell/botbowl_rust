//! Solved-subtree pruning: once every leaf under a node is terminal, the
//! node's aggregate is final ("solved"), selection skips it, and a solved
//! *root* makes `step()` a no-op so callers can stop their step loop.

use std::ops::Deref;
use std::sync::Arc;

use recon_mcts::prelude::*;

/// A tiny two-level game tree:
///
/// ```text
///        0 (root)
///      /   \
///    a=1    a=2
///     |      |
///    10     20        (interior, one action each)
///     |      |
///   110    120        (terminal: no actions)
/// ```
///
/// States < 100 branch; states ≥ 100 are terminal. Scores are the state
/// value, and backprop takes the max, so the solved root must aggregate
/// to 120.
struct TwoLevelGame;

#[derive(Clone, Debug, Hash, PartialEq)]
struct P;

impl GameDynamics for TwoLevelGame {
    type Player = P;
    type State = u32;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        match state {
            0 => Some(vec![(P, 1), (P, 2)]),
            s if *s < 100 => Some(vec![(P, s + 100)]),
            _ => None, // terminal
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        match state {
            0 => Some(action * 10),
            _ => Some(*action), // interior actions carry the target state
        }
    }

    fn select_node<II, Q, A>(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        _parent_node_state: &Self::State,
        _purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        II: IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        // Prefer unexplored children, else the first offered — enough to
        // sweep this toy tree.
        let mut first = None;
        for (q, a) in scores_and_actions {
            if first.is_none() {
                first = Some(*a);
            }
            if q.is_none() {
                return *a;
            }
        }
        first.expect("selection must be offered at least one child")
    }

    fn backprop_scores<II, Q, A>(
        &self,
        _player: &Self::Player,
        _score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        child_scores_and_actions
            .into_iter()
            .map(|(q, _)| *q)
            .fold(None, |acc, s| Some(acc.map_or(s, |a: f64| a.max(s))))
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        Some(f64::from(*state))
    }
}

#[test]
fn exhausted_tree_solves_and_steps_become_noops() {
    let tree = Arc::new(Tree::new(TwoLevelGame, StoreState, P, 0u32));
    assert!(!tree.is_solved(), "a fresh tree must not be solved");

    // The whole game has 5 nodes; a handful of steps must exhaust it.
    // (Each step materialises at most one node.)
    let mut steps_to_solve = None;
    for i in 0..32 {
        if tree.is_solved() {
            steps_to_solve = Some(i);
            break;
        }
        tree.step();
    }
    let steps = steps_to_solve.expect("a 5-node game must solve within 32 steps");
    assert!(steps >= 4, "cannot be solved before every node was materialised");

    // Solved root: further steps are no-ops and return None.
    assert!(tree.step().is_none(), "step on a solved tree must be a no-op");

    // The final aggregate must be the true max-backprop value (120),
    // i.e. solving happened *after* full exploration, not instead of it.
    let (action, info) = tree
        .get_next_move_info()
        .expect("solved tree still reports root children")
        .into_iter()
        .max_by(|(_, a), (_, b)| {
            a.score
                .unwrap_or(f64::MIN)
                .partial_cmp(&b.score.unwrap_or(f64::MIN))
                .unwrap()
        })
        .unwrap();
    assert_eq!(action, 2, "the a=2 branch dominates (leaf 120 > 110)");
    assert_eq!(info.score, Some(120.0));
}

//! Regression: dropping a deep search DAG must run at constant stack
//! depth. The old `on_drop` recursed once per level (drop → on_drop →
//! drain children → drop …), which overflowed the stack on deep graphs —
//! first seen with multi-thousand-node chains in Blood Bowl small-board
//! searches.

use std::ops::Deref;

use recon_mcts::prelude::*;

/// A linear chain: state `s` has `width` actions **all leading to
/// `s + 1`**, until `len` (terminal). With `width == 1` this is the
/// deepest possible tree per node count; with `width > 1` every level
/// recombines into a single child with multiple parent edges (a diamond
/// per level), which exercises the multi-parent teardown ordering: a
/// dying parent must materialise a last-reference child's state *before*
/// removing its edge, even when the child's other parents died first.
struct LineGame {
    len: u32,
    width: u32,
}

#[derive(Clone, Debug, Hash, PartialEq)]
struct P;

impl GameDynamics for LineGame {
    type Player = P;
    type State = u32;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        if *state < self.len {
            // Actions `k * (len + 1) + (s + 1)` for k in 0..width —
            // distinct actions, identical successor state (recombination).
            Some((0..self.width).map(|k| (P, k * (self.len + 1) + state + 1)).collect())
        } else {
            None // terminal
        }
    }

    fn apply_action(&self, _state: Self::State, action: &Self::Action) -> Option<Self::State> {
        Some(action % (self.len + 1))
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
        *scores_and_actions
            .into_iter()
            .next()
            .expect("selection must be offered at least one child")
            .1
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
        child_scores_and_actions.into_iter().next().map(|(q, _)| *q)
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
fn deep_linear_tree_drops_without_stack_overflow() {
    const DEPTH: u32 = 2_000;

    let tree = Tree::new(LineGame { len: DEPTH, width: 1 }, StoreState, P, 0u32);
    // Each step materialises at most one node; stop early once solved.
    for _ in 0..=DEPTH + 1 {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }

    // Hand the last reference to a deliberately tiny-stack thread. The
    // recursive drop needed O(DEPTH) frames and overflowed here (aborting
    // the process); the iterative teardown runs at constant depth.
    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(move || drop(tree))
        .unwrap()
        .join()
        .unwrap();
}

/// Every level a diamond: two distinct actions lead to the same child, so
/// each node in the chain has two parent edges. Dropping the tree must
/// materialise each last-reference child's state *before* removing the
/// final parent edge — a naive worklist that parks *every* drained child
/// inflates the child's strong count past the `== 1` check its other
/// dying parent relies on, leaving a registered node with no state and no
/// parents (`child needs a parent` / `can't calculate state` panics).
#[test]
fn recombined_diamond_chain_drops_cleanly() {
    const DEPTH: u32 = 300;

    // GetState memory: states are derived from parents on demand, so the
    // materialise-before-unlink ordering actually matters here (StoreState
    // would mask it — every node permanently holds its state).
    let tree = Tree::new(LineGame { len: DEPTH, width: 2 }, GetState, P, 0u32);
    for _ in 0..3 * DEPTH {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }

    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(move || drop(tree))
        .unwrap()
        .join()
        .unwrap();
}

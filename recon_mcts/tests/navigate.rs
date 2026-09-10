//! Read-only navigation of a finished tree: `Tree::get_root_node`,
//! `Node::get_children_info`, `Node::get_child`.
//!
//! `SearchTree::get_next_move_info` only ever reaches the root's children, so
//! anything that wants to *inspect* a DAG deeper than one level (a UI walking
//! into the search, a diagnostic printing a principal variation) needs these.
//! The contract they must keep is that inspection is inert: no descent, no
//! visit bump, no virtual loss.

use std::ops::Deref;
use std::sync::Arc;

use recon_mcts::prelude::*;

/// ```text
///        0 (root)
///      /   \
///    a=1    a=2
///     |      |
///    10     20        (interior, one action each)
///     |      |
///   110    120        (terminal)
/// ```
struct Chain;

#[derive(Clone, Debug, Hash, PartialEq)]
struct P;

impl GameDynamics for Chain {
    type Player = P;
    type State = u32;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        match state {
            0 => Some(vec![(P, 1), (P, 2)]),
            s if *s < 100 => Some(vec![(P, s + 100)]),
            _ => None,
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        match state {
            0 => Some(action * 10),
            _ => Some(*action),
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

fn solved_tree() -> Arc<TreeAlias<Chain, StoreState>> {
    let tree = Arc::new(Tree::new(Chain, StoreState, P, 0u32));
    for _ in 0..32 {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }
    assert!(tree.is_solved(), "the 5-node game should solve");
    tree
}

#[test]
fn navigating_walks_the_whole_chain() {
    let tree = solved_tree();
    let root = tree.get_root_node();
    let root_info = root.get_node_info();
    assert_eq!(root_info.state, Some(0));
    assert_eq!(root_info.depth, 0);

    let mut edges = root.get_children_info().expect("an expanded root has a children map");
    edges.sort_by_key(|(a, _)| *a);
    assert_eq!(edges.iter().map(|(a, _)| *a).collect::<Vec<_>>(), vec![1, 2]);
    // Both branches are proven out, and the a=2 branch aggregates to 120.
    let by_action = |want: u32| edges.iter().find(|(a, _)| *a == want).map(|(_, i)| i.clone()).unwrap();
    assert_eq!(by_action(1).state, Some(10));
    assert_eq!(by_action(2).state, Some(20));
    assert_eq!(by_action(2).score, Some(120.0));
    assert!(by_action(2).solved);

    // One level down by action, then the terminal below it.
    let mid = root.get_child(&2).expect("a=2 is materialised");
    let mid_info = mid.get_node_info();
    assert_eq!(mid_info.state, Some(20));
    assert_eq!(mid_info.depth, 1);
    assert_eq!(mid_info.n_parents, 1);

    let mid_edges = mid.get_children_info().expect("20 branches to 120");
    assert_eq!(mid_edges.len(), 1);
    let (leaf_action, leaf_info) = &mid_edges[0];
    assert_eq!(*leaf_action, 120);
    assert_eq!(leaf_info.state, Some(120));
    assert_eq!(leaf_info.score, Some(120.0));
    assert!(matches!(leaf_info.n_children, Status::Terminal));

    let leaf = mid.get_child(&120).expect("120 is materialised");
    assert!(
        leaf.get_children_info().is_none(),
        "a terminal node has no children map to walk into"
    );
}

#[test]
fn navigating_does_not_touch_the_tree() {
    let tree = solved_tree();
    let before = tree.get_root_info();

    // Walk everything reachable, repeatedly.
    for _ in 0..10 {
        let root = tree.get_root_node();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let _ = node.get_node_info();
            if let Some(children) = node.get_children_info() {
                for (action, _) in children {
                    if let Some(child) = node.get_child(&action) {
                        stack.push(child);
                    }
                }
            }
        }
    }

    let after = tree.get_root_info();
    assert_eq!(before.score, after.score, "inspection changed the root aggregate");
    assert_eq!(before.depth, after.depth);
    assert_eq!(before.solved, after.solved);
    assert!(tree.is_solved(), "inspection unsolved the tree");
}

#[test]
fn get_child_returns_none_for_an_edge_that_does_not_exist() {
    let tree = solved_tree();
    let root = tree.get_root_node();
    assert!(root.get_child(&99).is_none());
}

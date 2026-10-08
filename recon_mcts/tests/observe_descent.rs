//! `GameDynamics::observe_descent`: once per descent, the edges it took (each with the player of
//! the node it left), the state it stopped at, and how it stopped. Observational only — the hook
//! has no way back into the tree.

use std::ops::Deref;
use std::sync::{Arc, Mutex};

use recon_mcts::prelude::*;

/// A diamond with a tail:
///
/// ```text
///        0 (A)
///      /   \
///    a=1    a=2
///     1(B)   2(B)
///      \   /
///       3 (A)        (reached from both 1 and 2: the second path recombines into a twin)
///       |
///       4            (terminal)
/// ```
///
/// Two players alternate, so an edge's player tells which side chose it.
#[derive(Default)]
struct Diamond {
    seen: Mutex<Vec<Observed>>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum P {
    A,
    B,
}

#[derive(Clone, Debug)]
struct Observed {
    edges: Vec<(P, u32)>,
    leaf: u32,
    end: DescentEnd,
}

/// Edges from the root to `state` along any path.
fn depth_of(state: u32) -> usize {
    match state {
        0 => 0,
        1 | 2 => 1,
        3 => 2,
        _ => 3,
    }
}

impl GameDynamics for Diamond {
    type Player = P;
    type State = u32;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<Self::Action>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        match state {
            0 => Some(vec![1, 2]),
            1 | 2 => Some(vec![3]),
            3 => Some(vec![4]),
            _ => None,
        }
    }

    fn apply_action(&self, _state: Self::State, action: &Self::Action) -> Option<Self::State> {
        Some(*action)
    }

    fn player_for_child(
        &self,
        parent_player: &Self::Player,
        _action: &Self::Action,
        _child_state: &Self::State,
    ) -> Self::Player {
        match parent_player {
            P::A => P::B,
            P::B => P::A,
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

    fn observe_descent<'a, E>(&self, edges: E, leaf: &Self::State, end: DescentEnd)
    where
        E: Iterator<Item = (&'a Self::Player, &'a Self::Action)>,
        Self::Player: 'a,
        Self::Action: 'a,
    {
        let edges = edges.map(|(p, a)| (*p, *a)).collect();
        self.seen.lock().unwrap().push(Observed {
            edges,
            leaf: *leaf,
            end,
        });
    }
}

/// Run the diamond to exhaustion; returns the tree and the descents run.
fn sweep() -> (Arc<Tree<NodeAlias<Diamond, StoreState>, Diamond>>, usize) {
    let tree = Arc::new(Tree::new(Diamond::default(), StoreState, P::A, 0u32));
    let mut steps = 0;
    while !tree.is_solved() {
        tree.step();
        steps += 1;
        assert!(steps < 100, "the diamond must solve in a handful of descents");
    }
    (tree, steps)
}

#[test]
fn every_descent_is_observed_once_with_its_edges_leaf_and_end() {
    let (tree, steps) = sweep();
    let seen = tree.get_game_dynamics().seen.lock().unwrap().clone();
    assert_eq!(seen.len(), steps, "one observation per descent that ran");

    for o in &seen {
        assert_eq!(
            o.edges.len(),
            depth_of(o.leaf),
            "a descent's edges are the plies from the root to its leaf, a twin swap adds none: {o:?}"
        );
        for (i, (p, _)) in o.edges.iter().enumerate() {
            let want = if i % 2 == 0 { P::A } else { P::B };
            assert_eq!(*p, want, "edge {i} names the player of the node it left: {o:?}");
        }
    }

    // The very first descent expands the root itself: no edges.
    assert_eq!(seen[0].edges.len(), 0);
    assert_eq!(seen[0].leaf, 0);
    assert_eq!(seen[0].end, DescentEnd::Expanded);

    // The terminal is first met as a fresh leaf with no actions (`Expanded`); a node whose children
    // are all solved ends the descent that finds it so as `Solved`. Only the terminal itself can end
    // a descent as `Terminal`.
    assert!(seen.iter().any(|o| o.leaf == 4 && o.end == DescentEnd::Expanded));
    assert!(seen.iter().all(|o| o.end != DescentEnd::Terminal || o.leaf == 4));
    assert!(seen.iter().any(|o| o.end == DescentEnd::Solved));
    assert_eq!(
        seen.last().unwrap().end,
        DescentEnd::Solved,
        "the search ends on a solved node"
    );
}

#[test]
fn the_recombined_path_reaches_the_twin_with_two_edges() {
    let (tree, _) = sweep();
    let seen = tree.get_game_dynamics().seen.lock().unwrap().clone();
    // State 3 is reached through both 1 and 2; whichever came second recombined into a twin.
    let via: Vec<u32> = seen
        .iter()
        .filter(|o| o.edges.len() >= 2)
        .map(|o| o.edges[0].1)
        .collect();
    assert!(
        via.contains(&1) && via.contains(&2),
        "both paths into 3 were descended: {via:?}"
    );
}

/// The lean inspection accessors a tree-statistics walk uses: children without state copies, the
/// mover, the stored state in place.
#[test]
fn children_mover_and_state_are_readable_without_cloning_states() {
    let (tree, _) = sweep();
    let root = tree.get_root_node();
    assert_eq!(root.mover(), Some(&P::A));
    assert_eq!(root.with_state(|s| s.copied()), Some(0));
    let mut children = root.get_children().expect("the root is expanded");
    children.sort_by_key(|(a, _)| *a);
    let actions: Vec<u32> = children.iter().map(|(a, _)| *a).collect();
    assert_eq!(actions, vec![1, 2]);
    for (a, child) in &children {
        assert_eq!(child.mover(), Some(&P::B));
        assert_eq!(child.with_state(|s| s.copied()), Some(*a));
    }
}

//! Plan 035: each node's mover is derived by `GameDynamics::player_for_child`
//! when the node is **materialised**, not when its action was enumerated.
//!
//! Two things need pinning, and they are the whole safety argument for moving
//! that computation later:
//!
//! * **T1 — the tag cannot be read early.** `Node::player()` goes through
//!   `OnceLock::get().expect(..)`, which is a guarantee rather than a test:
//!   it holds in every build, every game, at any worker count. What a test
//!   *can* add is documenting the contract on the inspection API, which is
//!   the one place a placeholder is legitimately visible — it must report
//!   `None`, not panic and not invent a mover.
//! * **T4 — the tag is right.** Every materialised node's reported player
//!   equals what `player_for_child` gives from the parent edge that reaches
//!   it. That is the game-agnostic statement of the whole plan.
//!
//! The game here alternates its mover **independently of its state** — the
//! same pile count is reachable at either parity, so the mover is genuinely
//! not a function of the state. That is deliberate: a single-player game (or
//! one whose state names its own mover) would make the T4 assertion
//! vacuously true. It is the same property nim has, and the reason
//! `player_for_child` takes `parent_player` rather than being a
//! `player_for_state`.

use std::collections::HashSet;
use std::ops::Deref;

use recon_mcts::prelude::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Side {
    A,
    B,
}

/// Subtract 1 or 2 from a pile, alternating turns. Pile 4 is reachable from 6
/// both as `6-2` (one ply, mover B) and `6-1-1` (two plies, mover A), so
/// `State` alone cannot name the mover — the recombining registry keys on
/// `(player, state)` and keeps both.
struct Countdown;

impl GameDynamics for Countdown {
    type Player = Side;
    type State = u32;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<Self::Action>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        match state {
            0 => None, // terminal
            1 => Some(vec![1]),
            _ => Some(vec![1, 2]),
        }
    }

    fn player_for_child(
        &self,
        parent_player: &Self::Player,
        _action: &Self::Action,
        _child_state: &Self::State,
    ) -> Self::Player {
        match parent_player {
            Side::A => Side::B,
            Side::B => Side::A,
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        state.checked_sub(*action)
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
        // Unscored (placeholder) children first, so the whole DAG gets
        // materialised quickly; otherwise first offered.
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
        player: &Self::Player,
        _score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        // Minimax, so the node's own player tag actually steers the result —
        // a wrong tag here is exactly the silent min/max flip plan 023 found.
        let it = child_scores_and_actions.into_iter().map(|(q, _)| *q);
        match player {
            Side::A => it.fold(None, |acc, s| Some(acc.map_or(s, |a: f64| a.max(s)))),
            Side::B => it.fold(None, |acc, s| Some(acc.map_or(s, |a: f64| a.min(s)))),
        }
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

/// T1: a child that has been enumerated but never descended into reports
/// `player: None` — it has no mover yet, by construction.
///
/// A single `step()` on a fresh tree expands the root into placeholders and
/// returns, so every root child is unmaterialised at that point.
#[test]
fn an_unmaterialised_placeholder_reports_no_player() {
    let tree = Tree::new(Countdown, StoreState, Side::A, 6u32);
    tree.step();

    let root = tree.get_root_node();
    assert_eq!(
        root.get_node_info().player,
        Some(Side::A),
        "the root is materialised at construction"
    );

    let children = root.get_children_info().expect("one step expands the root");
    assert_eq!(children.len(), 2, "pile 6 offers -1 and -2");
    for (action, info) in &children {
        assert_eq!(
            info.player, None,
            "child {} was enumerated but never descended into — it cannot have a mover yet",
            action
        );
        // The rest of the placeholder read-out, for the same reason.
        assert_eq!(info.score, None, "child {} has not been scored", action);
        assert!(
            matches!(info.n_children, Status::Pending),
            "child {} has not been expanded",
            action
        );
    }
}

/// T4: for every materialised node in the finished DAG, the reported player is
/// exactly what `player_for_child` yields from the parent edge that reaches
/// it. Game-agnostic — this is the statement plan 035 rests on.
#[test]
fn every_materialised_node_carries_the_tag_player_for_child_would_give() {
    let tree = Tree::new(Countdown, StoreState, Side::A, 6u32);
    for _ in 0..500 {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }
    assert!(tree.is_solved(), "the countdown game should solve");

    let root = tree.get_root_node();
    let mut checked = 0usize;
    let mut placeholders = 0usize;
    let mut seen: HashSet<*const ()> = HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let info = node.get_node_info();
        let Some(parent_player) = info.player else {
            // An unmaterialised placeholder: nothing to check, and nothing
            // below it to walk into.
            placeholders += 1;
            continue;
        };
        let Some(children) = node.get_children_info() else {
            continue;
        };
        for (action, child_info) in children {
            let Some(child) = node.get_child(&action) else {
                continue;
            };
            if let Some(child_player) = child_info.player {
                let state = child_info.state.expect("StoreState keeps every node's state");
                assert_eq!(
                    child_player,
                    GameDynamics::player_for_child(&Countdown, &parent_player, &action, &state),
                    "node at state {} reached by {:?} --{}--> \
                     carries a mover the dynamics would not have given it",
                    state,
                    parent_player,
                    action
                );
                checked += 1;
            }
            // The DAG recombines, so guard against re-walking shared nodes.
            if seen.insert(&*child as *const _ as *const ()) {
                stack.push(child);
            }
        }
    }
    assert!(
        checked >= 8,
        "expected a real walk over the solved DAG, checked only {} edges \
         ({} placeholders skipped)",
        checked,
        placeholders
    );
}

//! Plan 070 (botbowl): a placeholder that materialises as an existing node (a twin) gives its
//! parent a scored child the parent's value has not seen. The parent must be backed up again, even
//! when the descent that continues into the twin ends below it without a backup reaching it.
//!
//! ```text
//!         0 (root)
//!        / \
//!       1   2          scored leaves (100, 200)
//!        \ /
//!         10           scored (1000): reached from 1 first, then from 2 as a twin
//!          |
//!         20           unscored (like a chance node with no outcome scored yet)
//!          |
//!         21           unscored, terminal
//! ```
//!
//! The backup is the sum of the scored children (`None` when there is none), so after the twin
//! splice node 2 must read 1000 and the root 2000. Before plan 070, the descent through the twin
//! ended at 20, whose backup is withheld (nothing scored below it), so 10 did not change, the walk
//! stopped there, and node 2 kept its leaf value 200 next to a child worth 1000.

use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};

use recon_mcts::prelude::*;

struct Splice {
    /// The root's choices once both its children are scored: 1, then 2.
    script: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
struct P;

impl GameDynamics for Splice {
    type Player = P;
    type State = u32;
    type Action = u32;
    type Score = i64;
    type ActionIter = Vec<u32>;

    fn available_actions(&self, _: &P, s: &u32) -> Option<Vec<u32>> {
        match s {
            0 => Some(vec![1, 2]),
            1 | 2 => Some(vec![10]),
            10 => Some(vec![20]),
            20 => Some(vec![21]),
            _ => None,
        }
    }

    /// The action is the state it leads to.
    fn apply_action(&self, _: u32, a: &u32) -> Option<u32> {
        Some(*a)
    }

    fn player_for_child(&self, _: &P, _: &u32, _: &u32) -> P {
        P
    }

    /// An unscored child first (the smallest), else the root follows its script.
    fn select_node<II, Q, A>(&self, _: Option<&i64>, _: &P, s: &u32, _: SelectNodeState, children: II) -> u32
    where
        II: IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<i64>>,
        A: Deref<Target = u32>,
    {
        let children: Vec<(bool, u32)> = children.into_iter().map(|(q, a)| (q.is_some(), *a)).collect();
        if let Some(a) = children.iter().filter(|(scored, _)| !scored).map(|(_, a)| *a).min() {
            return a;
        }
        if *s == 0 {
            return [1, 2][self.script.fetch_add(1, Ordering::Relaxed) % 2];
        }
        children.iter().map(|(_, a)| *a).min().unwrap()
    }

    fn backprop_scores<II, Q, A>(&self, _: &P, _: Option<&i64>, children: II) -> Option<i64>
    where
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = u32>,
        Q: Deref<Target = i64>,
    {
        children
            .into_iter()
            .map(|(q, _)| *q)
            .fold(None, |acc, v| Some(acc.unwrap_or(0) + v))
    }

    fn score_leaf(&self, _: Option<&i64>, _: &P, s: &u32) -> Option<i64> {
        match s {
            1 => Some(100),
            2 => Some(200),
            10 => Some(1000),
            _ => None,
        }
    }
}

fn value(tree: &TreeAlias<Splice, StoreState>, path: &[u32]) -> Option<i64> {
    let mut node = tree.get_root_node();
    for a in path {
        node = node.get_child(a)?;
    }
    node.with_score(|q| q.copied())
}

#[test]
fn a_twin_spliced_under_a_parent_backs_the_parent_up() {
    let tree = Tree::new(
        Splice {
            script: AtomicUsize::new(0),
        },
        StoreState,
        P,
        0u32,
    );
    // 1: expand the root. 2, 3: materialise 1 and 2. 4: 1 -> 10 (a miss: 10 is scored and backed
    // up through 1). 5: 2 -> 10 is a twin; the descent goes on to 20 and stops there, unscored.
    for _ in 0..5 {
        tree.step();
    }
    assert_eq!(value(&tree, &[1, 10]), Some(1000));
    assert_eq!(value(&tree, &[2, 10]), Some(1000), "node 2's child is the twin");
    assert_eq!(value(&tree, &[1, 10, 20]), None, "20 has nothing scored below it");
    assert_eq!(value(&tree, &[1]), Some(1000));
    assert_eq!(value(&tree, &[2]), Some(1000), "node 2 must see the twin it gained");
    assert_eq!(value(&tree, &[]), Some(2000));
}

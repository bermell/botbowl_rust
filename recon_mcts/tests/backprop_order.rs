//! Plan 070 (botbowl): backprop must leave no stale value behind, whatever order the DAG's
//! containers hand nodes back in.
//!
//! The game is a pair of counters, each stepped by 1 or 2 (or both by 1), so one state is reached
//! by paths of different lengths and the DAG is full of transpositions whose depth must be raised
//! after the fact: the case where a child could end up level with its parent, and a depth-ordered
//! walk could then recompute the parent before the child (the exp079 hypothesis). The backup is
//! the sum of the children's values, so any node recomputed too early (or not at all) shows up as
//! stored ≠ recomputed. On one thread neither happens: depth is maintained as one more than the
//! deepest parent, so the walk is a reverse topological order whatever order `parents` iterates
//! in. (The stale value plan 070 did find is a twin splice: `tests/twin_splice.rs`.)
//!
//! After every step the whole DAG is audited: every edge goes strictly deeper, and every expanded
//! node's stored value equals the backup of its children's current values.

use std::collections::HashSet;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use recon_mcts::prelude::*;

/// The game ends once the two counters sum to this.
const END: u32 = 24;
/// Each move: (+a, +b).
const MOVES: [(u32, u32); 5] = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)];

struct Steps {
    rng: AtomicU64,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
struct P;

/// A pure value per state.
fn leaf_value(s: (u32, u32)) -> i64 {
    let mut x = (u64::from(s.0) << 32 | u64::from(s.1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    (x % 1000) as i64 - 500
}

fn backup<'a>(children: impl Iterator<Item = &'a i64>) -> Option<i64> {
    children.fold(None, |acc, v| Some(acc.unwrap_or(0).wrapping_add(*v)))
}

impl GameDynamics for Steps {
    type Player = P;
    type State = (u32, u32);
    type Action = u32;
    type Score = i64;
    type ActionIter = Vec<u32>;

    fn available_actions(&self, _: &P, s: &(u32, u32)) -> Option<Vec<u32>> {
        (s.0 + s.1 < END).then(|| (0..MOVES.len() as u32).collect())
    }

    fn apply_action(&self, s: (u32, u32), a: &u32) -> Option<(u32, u32)> {
        let (da, db) = MOVES[*a as usize];
        Some((s.0 + da, s.1 + db))
    }

    fn player_for_child(&self, _: &P, _: &u32, _: &(u32, u32)) -> P {
        P
    }

    /// A pseudo-random child (selection need not be pure), so the steps cover the DAG in an
    /// arbitrary order.
    fn select_node<II, Q, A>(&self, _: Option<&i64>, _: &P, _: &(u32, u32), _: SelectNodeState, children: II) -> u32
    where
        II: IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<i64>>,
        A: Deref<Target = u32>,
    {
        let mut actions: Vec<u32> = children.into_iter().map(|(_, a)| *a).collect();
        actions.sort_unstable();
        let r = self
            .rng
            .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        actions[(r >> 33) as usize % actions.len()]
    }

    fn backprop_scores<II, Q, A>(&self, _: &P, _: Option<&i64>, children: II) -> Option<i64>
    where
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = u32>,
        Q: Deref<Target = i64>,
    {
        let values: Vec<i64> = children.into_iter().map(|(q, _)| *q).collect();
        backup(values.iter())
    }

    fn score_leaf(&self, _: Option<&i64>, _: &P, s: &(u32, u32)) -> Option<i64> {
        Some(leaf_value(*s))
    }
}

#[derive(Debug, Default, PartialEq)]
struct Audit {
    nodes: usize,
    depth_violations: usize,
    stale: usize,
}

fn audit(tree: &TreeAlias<Steps, StoreState>) -> Audit {
    let mut out = Audit::default();
    let mut seen = HashSet::new();
    let mut stack = vec![tree.get_root_node()];
    while let Some(node) = stack.pop() {
        if !seen.insert(&*node as *const _ as usize) {
            continue;
        }
        out.nodes += 1;
        let Some(children) = node.get_children() else { continue };
        let mut values = Vec::new();
        for (_, c) in children {
            if c.mover().is_none() {
                continue;
            }
            if c.depth() <= node.depth() {
                out.depth_violations += 1;
            }
            if let Some(v) = c.with_score(|q| q.copied()) {
                values.push(v);
            }
            stack.push(c);
        }
        if let Some(again) = backup(values.iter()) {
            if node.with_score(|q| q.copied()) != Some(again) {
                out.stale += 1;
            }
        }
    }
    out
}

#[test]
fn one_worker_leaves_no_stale_value_in_a_transposing_dag() {
    for seed in 0..20u64 {
        let tree = Tree::new(
            Steps {
                rng: AtomicU64::new(seed),
            },
            StoreState,
            P,
            (0u32, 0u32),
        );
        for step in 0..1500 {
            tree.step();
            let a = audit(&tree);
            assert_eq!((a.depth_violations, a.stale), (0, 0), "seed {seed} step {step}: {a:?}");
        }
    }
}

/// Several workers race on one tree (a measurement, not a gate). Concurrent depth updates and
/// backups are not serialised, so a run can end with an edge that does not go deeper, or a value
/// that no later walk refreshed; the earlier 1-D version of this game showed one depth inversion in
/// ten seeds. botbowl's loop searches on one thread (Gumbel root, `workers = 1`), where the test
/// above is exact. `cargo test --test backprop_order -- --ignored --nocapture` prints the counts.
#[test]
#[ignore = "multi-threaded search is not exact (plan 070); prints counts"]
fn concurrent_workers_audit() {
    let mut total = Audit::default();
    for seed in 0..10u64 {
        let tree = Arc::new(Tree::new(
            Steps {
                rng: AtomicU64::new(seed),
            },
            StoreState,
            P,
            (0u32, 0u32),
        ));
        std::thread::scope(|s| {
            for _ in 0..4 {
                let t = Arc::clone(&tree);
                s.spawn(move || {
                    for _ in 0..1000 {
                        t.step();
                    }
                });
            }
        });
        let a = audit(&tree);
        total.nodes += a.nodes;
        total.depth_violations += a.depth_violations;
        total.stale += a.stale;
    }
    eprintln!("4 workers, 10 seeds: {total:?}");
}

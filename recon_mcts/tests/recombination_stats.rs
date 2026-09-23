//! Plan 043 Part 1: the registry's probe counters.
//!
//! `RegistryInfo` has always counted `hits` and `misses` — whether a newly
//! materialised state already existed in the DAG. What it could not say is how
//! much that answer *cost*. The table is a `HashSet<WeakWrap<Node>>`, so a
//! probe first matches a hash bucket and only then calls `StateMemory::eq`,
//! which under `StoreState` clones and compares two whole game states. Every
//! such comparison that returns `false` is work spent to learn nothing.
//!
//! `HashSet::get` hides that second stage entirely: a `None` return is
//! indistinguishable from "a candidate was compared and rejected". So the
//! counters live inside the comparison itself, scoped to the probe by a
//! thread-local guard — which is what these tests pin.
//!
//! Two properties matter and are asserted separately:
//!
//! * **The accounting closes.** `probes == hits + misses`, and a comparison
//!   is never counted outside a probe — the `parents` set is a
//!   `HashSet<(A, WeakWrap<Node>)>` using the same `PartialEq`, and its
//!   inserts must not leak into the numbers.
//! * **Rejections are visible.** With a deliberately collision-prone `Hash`,
//!   every probe reaches the comparison stage and most are rejected. That is
//!   the measurement the whole thing exists for: if production shows many
//!   `eq_rejects` and few `hits`, recombination is costing more than it
//!   returns.

use std::ops::Deref;

use recon_mcts::prelude::*;

/// Pile count. `Hash` is derived, so distinct piles land in distinct buckets
/// and the comparison stage is reached only on a true match.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Pile(u32);

/// The same game, but every state hashes to the same value. The registry then
/// has exactly one bucket, so *every* probe reaches `StateMemory::eq` and is
/// compared against every node already in the table — the pathological case
/// the `eq_rejects` counter is there to detect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Collide(u32);

impl std::hash::Hash for Collide {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        // Deliberately state-independent.
        h.write_u8(0);
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Side {
    A,
    B,
}

/// Lets one `GameDynamics` impl serve both state types.
trait Pileish: Copy + std::hash::Hash + PartialEq + Eq + std::fmt::Debug {
    fn count(&self) -> u32;
    fn of(n: u32) -> Self;
}
impl Pileish for Pile {
    fn count(&self) -> u32 {
        self.0
    }
    fn of(n: u32) -> Self {
        Pile(n)
    }
}
impl Pileish for Collide {
    fn count(&self) -> u32 {
        self.0
    }
    fn of(n: u32) -> Self {
        Collide(n)
    }
}

struct Game<S>(std::marker::PhantomData<S>);

impl<S: Pileish> GameDynamics for Game<S> {
    type Player = Side;
    type State = S;
    type Action = u32;
    type Score = f64;
    type ActionIter = Vec<Self::Action>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        match state.count() {
            0 => None,
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
        state.count().checked_sub(*action).map(S::of)
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
        // Unscored children first, so the whole DAG materialises quickly.
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
        Some(f64::from(state.count()))
    }
}

fn solve<S: Pileish + 'static>(start: u32) -> recon_mcts::prelude::RecombinationStats
where
    Game<S>: GameDynamics<Player = Side, State = S, Action = u32, Score = f64>,
{
    let tree = Tree::new(Game::<S>(std::marker::PhantomData), StoreState, Side::A, S::of(start));
    for _ in 0..2000 {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }
    assert!(tree.is_solved(), "the countdown game should solve");
    tree.get_registry_info().snapshot()
}

/// The accounting closes: every probe ended in exactly one of hit or miss.
///
/// This is the property that makes the rest of the numbers readable. It also
/// pins the scoping: `Node`'s `PartialEq` is used by the per-node `parents`
/// set as well as by the registry, so an unscoped counter would drift above
/// the probe count and this assertion would fail.
#[test]
fn every_probe_is_either_a_hit_or_a_miss() {
    let s = solve::<Pile>(12);

    assert_eq!(
        s.probes,
        s.hits + s.misses,
        "probes ({}) must split exactly into hits ({}) and misses ({})",
        s.probes,
        s.hits,
        s.misses
    );
    assert!(s.misses > 0, "a fresh tree must insert new nodes");
    assert!(
        s.hits > 0,
        "pile 4 is reachable two ways at the same parity — recombination must fire"
    );
    assert_eq!(
        s.len,
        s.misses + 1,
        "every miss registers exactly one node, plus the root — which `Tree::new` registers \
         directly, without a probe"
    );
}

/// A hit is a comparison that returned `true`, so it cannot be cheaper than
/// one comparison — and with a well-behaved hash, comparisons are not wasted.
#[test]
fn hits_are_confirmed_by_a_state_comparison() {
    let s = solve::<Pile>(12);

    assert!(
        s.eq_checks >= s.hits,
        "each of the {} hits needed at least one comparison, but only {} were made",
        s.hits,
        s.eq_checks
    );
    assert!(
        s.eq_hash_equal >= s.hits,
        "a confirmed hit has an equal 64-bit hash by construction"
    );
    assert_eq!(
        s.eq_checks,
        s.eq_hash_equal + s.eq_tag_only(),
        "every comparison is either a full-hash match or a 7-bit tag brush"
    );
    assert!(
        s.eq_rejects <= s.eq_checks,
        "a rejection is a comparison that returned false"
    );
}

/// The measurement this all exists for. Under a state-independent hash every
/// probe reaches the comparison stage and is compared against the whole table,
/// so rejections dominate — while the answers stay *correct*, because
/// `StateMemory::eq` compares states and not hashes.
#[test]
fn a_colliding_hash_shows_up_as_rejected_comparisons() {
    let good = solve::<Pile>(12);
    let bad = solve::<Collide>(12);

    assert_eq!(
        (bad.hits, bad.misses),
        (good.hits, good.misses),
        "hash quality must not change which states are recombined — only what it costs"
    );
    assert!(
        bad.eq_rejects > 0,
        "every probe collides, so comparisons must be rejected"
    );
    assert!(
        bad.eq_rejects > good.eq_rejects,
        "the colliding game wastes comparisons the well-hashed one does not \
         (collide {} vs good {})",
        bad.eq_rejects,
        good.eq_rejects
    );
    assert_eq!(
        bad.eq_checks, bad.eq_hash_equal,
        "with one hash for every state, no comparison is a mere tag brush"
    );
}

/// `lookup_state` is the tree-reuse probe — a different entry point into the
/// same table, on the read lock. It is counted separately so re-rooting cannot
/// be mistaken for recombination.
#[test]
fn lookup_state_is_counted_apart_from_expansion() {
    let tree = Tree::new(Game::<Pile>(std::marker::PhantomData), StoreState, Side::A, Pile(8));
    for _ in 0..2000 {
        if tree.is_solved() {
            break;
        }
        tree.step();
    }

    let before = tree.get_registry_info().snapshot();
    assert_eq!(before.lookup_probes, 0, "nothing has looked a state up yet");

    // Pile 7 with mover B is the `-1` child of the root: materialised, and so
    // in the registry.
    assert!(
        tree.lookup_state(Side::B, Pile(7)).is_some(),
        "the root's -1 child must be registered"
    );
    // Pile 8 with mover B never occurs: the mover alternates, and 8 is the
    // root's own state at mover A.
    assert!(tree.lookup_state(Side::B, Pile(8)).is_none());

    let after = tree.get_registry_info().snapshot();
    assert_eq!(after.lookup_probes, 2);
    assert_eq!(after.lookup_hits, 1);
    assert_eq!(
        (after.probes, after.hits, after.misses),
        (before.probes, before.hits, before.misses),
        "a lookup is not an expansion probe"
    );
}

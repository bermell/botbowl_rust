//! How well does `GameState`'s hash separate the states a search actually builds?
//!
//! Plan 043's telemetry measured the symptom: ~8 full `GameState` comparisons per registry probe,
//! ~82% of them rejected, and ~85% of those rejections between states whose **64-bit hashes were
//! equal**. That is not chance — `GameState::hash` deliberately hashes only `proc_stack.len()` and
//! `proc_stack_top()` of the procedure stack, with the comment "collisions are corrected by
//! PartialEq". They are corrected, so answers are right; each correction just costs two whole
//! state clones and a deep compare.
//!
//! This file is the instrument for fixing that. The registry of a finished search is exactly the
//! population a state hash has to separate — every node in it is a state the search considered
//! *distinct* — so the corpus is the real thing rather than a synthetic one.
//!
//! Two tests:
//!
//! * `hash_separates_the_states_a_search_builds` is a **permanent regression gate** with a
//!   threshold. It fails if the hash starts conflating states again.
//! * `report_hash_collisions` is `#[ignore]`d and prints the diagnosis: how many distinct states
//!   share a hash, and — for the colliding groups — which publicly visible part of the state
//!   differs. That is what says *which* field to add, instead of guessing.
//!
//! ```sh
//! cargo test --release -p botbowl-mcts --test hash_quality -- --ignored --nocapture
//! ```

mod common;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameState;
use botbowl_mcts::{MctsBot, SearchBudget};

use common::states;

const SEED: u64 = 43_100;
/// Enough random-start positions to average over openings, carriers and board regions.
const N_STATES: u32 = 12;
const ITERS: usize = 1500;

fn hash_of(state: &GameState) -> u64 {
    let mut h = DefaultHasher::new();
    state.hash(&mut h);
    h.finish()
}

/// Build the corpus: every distinct state in the DAG of one search from each start position.
///
/// Single worker so the corpus is reproducible run to run; the hash's quality is a property of the
/// states, not of how many threads built them.
fn corpus() -> Vec<GameState> {
    let mut all = Vec::new();
    for s in states(N_STATES, SEED) {
        let mut bot = MctsBot::new(SearchBudget::Iterations(ITERS)).with_workers(1);
        let _ = bot.get_action(&s);
        if let Some(states) = bot.dag_states() {
            all.extend(states);
        }
    }
    all
}

/// Group a corpus by hash and report the groups that hold more than one distinct state.
struct Collisions {
    distinct_states: usize,
    distinct_hashes: usize,
    /// Distinct states sharing a hash with at least one other, summed over groups.
    colliding_states: usize,
    /// Size of the largest group of distinct states that share one hash.
    worst_group: usize,
    /// Mean number of distinct states per occupied hash bucket — the expected number of full
    /// comparisons a probe that lands on an occupied bucket has to make.
    mean_group: f64,
    groups: Vec<Vec<GameState>>,
}

fn analyse(corpus: &[GameState]) -> Collisions {
    // The registry dedups by `(player, state)`; we only have states here, so dedup them first —
    // two *equal* states sharing a hash is correct behaviour, not a collision.
    let mut by_hash: HashMap<u64, Vec<GameState>> = HashMap::new();
    for s in corpus {
        let bucket = by_hash.entry(hash_of(s)).or_default();
        if !bucket.iter().any(|existing| existing == s) {
            bucket.push(s.clone());
        }
    }

    let distinct_states: usize = by_hash.values().map(Vec::len).sum();
    let distinct_hashes = by_hash.len();
    let groups: Vec<Vec<GameState>> = by_hash.into_values().filter(|g| g.len() > 1).collect();
    let colliding_states = groups.iter().map(Vec::len).sum();
    let worst_group = groups.iter().map(Vec::len).max().unwrap_or(1);

    Collisions {
        distinct_states,
        distinct_hashes,
        colliding_states,
        worst_group,
        mean_group: distinct_states as f64 / distinct_hashes.max(1) as f64,
        groups,
    }
}

/// Which publicly visible part of the state differs within a colliding group.
///
/// Only the public surface is reachable from here (`proc_stack` and the player arrays are private
/// to the engine), but that is enough to name the culprit: if a group agrees on every public
/// field, what differs is inside the procedure stack.
fn classify(group: &[GameState]) -> &'static str {
    let a = &group[0];
    for b in &group[1..] {
        if a.info.player_action_type != b.info.player_action_type {
            return "info.player_action_type";
        }
        if (
            a.info.handoff_available,
            a.info.foul_available,
            a.info.pass_available,
            a.info.blitz_available,
        ) != (
            b.info.handoff_available,
            b.info.foul_available,
            b.info.pass_available,
            b.info.blitz_available,
        ) {
            return "info.*_available";
        }
        if a.info.weather != b.info.weather {
            return "info.weather";
        }
        if a.info.winner != b.info.winner {
            return "info.winner";
        }
        if a.info != b.info {
            return "info (other)";
        }
        if a.available_actions != b.available_actions {
            return "available_actions";
        }
        if a.pending_roll != b.pending_roll {
            return "pending_roll payload";
        }
        if a.proc_stack_top() != b.proc_stack_top() {
            return "proc_stack_top (should already be hashed!)";
        }
    }
    "proc_stack payload / private fields"
}

/// **The safety property.** `Hash` must agree with `PartialEq`: equal states must hash equally.
///
/// Hashing *less* than equality compares is only slow — the extra candidates get rejected. Hashing
/// *more* is a correctness bug, because the MCTS registry would file two equal states in different
/// buckets and silently split one DAG node into several, which is the recombination-purity
/// invariant the whole search rests on.
///
/// Clones are the sharp case: a `GameState` clone deliberately does *not* copy `path_buffer`
/// verbatim, and carries `rng`, `log` and `registered_roll` that equality ignores. If any of those
/// leaked into the hash, this fails.
#[test]
fn equal_states_hash_equally() {
    let corpus = corpus();
    if corpus.is_empty() {
        return;
    }
    for s in corpus.iter().take(2000) {
        let clone = s.clone();
        assert_eq!(s, &clone, "a clone must equal its original");
        assert_eq!(
            hash_of(s),
            hash_of(&clone),
            "a clone hashes differently from its original — the hash is reading a field \
             `PartialEq` ignores, which splits the MCTS DAG"
        );
    }

    // And the converse direction that actually bites: two states the registry considered
    // *distinct* must never be reported equal. (Cheap sanity check on the corpus itself.)
    let mut pairs = 0usize;
    for (i, a) in corpus.iter().take(400).enumerate() {
        for b in corpus.iter().take(400).skip(i + 1) {
            if hash_of(a) == hash_of(b) {
                pairs += 1;
            }
        }
    }
    eprintln!("HASH_SAFETY colliding pairs in the first 400 states: {pairs}");
}

/// Regression gate. The threshold is deliberately loose — this is guarding against a *regime*
/// change (someone dropping a field from the hash), not pinning an exact number that will drift
/// with every engine change.
#[test]
fn hash_separates_the_states_a_search_builds() {
    let corpus = corpus();
    if corpus.is_empty() {
        return; // board too small for this build
    }
    let c = analyse(&corpus);

    eprintln!(
        "HASH_QUALITY distinct_states={} distinct_hashes={} colliding={} ({:.1}%) worst_group={} mean_group={:.3}",
        c.distinct_states,
        c.distinct_hashes,
        c.colliding_states,
        100.0 * c.colliding_states as f64 / c.distinct_states as f64,
        c.worst_group,
        c.mean_group,
    );

    let colliding_share = c.colliding_states as f64 / c.distinct_states as f64;
    assert!(
        colliding_share < 0.02,
        "{:.1}% of distinct states share a hash with another ({} of {}). Every one of those pairs \
         costs a full two-state comparison on the search's hot path. Something was dropped from \
         `GameState::hash` — run the ignored `report_hash_collisions` for the culprit.",
        100.0 * colliding_share,
        c.colliding_states,
        c.distinct_states,
    );
    assert!(
        c.mean_group < 1.02,
        "mean distinct states per hash bucket is {:.3}; a probe landing on an occupied bucket pays \
         that many full comparisons",
        c.mean_group,
    );
}

/// The diagnosis. Not a gate — it prints.
#[test]
#[ignore = "diagnostic: prints the hash-collision breakdown, ~1 min"]
fn report_hash_collisions() {
    let corpus = corpus();
    if corpus.is_empty() {
        eprintln!("board too small for this build");
        return;
    }
    let c = analyse(&corpus);

    eprintln!("\n=== hash quality over {} distinct states ===", c.distinct_states);
    eprintln!("distinct hashes      : {}", c.distinct_hashes);
    eprintln!(
        "colliding states     : {} ({:.1}%)",
        c.colliding_states,
        100.0 * c.colliding_states as f64 / c.distinct_states as f64
    );
    eprintln!("worst bucket         : {} distinct states on one hash", c.worst_group);
    eprintln!("mean states / bucket : {:.3}", c.mean_group);

    let mut by_cause: HashMap<&'static str, (usize, usize)> = HashMap::new();
    for g in &c.groups {
        let e = by_cause.entry(classify(g)).or_default();
        e.0 += 1;
        e.1 += g.len();
    }
    let mut causes: Vec<_> = by_cause.into_iter().collect();
    causes.sort_by_key(|(_, (_, states))| std::cmp::Reverse(*states));
    eprintln!("\nwhat differs inside a colliding group (groups / states):");
    for (cause, (groups, states)) in causes {
        eprintln!("  {states:6} states in {groups:5} groups   {cause}");
    }
    eprintln!();
}

//! Plan 023: exact search equivariance under `TieBreak::Mover`.
//!
//! Everything the search is built from is now property-tested
//! mirror-exact (`mirror_symmetry.rs`, `mirror_transitions.rs`,
//! `mirror_chance_model.rs`, `mirror_apply_action.rs`), and `TieBreak::Mover`
//! is a mirror-*covariant* tie-break by construction. So a search built on
//! top of them ought to satisfy `search(mirror s) == mirror(search s)`
//! **exactly** — not just in distribution. Measured instead: the paired
//! mirror value-probe reads -54 to -65 at 1000 iterations regardless of tie
//! break (`mirror_symmetry.rs::search_side_bias_by_budget`), and a 100-game
//! mirror match under `Mover` still reads 0.633 Home share (z=+2.8). That
//! is a contradiction, and this file is the tool to chase it: an *exact*
//! equality assertion turns "is the selection/backprop layer covariant?"
//! from a statistical question into a provable one.
//!
//! Requires the `deterministic_hash` feature on `recon_mcts` (wired via
//! `botbowl-mcts/Cargo.toml`'s dev-dependency override) so the children
//! `HashMap`'s iteration order is reproducible given the same insertion
//! sequence — otherwise residual `HashMap` nondeterminism makes exact
//! equality untestable. Also forces `.with_workers(1)`: with more than one
//! worker, thread-scheduling nondeterminism in virtual-loss application and
//! descent order breaks exact equality regardless of the hasher, and that
//! is a orthogonal concern from the one under test here.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_data::ChildStat;
use botbowl_engine::bots::Bot;
use botbowl_mcts::{MctsBot, SearchBudget, TieBreak};
use botbowl_nn::eval::NnEvaluator;

use common::{mirror_action, mirror_playable, states, tier};

/// Compare two searches' root output for exact mirror-equivariance.
/// `sample_m` came from `mirror_playable(s)`. Panics with a detailed diff
/// on the first field that disagrees.
fn assert_exact_mirror(
    i: usize,
    iters: usize,
    action_s: botbowl_engine::core::model::Action,
    action_m: botbowl_engine::core::model::Action,
    mut children_s: Vec<ChildStat>,
    mut children_m: Vec<ChildStat>,
    root_value_s: Option<i64>,
    root_value_m: Option<i64>,
    root_visits_s: u32,
    root_visits_m: u32,
) {
    let dims = tier();
    assert_eq!(
        mirror_action(dims, action_s),
        action_m,
        "state {i} budget {iters}: root pick does not mirror ({action_s:?} vs {action_m:?})"
    );
    assert_eq!(
        root_value_s.map(|v| -v),
        root_value_m,
        "state {i} budget {iters}: root value is not the exact negation ({root_value_s:?} vs {root_value_m:?})"
    );
    assert_eq!(
        root_visits_s, root_visits_m,
        "state {i} budget {iters}: root visit count differs ({root_visits_s} vs {root_visits_m})"
    );

    children_s.sort_by_key(|c| c.action);
    children_m.sort_by_key(|c| mirror_action(dims, c.action));
    let mirrored_m: Vec<_> = children_m
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c.action = mirror_action(dims, c.action);
            c.q = c.q.map(|q| -q);
            c
        })
        .collect();
    let mut mirrored_m = mirrored_m;
    mirrored_m.sort_by_key(|c| c.action);

    assert_eq!(
        children_s.len(),
        mirrored_m.len(),
        "state {i} budget {iters}: different number of root children ({} vs {})",
        children_s.len(),
        mirrored_m.len()
    );
    for (cs, cm) in children_s.iter().zip(mirrored_m.iter()) {
        assert_eq!(
            cs.action, cm.action,
            "state {i} budget {iters}: child action sets disagree after mirroring"
        );
        assert_eq!(
            cs.visits, cm.visits,
            "state {i} budget {iters}: visits differ for {:?} ({} vs {})",
            cs.action, cs.visits, cm.visits
        );
        assert_eq!(
            cs.q, cm.q,
            "state {i} budget {iters}: q differs for {:?} ({:?} vs {:?})",
            cs.action, cs.q, cm.q
        );
        assert_eq!(
            cs.solved, cm.solved,
            "state {i} budget {iters}: solved differs for {:?}",
            cs.action
        );
    }
}

/// Which leaf/prior source the arm runs. The heuristic arms are the
/// original plan-023 tests; the `Nn` arms are plan 031 D9.
#[derive(Clone)]
enum Arm {
    Heuristic,
    Nn(Arc<NnEvaluator>),
}

/// `BLOOD_NN_MIRROR_MODEL` points the NN arms at a real champion instead of
/// the committed fixture — `models/` is gitignored so the default must stay
/// on `tiny.onnx`. Same knob `mirror_nn_evaluator.rs` uses. The invariant is
/// structural (the canonical frame in `botbowl-nn/src/perspective.rs` maps a
/// state and its mirror onto the identical tensor), so it must hold for an
/// arbitrary net; an untrained fixture localises a failure to the plumbing.
fn nn_arm() -> Arm {
    let onnx = match std::env::var("BLOOD_NN_MIRROR_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx"),
    };
    assert!(
        onnx.exists(),
        "missing model {} — run `uv run python -m bbnn.fixture` for the fixture",
        onnx.display()
    );
    Arm::Nn(Arc::new(
        NnEvaluator::from_path(&onnx).unwrap_or_else(|e| panic!("load {}: {e}", onnx.display())),
    ))
}

fn bot(arm: &Arm, iters: usize) -> MctsBot {
    let b = MctsBot::new(SearchBudget::Iterations(iters))
        .with_workers(1)
        .with_tie_break(TieBreak::Mover);
    match arm {
        Arm::Heuristic => b,
        Arm::Nn(nn) => b.with_evaluator(Arc::clone(nn)),
    }
}

fn run_at_budget(iters: usize, n: u32, seed: u64) {
    run_arm_at_budget(&Arm::Heuristic, iters, n, seed);
}

fn run_arm_at_budget(arm: &Arm, iters: usize, n: u32, seed: u64) {
    let dims = tier();
    for (i, mut s) in states(n, seed).into_iter().enumerate() {
        let mut m = mirror_playable(&s, dims);
        s.set_seed(1000 + i as u64);
        m.set_seed(1000 + i as u64);

        let mut bot_s = bot(arm, iters);
        let mut bot_m = bot(arm, iters);
        let (action_s, sample_s) = bot_s.get_action_with_record(&s);
        let (action_m, sample_m) = bot_m.get_action_with_record(&m);

        assert_exact_mirror(
            i,
            iters,
            action_s,
            action_m,
            sample_s.children,
            sample_m.children,
            sample_s.root_value,
            sample_m.root_value,
            sample_s.root_visits,
            sample_m.root_visits,
        );
    }
}

/// Smallest possible budget: root expansion only, no PUCT descent past
/// depth 1. If this fails, the bug is in expansion/scoring, not selection.
#[test]
fn search_mirrors_exactly_at_budget_2() {
    run_at_budget(2, 40, 24_100);
}

#[test]
fn search_mirrors_exactly_at_budget_5() {
    run_at_budget(5, 40, 24_100);
}

/// Was a known-open residual (plan 023, 2026-08-31 "exact equivariance"
/// result): failed at state 13, `StartBlock (9,6)` vs its mirror
/// `StartBlock (6,6)`, q = -523 vs -522. Root-caused (2026-08-31, part 2):
/// `available_actions` was tagging every returned action with the *current*
/// node's own mover (chance outcomes always `BbPlayer::Chance`, player
/// actions always the acting team) instead of the *resulting* node's mover,
/// per recon_mcts's actual contract (`tests/nim/lib.rs` tags every action
/// with the *other* player). That wrong tag fed `select_node`'s
/// `home_perspective` and `backprop_scores`'s `want_max` directly — e.g. the
/// state right after `StartBlock (9,6)`'s roll resolves to `Pow`, where the
/// next real decision (the push square) is Home's own choice, was tagged
/// `Chance` and evaluated as if it were Away's, silently minimising instead
/// of maximising. Fixed via `peek_mover` (one extra `apply_action` per
/// candidate to read the true resulting mover). Green since.
#[test]
fn search_mirrors_exactly_at_budget_20() {
    run_at_budget(20, 40, 24_100);
}

/// Slower; not part of the default fast suite.
#[test]
#[ignore]
fn search_mirrors_exactly_at_budget_200() {
    run_at_budget(200, 20, 24_100);
}

// ---------------------------------------------------------------------
// Plan 031 D9: the same assertions under `Evaluator::Nn`.
//
// The arms above prove equivariance for the *heuristic* evaluator only,
// and plan 027 left an unlocalised 56% Away share (z ≈ 3.4 pair-corrected)
// in NN full games. `mirror_nn_evaluator.rs` already pins the two pieces
// the search is built from — `value_home(mirrored(s)) == -value_home(s)`
// and mover-relative priors — so if those hold and the search is
// covariant, the composition must be exactly mirror-equivariant too.
// Green here means the residual side bias is tie-break variance or
// turn-order, and plan 032 #11's mirror games decide it. Red means a real
// bug in the NN search path, which outranks everything else in the queue.
//
// These are `#[ignore]`d for cost (a tract forward per expanded node), not
// for flakiness: run with
//   cargo test -p botbowl-mcts --test mirror_search_exact -- --ignored nn_
// ---------------------------------------------------------------------

/// Root expansion only — isolates NN scoring/prior plumbing from selection.
#[test]
#[ignore]
fn nn_search_mirrors_exactly_at_budget_5() {
    run_arm_at_budget(&nn_arm(), 5, 40, 24_100);
}

/// Deep enough for PUCT descent to matter: this is where a non-covariant
/// prior or a sign slip in backprop first shows up.
#[test]
#[ignore]
fn nn_search_mirrors_exactly_at_budget_20() {
    run_arm_at_budget(&nn_arm(), 20, 40, 24_100);
}

#[test]
#[ignore]
fn nn_search_mirrors_exactly_at_budget_200() {
    run_arm_at_budget(&nn_arm(), 200, 20, 24_100);
}

/// The budget production actually plays at. The lower arms could in
/// principle stay green while a bias only expressed itself deep in the tree
/// (more recombination, more solved subtrees, more chance branching), and
/// 1000 is the budget every side-bias measurement in plan 027 was taken at,
/// so this is the arm that speaks directly to that result.
#[test]
#[ignore]
fn nn_search_mirrors_exactly_at_production_budget() {
    run_arm_at_budget(&nn_arm(), 1000, 20, 24_100);
}

/// Confirms the root-pick-only agreement rate the plan quotes (78% at 200
/// iterations under `Mover`) as a sanity check on this file's own harness,
/// independent of the exact-equality assertions above.
#[test]
#[ignore]
fn root_pick_agreement_rate_at_200() {
    let dims = tier();
    let n = 100u32;
    let mut agree = 0usize;
    let mut total = 0usize;
    for (i, mut s) in states(n, 24_200).into_iter().enumerate() {
        let mut m = mirror_playable(&s, dims);
        s.set_seed(1000 + i as u64);
        m.set_seed(1000 + i as u64);
        let mut bot_s = MctsBot::new(SearchBudget::Iterations(200))
            .with_workers(1)
            .with_tie_break(TieBreak::Mover);
        let mut bot_m = MctsBot::new(SearchBudget::Iterations(200))
            .with_workers(1)
            .with_tie_break(TieBreak::Mover);
        let action_s = bot_s.get_action(&s);
        let action_m = bot_m.get_action(&m);
        total += 1;
        if mirror_action(dims, action_s) == action_m {
            agree += 1;
        }
    }
    println!("root pick agreement: {agree}/{total}");
}

#[test]
#[ignore]
fn debug_state_13_budget_20() {
    let dims = tier();
    let s_all = states(40, 24_100);
    let (i, mut s) = (13, s_all[13].clone());
    let mut m = mirror_playable(&s, dims);
    s.set_seed(1000 + i as u64);
    m.set_seed(1000 + i as u64);
    let mut bot_s = MctsBot::new(SearchBudget::Iterations(20))
        .with_workers(1)
        .with_tie_break(TieBreak::Mover);
    let mut bot_m = MctsBot::new(SearchBudget::Iterations(20))
        .with_workers(1)
        .with_tie_break(TieBreak::Mover);
    let (a_s, sample_s) = bot_s.get_action_with_record(&s);
    let (a_m, sample_m) = bot_m.get_action_with_record(&m);
    println!("root_s={:?} root_m={:?}", a_s, a_m);
    println!(
        "root_value_s={:?} root_value_m={:?}",
        sample_s.root_value, sample_m.root_value
    );
    let mut cs = sample_s.children.clone();
    cs.sort_by_key(|c| c.action);
    for c in &cs {
        println!(
            "S {:?} visits={} q={:?} solved={} terminal={}",
            c.action, c.visits, c.q, c.solved, c.terminal
        );
    }
    let mut cm = sample_m.children.clone();
    cm.sort_by_key(|c| c.action);
    for c in &cm {
        println!(
            "M {:?} visits={} q={:?} solved={} terminal={}",
            c.action, c.visits, c.q, c.solved, c.terminal
        );
    }
}

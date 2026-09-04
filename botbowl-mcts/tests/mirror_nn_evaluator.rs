//! Mirror-invariance property tests for the **NN** evaluator (plan 027).
//!
//! Plan 023 property-tested the whole heuristic pipeline for mirror
//! invariance — `leaf_score` antisymmetry, mover-relative priors and
//! pruning, the chance model, `apply_action`, and exact search equality.
//! Every one of those tests runs the *scripted* evaluator. The neural
//! network was never covered, and it is the one component plan 027's
//! mirrors implicate: a true heuristic mirror splits Home/Away 50.5%
//! (n=107, z=+0.10) while a true `nn-value` mirror on the same harness
//! reads 61.1% Away.
//!
//! The invariant should hold **exactly**, not approximately, and the
//! reason is worth stating because it makes the test strict rather than
//! statistical. `perspective.rs` canonicalises every encoded board into
//! the frame where *the mover attacks `x = 1`*: `Home` verbatim, `Away`
//! mirrored across the vertical axis. `GameState::mirrored()` reflects
//! the board and swaps the teams, so it also swaps which team is to
//! move — and canonicalisation maps the mirrored state back onto the
//! *identical* tensor. Same input, same network, same mover-centric
//! scalar. Only the Home-centric sign differs:
//!
//! ```text
//! value_home(mirrored(s)) == -value_home(s)
//! ```
//!
//! This is also why the trainer augments in y only (`data.py:18`: "The
//! x-mirror is NOT [applied]") — x-symmetry is supposed to be structural,
//! not learned. If these tests fail, that structural guarantee is broken
//! and no amount of training will fix it.
//!
//! Uses the committed `tiny.onnx` fixture: an untrained seeded net. That
//! is the right choice here — the invariant is a property of the
//! *encoding and perspective plumbing*, not of the weights, so it must
//! hold for an arbitrary network. A failure on an untrained net localises
//! the bug to the plumbing immediately.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use botbowl_engine::core::model::Action as EngineAction;
use botbowl_nn::eval::NnEvaluator;

use common::{mirror_action, states, tier};

fn evaluator() -> Arc<NnEvaluator> {
    // `BLOOD_NN_MIRROR_MODEL` points these at a real champion instead of
    // the fixture. `models/` is gitignored so CI must stay on the fixture,
    // but the invariant is structural and a trained net is the case we
    // actually care about — run it by hand when investigating a side bias:
    //   BLOOD_NN_MIRROR_MODEL=models/bbnet_14x7_gen03.onnx cargo test ...
    let onnx = match std::env::var("BLOOD_NN_MIRROR_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx"),
    };
    assert!(
        onnx.exists(),
        "missing model {} — run `uv run python -m bbnn.fixture` for the fixture",
        onnx.display()
    );
    Arc::new(NnEvaluator::from_path(&onnx).unwrap_or_else(|e| panic!("load {}: {e}", onnx.display())))
}

/// The core claim: the Home-centric NN value flips sign exactly under
/// reflection + team swap. This is the NN counterpart of
/// `mirror_symmetry::leaf_score_is_antisymmetric_under_mirroring`.
#[test]
fn nn_value_is_antisymmetric_under_mirroring() {
    let nn = evaluator();
    let mut worst: Option<(usize, i64, i64)> = None;
    for (i, s) in states(200, 27_000).into_iter().enumerate() {
        let v = nn.value_home_i64(&s);
        let vm = nn.value_home_i64(&s.mirrored());
        if vm != -v {
            let gap = (vm + v).abs();
            if worst.is_none_or(|(_, pv, pvm)| gap > (pvm + pv).abs()) {
                worst = Some((i, v, vm));
            }
        }
    }
    if let Some((i, v, vm)) = worst {
        panic!(
            "NN value is not antisymmetric under mirroring.\n\
             worst case: state {i}: value_home(s) = {v}, value_home(mirrored(s)) = {vm}, \
             expected {}. Residual {}.\n\
             The canonical frame in botbowl-nn/src/perspective.rs is supposed to make a state \
             and its mirror encode to the identical tensor, so this is a plumbing bug, not a \
             weights issue — the fixture net is untrained.",
            -v,
            vm + v
        );
    }
}

/// Doubling the mirror must be the identity for the value too, otherwise
/// the test above could pass against a transform that loses information.
#[test]
fn nn_value_survives_a_double_mirror() {
    let nn = evaluator();
    for (i, s) in states(60, 27_001).into_iter().enumerate() {
        assert_eq!(
            nn.value_home_i64(&s.mirrored().mirrored()),
            nn.value_home_i64(&s),
            "state {i}: double mirror changed the NN value"
        );
    }
}

/// Priors must be mover-relative: the prior the net assigns an action
/// should equal the prior it assigns that action's mirror image in the
/// mirrored state. A violation is the same shape as the `ScriptedBot`
/// touchback bug plan 023 found — one rule quietly helping one side.
#[test]
fn nn_priors_are_mover_relative() {
    let nn = evaluator();
    let dims = tier();
    let mut checked = 0usize;
    for (i, s) in states(120, 27_002).into_iter().enumerate() {
        let actions: Vec<EngineAction> = s.get_all_actions();
        if actions.is_empty() {
            continue;
        }
        let m = s.mirrored();
        let mirrored_actions: Vec<EngineAction> = actions.iter().map(|a| mirror_action(dims, *a)).collect();
        let m_legal: Vec<EngineAction> = m.get_all_actions();
        // Only compare when the mirror really does map the legal set onto
        // itself; mirror_symmetry.rs already pins that separately, and a
        // mismatch there is that test's failure to report, not this one's.
        if !mirrored_actions.iter().all(|a| m_legal.contains(a)) {
            continue;
        }
        let p = nn.priors(&s, &actions);
        let pm = nn.priors(&m, &mirrored_actions);
        for (k, (a, b)) in p.iter().zip(pm.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "state {i}, action {k} ({:?}): prior {a} but mirrored prior {b}",
                actions[k]
            );
        }
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} states were comparable — the test is not exercising much");
}

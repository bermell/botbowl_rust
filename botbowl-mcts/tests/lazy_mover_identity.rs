//! Plan 035 T2: the lazy-mover-tag refactor must be a **no-op on search
//! output**.
//!
//! Plan 023 tagged each action with the resulting node's mover by running a
//! full `apply_action` per candidate (`peek_mover`) at enumeration time. Plan
//! 035 moves that computation to `materialize_placeholder`, where the true
//! child state is already in hand. The two tags are identical by construction
//! — `player_for_state(apply_action(parent, a))` either way, given that
//! `apply_action` is pure and deterministic (which `mirror_apply_action.rs`
//! pins, see T3). So the hash inputs, node identity, registry hits and
//! `deterministic_hash` iteration order are all preserved bit-for-bit, and
//! anything less than byte equality of the root read-out is a bug.
//!
//! Procedure: generate and commit the goldens on master, do the refactor,
//! require these to pass unchanged.
//!
//! Same reproducibility setup as `mirror_search_exact.rs`: the
//! `deterministic_hash` feature on `recon_mcts` (wired via this crate's
//! dev-dependency override), `.with_workers(1)` (thread scheduling breaks
//! exact equality regardless of the hasher), `TieBreak::Mover`, and `tier()`
//! pinned to 16x9 so the goldens are not board-size dependent.
//!
//! Regenerate with `BLESS=1`:
//!   BLESS=1 cargo test -p botbowl-mcts --test lazy_mover_identity
//!   BLESS=1 cargo test -p botbowl-mcts --test lazy_mover_identity -- --ignored
//!
//! **Blessing is a reviewable event.** Byte-exact root output over 40 states
//! is a function of every prior, pruning rule and heuristic constant — exactly
//! what this repo's current focus keeps changing. The default-on arm is
//! deliberately scoped to `Heuristic` at one budget to keep the churn low; the
//! full matrix is `#[ignore]`d and exists as the one-shot gate for plan 035.
//!
//! **`lazy_mover_goldens_full.txt` is currently STALE.** Its NN arms were
//! blessed against the C=103 encoder; the tensor layout has since changed
//! repeatedly (unpaired per-player planes, the endzone planes, the path-
//! probability plane — C=59), so the
//! `#[ignore]`d full matrix will fail until it is re-blessed. The default-on
//! heuristic arm is unaffected and up to date: it never touches the encoder,
//! and the random-start states it runs on have not moved. Re-bless the full
//! matrix once the encoder schema settles — it costs ~14 minutes.
//!
//! The NN arms are additionally **only reliable on one machine**: the ONNX
//! runtime's intra-op threading and kernel selection can differ per host, so a
//! committed NN golden may fail elsewhere through no fault of the search.
//! Generate before and after on the same machine in the same session.

mod common;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use botbowl_data::ChildStat;
use botbowl_mcts::{MctsBot, SearchBudget, TieBreak};
use botbowl_nn::eval::NnEvaluator;

use common::states;

const SEED: u64 = 35_100;
const N_STATES: u32 = 40;

#[derive(Clone)]
enum Arm {
    Heuristic,
    Nn(Arc<NnEvaluator>),
}

impl Arm {
    fn label(&self) -> &'static str {
        match self {
            Arm::Heuristic => "heuristic",
            Arm::Nn(_) => "nn-tiny",
        }
    }
}

/// Same fixture `mirror_search_exact.rs` uses — `models/` is gitignored, so
/// the committed default has to be the tiny fixture net.
fn nn_arm() -> Arm {
    let onnx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../botbowl-nn/tests/fixtures/tiny.onnx");
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

/// One `(state, budget, arm)` cell, rendered as sorted plain text so a diff
/// names the exact child that moved. Plain text rather than JSON so the test
/// needs no serialisation dependency and a failure is readable in the diff.
fn render_cell(arm: &Arm, iters: usize, i: usize, s: &botbowl_engine::core::gamestate::GameState, out: &mut String) {
    let mut s = s.clone();
    s.set_seed(1000 + i as u64);
    let mut b = bot(arm, iters);
    let (action, sample) = b.get_action_with_record(&s);

    let _ = writeln!(
        out,
        "arm={} budget={iters} state={i} pick={action:?} root_value={:?} root_visits={}",
        arm.label(),
        sample.root_value,
        sample.root_visits,
    );
    let mut children: Vec<ChildStat> = sample.children;
    // Sort by action so a `Q = None` ordering difference among never-visited
    // children cannot masquerade as a real diff — and so a real diff points
    // at a named action.
    children.sort_by_key(|c| c.action);
    for c in &children {
        let _ = writeln!(
            out,
            "  {:?} visits={} q={:?} prior={:?} solved={} terminal={}",
            c.action, c.visits, c.q, c.prior, c.solved, c.terminal
        );
    }
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

/// `BLESS=1` writes `actual` to the golden and passes; otherwise compare and
/// report the first differing line.
fn check_or_bless(path: &Path, actual: &str) {
    if std::env::var("BLESS").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create tests/data");
        std::fs::write(path, actual).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        eprintln!("BLESSED {} ({} bytes)", path.display(), actual.len());
        return;
    }
    let expected = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing golden {} ({e}) — generate it on master with \
             BLESS=1 cargo test -p botbowl-mcts --test lazy_mover_identity",
            path.display()
        )
    });
    if expected == actual {
        return;
    }
    let first_diff = expected
        .lines()
        .zip(actual.lines())
        .enumerate()
        .find(|(_, (e, a))| e != a)
        .map(|(n, (e, a))| format!("line {}:\n  golden: {e}\n  actual: {a}", n + 1))
        .unwrap_or_else(|| {
            format!(
                "line counts differ: golden {} lines, actual {} lines",
                expected.lines().count(),
                actual.lines().count()
            )
        });
    panic!(
        "search output moved — plan 035's refactor must be a no-op on search output.\n\
         golden: {}\n{first_diff}\n\
         If this is an intended capability change, re-bless with BLESS=1 and review the diff.",
        path.display()
    );
}

fn run_matrix(arms: &[Arm], budgets: &[usize], name: &str) {
    let corpus = states(N_STATES, SEED);
    let mut out = String::new();
    for arm in arms {
        for &iters in budgets {
            for (i, s) in corpus.iter().enumerate() {
                render_cell(arm, iters, i, s, &mut out);
            }
        }
    }
    check_or_bless(&golden_path(name), &out);
}

/// The persistent arm: `Heuristic` at one budget. Cheap enough for the
/// default suite, and narrow enough that a bless is a reviewable event.
#[test]
fn search_output_unchanged_heuristic_200() {
    run_matrix(&[Arm::Heuristic], &[200], "lazy_mover_goldens.txt");
}

/// The plan-035 one-shot gate: both budgets x both evaluators. `#[ignore]`d
/// for cost (a tract forward per expanded node) and because the NN arms are
/// only reliable on the machine that blessed them.
#[test]
#[ignore = "plan 035 one-shot gate — slow (NN forwards); run with --ignored"]
fn search_output_unchanged_full_matrix() {
    run_matrix(&[Arm::Heuristic, nn_arm()], &[200, 1000], "lazy_mover_goldens_full.txt");
}

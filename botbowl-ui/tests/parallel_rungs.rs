//! `eval --parallel-games` must change only *when* rung games run, never
//! what they produce (plan 024 Stage 4b).
//!
//! Unlike `dataset`, eval **can** assert exact equivalence, and does. The
//! reason is that `play_game` seeds both bots from the game index
//! (`candidate.set_seed`/`opponent.set_seed`), and `ScriptedBot` and
//! `RandomBot` actually honour a seed — so a scripted-vs-random rung is a
//! pure function of `(seed, game index)`. `dataset` could not do this
//! because it always drives `MctsBot`, which plan 020 records as not
//! seed-reproducible at all (`recon_mcts` iterates a std `HashMap`).
//!
//! Deliberately using the deterministic bots rather than the MCTS ones is
//! what makes a failure here mean "the parallel plumbing is wrong" instead
//! of "the search tie-broke differently".

use std::collections::BTreeSet;
use std::process::Command;

const GAMES: u32 = 12;
const SEED: u64 = 555;

/// Only meaningful on a tier where a full game is quick. See
/// `parallel_games.rs` for the same guard and the reasoning behind it.
fn small_tier_or_skip(test: &str) -> bool {
    let w = botbowl_engine::core::model::WIDTH;
    if w > 20 {
        eprintln!("skipped {test}: board is {w} wide; run at the 14x7 tier");
        return false;
    }
    true
}

/// Run one scripted-vs-random rung and return `(per_game_rows, report_json)`.
fn run_rung(parallel: u32, tag: &str, dir: &std::path::Path) -> (Vec<String>, serde_json::Value) {
    let exe = env!("CARGO_BIN_EXE_botbowl-ui");
    let per_game = dir.join(format!("{tag}.jsonl"));
    let report = dir.join(format!("{tag}.json"));
    let out = Command::new(exe)
        .args([
            "eval",
            "--candidate-bot",
            "scripted",
            "--rungs",
            "random",
            "--skip-lectures",
            "--games",
            &GAMES.to_string(),
            "--seed",
            &SEED.to_string(),
            "--parallel-games",
            &parallel.to_string(),
            "--per-game-out",
            per_game.to_str().unwrap(),
            "--out",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("run eval");
    assert!(
        out.status.success(),
        "eval --parallel-games {parallel} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut rows: Vec<String> = std::fs::read_to_string(&per_game)
        .expect("per-game jsonl")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    // Sort: line order is the one thing parallelism is allowed to change.
    rows.sort();
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).expect("report")).expect("report json");
    (rows, report)
}

#[test]
fn parallel_rungs_match_sequential_exactly() {
    if !small_tier_or_skip("parallel_rungs_match_sequential_exactly") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("bb-parallel-rungs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");

    let (seq_rows, seq_report) = run_rung(1, "seq", &dir);
    let (par_rows, par_report) = run_rung(4, "par", &dir);

    assert_eq!(seq_rows.len(), GAMES as usize, "sequential wrote {} rows, expected {GAMES}", seq_rows.len());
    assert_eq!(
        par_rows.len(),
        seq_rows.len(),
        "parallel wrote {} rows, sequential {} — a game was dropped or duplicated",
        par_rows.len(),
        seq_rows.len()
    );
    for (i, (a, b)) in seq_rows.iter().zip(&par_rows).enumerate() {
        assert_eq!(a, b, "rung game {i} differs between --parallel-games 1 and 4");
    }

    // Every game index appears exactly once — the guard on the shared
    // `next_game` hand-out.
    let indices: BTreeSet<u64> = par_rows
        .iter()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("row json")["game"].as_u64().unwrap())
        .collect();
    assert_eq!(indices, (0..GAMES as u64).collect::<BTreeSet<_>>(), "game indices are not 0..{GAMES}");

    // And the aggregate the promotion gate reads must be identical — the
    // guard on the `LadderRow` mutex. A lost increment shows up only here.
    assert_eq!(
        seq_report["ladder"][0], par_report["ladder"][0],
        "ladder row differs between sequential and parallel — a counter update was lost"
    );

    std::fs::remove_dir_all(&dir).ok();
}

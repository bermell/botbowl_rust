//! What `--parallel-games` must not break (plan 024 Stage 4).
//!
//! # Why this does not assert that parallel == sequential
//!
//! The obvious test — same seeds, `--parallel-games 1` vs `4`, assert
//! identical trajectories — is **not achievable for `dataset`**, and it is
//! worth writing down why so nobody adds it back. `dataset` always drives
//! `MctsBot` (`--evaluator heuristic` swaps the leaf evaluator, not the
//! bot), and plan 020 records that MctsBot games are not reproducible
//! from a seed at all: `recon_mcts` iterates a std `HashMap` whose order
//! is randomised per process, so near-tied PUCT decisions break
//! differently on every run. Measured here: two *sequential* runs of the
//! identical command produced 0 of 6 identical trajectories. A
//! parallel-vs-sequential equality assertion would therefore fail
//! whatever the parallel code did, and passing it would prove nothing.
//!
//! So this asserts the properties parallelism genuinely puts at risk,
//! which are about the plumbing rather than the search:
//!
//! * **Every game runs exactly once** — the guard on the `next_game`
//!   hand-out. A racy counter shows up as a dropped or duplicated seed.
//! * **No line is corrupted** — the guard on the writer `Mutex`.
//!   `serde_json` writes straight into the `BufWriter`, so an unguarded
//!   concurrent write interleaves *within* a line and the file stops
//!   being JSONL. This is the failure that would silently poison a
//!   corpus, and the only one a reader would not immediately notice.
//!
//! Distributional equivalence between a parallel corpus and a sequential
//! one is the plan's Stage-4 acceptance test and needs hundreds of games;
//! it does not belong in the unit suite.

use std::collections::BTreeSet;
use std::process::Command;

const GAMES: u32 = 12;
const SEED: u64 = 7700;

fn generate(parallel: u32, out: &str) -> String {
    let exe = env!("CARGO_BIN_EXE_botbowl-ui");
    let out_run = Command::new(exe)
        .args([
            "dataset",
            "--mode",
            "random-start",
            "--games",
            &GAMES.to_string(),
            "--seed",
            &SEED.to_string(),
            // Tiny budget: this test is about the plumbing, not the search.
            "--mcts-iters",
            "8",
            "--evaluator",
            "heuristic",
            "--truncate",
            "--parallel-games",
            &parallel.to_string(),
            "--out",
            out,
        ])
        .output()
        .expect("run dataset");
    assert!(
        out_run.status.success(),
        "dataset --parallel-games {parallel} failed: {}",
        String::from_utf8_lossy(&out_run.stderr)
    );
    std::fs::read_to_string(out).expect("read jsonl")
}

/// Each of the N games is written exactly once, and every line is intact
/// JSON — under enough concurrency to actually race the writer.
#[test]
fn parallel_games_writes_every_game_exactly_once_and_never_tears_a_line() {
    let dir = std::env::temp_dir().join(format!("bb-parallel-games-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("par.jsonl");

    let text = generate(6, path.to_str().unwrap());
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();

    assert_eq!(
        lines.len(),
        GAMES as usize,
        "wrote {} lines for {GAMES} games — a game was dropped or duplicated",
        lines.len()
    );

    let mut seeds = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        // A torn line is the writer-lock failure, and it presents exactly
        // here: `serde_json` interleaves two trajectories mid-object.
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {i} is not valid JSON ({e}) — concurrent writes tore it apart"));
        let seed = v["meta"]["seed"].as_u64().expect("meta.seed");
        assert!(seeds.insert(seed), "seed {seed} was written twice");
        assert!(
            !v["samples"].as_array().expect("samples array").is_empty(),
            "seed {seed} produced no samples"
        );
    }

    let expected: BTreeSet<u64> = (0..GAMES as u64).map(|g| SEED + g).collect();
    assert_eq!(seeds, expected, "the set of generated seeds is not {SEED}..{}", SEED + GAMES as u64);

    std::fs::remove_dir_all(&dir).ok();
}

/// `--parallel-games 1` must remain exactly today's code path: same game
/// count, same seeds, one line each.
#[test]
fn parallel_games_one_is_the_sequential_path() {
    let dir = std::env::temp_dir().join(format!("bb-parallel-games-seq-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("seq.jsonl");

    let text = generate(1, path.to_str().unwrap());
    let seeds: Vec<u64> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("valid JSON")["meta"]["seed"].as_u64().unwrap())
        .collect();

    // Sequentially, order is still game order — worth pinning, because it
    // is the one ordering guarantee parallelism gives up.
    let expected: Vec<u64> = (0..GAMES as u64).map(|g| SEED + g).collect();
    assert_eq!(seeds, expected, "sequential output is no longer in game order");

    std::fs::remove_dir_all(&dir).ok();
}

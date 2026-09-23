//! Does a more discriminating `GameState::hash` actually make the search faster?
//!
//! It is a real trade, not a free win. The deeper hash walks the whole procedure stack, both
//! player arrays and the offered action set on **every** node insert; what it buys back is the
//! full two-state comparisons a colliding probe used to pay, each of which clones two whole
//! `GameState`s. Which side wins is an empirical question about this game's state size and this
//! search's collision rate, so it gets measured rather than argued.
//!
//! Controlled on purpose: a fixed set of start positions, a fixed iteration budget, one worker.
//! That removes game-length variance, so the difference between two builds is the search itself
//! and not how long the games happened to run.
//!
//! ```sh
//! cargo test --release -p botbowl-mcts --test hash_bench -- --ignored --nocapture
//! ```
//!
//! Run it in a worktree at the old commit and again here, and compare the `HASH_BENCH` lines.

mod common;

use std::time::Instant;

use botbowl_engine::bots::Bot;
use botbowl_mcts::{MctsBot, SearchBudget};

use common::states;

/// Independent replicates. Each is a different set of start positions, so the spread across them
/// is the run-to-run variance a single number would hide.
const SEEDS: [u64; 5] = [43_201, 43_202, 43_203, 43_204, 43_205];
const N_STATES: u32 = 8;
const ITERS: usize = 2000;

struct Replicate {
    searches: u64,
    nodes: u64,
    probes: u64,
    eq_checks: u64,
    eq_rejects: u64,
    hits: u64,
    secs: f64,
}

fn run(seed: u64) -> Option<Replicate> {
    let corpus = states(N_STATES, seed);
    if corpus.is_empty() {
        return None;
    }
    // One bot per start position, as a game would have: the first decision has no tree to reuse
    // and the rest may. Warm the allocator with a throwaway search so the first replicate is not
    // penalised.
    let mut total = Replicate {
        searches: 0,
        nodes: 0,
        probes: 0,
        eq_checks: 0,
        eq_rejects: 0,
        hits: 0,
        secs: 0.0,
    };
    for s in &corpus {
        let mut bot = MctsBot::new(SearchBudget::Iterations(ITERS)).with_workers(1);
        let t0 = Instant::now();
        // Three consecutive decisions, so tree reuse is exercised the way a real turn does.
        let mut state = s.clone();
        for _ in 0..3 {
            if state.info.game_over || state.available_actions.team.is_none() {
                break;
            }
            let action = bot.get_action(&state);
            if state.step(action).is_err() {
                break;
            }
        }
        total.secs += t0.elapsed().as_secs_f64();
        let t = bot.telemetry();
        total.searches += t.searches;
        total.probes += t.recombination.probes;
        total.eq_checks += t.recombination.eq_checks;
        total.eq_rejects += t.recombination.eq_rejects;
        total.hits += t.recombination.hits;
        total.nodes += t.recombination.misses;
    }
    Some(total)
}

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

#[test]
#[ignore = "benchmark: 5 replicates x 8 positions x 3 decisions, ~2 min"]
fn hash_cost_and_benefit() {
    let mut reps = Vec::new();
    for seed in SEEDS {
        match run(seed) {
            Some(r) => reps.push(r),
            None => {
                eprintln!("board too small for this build");
                return;
            }
        }
    }

    let per = |f: &dyn Fn(&Replicate) -> f64| reps.iter().map(|r| f(r)).collect::<Vec<_>>();
    let (eq_per_probe, eq_sd) = mean_sd(&per(&|r| r.eq_checks as f64 / r.probes.max(1) as f64));
    let (reject, reject_sd) = mean_sd(&per(&|r| r.eq_rejects as f64 / r.eq_checks.max(1) as f64));
    let (hit, hit_sd) = mean_sd(&per(&|r| r.hits as f64 / r.probes.max(1) as f64));
    let (nps, nps_sd) = mean_sd(&per(&|r| r.nodes as f64 / r.secs));
    let (sps, sps_sd) = mean_sd(&per(&|r| r.searches as f64 / r.secs));

    // Per-replicate lines first. Both builds run the **identical** search — the hash changes
    // bucket layout, never node identity, which `total_nodes` confirms — so the honest comparison
    // between two builds is *paired* on the seed. Pooling replicates instead buries the effect
    // under the between-position variance, which is several times larger.
    eprintln!();
    for (seed, r) in SEEDS.iter().zip(&reps) {
        eprintln!(
            "HASH_PAIR seed={seed} nodes={} secs={:.4} nodes_per_sec={:.0} eq_per_probe={:.4}",
            r.nodes,
            r.secs,
            r.nodes as f64 / r.secs,
            r.eq_checks as f64 / r.probes.max(1) as f64,
        );
    }

    // One grep-able line per metric, mean over replicates with the sample SD — so a comparison
    // between two builds can say whether a difference is bigger than the noise.
    eprintln!("\n=== hash bench: {} replicates ===", reps.len());
    eprintln!("HASH_BENCH eq_per_probe   {eq_per_probe:8.4} +/- {eq_sd:.4}");
    eprintln!("HASH_BENCH eq_reject_rate {reject:8.4} +/- {reject_sd:.4}");
    eprintln!("HASH_BENCH recomb_hit_rate{hit:8.4} +/- {hit_sd:.4}");
    eprintln!("HASH_BENCH nodes_per_sec  {nps:8.0} +/- {nps_sd:.0}");
    eprintln!("HASH_BENCH searches_per_s {sps:8.2} +/- {sps_sd:.2}");
    let total_nodes: u64 = reps.iter().map(|r| r.nodes).sum();
    let total_secs: f64 = reps.iter().map(|r| r.secs).sum();
    eprintln!("HASH_BENCH total_nodes    {total_nodes} in {total_secs:.2}s\n");
}

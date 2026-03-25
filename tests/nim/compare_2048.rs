//! Compare average final scores: one-step heuristic vs MCTS (same schedule as `benchmark_2048`).
//!
//! Usage: `compare_2048 [num_games] [mcts_iterations_per_action] [base_seed]`
//! Defaults: 10 games, 500 MCTS iterations, base seed 0. Game `i` uses seed `base_seed + i`.

use recon_mcts_test_nim::play_2048::{self, DEFAULT_WARMUP_STEPS};

fn median(mut xs: Vec<i32>) -> f64 {
    xs.sort_unstable();
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2] as f64
    } else {
        (xs[n / 2 - 1] as f64 + xs[n / 2] as f64) / 2.0
    }
}

fn print_block(label: &str, scores: &[i32]) {
    let sum: i64 = scores.iter().map(|&x| x as i64).sum();
    let mean = sum as f64 / scores.len() as f64;
    let min = *scores.iter().min().unwrap();
    let max = *scores.iter().max().unwrap();
    let med = median(scores.to_vec());
    println!("{}", label);
    println!("  mean:   {:.1}", mean);
    println!("  median: {:.1}", med);
    println!("  min:    {}", min);
    println!("  max:    {}", max);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let num_games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let mcts_iterations: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(500);
    let base_seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    println!(
        "2048 comparison: {} games, {} MCTS steps/move (warmup {}), seeds {}..{}",
        num_games,
        mcts_iterations,
        DEFAULT_WARMUP_STEPS,
        base_seed,
        base_seed.saturating_add(num_games as u64 - 1)
    );
    println!();

    let mut heuristic_scores = Vec::with_capacity(num_games);
    let mut mcts_scores = Vec::with_capacity(num_games);

    for i in 0..num_games {
        let seed = base_seed.wrapping_add(i as u64);
        let h = play_2048::run_heuristic_game(seed);
        let m = play_2048::run_mcts_game(seed, mcts_iterations, DEFAULT_WARMUP_STEPS);
        heuristic_scores.push(h);
        mcts_scores.push(m);
        println!("game {:>3}: heuristic {:>6}  mcts {:>6}", i + 1, h, m);
    }

    println!();
    print_block(
        "Heuristic (one-step snake / empty / score)",
        &heuristic_scores,
    );
    println!();
    print_block(
        &format!(
            "MCTS ({} iters/move, {} warmup)",
            mcts_iterations, DEFAULT_WARMUP_STEPS
        ),
        &mcts_scores,
    );
}

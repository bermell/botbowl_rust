use recon_mcts_test_nim::play_2048::{self, DEFAULT_WARMUP_STEPS};
use recon_mcts_test_nim::test_mcts_2048::Game2048Dynamics;

fn run_single_game(mcts_iterations: usize) -> i32 {
    let seed = rand::random::<u64>();
    play_2048::run_mcts_game(
        seed,
        mcts_iterations,
        DEFAULT_WARMUP_STEPS,
        Game2048Dynamics::default(),
    )
}

fn run_benchmark(num_games: usize, mcts_iterations: usize) {
    println!(
        "Running benchmark with {} games and {} MCTS iterations per move",
        num_games, mcts_iterations
    );
    println!();

    let mut scores = Vec::new();

    for i in 1..=num_games {
        print!("Game {}: ", i);
        let score = run_single_game(mcts_iterations);
        scores.push(score);
        println!("{}", score);
    }

    println!();

    // Calculate statistics
    let sum: i32 = scores.iter().sum();
    let average = sum as f64 / num_games as f64;
    let min = scores.iter().min().copied().unwrap_or(0);
    let max = scores.iter().max().copied().unwrap_or(0);

    println!("Results:");
    println!("  Average score: {:.1}", average);
    println!("  Min score: {}", min);
    println!("  Max score: {}", max);
}

fn main() {
    run_benchmark(2, 500);
}

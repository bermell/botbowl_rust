use std::sync::atomic::AtomicU32;

use rand;
use recon_mcts::{GetState, SearchTree, Tree};

use recon_mcts_test_nim::game_2048::{Coord, Direction, Game2048, GameState};
use recon_mcts_test_nim::test_mcts_2048::ActionChance;

fn run_single_game(mcts_iterations: usize) -> u32 {
    // Pick a random starting square for the initial tile
    let random_row = rand::random::<u16>() % 4;
    let random_col = rand::random::<u16>() % 4;
    let initial_coord = Coord {
        row: random_row as usize,
        col: random_col as usize,
    };

    // Initialize the game with a single tile of value 2
    let mut game = Game2048::new_game(initial_coord, 2);

    // Play the game until it's done
    while game.state != GameState::Done {
        if game.state == GameState::WaitingForAction {
            // Reset tree for current game state
            let tree = Tree::new(game, GetState, (), game);

            // Run MCTS iterations to evaluate possible moves
            for _ in 0..mcts_iterations {
                tree.step();
            }

            // Get the best move based on MCTS evaluation
            if let Some(move_info) = tree.get_next_move_info() {
                let mut action_scores: Vec<(Direction, i32, AtomicU32)> = Vec::new();

                // Extract action scores from move info
                for (ac, move_info_item) in move_info {
                    if let ActionChance::Action(action) = ac {
                        if let Some(score_item) = move_info_item.score {
                            action_scores.push((action, score_item.score, score_item.visits));
                        }
                    }
                }

                // Find the best action based on highest score
                let best_action = action_scores
                    .iter()
                    .max_by(|a, b| a.1.cmp(&b.1))
                    .map(|(action, _, _)| *action);

                // Apply the best action
                if let Some(action) = best_action {
                    game.step_action(action);
                }
            }
        } else if game.state == GameState::WaitingForRandom {
            // Handle random tile spawn
            let chances = game.available_chance();
            if !chances.is_empty() {
                let rand_idx = rand::random::<u16>() % (chances.len() as u16);
                let (coord, value, _) = &chances[rand_idx as usize];
                game.step_random(*coord, *value);
            }
        }
    }

    game.score
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
    let sum: u32 = scores.iter().sum();
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

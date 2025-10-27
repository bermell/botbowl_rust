use recon_mcts::{GetState, SearchTree, Status, Tree};

use recon_mcts_test_nim::game_2048::{Coord, Game2048};
use recon_mcts_test_nim::test_mcts_2048::ActionChance;

fn run_single_game(mcts_iterations: usize) -> i32 {
    // Pick a random starting square for the initial tile
    let random_row = rand::random::<u16>() % 4;
    let random_col = rand::random::<u16>() % 4;
    let initial_coord = Coord {
        row: random_row as usize,
        col: random_col as usize,
    };

    // Initialize the game with a single tile of value 2
    let game = Game2048::new_game(initial_coord, 2);
    let tree = Tree::new(game, GetState, (), game);

    // Play the game until it's done
    loop {
        for _ in 0..50 {
            tree.step();
        }
        let next_move_info = tree.get_next_move_info();
        if next_move_info.is_none() {
            // Game is over
            println!("Game over (1)!");
            break;
        }
        let root_actions = next_move_info.unwrap();
        if root_actions.is_empty() {
            // No available actions, game over
            println!("Game over (2)!");
            break;
        }
        let first_action = root_actions[0].0;

        // TODO: determine if root node is a chance node or action node
        let next_action = match first_action {
            ActionChance::Action(_) => {
                // Run MCTS iterations to evaluate possible moves and select the best one
                for _ in 0..mcts_iterations {
                    tree.step();
                }
                match tree.best_action() {
                    Status::Action(best_action) => best_action,
                    _ => {
                        panic!("Expected best action to be an action node");
                    }
                }
            }
            ActionChance::Chance(_, _, _) => {
                // For chance nodes, we can just sample the next tile placement
                // we know all chance nodes have equal probability
                //
                let rand_idx = rand::random::<u16>() % (root_actions.len() as u16);
                root_actions[rand_idx as usize].0
            }
        };
        tree.apply_action(&next_action);
    }

    // Return the final score from the tree's final game state
    tree.get_root_info().score.unwrap().score
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

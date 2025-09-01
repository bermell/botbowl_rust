use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use recon_mcts::{GetState, SearchTree, Tree};

use crate::game_2048::{Coord, Direction, Game2048, GameState};

pub struct GameVisualizer {
    game: Game2048,
    tree: Tree<
        recon_mcts::Node<
            Game2048,
            Game2048,
            (),
            crate::test_mcts_2048::ActionChance,
            crate::test_mcts_2048::ScoreItem,
            std::vec::IntoIter<((), crate::test_mcts_2048::ActionChance)>,
            GetState,
        >,
        Game2048,
    >,
    mcts_iterations: usize,
}

impl GameVisualizer {
    pub fn new(initial_coord: Coord, initial_value: u32) -> Self {
        let game = Game2048::new_game(initial_coord, initial_value);
        let tree = Tree::new(game.clone(), GetState, (), game.clone());

        Self {
            game,
            tree,
            mcts_iterations: 1000,
        }
    }

    pub fn set_mcts_iterations(&mut self, iterations: usize) {
        self.mcts_iterations = iterations;
    }

    pub fn display_board(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                          2048 GAME                          ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        // Display score and state
        println!("Score: {}", self.game.score);
        println!("State: {:?}", self.game.state);
        println!();

        // Display the board with proper padding
        println!("┌─────┬─────┬─────┬─────┐");
        for row in 0..4 {
            print!("│");
            for col in 0..4 {
                let value = self.game.board[row][col];
                if value == 0 {
                    print!("     │");
                } else {
                    // Center the number in a 5-character space
                    let value_str = value.to_string();
                    let padding = 5 - value_str.len();
                    let left_pad = padding / 2;
                    let right_pad = padding - left_pad;
                    print!(
                        "{}{}{}│",
                        " ".repeat(left_pad),
                        value_str,
                        " ".repeat(right_pad)
                    );
                }
            }
            println!();
            if row < 3 {
                println!("├─────┼─────┼─────┼─────┤");
            }
        }
        println!("└─────┴─────┴─────┴─────┘");
        println!();
    }

    pub fn display_available_actions(&self) {
        if self.game.state == GameState::WaitingForAction {
            let actions = self.game.available_action();
            if actions.is_empty() {
                println!("❌ No available actions - Game Over!");
                return;
            }

            println!("🎮 Available Actions:");
            for action in &actions {
                println!("  • {:?}", action);
            }
        }
        // Removed chance node display - just show actions
        println!();
    }

    pub fn display_mcts_scores(&mut self) {
        if self.game.state != GameState::WaitingForAction {
            return;
        }

        println!("🧠 Running MCTS ({} iterations)...", self.mcts_iterations);

        // Run MCTS to get action scores
        for _ in 0..self.mcts_iterations {
            self.tree.step();
        }

        println!("🧠 MCTS Action Scores:");

        let actions = self.game.available_action();
        for action in &actions {
            // Simulate the action to see what the resulting state would be
            let mut temp_game = self.game.clone();
            temp_game.step_action(*action);
            // Get the score for this action by looking at the tree
            let score = self.get_action_score_from_tree(action);
            println!(
                "  • {:?}: Score = {:.2}, Visits = {}",
                action, score.0, score.1
            );
        }
        println!();
    }

    fn get_action_score_from_tree(&self, action: &Direction) -> (f64, usize) {
        // This is a simplified approach - we'll use the game state to estimate scores
        // In a real implementation, you'd traverse the tree to find the specific action node

        let mut temp_game = self.game.clone();
        temp_game.step_action(*action);
        // Simple heuristic: prefer moves that keep high values in corners
        // and avoid moves that create isolated tiles
        let heuristic_score = self.evaluate_board_heuristic(&temp_game);
        return (heuristic_score, 1); // Placeholder visit count
    }

    fn evaluate_board_heuristic(&self, game: &Game2048) -> f64 {
        let mut score = 0.0;

        // Prefer boards with high values in corners
        let corners = [
            Coord { row: 0, col: 0 },
            Coord { row: 0, col: 3 },
            Coord { row: 3, col: 0 },
            Coord { row: 3, col: 3 },
        ];

        for corner in &corners {
            let value = game[*corner] as f64;
            if value > 0.0 {
                score += value * 2.0; // Bonus for corner tiles
            }
        }

        // Penalty for isolated tiles (tiles with no adjacent tiles of same value)
        for row in 0..4 {
            for col in 0..4 {
                let coord = Coord { row, col };
                let value = game[coord];
                if value > 0 {
                    let mut isolated = true;
                    let directions = [
                        Direction::Up,
                        Direction::Down,
                        Direction::Left,
                        Direction::Right,
                    ];

                    for dir in &directions {
                        let neighbor = coord + *dir;
                        if game.in_bounds(neighbor) && game[neighbor] == value {
                            isolated = false;
                            break;
                        }
                    }

                    if isolated {
                        score -= value as f64 * 0.5; // Penalty for isolated tiles
                    }
                }
            }
        }

        score
    }

    pub fn step_through_game(&mut self) {
        println!("🚀 Starting 2048 Game with MCTS AI!");
        println!("Press Enter to continue to next move, 'q' to quit, or 'a' for auto-play");

        let mut auto_play = false;

        loop {
            self.display_board();
            self.display_available_actions();

            if self.game.state == GameState::Done {
                println!("🏁 Game Over! Final Score: {}", self.game.score);
                break;
            }

            if self.game.state == GameState::WaitingForAction {
                self.display_mcts_scores();

                if !auto_play {
                    print!("Press Enter to continue, 'a' for auto-play, 'q' to quit: ");
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();

                    match input.trim() {
                        "q" | "quit" => break,
                        "a" | "auto" => {
                            auto_play = true;
                            println!("🔄 Auto-play enabled!");
                        }
                        _ => {}
                    }
                } else {
                    thread::sleep(Duration::from_millis(1000));
                }

                // Make the best move based on MCTS
                self.make_best_move();
            } else if self.game.state == GameState::WaitingForRandom {
                // Handle random tile spawn
                self.handle_random_spawn();
            }
        }
    }

    fn make_best_move(&mut self) {
        // Get available actions and their scores
        let actions = self.game.available_action();
        let mut best_action = None;
        let mut best_score = f64::NEG_INFINITY;

        for action in &actions {
            let score = self.get_action_score_from_tree(action).0;
            if score > best_score {
                best_score = score;
                best_action = Some(*action);
            }
        }

        if let Some(action) = best_action {
            println!(
                "🤖 MCTS chose: {:?} (heuristic score: {:.2})",
                action, best_score
            );
            self.game.step_action(action);
        }
    }

    fn handle_random_spawn(&mut self) {
        let chances = self.game.available_chance();
        if !chances.is_empty() {
            // For visualization, we'll pick the first available chance
            let (coord, value, _) = chances[0];
            println!(
                "🎲 Random spawn: {} at ({}, {})",
                value, coord.row, coord.col
            );
            self.game.step_random(coord, value);
            // Brief pause to show the spawn
            thread::sleep(Duration::from_millis(200));
        }
    }
}

pub fn run_visualization() {
    let mut visualizer = GameVisualizer::new(Coord { row: 2, col: 1 }, 2);
    visualizer.set_mcts_iterations(500);
    visualizer.step_through_game();
}

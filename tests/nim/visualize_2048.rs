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
        self.display_board_with_action_table(None, None);
    }

    fn generate_board_lines(&self) -> Vec<String> {
        let mut board_lines = Vec::new();
        board_lines.push(format!("Score: {}", self.game.score));
        board_lines.push(String::new());
        board_lines.push(" ┌─────┬─────┬─────┬─────┐".to_string());

        for row in 0..4 {
            let mut line = String::new();
            line.push('│');
            for col in 0..4 {
                let value = self.game.board[row][col];
                if value == 0 {
                    line.push_str("     │");
                } else {
                    let value_str = value.to_string();
                    let padding = 5 - value_str.len();
                    let left_pad = padding / 2;
                    let right_pad = padding - left_pad;
                    line.push_str(&format!(
                        "{}{}{}│",
                        " ".repeat(left_pad),
                        value_str,
                        " ".repeat(right_pad)
                    ));
                }
            }
            board_lines.push(line);

            if row < 3 {
                board_lines.push("├─────┼─────┼─────┼─────┤".to_string());
            }
        }
        board_lines.push("└─────┴─────┴─────┴─────┘".to_string());

        board_lines
    }

    fn generate_action_table_lines(
        &mut self,
        action_scores: &[(Direction, f64, usize)],
        best_action: Option<Direction>,
    ) -> Vec<String> {
        let mut table_lines = Vec::new();

        table_lines.push(format!(
            "🧠 Running MCTS ({} iterations)...",
            self.mcts_iterations
        ));
        table_lines.push("┌──────────────┬──────────┬─────────┬─────────┐".to_string());
        table_lines.push("│    Action    │   Score  │  Visits │ Selected│".to_string());
        table_lines.push("├──────────────┼──────────┼─────────┼─────────┤".to_string());

        for (action, score, visits) in action_scores {
            let marker = if Some(*action) == best_action {
                "        ✓"
            } else {
                "         "
            };
            let action_str = format!("{:?}", action);
            table_lines.push(format!(
                "│ {:<12} │ {:>8.2} │ {:>7} │{}│",
                action_str, score, visits, marker
            ));
        }

        table_lines.push("└──────────────┴──────────┴─────────┴─────────┘".to_string());
        table_lines.push(String::new());

        table_lines
    }

    fn print_combined(
        &self,
        board_lines: Vec<String>,
        table_lines: Option<Vec<String>>,
        chosen_action: Option<Direction>,
    ) {
        println!("══════════════════════════════════════════════════════════════");

        if let Some(table_lines) = table_lines {
            let mut empty_line = String::new();
            for _ in 0..47 {
                empty_line.push(' ');
            }

            // Print lines side-by-side
            for i in 0..board_lines.len() {
                let board_line = board_lines.get(i).unwrap_or(&empty_line);

                if i < 2 {
                    // For score and blank line, just print the board line
                    println!("{}", board_line);
                } else {
                    let table_line = table_lines.get(i - 2).unwrap_or(&empty_line);
                    println!("{:<45} {}", table_line, board_line);

                    // If this is the last board line and we have a chosen action, print it with an arrow
                    if i == board_lines.len() - 1 {
                        if let Some(action) = chosen_action {
                            let arrow = match action {
                                Direction::Up => "↑",
                                Direction::Down => "↓",
                                Direction::Left => "←",
                                Direction::Right => "→",
                            };
                            println!("{:<45}  {} {:?}", "", arrow, action);
                            println!();
                        }
                    }
                }
            }
        } else {
            // No action table, just print the board
            for (i, line) in board_lines.iter().enumerate() {
                println!("{}", line);

                if i == board_lines.len() - 1 {
                    if let Some(action) = chosen_action {
                        let arrow = match action {
                            Direction::Up => "↑",
                            Direction::Down => "↓",
                            Direction::Left => "←",
                            Direction::Right => "→",
                        };
                        println!("  {} {:?}", arrow, action);
                        println!();
                    }
                }
            }
        }
    }

    fn display_board_with_action_table(
        &self,
        action_table: Option<String>,
        chosen_action: Option<Direction>,
    ) {
        let board_lines = self.generate_board_lines();

        if let Some(table) = action_table {
            let table_lines: Vec<String> = table.lines().map(|s| s.to_string()).collect();
            self.print_combined(board_lines, Some(table_lines), chosen_action);
        } else {
            self.print_combined(board_lines, None, chosen_action);
        }
    }

    pub fn build_actions_table(&mut self) -> (String, Option<Direction>) {
        if self.game.state != GameState::WaitingForAction {
            return (String::new(), None);
        }

        let actions = self.game.available_action();
        if actions.is_empty() {
            return ("❌ No available actions - Game Over!\n".to_string(), None);
        }

        // Run MCTS to get action scores
        for _ in 0..self.mcts_iterations {
            self.tree.step();
        }

        // Collect action scores and find the best one
        let mut action_scores: Vec<(Direction, f64, usize)> = Vec::new();
        let mut best_action = None;
        let mut best_score = f64::NEG_INFINITY;

        for action in &actions {
            let score = self.get_action_score_from_tree(action);
            action_scores.push((*action, score.0, score.1));
            if score.0 > best_score {
                best_score = score.0;
                best_action = Some(*action);
            }
        }

        // Sort actions by name for consistent ordering
        action_scores.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        // Generate action table lines and convert to string
        let table_lines = self.generate_action_table_lines(&action_scores, best_action);
        let output = table_lines.join("\n");

        (output, best_action)
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
            if self.game.state == GameState::Done {
                self.display_board();
                println!("🏁 Game Over! Final Score: {}", self.game.score);
                break;
            }

            if self.game.state == GameState::WaitingForAction {
                let (action_table, chosen_action) = self.build_actions_table();
                self.display_board_with_action_table(Some(action_table), chosen_action);

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
                    thread::sleep(Duration::from_millis(100));
                }

                // Make the best move based on MCTS
                if let Some(action) = chosen_action {
                    self.game.step_action(action);
                }
            } else if self.game.state == GameState::WaitingForRandom {
                // Handle random tile spawn
                self.handle_random_spawn();
            }
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
        }
    }
}

pub fn run_visualization() {
    let mut visualizer = GameVisualizer::new(Coord { row: 2, col: 1 }, 2);
    visualizer.set_mcts_iterations(500);
    visualizer.step_through_game();
}

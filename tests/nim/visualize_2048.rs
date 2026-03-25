use std::io::{self, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use recon_mcts::{GetState, SearchTree, Tree};

use crate::game_2048::{Coord, Direction, Game2048, GameState};
use crate::test_mcts_2048::Game2048Dynamics;

type Game2048MctsTree = Tree<
    recon_mcts::Node<
        Game2048Dynamics,
        Game2048,
        (),
        crate::test_mcts_2048::ActionChance,
        crate::test_mcts_2048::ScoreItem,
        std::vec::IntoIter<((), crate::test_mcts_2048::ActionChance)>,
        GetState,
    >,
    Game2048Dynamics,
>;

pub struct GameVisualizer {
    game: Game2048,
    dynamics: Game2048Dynamics,
    tree: Game2048MctsTree,
    mcts_iterations: usize,
}

impl GameVisualizer {
    pub fn new(initial_coord: Coord, initial_value: u32) -> Self {
        let game = Game2048::new_game(initial_coord, initial_value);
        let dynamics = Game2048Dynamics::default();
        let tree = Tree::new(dynamics, GetState, (), game);

        Self {
            game,
            dynamics,
            tree,
            mcts_iterations: 1000,
        }
    }

    pub fn set_mcts_iterations(&mut self, iterations: usize) {
        self.mcts_iterations = iterations;
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
        action_scores: &[(Direction, i32, AtomicU32)],
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
            let visits_converted = visits.load(Ordering::Relaxed);
            let action_str = format!("{:?}", action);
            table_lines.push(format!(
                "│ {:<12} │ {:>8.2} │ {:>7} │{}│",
                action_str, score, visits_converted, marker
            ));
        }

        table_lines.push("└──────────────┴──────────┴─────────┴─────────┘".to_string());
        table_lines.push(String::new());

        table_lines
    }

    fn generate_combined_lines(
        &self,
        board_lines: Vec<String>,
        table_lines: Option<Vec<String>>,
        chosen_action: Option<Direction>,
    ) -> Vec<String> {
        let mut output = Vec::new();
        output.push("══════════════════════════════════════════════════════════════".to_string());

        if let Some(table_lines) = table_lines {
            let mut empty_line = String::new();
            for _ in 0..47 {
                empty_line.push(' ');
            }

            // Combine lines side-by-side
            for i in 0..board_lines.len() {
                let board_line = board_lines.get(i).unwrap_or(&empty_line);

                if i < 2 {
                    // For score and blank line, just use the board line
                    output.push(board_line.clone());
                } else {
                    let table_line = table_lines.get(i - 2).unwrap_or(&empty_line);
                    output.push(format!("{:<45} {}", table_line, board_line));

                    // If this is the last board line and we have a chosen action, add it with an arrow
                    if i == board_lines.len() - 1 {
                        if let Some(action) = chosen_action {
                            let arrow = match action {
                                Direction::Up => "↑",
                                Direction::Down => "↓",
                                Direction::Left => "←",
                                Direction::Right => "→",
                            };
                            output.push(format!("{:<45}  {} {:?}", "", arrow, action));
                            output.push(String::new());
                        }
                    }
                }
            }
        } else {
            // No action table, just use the board
            for (i, line) in board_lines.iter().enumerate() {
                output.push(line.clone());

                if i == board_lines.len() - 1 {
                    if let Some(action) = chosen_action {
                        let arrow = match action {
                            Direction::Up => "↑",
                            Direction::Down => "↓",
                            Direction::Left => "←",
                            Direction::Right => "→",
                        };
                        output.push(format!("  {} {:?}", arrow, action));
                        output.push(String::new());
                    }
                }
            }
        }

        output
    }

    fn generate_display_lines(
        &self,
        action_table: Option<String>,
        chosen_action: Option<Direction>,
    ) -> Vec<String> {
        let board_lines = self.generate_board_lines();

        if let Some(table) = action_table {
            let table_lines: Vec<String> = table.lines().map(|s| s.to_string()).collect();
            self.generate_combined_lines(board_lines, Some(table_lines), chosen_action)
        } else {
            self.generate_combined_lines(board_lines, None, chosen_action)
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

        // Run MCTS to get action scores. Though first we need to reset the tree

        self.tree = Tree::new(self.dynamics, GetState, (), self.game);
        for _ in 0..self.mcts_iterations {
            self.tree.step();
        }

        let mut action_scores: Vec<(Direction, i32, AtomicU32)> = Vec::new();

        for (ac, move_info) in self.tree.get_next_move_info().unwrap() {
            if let crate::test_mcts_2048::ActionChance::Action(action) = ac {
                let score_item = move_info.score.unwrap();
                action_scores.push((action, score_item.score, score_item.visits));
            }
        }
        let best_action = action_scores
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(action, _, _)| *action);

        // Sort actions by name for consistent ordering
        action_scores.sort_by(|a, b| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)));

        // Generate action table lines and convert to string
        let table_lines = self.generate_action_table_lines(&action_scores, best_action);
        let output = table_lines.join("\n");

        (output, best_action)
    }

    pub fn step_through_game(&mut self) {
        println!("🚀 Starting 2048 Game with MCTS AI!");
        println!("Press Enter to continue to next move, 'q' to quit, or 'a' for auto-play");

        let mut auto_play = false;

        loop {
            if self.game.state == GameState::Done {
                let display_lines = self.generate_display_lines(None, None);
                let game_over = format!("🏁 Game Over! Final Score: {}", self.game.score);

                let mut final_output = display_lines.join("\n");
                final_output.push('\n');
                final_output.push_str(&game_over);
                println!("{}", final_output);
                break;
            }

            if self.game.state == GameState::WaitingForAction {
                let (action_table, chosen_action) = self.build_actions_table();
                let display_lines = self.generate_display_lines(Some(action_table), chosen_action);

                let mut output = display_lines.join("\n");
                output.push('\n');

                if !auto_play {
                    output.push_str("Press Enter to continue, 'a' for auto-play, 'q' to quit: ");

                    println!("{}", output);
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
                    println!("{}", output);
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
        assert!(!chances.is_empty());
        let rand_idx = rand::random::<u16>() % (chances.len() as u16);
        let (coord, value, _) = &chances[rand_idx as usize];
        self.game.step_random(*coord, *value);
    }
}

pub fn run_visualization() {
    let mut visualizer = GameVisualizer::new(Coord { row: 2, col: 1 }, 2);
    visualizer.set_mcts_iterations(5000);
    visualizer.step_through_game();
}

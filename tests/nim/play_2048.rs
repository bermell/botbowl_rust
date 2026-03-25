//! Shared full-game drivers for 2048: seeded RNG for the initial tile and uniform random spawns.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use recon_mcts::{GetState, SearchTree, Status, Tree};

use crate::game_2048::{Coord, Game2048, GameState};
use crate::test_mcts_2048::ActionChance;

/// Matches [`crate::benchmark_2048`] warm-up before reading the root children.
pub const DEFAULT_WARMUP_STEPS: usize = 50;

pub fn new_game_with_rng<R: Rng>(rng: &mut R) -> Game2048 {
    let random_row = rng.random_range(0usize..4);
    let random_col = rng.random_range(0usize..4);
    Game2048::new_game(
        Coord {
            row: random_row,
            col: random_col,
        },
        2,
    )
}

fn apply_random_chance<R: Rng>(game: &mut Game2048, rng: &mut R) {
    let chances = game.available_chance();
    assert!(!chances.is_empty(), "expected spawn choices");
    let idx = rng.random_range(0..chances.len());
    let (c, v, _) = chances[idx];
    game.step_random(c, v);
}

pub fn run_heuristic_game(seed: u64) -> i32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut game = new_game_with_rng(&mut rng);
    while game.state != GameState::Done {
        match game.state {
            GameState::WaitingForAction => {
                let dir = game.best_direction_heuristic();
                game.step_action(dir);
            }
            GameState::WaitingForRandom => apply_random_chance(&mut game, &mut rng),
            GameState::Done => break,
        }
    }
    game.score as i32
}

pub fn run_mcts_game(seed: u64, mcts_iterations: usize, warmup_steps: usize) -> i32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let game = new_game_with_rng(&mut rng);
    let tree = Tree::new(game, GetState, (), game);

    loop {
        for _ in 0..warmup_steps {
            tree.step();
        }
        let Some(root_actions) = tree.get_next_move_info() else {
            break;
        };
        if root_actions.is_empty() {
            break;
        }

        let is_action_node = matches!(root_actions[0].0, ActionChance::Action(_));
        let next_action = if is_action_node {
            for _ in 0..mcts_iterations {
                tree.step();
            }
            match tree.best_action() {
                Status::Action(best_action) => best_action,
                _ => panic!("expected best action at an action node"),
            }
        } else {
            let rand_idx = rng.random_range(0..root_actions.len());
            root_actions[rand_idx].0
        };
        tree.apply_action(&next_action);
    }

    tree.get_root_info().score.unwrap().score
}

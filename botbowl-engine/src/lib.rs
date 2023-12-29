#![allow(clippy::new_ret_no_self)]
use crate::core::gamestate::GameStateBuilder;

use crate::core::gamestate::GameState;

pub mod bots;
pub mod core;

//TODO: this shouldn't be here
pub fn standard_state() -> GameState {
    GameStateBuilder::new()
        .add_home_players(&[(1, 2), (2, 2), (3, 1)])
        .add_away_players(&[(5, 2), (5, 5), (2, 3)])
        .add_ball((3, 2))
        .build()
}

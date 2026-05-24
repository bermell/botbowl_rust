//! Library surface for botbowl-ui. The binary is in `main.rs`; this lib target lets integration
//! tests in `tests/` call directly into the deterministic snapshot renderer.

pub mod bot_factory;
pub mod cli;
pub mod player_drawings;
pub mod render;
pub mod snapshot;
pub mod tui;

use botbowl_engine::core::{
    game_runner::{BotGameRunnerBuilder, GameRunner},
    gamestate::GameState,
};
use cli::BotKind;
use ratatui::{backend::TestBackend, Terminal};
use std::io;

/// Render a single deterministic frame for the given `state` into a plain-text buffer.
pub fn render_state(state: &GameState, width: u16, height: u16) -> io::Result<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|frame| render::draw(frame, state, &[]))?;
    Ok(format!("{}", terminal.backend()))
}

/// Render a deterministic frame of a fresh seeded agent-vs-agent game after stepping `step`
/// micro-steps. Returns the plain-text snapshot as produced by `TestBackend`'s `Display` impl.
pub fn render_seeded_snapshot(seed: u64, step: usize, width: u16, height: u16) -> io::Result<String> {
    render_seeded_snapshot_with_bots(seed, step, width, height, BotKind::Random, BotKind::Random, 1000)
}

pub fn render_seeded_snapshot_with_bots(
    seed: u64,
    step: usize,
    width: u16,
    height: u16,
    home_bot: BotKind,
    away_bot: BotKind,
    mcts_iters: usize,
) -> io::Result<String> {
    let mut runner = BotGameRunnerBuilder::new()
        .set_seed(seed)
        .set_home_bot(bot_factory::make_bot(home_bot, mcts_iters))
        .set_away_bot(bot_factory::make_bot(away_bot, mcts_iters))
        .build();
    for _ in 0..step {
        if runner.game_over() {
            break;
        }
        runner.step();
    }
    render_state(runner.get_state(), width, height)
}

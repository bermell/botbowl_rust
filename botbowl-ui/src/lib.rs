//! Library surface for botbowl-ui. The binary is in `main.rs`; this lib target lets integration
//! tests in `tests/` call directly into the deterministic snapshot renderer.

pub mod cli;
pub mod player_drawings;
pub mod render;
pub mod snapshot;

use botbowl_engine::core::game_runner::{BotGameRunnerBuilder, GameRunner};
use ratatui::{backend::TestBackend, Terminal};
use std::io;

/// Render a deterministic frame of a fresh seeded agent-vs-agent game after stepping `step`
/// micro-steps. Returns the plain-text snapshot as produced by `TestBackend`'s `Display` impl.
pub fn render_seeded_snapshot(seed: u64, step: usize, width: u16, height: u16) -> io::Result<String> {
    let mut runner = BotGameRunnerBuilder::new().set_seed(seed).build();
    for _ in 0..step {
        if runner.game_over() {
            break;
        }
        runner.step();
    }
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|frame| render::draw(frame, runner.get_state(), &[]))?;
    Ok(format!("{}", terminal.backend()))
}

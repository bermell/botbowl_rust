use std::io::{self, Write};

use botbowl_engine::core::game_runner::{BotGameRunner, BotGameRunnerBuilder, GameRunner, Recording};
use ratatui::{backend::TestBackend, Terminal};

use crate::cli::SnapshotArgs;
use crate::render;

pub fn run(args: SnapshotArgs) -> io::Result<()> {
    let (width, height) = args.size;

    let frame = match (&args.replay, args.seed) {
        (Some(path), _) => render_replay(path, args.step, width, height)?,
        (None, Some(seed)) => render_seeded(seed, args.step, width, height)?,
        (None, None) => {
            eprintln!("snapshot requires either --replay PATH or --seed N");
            std::process::exit(2);
        }
    };

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(frame.as_bytes())?;
    Ok(())
}

fn render_seeded(seed: u64, step: usize, width: u16, height: u16) -> io::Result<String> {
    let mut runner: BotGameRunner = BotGameRunnerBuilder::new().set_seed(seed).build();
    for _ in 0..step {
        if runner.game_over() {
            break;
        }
        runner.step();
    }
    render_frame(runner.get_state(), width, height)
}

fn render_replay(path: &str, step: usize, width: u16, height: u16) -> io::Result<String> {
    let mut recording = Recording::from_file(path);
    let last_valid = recording.total_steps().saturating_sub(1);
    let target = step.min(last_valid);
    while recording.current_step() < target {
        recording.step();
    }
    render_frame(recording.get_state(), width, height)
}

fn render_frame(
    state: &botbowl_engine::core::gamestate::GameState,
    width: u16,
    height: u16,
) -> io::Result<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|frame| render::draw(frame, state, &[]))?;
    Ok(format!("{}", terminal.backend()))
}

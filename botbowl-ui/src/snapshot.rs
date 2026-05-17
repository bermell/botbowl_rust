use std::io::{self, Write};

use botbowl_engine::core::game_runner::{GameRunner, Recording};

use crate::cli::SnapshotArgs;
use crate::{render_seeded_snapshot, render_state};

pub fn run(args: SnapshotArgs) -> io::Result<()> {
    let (width, height) = args.size;

    let frame = match (&args.replay, args.seed) {
        (Some(path), _) => render_replay(path, args.step, width, height)?,
        (None, Some(seed)) => render_seeded_snapshot(seed, args.step, width, height)?,
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

fn render_replay(path: &str, step: usize, width: u16, height: u16) -> io::Result<String> {
    let mut recording = Recording::from_file(path);
    if recording.total_steps() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("recording '{path}' has no states"),
        ));
    }
    let last_valid = recording.total_steps() - 1;
    let target = step.min(last_valid);
    while recording.current_step() < target {
        recording.step();
    }
    render_state(recording.get_state(), width, height)
}

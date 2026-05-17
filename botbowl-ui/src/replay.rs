use std::{io, time::Duration};

use botbowl_engine::core::game_runner::{GameRunner, Recording};
use crossterm::event::{self, Event, KeyCode};

use botbowl_ui::cli::ReplayArgs;
use botbowl_ui::{render, tui};

pub fn run(args: ReplayArgs) -> io::Result<()> {
    let mut recording = Recording::from_file(&args.path);

    let mut terminal = tui::init_terminal()?;
    let mut log: Vec<String> = Vec::new();

    let result: io::Result<()> = loop {
        let footer = format!(
            "step {}/{}",
            recording.current_step(),
            recording.total_steps().saturating_sub(1)
        );
        log.clear();
        log.push(footer);
        log.push("← prev   → next   q quit".to_string());

        if let Err(e) = terminal.draw(|frame| render::draw(frame, recording.get_state(), &log)) {
            break Err(e);
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                    KeyCode::Right | KeyCode::Char(' ') => {
                        if recording.current_step() + 1 < recording.total_steps() {
                            recording.step();
                        }
                    }
                    KeyCode::Left => {
                        recording.step_back();
                    }
                    _ => {}
                }
            }
        }
    };

    tui::restore_terminal()?;
    result
}

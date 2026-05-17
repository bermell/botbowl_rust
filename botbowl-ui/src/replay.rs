use std::{
    io::{self, stdout, Stdout},
    time::Duration,
};

use botbowl_engine::core::game_runner::{GameRunner, Recording};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;

use botbowl_ui::cli::ReplayArgs;
use botbowl_ui::render;

pub fn run(args: ReplayArgs) -> io::Result<()> {
    let mut recording = Recording::from_file(&args.path);

    let mut terminal = init_terminal()?;
    let mut log: Vec<String> = Vec::new();

    let result: io::Result<()> = loop {
        let footer = format!("step {}/{}", recording.current_step(), recording.total_steps().saturating_sub(1));
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

    restore_terminal()?;
    result
}

fn init_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

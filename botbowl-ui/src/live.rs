use std::{
    io::{self, stdout, Stdout},
    time::{Duration, Instant},
};

use botbowl_engine::core::game_runner::{BotGameRunnerBuilder, GameRunner};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;

use botbowl_ui::cli::LiveArgs;
use botbowl_ui::render;

pub fn run(args: LiveArgs) -> io::Result<()> {
    let mut builder = BotGameRunnerBuilder::new();
    if let Some(seed) = args.seed {
        builder = builder.set_seed(seed);
    }
    if let Some(ref path) = args.save {
        builder = builder.set_replay_file(path);
    }
    let mut game = builder.build();

    let mut terminal = init_terminal()?;
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(80);
    let mut do_step = false;
    let mut log: Vec<String> = Vec::new();

    let result: io::Result<()> = loop {
        if let Err(e) = terminal.draw(|frame| render::draw(frame, game.get_state(), &log)) {
            break Err(e);
        }
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                    KeyCode::Char('s') => do_step = !do_step,
                    KeyCode::Char(' ') | KeyCode::Right => {
                        if !game.game_over() {
                            push_log(&mut log, game.get_state());
                            game.step();
                        }
                    }
                    _ => {}
                }
            }
        }
        if do_step && !game.game_over() {
            push_log(&mut log, game.get_state());
            game.step();
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    };

    restore_terminal()?;
    if args.save.is_some() {
        game.save_to_file();
    }
    result
}

fn push_log(
    log: &mut Vec<String>,
    state: &botbowl_engine::core::gamestate::GameState,
) {
    let team = state
        .available_actions
        .team
        .map(|t| format!("{t:?}"))
        .unwrap_or_else(|| "-".to_string());
    let proc = state.proc_stack_top().unwrap_or("(empty)");
    log.push(format!("{team:<5} {proc}"));
    if log.len() > 200 {
        log.drain(0..(log.len() - 200));
    }
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

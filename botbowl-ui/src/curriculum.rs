use std::{
    io,
    time::{Duration, Instant},
};

use botbowl_curriculum::{available_lectures, make_lecture, Difficulty, LectureSession, LectureStatus};
use crossterm::event::{self, Event, KeyCode};

use botbowl_ui::bot_factory::make_bot;
use botbowl_ui::cli::{CliDifficulty, CurriculumArgs};
use botbowl_ui::{render, tui};

pub fn run(args: CurriculumArgs) -> io::Result<()> {
    let difficulty = match args.difficulty {
        CliDifficulty::Easy => Difficulty::Easy,
        CliDifficulty::Medium => Difficulty::Medium,
        CliDifficulty::Hard => Difficulty::Hard,
    };

    let lecture = match make_lecture(&args.name, difficulty) {
        Some(l) => l,
        None => {
            eprintln!(
                "unknown lecture: {:?} ({:?}).\nAvailable lectures:",
                args.name, difficulty
            );
            for (name, diff) in available_lectures() {
                eprintln!("  --name {name:?} --difficulty {diff:?}");
            }
            return Ok(());
        }
    };

    let mut agent = make_bot(args.bot, args.mcts_iters);
    let mut session = LectureSession::new(lecture.as_ref(), args.seed, args.max_steps, agent.as_mut());

    let mut log: Vec<String> = Vec::new();
    log.push(format!(
        "Lecture: {} ({:?}) | bot: {:?} | seed: {} | max_steps: {}",
        lecture.name(),
        difficulty,
        args.bot,
        args.seed,
        args.max_steps,
    ));

    let mut terminal = tui::init_terminal()?;
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(80);
    let mut do_step = false;
    let mut verdict_logged = false;

    let result: io::Result<()> = loop {
        if let Err(e) = terminal.draw(|frame| render::draw(frame, session.state(), &log)) {
            break Err(e);
        }

        if session.is_finished() && !verdict_logged {
            log.push(verdict_line(session.status(), session.steps_taken()));
            verdict_logged = true;
            do_step = false;
        }

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                    KeyCode::Char('s') => {
                        if !session.is_finished() {
                            do_step = !do_step;
                        }
                    }
                    KeyCode::Char(' ') | KeyCode::Right => {
                        if !session.is_finished() {
                            push_state_log(&mut log, session.state());
                            session.step(agent.as_mut());
                        }
                    }
                    _ => {}
                }
            }
        }
        if do_step && !session.is_finished() {
            push_state_log(&mut log, session.state());
            session.step(agent.as_mut());
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    };

    tui::restore_terminal()?;
    result
}

fn verdict_line(status: LectureStatus, steps: u32) -> String {
    match status {
        LectureStatus::Success => format!("===== LECTURE PASSED ===== ({steps} micro-steps)"),
        LectureStatus::Failure => format!("===== LECTURE FAILED ===== ({steps} micro-steps)"),
        LectureStatus::InProgress => {
            format!("===== LECTURE TIMED OUT ===== ({steps} micro-steps, raise --max-steps)")
        }
    }
}

fn push_state_log(log: &mut Vec<String>, state: &botbowl_engine::core::gamestate::GameState) {
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

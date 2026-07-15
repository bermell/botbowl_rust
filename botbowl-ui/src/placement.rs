//! Interactive tuner for random-start placement biases (plan 019).
//!
//! Renders a freshly generated random state and lets you feel out the bias
//! variables: space generates a new state, 1-9 select a variable, up/down
//! adjust it (regenerating the same sample for a like-for-like comparison),
//! b cycles the board size, q/Esc quits.

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use botbowl_curriculum::{
    generate_random_start, RandomStartConfig, BIAS_MAX, BIAS_MIN, DECAY_MAX, DECAY_MIN, TEMP_MAX, TEMP_MIN,
};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::BoardDims;
use botbowl_ui::cli::PlacementArgs;
use botbowl_ui::{render, tui};

/// Board presets to cycle with `b`, in engine dims (playable + 2 border);
/// the smallest is the engine's documented small-board test floor. Presets
/// beyond the compiled capacity are skipped at startup.
const BOARD_PRESETS: [(i8, i8, usize); 5] = [(28, 17, 11), (22, 13, 8), (16, 9, 5), (12, 7, 4), (10, 5, 2)];

fn available_boards() -> Vec<BoardDims> {
    let capacity = BoardDims::default();
    let mut boards = vec![BoardDims::from_env()];
    for &(w, h, p) in &BOARD_PRESETS {
        if w <= capacity.width && h <= capacity.height && p <= capacity.team_size {
            let dims = BoardDims::new(w, h, p);
            if !boards.contains(&dims) {
                boards.push(dims);
            }
        }
    }
    boards
}

struct Knob {
    name: &'static str,
    step: f32,
    min: f32,
    max: f32,
    get: fn(&RandomStartConfig) -> f32,
    set: fn(&mut RandomStartConfig, f32),
}

const KNOBS: [Knob; 9] = [
    Knob {
        name: "ball_dist",
        step: 0.05,
        min: DECAY_MIN,
        max: DECAY_MAX,
        get: |c| c.ball_distance,
        set: |c, v| c.ball_distance = v,
    },
    Knob {
        name: "front_line",
        step: 0.05,
        min: DECAY_MIN,
        max: DECAY_MAX,
        get: |c| c.front_line,
        set: |c, v| c.front_line = v,
    },
    Knob {
        name: "mark_mate",
        step: 0.25,
        min: BIAS_MIN,
        max: BIAS_MAX,
        get: |c| c.mark_teammate,
        set: |c, v| c.mark_teammate = v,
    },
    Knob {
        name: "mark_opp",
        step: 0.25,
        min: BIAS_MIN,
        max: BIAS_MAX,
        get: |c| c.mark_opponent,
        set: |c, v| c.mark_opponent = v,
    },
    Knob {
        name: "own_side",
        step: 0.25,
        min: BIAS_MIN,
        max: BIAS_MAX,
        get: |c| c.own_side,
        set: |c, v| c.own_side = v,
    },
    Knob {
        name: "temp",
        step: 0.1,
        min: TEMP_MIN,
        max: TEMP_MAX,
        get: |c| c.temperature,
        set: |c, v| c.temperature = v,
    },
    Knob {
        name: "carried",
        step: 0.05,
        min: 0.0,
        max: 1.0,
        get: |c| c.carried_prob,
        set: |c, v| c.carried_prob = v,
    },
    Knob {
        name: "line_frac",
        step: 0.05,
        min: 0.0,
        max: 1.0,
        get: |c| c.line_fraction,
        set: |c, v| c.line_fraction = v,
    },
    Knob {
        name: "pocket_frac",
        step: 0.05,
        min: 0.0,
        max: 1.0,
        get: |c| c.pocket_fraction,
        set: |c, v| c.pocket_fraction = v,
    },
];

fn generate(cfg: &RandomStartConfig, base_seed: u64, counter: u64) -> GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(counter));
    generate_random_start(cfg, &mut rng)
}

/// Two knobs per line so all nine fit the 5-line log panel, plus a hint.
fn status_lines(
    cfg: &RandomStartConfig,
    selected: usize,
    base_seed: u64,
    counter: u64,
    board: BoardDims,
) -> Vec<String> {
    let cell = |i: usize| -> String {
        let marker = if i == selected { ">" } else { " " };
        format!("{marker}{}.{} = {:<7.2}", i + 1, KNOBS[i].name, (KNOBS[i].get)(cfg))
    };
    let mut lines: Vec<String> = (0..4)
        .map(|row| format!("{:<28}{}", cell(2 * row), cell(2 * row + 1)))
        .collect();
    lines.push(format!(
        "{:<28}[board {}x{}/{} | seed {} | space:new 1-9:select up/down:adjust b:board q:quit]",
        cell(8),
        board.width - 2,
        board.height - 2,
        board.team_size,
        base_seed + counter
    ));
    lines
}

pub fn run(args: PlacementArgs) -> io::Result<()> {
    let boards = available_boards();
    let mut board_idx = 0usize;
    let mut cfg = args.bias.to_config();
    cfg.board_dims = Some(boards[board_idx]);
    let mut selected = 0usize;
    let mut counter = 0u64;
    let mut state = generate(&cfg, args.seed, counter);

    let mut terminal = tui::init_terminal()?;
    let result: io::Result<()> = loop {
        let log = status_lines(&cfg, selected, args.seed, counter, boards[board_idx]);
        if let Err(e) = terminal.draw(|frame| render::draw(frame, &state, &log)) {
            break Err(e);
        }
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                    KeyCode::Char(' ') => {
                        counter += 1;
                        state = generate(&cfg, args.seed, counter);
                    }
                    KeyCode::Char(c @ '1'..='9') => {
                        selected = c as usize - '1' as usize;
                    }
                    KeyCode::Char('b') => {
                        board_idx = (board_idx + 1) % boards.len();
                        cfg.board_dims = Some(boards[board_idx]);
                        state = generate(&cfg, args.seed, counter);
                    }
                    KeyCode::Up | KeyCode::Down => {
                        let knob = &KNOBS[selected];
                        let sign = if key.code == KeyCode::Up { 1.0 } else { -1.0 };
                        let value = ((knob.get)(&cfg) + sign * knob.step).clamp(knob.min, knob.max);
                        (knob.set)(&mut cfg, value);
                        // Same counter: re-sample the same state under the new
                        // biases so their effect is visible like-for-like.
                        state = generate(&cfg, args.seed, counter);
                    }
                    _ => {}
                }
            }
        }
    };

    tui::restore_terminal()?;
    result
}

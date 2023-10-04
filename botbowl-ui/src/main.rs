use botbowl_engine::core::gamestate::{GameState, GameStateBuilder};
use botbowl_engine::core::model::{
    BallState, PlayerID, PlayerStatus, Position, TeamType, HEIGHT, WIDTH,
};
use botbowl_engine::core::table;

use std::{
    fmt::Write as fmt_Write,
    io::{self, stdin, Stdout, Write},
};
use termion::{self, color, input::TermRead, raw::RawTerminal};
use termion::{event::Key, raw::IntoRawMode};

const SQUARE_HEIGHT: usize = 2;
const SQUARE_WIDTH: usize = 4;

fn main() {
    println!("Hello world!");
    let mut state = GameStateBuilder::new()
        .add_home_players(&[(1, 1), (2, 2), (3, 1)])
        .add_away_players(&[(26, 15), (25, 14), (24, 15)])
        .add_ball((3, 3))
        .build();

    state.step_positional(table::PosAT::StartMove, state.get_player_unsafe(1).position);
    state.get_mut_player_unsafe(2).used = true;
    state.get_mut_player_unsafe(4).status = PlayerStatus::Stunned;
    state.get_mut_player_unsafe(5).status = PlayerStatus::Down;

    let mut rend = Renderer::new();
    rend.curser_pos = Some(Position::new((5, 5)));
    rend.run_loop(&mut state);
}

fn letter_to_ball_carrier(s: String) -> String {
    match s.as_str() {
        "H" => "\u{1E26}".to_owned(),
        "A" => "Ä".to_owned(),
        _ => unreachable!(),
    }
}

fn pos_to_tty_coord(pos: Position) -> (u16, u16) {
    let (mut x, mut y) = <(u16, u16)>::from(pos);
    x *= SQUARE_WIDTH as u16;
    y *= SQUARE_HEIGHT as u16;
    x -= 2;
    (x, y)
}
fn player_repr(game_state: &GameState, id: PlayerID) -> String {
    let p = game_state.get_player_unsafe(id);

    let mut player_char = {
        match p.stats.team {
            TeamType::Home => "H".to_owned(),
            TeamType::Away => "A".to_owned(),
        }
    };
    if p.status != PlayerStatus::Up {
        player_char = player_char.to_lowercase();
    } else if matches!(game_state.ball, BallState::Carried(id) if id == p.id) {
        player_char = letter_to_ball_carrier(player_char);
    }

    let mut fmt = "".to_owned();
    //if active player, set blinkning
    match game_state.info.active_player {
        Some(id) if id == p.id => {
            write!(&mut fmt, "{}", termion::style::Blink).unwrap();
        }
        _ => (),
    }

    //Set color
    match p.stats.team {
        TeamType::Home => write!(&mut fmt, "{}", color::Fg(color::Red)).unwrap(),
        TeamType::Away => write!(&mut fmt, "{}", color::Fg(color::LightBlue)).unwrap(),
    }

    let (prefix, suffix): (&str, &str) = match (p.used, p.status) {
        (false, PlayerStatus::Up) => ("(", ")"),
        (false, PlayerStatus::Down) => ("-", "-"),
        (false, PlayerStatus::Stunned) => ("=", "="),
        (true, PlayerStatus::Up) => (" ", " "),
        (true, PlayerStatus::Down) => ("-", "-"),
        (true, PlayerStatus::Stunned) => ("=", "="),
    };
    format!(
        "{}{}{}{}{}{}",
        fmt,
        prefix,
        player_char,
        suffix,
        termion::style::Reset,
        color::Fg(color::Reset)
    )
}

pub struct Renderer {
    pub curser_pos: Option<Position>,
}
impl Renderer {
    pub fn new() -> Renderer {
        Renderer { curser_pos: None }
    }

    fn select_current_cursor(&self, state: &mut GameState) {
        let pos = self.curser_pos.unwrap();
    }
    pub fn run_loop(&mut self, state: &mut GameState) {
        let stdin = stdin();
        let mut stdout = io::stdout().into_raw_mode().unwrap();
        let mut pos = self.curser_pos.unwrap();

        self.draw(&mut stdout, state);
        stdout.flush().unwrap();

        for c in stdin.keys() {
            match c.unwrap() {
                Key::Char('w') => pos += (0, -1),
                Key::Char('a') => pos += (-1, 0),
                Key::Char('d') => pos += (1, 0),
                Key::Char('s') => pos += (0, 1),
                Key::Char('q') => break,
                _ => (),
            }
            if !pos.is_out() {
                self.curser_pos = Some(pos);
            }
            self.draw(&mut stdout, state);
            stdout.flush().unwrap();
        }
    }
    pub fn draw(&mut self, stdout: &mut RawTerminal<Stdout>, state: &GameState) {
        let (width, height) = termion::terminal_size().unwrap();
        let min_width = 6 + SQUARE_WIDTH as u16 * 26;
        assert!(
            width > min_width,
            "width ({}) should be > {}",
            width,
            min_width
        );
        let min_height = 2 + SQUARE_HEIGHT as u16 * 15;
        assert!(
            height > min_height,
            "height ({}) should be > {}",
            height,
            min_height
        );

        // Get and lock the stdios.
        // let stdin = io::stdin();
        // let stdin = stdin.lock();
        // let stderr = io::stderr();
        // let mut stderr = stderr.lock();

        write!(stdout, "{}", termion::clear::All).unwrap();

        //draw crowd
        let crowd_y = (HEIGHT * SQUARE_HEIGHT - 3) as u16;
        let crowd_x = (WIDTH * SQUARE_WIDTH - 6) as u16;
        let long_side = "▒".repeat(crowd_x as usize);
        write!(stdout, "{}{}", termion::cursor::Goto(1, 1), long_side).unwrap();
        write!(stdout, "{}{}", termion::cursor::Goto(1, crowd_y), long_side).unwrap();

        for y in 1..crowd_y {
            write!(stdout, "{}▒▒", termion::cursor::Goto(1, y)).unwrap();
            write!(stdout, "{}▒▒", termion::cursor::Goto(crowd_x, y)).unwrap();
        }
        // draw vertical lines (TD zones and line of scrimmage)
        Position::all_positions()
            .filter(|pos| !pos.is_out())
            .filter(|&Position { x, y: _ }| x == 1 || x == 13 || x == 25)
            .map(pos_to_tty_coord)
            .for_each(|(x, y)| {
                write!(stdout, "{}|", termion::cursor::Goto(x + 4, y)).unwrap();
            });

        // draw horizontal lines (wings)
        Position::all_positions()
            .filter(|pos| !pos.is_out())
            .filter(|&Position { x: _, y }| y == 4 || y == 11)
            .map(pos_to_tty_coord)
            .for_each(|(x, y)| {
                write!(stdout, "{}-", termion::cursor::Goto(x + 2, y + 1)).unwrap();
            });

        //draw squares
        Position::all_positions()
            .filter(|pos| !pos.is_out())
            .filter(|&Position { x, y }| y != 15 && x != 1)
            .map(pos_to_tty_coord)
            .for_each(|(x, y)| {
                write!(stdout, "{}+", termion::cursor::Goto(x, y + 1)).unwrap();
            });

        // draw players
        Position::all_positions()
            .filter_map(|pos| state.get_player_id_at(pos))
            .for_each(|id| {
                let (x, y) = pos_to_tty_coord(state.get_player_unsafe(id).position);
                let s = player_repr(state, id);
                write!(stdout, "{}{}", termion::cursor::Goto(x + 1, y), s).unwrap();
            });
        // draw ball
        match state.ball {
            BallState::OnGround(p) => {
                let (x, y) = pos_to_tty_coord(p);
                write!(stdout, "{}-b-", termion::cursor::Goto(x + 1, y)).unwrap();
            }
            BallState::InAir(p) => {
                let (x, y) = pos_to_tty_coord(p);
                if state.get_player_at(p).is_some() {
                    write!(stdout, "{}\u{1E07}", termion::cursor::Goto(x + 2, y)).unwrap();
                } else {
                    write!(stdout, "{}^b^", termion::cursor::Goto(x + 1, y)).unwrap();
                }
            }
            _ => (),
        };

        if let Some((x, y)) = self.curser_pos.map(pos_to_tty_coord) {
            let white = color::Bg(color::White);
            write!(stdout, "{}{} ", termion::cursor::Goto(x, y), white).unwrap();
            write!(stdout, "{}{} ", termion::cursor::Goto(x + 2, y + 1), white).unwrap();
            write!(stdout, "{}{} ", termion::cursor::Goto(x + 2, y - 1), white).unwrap();
            write!(stdout, "{}{} ", termion::cursor::Goto(x + 4, y), white).unwrap();
            write!(
                stdout,
                "{}{}",
                termion::style::Reset,
                color::Fg(color::Reset)
            )
            .unwrap();
        }

        write!(stdout, "{}", termion::cursor::Goto(1, 35)).unwrap();
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

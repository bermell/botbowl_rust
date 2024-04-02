pub mod player_drawings;

use std::{
    io::{self, stdout, Stdout},
    rc::Rc,
    time::{Duration, Instant},
    u16,
};

use botbowl_engine::core::{
    game_runner::{BotGameRunner, BotGameRunnerBuilder, GameRunner},
    gamestate::{GameState, GameStateBuilder},
    model::{BallState, FieldedPlayer, PlayerStatus, Position, TeamType},
};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use player_drawings::{player_2x1, player_4x2, player_6x3, player_8x4};
use ratatui::{
    prelude::*,
    widgets::{
        canvas::{Canvas, Circle},
        *,
    },
};

fn main() -> io::Result<()> {
    App::run()
}

#[allow(dead_code)]
struct App {
    game: BotGameRunner,
}

impl App {
    fn new() -> App {
        let runner = BotGameRunnerBuilder::new().build();
        App { game: runner }
    }
    fn new_show_state() -> App {
        let down_pos = Position::from((1, 1));
        let stunned_pos = Position::from((2, 2));
        let ball_pos = Position::from((3, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(down_pos)
            .add_home_player(stunned_pos)
            .add_home_player(ball_pos)
            .add_ball((ball_pos.x, ball_pos.y))
            .build();

        state.get_mut_player_at_unsafe(down_pos).status = PlayerStatus::Down;
        state.get_mut_player_at_unsafe(stunned_pos).status = PlayerStatus::Stunned;

        let runner = BotGameRunnerBuilder::new().set_state(state).build();
        App { game: runner }
    }

    pub fn run() -> io::Result<()> {
        let mut terminal = init_terminal()?;
        let mut app = App::new_show_state();
        let mut last_tick = Instant::now();
        let mut do_step = false;
        let tick_rate = Duration::from_millis(40);
        loop {
            let _ = terminal.draw(|frame| app.ui(frame));
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if let KeyCode::Char('q') = key.code {
                        break;
                    }
                }
                if let Event::Key(key) = event::read()? {
                    if let KeyCode::Char('s') = key.code {
                        do_step = true;
                    }
                }
            }
            if do_step {
                app.game.step();
                do_step = false;
            }
            if app.game.game_over() {
                break;
            }

            if last_tick.elapsed() >= tick_rate {
                app.on_tick();
                last_tick = Instant::now();
            }
        }
        restore_terminal()
    }

    fn on_tick(&mut self) {}

    fn ui(&self, frame: &mut Frame) {
        const ROWS: u16 = 15;
        const COLS: u16 = 26;
        let rect_size = frame.size();
        let allowed_square_sizes = &[(10, 5), (8, 4), (6, 3), (4, 2), (2, 1)];
        let (square_width, square_height) = allowed_square_sizes
            .iter()
            .find(|(w, h)| rect_size.width / COLS >= *w && rect_size.height / ROWS >= *h)
            .unwrap_or(&(1, 2));
        let pitch_width = square_width * COLS;
        let pitch_height = square_height * ROWS;

        let pitch_intermediate = Layout::default()
            .direction(Direction::Vertical)
            .margin(0)
            .constraints([
                Constraint::Length((rect_size.height - pitch_height) / 2),
                Constraint::Length(pitch_height),
                Constraint::Length((rect_size.height - pitch_height) / 2),
            ])
            .split(frame.size());

        frame.render_widget(self.info_widget(), pitch_intermediate[0]);

        let pitch = Layout::default()
            .direction(Direction::Horizontal)
            .margin(0)
            .constraints([
                Constraint::Length((rect_size.width - pitch_width) / 2),
                Constraint::Length(pitch_width),
                Constraint::Length((rect_size.width - pitch_width) / 2),
            ])
            .split(pitch_intermediate[1])[1];

        let rows = split_rows(&pitch, *square_height, ROWS).to_vec();
        let squares = rows
            .iter()
            .map(|row| split_cols(row, *square_width, COLS).to_vec())
            .collect::<Vec<_>>();
        for (y, rows) in squares.iter().enumerate() {
            for (x, chunk) in rows.iter().enumerate() {
                let pos = Position::from((x + 1, y + 1));
                let bg_color = match (pos.x + pos.y) % 2 {
                    0 => Color::Reset,
                    _ => Color::Black,
                };
                let ball: bool = match self.game.get_state().ball {
                    BallState::OffPitch => false,
                    BallState::OnGround(p) => p == pos,
                    BallState::Carried(_) => false,
                    BallState::InAir(p) => p == pos,
                };
                let td = self.game.get_state().get_endzone_x(TeamType::Home) == pos.x
                    || self.game.get_state().get_endzone_x(TeamType::Away) == pos.x;

                if let Some(player) = self.game.get_state().get_player_at(pos) {
                    let paragraph =
                        player_paragraph(player, self.game.get_state(), bg_color, *chunk);
                    frame.render_widget(paragraph, *chunk);
                } else if ball {
                    frame.render_widget(self.ball_canvas(bg_color), *chunk);
                } else if td {
                    let fg_color = match bg_color {
                        Color::Reset => Color::DarkGray,
                        _ => Color::DarkGray,
                    };
                    frame.render_widget(
                        self.td_square_canvas(bg_color, fg_color, pos.y as usize),
                        *chunk,
                    );
                } else {
                    frame.render_widget(self.square_canvas(bg_color), *chunk);
                }
            }
        }
    }

    fn info_widget(&self) -> impl Widget + '_ {
        let info = &self.game.get_state().info;
        let home_turn = format!("Home: {}", info.home_turn);
        let away_turn = format!("Away: {}", info.away_turn);
        let line = match info.team_turn {
            TeamType::Home => {
                vec![
                    Span::styled(away_turn, Style::default()),
                    Span::from(" "),
                    Span::styled(home_turn, Style::default().underlined()),
                ]
            }
            TeamType::Away => {
                vec![
                    Span::styled(away_turn, Style::default().underlined()),
                    Span::from(" "),
                    Span::styled(home_turn, Style::default()),
                ]
            }
        };

        let text = Line::from(line);
        Paragraph::new(text.clone())
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray))
    }

    fn td_square_canvas(&self, bg_color: Color, fg_color: Color, y: usize) -> impl Widget + '_ {
        let td_chars = "    TOUCHDOWN     ".chars().collect::<Vec<_>>();

        Canvas::default()
            .background_color(bg_color)
            .marker(Marker::Braille)
            .paint(move |ctx| {
                ctx.print(50.0, 50.0, td_chars[y].to_string().fg(fg_color));
            }) // Needed to infer the return type
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 100.0])
    }
    // touchdown 9 charactera
    fn square_canvas(&self, bg_color: Color) -> impl Widget + '_ {
        Canvas::default()
            .background_color(bg_color)
            .marker(Marker::Braille)
            .paint(|_| {}) // Needed to infer the return type
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 100.0])
    }
    fn ball_canvas(&self, bg_color: Color) -> impl Widget + '_ {
        Canvas::default()
            .background_color(bg_color)
            .marker(Marker::Braille)
            .paint(move |ctx| {
                ctx.draw(&Circle {
                    x: 50.0,
                    y: 50.0,
                    radius: 15.0,
                    color: Color::Yellow,
                });
            })
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 100.0])
    }
}
fn split_rows(area: &Rect, row_height: u16, num_rows: u16) -> Rc<[Rect]> {
    let list_layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints(
            (0..num_rows)
                .map(|_| Constraint::Length(row_height))
                .collect::<Vec<_>>(),
        );

    list_layout.split(*area)
}

fn split_cols(area: &Rect, col_width: u16, num_cols: u16) -> Rc<[Rect]> {
    let list_layout = Layout::default()
        .direction(Direction::Horizontal)
        .margin(0)
        .constraints(
            (0..num_cols)
                .map(|_| Constraint::Length(col_width))
                .collect::<Vec<_>>(),
        );

    list_layout.split(*area)
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

fn player_paragraph<'a>(
    player: &'a FieldedPlayer,
    game_state: &'a GameState,
    bg_color: Color,
    rect: Rect,
) -> Paragraph<'a> {
    let (h, w) = (rect.height, rect.width);
    let text = match (w, h) {
        (8, 4) => player_8x4(player, game_state),
        (6, 3) => player_6x3(player, game_state),
        (4, 2) => player_4x2(player, game_state),
        (2, 1) => player_2x1(player, game_state),
        _ => panic!("Invalid Rectangle size {}, {}", h, w),
    };
    Paragraph::new(text)
        .alignment(Alignment::Center)
        .style(Style::default().bg(bg_color))
}

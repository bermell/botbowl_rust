use std::rc::Rc;

use botbowl_engine::core::{
    gamestate::GameState,
    model::{BallState, FieldedPlayer, Position, TeamType},
};
use ratatui::{
    prelude::*,
    widgets::{
        canvas::{Canvas, Circle},
        *,
    },
};

use crate::player_drawings::{player_2x1, player_4x2, player_6x3, player_8x4};

const ROWS: u16 = 15;
const COLS: u16 = 26;

pub fn draw(frame: &mut Frame, state: &GameState, log: &[String]) {
    let rect_size = frame.size();

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(0),    // body
            Constraint::Length(log_panel_height(log)),
        ])
        .split(rect_size);

    frame.render_widget(header_widget(state), outer[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .margin(0)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(side_panel_width(rect_size.width)),
        ])
        .split(outer[1]);

    draw_pitch(frame, state, body[0]);
    frame.render_widget(side_panel_widget(state), body[1]);

    if !log.is_empty() {
        frame.render_widget(log_widget(log), outer[2]);
    }
}

fn side_panel_width(total_width: u16) -> u16 {
    // Reserve up to 38 cols for the side panel, but never more than half the screen
    std::cmp::min(38, total_width / 2)
}

fn log_panel_height(log: &[String]) -> u16 {
    if log.is_empty() {
        0
    } else {
        // up to 5 lines plus 1-line border
        std::cmp::min(log.len() as u16, 5) + 2
    }
}

fn draw_pitch(frame: &mut Frame, state: &GameState, area: Rect) {
    let allowed_square_sizes = &[(10, 5), (8, 4), (6, 3), (4, 2), (2, 1)];
    let (square_width, square_height) = allowed_square_sizes
        .iter()
        .find(|(w, h)| area.width / COLS >= *w && area.height / ROWS >= *h)
        .copied()
        .unwrap_or((2, 1));
    let pitch_width = square_width * COLS;
    let pitch_height = square_height * ROWS;

    if area.width < pitch_width || area.height < pitch_height {
        // Fallback: not enough room — render a placeholder rather than panic.
        let msg = Paragraph::new("terminal too small for pitch")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(msg, area);
        return;
    }

    let pad_y_top = (area.height - pitch_height) / 2;
    let pad_x_left = (area.width - pitch_width) / 2;

    let pitch = Rect {
        x: area.x + pad_x_left,
        y: area.y + pad_y_top,
        width: pitch_width,
        height: pitch_height,
    };

    let rows: Rc<[Rect]> = split_rows(&pitch, square_height, ROWS);
    let squares: Vec<Vec<Rect>> = rows
        .iter()
        .map(|row| split_cols(row, square_width, COLS).to_vec())
        .collect();

    for (y, row_rects) in squares.iter().enumerate() {
        for (x, chunk) in row_rects.iter().enumerate() {
            let pos = Position::from((x + 1, y + 1));
            let bg_color = match (pos.x + pos.y) % 2 {
                0 => Color::Reset,
                _ => Color::DarkGray,
            };
            let ball = match state.ball {
                BallState::OffPitch => false,
                BallState::OnGround(p) => p == pos,
                BallState::Carried(_) => false,
                BallState::InAir(p) => p == pos,
            };
            let td = state.get_endzone_x(TeamType::Home) == pos.x
                || state.get_endzone_x(TeamType::Away) == pos.x;

            if let Some(player) = state.get_player_at(pos) {
                let paragraph = player_paragraph(player, state, bg_color, *chunk);
                frame.render_widget(paragraph, *chunk);
            } else if ball {
                frame.render_widget(ball_canvas(bg_color), *chunk);
            } else if td {
                frame.render_widget(
                    td_square_canvas(bg_color, Color::Gray, pos.y as usize),
                    *chunk,
                );
            } else {
                frame.render_widget(square_canvas(bg_color), *chunk);
            }
        }
    }
}

fn header_widget(state: &GameState) -> impl Widget + '_ {
    let info = &state.info;
    let score = format!("{} - {}", state.home.score, state.away.score);
    let half = format!("H{}", info.half.max(1));
    let home_turn = format!("Home T{}", info.home_turn);
    let away_turn = format!("Away T{}", info.away_turn);
    let (home_style, away_style) = match info.team_turn {
        TeamType::Home => (Style::default().underlined(), Style::default()),
        TeamType::Away => (Style::default(), Style::default().underlined()),
    };
    let line = Line::from(vec![
        Span::styled(half, Style::default().fg(Color::Cyan)),
        Span::from("  "),
        Span::styled(home_turn, home_style),
        Span::from("   "),
        Span::styled(score, Style::default().fg(Color::Yellow)),
        Span::from("   "),
        Span::styled(away_turn, away_style),
    ]);
    Paragraph::new(line)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray))
}

fn side_panel_widget(state: &GameState) -> impl Widget + '_ {
    let proc_line = match state.proc_stack_top() {
        Some(name) => format!("Proc: {name}"),
        None => "Proc: (empty)".to_string(),
    };
    let team_line = match state.available_actions.team {
        Some(team) => format!("To act: {:?}", team),
        None => "To act: -".to_string(),
    };

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(proc_line, Style::default().fg(Color::Cyan))),
        Line::from(Span::styled(team_line, Style::default().fg(Color::Gray))),
        Line::from(""),
        Line::from(Span::styled(
            "Actions:",
            Style::default().fg(Color::White).bold(),
        )),
    ];

    let mut actions: Vec<String> = state
        .available_actions
        .get_all()
        .iter()
        .map(|a| format!("{a:?}"))
        .collect();
    actions.sort();
    if actions.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for action in actions.iter().take(12) {
            lines.push(Line::from(Span::raw(format!("  {action}"))));
        }
        if actions.len() > 12 {
            lines.push(Line::from(Span::styled(
                format!("  + {} more", actions.len() - 12),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::LEFT)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false })
}

fn log_widget(log: &[String]) -> impl Widget + '_ {
    let take = std::cmp::min(log.len(), 5);
    let start = log.len() - take;
    let lines: Vec<Line> = log[start..]
        .iter()
        .map(|s| Line::from(Span::raw(s.clone())))
        .collect();
    Paragraph::new(lines).block(Block::default().borders(Borders::TOP).title("log"))
}

fn split_rows(area: &Rect, row_height: u16, num_rows: u16) -> Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints(
            (0..num_rows)
                .map(|_| Constraint::Length(row_height))
                .collect::<Vec<_>>(),
        )
        .split(*area)
}

fn split_cols(area: &Rect, col_width: u16, num_cols: u16) -> Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Horizontal)
        .margin(0)
        .constraints(
            (0..num_cols)
                .map(|_| Constraint::Length(col_width))
                .collect::<Vec<_>>(),
        )
        .split(*area)
}

fn td_square_canvas(bg_color: Color, fg_color: Color, y: usize) -> impl Widget {
    let td_chars = "    TOUCHDOWN     ".chars().collect::<Vec<_>>();
    Canvas::default()
        .background_color(bg_color)
        .marker(Marker::Braille)
        .paint(move |ctx| {
            ctx.print(50.0, 50.0, td_chars[y].to_string().fg(fg_color));
        })
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0])
}

fn square_canvas(bg_color: Color) -> impl Widget {
    Canvas::default()
        .background_color(bg_color)
        .marker(Marker::Braille)
        .paint(|_| {})
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 100.0])
}

fn ball_canvas(bg_color: Color) -> impl Widget {
    Canvas::default()
        .background_color(bg_color)
        .marker(Marker::Braille)
        .paint(|ctx| {
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

fn player_paragraph<'a>(
    player: &'a FieldedPlayer,
    state: &'a GameState,
    bg_color: Color,
    rect: Rect,
) -> Paragraph<'a> {
    let (h, w) = (rect.height, rect.width);
    let text = match (w, h) {
        (10, 5) => player_8x4(player, state),
        (8, 4) => player_8x4(player, state),
        (6, 3) => player_6x3(player, state),
        (4, 2) => player_4x2(player, state),
        (2, 1) => player_2x1(player, state),
        _ => player_2x1(player, state),
    };
    Paragraph::new(text)
        .alignment(Alignment::Center)
        .style(Style::default().bg(bg_color))
}

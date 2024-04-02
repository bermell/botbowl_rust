use botbowl_engine::core::model::{FieldedPlayer, TeamType};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

fn player_style(player: &FieldedPlayer) -> Style {
    match player.stats.team {
        TeamType::Home => Style::default().fg(Color::Red),
        TeamType::Away => Style::default().fg(Color::LightBlue),
    }
}
fn ball_style() -> Style {
    Style::default().fg(Color::Yellow)
}
fn ball_2x1() -> Span<'static> {
    Span::styled("⠪⠕", ball_style())
}
fn ball_1x1() -> Span<'static> {
    Span::styled("●", ball_style())
}

pub fn player_8x4(player: &FieldedPlayer, is_carrier: bool) -> Vec<Line> {
    // A player made of 8x4 characters, outline:
    // "   ██   "
    // " --▐▌-- "
    // "   ▐▌   "
    // "  /  \  "
    let style = player_style(player);
    let l1 = Line::from(vec![Span::styled("   ▆▆   ", style)]);
    let l2 = Line::from(vec![Span::styled(" --▐▌-- ", style)]);
    let l3 = Line::from(vec![Span::styled("   ▐▌   ", style)]);
    let l4 = Line::from(vec![Span::styled("  /  \\  ", style)]);

    let l2_ball = Line::from(vec![Span::styled(" --▐▌-", style), ball_2x1()]);

    if is_carrier {
        vec![l1, l2_ball, l3, l4]
    } else {
        vec![l1, l2, l3, l4]
    }
}
pub fn player_6x3(player: &FieldedPlayer, is_carrier: bool) -> Vec<Line> {
    // A player made of 6x3 characters, outline:
    // "  ▆▆  "
    // " -▐▌- "
    // "  /\  "
    let style = player_style(player);
    let l1 = Line::from(vec![Span::styled("  ▆▆  ", style)]);
    let l2 = Line::from(vec![Span::styled(" -▐▌- ", style)]);
    let l3 = Line::from(vec![Span::styled("  /\\  ", style)]);

    let l2_ball = Line::from(vec![Span::styled(" -▐▌", style), ball_2x1()]);
    if is_carrier {
        vec![l1, l2_ball, l3]
    } else {
        vec![l1, l2, l3]
    }
}
pub fn player_4x2(player: &FieldedPlayer, is_carrier: bool) -> Vec<Line> {
    // A player made of 4x2 characters, outline:
    // "-▝▘-"
    // " /\ "
    let style = player_style(player);
    if is_carrier {
        let l1_ball = Line::from(vec![Span::styled("-▝▘", style), ball_1x1()]);
        let l2 = Line::from(vec![Span::styled(" /\\ ", style)]);
        vec![l1_ball, l2]
    } else {
        let l1 = Line::from(vec![Span::styled("-▝▘-", style)]);
        let l2 = Line::from(vec![Span::styled(" /\\ ", style)]);
        vec![l1, l2]
    }
}
pub fn player_2x1(player: &FieldedPlayer, is_carrier: bool) -> Vec<Line> {
    // A player made of 2x1 characters, outline:
    // "☺-"
    let style = player_style(player);
    if is_carrier {
        let l1_ball = Line::from(vec![Span::styled("☺", style), ball_1x1()]);
        vec![l1_ball]
    } else {
        let l1 = Line::from(vec![Span::styled("☺-", style)]);
        vec![l1]
    }
}

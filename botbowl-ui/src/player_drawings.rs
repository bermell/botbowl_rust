use botbowl_engine::core::model::{BallState, FieldedPlayer, TeamType};
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
fn ball_style(_ball: BallState) -> Style {
    Style::default().fg(Color::Yellow)
}
fn ball_2x1(ball: BallState) -> Span<'static> {
    Span::styled("⠪⠕", ball_style(ball))
}

pub fn player_8x4(player: &FieldedPlayer, ball: BallState) -> Vec<Line> {
    // A player made of 8x4 characters, outline:
    // "   ██   "
    // " --▐▌-- "
    // "   ▐▌   "
    // "  /  \  "
    let style = player_style(player);
    let l1 = Line::from(vec![Span::styled("   ▆▆   ", style)]);
    let l2 = Line::from(vec![Span::styled(" --▐▌-- ", style)]);
    let lb = Line::from(vec![Span::styled(" --▐▌-", style), ball_2x1(ball)]);
    let l3 = Line::from(vec![Span::styled("   ▐▌   ", style)]);
    let l4 = Line::from(vec![Span::styled("  /  \\  ", style)]);

    match ball {
        BallState::Carried(_) => {
            vec![l1, lb]
        }
        _ => {
            vec![l1, l2, l3, l4]
        }
    }
}
pub fn player_6x3(player: &FieldedPlayer, ball: BallState) -> Vec<Line> {
    Vec::new()
}

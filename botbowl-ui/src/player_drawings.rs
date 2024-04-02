use botbowl_engine::core::{
    gamestate::GameState,
    model::{BallState, FieldedPlayer, PlayerStatus, TeamType},
};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

fn player_style(player: &FieldedPlayer, game_state: &GameState) -> Style {
    match player.stats.team {
        TeamType::Home => Style::default().fg(Color::Red),
        TeamType::Away => Style::default().fg(Color::LightBlue),
    }
}
fn is_carrier(player: &FieldedPlayer, game_state: &GameState) -> bool {
    match game_state.ball {
        BallState::Carried(id) => id == player.id,
        _ => false,
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

pub fn player_8x4<'a>(player: &FieldedPlayer, game_state: &GameState) -> Vec<Line<'a>> {
    let style = player_style(player, game_state);
    let is_carrier = is_carrier(player, game_state);

    if player.status != PlayerStatus::Up {
        let empty = Line::from(vec![Span::styled("        ", style)]);
        let star = Line::from(vec![Span::styled(" ***    ", style)]);
        let l3 = Line::from(vec![Span::styled(" ▄▄▁▃▃▬ ", style)]);
        let l4 = Line::from(vec![Span::styled(" ▀▀▔▀▀▬ ", style)]);
        if player.status == PlayerStatus::Down {
            return vec![empty.clone(), empty, l3, l4];
        }
        return vec![empty, star, l3, l4];
    }

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
pub fn player_6x3<'a>(player: &FieldedPlayer, game_state: &GameState) -> Vec<Line<'a>> {
    let style = player_style(player, game_state);
    let is_carrier = is_carrier(player, game_state);

    if player.status != PlayerStatus::Up {
        let empty = Line::from(vec![Span::styled("      ", style)]);
        let star = Line::from(vec![Span::styled(" ***  ", style)]);
        let l2 = Line::from(vec![Span::styled(" ▄▁▃▬ ", style)]);
        let l3 = Line::from(vec![Span::styled(" ▀▔▀▬ ", style)]);
        if player.status == PlayerStatus::Down {
            return vec![empty, l2, l3];
        }
        return vec![star, l2, l3];
    }

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
pub fn player_4x2<'a>(player: &FieldedPlayer, game_state: &GameState) -> Vec<Line<'a>> {
    let style = player_style(player, game_state);
    let is_carrier = is_carrier(player, game_state);

    if player.status != PlayerStatus::Up {
        let empty = Line::from(vec![Span::styled("    ", style)]);
        let star = Line::from(vec![Span::styled(" ** ", style)]);
        let l2 = Line::from(vec![Span::styled(" o-<", style)]);
        if player.status == PlayerStatus::Down {
            return vec![empty, l2];
        }
        return vec![star, l2];
    }

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
pub fn player_2x1<'a>(player: &FieldedPlayer, game_state: &GameState) -> Vec<Line<'a>> {
    let style = player_style(player, game_state);
    let is_carrier = is_carrier(player, game_state);

    if player.status != PlayerStatus::Up {
        if player.status == PlayerStatus::Down {
            return vec![Line::from(vec![Span::styled("--", style)])];
        }
        return vec![Line::from(vec![Span::styled("✧-", style)])];
    }

    if is_carrier {
        vec![Line::from(vec![Span::styled("☺", style), ball_1x1()])]
    } else {
        vec![Line::from(vec![Span::styled("☺-", style)])]
    }
}

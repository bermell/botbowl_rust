use botbowl_engine::core::{
    gamestate::GameState,
    model::{BallState, FieldedPlayer, PlayerStatus, TeamType},
};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

fn player_style(player: &FieldedPlayer, game_state: &GameState) -> Style {
    let is_active_player = is_active(player, game_state);
    let is_used = player.used && !is_active_player;

    let fg_team = match (player.stats.team, is_used) {
        (TeamType::Home, false) => Color::Red,
        (TeamType::Home, true) => Color::Indexed(52),
        (TeamType::Away, false) => Color::LightBlue,
        (TeamType::Away, true) => Color::Indexed(22),
    };

    Style::default().fg(fg_team)
}

fn is_active(player: &FieldedPlayer, game_state: &GameState) -> bool {
    game_state
        .get_active_player()
        .map_or(false, |active| active.id == player.id)
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
    let border_style = Style::default().fg(Color::White);
    let border = match is_active(player, game_state) {
        true => Span::styled("|", border_style),
        false => Span::styled(" ", border_style),
    };

    let fm =
        |s: &'static str| Line::from(vec![border.clone(), Span::styled(s, style), border.clone()]);

    if player.status != PlayerStatus::Up {
        let empty = fm("      ");
        let star_ = fm("***   ");
        let l3___ = fm("▄▄▁▃▃▬");
        let l4___ = fm("▀▀▔▀▀▬");
        if player.status == PlayerStatus::Down {
            return vec![empty.clone(), empty, l3___, l4___];
        }
        return vec![empty, star_, l3___, l4___];
    }

    let l1 = fm("  ▆▆  ");
    let l2 = fm("--▐▌--");
    let l3 = fm("  ▐▌  ");
    let l4 = fm(" /  \\ ");

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
    let border_style = Style::default().fg(Color::White);
    let border = match is_active(player, game_state) {
        true => Span::styled("|", border_style),
        false => Span::styled(" ", border_style),
    };

    let fm =
        |s: &'static str| Line::from(vec![border.clone(), Span::styled(s, style), border.clone()]);

    if player.status != PlayerStatus::Up {
        let empty = fm("    ");
        let star = fm("*** ");
        let l2 = fm("▄▁▃▬");
        let l3 = fm("▀▔▀▬");
        if player.status == PlayerStatus::Down {
            return vec![empty, l2, l3];
        }
        return vec![star, l2, l3];
    }

    let l1 = fm(" ▆▆ ");
    let l2 = fm("-▐▌-");
    let l3 = fm(" /\\ ");

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

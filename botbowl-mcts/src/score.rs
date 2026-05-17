use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{BallState, TeamType};

/// MVP leaf-scoring ladder, scored from the Home team's perspective.
/// Returns a signed integer — positive favours Home, negative favours
/// Away. Three tiers in the MVP (tier-3 / player value × health is the
/// remaining follow-up):
///
/// 1. **Game score**, weighted ×1000. The dominant signal — a touchdown
///    swamps every other consideration.
/// 2. **Ball control category**, weighted ×10. Mid-strength navigation
///    signal: keeping the ball matters more than position drift.
/// 3. **Carrier distance to endzone**, unweighted (×1). Fine-grained
///    nudge that gives the search a gradient *between* states that
///    differ only in carrier position. Without it the leaf score is
///    constant for any in-turn move and UCT can't differentiate.
pub fn leaf_score(state: &GameState) -> i64 {
    let score_delta = state.home.score as i64 - state.away.score as i64;
    let ball_tier = ball_control_value(state);
    let carrier_tier = carrier_distance_value(state);
    score_delta * 1000 + ball_tier * 10 + carrier_tier
}

/// Tier-4: closer-to-own-endzone is better when Home carries the ball,
/// worse when Away does. Returns a value in roughly [-26, +26] (pitch
/// width is 28).
fn carrier_distance_value(state: &GameState) -> i64 {
    let BallState::Carried(id) = state.ball else {
        return 0;
    };
    let Ok(carrier) = state.get_player(id) else {
        return 0;
    };
    let endzone_x = state.get_endzone_x(carrier.stats.team) as i64;
    let dist = (carrier.position.x as i64 - endzone_x).abs();
    // 26-dist so smaller distances → larger values. Sign flipped when
    // the opponent carries.
    let magnitude = 26 - dist;
    match carrier.stats.team {
        TeamType::Home => magnitude,
        TeamType::Away => -magnitude,
    }
}

fn ball_control_value(state: &GameState) -> i64 {
    match state.ball {
        BallState::Carried(id) => match state.get_player(id).ok().map(|p| p.stats.team) {
            Some(TeamType::Home) => 50,
            Some(TeamType::Away) => -50,
            None => 0,
        },
        BallState::OnGround(ball_pos) => {
            let mut home_adj = 0;
            let mut away_adj = 0;
            for player in state.get_adj_players(ball_pos) {
                match player.stats.team {
                    TeamType::Home => home_adj += 1,
                    TeamType::Away => away_adj += 1,
                }
            }
            match (home_adj, away_adj) {
                (h, 0) if h > 0 => 20,
                (0, a) if a > 0 => -20,
                _ => 0,
            }
        }
        BallState::InAir(_) | BallState::OffPitch => 0,
    }
}

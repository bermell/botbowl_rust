//! Domain-knowledge action priors for PUCT selection.
//!
//! `prior_for` returns a *relative* weight: priors are not normalised
//! here, so individual rules read naturally as the ×N multipliers from
//! the design doc. The PUCT formula in `dynamics::select_node` consumes
//! the raw weight directly — `recon_mcts` does not require priors to sum
//! to one, and PUCT's `c · P · √N(parent) / (1 + N(a))` term is monotonic
//! in `P`.
//!
//! All rules must be a pure function of `(state, action)` — same input,
//! same output. The selection step calls this lazily on every descent
//! rather than storing the prior on the score, because the engine's
//! `GameState` doesn't pass cheaply through `score_leaf`.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, BallState};
use botbowl_engine::core::table::{PosAT, SimpleAT};

use crate::action::BbAction;

const BASE: f32 = 1.0;
const W_PICKUP_BALL: f32 = 10.0;
const W_BLITZ_CARRIER: f32 = 10.0;
const W_MARK_CARRIER: f32 = 5.0;
const W_CARRIER_TOWARDS_ENDZONE: f32 = 5.0;
const W_END_TURN: f32 = 0.2;

/// Returns a non-negative relative weight to bias selection toward
/// domain-good actions. Chance actions always return [`BASE`] — priors
/// don't influence stochastic outcome selection in v1.
pub fn prior_for(state: &GameState, action: &BbAction) -> f32 {
    let engine_action = match action {
        BbAction::Player(a) => *a,
        BbAction::Chance { .. } => return BASE,
    };

    // ×0.2 — discourage turn-ending choices when domain-good actions are
    // available. Always applied (rather than gating on "are there other
    // siblings?") so this stays a pure function of (state, action); when
    // EndTurn is genuinely the only legal action, the absolute weight is
    // irrelevant — PUCT is invariant to a uniform scale of priors.
    if matches!(
        engine_action,
        EngineAction::Simple(SimpleAT::EndPlayerTurn) | EngineAction::Simple(SimpleAT::EndTurn)
    ) {
        return W_END_TURN;
    }

    let (action_type, dest) = match engine_action {
        EngineAction::Positional(at, pos) => (at, pos),
        EngineAction::Simple(_) => return BASE,
    };

    let agent_team = match state.available_actions.team {
        Some(t) => t,
        None => return BASE,
    };

    let active_pos = state.get_active_player().map(|p| p.position);
    let active_id = state.info.active_player;

    let mut prior = BASE;

    // Rule ×10 — pickup: Move onto a free ball.
    if action_type == PosAT::Move {
        if let BallState::OnGround(ball_pos) = state.ball {
            if dest == ball_pos {
                prior *= W_PICKUP_BALL;
            }
        }
    }

    // Rule ×10 — blitz the ball carrier: Block ending on opponent carrier.
    if action_type == PosAT::Block {
        if let BallState::Carried(carrier_id) = state.ball {
            if let Ok(carrier) = state.get_player(carrier_id) {
                if carrier.position == dest && carrier.stats.team != agent_team {
                    prior *= W_BLITZ_CARRIER;
                }
            }
        }
    }

    // Rule ×5 — move to mark the opponent carrier: Move ending adjacent
    // to (but not on top of) the opposing carrier.
    if action_type == PosAT::Move {
        if let BallState::Carried(carrier_id) = state.ball {
            if let Ok(carrier) = state.get_player(carrier_id) {
                if carrier.stats.team != agent_team
                    && dest != carrier.position
                    && dest.distance_to(&carrier.position) == 1
                {
                    prior *= W_MARK_CARRIER;
                }
            }
        }
    }

    // Rule ×5 — move ball carrier toward own endzone: only fires when
    // the active player IS the ball carrier and the destination's x is
    // strictly closer to the agent's endzone x.
    if action_type == PosAT::Move {
        if let (BallState::Carried(carrier_id), Some(aid)) = (state.ball, active_id) {
            if carrier_id == aid {
                if let Some(from) = active_pos {
                    let endzone_x = state.get_endzone_x(agent_team) as i64;
                    let from_dist = (from.x as i64 - endzone_x).abs();
                    let to_dist = (dest.x as i64 - endzone_x).abs();
                    if to_dist < from_dist {
                        prior *= W_CARRIER_TOWARDS_ENDZONE;
                    }
                }
            }
        }
    }

    prior
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{BbAction, ChanceOutcome};
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{Action as EA, BallState, Position, TeamType};
    use botbowl_engine::core::table::{PosAT, SimpleAT};

    fn pa(a: EA) -> BbAction {
        BbAction::Player(a)
    }

    /// Build a Home-turn state with one Home and one Away player and the
    /// ball where the caller wants it. Active player set explicitly.
    fn state_with(
        home_pos: Position,
        away_pos: Position,
        ball: BallState,
        active_team: TeamType,
        active_at: Position,
    ) -> GameState {
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        state.ball = ball;
        state.available_actions.team = Some(active_team);
        let id = state.get_player_id_at(active_at).unwrap();
        state.set_active_player(id);
        state
    }

    #[test]
    fn chance_actions_get_base() {
        let state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        let a = BbAction::chance(ChanceOutcome::Pass, 0.5);
        assert_eq!(prior_for(&state, &a), BASE);
    }

    #[test]
    fn end_player_turn_gets_low_prior() {
        let state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        let a = pa(EA::Simple(SimpleAT::EndPlayerTurn));
        assert_eq!(prior_for(&state, &a), W_END_TURN);
    }

    #[test]
    fn end_turn_gets_low_prior() {
        let state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        let a = pa(EA::Simple(SimpleAT::EndTurn));
        assert_eq!(prior_for(&state, &a), W_END_TURN);
    }

    #[test]
    fn pickup_ball_boosts_move_onto_ball() {
        let ball_pos = Position::new((10, 7));
        let state = state_with(
            Position::new((5, 5)),
            Position::new((20, 7)),
            BallState::OnGround(ball_pos),
            TeamType::Home,
            Position::new((5, 5)),
        );
        let a = pa(EA::Positional(PosAT::Move, ball_pos));
        assert_eq!(prior_for(&state, &a), W_PICKUP_BALL);
    }

    #[test]
    fn move_to_non_ball_square_is_base() {
        let state = state_with(
            Position::new((5, 5)),
            Position::new((20, 7)),
            BallState::OnGround(Position::new((10, 7))),
            TeamType::Home,
            Position::new((5, 5)),
        );
        let a = pa(EA::Positional(PosAT::Move, Position::new((6, 5))));
        assert_eq!(prior_for(&state, &a), BASE);
    }

    #[test]
    fn block_on_opposing_carrier_is_blitz_carrier() {
        let mut state = state_with(
            Position::new((5, 5)),
            Position::new((6, 5)),
            BallState::OffPitch,
            TeamType::Home,
            Position::new((5, 5)),
        );
        let away_id = state.get_player_id_at(Position::new((6, 5))).unwrap();
        state.ball = BallState::Carried(away_id);
        let a = pa(EA::Positional(PosAT::Block, Position::new((6, 5))));
        assert_eq!(prior_for(&state, &a), W_BLITZ_CARRIER);
    }

    #[test]
    fn move_adjacent_to_opposing_carrier_marks() {
        let mut state = state_with(
            Position::new((5, 5)),
            Position::new((10, 5)),
            BallState::OffPitch,
            TeamType::Home,
            Position::new((5, 5)),
        );
        let away_id = state.get_player_id_at(Position::new((10, 5))).unwrap();
        state.ball = BallState::Carried(away_id);
        // Move to (9,5) — adjacent to the carrier at (10,5).
        let a = pa(EA::Positional(PosAT::Move, Position::new((9, 5))));
        assert_eq!(prior_for(&state, &a), W_MARK_CARRIER);
    }

    #[test]
    fn move_carrier_closer_to_endzone() {
        // Home endzone is at x=1. Carrier starts at (5,7), moves to (4,7).
        let mut state = state_with(
            Position::new((5, 7)),
            Position::new((20, 7)),
            BallState::OffPitch,
            TeamType::Home,
            Position::new((5, 7)),
        );
        let home_id = state.get_player_id_at(Position::new((5, 7))).unwrap();
        state.ball = BallState::Carried(home_id);
        let a = pa(EA::Positional(PosAT::Move, Position::new((4, 7))));
        assert_eq!(prior_for(&state, &a), W_CARRIER_TOWARDS_ENDZONE);
    }

    #[test]
    fn move_carrier_away_from_endzone_is_base() {
        let mut state = state_with(
            Position::new((5, 7)),
            Position::new((20, 7)),
            BallState::OffPitch,
            TeamType::Home,
            Position::new((5, 7)),
        );
        let home_id = state.get_player_id_at(Position::new((5, 7))).unwrap();
        state.ball = BallState::Carried(home_id);
        // Home endzone x=1; (6,7) is *farther* from it than (5,7).
        let a = pa(EA::Positional(PosAT::Move, Position::new((6, 7))));
        assert_eq!(prior_for(&state, &a), BASE);
    }

    #[test]
    fn non_carrier_move_does_not_get_carrier_boost() {
        let mut state = state_with(
            Position::new((5, 7)),
            Position::new((20, 7)),
            BallState::OffPitch,
            TeamType::Home,
            Position::new((5, 7)),
        );
        // Carrier is the AWAY player, but active is HOME. Active is not
        // the carrier so the carrier-toward-endzone rule must not fire.
        let away_id = state.get_player_id_at(Position::new((20, 7))).unwrap();
        state.ball = BallState::Carried(away_id);
        let a = pa(EA::Positional(PosAT::Move, Position::new((4, 7))));
        // Not adjacent to away carrier, not a pickup, not a block —
        // expect base.
        assert_eq!(prior_for(&state, &a), BASE);
    }

    #[test]
    fn pickup_and_toward_endzone_compose() {
        // The ball is on the ground in front of the carrier? No — pickup
        // implies the active player is *not* yet the carrier. Carrier
        // boost requires Carried + active==carrier, so pickup and carrier
        // boosts are mutually exclusive by construction. This test
        // documents that mutual exclusivity rather than asserting both
        // multipliers stack.
        let ball_pos = Position::new((4, 7));
        let state = state_with(
            Position::new((5, 7)),
            Position::new((20, 7)),
            BallState::OnGround(ball_pos),
            TeamType::Home,
            Position::new((5, 7)),
        );
        let a = pa(EA::Positional(PosAT::Move, ball_pos));
        assert_eq!(prior_for(&state, &a), W_PICKUP_BALL);
    }
}

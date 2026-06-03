//! Domain-knowledge action pruning applied inside `available_actions`.
//!
//! Every rule here MUST be a pure function of `(state, action)`. recon_mcts
//! recombines nodes by state hash; if two paths to the same state return
//! different subsets of legal actions the DAG silently splits, breaking
//! recombination. See `recon_mcts/src/tree.rs:462-492`.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{other_team, Action as EngineAction, BallState, PlayerStatus, Position};
use botbowl_engine::core::table::{PosAT, SimpleAT};

/// Returns true when the engine-legal `action` should be hidden from MCTS.
pub fn should_prune(state: &GameState, action: &EngineAction) -> bool {
    prune_end_player_turn_before_any_move(state, action)
        || prune_off_ball_when_pass_or_handoff(state, action)
        || prune_handoff_dest_must_neighbour_teammate(state, action)
        || prune_blitz_dest_must_neighbour_opponent(state, action)
        || prune_start_handoff_without_recipient(state, action)
}

/// **P1** — disallow ending a player's turn immediately after activating
/// them, before any movement has been made. Without this the search wastes
/// huge fan-out on "activate → end turn" no-op branches that produce a
/// state indistinguishable from never activating that player.
fn prune_end_player_turn_before_any_move(state: &GameState, action: &EngineAction) -> bool {
    if !matches!(action, EngineAction::Simple(SimpleAT::EndPlayerTurn)) {
        return false;
    }
    match state.get_active_player() {
        Some(p) => p.moves == 0,
        None => false,
    }
}

/// **P2 / P3** — when the active player was activated with `StartPass` or
/// `StartHandoff` but doesn't have the ball, the only sensible destination
/// is the ball itself. Any positional action targeting some other square
/// is wasted activation (the engine will still want a pass/handoff next).
///
/// Pure on `(state, action)` because `player_action_type` and
/// `BallState::OnGround(...)` are both fields read straight off `state`.
fn prune_off_ball_when_pass_or_handoff(state: &GameState, action: &EngineAction) -> bool {
    let dest = match action {
        EngineAction::Positional(_, pos) => *pos,
        EngineAction::Simple(_) => return false,
    };
    let action_type = match state.info.player_action_type {
        Some(at) => at,
        None => return false,
    };
    if action_type != PosAT::StartPass && action_type != PosAT::StartHandoff {
        return false;
    }
    // The player is the carrier — pass/handoff path filtering doesn't apply
    // here; that's P4's job for handoff and a similar pass-throw filter
    // would belong elsewhere.
    let carrier_is_active = matches!(
        (state.ball, state.info.active_player),
        (BallState::Carried(cid), Some(aid)) if cid == aid
    );
    if carrier_is_active {
        return false;
    }
    let ball_pos = match state.ball {
        BallState::OnGround(p) => p,
        // No on-ground ball: pickup isn't an option this turn; treat as
        // "wasted activation either way" and prune any positional action.
        _ => return true,
    };
    dest != ball_pos
}

/// **P4** — when the active player was activated with `StartHandoff` and
/// is carrying the ball, only allow movement destinations that end
/// adjacent to a standing teammate (a valid hand-off recipient). Anywhere
/// else and the activation has no follow-through.
///
/// `PosAT::Handoff` itself is not a Move — it's the throw — and is left
/// alone here.
fn prune_handoff_dest_must_neighbour_teammate(state: &GameState, action: &EngineAction) -> bool {
    if state.info.player_action_type != Some(PosAT::StartHandoff) {
        return false;
    }
    let (at, dest) = match action {
        EngineAction::Positional(at, pos) => (*at, *pos),
        EngineAction::Simple(_) => return false,
    };
    if at != PosAT::Move {
        return false;
    }
    let active_id = match state.info.active_player {
        Some(id) => id,
        None => return false,
    };
    // Only fire when the carrier is the one moving — pre-pickup is
    // already handled by P3.
    let carrying = matches!(state.ball, BallState::Carried(cid) if cid == active_id);
    if !carrying {
        return false;
    }
    let agent_team = match state.get_active_player() {
        Some(p) => p.stats.team,
        None => return false,
    };
    !has_standing_teammate_adjacent(state, dest, agent_team, active_id)
}

/// **P5** — when the active player was activated with `StartBlitz` and
/// hasn't yet resolved the blitz block, only allow movement destinations
/// that end adjacent to an opposing player. (Block actions themselves are
/// left untouched.) After the block resolves the engine flips
/// `player_action_type` to `StartMove` (see `block_procs.rs:336`) and
/// this rule disengages.
fn prune_blitz_dest_must_neighbour_opponent(state: &GameState, action: &EngineAction) -> bool {
    if state.info.player_action_type != Some(PosAT::StartBlitz) {
        return false;
    }
    let (at, dest) = match action {
        EngineAction::Positional(at, pos) => (*at, *pos),
        EngineAction::Simple(_) => return false,
    };
    if at != PosAT::Move {
        return false;
    }
    let agent_team = match state.get_active_player() {
        Some(p) => p.stats.team,
        None => return false,
    };
    !has_opponent_adjacent(state, dest, agent_team)
}

/// **P7** — `StartHandoff` is a strictly worse activation when there
/// is no other standing teammate on the pitch (the player can never
/// actually hand off the ball). This rule blocks MCTS from picking
/// it upstream so the search doesn't waste budget burrowing into a
/// state where P4 will have nothing to recommend.
///
/// Compared to P4 this is a coarser check (any standing teammate
/// anywhere, not "reachable adjacent teammate"), but it cleanly
/// rejects the pathological case the curriculum's single-player
/// lectures land on.
fn prune_start_handoff_without_recipient(state: &GameState, action: &EngineAction) -> bool {
    if !matches!(action, EngineAction::Positional(PosAT::StartHandoff, _)) {
        return false;
    }
    // The active player at this point is whoever was about to be
    // activated — not yet `state.info.active_player` (that's set by
    // the StartHandoff procedure itself once accepted). We rely on
    // the action's team via `available_actions.team` here.
    let team = match state.available_actions.team {
        Some(t) => t,
        None => return false,
    };
    let candidate_id = match action {
        EngineAction::Positional(_, pos) => state.get_player_at(*pos).map(|p| p.id),
        EngineAction::Simple(_) => None,
    };
    state
        .get_players_on_pitch_in_team(team)
        .all(|p| p.status != PlayerStatus::Up || Some(p.id) == candidate_id)
}

fn has_standing_teammate_adjacent(
    state: &GameState,
    dest: Position,
    agent_team: botbowl_engine::core::model::TeamType,
    exclude_id: botbowl_engine::core::model::PlayerID,
) -> bool {
    state
        .get_players_on_pitch_in_team(agent_team)
        .any(|p| p.id != exclude_id && p.status == PlayerStatus::Up && p.position.distance_to(&dest) == 1)
}

fn has_opponent_adjacent(
    state: &GameState,
    dest: Position,
    agent_team: botbowl_engine::core::model::TeamType,
) -> bool {
    let opponent = other_team(agent_team);
    state
        .get_players_on_pitch_in_team(opponent)
        .any(|p| p.status == PlayerStatus::Up && p.position.distance_to(&dest) == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{Action as EA, Position};
    use botbowl_engine::core::table::{PosAT, SimpleAT};

    /// Sets up a Home-turn state with a single Home player on the pitch
    /// and that player activated via StartMove, mimicking the engine state
    /// right after a `StartMove` action and before any move action.
    fn state_just_activated() -> GameState {
        let mut state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        // Drive engine into the active-player state by issuing a StartMove
        // on the home player. We don't care which exact path the engine
        // takes — just that get_active_player() returns Some and moves==0.
        let id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(id);
        state
    }

    #[test]
    fn end_player_turn_pruned_when_active_player_has_not_moved() {
        let state = state_just_activated();
        let action = EA::Simple(SimpleAT::EndPlayerTurn);
        assert!(should_prune(&state, &action));
    }

    #[test]
    fn end_player_turn_allowed_once_player_has_moved() {
        let mut state = state_just_activated();
        state.get_active_player_mut().unwrap().moves = 1;
        let action = EA::Simple(SimpleAT::EndPlayerTurn);
        assert!(!should_prune(&state, &action));
    }

    #[test]
    fn end_player_turn_allowed_when_no_active_player() {
        let state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        let action = EA::Simple(SimpleAT::EndPlayerTurn);
        // No active player → P1 doesn't apply.
        assert!(!should_prune(&state, &action));
    }

    #[test]
    fn end_turn_is_not_pruned_by_p1() {
        // EndTurn (whole-team) is distinct from EndPlayerTurn; P1 only
        // covers EndPlayerTurn — ending the entire turn after activating
        // is a legitimate strategic choice.
        let state = state_just_activated();
        let action = EA::Simple(SimpleAT::EndTurn);
        assert!(!should_prune(&state, &action));
    }

    #[test]
    fn other_actions_pass_through() {
        let state = state_just_activated();
        let m = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(!should_prune(&state, &m));
    }

    /// Determinism check — same state + same action always yields same
    /// answer, never reads RNG.
    #[test]
    fn deterministic_on_state() {
        let state = state_just_activated();
        let action = EA::Simple(SimpleAT::EndPlayerTurn);
        let a = should_prune(&state, &action);
        let b = should_prune(&state, &action);
        let c = should_prune(&state, &action);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    // --- P2 / P3: pass / handoff off-ball path filter -----------------

    /// Active home player at (5,5), ball on ground at (10,7), player
    /// activated under `action_type`. P2/P3 should let the player move to
    /// the ball square but prune any other Move destination.
    fn off_ball_state(action_type: PosAT) -> (GameState, Position) {
        let ball_pos = Position::new((10, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        state.ball = botbowl_engine::core::model::BallState::OnGround(ball_pos);
        let id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(id);
        state.info.player_action_type = Some(action_type);
        (state, ball_pos)
    }

    #[test]
    fn pass_mode_off_ball_move_to_ball_passes() {
        let (state, ball_pos) = off_ball_state(PosAT::StartPass);
        let a = EA::Positional(PosAT::Move, ball_pos);
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn pass_mode_off_ball_move_to_non_ball_pruned() {
        let (state, _) = off_ball_state(PosAT::StartPass);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn handoff_mode_off_ball_move_to_ball_passes() {
        let (state, ball_pos) = off_ball_state(PosAT::StartHandoff);
        let a = EA::Positional(PosAT::Move, ball_pos);
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn handoff_mode_off_ball_move_to_non_ball_pruned() {
        let (state, _) = off_ball_state(PosAT::StartHandoff);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn move_mode_off_ball_not_pruned() {
        // Without a pass/handoff activation we don't apply the filter,
        // even if the player is off-ball.
        let (mut state, _) = off_ball_state(PosAT::StartMove);
        state.info.player_action_type = Some(PosAT::StartMove);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(!should_prune(&state, &a));
    }

    // --- P4: handoff destination must neighbour a standing teammate ---

    #[test]
    fn handoff_carrier_dest_adjacent_to_teammate_passes() {
        // Home carrier at (5,5); friendly receiver at (10,7). Move to
        // (10,6) — distance 1 from the receiver — should be allowed.
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(5, 5), (10, 7)])
            .add_away_player(Position::new((20, 10)))
            .build();
        let carrier_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.ball = botbowl_engine::core::model::BallState::Carried(carrier_id);
        state.set_active_player(carrier_id);
        state.info.player_action_type = Some(PosAT::StartHandoff);
        let a = EA::Positional(PosAT::Move, Position::new((10, 6)));
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn handoff_carrier_dest_not_adjacent_pruned() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(5, 5), (10, 7)])
            .add_away_player(Position::new((20, 10)))
            .build();
        let carrier_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.ball = botbowl_engine::core::model::BallState::Carried(carrier_id);
        state.set_active_player(carrier_id);
        state.info.player_action_type = Some(PosAT::StartHandoff);
        // (15,2) is nowhere near the receiver at (10,7).
        let a = EA::Positional(PosAT::Move, Position::new((15, 2)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn handoff_carrier_dest_adjacent_to_self_not_a_teammate() {
        // Only teammate is the active carrier itself — we exclude the
        // active player, so no eligible recipient and the move is pruned.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        let carrier_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.ball = botbowl_engine::core::model::BallState::Carried(carrier_id);
        state.set_active_player(carrier_id);
        state.info.player_action_type = Some(PosAT::StartHandoff);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    // --- P5: blitz destination must neighbour an opponent --------------

    #[test]
    fn blitz_mode_dest_adjacent_to_opponent_passes() {
        // Home at (5,5), Away at (10,10). Move to (9,10) — adjacent.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((10, 10)))
            .build();
        let home_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(home_id);
        state.info.player_action_type = Some(PosAT::StartBlitz);
        let a = EA::Positional(PosAT::Move, Position::new((9, 10)));
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_dest_not_adjacent_to_opponent_pruned() {
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        let home_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(home_id);
        state.info.player_action_type = Some(PosAT::StartBlitz);
        // (6,5) is nowhere near the only opponent at (20,20).
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_block_action_not_pruned() {
        // Block actions in blitz mode are the whole point — P5 only
        // touches Move actions.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((6, 5)))
            .build();
        let home_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(home_id);
        state.info.player_action_type = Some(PosAT::StartBlitz);
        let a = EA::Positional(PosAT::Block, Position::new((6, 5)));
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_disengages_after_block_resolution() {
        // Engine flips player_action_type to StartMove once the blitz
        // block is resolved (block_procs.rs:336). P5 must stop pruning.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        let home_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(home_id);
        state.info.player_action_type = Some(PosAT::StartMove);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(!should_prune(&state, &a));
    }

    // --- P7: StartHandoff requires at least one recipient teammate --

    #[test]
    fn start_handoff_pruned_when_no_teammate_exists() {
        // Only one home player on pitch — nobody to receive a handoff.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        state.available_actions.team = Some(botbowl_engine::core::model::TeamType::Home);
        let a = EA::Positional(PosAT::StartHandoff, Position::new((5, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn start_handoff_allowed_with_another_teammate_up() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(5, 5), (10, 7)])
            .add_away_player(Position::new((20, 10)))
            .build();
        state.available_actions.team = Some(botbowl_engine::core::model::TeamType::Home);
        let a = EA::Positional(PosAT::StartHandoff, Position::new((5, 5)));
        assert!(!should_prune(&state, &a));
    }

    /// Determinism check for P5 — pure on state.
    #[test]
    fn p5_blitz_pruning_is_deterministic() {
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((20, 10)))
            .build();
        let home_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.set_active_player(home_id);
        state.info.player_action_type = Some(PosAT::StartBlitz);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
    }
}

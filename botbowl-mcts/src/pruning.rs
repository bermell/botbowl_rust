//! Domain-knowledge action pruning applied inside `available_actions`.
//!
//! Every rule here MUST be a pure function of `(state, action)`. recon_mcts
//! recombines nodes by state hash; if two paths to the same state return
//! different subsets of legal actions the DAG silently splits, breaking
//! recombination. See `recon_mcts/src/tree.rs:462-492`.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::SimpleAT;

/// Returns true when the engine-legal `action` should be hidden from MCTS.
pub fn should_prune(state: &GameState, action: &EngineAction) -> bool {
    prune_end_player_turn_before_any_move(state, action)
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
}

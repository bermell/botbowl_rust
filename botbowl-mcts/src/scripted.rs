//! Scripted player-action picks applied inside `apply_action`.
//!
//! Every helper here is a pure function of `GameState` that returns
//! `Some(action)` when the state offers a decision MCTS shouldn't
//! waste search budget on (the rules effectively settle it for us),
//! or `None` to let normal MCTS expansion handle the node.
//!
//! These are stitched together by [`scripted_player_pick`], which
//! returns the first scripted choice that applies. The
//! `apply_action` quiescent loop calls that helper after every engine
//! step; when it returns `Some`, the engine is advanced again before
//! returning to MCTS.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::SimpleAT;

use crate::block_dice;

/// Returns a single scripted engine action if one applies to the
/// current player-decision state, otherwise `None`. Caller must
/// already have checked that the state is at a player decision (no
/// `pending_roll`, not `game_over`).
pub fn scripted_player_pick(state: &GameState) -> Option<EngineAction> {
    if let Some(a) = block_dice::scripted_pick(state) {
        return Some(a);
    }
    if let Some(a) = coin_toss_pick(state) {
        return Some(a);
    }
    if let Some(a) = kick_receive_pick(state) {
        return Some(a);
    }
    None
}

/// Coin toss: always pick `Heads`. The win/loss is 50/50 so the
/// choice doesn't matter strategically — pinning it removes a
/// pointless 2-way branch.
pub fn coin_toss_pick(state: &GameState) -> Option<EngineAction> {
    let simple = state.available_actions.get_simple();
    if simple.contains(&SimpleAT::Heads) && simple.contains(&SimpleAT::Tails) {
        return Some(EngineAction::Simple(SimpleAT::Heads));
    }
    None
}

/// Kick-or-receive after winning the toss: always `Receive`.
/// Receiving lets the agent score on the first drive of the half;
/// against typical curriculum lectures this is uniformly the right
/// call.
pub fn kick_receive_pick(state: &GameState) -> Option<EngineAction> {
    let simple = state.available_actions.get_simple();
    if simple.contains(&SimpleAT::Receive) && simple.contains(&SimpleAT::Kick) {
        return Some(EngineAction::Simple(SimpleAT::Receive));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::{BuilderState, GameStateBuilder};

    #[test]
    fn coin_toss_state_collapses_to_heads() {
        let state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
        let pick = coin_toss_pick(&state).expect("coin toss state should be picked");
        assert_eq!(pick, EngineAction::Simple(SimpleAT::Heads));
        // The aggregate dispatcher prefers block-die over coin toss in
        // principle, but block-die isn't on offer here — the result
        // should also be Heads.
        let pick_any = scripted_player_pick(&state).expect("scripted dispatcher should fire");
        assert_eq!(pick_any, EngineAction::Simple(SimpleAT::Heads));
    }

    #[test]
    fn non_scripted_state_returns_none() {
        // A plain post-kickoff turn state offers genuine choices (Move,
        // Block, EndTurn, ...) — none of the scripted helpers should fire.
        let state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(botbowl_engine::core::model::Position::new((5, 5)))
            .add_away_player(botbowl_engine::core::model::Position::new((20, 10)))
            .build();
        assert!(scripted_player_pick(&state).is_none());
    }
}

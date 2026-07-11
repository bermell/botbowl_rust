//! Domain-knowledge action pruning applied inside `available_actions`.
//!
//! Every rule here MUST be a pure function of `(state, action)`. recon_mcts
//! recombines nodes by state hash; if two paths to the same state return
//! different subsets of legal actions the DAG silently splits, breaking
//! recombination. See `recon_mcts/src/tree.rs:462-492`.
//!
//! ## Rule registry
//!
//! Rules are dispatched by action type so it's immediately clear what each
//! rule targets:
//!
//! | Action                      | Rules applied          |
//! |-----------------------------|------------------------|
//! | `Simple(EndPlayerTurn)`     | P1                     |
//! | `Positional(StartHandoff)`  | P7                     |
//! | `Positional(Move)`          | P2/P3, P4, P5, P8      |
//! | everything else             | *(none)*               |

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{other_team, Action as EngineAction, BallState, PlayerStatus, Position};
use botbowl_engine::core::table::{PosAT, SimpleAT};

/// Returns true when the engine-legal `action` should be hidden from MCTS.
pub fn should_prune(state: &GameState, action: &EngineAction) -> bool {
    match action {
        EngineAction::Simple(SimpleAT::EndPlayerTurn) => prune_end_player_turn_before_any_move(state),
        EngineAction::Positional(PosAT::StartHandoff, pos) => prune_start_handoff_pass(state, *pos),
        EngineAction::Positional(PosAT::StartPass, pos) => prune_start_handoff_pass(state, *pos),
        EngineAction::Positional(PosAT::StartBlitz, _) => prune_start_blitz(state),
        EngineAction::Positional(PosAT::Move, pos) => prune_move_action(state, *pos),
        _ => false,
    }
}

fn prune_move_action(state: &GameState, pos: Position) -> bool {
    prune_off_ball_when_pass_or_handoff(state, pos)
        || prune_move_action_when_ball_carrier_if_start_handoff_pass(state)
        || prune_move_when_blitzing(state, pos)
        || prune_redundant_move_after_first(state)
}
/// **P1** — disallow ending a player's turn immediately after activating
/// them, before any movement has been made. Without this the search wastes
/// huge fan-out on "activate → end turn" no-op branches that produce a
/// state indistinguishable from never activating that player.
fn prune_end_player_turn_before_any_move(state: &GameState) -> bool {
    match state.get_active_player() {
        Some(p) => p.moves == 0,
        None => false,
    }
}

fn prune_start_handoff_pass(state: &GameState, pos: Position) -> bool {
    let activating_player = state.get_player_id_at(pos).unwrap();
    match state.ball {
        BallState::OnGround(_) => prune_start_handoff_pass_without_recipient(state, pos), // Ball on ground, allow
        BallState::Carried(carrier_id) if carrier_id == activating_player => {
            prune_start_handoff_pass_without_recipient(state, pos)
        }
        BallState::Carried(_) => true, // not allowed, you are not carrier
        BallState::OffPitch => true,
        BallState::InAir(_) => true,
    }
}

//fn prune_start_blitz
fn prune_start_blitz(state: &GameState) -> bool {
    let agent_team = match state.get_active_teamtype() {
        Some(team) => team,
        None => unreachable!(),
    };
    !state
        .get_players_on_pitch_in_team(other_team(agent_team))
        .any(|p| p.status == PlayerStatus::Up)
}

/// **P2 / P3** — when the active player was activated with `StartPass` or
/// `StartHandoff` but doesn't have the ball, the only sensible destination
/// is the ball itself. Any `Move` targeting some other square is wasted
/// activation (the engine will still want a pass/handoff next).
///
/// Pure on `(state, dest)` because `player_action_type` and
/// `BallState::OnGround(...)` are both fields read straight off `state`.
fn prune_off_ball_when_pass_or_handoff(state: &GameState, dest: Position) -> bool {
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
        // "wasted activation either way" and prune.
        _ => return true,
    };
    dest != ball_pos
}

/// **P4** — when the active player was activated with `StartHandoff` or `StartPass` and
/// is carrying the ball, disallow `Move` actions
fn prune_move_action_when_ball_carrier_if_start_handoff_pass(state: &GameState) -> bool {
    let at = state.info.player_action_type.unwrap();
    if at != PosAT::StartHandoff && at != PosAT::StartPass {
        return false;
    }
    let active_id = state.info.active_player.unwrap();
    matches!(state.ball, BallState::Carried(cid) if cid == active_id) // ie active player is carrier
}

/// **P5** — when the active player was activated with `StartBlitz`, the only
/// sensible pre-block `Move` is a single positioning move onto a square
/// adjacent to a standing opponent (so `Block` becomes available next). Any
/// other pre-block move — a second one, or one that doesn't reach an
/// opponent — is pruned. A path-style move reaches any reachable square in
/// one action, so one move action is sufficient positioning.
/// After the block resolves the engine flips `player_action_type` to
/// `StartMove` (see `block_procs.rs:336`) and this rule disengages.
fn prune_move_when_blitzing(state: &GameState, dest: Position) -> bool {
    if state.info.player_action_type != Some(PosAT::StartBlitz) {
        return false;
    }
    let active = match state.get_active_player() {
        Some(p) => p,
        None => return true,
    };
    if active.moves > 0 {
        return true;
    }
    let agent_team = match state.get_active_teamtype() {
        Some(team) => team,
        None => return true,
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
fn prune_start_handoff_pass_without_recipient(state: &GameState, candidate_pos: Position) -> bool {
    // The active player at this point is whoever was about to be
    // activated — not yet `state.info.active_player` (that's set by
    // the StartHandoff procedure itself once accepted). We rely on
    // the action's team via `available_actions.team` here.
    let team = state.get_active_teamtype().unwrap();
    let candidate_id = state.get_player_id_at(candidate_pos).unwrap();
    !state
        .get_players_on_pitch_in_team(team)
        .any(|p| p.id != candidate_id && p.status == PlayerStatus::Up)
}

/// **P8** — collapse the "move again" duplication (plan 003 §1). After a
/// plain-move (`StartMove`) player has completed at least one move action,
/// the engine keeps re-offering every reachable square plus `EndPlayerTurn`
/// (`movement_procs.rs` loops `MoveAction` back to `Init`), flooding the
/// tree with permutations that reach the same final position. Once the
/// player has moved, prune every `Move` continuation so only `EndPlayerTurn`
/// survives; the dynamics quiescent loop (`sole_legal_action`) then
/// auto-applies it.
///
/// **Exception — the pickup and blitz bonuses.** A player who *picked up the
/// ball this activation* is owed exactly one follow-up move action: the
/// pathfinder stops a non-carrier's path at the ball square
/// (`pathing.rs::can_continue_expanding`), so they need a fresh
/// carrier-routed path to actually run with it. Likewise a player who just
/// resolved a blitz block is owed exactly one post-block move (e.g. to pick
/// up the ball). The engine records these entitlements in
/// `state.info.pickup_this_activation` / `state.info.blitz_this_activation`
/// and clears each the moment its follow-up move is selected
/// (`movement_procs.rs`) — so each bonus grants exactly one extra move
/// action, not unlimited (if that move itself picks up the ball, the pickup
/// flag re-arms the bonus for one more).
///
/// Why this needs engine flags rather than reading the board: a
/// just-picked-up (or just-post-block) carrier and an already-carrying mover
/// can reach a byte-identical `GameState`, so no pure function of the board
/// alone could distinguish them. The flags make them genuinely distinct
/// states, keeping this rule pure on `(state, action)` and
/// recombination-safe.
///
/// Scoped to `StartMove`, which also covers the post-blitz state (the engine
/// flips `player_action_type` to `StartMove` once a blitz block resolves —
/// `block_procs.rs:336`). Pass/handoff/foul follow-through is handled by
/// P2–P4; the pre-block blitz move is handled by P5.
fn prune_redundant_move_after_first(state: &GameState) -> bool {
    if state.info.player_action_type != Some(PosAT::StartMove) {
        return false;
    }
    let active = match state.get_active_player() {
        Some(p) => p,
        None => return false,
    };
    // Hasn't moved yet — the first move is exactly what we want to keep.
    // (A downed player's standup is bundled into its first move action, so
    // `moves == 0` still correctly means "no move action taken yet".)
    if active.moves == 0 {
        return false;
    }
    if state.info.pickup_this_activation || state.info.blitz_this_activation {
        return false;
    }
    true
}
#[allow(dead_code)]
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

fn has_opponent_adjacent(state: &GameState, dest: Position, agent_team: botbowl_engine::core::model::TeamType) -> bool {
    let opponent = other_team(agent_team);
    state
        .get_players_on_pitch_in_team(opponent)
        .any(|p| p.status == PlayerStatus::Up && p.position.distance_to(&dest) == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::dices::BlockDice;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{Action as EA, Position};
    use botbowl_engine::core::table::{PosAT, SimpleAT};

    /// Sets up a Home-turn state with a single Home player on the pitch
    /// and that player activated via StartMove, mimicking the engine state
    /// right after a `StartMove` action and before any move action.
    fn state_just_activated() -> GameState {
        let mut state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartMove, Position::new((5, 5))));
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
    ///
    /// A second home player at (3,3) satisfies the recipient requirement so
    /// StartPass / StartHandoff are engine-legal before activation.
    fn off_ball_state(action_type: PosAT) -> (GameState, Position) {
        let home_pos = Position::new((5, 5));
        let ball_pos = Position::new((10, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(Position::new((3, 3)))
            .add_away_player(Position::new((20, 10)))
            .add_ball_pos(home_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(action_type, home_pos));
        state.ball = botbowl_engine::core::model::BallState::OnGround(ball_pos);
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
        let (state, _) = off_ball_state(PosAT::StartMove);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(!should_prune(&state, &a));
    }

    // --- P5: blitz destination must neighbour an opponent --------------

    #[test]
    fn blitz_mode_dest_not_adjacent_to_opponent_pruned() {
        let home_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(Position::new((20, 10)))
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        // (6,5) is nowhere near the only opponent at (20,10).
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_dest_adjacent_to_standing_opponent_allowed() {
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        // (6,6) neighbours the standing opponent at (7,7).
        let a = EA::Positional(PosAT::Move, Position::new((6, 6)));
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_dest_adjacent_only_to_down_opponent_pruned() {
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        state.get_mut_player_at_unsafe(opp_pos).status = PlayerStatus::Down;
        let a = EA::Positional(PosAT::Move, Position::new((6, 6)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_second_pre_block_move_pruned() {
        // Only one pre-block positioning move is allowed; once the active
        // player has taken a move action this activation, further moves
        // (even onto squares adjacent to the opponent) are pruned.
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        state.get_active_player_mut().unwrap().moves = 1;
        let a = EA::Positional(PosAT::Move, Position::new((6, 6)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn blitz_mode_block_action_not_pruned() {
        // Block actions in blitz mode are the whole point — P5 only
        // touches Move actions.
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((6, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        let a = EA::Positional(PosAT::Block, opp_pos);
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn start_blitz_pruned_when_no_standing_opponent() {
        // No standing opponent → no valid blitz target → No reason to do StartBlitz
        let opp_pos = Position::new((20, 10));
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(opp_pos)
            .build();
        // set away player down
        let start_blitz_action = EA::Positional(PosAT::StartBlitz, Position::new((5, 5)));
        assert!(!should_prune(&state, &start_blitz_action));
        state.get_mut_player_at_unsafe(opp_pos).status = PlayerStatus::Down;
        assert!(should_prune(&state, &start_blitz_action));
    }

    fn check_prune_and_step(state: &mut GameState, action: EA) {
        assert!(!should_prune(state, &action));
        assert!(state.is_legal_action(&action));
        state.step(action).unwrap();
    }
    fn assert_is_pruned(state: &GameState, action: &EA) {
        assert!(state.is_legal_action(action));
        assert!(should_prune(state, action));
    }
    fn assert_is_pruned_positional(state: &GameState, at: PosAT, pos: Position) {
        assert_is_pruned(state, &EA::Positional(at, pos));
    }

    #[allow(dead_code)]
    fn assert_is_pruned_simple(state: &GameState, at: SimpleAT) {
        assert_is_pruned(state, &EA::Simple(at));
    }

    #[test]
    fn pruned_action_when_passing() {
        let home_pos = Position::new((5, 5));
        let home_pos_2 = Position::new((10, 10));
        let opp_pos = Position::new((20, 10));
        let ball_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(home_pos_2)
            .add_away_player(opp_pos)
            .add_ball_pos(ball_pos)
            .build();

        check_prune_and_step(&mut state, EA::Positional(PosAT::StartPass, home_pos));
        assert_is_pruned_positional(&state, PosAT::Move, Position::new((6, 5)));

        state.fix_d6(6); //pickup
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, ball_pos));
        // assert_is_pruned_positional(&state, PosAT::Move, Position::new((6, 5)));
        state.fix_d6(6); //pass
        state.fix_d6(6); // catch
        check_prune_and_step(&mut state, EA::Positional(PosAT::Pass, home_pos_2));
    }

    #[test]
    fn pruned_action_when_picking_up() {
        let home_pos = Position::new((5, 5));
        let ball_pos = Position::new((7, 7));
        let final_pos = Position::new((10, 10));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_ball_pos(ball_pos)
            .build();

        check_prune_and_step(&mut state, EA::Positional(PosAT::StartMove, home_pos));

        state.fix_d6(6); //pickup
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, ball_pos));

        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, final_pos));

        assert_is_pruned_positional(&state, PosAT::Move, Position::new((11, 11)));
    }
    #[test]
    fn pruned_move_when_blitzing() {
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();

        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));

        // (6,6) is adjacent to the opponent at (7,7) — the single allowed
        // pre-block positioning move.
        assert!(!should_prune(
            &state,
            &EA::Positional(PosAT::Move, Position::new((6, 6)))
        ));

        state.fix_blockdice(BlockDice::Pow);
        check_prune_and_step(&mut state, EA::Positional(PosAT::Block, opp_pos));
        check_prune_and_step(&mut state, EA::Simple(SimpleAT::SelectPow));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Push, opp_pos + (1, 1)));
        state.fix_d6(1);
        state.fix_d6(1);
        check_prune_and_step(&mut state, EA::Positional(PosAT::FollowUp, opp_pos + (-1, -1)));

        // check that we can move once more after resolving the blitz block.
        assert!(!should_prune(&state, &EA::Positional(PosAT::Move, home_pos + (1, 0))));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, home_pos + (1, 0)));
    }

    /// Full target blitz sequence (plan 019): pre-block positioning move →
    /// block → push/follow-up → post-block move onto the ball (bonus move
    /// #1, also clears `blitz_this_activation`) → pickup bonus move #2 →
    /// everything pruned but `EndPlayerTurn`.
    #[test]
    fn full_blitz_sequence_bounded_by_pickup_bonus() {
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let ball_pos = Position::new((8, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .add_ball_pos(ball_pos)
            .build();

        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));

        // Pre-block positioning move onto the one allowed adjacent square.
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, Position::new((6, 6))));

        state.fix_blockdice(BlockDice::Pow);
        check_prune_and_step(&mut state, EA::Positional(PosAT::Block, opp_pos));
        check_prune_and_step(&mut state, EA::Simple(SimpleAT::SelectPow));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Push, opp_pos + (1, 1)));
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        check_prune_and_step(&mut state, EA::Positional(PosAT::FollowUp, opp_pos));

        // Post-block move onto the ball: allowed by the blitz bonus, and
        // picking up re-arms the bonus for one more move.
        state.fix_d6(6); //pickup
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, ball_pos));
        assert!(!state.info.blitz_this_activation);
        assert!(state.info.pickup_this_activation);

        assert!(!should_prune(
            &state,
            &EA::Positional(PosAT::Move, Position::new((9, 7)))
        ));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, Position::new((9, 7))));

        // Both bonuses spent — further moves are pruned, EndPlayerTurn is not.
        assert_is_pruned_positional(&state, PosAT::Move, Position::new((10, 7)));
        assert!(!should_prune(&state, &EA::Simple(SimpleAT::EndPlayerTurn)));
    }

    #[test]
    fn post_block_move_off_ball_grants_no_further_bonus() {
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();

        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, Position::new((6, 6))));

        state.fix_blockdice(BlockDice::Pow);
        check_prune_and_step(&mut state, EA::Positional(PosAT::Block, opp_pos));
        check_prune_and_step(&mut state, EA::Simple(SimpleAT::SelectPow));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Push, opp_pos + (1, 1)));
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        check_prune_and_step(&mut state, EA::Positional(PosAT::FollowUp, opp_pos));

        // Post-block move to a non-ball square: no pickup, so the bonus is
        // spent and no more moves are owed.
        check_prune_and_step(&mut state, EA::Positional(PosAT::Move, Position::new((9, 7))));
        assert!(!state.info.blitz_this_activation);
        assert!(!state.info.pickup_this_activation);

        assert_is_pruned_positional(&state, PosAT::Move, Position::new((10, 7)));
        assert!(!should_prune(&state, &EA::Simple(SimpleAT::EndPlayerTurn)));
    }

    #[test]
    fn start_handoff_pass_pruned_when_opponent_carry() {
        // Oppenent carries ball -> no reason to START_HANDOFF or START_PASS
        let home_pos = Position::new((5, 5));
        let home_pos_2 = Position::new((6, 6));
        let opp_pos = Position::new((20, 10));
        let state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(home_pos_2)
            .add_away_player(opp_pos)
            .add_ball_pos(opp_pos)
            .build();
        let start_handoff = EA::Positional(PosAT::StartHandoff, home_pos);
        let start_pass = EA::Positional(PosAT::StartPass, home_pos);
        let start_blitz = EA::Positional(PosAT::StartBlitz, home_pos);
        let start_move = EA::Positional(PosAT::StartMove, home_pos);

        assert!(!should_prune(&state, &start_blitz));
        assert!(!should_prune(&state, &start_move));
        assert!(should_prune(&state, &start_handoff));
        assert!(should_prune(&state, &start_pass));
    }

    #[test]
    fn blitz_mode_disengages_after_block_resolution() {
        // Engine flips player_action_type to StartMove once the blitz
        // block is resolved (block_procs.rs:336). P5 must stop pruning.
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        state.fix_blockdice(BlockDice::Pow);
        check_prune_and_step(&mut state, EA::Positional(PosAT::Block, opp_pos));
        check_prune_and_step(&mut state, EA::Simple(SimpleAT::SelectPow));
        check_prune_and_step(&mut state, EA::Positional(PosAT::Push, opp_pos + (1, 1)));
        state.fix_d6(1);
        state.fix_d6(1);
        check_prune_and_step(&mut state, EA::Positional(PosAT::FollowUp, opp_pos + (-1, -1)));
        assert!(!should_prune(&state, &EA::Positional(PosAT::Move, home_pos + (1, 0))));
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
        let state = GameStateBuilder::new()
            .add_home_players(&[(5, 5), (10, 7)])
            .add_away_player(Position::new((20, 10)))
            .add_ball_pos(Position::new((2, 2)))
            .build();
        let a = EA::Positional(PosAT::StartHandoff, Position::new((5, 5)));
        assert!(state.is_legal_action(&a));
        assert!(!should_prune(&state, &a));
    }

    // --- P8: collapse "move again" after the first move action ---------

    /// Home player at (5,5), activated as `StartMove`. Helper drives the
    /// engine into the active-player state and lets the caller dial in
    /// `moves` and the pickup flag.
    fn startmove_state(moves: u8, picked_up: bool) -> (GameState, Position) {
        let pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(pos)
            .add_away_player(Position::new((20, 10)))
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartMove, pos));
        state.get_active_player_mut().unwrap().moves = moves;
        state.info.pickup_this_activation = picked_up;
        (state, pos)
    }

    #[test]
    fn move_after_first_move_is_pruned() {
        let (state, _) = startmove_state(1, false);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(
            should_prune(&state, &a),
            "a settled mover's extra move should be pruned"
        );
    }

    #[test]
    fn move_before_first_move_not_pruned() {
        let (state, _) = startmove_state(0, false);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(!should_prune(&state, &a), "the first move must be allowed");
    }

    #[test]
    fn end_player_turn_survives_p8_after_moving() {
        // With moves > 0, P1 no longer prunes EndPlayerTurn and P8 must not
        // either (it only touches positional continuations) — leaving it as
        // the sole survivor for the quiescent loop to auto-apply.
        let (state, _) = startmove_state(1, false);
        let a = EA::Simple(SimpleAT::EndPlayerTurn);
        assert!(!should_prune(&state, &a));
    }

    #[test]
    fn pickup_bonus_allows_one_more_move() {
        // Just picked up the ball this activation → owed one follow-up move.
        let (state, _) = startmove_state(2, true);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(
            !should_prune(&state, &a),
            "a player who just picked up should get one more move action"
        );
    }

    #[test]
    fn pickup_bonus_only_lasts_one_move() {
        // Once the bonus is consumed (engine clears the flag on the next move
        // selection), a further move is pruned again — so picked-up == one
        // extra move, not unlimited.
        let (state, _) = startmove_state(4, false);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert!(should_prune(&state, &a));
    }

    #[test]
    fn p8_does_not_touch_non_startmove() {
        // P8 is scoped to StartMove; other activation types (blitz, pass, etc.)
        // are governed by P2–P5 instead. Verify the rule itself stays out.
        let home_pos = Position::new((5, 5));
        let opp_pos = Position::new((7, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(opp_pos)
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        assert!(!prune_redundant_move_after_first(&state));
    }

    #[test]
    fn p8_is_deterministic() {
        let (state, _) = startmove_state(1, false);
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
    }

    /// Determinism check for P5 — pure on state.
    #[test]
    fn p5_blitz_pruning_is_deterministic() {
        let home_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(Position::new((20, 10)))
            .build();
        check_prune_and_step(&mut state, EA::Positional(PosAT::StartBlitz, home_pos));
        let a = EA::Positional(PosAT::Move, Position::new((6, 5)));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
        assert_eq!(should_prune(&state, &a), should_prune(&state, &a));
    }
}

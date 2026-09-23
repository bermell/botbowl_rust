//! Scripted block-die selection for the MCTS bot.
//!
//! When the engine offers block-die choices via simple actions
//! (`SelectPow` / `SelectPowPush` / `SelectPush` / `SelectBothDown` /
//! `SelectSkull`), MCTS would otherwise treat each variant as its own
//! tree branch — a waste of search budget for a decision the rules of
//! Blood Bowl resolve deterministically given attacker/defender skills.
//!
//! `scripted_pick` returns the best die for whichever side is currently
//! picking. It is wired in at exactly one place: the first arm of
//! `scripted::scripted_player_pick`, which `apply_action`'s quiescent loop
//! consults before every advance. So the search does not *see* a collapsed
//! fan-out — it never creates the node at all, because the loop plays the
//! die and walks on.
//!
//! **It is deliberately not called from `available_actions`.** That matters
//! more than it looks. The engine stops and asks a real player which die to
//! use, so `MctsBot::get_action` does get called on a mid-block-die root —
//! and there `available_actions` offers the full fan, which the search then
//! explores. Two consequences, both measured in
//! `tests/block_reuse.rs` (and queued as plans/032 item 13):
//!
//! * the previous search cannot have materialised that root, so a block-die
//!   decision reuses the cached tree **0.8%** of the time against 65-100%
//!   for every other procedure, and rebuilds instead;
//! * the bot picks a different die from this script **51%** of the time, so
//!   the tree is valuing every future block under a policy the bot does not
//!   actually follow.
//!
//! The picker is identified by `state.available_actions.team`: when
//! that team matches the active player's team, the attacker is picking
//! (advantage block, `NumBlockDices::{One,Two,Three}`); otherwise the
//! defender is picking the result of an uphill block
//! (`NumBlockDices::{TwoUphill,ThreeUphill}`).
//!
//! The attacker-picking logic mirrors `scripted_bot::pick_block_die`
//! (engine/src/scripted_bot.rs:303-360). We don't import it because
//! that helper has no uphill handling — the engine's scripted bot
//! always controls the attacking team and never has to pick as the
//! defender.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, FieldedPlayer, PlayerStatus};
use botbowl_engine::core::table::{SimpleAT, Skill};

/// Returns the scripted-best block-die action if the state currently
/// offers a block-die choice, otherwise `None`. Pure function of state.
pub fn scripted_pick(state: &GameState) -> Option<EngineAction> {
    let simple = state.available_actions.get_simple();
    let is_block_choice = simple.contains(&SimpleAT::SelectPow)
        || simple.contains(&SimpleAT::SelectPowPush)
        || simple.contains(&SimpleAT::SelectPush)
        || simple.contains(&SimpleAT::SelectBothDown)
        || simple.contains(&SimpleAT::SelectSkull);
    if !is_block_choice {
        return None;
    }

    let (attacker, defender) = active_block_attacker_defender(state);

    // Picker = whichever team is being asked. `available_actions.team`
    // is set by the Block procedure (block_procs.rs:304-309) to the
    // attacker on a normal (downhill) block and to the defender on an
    // uphill block. Attacker not being set should not happen in
    // practice, but if it does we fall back to attacker-picks logic.
    let attacker_team = attacker.map(|a| a.stats.team);
    let picking_team = state.available_actions.team;
    let defender_is_picking = picking_team.is_some() && attacker_team.is_some() && picking_team != attacker_team;

    let pick_at = if defender_is_picking {
        pick_for_defender(simple, attacker, defender)
    } else {
        pick_for_attacker(simple, attacker, defender)
    }?;
    Some(EngineAction::Simple(pick_at))
}

/// Attacker preference: Pow > PowPush (unless defender Dodge) > BothDown
/// (only if attacker has Block and defender doesn't) > Push > BothDown
/// (any) > PowPush (any) > Skull. Mirrors `scripted_bot::pick_block_die`
/// — with one deliberate hole: when the attacker has Block, the
/// defender doesn't, and the roll offers *both* a knockdown-and-push die
/// and `BothDown`, this returns `None`. Knocking the defender down in
/// place versus down-and-pushed is a genuine positional choice, and
/// `roll_outcomes::block_outcomes` emits exactly that roll as its
/// `[Pow, BothDown, ..]` child so the search can decide (plan 036).
fn pick_for_attacker(
    simple: &std::collections::HashSet<SimpleAT>,
    attacker: Option<&FieldedPlayer>,
    defender: Option<&FieldedPlayer>,
) -> Option<SimpleAT> {
    let defender_dodges = defender.map(|d| d.has_skill(Skill::Dodge)).unwrap_or(false);
    let attacker_has_block = attacker.map(|a| a.has_skill(Skill::Block)).unwrap_or(false);
    let defender_has_block = defender.map(|d| d.has_skill(Skill::Block)).unwrap_or(false);
    let both_down_fells_defender_only = attacker_has_block && !defender_has_block;

    let knockdown_and_push = if simple.contains(&SimpleAT::SelectPow) {
        Some(SimpleAT::SelectPow)
    } else if simple.contains(&SimpleAT::SelectPowPush) && !defender_dodges {
        Some(SimpleAT::SelectPowPush)
    } else {
        None
    };
    if let Some(pick) = knockdown_and_push {
        if both_down_fells_defender_only && simple.contains(&SimpleAT::SelectBothDown) {
            return None; // down-in-place vs down-and-pushed: let the search choose
        }
        return Some(pick);
    }
    if simple.contains(&SimpleAT::SelectBothDown) && both_down_fells_defender_only {
        return Some(SimpleAT::SelectBothDown);
    }
    if simple.contains(&SimpleAT::SelectPush) {
        return Some(SimpleAT::SelectPush);
    }
    if simple.contains(&SimpleAT::SelectPowPush) {
        return Some(SimpleAT::SelectPowPush);
    }
    if simple.contains(&SimpleAT::SelectBothDown) {
        return Some(SimpleAT::SelectBothDown);
    }
    if simple.contains(&SimpleAT::SelectSkull) {
        return Some(SimpleAT::SelectSkull);
    }
    None
}

/// Defender preference (uphill block): Skull > Push > BothDown (only if
/// defender has Block and attacker doesn't) > PowPush > BothDown (any) >
/// Pow. The defender picks the outcome that's least bad for them.
fn pick_for_defender(
    simple: &std::collections::HashSet<SimpleAT>,
    attacker: Option<&FieldedPlayer>,
    defender: Option<&FieldedPlayer>,
) -> Option<SimpleAT> {
    if simple.contains(&SimpleAT::SelectSkull) {
        return Some(SimpleAT::SelectSkull);
    }
    if simple.contains(&SimpleAT::SelectPush) {
        return Some(SimpleAT::SelectPush);
    }
    if simple.contains(&SimpleAT::SelectBothDown) {
        let attacker_has_block = attacker.map(|a| a.has_skill(Skill::Block)).unwrap_or(false);
        let defender_has_block = defender.map(|d| d.has_skill(Skill::Block)).unwrap_or(false);
        if defender_has_block && !attacker_has_block {
            return Some(SimpleAT::SelectBothDown);
        }
    }
    if simple.contains(&SimpleAT::SelectPowPush) {
        return Some(SimpleAT::SelectPowPush);
    }
    if simple.contains(&SimpleAT::SelectBothDown) {
        return Some(SimpleAT::SelectBothDown);
    }
    if simple.contains(&SimpleAT::SelectPow) {
        return Some(SimpleAT::SelectPow);
    }
    None
}

fn active_block_attacker_defender(state: &GameState) -> (Option<&FieldedPlayer>, Option<&FieldedPlayer>) {
    let attacker = state.get_active_player();
    let defender = attacker.and_then(|att| {
        state
            .get_adj_players(att.position)
            .find(|p| p.stats.team != att.stats.team && p.status == PlayerStatus::Up)
    });
    (attacker, defender)
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{Position, TeamType};

    fn block_state(home_pos: Position, away_pos: Position, picking: TeamType, active_at: Position) -> GameState {
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        state.available_actions.team = Some(picking);
        let id = state.get_player_id_at(active_at).unwrap();
        state.set_active_player(id);
        state
    }

    fn offer(state: &mut GameState, ats: &[SimpleAT]) {
        for at in ats {
            state.available_actions.insert_simple(*at);
        }
    }

    #[test]
    fn returns_none_when_no_block_choice_present() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        // Some unrelated simple action — should not be confused for a block pick.
        offer(&mut state, &[SimpleAT::EndPlayerTurn]);
        assert!(scripted_pick(&state).is_none());
    }

    #[test]
    fn attacker_picks_pow_when_offered() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        offer(
            &mut state,
            &[SimpleAT::SelectPow, SimpleAT::SelectPush, SimpleAT::SelectSkull],
        );
        assert_eq!(scripted_pick(&state), Some(EngineAction::Simple(SimpleAT::SelectPow)));
    }

    #[test]
    fn attacker_falls_through_pow_push_to_push_when_defender_has_dodge() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        // Give the defender Dodge so PowPush becomes risky.
        let def_id = state.get_player_id_at(Position::new((6, 5))).unwrap();
        state.get_mut_player(def_id).unwrap().stats.give_skill(Skill::Dodge);
        offer(&mut state, &[SimpleAT::SelectPowPush, SimpleAT::SelectPush]);
        assert_eq!(scripted_pick(&state), Some(EngineAction::Simple(SimpleAT::SelectPush)));
    }

    #[test]
    fn attacker_picks_both_down_when_attacker_has_block_and_defender_does_not() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        let att_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.get_mut_player(att_id).unwrap().stats.give_skill(Skill::Block);
        offer(&mut state, &[SimpleAT::SelectBothDown, SimpleAT::SelectPush]);
        assert_eq!(
            scripted_pick(&state),
            Some(EngineAction::Simple(SimpleAT::SelectBothDown))
        );
    }

    #[test]
    fn attacker_with_block_gets_a_real_choice_between_pow_and_both_down() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        let att_id = state.get_player_id_at(Position::new((5, 5))).unwrap();
        state.get_mut_player(att_id).unwrap().stats.give_skill(Skill::Block);
        offer(&mut state, &[SimpleAT::SelectBothDown, SimpleAT::SelectPow]);
        assert_eq!(
            scripted_pick(&state),
            None,
            "down-in-place vs down-and-pushed is the search's call"
        );

        // Same with PowPush standing in for Pow (no Dodge on the defender)...
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        state.get_mut_player(att_id).unwrap().stats.give_skill(Skill::Block);
        offer(&mut state, &[SimpleAT::SelectBothDown, SimpleAT::SelectPowPush]);
        assert_eq!(scripted_pick(&state), None);

        // ...but not when the defender has Block too: BothDown does nothing, Pow wins.
        let def_id = state.get_player_id_at(Position::new((6, 5))).unwrap();
        state.get_mut_player(def_id).unwrap().stats.give_skill(Skill::Block);
        assert_eq!(
            scripted_pick(&state),
            Some(EngineAction::Simple(SimpleAT::SelectPowPush))
        );
    }

    #[test]
    fn attacker_avoids_both_down_when_attacker_lacks_block() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        offer(&mut state, &[SimpleAT::SelectBothDown, SimpleAT::SelectPush]);
        assert_eq!(scripted_pick(&state), Some(EngineAction::Simple(SimpleAT::SelectPush)));
    }

    #[test]
    fn defender_picks_skull_when_offered() {
        // Picking team != attacker team → uphill block, defender picks.
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Away,
            Position::new((5, 5)),
        );
        offer(
            &mut state,
            &[SimpleAT::SelectSkull, SimpleAT::SelectPow, SimpleAT::SelectPowPush],
        );
        assert_eq!(scripted_pick(&state), Some(EngineAction::Simple(SimpleAT::SelectSkull)));
    }

    #[test]
    fn defender_picks_push_over_pow_when_skull_unavailable() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Away,
            Position::new((5, 5)),
        );
        offer(&mut state, &[SimpleAT::SelectPow, SimpleAT::SelectPush]);
        assert_eq!(scripted_pick(&state), Some(EngineAction::Simple(SimpleAT::SelectPush)));
    }

    #[test]
    fn defender_picks_both_down_only_when_defender_has_block_advantage() {
        // Defender has Block, attacker doesn't → BothDown safe for defender.
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Away,
            Position::new((5, 5)),
        );
        let def_id = state.get_player_id_at(Position::new((6, 5))).unwrap();
        state.get_mut_player(def_id).unwrap().stats.give_skill(Skill::Block);
        offer(&mut state, &[SimpleAT::SelectBothDown, SimpleAT::SelectPow]);
        assert_eq!(
            scripted_pick(&state),
            Some(EngineAction::Simple(SimpleAT::SelectBothDown))
        );
    }

    #[test]
    fn determinism_same_state_same_pick() {
        let mut state = block_state(
            Position::new((5, 5)),
            Position::new((6, 5)),
            TeamType::Home,
            Position::new((5, 5)),
        );
        offer(
            &mut state,
            &[SimpleAT::SelectPow, SimpleAT::SelectPush, SimpleAT::SelectBothDown],
        );
        let p1 = scripted_pick(&state);
        let p2 = scripted_pick(&state);
        let p3 = scripted_pick(&state);
        assert_eq!(p1, p2);
        assert_eq!(p2, p3);
    }
}

//! Plan 049 finding 7: can the value head tell a failed roll's reroll prompt from the state a
//! success would have left?
//!
//! The encoder has no input for the decision being asked (no procedure or action-type feature),
//! so a failed dodge that stops for "use a reroll?" may encode exactly like the next move decision
//! after a successful one: same square, same player, same rerolls. Those two states have very
//! different values (the audit measured a mean mover outcome of −0.36 at dodge prompts), so if the
//! tensors are identical the value head can only average them.
//!
//! Measured 2026-09-29: the two differ in exactly one plane, `path_prob` (the success state shows
//! the mover's remaining paths, the prompt shows none); every other plane and every global feature
//! is identical. So the head *can* tell them apart, but only through an absent path plane, and
//! nothing encodes which roll is being rerolled. This pins the one thing that separates them.

use botbowl_engine::core::dices::RollResult;
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, Position, SomeProcInput};
use botbowl_engine::core::table::PosAT;
use botbowl_nn::encode::{encode_raw, global_feature_names, spatial_channel_names};

/// A home player next to an away player, dodging away; paused on the dodge roll.
fn paused_on_dodge() -> GameState {
    let dodger = Position::new((5, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(dodger)
        .add_away_player(Position::new((6, 5)))
        .add_home_player(Position::new((2, 2)))
        .build();
    state.home.rerolls = 2;
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, dodger)));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(
        PosAT::Move,
        Position::new((4, 4)),
    )));
    assert!(state.pending_roll.is_some(), "expected the dodge roll");
    state
}

#[test]
fn a_failed_dodge_prompt_against_the_state_a_success_leaves() {
    let base = paused_on_dodge();

    let mut success = base.clone();
    success.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Pass));
    let mut failed = base.clone();
    failed.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));

    assert!(success.available_actions.team.is_some(), "success: a decision");
    assert!(failed.available_actions.team.is_some(), "failure: the reroll prompt");
    eprintln!("success top proc: {:?}", success.proc_stack_top());
    eprintln!("failed  top proc: {:?}", failed.proc_stack_top());
    eprintln!("failed actions: {:?}", failed.get_all_actions());

    let (a, b) = (encode_raw(&success), encode_raw(&failed));
    let spatial_diff: Vec<usize> = a
        .spatial
        .iter()
        .zip(&b.spatial)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i / (a.h * a.w))
        .collect();
    let mut planes = spatial_diff.clone();
    planes.dedup();
    let channel = spatial_channel_names();
    let planes: Vec<String> = planes.iter().map(|&c| channel[c].clone()).collect();
    let plane = a.h * a.w;
    let ones = |e: &botbowl_nn::encode::EncodedRaw, name: &str| {
        let c = channel.iter().position(|n| n == name).unwrap();
        e.spatial[c * plane..(c + 1) * plane]
            .iter()
            .filter(|v| **v != 0)
            .count()
    };
    for name in &planes {
        eprintln!(
            "  {name}: nonzero cells success {} / failed {}",
            ones(&a, name),
            ones(&b, name)
        );
    }
    let names = global_feature_names();
    let global_diff: Vec<String> = a
        .global
        .iter()
        .zip(&b.global)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, (x, y))| format!("{}: {x} vs {y}", names.get(i).cloned().unwrap_or_else(|| i.to_string())))
        .collect();
    eprintln!(
        "REROLL_PROMPT_ENCODING spatial cells differing: {} in planes {:?}; global differing: {:?}",
        spatial_diff.len(),
        planes,
        global_diff
    );
    assert!(
        !spatial_diff.is_empty() || !global_diff.is_empty(),
        "a failed dodge's reroll prompt encodes exactly like the state a success leaves — the value \
         head cannot tell them apart"
    );
}

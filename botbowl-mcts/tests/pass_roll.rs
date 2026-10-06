//! The pass roll inside the search.
//!
//! The pass is a raw D6 that the `Pass` proc reads face by face (accurate on the passer's target,
//! fumble on a 1, wildly inaccurate when the modifier takes it to 1, inaccurate otherwise). The
//! search used to script it to a 1, so every pass fumbled in-tree. Enumerating all six faces is
//! not enough on its own: faces the proc treats alike reach the same state, and recon_mcts cannot
//! hold two chance edges from one parent into one recombined child (it panicked with "could not
//! remove dropped node as child's parents"). So faces that coincide are merged into one child
//! carrying their summed probability.

use botbowl_engine::core::dices::{RequestedRoll, RollResult};
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, Position, SomeProcInput};
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::{BbAction, BbPlayer, BloodBowlDynamics};
use recon_mcts::GameDynamics;

/// A home thrower with the ball and a free receiver three squares away, paused on the pass D6.
fn paused_on_pass_roll() -> GameState {
    let thrower = Position::new((5, 5));
    let receiver = Position::new((8, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(thrower)
        .add_home_player(receiver)
        .add_away_player(Position::new((15, 2)))
        .add_ball_pos(thrower)
        .build();
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartPass, thrower)));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Pass, receiver)));
    assert_eq!(state.pending_roll, Some(RequestedRoll::D6), "expected the pass roll");
    state
}

#[test]
fn the_pass_roll_offers_every_distinct_outcome_once() {
    let state = paused_on_pass_roll();
    let gd = BloodBowlDynamics::default();
    let outcomes: Vec<BbAction> = gd
        .available_actions(&BbPlayer::Chance, &state)
        .unwrap()
        .into_iter()
        .collect();

    let total: f32 = outcomes.iter().map(|a| a.prob_f32().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}");
    assert!(
        (2..=4).contains(&outcomes.len()),
        "accurate / inaccurate / wildly inaccurate / fumble at most, got {outcomes:?}"
    );

    // Every child reaches a different state: no two edges into one recombined node.
    let next: Vec<GameState> = outcomes
        .iter()
        .map(|a| gd.apply_action(state.clone(), a).expect("a legal roll outcome"))
        .collect();
    for i in 0..next.len() {
        for j in i + 1..next.len() {
            assert!(
                next[i] != next[j],
                "outcomes {:?} and {:?} coincide",
                outcomes[i],
                outcomes[j]
            );
        }
    }

    // The fumble is its own 1-in-6 child, and it is no longer the only one.
    let fumble = outcomes
        .iter()
        .find(|a| matches!(a, BbAction::Chance { result: RollResult::D6(d), .. } if *d as u8 == 1))
        .expect("a fumble child");
    assert!((fumble.prob_f32().unwrap() - 1.0 / 6.0).abs() < 1e-5);
}

#[test]
fn a_search_through_passes_completes() {
    use botbowl_engine::bots::Bot;
    use botbowl_mcts::{MctsBot, SearchBudget};
    // Before the merge this panicked (or deadlocked) on dropping the tree.
    let mut thrower_side = GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_home_player(Position::new((8, 5)))
        .add_home_player(Position::new((8, 3)))
        .add_away_player(Position::new((15, 2)))
        .add_ball_pos(Position::new((5, 5)))
        .build();
    thrower_side.set_dice_mode(DiceMode::RollDice);
    let mut bot = MctsBot::new(SearchBudget::Iterations(400)).with_workers(1);
    for _ in 0..3 {
        if thrower_side.info.game_over || thrower_side.available_actions.team.is_none() {
            break;
        }
        let a = bot.get_action(&thrower_side);
        thrower_side.step(a).unwrap();
    }
}

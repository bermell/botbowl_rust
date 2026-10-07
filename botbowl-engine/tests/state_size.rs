//! `GameState` is cloned, hashed and compared for every node the MCTS search creates, and a
//! search stores one per node, so its size is both the search's memory footprint and a good
//! part of its CPU. These tests pin the size so it cannot quietly regress: a board square is one
//! byte, a square's positional offerings are two, and the procedure stack and action set are
//! inline. Measured on the default 28x17/11 capacity when this was written: `GameState` 4.4 KB
//! (was 10.0 KB plus an 11.5 KB boxed `AvailableActions`), `AvailableActions` 1.0 KB.

use std::mem::size_of;

use botbowl_engine::core::dices::resolve_with_rng;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder, PROC_STACK_INLINE};
use botbowl_engine::core::model::{AvailableActions, BoardCell, BoardDims, SomeProcInput, HEIGHT, WIDTH};
use botbowl_engine::core::table::{PosATSet, SimpleATSet};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const CELLS: usize = WIDTH * HEIGHT;

#[test]
fn board_square_is_one_byte_and_offerings_are_two() {
    assert_eq!(size_of::<BoardCell>(), 1);
    assert_eq!(size_of::<PosATSet>(), 2);
    assert_eq!(size_of::<SimpleATSet>(), 4);
}

#[test]
fn available_actions_is_a_small_copy_value() {
    // One `PosATSet` a square, plus the team, the simple set, the `Option` tag and padding.
    assert!(
        size_of::<AvailableActions>() <= 2 * CELLS + 64,
        "AvailableActions is {} bytes for {CELLS} squares",
        size_of::<AvailableActions>()
    );
}

#[test]
fn game_state_stays_small() {
    // The board (one byte a square), the inline procedure stack, two roster arrays at ~40 bytes a
    // slot, the rng, and some hundreds of bytes of scalars. The bound is generous on purpose: it
    // catches a `Vec` or a wide `Option` creeping into a per-square or per-frame field, not a
    // new scalar.
    let proc_stack = PROC_STACK_INLINE * 128;
    let bound = CELLS + proc_stack + 4096;
    assert!(
        size_of::<GameState>() <= bound,
        "GameState is {} bytes, bound {bound} ({CELLS} squares)",
        size_of::<GameState>()
    );
}

/// Plays a whole random game and returns the deepest procedure stack it saw.
fn deepest_proc_stack(dims: BoardDims, seed: u64) -> usize {
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut deepest = 0;
    let mut steps = 0;
    while !state.info.game_over {
        steps += 1;
        assert!(steps < 400_000, "dims {dims:?} seed {seed}: game did not finish");
        deepest = deepest.max(state.proc_stack_iter().count());
        match state.pending_roll {
            Some(requested) => {
                let result = resolve_with_rng(requested, &mut rng);
                state.step_with_roll_or_action(SomeProcInput::Roll(result));
            }
            None => {
                let actions = state.get_all_actions();
                let action = actions[rng.gen_range(0..actions.len())];
                state.step_with_roll_or_action(SomeProcInput::Action(action));
            }
        }
    }
    deepest
}

/// The inline capacity is headroom over what a game uses, so a clone never allocates for the
/// stack. If this fails, raise `PROC_STACK_INLINE` rather than weakening the test.
#[test]
fn a_full_game_never_spills_the_inline_proc_stack() {
    let dims = BoardDims::from_env();
    let deepest = (0..3).map(|seed| deepest_proc_stack(dims, seed)).max().unwrap();
    assert!(
        deepest <= PROC_STACK_INLINE,
        "a random game reached a procedure stack {deepest} deep; PROC_STACK_INLINE is {PROC_STACK_INLINE}"
    );
    eprintln!("deepest proc stack over 3 random games on {dims:?}: {deepest} (inline capacity {PROC_STACK_INLINE})");
}

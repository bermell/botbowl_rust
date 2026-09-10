//! Plan 034 phase 0, decision 9: `DiceMode::RegisterRolls` is the contract the
//! web server drives the engine on, so a *whole* game must be playable through
//! `step_with_roll_or_action` with the caller rolling every die.
//!
//! Before this test, `RegisterRolls` was only exercised over MCTS horizons
//! (one turn, no half/kickoff/game-over transitions), so the coin toss, the
//! kickoff table, KO/casualty recovery between halves and the game-over
//! handshake were all unvisited under externally-supplied rolls. The plan says
//! any assert this trips is an engine bug to fix in the engine, not something
//! to route around in the server.
//!
//! It doubles as the capacity-padding guard: a 14x7 playable board is played
//! under whatever capacity the binary was compiled at, so a `WIDTH`/`HEIGHT`
//! constant leaking into gameplay code shows up here as an out-of-bounds edge
//! in the wrong place.

use botbowl_engine::core::dices::resolve_with_rng;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, BoardDims, MicroStepState, SomeProcInput, HEIGHT, TEAM_SIZE, WIDTH};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A micro-step budget generous enough for a full random game (measured
/// ~4k-25k micro-steps) but small enough to fail fast on a livelock.
const STEP_BUDGET: usize = 400_000;

struct GameStats {
    steps: usize,
    actions: usize,
    rolls: usize,
}

/// Play one full game with the *caller* supplying every roll, asserting the
/// `RegisterRolls` invariants at each pause.
fn play_full_game(dims: BoardDims, seed: u64) -> GameStats {
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut stats = GameStats {
        steps: 0,
        actions: 0,
        rolls: 0,
    };

    while !state.info.game_over {
        assert!(
            stats.steps < STEP_BUDGET,
            "dims {dims:?} seed {seed}: game did not finish within {STEP_BUDGET} micro-steps \
             (proc={:?}, half={}, turns={}/{})",
            state.proc_stack_top(),
            state.info.half,
            state.info.home_turn,
            state.info.away_turn,
        );
        stats.steps += 1;

        let outcome = match state.pending_roll {
            Some(requested) => {
                // A pause on a roll must not also offer actions — the caller
                // would have no way to know which input the engine wants.
                assert!(
                    state.get_available_actions().is_empty(),
                    "dims {dims:?} seed {seed}: pending roll {requested:?} *and* available actions at {:?}",
                    state.proc_stack_top(),
                );
                let result = resolve_with_rng(requested, &mut rng);
                assert!(
                    requested.is_compatible(result),
                    "resolve_with_rng produced {result:?}, incompatible with {requested:?}"
                );
                stats.rolls += 1;
                state.step_with_roll_or_action(SomeProcInput::Roll(result))
            }
            None => {
                let actions = state.get_all_actions();
                assert!(
                    !actions.is_empty(),
                    "dims {dims:?} seed {seed}: engine wants an action but offers none at {:?}",
                    state.proc_stack_top(),
                );
                let action = actions[rng.gen_range(0..actions.len())];
                assert_positions_are_on_the_runtime_board(&state, action, dims, seed);
                stats.actions += 1;
                state.step_with_roll_or_action(SomeProcInput::Action(action))
            }
        };

        // `RunAgain` is consumed inside `step_with_roll_or_action`; the three
        // states below are the complete external contract.
        match outcome {
            MicroStepState::NeedAction => assert!(
                state.pending_roll.is_none(),
                "NeedAction with a pending roll still set at {:?}",
                state.proc_stack_top()
            ),
            MicroStepState::NeedRoll => assert!(
                state.pending_roll.is_some(),
                "NeedRoll without a pending_roll at {:?}",
                state.proc_stack_top()
            ),
            MicroStepState::GameOver => assert!(state.info.game_over, "GameOver without info.game_over"),
            MicroStepState::RunAgain => {
                panic!("step_with_roll_or_action leaked RunAgain — it is supposed to loop internally")
            }
        }
    }

    assert!(state.info.half >= 2, "game over before the second half");
    assert!(
        state.info.winner.is_some() || state.home.score == state.away.score,
        "no winner recorded on a decided game: {} - {}",
        state.home.score,
        state.away.score
    );
    stats
}

/// Every position the engine offers must be inside the *runtime* board, not
/// merely inside the compiled capacity. This is what catches a `WIDTH`/`HEIGHT`
/// constant leaking into gameplay code on a smaller board.
fn assert_positions_are_on_the_runtime_board(state: &GameState, action: Action, dims: BoardDims, seed: u64) {
    if let Action::Positional(at, pos) = action {
        assert!(
            !state.is_out(pos),
            "dims {dims:?} seed {seed}: offered {at:?} at out-of-bounds {pos} \
             (runtime board is x in 1..={}, y in 1..={})",
            dims.width - 2,
            dims.height - 2,
        );
    }
}

/// The plan's first target board: 14x7 playable (16x9 engine), 4 players a
/// side. Skipped when the binary was compiled at a smaller capacity.
const SMALL: (i8, i8, usize) = (16, 9, 4);

fn capacity_fits(w: i8, h: i8, team_size: usize) -> bool {
    WIDTH as i8 >= w && HEIGHT as i8 >= h && TEAM_SIZE >= team_size
}

#[test]
fn full_game_under_register_rolls_on_the_small_board() {
    let (w, h, team_size) = SMALL;
    if !capacity_fits(w, h, team_size) {
        eprintln!("skipped: compiled capacity {WIDTH}x{HEIGHT}/{TEAM_SIZE} is smaller than {w}x{h}/{team_size}");
        return;
    }
    let dims = BoardDims::new(w, h, team_size);
    for seed in 0..40 {
        let stats = play_full_game(dims, seed);
        assert!(stats.rolls > 0 && stats.actions > 0);
    }
}

#[test]
fn full_game_under_register_rolls_at_compiled_capacity() {
    // The other end of the range: whatever board this binary was built for,
    // which for the default build is the full 26x15/11 pitch with the kickoff
    // table enabled (`team_size >= 7`).
    let dims = BoardDims::default();
    for seed in 0..15 {
        let stats = play_full_game(dims, seed);
        assert!(stats.rolls > 0 && stats.actions > 0);
    }
}

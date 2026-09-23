//! `view::derive` must not panic, and must not lie, on **any** state the
//! engine can reach — not just the tidy ones a unit test constructs.
//!
//! This exists because it should have caught a real bug and didn't: `Kickoff`
//! sets `BallState::InAir(aim + direction * len)` with `len` capped only at
//! `max_scatter()`, which on a narrow board puts the ball at a *negative*
//! coordinate for a step or two. `Position` is `i8`, so the unchecked
//! `pos.y as usize` in the view's index wrapped to ~2^64 and the multiply
//! overflowed — panicking the session thread mid-kickoff, which (before the
//! websocket learned to notice) left the browser clicking into a dead socket.
//!
//! The websocket integration test plays whole games too, but it only sees the
//! states where the *human* is asked to move. This one derives a view at every
//! single pause, including the mid-kickoff ones.

use botbowl_engine::core::dices::resolve_with_rng;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, BallState, BoardDims, Position, SomeProcInput, HEIGHT, TEAM_SIZE, WIDTH};
use botbowl_web_server::view::{derive, DeriveCtx};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn check(state: &GameState) {
    let view = derive(state, &DeriveCtx::default());
    let dims = state.board_dims;

    assert_eq!(
        view.squares.len(),
        dims.width as usize * dims.height as usize,
        "the grid must cover the whole runtime board"
    );
    for (i, sq) in view.squares.iter().enumerate() {
        assert_eq!(
            view.dims.index(sq.pos),
            i,
            "square {:?} is not at its own index",
            sq.pos
        );
    }

    // Everything the view offers must be legal, and everything legal must be
    // offered — a UI that shows an illegal move gets an error on click, and
    // one that hides a legal move can deadlock the game.
    let mut offered: Vec<Action> = view
        .squares
        .iter()
        .flat_map(|sq| {
            sq.actions.iter().map(move |at| {
                Action::Positional(
                    botbowl_web_server::mirror::pos_at_from_proto(*at),
                    Position::new((sq.pos.x, sq.pos.y)),
                )
            })
        })
        .collect();
    offered.extend(
        view.simple_actions
            .iter()
            .map(|a| Action::Simple(botbowl_web_server::mirror::simple_at_from_proto(a.at))),
    );
    offered.sort();

    if state.info.game_over {
        assert!(offered.is_empty(), "nothing is offered after the whistle");
        assert!(view.to_act.is_none());
        return;
    }

    let mut legal = state.get_all_actions();
    legal.sort();
    assert_eq!(
        offered,
        legal,
        "the view and the engine disagree at {:?}",
        state.proc_stack_top()
    );

    if state.pending_roll.is_none() {
        assert!(
            !offered.is_empty(),
            "the engine wants an action at {:?} but the view offers none",
            state.proc_stack_top()
        );
    }

    // A probability is a probability.
    for sq in &view.squares {
        if let Some(p) = sq.move_prob {
            assert!((0.0..=1.0).contains(&p), "{:?} has move_prob {p}", sq.pos);
        }
    }
}

fn play(dims: BoardDims, seed: u64) -> usize {
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut steps = 0usize;

    while !state.info.game_over && steps < 400_000 {
        steps += 1;
        check(&state);
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
    check(&state);
    steps
}

fn capacity_fits(w: i8, h: i8, team_size: usize) -> bool {
    WIDTH as i8 >= w && HEIGHT as i8 >= h && TEAM_SIZE >= team_size
}

#[test]
fn the_view_survives_every_state_of_a_small_board_game() {
    if !capacity_fits(16, 9, 4) {
        eprintln!("skipped: capacity too small");
        return;
    }
    let dims = BoardDims::new(16, 9, 4);
    for seed in 0..6 {
        let steps = play(dims, seed);
        assert!(steps > 100);
    }
}

/// The narrow case that used to reliably turn up in the random-play fuzz
/// above, stated directly instead: force a kickoff-deviate roll large enough
/// to put the ball off the grid on a narrow board, independent of how likely
/// that roll is to come up in real play. `BoardDims::scatter_divisor` (added
/// after this file was written) intentionally scales the deviate roll down
/// on narrow boards, which made an off-grid kick rare enough that
/// `the_view_survives_every_state_of_a_small_board_game` could no longer be
/// relied on to hit it within a reasonable sample — this test doesn't depend
/// on that probability at all.
#[test]
fn a_kickoff_deviate_off_the_grid_does_not_panic_the_view() {
    if !capacity_fits(16, 9, 4) {
        return;
    }
    let dims = BoardDims::new(16, 9, 4);
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::Kickoff { turn: 1 })
        .build();
    state.set_logging_state(false);
    // The largest deviate roll this board's scatter_divisor still allows,
    // aimed straight up: enough to carry the ball past y=0 from a
    // receiving-half-centred aim on this narrow a board.
    state.fix_d6(6);
    state.fix_d8_direction(botbowl_engine::core::model::Direction::up());
    state.step_simple(botbowl_engine::core::table::SimpleAT::KickoffAimMiddle);
    check(&state); // must not panic
}

#[test]
fn the_view_survives_every_state_at_the_compiled_capacity() {
    let dims = BoardDims::default();
    for seed in 100..102 {
        let steps = play(dims, seed);
        assert!(steps > 100);
    }
}

/// The narrow case, stated directly rather than waiting for a random game to
/// find it.
#[test]
fn a_ball_in_flight_past_the_touchline_is_simply_not_drawn() {
    if !capacity_fits(16, 9, 4) {
        return;
    }
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(16, 9, 4))
        .add_home_players(&[(9, 3)])
        .build();
    state.set_logging_state(false);
    for pos in [(-4, -3), (-1, 4), (20, 4), (4, 40)] {
        state.set_ball(BallState::InAir(Position::new(pos)));
        let view = derive(&state, &DeriveCtx::default());
        assert!(
            view.squares.iter().all(|s| s.ball.is_none()),
            "a ball at {pos:?} is off the grid and must not be drawn"
        );
    }
}

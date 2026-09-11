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

fn play(dims: BoardDims, seed: u64) -> (usize, bool) {
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut steps = 0usize;
    // Did this game ever put the ball off the grid? That is the case the
    // panic came from, so the test is only meaningful if it happens.
    let mut saw_ball_off_grid = false;

    while !state.info.game_over && steps < 400_000 {
        steps += 1;
        check(&state);
        if let BallState::InAir(pos) | BallState::OnGround(pos) = state.ball {
            if pos.x < 0 || pos.y < 0 || pos.x >= dims.width || pos.y >= dims.height {
                saw_ball_off_grid = true;
            }
        }
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
    (steps, saw_ball_off_grid)
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
    let mut ever_off_grid = false;
    for seed in 0..6 {
        let (steps, off_grid) = play(dims, seed);
        ever_off_grid |= off_grid;
        assert!(steps > 100);
    }
    assert!(
        ever_off_grid,
        "no kick left the grid in six games — the regression this test exists for was not exercised"
    );
}

#[test]
fn the_view_survives_every_state_at_the_compiled_capacity() {
    let dims = BoardDims::default();
    for seed in 100..102 {
        let (steps, _) = play(dims, seed);
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

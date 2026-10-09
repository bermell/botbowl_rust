//! Plan 061 (a)-(c): the roll models behind `RollModel`, each off by default.
//!
//! - `BounceModel::Catch`: a bounce onto a standing player is a catch attempt (the engine's
//!   `Bounce` hands it to `Catch`), onto a player on the ground it bounces on; the old `Settle`
//!   model dropped both whenever the ball could land.
//! - `PassScatterModel::Grouped`: an inaccurate pass's scatter (3 D8) and a wildly inaccurate
//!   pass's deviate (D6 x D8) at their exact landing distribution, grouped by consequence: one
//!   child per player the ball lands on (a catch), one for "goes out" (a throw-in), and the empty
//!   squares' mass carried by the most likely few.
//! - `ThrowInModel::Grouped`: the same for a throw-in's 3 directions x 2D6, with a re-throw (and
//!   any throw-in deep in a chain) falling back to the scripted throw, so chains end.
//!
//! Every enumeration must be a proper distribution, give each child its own state (recon_mcts
//! cannot hold two chance edges into one child), carry dice that make the engine produce the
//! claimed landing, and mirror with the board.

use std::collections::BTreeMap;

use botbowl_engine::core::dices::{RequestedRoll, RollResult, Sum2D6, D3, D6, D8};
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{
    Action, BallState, BoardDims, Coord, Direction, PlayerStatus, Position, SomeProcInput, TeamType,
};
use botbowl_engine::core::procedures::AnyProc;
use botbowl_engine::core::table::{PosAT, SimpleAT};
use botbowl_mcts::dynamics::ChanceModel;
use botbowl_mcts::roll_outcomes::{self, BounceModel, PassScatterModel, RollModel, ThrowInModel};
use botbowl_mcts::{BbAction, BloodBowlDynamics};
use recon_mcts::GameDynamics;

fn catch_model() -> RollModel {
    RollModel {
        bounce: BounceModel::Catch,
        ..RollModel::default()
    }
}

fn grouped_passes() -> RollModel {
    RollModel {
        pass_scatter: PassScatterModel::Grouped,
        ..RollModel::default()
    }
}

fn grouped_throw_ins() -> RollModel {
    RollModel {
        throw_in: ThrowInModel::Grouped,
        ..RollModel::default()
    }
}

fn children(state: &GameState, rolls: RollModel) -> Vec<(RollResult, f32)> {
    let req = state.pending_roll.expect("paused on a roll");
    roll_outcomes::enumerate_full(state, &req, ChanceModel::Exact, rolls)
        .into_iter()
        .map(|a| match a {
            BbAction::Chance { result, prob_bits } => (result, f32::from_bits(prob_bits)),
            other => panic!("expected a chance child, got {other:?}"),
        })
        .collect()
}

fn assert_distribution(outcomes: &[(RollResult, f32)]) {
    let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}: {outcomes:?}");
    assert!(outcomes.iter().all(|(_, p)| *p > 0.0), "{outcomes:?}");
}

/// `state` after the engine resolves `r` (no quiescent stepping).
fn after(state: &GameState, r: RollResult) -> GameState {
    let mut s = state.clone();
    s.step_with_roll_or_action(SomeProcInput::Roll(r));
    s
}

/// No two children reach the same state through the search's own `apply_action`.
fn assert_distinct_children(state: &GameState, outcomes: &[(RollResult, f32)]) {
    let gd = BloodBowlDynamics::default();
    let next: Vec<GameState> = outcomes
        .iter()
        .map(|(r, p)| gd.apply_action(state.clone(), &BbAction::chance(*r, *p)).unwrap())
        .collect();
    for i in 0..next.len() {
        for j in i + 1..next.len() {
            assert!(
                next[i] != next[j],
                "children {:?} and {:?} reach the same state",
                outcomes[i].0,
                outcomes[j].0
            );
        }
    }
}

fn set_status(state: &mut GameState, pos: Position, status: PlayerStatus) {
    let id = state.get_player_id_at(pos).unwrap();
    state.get_mut_player(id).unwrap().status = status;
}

// ---- (a) bounce -------------------------------------------------------------------------------

/// A home player at `ball - (1, 0)` moves onto the loose ball, fails the pickup and declines the
/// reroll: the engine pauses on the bounce's D8.
fn paused_on_bounce(ball: Position, setup: impl FnOnce(&mut GameStateBuilder)) -> GameState {
    let start = Position::new((ball.x - 1, ball.y));
    let mut builder = GameStateBuilder::new();
    builder.add_home_player(start).add_ball_pos(ball);
    setup(&mut builder);
    let mut state = builder.build();
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start)));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, ball)));
    state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
    assert_eq!(state.pending_roll, Some(RequestedRoll::D8));
    assert_eq!(state.proc_stack_top(), Some("Bounce"));
    state
}

fn d8_to(from: Position, to: Position) -> RollResult {
    RollResult::D8(D8::from(Direction::from((to.x - from.x, to.y - from.y))))
}

#[test]
fn a_bounce_onto_a_standing_player_is_a_catch_attempt_and_onto_a_downed_one_bounces_on() {
    let ball = Position::new((6, 5));
    let standing = Position::new((7, 5));
    let down = Position::new((7, 6));
    let mut state = paused_on_bounce(ball, |b| {
        b.add_away_player(standing).add_away_player(down);
    });
    set_status(&mut state, down, PlayerStatus::Down);

    // The shipped model drops both player squares (renormalised over the 6 free ones; the mover
    // stands on the ball's square, not next to it).
    let settle = children(&state, RollModel::default());
    assert_eq!(settle.len(), 6, "{settle:?}");

    let outcomes = children(&state, catch_model());
    assert_distribution(&outcomes);
    assert_eq!(outcomes.len(), 8, "every direction is its own child: {outcomes:?}");
    assert!(outcomes.iter().all(|(_, p)| (*p - 0.125).abs() < 1e-6), "{outcomes:?}");
    let catch = after(&state, d8_to(ball, standing));
    assert!(outcomes.iter().any(|(r, _)| *r == d8_to(ball, standing)));
    assert_eq!(
        catch.proc_stack_top(),
        Some("Catch"),
        "a standing player tries to catch"
    );
    assert!(matches!(catch.pending_roll, Some(RequestedRoll::D6PassFail(_))));
    let on = after(&state, d8_to(ball, down));
    assert_eq!(on.proc_stack_top(), Some("Bounce"), "a downed player cannot catch");
    assert_eq!(on.ball, BallState::InAir(down));
    assert_eq!(on.pending_roll, Some(RequestedRoll::D8));
    assert_distinct_children(&state, &outcomes);
}

#[test]
fn a_bounce_never_returns_to_a_player_square_it_has_been_through() {
    let ball = Position::new((6, 5));
    let standing = Position::new((7, 5));
    let mut state = paused_on_bounce(ball, |b| {
        b.add_away_player(standing);
    });
    state.bounce_squares.push(standing);
    let outcomes = children(&state, catch_model());
    assert_distribution(&outcomes);
    assert_eq!(outcomes.len(), 7, "the visited player square is dropped: {outcomes:?}");
    assert!(!outcomes.iter().any(|(r, _)| *r == d8_to(ball, standing)));
}

#[test]
fn a_bounce_against_the_edge_keeps_one_out_of_bounds_child() {
    let ball = Position::new((6, 1));
    let state = paused_on_bounce(ball, |b| {
        b.add_away_player(Position::new((7, 2)));
    });
    let outcomes = children(&state, catch_model());
    assert_distribution(&outcomes);
    // 3 out of bounds (one child, 3/8), the mover's own square is not a neighbour; 4 empty + 1 player.
    assert_eq!(outcomes.len(), 6, "{outcomes:?}");
    let out: Vec<_> = outcomes
        .iter()
        .filter(|(r, _)| matches!(r, RollResult::D8(d) if state.is_out(ball + Direction::from(*d))))
        .collect();
    assert_eq!(out.len(), 1);
    assert!((out[0].1 - 3.0 / 8.0).abs() < 1e-6);
    assert_eq!(
        out[0].0,
        RollResult::D8(D8::from(Direction::up())),
        "the straight-out exit"
    );
}

/// A kickoff bounce: every direction that ends in a touchback (out, or onto the kicking half)
/// reaches the same state, so they are one child.
#[test]
fn kickoff_bounce_touchbacks_are_one_child() {
    let mut base = GameStateBuilder::new_at_kickoff();
    base.set_dice_mode(DiceMode::RegisterRolls);
    base.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::KickoffAimMiddle)));
    assert_eq!(base.pending_roll, Some(RequestedRoll::Deviate));
    let kicking = base.info.kicking_this_drive;
    // Find a kick that comes down on an empty square next to the kicking half or the sideline.
    let mut found = None;
    'search: for n in (1..=6u8).rev() {
        for d in 1..=8u8 {
            let mut s = base.clone();
            s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Deviate(
                D6::try_from(n).unwrap(),
                D8::try_from(d).unwrap(),
            )));
            if s.pending_roll == Some(RequestedRoll::Sum2D6) {
                s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Sum2D6(Sum2D6::Four)));
            }
            let kick_bounce = matches!(s.proc_stack_peek(), Some(AnyProc::Bounce(b)) if b.is_kick());
            if s.pending_roll != Some(RequestedRoll::D8) || !kick_bounce {
                continue;
            }
            let ball = s.get_ball_position().unwrap();
            let touchbacks = Direction::all_directions_as_array()
                .iter()
                .filter(|dir| s.is_out(ball + **dir) || s.is_on_team_side(ball + **dir, kicking))
                .count();
            if touchbacks >= 2 {
                found = Some((s, touchbacks));
                break 'search;
            }
        }
    }
    let (state, touchbacks) = found.expect("a kick landing next to the kicking half or the sideline");
    let outcomes = children(&state, catch_model());
    assert_distribution(&outcomes);
    assert!(
        outcomes.len() <= 8 - touchbacks + 1,
        "{touchbacks} touchback directions must be one child: {outcomes:?}"
    );
    assert_distinct_children(&state, &outcomes);
}

// ---- (b) pass scatter / deviate ---------------------------------------------------------------

/// A home thrower at `thrower` with the ball passes to the square `target`; the D6 is resolved to
/// the first face that asks for `want` (Scatter or Deviate).
fn paused_on_pass_landing(
    dims: Option<BoardDims>,
    thrower: Position,
    target: Position,
    others: &[(Position, TeamType, PlayerStatus)],
    want: RequestedRoll,
) -> GameState {
    let mut builder = GameStateBuilder::new();
    if let Some(d) = dims {
        builder.with_board_dims(d);
    }
    builder.add_home_player(thrower).add_ball_pos(thrower);
    for (p, team, _) in others {
        match team {
            TeamType::Home => builder.add_home_player(*p),
            TeamType::Away => builder.add_away_player(*p),
        };
    }
    let mut base = builder.build();
    for (p, _, status) in others {
        set_status(&mut base, *p, *status);
    }
    pin(&mut base, thrower);
    base.set_dice_mode(DiceMode::RegisterRolls);
    base.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartPass, thrower)));
    base.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Pass, target)));
    assert_eq!(base.pending_roll, Some(RequestedRoll::D6));
    for face in 1..=6u8 {
        let mut s = base.clone();
        s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::D6(D6::try_from(face).unwrap())));
        if s.pending_roll == Some(want) {
            return s;
        }
    }
    panic!("no pass face asks for {want:?}")
}

/// No moves left: the pass is thrown from where the thrower stands, so its range (and the
/// wildly-inaccurate face) is the test's to choose.
fn pin(state: &mut GameState, thrower: Position) {
    let id = state.get_player_id_at(thrower).unwrap();
    let p = state.get_mut_player(id).unwrap();
    p.stats.ma = 0;
    p.moves = 2;
}

/// Brute force: where every dice combination of the pending scatter/deviate lands, from the
/// engine itself. `Some(square)` lands in bounds, `None` goes out (a throw-in).
fn engine_landings(state: &GameState) -> Vec<(RollResult, Option<Position>)> {
    let combos: Vec<RollResult> = match state.pending_roll {
        Some(RequestedRoll::Scatter) => {
            let mut v = Vec::new();
            for a in 1..=8u8 {
                for b in 1..=8u8 {
                    for c in 1..=8u8 {
                        let d = |x: u8| D8::try_from(x).unwrap();
                        v.push(RollResult::Scatter(d(a), d(b), d(c)));
                    }
                }
            }
            v
        }
        Some(RequestedRoll::Deviate) => (1..=6u8)
            .flat_map(|n| {
                (1..=8u8).map(move |d| RollResult::Deviate(D6::try_from(n).unwrap(), D8::try_from(d).unwrap()))
            })
            .collect(),
        other => panic!("not a pass landing: {other:?}"),
    };
    combos.into_iter().map(|r| (r, landing_of(state, r))).collect()
}

/// Where the ball comes down after the engine resolves `r` (stepping past interception prompts
/// cannot happen in these setups: no opponent stands between).
fn landing_of(state: &GameState, r: RollResult) -> Option<Position> {
    let s = after(state, r);
    if s.proc_stack_top() == Some("ThrowIn") {
        return None;
    }
    match s.ball {
        BallState::InAir(p) | BallState::OnGround(p) => Some(p),
        other => panic!(
            "unexpected ball after the pass: {other:?} (top {:?})",
            s.proc_stack_top()
        ),
    }
}

fn check_pass_landing(state: &GameState) {
    let brute = engine_landings(state);
    let total = brute.len() as f32;
    let outcomes = children(state, grouped_passes());
    assert_distribution(&outcomes);
    assert!(outcomes.len() <= 10, "fan-out {}: {outcomes:?}", outcomes.len());
    assert_distinct_children(state, &outcomes);
    // Exact mass of each catch square (a player stands there) and of going out.
    let mut catch_mass: BTreeMap<(Coord, Coord), f32> = BTreeMap::new();
    let mut out_mass = 0.0;
    for (_, l) in &brute {
        match l {
            Some(p) if state.get_player_at(*p).is_some() => *catch_mass.entry((p.x, p.y)).or_default() += 1.0 / total,
            Some(_) => {}
            None => out_mass += 1.0 / total,
        }
    }
    for (r, p) in &outcomes {
        match landing_of(state, *r) {
            None => assert!((p - out_mass).abs() < 1e-5, "out child {p} vs exact {out_mass}"),
            Some(l) if state.get_player_at(l).is_some() => {
                let want = catch_mass[&(l.x, l.y)];
                assert!((p - want).abs() < 1e-5, "catch child at {l:?}: {p} vs exact {want}");
            }
            Some(_) => {}
        }
    }
    let mass = |pred: &dyn Fn(Option<Position>) -> bool| -> f32 {
        outcomes
            .iter()
            .filter(|(r, _)| pred(landing_of(state, *r)))
            .map(|(_, p)| p)
            .sum()
    };
    let empty = mass(&|l| l.is_some_and(|p| state.get_player_at(p).is_none()));
    let exact_empty = 1.0 - out_mass - catch_mass.values().sum::<f32>();
    assert!(
        (empty - exact_empty).abs() < 1e-5,
        "empty mass {empty} vs exact {exact_empty}"
    );
}

#[test]
fn an_inaccurate_pass_scatters_at_its_exact_landing_odds() {
    let thrower = Position::new((5, 5));
    let target = Position::new((9, 5));
    // Teammates where the ball comes down (one on the ground: `DeflectOrResolve` still hands it a
    // catch), the opponent behind the thrower where it cannot intercept.
    let others = [
        (target, TeamType::Home, PlayerStatus::Up),
        (Position::new((10, 6)), TeamType::Home, PlayerStatus::Up),
        (Position::new((8, 3)), TeamType::Home, PlayerStatus::Down),
        (Position::new((2, 5)), TeamType::Away, PlayerStatus::Up),
    ];
    let state = paused_on_pass_landing(None, thrower, target, &others, RequestedRoll::Scatter);
    // The shipped model: one scripted child.
    assert_eq!(children(&state, RollModel::default()).len(), 1);
    check_pass_landing(&state);
}

#[test]
fn a_scatter_near_the_sideline_has_one_out_of_bounds_child() {
    let thrower = Position::new((5, 3));
    let target = Position::new((8, 1));
    let others = [(target, TeamType::Home, PlayerStatus::Up)];
    let state = paused_on_pass_landing(None, thrower, target, &others, RequestedRoll::Scatter);
    check_pass_landing(&state);
    let outs = children(&state, grouped_passes())
        .into_iter()
        .filter(|(r, _)| landing_of(&state, *r).is_none())
        .count();
    assert_eq!(outs, 1);
}

#[test]
fn a_wildly_inaccurate_pass_deviates_at_its_exact_landing_odds() {
    // A long pass (-2) from near the sideline: a 2 or 3 is wildly inaccurate.
    let thrower = Position::new((5, 2));
    let target = Position::new((12, 5));
    let others = [
        (target, TeamType::Home, PlayerStatus::Up),
        (Position::new((6, 3)), TeamType::Home, PlayerStatus::Up),
        (Position::new((5, 4)), TeamType::Home, PlayerStatus::Up),
        (Position::new((24, 14)), TeamType::Away, PlayerStatus::Up),
    ];
    let state = paused_on_pass_landing(None, thrower, target, &others, RequestedRoll::Deviate);
    assert_eq!(children(&state, RollModel::default()).len(), 1);
    check_pass_landing(&state);
}

/// Mirror the whole setup (squares reflected, teams swapped): the enumerated landing squares and
/// their probabilities must reflect too.
#[test]
fn pass_landing_children_mirror_with_the_board() {
    let dims = GameStateBuilder::new().build().board_dims;
    let flip = |p: Position| Position::new((dims.width - 1 - p.x, p.y));
    for want in [RequestedRoll::Scatter, RequestedRoll::Deviate] {
        let thrower = Position::new((6, 4));
        let target = Position::new((13, 2));
        let others = [
            (target, TeamType::Home, PlayerStatus::Up),
            (Position::new((14, 3)), TeamType::Home, PlayerStatus::Up),
            (Position::new((7, 6)), TeamType::Home, PlayerStatus::Up),
            (Position::new((24, 14)), TeamType::Away, PlayerStatus::Up),
        ];
        let state = paused_on_pass_landing(None, thrower, target, &others, want);
        // The mirror: Away throws from the reflected square. Build it with Away as the passing
        // team by giving the turn to Away through the mirrored builder state.
        let m_others: Vec<_> = others
            .iter()
            .map(|(p, t, s)| (flip(*p), botbowl_engine::core::model::other_team(*t), *s))
            .collect();
        let mirror = paused_on_pass_landing_away(flip(thrower), flip(target), &m_others, want);
        let squares = |s: &GameState, flipped: bool| -> Vec<String> {
            let mut v: Vec<String> = children(s, grouped_passes())
                .into_iter()
                .map(|(r, p)| {
                    let l = landing_of(s, r).map(|l| if flipped { flip(l) } else { l });
                    format!("{l:?}@{p:.6}")
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(squares(&state, true), squares(&mirror, false), "{want:?}");
    }
}

/// [`paused_on_pass_landing`] with Away passing: `others` are given in Away's frame.
fn paused_on_pass_landing_away(
    thrower: Position,
    target: Position,
    others: &[(Position, TeamType, PlayerStatus)],
    want: RequestedRoll,
) -> GameState {
    let mut builder = GameStateBuilder::new();
    builder
        .set_state(botbowl_engine::core::gamestate::BuilderState::Turn { turn: 1 })
        .add_away_player(thrower)
        .add_ball_pos(thrower);
    for (p, team, _) in others {
        match team {
            TeamType::Home => builder.add_home_player(*p),
            TeamType::Away => builder.add_away_player(*p),
        };
    }
    let mut base = builder.build();
    for (p, _, status) in others {
        set_status(&mut base, *p, *status);
    }
    // Hand the turn to Away: end Home's turn.
    if base.info.team_turn != TeamType::Away {
        base.step(Action::Simple(SimpleAT::EndTurn)).unwrap();
    }
    assert_eq!(base.info.team_turn, TeamType::Away);
    pin(&mut base, thrower);
    base.set_dice_mode(DiceMode::RegisterRolls);
    base.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartPass, thrower)));
    base.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Pass, target)));
    assert_eq!(base.pending_roll, Some(RequestedRoll::D6));
    for face in 1..=6u8 {
        let mut s = base.clone();
        s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::D6(D6::try_from(face).unwrap())));
        if s.pending_roll == Some(want) {
            return s;
        }
    }
    panic!("no pass face asks for {want:?}")
}

// ---- (c) throw-in -----------------------------------------------------------------------------

/// The ball on `ball` (an edge square) bounces out by `out`: the engine pauses on the throw-in.
fn paused_on_throw_in(
    dims: (Coord, Coord, usize),
    ball: Position,
    out: Direction,
    others: &[(Position, TeamType, PlayerStatus)],
) -> Option<GameState> {
    let dims = BoardDims::try_new(dims.0, dims.1, dims.2).ok()?;
    let start = Position::new((ball.x - 1, ball.y));
    let mut builder = GameStateBuilder::new();
    builder.with_board_dims(dims).add_home_player(start).add_ball_pos(ball);
    for (p, team, _) in others {
        match team {
            TeamType::Home => builder.add_home_player(*p),
            TeamType::Away => builder.add_away_player(*p),
        };
    }
    let mut state = builder.build();
    for (p, _, status) in others {
        set_status(&mut state, *p, *status);
    }
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start)));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, ball)));
    state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));
    state.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
    state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::D8(D8::from(out))));
    (state.pending_roll == Some(RequestedRoll::ThrowIn)).then_some(state)
}

fn throw_in_combos() -> Vec<(RollResult, u32)> {
    let mut v = Vec::new();
    for d in 1..=3u8 {
        for a in 1..=6u8 {
            for b in 1..=6u8 {
                v.push((
                    RollResult::ThrowIn {
                        direction: D3::try_from(d).unwrap(),
                        distance: Sum2D6::try_from(a + b).unwrap(),
                    },
                    1,
                ));
            }
        }
    }
    v
}

/// What the engine does with a throw-in roll: `Err(())` re-throws (landed out), `Ok((square,
/// catch))` lands on `square`, `catch` when a standing player there tries to catch it.
fn throw_in_landing(state: &GameState, r: RollResult) -> Result<(Position, bool), ()> {
    let s = after(state, r);
    if s.pending_roll == Some(RequestedRoll::ThrowIn) {
        return Err(());
    }
    let BallState::InAir(p) = s.ball else {
        panic!("throw-in left the ball {:?}", s.ball)
    };
    Ok((p, s.proc_stack_top() == Some("Catch")))
}

#[test]
fn a_throw_in_lands_at_its_exact_odds_grouped_by_consequence() {
    // Full pitch, ball out over the bottom sideline at x = 10; a standing receiver and a downed
    // player where throws land.
    let ball = Position::new((10, 15));
    let others = [
        (Position::new((10, 10)), TeamType::Away, PlayerStatus::Up),
        (Position::new((12, 12)), TeamType::Home, PlayerStatus::Down),
    ];
    let state = paused_on_throw_in((28, 17, 11), ball, Direction::down(), &others).expect("full pitch");
    assert_eq!(
        children(&state, RollModel::default()).len(),
        1,
        "shipped: one scripted throw"
    );
    let outcomes = children(&state, grouped_throw_ins());
    assert_distribution(&outcomes);
    assert!(outcomes.len() <= 10, "{outcomes:?}");
    assert_distinct_children(&state, &outcomes);
    let combos = throw_in_combos();
    let total = combos.len() as f32;
    let mut catch: BTreeMap<(Coord, Coord), f32> = BTreeMap::new();
    let mut out = 0.0;
    for (r, _) in &combos {
        match throw_in_landing(&state, *r) {
            Err(()) => out += 1.0 / total,
            Ok((p, true)) => *catch.entry((p.x, p.y)).or_default() += 1.0 / total,
            Ok(_) => {}
        }
    }
    assert!(catch.contains_key(&(10, 10)), "the receiver is reachable: {catch:?}");
    for (r, p) in &outcomes {
        match throw_in_landing(&state, *r) {
            Err(()) => assert!((p - out).abs() < 1e-5, "out {p} vs {out}"),
            Ok((l, true)) => assert!((p - catch[&(l.x, l.y)]).abs() < 1e-5, "catch at {l:?}"),
            Ok(_) => {}
        }
    }
}

#[test]
fn a_rethrow_falls_back_to_the_scripted_throw() {
    let ball = Position::new((10, 15));
    let state = paused_on_throw_in((28, 17, 11), ball, Direction::down(), &[]).expect("full pitch");
    let out_child = children(&state, grouped_throw_ins())
        .into_iter()
        .find(|(r, _)| throw_in_landing(&state, *r).is_err());
    let Some((r, _)) = out_child else {
        // A throw from mid-sideline on the full pitch can go out over the far side only with a
        // long diagonal; if none does, there is nothing to re-throw.
        return;
    };
    let rethrow = after(&state, r);
    assert_eq!(rethrow.pending_roll, Some(RequestedRoll::ThrowIn));
    let again = children(&rethrow, grouped_throw_ins());
    assert_eq!(again.len(), 1, "a re-throw is scripted: {again:?}");
    assert!(throw_in_landing(&rethrow, again[0].0).is_ok(), "and lands in bounds");
}

/// The 14x5 lesson (plan 060 §6): every chain of throw-ins and bounces off an edge must end. Walk
/// the whole chance tree below each edge throw-in on the loop's narrow boards (taking every
/// child, declining rerolls, stopping at any other decision or settled ball) and bound its size.
#[test]
fn every_throw_in_chain_ends_on_narrow_boards() {
    // (engine width, height, team size): 8x3, 12x5, 14x5, 14x7.
    let boards = [(10, 5, 2), (14, 7, 3), (16, 7, 3), (16, 9, 4)];
    let mut checked = 0;
    for (w, h, t) in boards {
        let (max_x, max_y) = (w - 2, h - 2);
        let exits = [
            (Position::new((max_x, h / 2)), Direction::from((1, 0))),
            (Position::new((w / 2, max_y)), Direction::from((0, 1))),
            (Position::new((w / 2, 1)), Direction::from((0, -1))),
            (Position::new((max_x, max_y)), Direction::from((1, 1))),
        ];
        for (ball, out) in exits {
            let Some(state) = paused_on_throw_in((w, h, t), ball, out, &[]) else {
                continue;
            };
            let model = RollModel {
                bounce: BounceModel::Catch,
                pass_scatter: PassScatterModel::Grouped,
                throw_in: ThrowInModel::Grouped,
            };
            let (nodes, depth) = chance_tree_size(&state, model, 0);
            assert!(
                nodes < 20_000 && depth <= 24,
                "{w}x{h} from {ball:?}: {nodes} chance nodes, depth {depth}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 8, "only {checked} exits fit this build");
}

/// (chance nodes, max depth) of the tree of rolls below `state`.
fn chance_tree_size(state: &GameState, model: RollModel, depth: usize) -> (usize, usize) {
    assert!(depth < 64, "runaway chain: {:?}", state.bounce_squares);
    let mut s = state.clone();
    // Step past the reroll offers a failed catch makes.
    while s.pending_roll.is_none() && !s.info.game_over {
        let actions = s.get_all_actions();
        if actions.contains(&Action::Simple(SimpleAT::DontUseReroll)) {
            s.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
        } else {
            return (0, depth);
        }
    }
    let Some(req) = s.pending_roll else { return (0, depth) };
    if !matches!(
        req,
        RequestedRoll::ThrowIn | RequestedRoll::D8 | RequestedRoll::D6PassFail(_)
    ) {
        return (0, depth);
    }
    let mut nodes = 1;
    let mut max_depth = depth;
    for a in roll_outcomes::enumerate_full(&s, &req, ChanceModel::Exact, model) {
        let BbAction::Chance { result, .. } = a else {
            unreachable!()
        };
        if matches!(req, RequestedRoll::D6PassFail(_)) && result == RollResult::Pass {
            continue; // caught
        }
        let (n, d) = chance_tree_size(&after(&s, result), model, depth + 1);
        nodes += n;
        max_depth = max_depth.max(d);
    }
    (nodes, max_depth)
}

//! Shared scaffolding for the plan-023 mirror-invariance tests.
//!
//! Blood Bowl is symmetric under "reflect the board about its vertical
//! midline and swap the two teams". Home attacks `x = 1` and Away attacks
//! `x = width - 2`, so any deterministic choice made in *absolute* board
//! coordinates is a side bias waiting to happen — plan 023 has caught four
//! of those, every one of which had first been argued symmetric by reading
//! the code. This module is the machinery for asserting the property
//! instead.

#![allow(dead_code)]

use botbowl_curriculum::random_start::{generate_random_start, RandomStartConfig};
use botbowl_engine::core::dices::{RollResult, D8};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{other_team, Action as EngineAction, BallState, BoardDims, Direction, Position, TeamType};
use botbowl_engine::core::table::SimpleAT;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The 14x7 curriculum tier (engine 16x9), the board every plan-023 mirror
/// measurement was taken on. Pinned rather than read from the environment
/// so the properties hold on any build.
pub fn tier() -> BoardDims {
    BoardDims::new(16, 9, 4)
}

pub fn states(n: u32, seed: u64) -> Vec<GameState> {
    let cfg = RandomStartConfig {
        board_dims: Some(tier()),
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| generate_random_start(&cfg, &mut rng)).collect()
}

pub fn flip(dims: BoardDims, p: Position) -> Position {
    Position::new((dims.width - 1 - p.x, p.y))
}

pub fn mirror_action(dims: BoardDims, a: EngineAction) -> EngineAction {
    match a {
        EngineAction::Positional(at, pos) => EngineAction::Positional(at, flip(dims, pos)),
        simple => simple,
    }
}

fn flip_d8(d: D8) -> D8 {
    let dir = Direction::from(d);
    D8::from(Direction {
        dx: -dir.dx,
        dy: dir.dy,
    })
}

/// The mirror image of a die result. Direction-valued results reflect in
/// `x`; everything else is side-agnostic and passes through. `None` for
/// `ThrowIn`, whose `D3` indexes a sideline-dependent direction table that
/// has no simple inverse — those pairs are skipped rather than guessed at.
pub fn mirror_roll(r: RollResult) -> Option<RollResult> {
    Some(match r {
        RollResult::D8(d) => RollResult::D8(flip_d8(d)),
        RollResult::Deviate(d6, d8) => RollResult::Deviate(d6, flip_d8(d8)),
        RollResult::Scatter(a, b, c) => RollResult::Scatter(flip_d8(a), flip_d8(b), flip_d8(c)),
        RollResult::ThrowIn { .. } => return None,
        other => other,
    })
}

/// Rebuild `s` reflected and team-swapped, through `GameStateBuilder`, so
/// the result is a real playable state — `GameState::mirrored` deliberately
/// leaves the procedure stack alone and so cannot be stepped.
///
/// Only valid for states from [`states`] (i.e. `generate_random_start`):
/// the reconstruction reproduces that generator's conventions, not an
/// arbitrary position's.
pub fn mirror_playable(s: &GameState, dims: BoardDims) -> GameState {
    let ball_pos = match s.ball {
        BallState::Carried(id) => s.get_player(id).expect("carrier exists").position,
        BallState::OnGround(p) | BallState::InAir(p) => p,
        BallState::OffPitch => panic!("random starts always have the ball on the pitch"),
    };
    let mut builder = GameStateBuilder::new();
    builder
        .with_board_dims(dims)
        .set_state(BuilderState::Turn { turn: 1 })
        .add_ball_pos(flip(dims, ball_pos));
    for p in s.get_players_on_pitch() {
        let pos = flip(dims, p.position);
        match p.stats.team {
            TeamType::Home => builder.add_away_player(pos),
            TeamType::Away => builder.add_home_player(pos),
        };
    }
    let mut m = builder.build();
    m.set_logging_state(false);
    // The builder hands turn 1 to Home; the mirror's mover is the swap of
    // `s`'s, so pass the turn on when `s` had Home to move.
    if s.available_actions.team == Some(TeamType::Home) {
        m.step_simple(SimpleAT::EndTurn);
    }
    m.info.home_turn = s.info.away_turn;
    m.info.away_turn = s.info.home_turn;
    if s.info.half == 2 {
        m.set_half(2);
    }
    m.home.score = s.away.score;
    m.away.score = s.home.score;
    m.info.kicking_first_half = other_team(s.info.kicking_first_half);
    m.info.kicking_this_drive = other_team(s.info.kicking_this_drive);
    // The one that matters most, and the one easiest to miss: `Half::step`
    // decides who takes the next team turn from *its own* copy of the
    // kicking team, not from `GameInfo`. Leave it and the mirrored state
    // gives its mover two consecutive turns.
    let kicking = s.kicking_this_half().expect("a half is in progress");
    assert!(m.set_kicking_this_half(other_team(kicking)), "mirror has no started half");
    m.set_dice_mode(DiceMode::RollDice);
    m
}

/// The **y** reflection: flip top-for-bottom and keep both teams where
/// they are. Blood Bowl is symmetric under this too, and — crucially — it
/// involves no Home/Away labels at all, so it is the control for any
/// measurement of x-mirror asymmetry. A statistic that reports a side bias
/// under the x mirror but nothing under the y mirror is measuring sides; one
/// that reports the same magnitude under both is measuring something else.
///
/// Same construction route and same caveats as [`mirror_playable`].
pub fn flip_y(dims: BoardDims, p: Position) -> Position {
    Position::new((p.x, dims.height - 1 - p.y))
}

pub fn mirror_y_playable(s: &GameState, dims: BoardDims) -> GameState {
    let ball_pos = match s.ball {
        BallState::Carried(id) => s.get_player(id).expect("carrier exists").position,
        BallState::OnGround(p) | BallState::InAir(p) => p,
        BallState::OffPitch => panic!("random starts always have the ball on the pitch"),
    };
    let mut builder = GameStateBuilder::new();
    builder
        .with_board_dims(dims)
        .set_state(BuilderState::Turn { turn: 1 })
        .add_ball_pos(flip_y(dims, ball_pos));
    for p in s.get_players_on_pitch() {
        let pos = flip_y(dims, p.position);
        match p.stats.team {
            TeamType::Home => builder.add_home_player(pos),
            TeamType::Away => builder.add_away_player(pos),
        };
    }
    let mut m = builder.build();
    m.set_logging_state(false);
    // No team swap here, so the mover is unchanged.
    if s.available_actions.team == Some(TeamType::Away) {
        m.step_simple(SimpleAT::EndTurn);
    }
    m.info.home_turn = s.info.home_turn;
    m.info.away_turn = s.info.away_turn;
    if s.info.half == 2 {
        m.set_half(2);
    }
    m.home.score = s.home.score;
    m.away.score = s.away.score;
    m.set_dice_mode(DiceMode::RollDice);
    m
}

pub fn mirror_y_action(dims: BoardDims, a: EngineAction) -> EngineAction {
    match a {
        EngineAction::Positional(at, pos) => EngineAction::Positional(at, flip_y(dims, pos)),
        simple => simple,
    }
}

/// A canonical, ID-free rendering of everything about a state that a
/// search or a bot can read. Player IDs are never printed — they are
/// assigned in construction order, so two states built by different routes
/// disagree on them while being the same position; anything that refers to
/// a player is rendered as its team plus square instead.
///
/// Pair with [`mirror_fingerprint`], which renders the *mirror* of a state
/// directly from the state, field by field. Comparing the two is the
/// mirror-invariance assertion. (Going through `GameState::mirrored()`
/// instead would not work here: that helper drops path offerings, so the
/// action lists would differ for a reason that has nothing to do with the
/// property under test.)
pub fn fingerprint(s: &GameState) -> String {
    render(s, None)
}

/// [`fingerprint`] of the y-reflection of `s` (top-for-bottom, teams
/// unchanged) — the reference for validating a y-mirror rebuild.
pub fn mirror_y_fingerprint(s: &GameState, dims: BoardDims) -> String {
    render(s, Some((dims, Axis::Y)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// Reflect in x **and** swap the benches — the side symmetry.
    X,
    /// Reflect in y only — the control symmetry, no team labels involved.
    Y,
}

/// [`fingerprint`] of the mirror of `s`, computed from `s` without ever
/// building a mirrored state.
pub fn mirror_fingerprint(s: &GameState, dims: BoardDims) -> String {
    render(s, Some((dims, Axis::X)))
}

fn render(s: &GameState, mirror: Option<(BoardDims, Axis)>) -> String {
    let team = |t: TeamType| match mirror {
        Some((_, Axis::X)) => other_team(t),
        _ => t,
    };
    let pos = |p: Position| match mirror {
        Some((dims, Axis::X)) => flip(dims, p),
        Some((dims, Axis::Y)) => flip_y(dims, p),
        None => p,
    };
    // A referenced player can have left the pitch (KO / casualty), in which
    // case there is no position to render. Both sides of a mirrored pair
    // reach that together, so a marker keeps them comparable.
    let player_ref = |id: Option<usize>| {
        id.map(|id| match s.get_player(id) {
            Ok(p) => format!("{:?}@{:?}", team(p.stats.team), pos(p.position)),
            Err(_) => "off-pitch".to_string(),
        })
    };

    let i = &s.info;
    let (home_turn, away_turn) = match mirror {
        Some((_, Axis::X)) => (i.away_turn, i.home_turn),
        _ => (i.home_turn, i.away_turn),
    };
    let (home_state, away_state) = match mirror {
        Some((_, Axis::X)) => (s.away, s.home),
        _ => (s.home, s.away),
    };
    let ball = match s.ball {
        BallState::Carried(id) => format!("Carried({})", player_ref(Some(id)).unwrap()),  // carrier is fielded by invariant
        BallState::OnGround(p) => format!("OnGround({:?})", pos(p)),
        BallState::InAir(p) => format!("InAir({:?})", pos(p)),
        BallState::OffPitch => "OffPitch".to_string(),
    };
    let mut players: Vec<String> = s
        .get_players_on_pitch()
        .map(|p| {
            format!(
                "{:?}@{:?} {:?} used={} moves={}",
                team(p.stats.team),
                pos(p.position),
                p.status,
                p.used,
                p.moves
            )
        })
        .collect();
    players.sort();
    let mut actions: Vec<EngineAction> = s
        .get_all_actions()
        .into_iter()
        .map(|a| match mirror {
            Some((dims, Axis::X)) => mirror_action(dims, a),
            Some((dims, Axis::Y)) => mirror_y_action(dims, a),
            None => a,
        })
        .collect();
    actions.sort();

    format!(
        "half={} turns={home_turn}/{away_turn} winner={:?} turnover={} active={:?} pat={:?} \
         team_turn={:?} over={} weather={:?} kick1st={:?} kickoff_by={:?} kick_drive={:?} \
         avail=[h{} f{} p{} b{}] td_by={:?} pickup={} blitz={}\n\
         home={home_state:?}\naway={away_state:?}\nball={ball}\nkicking_this_half={:?}\n\
         pending={:?}\nplayers={players:?}\nactions={actions:?}",
        i.half,
        i.winner.map(team),
        i.turnover,
        player_ref(i.active_player),
        i.player_action_type,
        team(i.team_turn),
        i.game_over,
        i.weather,
        team(i.kicking_first_half),
        i.kickoff_by_team.map(team),
        team(i.kicking_this_drive),
        i.handoff_available as u8,
        i.foul_available as u8,
        i.pass_available as u8,
        i.blitz_available as u8,
        player_ref(i.handle_td_by),
        i.pickup_this_activation,
        i.blitz_this_activation,
        s.kicking_this_half().map(team),
        s.pending_roll,
    )
}

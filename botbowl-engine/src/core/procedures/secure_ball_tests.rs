//! S3 Secure the Ball (the-game-of-blood-bowl, "SECURE THE BALL ACTION" and "SECURE THE BALL
//! ACTIONS!"): "A player may only declare a Secure the Ball Action if the ball is loose on the
//! ground and is not within 2 squares of any Standing opposition players ... may first make a free
//! Move Action, however, this Move Action must finish on a square containing a ball otherwise a
//! Turnover is caused. ... does not make an Agility Test; instead they roll a D6 and on a 2+ they
//! will automatically pick up the ball and their activation immediately ends. On a 1, the player
//! fails to pick up the ball and a Turnover is caused - the ball will then Bounce from the
//! player's square." "Only a single Secure the Ball Action can be declared each Turn."
use crate::core::dices::{D6Target, D8};
use crate::core::gamestate::GameStateBuilder;
use crate::core::model::Direction;
use crate::core::model::{Action, BallState, PlayerStatus, Position, TeamType, Weather};
use crate::core::pathing::{CustomIntoIter, PathFinder, PathingEvent, PositionOrEvent};
use crate::core::table::{PosAT, SimpleAT, Skill};

const START: (i8, i8) = (2, 5);
const BALL: (i8, i8) = (5, 5);

fn pos(xy: (i8, i8)) -> Position {
    Position::new(xy)
}

fn secure_offered(state: &crate::core::gamestate::GameState, at: (i8, i8)) -> bool {
    state.is_legal_action(&Action::Positional(PosAT::StartSecureBall, pos(at)))
        && state
            .get_all_actions()
            .contains(&Action::Positional(PosAT::StartSecureBall, pos(at)))
}

/// One Home player at `START`, the ball loose at `BALL`, one Away player far away; no re-rolls.
fn loose_ball_state() -> crate::core::gamestate::GameState {
    let mut state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_away_player(pos((14, 9)))
        .add_ball_pos(pos(BALL))
        .build();
    state.home.rerolls = 0;
    state.away.rerolls = 0;
    state
}

#[test]
fn secure_the_ball_is_offered_for_a_loose_ball_clear_of_opponents() {
    let state = loose_ball_state();
    assert!(state.home_to_act());
    assert!(secure_offered(&state, START));
}

#[test]
fn a_standing_opponent_within_two_squares_of_the_ball_forbids_it() {
    let state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_away_player(pos((7, 7)))
        .add_ball_pos(pos(BALL))
        .build();
    assert!(!secure_offered(&state, START));
    // Three squares away does not count.
    let state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_away_player(pos((8, 7)))
        .add_ball_pos(pos(BALL))
        .build();
    assert!(secure_offered(&state, START));
}

/// Only a *Standing* opposition player stops it: a Prone (or Stunned) one does not.
#[test]
fn a_prone_opponent_within_two_squares_does_not_forbid_it() {
    let mut state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_away_player(pos((6, 6)))
        .add_ball_pos(pos(BALL))
        .build();
    let away = state.get_player_id_at(pos((6, 6))).unwrap();
    state.get_mut_player_unsafe(away).status = PlayerStatus::Down;
    state.reask_turn_decision().unwrap();
    assert!(secure_offered(&state, START));
}

/// A team-mate near the ball is no obstacle either.
#[test]
fn a_teammate_near_the_ball_does_not_forbid_it() {
    let state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_home_player(pos((6, 5)))
        .add_away_player(pos((14, 9)))
        .add_ball_pos(pos(BALL))
        .build();
    assert!(secure_offered(&state, START));
    assert!(secure_offered(&state, (6, 5)));
}

#[test]
fn not_offered_when_the_ball_is_held() {
    let state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_home_player(pos(BALL))
        .add_away_player(pos((14, 9)))
        .add_ball_pos(pos(BALL))
        .build();
    assert!(matches!(state.ball, BallState::Carried(_)));
    assert!(!secure_offered(&state, START));
    assert!(!secure_offered(&state, BALL));
}

/// "If this is not possible, a player cannot declare this Action": the free Move must be able to
/// end on the ball.
#[test]
fn not_offered_when_the_ball_square_is_out_of_reach() {
    let state = GameStateBuilder::new()
        .add_home_player(pos((1, 1)))
        .add_home_player(pos((4, 4)))
        .add_away_player(pos((14, 9)))
        .add_ball_pos(pos((12, 5)))
        .build();
    // MA 6 + 2 Rushes = 8 squares; (1,1) is 11 away, (4,4) is 8 away.
    assert!(!secure_offered(&state, (1, 1)));
    assert!(secure_offered(&state, (4, 4)));
}

#[test]
fn a_two_secures_the_ball_and_ends_the_activation() {
    let mut state = loose_ball_state();
    let id = state.get_player_id_at(pos(START)).unwrap();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.fix_d6(2);
    state.step_positional(PosAT::Move, pos(BALL));

    assert_eq!(state.ball, BallState::Carried(id));
    assert!(state.home_to_act());
    assert!(!state.info.turnover);
    assert_eq!(state.proc_stack_top(), Some("Turn"));
    assert!(state.get_player_unsafe(id).used);
    assert_eq!(state.info.active_player, None);
}

/// No Agility Test: neither the player's AG nor the pick-up modifiers (rain here; marking cannot
/// apply, no Standing opponent is near) move the 2+.
#[test]
fn the_two_plus_ignores_agility_and_pickup_modifiers() {
    let mut state = loose_ball_state();
    state.info.weather = Weather::Rain;
    let id = state.get_player_id_at(pos(START)).unwrap();
    state.get_mut_player_unsafe(id).stats.agility = 6;
    state.reask_turn_decision().unwrap();
    state.step_positional(PosAT::StartSecureBall, pos(START));

    let path = state.get_paths().unwrap()[pos(BALL)]
        .clone()
        .expect("a path onto the ball");
    let events: Vec<PathingEvent> = path
        .iter()
        .filter_map(|e| match e {
            PositionOrEvent::Event(ev) => Some(ev),
            _ => None,
        })
        .collect();
    assert_eq!(events, vec![PathingEvent::SecureBall(D6Target::TwoPlus)]);

    state.fix_d6(2);
    state.step_positional(PosAT::Move, pos(BALL));
    assert_eq!(state.ball, BallState::Carried(id));
}

#[test]
fn a_one_is_a_turnover_and_the_ball_bounces_from_the_players_square() {
    let mut state = loose_ball_state();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.fix_d6(1);
    state.fix_d8(D8::One as u8);
    state.step_positional(PosAT::Move, pos(BALL));

    let direction = Direction::from(D8::One);
    assert_eq!(state.ball, BallState::OnGround(pos(BALL) + direction));
    assert!(state.away_to_act());
}

/// A team re-roll may re-roll the D6 (the re-roll rules exclude only Scatter, Armour, Injury, ...).
#[test]
fn a_team_reroll_may_reroll_the_secure_roll() {
    let mut state = loose_ball_state();
    state.home.rerolls = 1;
    let id = state.get_player_id_at(pos(START)).unwrap();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.fix_d6(1);
    state.step_positional(PosAT::Move, pos(BALL));
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
    state.fix_d6(2);
    state.step_simple(SimpleAT::UseReroll);
    assert_eq!(state.ball, BallState::Carried(id));
    assert!(state.home_to_act());
}

/// Sure Hands: "may re-roll the D6 when attempting to pick up the ball, though not when making a
/// Secure the Ball Action".
#[test]
fn sure_hands_does_not_reroll_the_secure_roll() {
    let mut state = loose_ball_state();
    let id = state.get_player_id_at(pos(START)).unwrap();
    state.get_mut_player_unsafe(id).stats.give_skill(Skill::SureHands);
    state.reask_turn_decision().unwrap();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.fix_d6(1);
    state.fix_d8(D8::One as u8);
    state.step_positional(PosAT::Move, pos(BALL));
    assert!(state.away_to_act());
    assert!(matches!(state.ball, BallState::OnGround(_)));
}

#[test]
fn ending_the_activation_off_the_ball_is_a_turnover() {
    let mut state = loose_ball_state();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.step_positional(PosAT::Move, pos((3, 5)));
    assert!(state.home_to_act());
    state.step_simple(SimpleAT::EndPlayerTurn);
    assert!(state.away_to_act());
    assert_eq!(state.ball, BallState::OnGround(pos(BALL)));
}

#[test]
fn only_one_secure_the_ball_per_team_turn() {
    let mut state = GameStateBuilder::new()
        .add_home_player(pos(START))
        .add_home_player(pos((2, 7)))
        .add_away_player(pos((14, 9)))
        .add_ball_pos(pos(BALL))
        .build();
    state.home.rerolls = 0;
    // Secure it, then put the ball back on the ground to isolate the once-per-turn rule.
    let id = state.get_player_id_at(pos(START)).unwrap();
    state.step_positional(PosAT::StartSecureBall, pos(START));
    state.fix_d6(6);
    state.step_positional(PosAT::Move, pos(BALL));
    assert_eq!(state.ball, BallState::Carried(id));
    state.set_ball(BallState::OnGround(pos((5, 7))));
    state.reask_turn_decision().unwrap();
    assert!(!state.info.secure_ball_available);
    assert!(!secure_offered(&state, (2, 7)));
    assert!(state.is_legal_action(&Action::Positional(PosAT::StartMove, pos((2, 7)))));

    // The next team turn offers it again.
    state.step_simple(SimpleAT::EndTurn);
    state.step_simple(SimpleAT::EndTurn);
    assert!(state.home_to_act());
    assert!(state.info.secure_ball_available);
    assert!(secure_offered(&state, (2, 7)));
}

/// Away gets it on its own turn, by the same rule.
#[test]
fn away_may_secure_the_ball_too() {
    let mut state = GameStateBuilder::new()
        .add_home_player(pos((14, 9)))
        .add_away_player(pos((8, 5)))
        .add_ball_pos(pos(BALL))
        .build();
    assert!(!secure_offered(&state, (14, 9)), "Home cannot reach the ball");
    state.step_simple(SimpleAT::EndTurn);
    assert!(state.away_to_act());
    assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
    assert!(secure_offered(&state, (8, 5)));
}

#[test]
fn pathing_reaches_the_ball_with_the_secure_event_only_under_secure_the_ball() {
    let state = loose_ball_state();
    let id = state.get_player_id_at(pos(START)).unwrap();
    // A plain move picks up with an Agility Test.
    let node = PathFinder::safest_path_to(&state, id, pos(BALL)).unwrap().unwrap();
    assert!(node
        .iter()
        .any(|e| matches!(e, PositionOrEvent::Event(PathingEvent::Pickup(_)))));
}

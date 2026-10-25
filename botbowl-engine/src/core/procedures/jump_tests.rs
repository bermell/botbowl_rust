//! S3 "Jumping over Players" (The Game of Blood Bowl, Jumping over Players) and the Leap skill
//! (Skills and Traits, Leap). Included from `movement_procs.rs`.
use crate::core::dices::D6Target;
use crate::core::gamestate::{GameState, GameStateBuilder};
use crate::core::model::*;
use crate::core::pathing::{CustomIntoIter, PathFinder, PathingEvent, PositionOrEvent};
use crate::core::table::*;

const MOVER: Position = Position { x: 3, y: 3 };
const DOWNED: Position = Position { x: 4, y: 3 };
const BEYOND: Position = Position { x: 5, y: 3 };

/// A Home AG 3+ mover at (3,3) with movement allowance `ma`, an Away player at (4,3) — Prone
/// when `downed_status` is `Down` — and, when `wall`, Home team-mates on the rest of column 4
/// from y=1 to y=5, so with a small MA the only way past is over (4,3). No team re-rolls.
fn jump_state(ma: u8, wall: bool, downed_status: PlayerStatus, extra_away: &[(Coord, Coord)]) -> GameState {
    let mut builder = GameStateBuilder::new();
    builder.add_home_player(MOVER).add_away_player(DOWNED).add_away_players(extra_away);
    if wall {
        builder.add_home_players(&[(4, 1), (4, 2), (4, 4), (4, 5)]);
    }
    let mut state = builder.build();
    state.get_mut_team(TeamType::Home).rerolls = 0;
    let id = state.get_player_id_at(MOVER).unwrap();
    state.get_mut_player_unsafe(id).stats.ma = ma;
    let downed = state.get_player_id_at(DOWNED).unwrap();
    state.get_mut_player_unsafe(downed).status = downed_status;
    state
}

fn mover(state: &GameState) -> PlayerID {
    state.get_player_id_at(MOVER).unwrap()
}

fn jump_event(node: &std::sync::Arc<crate::core::pathing::Node>) -> Option<PathingEvent> {
    node.iter().find_map(|item| match item {
        PositionOrEvent::Event(e @ PathingEvent::Jump { .. }) => Some(e),
        _ => None,
    })
}

/// Pathfinding: with the way past blocked, the path over the Prone player is a single Agility
/// Test (AG 3+, no Marking: 3+), costing two squares of Movement Allowance, at 2/3.
#[test]
fn a_path_jumps_over_a_prone_player() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    state.step_positional(PosAT::StartMove, MOVER);
    let node = state.get_paths().unwrap()[BEYOND].clone().expect("a path over the prone player");
    let items: Vec<PositionOrEvent> = node.iter().collect();
    assert_eq!(
        items,
        vec![PositionOrEvent::Event(PathingEvent::Jump {
            to: BEYOND,
            target: D6Target::ThreePlus
        })]
    );
    assert!((node.prob - 4.0 / 6.0).abs() < 1e-6, "prob {}", node.prob);
    // Squares adjacent to the jumper are not jump targets, and the square of the downed player
    // is occupied: neither is reachable here.
    assert!(state.get_paths().unwrap()[DOWNED].is_none());
}

/// Jump passed: the player is placed Standing in the target square and may continue their
/// activation — here one more square, on the same path.
#[test]
fn a_passed_jump_lands_standing_and_the_path_continues() {
    let mut state = jump_state(3, true, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    let further = BEYOND + (1, 0);
    state.fix_d6(3); // Jump: AG 3+
    state.step_positional(PosAT::Move, further);
    let player = state.get_player_unsafe(id);
    assert_eq!(player.position, further);
    assert_eq!(player.status, PlayerStatus::Up);
    assert_eq!(player.moves_left(), 0, "two squares for the jump, one for the step");
    assert!(!state.away_to_act());
}

/// Jump failed (not a natural 1): placed in the target square, Falls Over — a turnover.
#[test]
fn a_failed_jump_falls_over_in_the_target_square() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    state.fix_d6(2); // Jump fails
    state.fix_d6(1); // armour
    state.fix_d6(1);
    state.step_positional(PosAT::Move, BEYOND);
    let player = state.get_player_unsafe(id);
    assert_eq!(player.position, BEYOND);
    assert_eq!(player.status, PlayerStatus::Down);
    assert!(state.away_to_act(), "turnover");
}

/// A natural 1 on the Jump: the player Falls Over in the square they are in.
#[test]
fn a_natural_one_falls_over_in_the_start_square() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    state.fix_d6(1); // Jump: a natural 1
    state.fix_d6(1); // armour
    state.fix_d6(1);
    state.step_positional(PosAT::Move, BEYOND);
    let player = state.get_player_unsafe(id);
    assert_eq!(player.position, MOVER);
    assert_eq!(player.status, PlayerStatus::Down);
    assert!(state.away_to_act(), "turnover");
}

/// Arm Bar (Skills and Traits): "If an opposing player Falls Over as a result of attempting to
/// Dodge, Leap or Jump away from a square in this player's Tackle Zone", +1 to the armour or
/// injury roll — whether the jumper falls in the target square or, on a natural 1, in place. An
/// 8 breaks AV 9+ only with it.
#[test]
fn arm_bar_hits_a_player_falling_from_a_jump_out_of_its_zone() {
    let fall = |arm_bar: bool, jump_roll: u8| {
        let mut state = jump_state(2, true, PlayerStatus::Down, &[(2, 3)]);
        let id = mover(&state);
        state.get_mut_player_unsafe(id).stats.armour = 9;
        if arm_bar {
            let marker = state.get_player_id_at(Position::new((2, 3))).unwrap();
            state.get_mut_player_unsafe(marker).stats.give_skill(Skill::ArmBar);
        }
        state.step_positional(PosAT::StartMove, MOVER);
        state.fix_d6(jump_roll); // the Jump (AG 3+, -1 for the marker: 4+) fails
        state.fix_d6(5); // armour: 8
        state.fix_d6(3);
        if arm_bar {
            state.fix_d6(1); // injury: Stunned
            state.fix_d6(2);
        }
        state.step_positional(PosAT::Move, BEYOND);
        state.get_player_unsafe(id).status
    };
    for jump_roll in [1, 3] {
        assert_eq!(fall(false, jump_roll), PlayerStatus::Down);
        assert_eq!(fall(true, jump_roll), PlayerStatus::Stunned);
    }
}

/// A failed Jump may be re-rolled with a team re-roll (no skill re-rolls it: Dodge is for
/// Dodging only).
#[test]
fn a_failed_jump_may_be_team_rerolled() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    state.get_mut_team(TeamType::Home).rerolls = 1;
    let id = mover(&state);
    state.get_mut_player_unsafe(id).stats.give_skill(Skill::Dodge);
    state.step_positional(PosAT::StartMove, MOVER);
    state.fix_d6(1); // Jump: a natural 1
    state.step_positional(PosAT::Move, BEYOND);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
    state.fix_d6(4); // re-rolled: passes
    state.step_simple(SimpleAT::UseReroll);
    assert_eq!(state.get_player_unsafe(id).position, BEYOND);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
    assert_eq!(state.get_team(TeamType::Home).rerolls, 0);
}

/// The modifier is the number of opposition players Marking the square the jumper is in or the
/// square jumped into, whichever is higher — not their sum, and no extra Dodge for leaving a
/// Marked square.
#[test]
fn the_jump_modifier_is_the_higher_of_the_two_squares_marking() {
    // Start Marked by two, target by one: -2.
    let state = jump_state(6, false, PlayerStatus::Down, &[(2, 2), (2, 4), (6, 3)]);
    let node = PathFinder::jump_path(&state, mover(&state), BEYOND).expect("a legal jump");
    let items: Vec<PositionOrEvent> = node.iter().collect();
    assert_eq!(
        items,
        vec![PositionOrEvent::Event(PathingEvent::Jump {
            to: BEYOND,
            target: D6Target::FivePlus
        })]
    );
    // Start Marked by one, target by two: -2 as well.
    let state = jump_state(6, false, PlayerStatus::Down, &[(2, 3), (6, 2), (6, 4)]);
    let node = PathFinder::jump_path(&state, mover(&state), BEYOND).expect("a legal jump");
    assert_eq!(
        jump_event(&node),
        Some(PathingEvent::Jump {
            to: BEYOND,
            target: D6Target::FivePlus
        })
    );
}

/// Short of Movement Allowance, the jumper Rushes — every Rush rolled before the Jump.
#[test]
fn rushes_are_rolled_before_the_jump() {
    let mut state = jump_state(6, true, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.get_mut_player_unsafe(id).moves = 5; // one square left: one Rush
    state.step_positional(PosAT::StartMove, MOVER);
    let node = state.get_paths().unwrap()[BEYOND].clone().unwrap();
    let items: Vec<PositionOrEvent> = node.iter().collect();
    assert_eq!(
        items,
        vec![
            PositionOrEvent::Event(PathingEvent::GFI(D6Target::TwoPlus)),
            PositionOrEvent::Event(PathingEvent::Jump {
                to: BEYOND,
                target: D6Target::ThreePlus
            }),
        ]
    );
    state.fix_d6(2); // Rush
    state.fix_d6(3); // Jump
    state.step_positional(PosAT::Move, BEYOND);
    let player = state.get_player_unsafe(id);
    assert_eq!(player.position, BEYOND);
    assert_eq!(player.moves_left(), 0);
    assert_eq!(player.gfis_left(), 1);
}

/// A failed Rush before a Jump: the player Falls Over in the square they are in.
#[test]
fn a_failed_rush_before_a_jump_falls_over_in_the_start_square() {
    let mut state = jump_state(6, true, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.get_mut_player_unsafe(id).moves = 6; // no squares left: two Rushes
    state.step_positional(PosAT::StartMove, MOVER);
    state.fix_d6(1); // Rush fails
    state.fix_d6(1); // armour
    state.fix_d6(1);
    state.step_positional(PosAT::Move, BEYOND);
    let player = state.get_player_unsafe(id);
    assert_eq!(player.position, MOVER);
    assert_eq!(player.status, PlayerStatus::Down);
    assert!(state.away_to_act(), "turnover");
}

/// Only over a Prone or Stunned player, and only into a square not adjacent to the jumper.
#[test]
fn jumps_are_only_over_downed_players_into_squares_not_adjacent_to_the_jumper() {
    let jumpable = |status: PlayerStatus, to: Position| {
        let mut state = jump_state(6, false, status, &[]);
        state.step_positional(PosAT::StartMove, MOVER);
        state.is_legal_action(&Action::Positional(PosAT::Jump, to))
    };
    assert!(jumpable(PlayerStatus::Down, BEYOND));
    assert!(jumpable(PlayerStatus::Stunned, BEYOND));
    assert!(jumpable(PlayerStatus::Down, BEYOND + (0, 1)), "diagonally beyond");
    assert!(!jumpable(PlayerStatus::Up, BEYOND), "not over a Standing player");
    assert!(!jumpable(PlayerStatus::Down, MOVER + (1, 1)), "adjacent to the jumper");
    assert!(!jumpable(PlayerStatus::Down, BEYOND + (1, 0)), "not adjacent to the downed player");
}

/// The hidden action: a human picks one specific jump. Legal, never listed for bots.
#[test]
fn the_jump_action_is_legal_but_hidden_from_bots() {
    let mut state = jump_state(6, false, PlayerStatus::Down, &[]);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    let jump = Action::Positional(PosAT::Jump, BEYOND);
    assert!(state.is_legal_action(&jump));
    assert!(state.get_all_actions().iter().all(|a| !matches!(a, Action::Positional(PosAT::Jump, _))));
    state.fix_d6(3);
    state.step_positional(PosAT::Jump, BEYOND);
    assert_eq!(state.get_player_unsafe(id).position, BEYOND);
    assert_eq!(state.get_player_unsafe(id).moves_left(), 4);
}

/// The toggle (`GameState::set_jump_paths`) takes jumps out of the pathfinding — here the square
/// beyond the wall is out of reach — while the hidden action still offers them.
#[test]
fn jump_paths_off_removes_jumps_from_paths_but_not_the_action() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    state.set_jump_paths(false);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    assert!(state.get_paths().unwrap()[BEYOND].is_none());
    assert!(state.is_legal_action(&Action::Positional(PosAT::Jump, BEYOND)));
    state.fix_d6(6);
    state.step_positional(PosAT::Jump, BEYOND);
    assert_eq!(state.get_player_unsafe(id).position, BEYOND);
}

/// Leap: over any single adjacent square, whatever is in it — a Standing opponent here. The
/// opponent Marks both squares (-1), and Leap's reduction stops at -1: 4+.
#[test]
fn leap_goes_over_a_standing_player() {
    let mut state = jump_state(2, true, PlayerStatus::Up, &[]);
    let id = mover(&state);
    state.get_mut_player_unsafe(id).stats.give_skill(Skill::Leap);
    state.step_positional(PosAT::StartMove, MOVER);
    let node = state.get_paths().unwrap()[BEYOND].clone().expect("a leap over the standing player");
    assert_eq!(
        jump_event(&node),
        Some(PathingEvent::Jump {
            to: BEYOND,
            target: D6Target::FourPlus
        })
    );
    assert!(state.is_legal_action(&Action::Positional(PosAT::Jump, BEYOND)));
    // ... and over an empty square or a team-mate.
    assert!(state.is_legal_action(&Action::Positional(PosAT::Jump, Position::new((5, 1)))));
    assert!(state.is_legal_action(&Action::Positional(PosAT::Jump, Position::new((3, 5)))));
}

/// Leap reduces the negative modifier by 1, to a minimum of -1: -1 stays -1, -2 is -1, -3 is -2.
#[test]
fn leap_reduces_the_modifier_by_one_to_at_least_minus_one() {
    let leap_target = |markers: &[(Coord, Coord)]| {
        let mut state = jump_state(6, false, PlayerStatus::Down, markers);
        let id = mover(&state);
        state.get_mut_player_unsafe(id).stats.give_skill(Skill::Leap);
        match jump_event(&PathFinder::jump_path(&state, id, BEYOND).unwrap()) {
            Some(PathingEvent::Jump { target, .. }) => target,
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(leap_target(&[]), D6Target::ThreePlus);
    assert_eq!(leap_target(&[(2, 3)]), D6Target::FourPlus, "-1 stays -1");
    assert_eq!(leap_target(&[(2, 2), (2, 4)]), D6Target::FourPlus, "-2 becomes -1");
    assert_eq!(leap_target(&[(2, 2), (2, 3), (2, 4)]), D6Target::FivePlus, "-3 becomes -2");
}

/// A Jump into the ball's square picks it up after the Jump.
#[test]
fn a_jump_onto_the_ball_picks_it_up_after_the_jump() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    state.ball = BallState::OnGround(BEYOND);
    let id = mover(&state);
    state.step_positional(PosAT::StartMove, MOVER);
    let node = state.get_paths().unwrap()[BEYOND].clone().unwrap();
    let items: Vec<PositionOrEvent> = node.iter().collect();
    assert_eq!(
        items,
        vec![
            PositionOrEvent::Event(PathingEvent::Jump {
                to: BEYOND,
                target: D6Target::ThreePlus
            }),
            PositionOrEvent::Event(PathingEvent::Pickup(D6Target::ThreePlus)),
        ]
    );
    state.fix_d6(3); // Jump
    state.fix_d6(3); // Pick-up
    state.step_positional(PosAT::Move, BEYOND);
    assert_eq!(state.ball, BallState::Carried(id));
}

/// Secure the Ball: a Jump onto the ball ends in the Secure the Ball's plain 2+, not the
/// Agility Test's pick-up.
#[test]
fn a_jump_onto_the_ball_secures_it_in_a_secure_the_ball_action() {
    let mut state = jump_state(2, true, PlayerStatus::Down, &[]);
    state.ball = BallState::OnGround(BEYOND);
    state.reask_turn_decision().unwrap();
    state.step_positional(PosAT::StartSecureBall, MOVER);
    let node = state.get_paths().unwrap()[BEYOND].clone().unwrap();
    let items: Vec<PositionOrEvent> = node.iter().collect();
    assert_eq!(
        items,
        vec![
            PositionOrEvent::Event(PathingEvent::Jump {
                to: BEYOND,
                target: D6Target::ThreePlus
            }),
            PositionOrEvent::Event(PathingEvent::SecureBall(D6Target::TwoPlus)),
        ]
    );
}

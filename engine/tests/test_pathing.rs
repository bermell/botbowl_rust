extern crate rust_bb;
use itertools::Either;
use rust_bb::core::dices::BlockDice;
use rust_bb::core::dices::D6Target;
use rust_bb::core::model::*;
use rust_bb::core::pathing::CustomIntoIter;
use rust_bb::core::pathing::NodeIteratorItem;
use rust_bb::core::table::*;
use rust_bb::core::{
    gamestate::GameStateBuilder,
    model::{Action, Position, TeamType},
    pathing::{PathFinder, PathingEvent},
    table::PosAT,
};
use std::iter::zip;
#[test]
fn one_long_path() -> Result<()> {
    let starting_pos = Position::new((1, 1));
    let state = GameStateBuilder::new()
        .add_home_player(starting_pos)
        .add_away_players(&[(1, 2), (2, 3), (2, 4), (5, 3), (6, 4)])
        .add_ball((4, 6))
        .build();
    let id = state.get_player_id_at(starting_pos).unwrap();
    let paths = PathFinder::player_paths(&state, id)?;

    let expected_steps: Vec<NodeIteratorItem> = vec![
        Either::Left(Position::new((2, 1))),
        Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
        Either::Left(Position::new((3, 1))),
        Either::Right(PathingEvent::Dodge(D6Target::ThreePlus)),
        Either::Left(Position::new((3, 2))),
        Either::Left(Position::new((4, 3))),
        Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
        Either::Left(Position::new((4, 4))),
        Either::Right(PathingEvent::Dodge(D6Target::FourPlus)),
        Either::Left(Position::new((4, 5))),
        Either::Right(PathingEvent::Dodge(D6Target::ThreePlus)),
        Either::Left(Position::new((4, 6))),
        Either::Right(PathingEvent::GFI(D6Target::TwoPlus)),
        Either::Right(PathingEvent::Pickup(D6Target::ThreePlus)),
    ];

    let expected_prob = 0.03086;
    let path = paths.get(4, 6).clone().unwrap();

    for (i, (expected, actual)) in zip(expected_steps, path.iter()).enumerate() {
        if expected != actual {
            panic!("Step {}: {:?} != {:?}", i, expected, actual);
        }
    }

    assert!((expected_prob - path.prob).abs() < 0.0001);

    Ok(())
}

#[test]
fn double_gfi_foul() {
    let start_pos = Position::new((10, 1));
    let target_pos = Position::new((13, 1));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(target_pos)
        .build();
    let victim_id = state.get_player_id_at(target_pos).unwrap();
    state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;
    let id = state.get_player_id_at(start_pos).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.get_mut_player_unsafe(id).moves = ma;
    assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
    assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

    state.step_positional(PosAT::StartFoul, start_pos);
    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_d6(2); //GFI

    state.step_positional(PosAT::Move, target_pos + (-1, 0));

    state.fixes.fix_d6(4);
    state.fixes.fix_d6(5);
    state.fixes.fix_d6(1);
    state.fixes.fix_d6(2);
    state.step_positional(PosAT::Foul, target_pos);

    assert_eq!(
        state
            .get_player_unsafe(id)
            .position
            .distance_to(&target_pos),
        1
    );
    assert_eq!(
        state.get_player_unsafe(victim_id).status,
        PlayerStatus::Stunned
    );
}
#[test]
fn double_gfi_handoff_with_incremental_steps() {
    let start_pos = Position::new((10, 1));
    let target_pos = Position::new((13, 1));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_home_player(target_pos)
        .add_ball_pos(start_pos)
        .build();
    let id = state.get_player_id_at(start_pos).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.get_mut_player_unsafe(id).moves = ma;
    assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
    assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

    state.step_positional(PosAT::StartHandoff, start_pos);
    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_d6(2); //GFI

    state.step_positional(PosAT::Move, target_pos + (-1, 0));

    state.fixes.fix_d6(4); //Catch
    state.step_positional(PosAT::Handoff, target_pos);

    let carrier_id = state.get_player_id_at(target_pos).unwrap();
    assert_eq!(state.ball, BallState::Carried(carrier_id));
    assert_eq!(
        state
            .get_player_unsafe(id)
            .position
            .distance_to(&target_pos),
        1
    );
}
#[test]
fn double_gfi_handoff() {
    let start_pos = Position::new((10, 1));
    let target_pos = Position::new((13, 1));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_home_player(target_pos)
        .add_ball_pos(start_pos)
        .build();
    let id = state.get_player_id_at(start_pos).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.get_mut_player_unsafe(id).moves = ma;
    assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
    assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

    state.step_positional(PosAT::StartHandoff, start_pos);
    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_d6(4); //Catch

    state.step_positional(PosAT::Handoff, target_pos);

    let carrier_id = state.get_player_id_at(target_pos).unwrap();
    assert_eq!(state.ball, BallState::Carried(carrier_id));
    assert_eq!(
        state
            .get_player_unsafe(id)
            .position
            .distance_to(&target_pos),
        1
    );
}
#[test]
fn double_gfi_blitz() {
    let start_pos = Position::new((10, 1));
    let target_pos = Position::new((12, 1));
    let push_pos = target_pos + (1, 0);
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(target_pos)
        .build();
    let id = state.get_player_id_at(start_pos).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.get_mut_player_unsafe(id).moves = ma;
    assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
    assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

    state.step_positional(PosAT::StartBlitz, start_pos);

    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_d6(2); //GFI
    state.fixes.fix_blockdice(BlockDice::Pow);
    state.step_positional(PosAT::Block, target_pos);

    state.step_simple(SimpleAT::SelectPow);
    state.step_positional(PosAT::Push, target_pos + (1, 0));
    state.fixes.fix_d6(1); //armor
    state.fixes.fix_d6(1); //armor
    state.step_positional(PosAT::FollowUp, target_pos);

    assert_eq!(
        state.get_player_at(push_pos).unwrap().status,
        PlayerStatus::Down
    );
    assert_eq!(state.get_player_at(target_pos).unwrap().id, id);
}
#[test]
fn foul_pathing() {
    let mut field = "".to_string();
    field += "a ah \n";
    field += "  hh \n";
    field += "  h  \n";
    let start_pos = Position::new((5, 5));
    let foul_pos = start_pos + (2, 0);
    let fouler_pos = foul_pos + (0, 2);
    let foul_from_pos = foul_pos + (1, -1);
    let mut state = GameStateBuilder::new().add_str(start_pos, &field).build();

    let victim_id = state.get_player_id_at(foul_pos).unwrap();
    state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;
    assert_eq!(
        state.get_player_unsafe(victim_id).stats.team,
        TeamType::Away
    );
    assert_eq!(
        state.get_player_unsafe(victim_id).status,
        PlayerStatus::Down
    );

    let id = state.get_player_id_at(fouler_pos).unwrap();
    state.step_positional(PosAT::StartFoul, fouler_pos);

    state.fixes.fix_d6(4); //armor
    state.fixes.fix_d6(2); //armor
    state.fixes.fix_d6(2); //injury
    state.fixes.fix_d6(3); //injury

    state.step_positional(PosAT::Foul, foul_pos);

    assert_eq!(state.get_player_unsafe(id).position, foul_from_pos);
}
#[test]
fn standup_pathing() {
    let start_pos = Position::new((5, 5));
    let target = Position::new((8, 8));
    let push_to = target + (1, 1);
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(target)
        .build();

    let id = state.get_player_id_at(start_pos).unwrap();
    state.get_mut_player_unsafe(id).status = PlayerStatus::Down;

    state.step_positional(PosAT::StartBlitz, start_pos);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);

    state.fixes.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, target);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
    assert_eq!(
        state.get_player_unsafe(id).moves_left(),
        state.get_player_unsafe(id).stats.ma - 3 - 3
    );

    state.step_simple(SimpleAT::SelectPush);
    state.step_positional(PosAT::Push, push_to);
    state.step_positional(PosAT::FollowUp, target);

    assert!(!state.is_legal_action(&Action::Positional(PosAT::Block, push_to)));
}

extern crate botbowl_engine;
use botbowl_engine::core::dices::BlockDice;
use botbowl_engine::core::dices::D8;
use botbowl_engine::core::model::*;
use botbowl_engine::core::table::*;
use botbowl_engine::core::{
    gamestate::GameStateBuilder,
    model::{Action, Position},
    table::PosAT,
};

#[test]
fn touchdown() {
    let start_pos = Position::new((2, 1));
    let td_pos = Position::new((1, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_ball_pos(start_pos)
        .build();

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    state.step_positional(PosAT::StartMove, start_pos);
    state.step_positional(PosAT::Move, td_pos);

    assert_eq!(state.home.score, 1);
    assert_eq!(state.away.score, 0);
    assert_eq!(state.get_players_on_pitch().count(), 0);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
}

#[test]
fn failed_gfi_touchdown() {
    let start_pos = Position::new((2, 5));
    let td_pos = Position::new((1, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_ball_pos(start_pos)
        .build();

    let id = state.get_player_id_at(start_pos).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.get_mut_player_unsafe(id).moves = ma;
    assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
    assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    state.step_positional(PosAT::StartMove, start_pos);
    state.fixes.fix_d6(1);
    state.step_positional(PosAT::Move, td_pos);

    state.fixes.fix_d8(4);
    state.fixes.fix_d6(1);
    state.fixes.fix_d6(1);
    state.step_simple(SimpleAT::DontUseReroll);

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    assert_eq!(state.ball, BallState::OnGround(td_pos + (1, 0)));
    assert_eq!(state.get_player_unsafe(id).position, td_pos);
}

#[test]
fn pushed_to_touchdown() {
    let carrier_pos = Position::new((2, 5));
    let blocker_pos = Position::new((3, 5));
    let td_pos = carrier_pos + (carrier_pos - blocker_pos);
    let mut state = GameStateBuilder::new()
        .add_home_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .add_away_player(blocker_pos)
        .build();

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);
    state.step_simple(SimpleAT::EndTurn);
    state.step_positional(PosAT::StartBlock, blocker_pos);
    state.fixes.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, carrier_pos);
    state.step_simple(SimpleAT::SelectPush);
    state.step_positional(PosAT::Push, td_pos);
    state.step_positional(PosAT::FollowUp, carrier_pos);

    assert_eq!(state.home.score, 1);
    assert_eq!(state.away.score, 0);
    assert_eq!(state.get_players_on_pitch().count(), 0);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
}

#[test]
fn no_td_when_knocked_down_with_ball() {
    let carrier_pos = Position::new((2, 5));
    let blocker_pos = Position::new((3, 5));
    let td_pos = carrier_pos + (carrier_pos - blocker_pos);
    let mut state = GameStateBuilder::new()
        .add_home_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .add_away_player(blocker_pos)
        .build();

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);
    state.step_simple(SimpleAT::EndTurn);
    state.step_positional(PosAT::StartBlock, blocker_pos);
    state.fixes.fix_blockdice(BlockDice::Pow);
    state.step_positional(PosAT::Block, carrier_pos);
    state.step_simple(SimpleAT::SelectPow);
    state.step_positional(PosAT::Push, td_pos);
    state.fixes.fix_d6(1);
    state.fixes.fix_d6(1);
    state.fixes.fix_d8(4);
    state.step_positional(PosAT::FollowUp, blocker_pos);
    // state.step_simple(SimpleAT::EndPlayerTurn);
    state.step_simple(SimpleAT::EndTurn);
    // state.step_positional(PosAT::StartMove, td_pos);
    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);
    assert_eq!(state.ball, BallState::OnGround(td_pos + (1, 0)));
}

#[test]
fn follow_up_to_touchdown() {
    let carrier_pos = Position::new((2, 5));
    let victim_pos = Position::new((1, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .add_away_player(victim_pos)
        .build();

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);
    state.step_positional(PosAT::StartBlock, carrier_pos);
    state.fixes.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, victim_pos);
    state.step_simple(SimpleAT::SelectPush);
    //no need to select push position because crowd
    state.fixes.fix_d6(1);
    state.fixes.fix_d6(1);
    state.step_positional(PosAT::FollowUp, victim_pos);

    assert_eq!(state.home.score, 1);
    assert_eq!(state.away.score, 0);
    assert_eq!(state.get_players_on_pitch().count(), 0);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
}

#[test]
fn touchdown_pickup_in_endzone() {
    let start_pos = Position::new((2, 5));
    let td_pos = Position::new((1, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_ball_pos(td_pos)
        .build();
    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    state.step_positional(PosAT::StartMove, start_pos);

    state.fixes.fix_d6(3);
    state.step_positional(PosAT::Move, td_pos);

    assert_eq!(state.home.score, 1);
    assert_eq!(state.away.score, 0);
    assert_eq!(state.get_players_on_pitch().count(), 0);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
}
#[test]
fn no_td_when_failed_pickup_in_endzone() {
    let start_pos = Position::new((2, 5));
    let td_pos = Position::new((1, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_ball_pos(td_pos)
        .build();
    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    state.step_positional(PosAT::StartMove, start_pos);

    state.fixes.fix_d6(2);
    state.step_positional(PosAT::Move, td_pos);

    state.fixes.fix_d8(4);
    state.step_simple(SimpleAT::DontUseReroll);

    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);
}
#[test]
fn touchdown_when_catching_bouncing_ball() {
    let mut field = "".to_string();
    field += "hh \n";
    field += " A \n";
    field += "  h\n";
    let td_pos = Position::new((1, 3));
    let carrier_pos = td_pos + (1, 1);
    let blocker_pos = carrier_pos + (1, 1);
    let push_pos = carrier_pos + (-1, 0);
    let mut state = GameStateBuilder::new().add_str(td_pos, &field).build();
    assert_eq!(state.home.score, 0);
    assert_eq!(state.away.score, 0);

    state.step_positional(PosAT::StartBlock, blocker_pos);

    state.fixes.fix_blockdice(BlockDice::Pow);
    state.fixes.fix_blockdice(BlockDice::Pow);

    state.step_positional(PosAT::Block, carrier_pos);
    state.step_simple(SimpleAT::SelectPow);
    state.step_positional(PosAT::Push, push_pos);

    state.fixes.fix_d6(1); //armor
    state.fixes.fix_d6(2); //armor
    state.fixes.fix_d6(4); //catch
    state
        .fixes
        .fix_d8(D8::from(Direction { dx: 0, dy: -1 }) as u8); // bounce direction up
    state.step_positional(PosAT::FollowUp, blocker_pos);

    assert_eq!(state.home.score, 1);
    assert_eq!(state.away.score, 0);

    assert_eq!(state.get_players_on_pitch().count(), 0);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
}

extern crate botbowl_engine;
use botbowl_engine::core::dices::BlockDice;
use botbowl_engine::core::model::*;
use botbowl_engine::core::table::*;
use botbowl_engine::core::{
    gamestate::GameStateBuilder,
    model::{Action, DugoutPlace, PlayerStats, Position, TeamType},
    table::PosAT,
};

#[test]
fn crowd_surf_ball_carrier() {
    let carrier_pos = Position::new((5, 1));
    let blocker_pos = Position::new((5, 2));
    let mut state = GameStateBuilder::new()
        .add_home_player(blocker_pos)
        .add_away_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .build();

    state.step_positional(PosAT::StartBlock, blocker_pos);

    state.fixes.fix_blockdice(BlockDice::Pow);

    state.step_positional(PosAT::Block, carrier_pos);
    state.step_simple(SimpleAT::SelectPow);

    state.fixes.fix_d6(1); //armor
    state.fixes.fix_d6(1); //armor
    state.fixes.fix_d6(3); //throw in direction down
    state.fixes.fix_d6(1); //throw in length
    state.fixes.fix_d6(1); //throw in length
    state.fixes.fix_d8(2); //bounce direction down

    state.step_positional(PosAT::FollowUp, carrier_pos);

    assert_eq!(state.ball, BallState::OnGround(Position::new((5, 4))));

    assert!(matches!(
        state.get_dugout().next(),
        Some(DugoutPlayer {
            place: DugoutPlace::Reserves,
            stats: PlayerStats {
                team: TeamType::Away,
                ..
            },
            ..
        })
    ));
}

#[test]
fn handoff() {
    let start_pos = Position::new((2, 1));
    let target_pos = Position::new((5, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_home_player(target_pos)
        .add_ball_pos(start_pos)
        .build();
    let start_id = state.get_player_id_at(start_pos).unwrap();
    let carrier_id = state.get_player_id_at(target_pos).unwrap();

    state.step_positional(PosAT::StartHandoff, start_pos);

    state.fixes.fix_d6(6);
    state.step_positional(PosAT::Handoff, target_pos);

    // state.fixes.fix_d6(3);
    // state.step_simple(SimpleAT::UseReroll);

    assert!(state.get_player_unsafe(start_id).used);
    assert_eq!(state.ball, BallState::Carried(carrier_id));
}
#[test]
fn can_only_handoff_when_carrying_the_ball() {
    let start_pos = Position::new((2, 1));
    let target_pos = Position::new((5, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_home_player(target_pos)
        .build();
    state.step_positional(PosAT::StartHandoff, start_pos);
    assert!(!state.is_legal_action(&Action::Positional(PosAT::Handoff, target_pos)));
}

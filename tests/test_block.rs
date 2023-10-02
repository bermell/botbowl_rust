extern crate rust_bb;
use rust_bb::core::dices::BlockDice;
use rust_bb::core::model::*;
use rust_bb::core::table::*;
use rust_bb::core::{
    gamestate::GameStateBuilder,
    model::{DugoutPlace, PlayerStats, Position, TeamType},
    table::PosAT,
};

#[test]
fn crowd_chain_push() {
    let mut field = "".to_string();
    field += " aa\n";
    field += " aa\n";
    field += "h  \n";
    let first_pos = Position::new((5, 1));
    let mut state = GameStateBuilder::new().add_str(first_pos, &field).build();

    state.step_positional(PosAT::StartBlock, Position::new((5, 3)));
    state.fixes.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, Position::new((6, 2)));
    state.step_simple(SimpleAT::SelectPush);

    state.step_positional(PosAT::Push, Position::new((6, 1)));
    state.fixes.fix_d6(1);
    state.fixes.fix_d6(1);

    state.step_positional(PosAT::FollowUp, Position::new((6, 2)));

    state.step_simple(SimpleAT::EndTurn);

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
fn blitz() {
    let start_pos = Position::new((2, 1));
    let target_pos = Position::new((5, 5));
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(target_pos)
        .build();
    state.step_positional(PosAT::StartBlitz, start_pos);

    state.fixes.fix_blockdice(BlockDice::Skull);
    state.step_positional(PosAT::Block, target_pos);
}

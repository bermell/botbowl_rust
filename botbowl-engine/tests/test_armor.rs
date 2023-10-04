extern crate botbowl_engine;
use botbowl_engine::core::model::*;
use botbowl_engine::core::{
    gamestate::GameStateBuilder,
    model::{DugoutPlace, PlayerStats, Position, TeamType},
    table::PosAT,
};

#[test]
fn foul_ejected_at_armor() {
    let start_pos = Position::new((5, 5));
    let foul_pos = start_pos + (2, 0);
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(foul_pos)
        .build();

    let victim_id = state.get_player_id_at(foul_pos).unwrap();
    state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;

    state.step_positional(PosAT::StartFoul, start_pos);

    state.fixes.fix_d6(5); //armor
    state.fixes.fix_d6(5); //armor
    state.fixes.fix_d6(2); //injury
    state.fixes.fix_d6(1); //injury

    state.step_positional(PosAT::Foul, foul_pos);

    assert!(matches!(
        state.get_dugout().next(),
        Some(DugoutPlayer {
            place: DugoutPlace::Ejected,
            stats: PlayerStats {
                team: TeamType::Home,
                ..
            },
            ..
        })
    ));
}
#[test]
fn foul_ejected_at_injury() {
    let start_pos = Position::new((5, 5));
    let foul_pos = start_pos + (2, 0);
    let mut state = GameStateBuilder::new()
        .add_home_player(start_pos)
        .add_away_player(foul_pos)
        .build();

    let victim_id = state.get_player_id_at(foul_pos).unwrap();
    state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;

    state.step_positional(PosAT::StartFoul, start_pos);

    state.fixes.fix_d6(5); //armor
    state.fixes.fix_d6(6); //armor
    state.fixes.fix_d6(2); //injury
    state.fixes.fix_d6(2); //injury

    state.step_positional(PosAT::Foul, foul_pos);

    assert!(matches!(
        state.get_dugout().next(),
        Some(DugoutPlayer {
            place: DugoutPlace::Ejected,
            stats: PlayerStats {
                team: TeamType::Home,
                ..
            },
            ..
        })
    ));
}

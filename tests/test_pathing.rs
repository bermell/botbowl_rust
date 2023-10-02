extern crate rust_bb;

use itertools::Either;
use rust_bb::core::dices::D6Target;
use rust_bb::core::model::Result;
use rust_bb::core::pathing::CustomIntoIter;
use rust_bb::core::pathing::NodeIteratorItem;
use rust_bb::core::{
    gamestate::GameStateBuilder,
    model::Position,
    pathing::{PathFinder, PathingEvent},
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

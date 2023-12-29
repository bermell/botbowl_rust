use serde::Serialize;

use crate::core::model::ProcInput;
use crate::core::model::{Action, AvailableActions, PlayerID, PlayerStatus, ProcState, Procedure};
use crate::core::pathing::{
    event_ends_player_action, CustomIntoIter, NodeIterator, PathFinder, PathingEvent,
    PositionOrEvent,
};
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::procedures::{ball_procs, block_procs};
use crate::core::table::*;

use crate::core::{dices::D6Target, gamestate::GameState};

use super::{casualty_procs, AnyProc};

#[derive(Debug, Serialize)]
pub struct GfiProc {
    target: D6Target,
    id: PlayerID,
}
impl GfiProc {
    fn new(id: PlayerID, target: D6Target) -> AnyProc {
        AnyProc::GfiProc(SimpleProcContainer::new(GfiProc { target, id }))
    }
}
impl SimpleProc for GfiProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::SureFeet)
    }

    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Debug, Serialize)]
pub struct StandUp {
    id: PlayerID,
}
impl StandUp {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::StandUp(StandUp { id })
    }
}
impl Procedure for StandUp {
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
        debug_assert_eq!(
            game_state.get_player_unsafe(self.id).status,
            PlayerStatus::Down
        );
        game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Up;
        game_state.get_mut_player_unsafe(self.id).add_move(3);

        ProcState::Done
    }
}
#[derive(Debug, Serialize)]
pub struct DodgeProc {
    target: D6Target,
    id: PlayerID,
}
impl DodgeProc {
    fn new(id: PlayerID, target: D6Target) -> AnyProc {
        AnyProc::DodgeProc(SimpleProcContainer::new(DodgeProc { target, id }))
    }
}
impl SimpleProc for DodgeProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Dodge)
    }

    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
fn proc_from_roll(roll: PathingEvent, active_player: PlayerID) -> AnyProc {
    match roll {
        PathingEvent::Dodge(target) => DodgeProc::new(active_player, target),
        PathingEvent::GFI(target) => GfiProc::new(active_player, target),
        PathingEvent::Pickup(target) => ball_procs::PickupProc::new(active_player, target),
        PathingEvent::Block(id, dices) => block_procs::Block::new(dices, id),
        PathingEvent::Handoff(id, target) => ball_procs::Catch::new(id, target),
        PathingEvent::Touchdown(id) => ball_procs::Touchdown::new(id),
        PathingEvent::Foul(victim, target) => {
            casualty_procs::Armor::new_foul(victim, target, active_player)
        }
        PathingEvent::StandUp => StandUp::new(active_player),
        PathingEvent::Pass { to, pass, modifer } => ball_procs::Pass::new(to, pass, modifer),
    }
}

#[derive(Debug, Serialize)]
enum MoveActionState {
    Init,
    ActivePath(NodeIterator),
    SelectPath,
}
#[derive(Debug, Serialize)]
pub struct MoveAction {
    player_id: PlayerID,
    state: MoveActionState,
}
impl MoveAction {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::MoveAction(MoveAction {
            state: MoveActionState::Init,
            player_id: id,
        })
    }
    fn continue_along_path(path: &mut NodeIterator, game_state: &mut GameState) -> ProcState {
        let player_id = game_state.info.active_player.unwrap();

        for next_event in path.by_ref() {
            match next_event {
                PositionOrEvent::Position(position) => {
                    game_state.move_player(player_id, position).unwrap();
                    game_state.log(format!("Moved to {:?}", position));
                    game_state.get_mut_player_unsafe(player_id).add_move(1);
                }
                PositionOrEvent::Event(roll) => {
                    if event_ends_player_action(&roll) {
                        game_state.get_mut_player_unsafe(player_id).used = true;
                    }
                    return ProcState::NotDoneNew(proc_from_roll(roll, player_id));
                }
            }
        }
        ProcState::NotDone
    }
    fn available_actions(&self, game_state: &GameState) -> Box<AvailableActions> {
        let player = game_state.get_player_unsafe(self.player_id);
        let mut aa = AvailableActions::new(player.stats.team);
        aa.insert_paths(PathFinder::player_paths(game_state, self.player_id).unwrap());
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }
}
impl Procedure for MoveAction {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        if game_state.info.handle_td_by.is_some() || game_state.info.turnover {
            // game_state.get_mut_player_unsafe(self.player_id).used = true;
            return ProcState::Done;
        }

        match game_state.get_player(self.player_id) {
            Ok(player) if player.used => return ProcState::Done,
            Err(_) => return ProcState::Done, // player not on field anymore
            _ => (),
        }

        match (input, &mut self.state) {
            (ProcInput::Nothing, MoveActionState::Init) => {
                self.state = MoveActionState::SelectPath;
                ProcState::NeedAction(self.available_actions(game_state))
            }
            (ProcInput::Nothing, MoveActionState::ActivePath(path)) => {
                let proc_state = MoveAction::continue_along_path(path, game_state);
                if path.is_empty() {
                    self.state = MoveActionState::Init;
                }
                proc_state
            }
            (ProcInput::Action(Action::Positional(_, position)), MoveActionState::SelectPath) => {
                let mut path = game_state
                    .available_actions
                    .take_path(position)
                    .unwrap()
                    .iter();
                let proc_state = MoveAction::continue_along_path(&mut path, game_state);
                if path.is_empty() {
                    self.state = MoveActionState::Init;
                } else {
                    self.state = MoveActionState::ActivePath(path);
                }
                proc_state
            }
            (ProcInput::Action(Action::Simple(SimpleAT::EndPlayerTurn)), _) => {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
                ProcState::Done
            }
            _ => unreachable!(),
        }
    }
}
#[cfg(test)]
mod tests {

    use crate::core::gamestate::GameState;
    use crate::core::pathing::{CustomIntoIter, PositionOrEvent};
    use std::collections::HashMap;
    use std::iter::zip;

    use crate::core::dices::{BlockDice, D6Target};
    use crate::core::model::*;
    use crate::core::pathing::{PathFinder, PathingEvent};
    use crate::core::table::*;
    use crate::core::{
        gamestate::GameStateBuilder,
        model::{Action, DugoutPlace, Position},
        table::PosAT,
    };
    use crate::standard_state;

    #[test]
    fn path_with_two_failures() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let target_pos = Position::new((3, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((1, 2)))
            .build();

        state.step_positional(PosAT::StartMove, start_pos);

        state.fixes.fix_d6(1);

        state.step_positional(PosAT::Move, target_pos);

        state.fixes.fix_d6(4); //succeed first reroll
        state.fixes.fix_d6(1); //fail next dodge
        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d6(1); //armor

        state.step_simple(SimpleAT::UseReroll);

        assert_eq!(
            state.get_player_at(target_pos).unwrap().status,
            PlayerStatus::Down
        );

        assert!(state.get_player_at(target_pos).unwrap().used);

        Ok(())
    }

    #[test]
    fn failed_dodge_ko() -> Result<()> {
        let mut state = standard_state();
        let id = state.get_player_id_at_coord(2, 2).unwrap();
        assert!(state.get_dugout().next().is_none());

        state.step_positional(PosAT::StartMove, Position::new((2, 2)));

        state.fixes.fix_d6(2);
        state.step_positional(PosAT::Move, Position::new((2, 1)));

        state.fixes.fix_d6(4); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d6(4); //injury
        state.fixes.fix_d6(5); //injury
        state.step_simple(SimpleAT::DontUseReroll);

        assert!(state.get_player_id_at_coord(2, 1).is_none());
        assert!(state.get_players_on_pitch().all(|player| player.id != id));

        assert!(matches!(
            state.get_dugout().next(),
            Some(DugoutPlayer {
                place: DugoutPlace::KnockOut,
                ..
            })
        ));

        assert_eq!(state.get_dugout().count(), 1);
        Ok(())
    }

    #[test]
    fn gfi_reroll() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

        let id = state.get_player_id_at(start_pos).unwrap();

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));

        state.fixes.fix_d6(1); //fail first (2+)
        state.step_positional(PosAT::Move, Position::new((9, 1)));

        state.fixes.fix_d6(2); //succeed with team reroll
        state.fixes.fix_d6(2); //succeed next gfi roll
        state.step_simple(SimpleAT::UseReroll);

        let state = state;
        let player = state.get_player(id).unwrap();
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Move, Position::new((9, 2)))));
        assert_eq!(state.get_player_id_at_coord(9, 1).unwrap(), id);
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll());
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2);
        // assert_eq!(state.get_legal_positions(PosAT::Move).len(), 0);
        assert_eq!(player.total_movement_left(), 0);
        assert_eq!(player.gfis_left(), 0);
        assert_eq!(player.moves_left(), 0);

        Ok(())
    }

    #[test]
    fn dodge_reroll() -> Result<()> {
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((2, 1)))
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();

        state.get_mut_player(id)?.stats.give_skill(Skill::Dodge);
        assert!(state.get_player(id).unwrap().has_skill(Skill::Dodge));

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));

        state.fixes.fix_d6(3); //fail first (4+)
        state.fixes.fix_d6(4); //Succeed on skill reroll
        state.fixes.fix_d6(2); //fail second dodge  (3+)

        state.step_positional(PosAT::Move, Position::new((3, 3)));
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge));

        state.fixes.fix_d6(3); //succeed with team reroll
        state.step_simple(SimpleAT::UseReroll);

        assert_eq!(state.get_player_id_at_coord(3, 3).unwrap(), id);
        assert!(!state.get_team_from_player(id).unwrap().can_use_reroll());
        assert_eq!(state.get_team_from_player(id).unwrap().rerolls, 2);
        assert_eq!(state.get_mut_player(id).unwrap().total_movement_left(), 6);
        assert_eq!(state.get_mut_player(id).unwrap().gfis_left(), 2);
        assert_eq!(state.get_mut_player(id).unwrap().moves_left(), 4);
        state.step_simple(SimpleAT::EndPlayerTurn);

        Ok(())
    }
    #[test]
    fn long_move_action() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let move_target = Position::new((2, 5));

        assert!(state.get_player_at(starting_pos).is_some());
        assert!(state.get_player_at(move_target).is_none());

        state.step_positional(PosAT::StartMove, starting_pos);

        state.fixes.fix_d6(6);
        state.fixes.fix_d6(6);
        state.fixes.fix_d6(6);
        state.step_positional(PosAT::Move, move_target);

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step_simple(SimpleAT::EndPlayerTurn);

        assert!(state.get_player_at(move_target).unwrap().used);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartMove, move_target)));

        Ok(())
    }

    #[test]
    fn start_move_action() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let move_target = Position::new((4, 1));

        assert!(state.get_player_at(starting_pos).is_some());
        assert!(state.get_player_at(move_target).is_none());

        state.step_positional(PosAT::StartMove, starting_pos);
        state.step_positional(PosAT::Move, move_target);

        assert!(state.get_player_at(starting_pos).is_none());
        assert!(state.get_player_at(move_target).is_some());

        state.step_simple(SimpleAT::EndPlayerTurn);

        assert!(state.get_player_at(move_target).unwrap().used);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartMove, move_target)));

        Ok(())
    }

    #[test]
    fn pathing() -> Result<()> {
        let mut state = standard_state();
        let starting_pos = Position::new((3, 1));
        let id = state.get_player_id_at(starting_pos).unwrap();
        state.step_positional(PosAT::StartMove, starting_pos);
        let paths = PathFinder::player_paths(&state, id)?;

        let mut errors = Vec::new();

        for x in 1..8 {
            for y in 1..8 {
                let pos = Position::new((x, y));
                match (state.get_player_id_at(pos), &paths[pos]) {
                    (Some(_), None) => (),
                    (None, Some(_)) => (),
                    (Some(_), Some(_)) => {
                        errors.push(format!("Found path already occupied square ({},{})", x, y))
                    }
                    (None, None) => errors.push(format!("Missing a path to ({},{})!", x, y)),
                }
            }
        }
        let no_errors: Vec<String> = Vec::new();
        assert_eq!(no_errors, errors);
        Ok(())
    }

    #[test]
    fn pathing_probs() -> Result<()> {
        let starting_pos = Position::new((3, 2));
        let state = GameStateBuilder::new()
            .add_home_player(starting_pos)
            .add_away_players(&[(1, 3), (3, 3), (4, 2)])
            .build();

        let id = state.get_player_id_at(starting_pos).unwrap();

        let paths = PathFinder::player_paths(&state, id)?;

        let mut pos_to_prob: HashMap<(usize, usize), Option<f32>> = HashMap::new();
        pos_to_prob.insert((1, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((1, 2), Some(2.0 / 3.0));
        pos_to_prob.insert((1, 3), None);
        pos_to_prob.insert((1, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((2, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((2, 2), Some(2.0 / 3.0));
        pos_to_prob.insert((2, 3), Some(1.0 / 3.0));
        pos_to_prob.insert((2, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((3, 1), Some(2.0 / 3.0));
        pos_to_prob.insert((3, 2), None);
        pos_to_prob.insert((3, 3), None);
        pos_to_prob.insert((3, 4), Some(2.0 / 9.0));
        pos_to_prob.insert((4, 1), Some(1.0 / 2.0));
        pos_to_prob.insert((4, 2), None);
        pos_to_prob.insert((4, 3), Some(1.0 / 3.0));
        pos_to_prob.insert((4, 4), Some(2.0 / 9.0));

        let mut errors = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for x in 1..5 {
            for y in 1..5 {
                match (pos_to_prob.get(&(x, y)).unwrap(), paths.get(x, y)) {
                    (Some(correct_prob), Some(path))
                        if (*correct_prob - path.prob).abs() > 0.001 =>
                    {
                        errors.push(format!(
                            "Path to ({}, {}) has wrong prob. \nExpected prob: {}\nGot prob: {}\n",
                            x, y, *correct_prob, path.prob
                        ))
                    }
                    (Some(correct_prob), Some(path))
                        if (*correct_prob - path.prob).abs() <= 0.001 => {}
                    (None, None) => (),
                    (Some(_), None) => errors.push(format!("No path to ({}, {})", x, y)),
                    (None, Some(path)) => errors.push(format!(
                        "There shouldn't be a path to ({}, {}). Found: {:?}",
                        x, y, path
                    )),
                    _ => (),
                }
            }
        }

        let no_errors: Vec<String> = Vec::new();
        assert_eq!(no_errors, errors);

        Ok(())
    }

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

        let expected_steps: Vec<PositionOrEvent> = vec![
            PositionOrEvent::Position(Position::new((2, 1))),
            PositionOrEvent::Event(PathingEvent::Dodge(D6Target::FourPlus)),
            PositionOrEvent::Position(Position::new((3, 1))),
            PositionOrEvent::Event(PathingEvent::Dodge(D6Target::ThreePlus)),
            PositionOrEvent::Position(Position::new((3, 2))),
            PositionOrEvent::Position(Position::new((4, 3))),
            PositionOrEvent::Event(PathingEvent::Dodge(D6Target::FourPlus)),
            PositionOrEvent::Position(Position::new((4, 4))),
            PositionOrEvent::Event(PathingEvent::Dodge(D6Target::FourPlus)),
            PositionOrEvent::Position(Position::new((4, 5))),
            PositionOrEvent::Event(PathingEvent::Dodge(D6Target::ThreePlus)),
            PositionOrEvent::Position(Position::new((4, 6))),
            PositionOrEvent::Event(PathingEvent::GFI(D6Target::TwoPlus)),
            PositionOrEvent::Event(PathingEvent::Pickup(D6Target::ThreePlus)),
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
    fn setup_simple_pass(
        interceptor: bool,
        distance: i8,
    ) -> (GameState, Position, Position, Position) {
        assert!(distance > 1);
        let start_pos = Position::new((3, 3));
        let target_pos = Position::new((3 + distance, 3));
        let away_player = Position::new((3 + distance / 2, 3));
        let mut builder = GameStateBuilder::new();
        builder
            .add_home_player(start_pos)
            .add_home_player(target_pos)
            .add_ball_pos(start_pos);
        if interceptor {
            builder.add_away_player(away_player);
        }
        let mut state = builder.build();

        //make sure player does not move
        let id = state.get_player_id_at(start_pos).unwrap();
        state.get_mut_player_unsafe(id).moves = state.get_player_unsafe(id).total_movement_left();

        state.step_positional(PosAT::StartPass, start_pos);
        (state, start_pos, target_pos, away_player)
    }
    #[test]
    fn pass_selecting_interceptor() {
        let mut field = "".to_string();
        field += "H aah  \n";
        field += "  a   a\n";
        let start_pos = Position::new((2, 1));
        let target_pos = start_pos + Direction::right() * 4;
        let deflect_pos = start_pos + Direction::right() * 2;
        let mut state = GameStateBuilder::new().add_str(start_pos, &field).build();
        let id = state.get_player_id_at(start_pos).unwrap();
        state.get_mut_player_unsafe(id).moves = state.get_player_unsafe(id).total_movement_left();
        state.step_positional(PosAT::StartPass, start_pos);
        state.fixes.fix_d6(6); //Pass
        state.step_positional(PosAT::Pass, target_pos);
        state.fixes.fix_d6(6); //deflect
        state.fixes.fix_d6(6); //Catch
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
        state.step_positional(PosAT::SelectPosition, deflect_pos);
        let carrier_id = state.get_player_id_at(deflect_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(carrier_id));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_successful() {
        let (mut state, _, target_pos, _) = setup_simple_pass(false, 2);
        state.fixes.fix_d6(6); //Pass
        state.fixes.fix_d6(6); //Catch
        state.step_positional(PosAT::Pass, target_pos);
        let carrier_id = state.get_player_id_at(target_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(carrier_id));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Home);
    }
    #[test]
    fn pass_successful_intercepted() {
        let (mut state, _, target_pos, interceptor) = setup_simple_pass(true, 4);
        assert_eq!(interceptor, Position::new((5, 3)));
        let interceptor_id = state.get_player_id_at(interceptor).unwrap();
        state.fixes.fix_d6(6); //Pass
        state.fixes.fix_d6(6); //deflect
        state.fixes.fix_d6(6); //Catch
        state.step_positional(PosAT::Pass, target_pos);
        assert_eq!(state.ball, BallState::Carried(interceptor_id));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_successful_deflect_failed_catch() {
        let (mut state, _, target_pos, interceptor) = setup_simple_pass(true, 4);
        assert_eq!(interceptor, Position::new((5, 3)));
        state.fixes.fix_d6(6); //Pass
        state.fixes.fix_d6(6); //deflect
        state.fixes.fix_d6(1); //Catch
        state.step_positional(PosAT::Pass, target_pos);
        state.fixes.fix_d8_direction(Direction::up()); //Catch
        state.step_simple(SimpleAT::DontUseReroll);
        assert_eq!(
            state.ball,
            BallState::OnGround(interceptor + Direction::up())
        );
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_failed_deflect() {
        let (mut state, _, target_pos, interceptor) = setup_simple_pass(true, 4);
        assert_eq!(interceptor, Position::new((5, 3)));
        state.fixes.fix_d6(6); //Pass
        state.fixes.fix_d6(1); //deflect
        state.step_positional(PosAT::Pass, target_pos);
        state.fixes.fix_d6(6); //Catch
        state.step_simple(SimpleAT::DontUseReroll);
        let carrier_id = state.get_player_id_at(target_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(carrier_id));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Home);
    }
    #[test]
    fn pass_fumbled() {
        let (mut state, start_pos, target_pos, _) = setup_simple_pass(false, 2);
        let bounce_direction = Direction::up();
        state.fixes.fix_d6(1); //Pass fumbled
        state.fixes.fix_d8_direction(bounce_direction);
        state.step_positional(PosAT::Pass, target_pos);
        assert_eq!(
            state.ball,
            BallState::OnGround(start_pos + bounce_direction)
        );
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_inaccurate_turnover() {
        let (mut state, _, target_pos, _) = setup_simple_pass(false, 7);
        let bounce_direction = Direction::down();
        state.fixes.fix_d6(4); //Pass failed
        state.fixes.fix_d8_direction(bounce_direction); //Scatter
        state.fixes.fix_d8_direction(bounce_direction); //Scatter
        state.fixes.fix_d8_direction(bounce_direction); //Scatter
        state.fixes.fix_d8_direction(bounce_direction); //Bounce
        state.step_positional(PosAT::Pass, target_pos);
        assert_eq!(
            state.ball,
            BallState::OnGround(target_pos + 4 * bounce_direction)
        );
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_wildly_inaccurate_turnover() {
        let (mut state, start_pos, target_pos, _) = setup_simple_pass(false, 10);
        let deviate_direction = Direction::down();
        let bounce_direction = Direction::right();
        let passer_id = state.get_player_id_at(start_pos).unwrap();
        let deviate_distance = 3;
        state.fixes.fix_d6(2); //Pass failed
        state.fixes.fix_d8_direction(deviate_direction); //Scatter
        state.fixes.fix_d8_direction(bounce_direction); //Scatter
        state.fixes.fix_d6(deviate_distance as u8);
        state.step_positional(PosAT::Pass, target_pos);
        let expected_ball_pos = state.get_player_unsafe(passer_id).position
            + deviate_distance * deviate_direction
            + bounce_direction;
        assert_eq!(state.ball, BallState::OnGround(expected_ball_pos));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }
    #[test]
    fn pass_wildly_inaccurate_out_of_bounds() {
        let (mut state, _start_pos, target_pos, _) = setup_simple_pass(false, 10);
        let deviate_direction = Direction::up();
        let bounce_direction = Direction::right();
        let out_of_bounds_pos = Position::new((3, 1));
        let deviate_distance = 6;
        state.fixes.fix_d6(2); //Pass failed
        state.fixes.fix_d8_direction(deviate_direction);
        state.fixes.fix_d6(deviate_distance as u8);
        state.fixes.fix_d6(3); //throw in length
        state.fixes.fix_d6(2); //throw in length
        state.fixes.fix_d3(2); //throw in direction: down
        state.fixes.fix_d8_direction(bounce_direction);
        state.step_positional(PosAT::Pass, target_pos);
        let expected_ball_pos = out_of_bounds_pos + Direction::down() * (3 + 2) + bounce_direction;
        assert_eq!(state.ball, BallState::OnGround(expected_ball_pos));
        assert_eq!(state.get_active_teamtype().unwrap(), TeamType::Away);
    }

    #[test]
    fn pass_avoid_intercepts() {
        let mut field = "".to_string();
        field += "H a h \n";
        field += "  a   \n";
        field += "      \n";
        let first_pos = Position::new((2, 2));
        let mut state = GameStateBuilder::new().add_str(first_pos, &field).build();
        let start_pos = Position::new((2, 2));
        let target_pos = Position::new((6, 2));
        let passer_id = state.get_player_id_at(start_pos).unwrap();
        state.step_positional(PosAT::StartPass, start_pos);
        state.fixes.fix_d6(6); //Pass
        state.fixes.fix_d6(6); //Catch
        state.step_positional(PosAT::Pass, target_pos);
        let carrier_id = state.get_player_id_at(target_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(carrier_id));
        assert!(state.get_player_unsafe(passer_id).position.y > 3);
        assert!(state.get_player_unsafe(passer_id).position.x > 4);
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

    #[test]
    fn move_into_fail_gfi_into_stun_into_move_again() {
        let start_pos = Position::new((1, 1));
        let move_target = Position::new((8, 1));

        let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

        let id = state.get_player_id_at(start_pos).unwrap();

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));

        state.fixes.fix_d6(1); //fail first (2+)
        state.step_positional(PosAT::Move, move_target);

        state.fixes.fix_d6(6); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d6(1); //injury
        state.fixes.fix_d6(2); //injury

        state.step_simple(SimpleAT::DontUseReroll);
        assert!(state.get_player_unsafe(id).used);
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Stunned);
        assert_eq!(state.get_player_unsafe(id).position, move_target);

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.home_to_act());
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Stunned);
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.home_to_act());
        let player = state.get_player_unsafe(id);
        assert_eq!(player.status, PlayerStatus::Down);
        assert_eq!(player.moves_left(), player.stats.ma);
        assert_eq!(player.gfis_left(), 2);
        state.step_positional(PosAT::StartMove, move_target)
    }
}

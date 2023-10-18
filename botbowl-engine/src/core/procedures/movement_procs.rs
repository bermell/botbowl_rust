use crate::core::model::{Action, AvailableActions, PlayerID, PlayerStatus, ProcState, Procedure};
use crate::core::pathing::{
    event_ends_player_action, CustomIntoIter, NodeIterator, PathFinder, PathingEvent,
};
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::procedures::{ball_procs, block_procs};
use crate::core::table::*;

use crate::core::{dices::D6Target, gamestate::GameState};

use super::casualty_procs;

#[derive(Debug)]
pub struct GfiProc {
    target: D6Target,
    id: PlayerID,
}
impl GfiProc {
    fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<GfiProc>> {
        SimpleProcContainer::new(GfiProc { target, id })
    }
}
impl SimpleProc for GfiProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::SureFeet)
    }

    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Debug)]
pub struct StandUp {
    id: PlayerID,
}
impl StandUp {
    pub fn new(id: PlayerID) -> Box<StandUp> {
        Box::new(StandUp { id })
    }
}
impl Procedure for StandUp {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        debug_assert_eq!(
            game_state.get_player_unsafe(self.id).status,
            PlayerStatus::Down
        );
        game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Up;
        game_state.get_mut_player_unsafe(self.id).add_move(3);

        ProcState::Done
    }
}
#[derive(Debug)]
pub struct DodgeProc {
    target: D6Target,
    id: PlayerID,
}
impl DodgeProc {
    fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<DodgeProc>> {
        SimpleProcContainer::new(DodgeProc { target, id })
    }
}
impl SimpleProc for DodgeProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Dodge)
    }

    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
fn proc_from_roll(roll: PathingEvent, active_player: PlayerID) -> Box<dyn Procedure> {
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
    }
}

#[derive(Debug)]
enum MoveActionState {
    Init,
    ActivePath(NodeIterator),
    SelectPath,
}
#[derive(Debug)]
pub struct MoveAction {
    player_id: PlayerID,
    state: MoveActionState,
}
impl MoveAction {
    pub fn new(id: PlayerID) -> Box<MoveAction> {
        Box::new(MoveAction {
            state: MoveActionState::Init,
            player_id: id,
        })
    }
    fn continue_along_path(path: &mut NodeIterator, game_state: &mut GameState) -> ProcState {
        let player_id = game_state.info.active_player.unwrap();

        for next_event in path.by_ref() {
            match next_event {
                itertools::Either::Left(position) => {
                    game_state.move_player(player_id, position).unwrap();
                    game_state.get_mut_player_unsafe(player_id).add_move(1);
                }
                itertools::Either::Right(roll) => {
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if game_state.info.handle_td_by.is_some() || game_state.info.turnover {
            // game_state.get_mut_player_unsafe(self.player_id).used = true;
            return ProcState::Done;
        }

        match game_state.get_player(self.player_id) {
            Ok(player) if player.used => return ProcState::Done,
            Err(_) => return ProcState::Done, // player not on field anymore
            _ => (),
        }

        match (action, &mut self.state) {
            (None, MoveActionState::Init) => {
                self.state = MoveActionState::SelectPath;
                ProcState::NeedAction(self.available_actions(game_state))
            }
            (None, MoveActionState::ActivePath(path)) => {
                let proc_state = MoveAction::continue_along_path(path, game_state);
                if path.is_empty() {
                    self.state = MoveActionState::Init;
                }
                proc_state
            }
            (Some(Action::Positional(_, position)), MoveActionState::SelectPath) => {
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
            (Some(Action::Simple(SimpleAT::EndPlayerTurn)), _) => {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
                ProcState::Done
            }
            _ => unreachable!(),
        }
    }
}
#[cfg(test)]
mod tests {

    use crate::core::model::*;
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

        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::Dodge));

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
}

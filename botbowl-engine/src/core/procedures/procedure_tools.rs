use serde::{Deserialize, Serialize};

use crate::core::dices::{D6Target, RequestedRoll, RollResult};
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{Action, AvailableActions, PlayerID, ProcState, Procedure};
use crate::core::table::{SimpleAT, Skill};

use super::AnyProc;

#[allow(unused_variables)]
pub trait SimpleProc {
    fn d6_target(&self) -> D6Target; //called immidiately before
    fn reroll_skill(&self) -> Option<Skill>;
    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        Vec::new()
    }
    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc>;
    fn player_id(&self) -> PlayerID;
}
impl From<Vec<AnyProc>> for ProcState {
    fn from(procs: Vec<AnyProc>) -> Self {
        match procs.len() {
            0 => ProcState::Done,
            // 1 => ProcState::DoneNew(procs.pop().unwrap()),
            _ => ProcState::DoneNewProcs(procs),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollProcState {
    Init,
    RerollUsed,
    //WaitingForSkillReroll,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleProcContainer<T: SimpleProc + std::fmt::Debug> {
    proc: T,
    state: RollProcState,
}
impl<T: SimpleProc + std::fmt::Debug> SimpleProcContainer<T> {
    pub fn new(proc: T) -> Self {
        SimpleProcContainer {
            proc,
            state: RollProcState::Init,
        }
    }
    pub fn id(&self) -> PlayerID {
        self.proc.player_id()
    }
}

impl<T> Procedure for SimpleProcContainer<T>
where
    T: SimpleProc + std::fmt::Debug,
{
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::D6PassFail(self.proc.d6_target()));
            }
            ProcInput::Roll(RollResult::Pass) => {
                return ProcState::from(self.proc.apply_success(game_state))
            }
            ProcInput::Roll(RollResult::Fail) if self.state == RollProcState::RerollUsed => {
                return ProcState::from(self.proc.apply_failure(game_state))
            }
            ProcInput::Roll(RollResult::Fail) => (/*figure out if reroll is available below*/),
            ProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)) => {
                return ProcState::from(self.proc.apply_failure(game_state));
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseReroll)) => {
                game_state.get_active_team_mut().unwrap().use_reroll();
                self.state = RollProcState::RerollUsed;
                return ProcState::NeedRoll(RequestedRoll::D6PassFail(self.proc.d6_target()));
            }
            _ => panic!("Unexpected input: {:?}", input),
        };

        match self.proc.reroll_skill() {
            Some(skill) if game_state.get_player_unsafe(self.id()).can_use_skill(skill) => {
                game_state.get_mut_player_unsafe(self.id()).use_skill(skill);
                self.state = RollProcState::RerollUsed;
                return ProcState::NeedRoll(RequestedRoll::D6PassFail(self.proc.d6_target()));
            }
            _ => (),
        }

        if game_state
            .get_team_from_player(self.id())
            .unwrap()
            .can_use_reroll()
        {
            let team = game_state.get_player_unsafe(self.id()).stats.team;
            let mut aa = AvailableActions::new(team);
            aa.insert_simple(SimpleAT::UseReroll);
            aa.insert_simple(SimpleAT::DontUseReroll);
            return ProcState::NeedAction(aa);
        }
        ProcState::from(self.proc.apply_failure(game_state))
    }
}

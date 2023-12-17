use crate::core::dices::{D6Target, RollTarget};
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{Action, AvailableActions, PlayerID, ProcState, Procedure};
use crate::core::table::{SimpleAT, Skill};

#[allow(unused_variables)]
pub trait SimpleProc {
    fn d6_target(&self) -> D6Target; //called immidiately before
    fn reroll_skill(&self) -> Option<Skill>;
    fn apply_success(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        Vec::new()
    }
    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>>;
    fn player_id(&self) -> PlayerID;
}
impl From<Vec<Box<dyn Procedure>>> for ProcState {
    fn from(procs: Vec<Box<dyn Procedure>>) -> Self {
        match procs.len() {
            0 => ProcState::Done,
            // 1 => ProcState::DoneNew(procs.pop().unwrap()),
            _ => ProcState::DoneNewProcs(procs),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RollProcState {
    Init,
    RerollUsed,
    //WaitingForSkillReroll,
}
#[derive(Debug)]
pub struct SimpleProcContainer<T: SimpleProc + std::fmt::Debug> {
    proc: T,
    state: RollProcState,
}
impl<T: SimpleProc + std::fmt::Debug> SimpleProcContainer<T> {
    pub fn new(proc: T) -> Box<Self> {
        Box::new(SimpleProcContainer {
            proc,
            state: RollProcState::Init,
        })
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
        // if action is DON*T REROLL, apply failure, return true
        match input {
            ProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)) => {
                return ProcState::from(self.proc.apply_failure(game_state));
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseReroll)) => {
                game_state.get_active_team_mut().unwrap().use_reroll();
                self.state = RollProcState::RerollUsed;
            }
            _ => (),
        }

        loop {
            let roll = game_state.get_d6_roll();
            if self.proc.d6_target().is_success(roll) {
                return ProcState::from(self.proc.apply_success(game_state));
            }
            if self.state == RollProcState::RerollUsed {
                break;
            }
            match self.proc.reroll_skill() {
                Some(skill) if game_state.get_player_unsafe(self.id()).can_use_skill(skill) => {
                    game_state.get_mut_player_unsafe(self.id()).use_skill(skill);
                    self.state = RollProcState::RerollUsed;
                    continue;
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
            } else {
                break;
            }
        }
        ProcState::from(self.proc.apply_failure(game_state))
    }
}

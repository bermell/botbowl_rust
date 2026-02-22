use serde::{Deserialize, Serialize};

use crate::core::{dices::{D16, RequestedRoll, RollResult}, gamestate::GameState, model::{ProcInput, ProcState, Procedure}, procedures::AnyProc};



#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrayersToNuffle {}
impl PrayersToNuffle {
    pub fn new() -> AnyProc {
        AnyProc::PrayersToNuffle(PrayersToNuffle {})

    }
}
impl Procedure for PrayersToNuffle {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let prayers_to_nuffles_roll = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::D16);
            }
            ProcInput::Roll(RollResult::D16(prayers_to_nuffle_roll)) => prayers_to_nuffle_roll,
            _ => panic!("Unexpected input {:?}", input),
        };
        let mut procs: Vec<AnyProc> = Vec::new();
        match prayers_to_nuffles_roll {
            D16::One => {},
            D16::Two => {},
            D16::Three => {},
            D16::Four => {},
            D16::Five => {},
            D16::Six => {},
            D16::Seven => {},
            D16::Eight => {},
            D16::Nine => {},
            D16::Ten => {},
            D16::Eleven => {},
            D16::Twelve => {},
            D16::Thirteen => {}
        }
        ProcState::from(procs)
    }
}
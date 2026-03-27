use serde::{Deserialize, Serialize};

use crate::core::{
    dices::{D6, D6Target, D16, RequestedRoll, RollResult, RollTarget},
    gamestate::GameState,
    model::{BallState, PlayerID, ProcInput, ProcState, Procedure},
    procedures::{AnyProc, ball_procs, casualty_procs},
};


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
        let procs: Vec<AnyProc> = Vec::new();
        match prayers_to_nuffles_roll {
            D16::One => {
                game_state.info.trapdoors_active = true;
            },
            D16::Two => {

            },
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrapdoorCheck {
    id: PlayerID,
    target: D6Target,
    on_safe_procs: Vec<AnyProc>,
}

impl TrapdoorCheck {
    pub fn new(id: PlayerID, target: D6Target) -> AnyProc {
        AnyProc::TrapdoorCheck(TrapdoorCheck {
            id,
            target,
            on_safe_procs: Vec::new(),
        })
    }
}

impl Procedure for TrapdoorCheck {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        if !game_state.info.trapdoors_active {
            return ProcState::Done;
        }

        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(roll)) if self.target.is_success(roll) => {
                ProcState::from(std::mem::take(&mut self.on_safe_procs))
            }
            ProcInput::Roll(RollResult::D6(D6::One)) => {
                //FAIL
                let mut procs: Vec<AnyProc> = Vec::new();
                let player_position = match game_state.get_player(self.id) {
                    Ok(player_) => player_.position,
                    Err(_) => panic!("Player with id {:?} not found.", self.id),
                }; 

                if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
                    game_state.ball = BallState::InAir(player_position);
                    procs.push(ball_procs::Bounce::new());
                }
                procs.push(casualty_procs::Injury::new_crowd(self.id));
                ProcState::from(procs)
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

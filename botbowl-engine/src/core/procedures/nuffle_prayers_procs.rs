use serde::{Deserialize, Serialize};

use crate::core::{
    dices::{D16, D6, RequestedRoll, RollResult},
    gamestate::GameState,
    model::{BallState, PlayerID, Position, ProcInput, ProcState, Procedure},
    procedures::{ball_procs, casualty_procs, AnyProc},
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
    expected_pos: Position,
    on_safe_procs: Vec<AnyProc>,
}

impl TrapdoorCheck {
    pub fn new(id: PlayerID, expected_pos: Position) -> AnyProc {
        AnyProc::TrapdoorCheck(TrapdoorCheck {
            id,
            expected_pos,
            on_safe_procs: Vec::new(),
        })
    }

    pub fn new_with_on_safe(id: PlayerID, expected_pos: Position, on_safe_procs: Vec<AnyProc>) -> AnyProc {
        AnyProc::TrapdoorCheck(TrapdoorCheck {
            id,
            expected_pos,
            on_safe_procs,
        })
    }
}

impl Procedure for TrapdoorCheck {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(roll)) => {
                let valid_target = game_state.info.trapdoors_active
                    && self.expected_pos.is_trapdoor_position()
                    && matches!(
                        game_state.get_player(self.id),
                        Ok(player) if player.position == self.expected_pos
                    );

                if !valid_target {
                    return ProcState::Done;
                }

                if roll == D6::One {
                    let mut procs: Vec<AnyProc> = Vec::new();
                    if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id)
                    {
                        game_state.ball = BallState::InAir(self.expected_pos);
                        procs.push(ball_procs::Bounce::new());
                    }
                    procs.push(casualty_procs::Injury::new_crowd(self.id));
                    ProcState::from(procs)
                } else {
                    ProcState::from(std::mem::take(&mut self.on_safe_procs))
                }
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

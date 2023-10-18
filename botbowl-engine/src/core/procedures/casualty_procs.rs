use crate::core::dices::{RollTarget, Sum2D6Target};
use crate::core::gamestate::GameState;
use crate::core::model::{Action, DugoutPlace, PlayerStatus, ProcState, Procedure};
use crate::core::model::{BallState, PlayerID};
use crate::core::procedures::ball_procs;

#[derive(Debug)]
pub struct Armor {
    id: PlayerID,
    foul_target: Option<(PlayerID, Sum2D6Target)>,
}
impl Armor {
    pub fn new(id: PlayerID) -> Box<Armor> {
        Box::new(Armor {
            id,
            foul_target: None,
        })
    }
    pub fn new_foul(id: PlayerID, target: Sum2D6Target, fouler_id: PlayerID) -> Box<Armor> {
        Box::new(Armor {
            id,
            foul_target: Some((fouler_id, target)),
        })
    }
}
impl Procedure for Armor {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let roll1 = game_state.get_d6_roll();
        let roll2 = game_state.get_d6_roll();
        let roll = roll1 + roll2;
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();
        let mut injury_proc = Injury::new(self.id);

        let target = if let Some((fouler_id, foul_target)) = self.foul_target {
            if roll1 == roll2 {
                procs.push(Ejection::new(fouler_id));
            } else {
                injury_proc.fouler = Some(fouler_id);
            }
            foul_target
        } else {
            game_state.get_player_unsafe(self.id).armor_target()
        };

        if target.is_success(roll) {
            procs.push(injury_proc);
        }

        ProcState::from(procs)
    }
}

#[derive(Debug)]
pub struct Ejection {
    id: PlayerID,
}
impl Ejection {
    pub fn new(id: PlayerID) -> Box<Ejection> {
        Box::new(Ejection { id })
    }
}
impl Procedure for Ejection {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let position = game_state.get_player_unsafe(self.id).position;
        game_state
            .unfield_player(self.id, DugoutPlace::Ejected)
            .unwrap();

        if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
            game_state.ball = BallState::InAir(position);
            ProcState::DoneNew(ball_procs::Bounce::new())
        } else {
            ProcState::Done
        }
    }
}

#[derive(Debug)]
pub struct Injury {
    id: PlayerID,
    crowd: bool,
    fouler: Option<PlayerID>,
}
impl Injury {
    pub fn new(id: PlayerID) -> Box<Injury> {
        Box::new(Injury {
            id,
            crowd: false,
            fouler: None,
        })
    }

    pub fn new_crowd(id: PlayerID) -> Box<Injury> {
        Box::new(Injury {
            id,
            crowd: true,
            fouler: None,
        })
    }
}
impl Procedure for Injury {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let cas_target = Sum2D6Target::TenPlus;
        let ko_target = Sum2D6Target::EightPlus;
        let roll1 = game_state.get_d6_roll();
        let roll2 = game_state.get_d6_roll();
        let roll = roll1 + roll2;
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();

        if self.fouler.is_some() && roll1 == roll2 {
            procs.push(Ejection::new(self.fouler.unwrap()))
        }

        if cas_target.is_success(roll) {
            game_state
                .unfield_player(self.id, DugoutPlace::Injuried)
                .unwrap();
        } else if ko_target.is_success(roll) {
            game_state
                .unfield_player(self.id, DugoutPlace::KnockOut)
                .unwrap();
        } else if self.crowd {
            game_state
                .unfield_player(self.id, DugoutPlace::Reserves)
                .unwrap();
        } else {
            game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Stunned;
        }
        ProcState::from(procs)
    }
}

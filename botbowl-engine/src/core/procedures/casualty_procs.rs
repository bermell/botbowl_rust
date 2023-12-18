use crate::core::dices::{RequestedRoll, RollResult, Sum2D6Target};
use crate::core::gamestate::GameState;
use crate::core::model::{BallState, PlayerID};
use crate::core::model::{DugoutPlace, PlayerStatus, ProcState, Procedure};
use crate::core::model::{InjuryOutcome, ProcInput};
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
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();
        let mut injury_proc = Injury::new(self.id);
        let armor_broken = match input {
            ProcInput::Nothing if self.foul_target.is_some() => {
                return ProcState::NeedRoll(RequestedRoll::FoulArmor(self.foul_target.unwrap().1));
            }
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::Sum2D6PassFail(
                    game_state.get_player_unsafe(self.id).armor_target(),
                ));
            }
            ProcInput::Roll(RollResult::FoulArmor { broken, ejected }) => {
                if ejected {
                    procs.push(Ejection::new(self.foul_target.unwrap().0));
                } else if broken {
                    // injury proc shall also check of ejection
                    injury_proc.fouler = Some(self.foul_target.unwrap().0);
                }
                broken
            }
            ProcInput::Roll(RollResult::Pass) => true,
            ProcInput::Roll(RollResult::Fail) => false,
            _ => panic!("Unexpected input"),
        };

        if armor_broken {
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
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
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
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();

        let injury_outcome = match input {
            ProcInput::Nothing if self.fouler.is_some() => {
                return ProcState::NeedRoll(RequestedRoll::FoulInjury(
                    Sum2D6Target::EightPlus,
                    Sum2D6Target::TenPlus,
                ));
            }
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::Sum2D6ThreeOutcomes(
                    Sum2D6Target::EightPlus,
                    Sum2D6Target::TenPlus,
                ));
            }
            ProcInput::Roll(RollResult::FoulInjury { outcome, ejected }) => {
                if ejected {
                    procs.push(Ejection::new(self.fouler.unwrap()));
                }
                outcome
            }
            ProcInput::Roll(RollResult::Fail) => InjuryOutcome::Stunned,
            ProcInput::Roll(RollResult::MiddleOutcome) => InjuryOutcome::KO,
            ProcInput::Roll(RollResult::Pass) => InjuryOutcome::Casualty,

            _ => panic!("Unexpected input"),
        };

        let dugout_place = match injury_outcome {
            InjuryOutcome::Casualty => Some(DugoutPlace::Injuried),
            InjuryOutcome::KO => Some(DugoutPlace::KnockOut),
            InjuryOutcome::Stunned if self.crowd => Some(DugoutPlace::Reserves),
            InjuryOutcome::Stunned => {
                game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Stunned;
                None
            }
        };

        if let Some(place) = dugout_place {
            game_state.unfield_player(self.id, place).unwrap();
        }
        ProcState::from(procs)
    }
}

#[cfg(test)]
mod tests {

    use crate::core::dices::D8;
    use crate::core::model::*;
    use crate::core::table::*;
    use crate::core::{gamestate::GameStateBuilder, model::Position, table::PosAT};
    #[test]
    fn bounce_on_knockdown() -> Result<()> {
        let start_pos = Position::new((2, 2));
        let move_to = Position::new((3, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(Position::new((1, 1)))
            .add_ball_pos(start_pos)
            .build();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);
        let id = state.get_player_id_at(start_pos).unwrap();

        assert_eq!(state.ball, BallState::Carried(id));
        state.step_positional(PosAT::StartMove, start_pos);

        state.fixes.fix_d6(2);

        state.step_positional(PosAT::Move, move_to);

        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d8(d8_fix as u8);

        state.step_simple(SimpleAT::DontUseReroll);

        let player = state.get_player_unsafe(id);
        assert!(player.used);
        assert_eq!(state.ball, BallState::OnGround(move_to + direction));

        Ok(())
    }

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
}

use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6Target};
use crate::core::gamestate::GameState;
use crate::core::model::{Action, AvailableActions, BallState, PlayerID};
use crate::core::model::{DugoutPlace, PlayerStatus, ProcState, Procedure};
use crate::core::model::{InjuryOutcome, ProcInput};
use crate::core::procedures::ball_procs;
use crate::core::table::ArgueTheCall;
use crate::core::table::SimpleAT;

use super::AnyProc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Armor {
    id: PlayerID,
    foul_target: Option<(PlayerID, Sum2D6Target)>,
}
impl Armor {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Armor(Armor {
            id,
            foul_target: None,
        })
    }
    pub fn new_foul(id: PlayerID, target: Sum2D6Target, fouler_id: PlayerID) -> AnyProc {
        AnyProc::Armor(Armor {
            id,
            foul_target: Some((fouler_id, target)),
        })
    }
}
impl Procedure for Armor {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let mut procs: Vec<AnyProc> = Vec::new();
        let mut injury_proc = Injury::new_pure(self.id);
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
                    procs.push(Ejection::new_foul(self.foul_target.unwrap().0));
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
            procs.push(AnyProc::Injury(injury_proc));
        }

        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum EjectionState {
    Init,
    AwaitArgument,
    AwaitRoll,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ejection {
    id: PlayerID,
    foul: bool,
    state: EjectionState,
}
impl Ejection {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Ejection(Ejection {
            id,
            foul: false,
            state: EjectionState::Init,
        })
    }
    pub fn new_foul(id: PlayerID) -> AnyProc {
        AnyProc::Ejection(Ejection {
            id,
            foul: true,
            state: EjectionState::Init,
        })
    }
    fn eject_player(&self, game_state: &mut GameState) -> ProcState {
        let position = game_state.get_player_unsafe(self.id).position;
        let ret = if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id)
        {
            game_state.ball = BallState::InAir(position);
            ProcState::DoneNew(ball_procs::Bounce::new())
        } else {
            ProcState::Done
        };
        game_state
            .unfield_player(self.id, DugoutPlace::Ejected)
            .unwrap();
        ret
    }
    fn turnover_and_eject(&self, game_state: &mut GameState) -> ProcState {
        game_state.info.turnover = true;
        self.eject_player(game_state)
    }
}
impl Procedure for Ejection {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            EjectionState::Init if !self.foul => self.eject_player(game_state),
            EjectionState::Init => {
                if !game_state
                    .get_team_from_player(self.id)
                    .unwrap()
                    .can_argue_the_call()
                {
                    return self.turnover_and_eject(game_state);
                }

                self.state = EjectionState::AwaitArgument;
                let mut aa = AvailableActions::new(game_state.get_player_unsafe(self.id).stats.team);
                aa.insert_simple(SimpleAT::ArgueTheCall);
                aa.insert_simple(SimpleAT::DontArgueTheCall);
                ProcState::NeedAction(aa)
            }
            EjectionState::AwaitArgument => match input {
                ProcInput::Action(Action::Simple(SimpleAT::DontArgueTheCall)) => {
                    self.turnover_and_eject(game_state)
                }
                ProcInput::Action(Action::Simple(SimpleAT::ArgueTheCall)) => {
                    self.state = EjectionState::AwaitRoll;
                    ProcState::NeedRoll(RequestedRoll::D6)
                }
                _ => panic!("Unexpected input"),
            },
            EjectionState::AwaitRoll => match input {
                ProcInput::Roll(RollResult::D6(roll)) => match ArgueTheCall::from(roll) {
                    ArgueTheCall::YoureOutaHere => {
                        game_state.get_mut_team_from_player(self.id).unwrap().eject_coach();
                        self.turnover_and_eject(game_state)
                    }
                    ArgueTheCall::IDontCare => self.turnover_and_eject(game_state),
                    ArgueTheCall::WellWhenYouPutItLikeThat => {
                        game_state.info.turnover = true;
                        ProcState::Done
                    }
                },
                _ => panic!("Unexpected input"),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Injury {
    id: PlayerID,
    crowd: bool,
    fouler: Option<PlayerID>,
}
impl Injury {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Injury(Injury {
            id,
            crowd: false,
            fouler: None,
        })
    }

    pub fn new_crowd(id: PlayerID) -> AnyProc {
        AnyProc::Injury(Injury {
            id,
            crowd: true,
            fouler: None,
        })
    }
    pub fn new_pure(id: PlayerID) -> Injury {
        Injury {
            id,
            crowd: false,
            fouler: None,
        }
    }
}
impl Procedure for Injury {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let mut procs: Vec<AnyProc> = Vec::new();

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
                    procs.push(Ejection::new_foul(self.fouler.unwrap()));
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

    use crate::core::dices::{D6, D8, RequestedRoll, RollResult};
    use crate::core::model::*;
    use crate::core::procedures::AnyProc;
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

        state.fixes.fix_d6(2); //fail dodge

        state.step_positional(PosAT::Move, move_to);

        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d8(d8_fix as u8);

        state.step_simple(SimpleAT::DontUseReroll);

        assert!(state.away_to_act());
        let player = state.get_player_unsafe(id);
        assert!(!player.used);
        assert_eq!(state.ball, BallState::OnGround(move_to + direction));

        Ok(())
    }

    mod ejection_tests {
        use crate::core::procedures::Ejection;

        use super::*;

        fn new_foul_ejection(id: PlayerID) -> Ejection {
            match Ejection::new_foul(id) {
                AnyProc::Ejection(proc) => proc,
                _ => unreachable!(),
            }
        }

        #[test]
        fn well_when_you_put_it_like_that_argue_the_call_result() {
            //should result in player being allowed to stay on pitch but still turnover
            let start_pos = Position::new((5, 5));
            let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

            let id = state.get_player_id_at(start_pos).unwrap();
            let mut ejection = new_foul_ejection(id);

            let proc_state = ejection.step(&mut state, ProcInput::Nothing);
            assert!(matches!(
                proc_state,
                ProcState::NeedAction(aa)
                    if aa.is_legal_action(Action::Simple(SimpleAT::ArgueTheCall))
                        && aa.is_legal_action(Action::Simple(SimpleAT::DontArgueTheCall))
            ));

            let proc_state = ejection.step(
                &mut state,
                ProcInput::Action(Action::Simple(SimpleAT::ArgueTheCall)),
            );
            assert!(matches!(proc_state, ProcState::NeedRoll(RequestedRoll::D6)));

            let proc_state = ejection.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Six)));
            assert!(matches!(proc_state, ProcState::Done));
            assert_eq!(state.get_player_id_at(start_pos), Some(id));
            assert!(state.info.turnover);
            assert!(state.home.can_argue_the_call());
            assert!(state.get_dugout().next().is_none());

        }

        #[test]
        fn youre_outa_here_argue_the_call_result() {
            //should result in the player being ejected and the coach being ejected and then turnover
            let start_pos = Position::new((5, 5));
            let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

            let id = state.get_player_id_at(start_pos).unwrap();
            let mut ejection = new_foul_ejection(id);

            assert!(matches!(
                ejection.step(&mut state, ProcInput::Nothing),
                ProcState::NeedAction(_)
            ));
            assert!(matches!(
                ejection.step(
                    &mut state,
                    ProcInput::Action(Action::Simple(SimpleAT::ArgueTheCall)),
                ),
                ProcState::NeedRoll(RequestedRoll::D6)
            ));

            assert!(matches!(
                ejection.step(&mut state, ProcInput::Roll(RollResult::D6(D6::One))),
                ProcState::Done
            ));
            assert_eq!(state.get_player_id_at(start_pos), None);
            assert!(state.info.turnover);
            assert!(!state.home.can_argue_the_call());
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
        fn i_dont_care_argue_the_call_result() {
            // should result in the player being ejected and then turnover
            let start_pos = Position::new((5, 5));
            let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

            let id = state.get_player_id_at(start_pos).unwrap();
            let mut ejection = new_foul_ejection(id);

            assert!(matches!(
                ejection.step(&mut state, ProcInput::Nothing),
                ProcState::NeedAction(_)
            ));
            assert!(matches!(
                ejection.step(
                    &mut state,
                    ProcInput::Action(Action::Simple(SimpleAT::ArgueTheCall)),
                ),
                ProcState::NeedRoll(RequestedRoll::D6)
            ));

            assert!(matches!(
                ejection.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Two))),
                ProcState::Done
            ));
            assert_eq!(state.get_player_id_at(start_pos), None);
            assert!(state.info.turnover);
            assert!(state.home.can_argue_the_call());
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
            state.step_simple(SimpleAT::DontArgueTheCall);

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
            state.step_simple(SimpleAT::DontArgueTheCall);

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
        fn ejection_of_ball_carrier_starts_bounce() {
            let start_pos = Position::new((5, 5));
            let mut state = GameStateBuilder::new()
                .add_home_player(start_pos)
                .add_ball_pos(start_pos)
                .build();

            let id = state.get_player_id_at(start_pos).unwrap();
            assert_eq!(state.ball, BallState::Carried(id));

            let mut ejection = Ejection::new(id);
            let proc_state = ejection.step(&mut state, ProcInput::Nothing);

            assert!(matches!(proc_state, ProcState::DoneNew(AnyProc::Bounce(_))));
            assert_eq!(state.ball, BallState::InAir(start_pos));
            assert_eq!(state.get_player_id_at(start_pos), None);
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
}

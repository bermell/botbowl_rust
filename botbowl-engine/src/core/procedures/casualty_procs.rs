use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, RollTarget, Sum2D6Target};
use crate::core::gamestate::GameState;
use crate::core::model::{BallState, PlayerID};
use crate::core::model::{DugoutPlace, PlayerStatus, ProcState, Procedure};
use crate::core::model::{InjuryOutcome, ProcInput};
use crate::core::procedures::ball_procs;

use super::AnyProc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Armor {
    id: PlayerID,
    foul_target: Option<(PlayerID, Sum2D6Target)>,
    /// Mighty Blow's +1, for the armour roll or else the injury roll. The armour roll then has
    /// three outcomes — holds, breaks only with the +1, breaks on its own — so the +1 goes to
    /// armour exactly when that is what breaks it, and to injury otherwise.
    #[serde(default)]
    mighty_blow: bool,
}
impl Armor {
    pub fn new(id: PlayerID) -> AnyProc {
        Armor::new_block(id, false)
    }
    pub fn new_block(id: PlayerID, mighty_blow: bool) -> AnyProc {
        AnyProc::Armor(Armor {
            id,
            foul_target: None,
            mighty_blow,
        })
    }
    pub fn new_foul(id: PlayerID, target: Sum2D6Target, fouler_id: PlayerID) -> AnyProc {
        AnyProc::Armor(Armor {
            id,
            foul_target: Some((fouler_id, target)),
            mighty_blow: false,
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
            ProcInput::Nothing if self.mighty_blow => {
                let target = game_state.get_player_unsafe(self.id).armor_target();
                let mut with_mighty_blow = target;
                with_mighty_blow.add_modifer(1);
                return ProcState::NeedRoll(RequestedRoll::Sum2D6ThreeOutcomes(with_mighty_blow, target));
            }
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::Sum2D6PassFail(
                    game_state.get_player_unsafe(self.id).armor_target(),
                ));
            }
            // Broken only thanks to Mighty Blow: it is spent on the armour.
            ProcInput::Roll(RollResult::MiddleOutcome) => true,
            // Broken on its own: Mighty Blow, if any, goes to the injury roll.
            ProcInput::Roll(RollResult::Pass) if self.mighty_blow => {
                injury_proc.mighty_blow = true;
                true
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
            procs.push(AnyProc::Injury(injury_proc));
        }

        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Ejection {
    id: PlayerID,
}
impl Ejection {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Ejection(Ejection { id })
    }
}
impl Procedure for Ejection {
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
        let position = game_state.get_player_unsafe(self.id).position;
        let ret = if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
            game_state.set_ball(BallState::InAir(position));
            // Defer the turnover until the loose ball has finished bouncing,
            // mirroring `TurnoverIfPossessionLost`'s push-after-`Bounce` order.
            ProcState::DoneNewProcs(vec![EjectionTurnover::new(), ball_procs::Bounce::new()])
        } else {
            game_state.info.turnover = true;
            ProcState::Done
        };
        game_state.unfield_player(self.id, DugoutPlace::Ejected).unwrap();
        ret
    }
}

/// A foul that gets a player sent off is always a turnover, regardless of
/// where the ball ends up. Pushed after `Bounce` (see `Ejection::step`) so
/// it resolves once the loose ball has settled.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EjectionTurnover;
impl EjectionTurnover {
    pub fn new() -> AnyProc {
        AnyProc::EjectionTurnover(EjectionTurnover)
    }
}
impl Procedure for EjectionTurnover {
    fn step(&mut self, game_state: &mut GameState, _input: ProcInput) -> ProcState {
        game_state.info.turnover = true;
        ProcState::Done
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Injury {
    id: PlayerID,
    crowd: bool,
    fouler: Option<PlayerID>,
    /// +1 to the roll (Mighty Blow).
    #[serde(default)]
    mighty_blow: bool,
}
impl Injury {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Injury(Injury {
            id,
            crowd: false,
            fouler: None,
            mighty_blow: false,
        })
    }

    pub fn new_crowd(id: PlayerID) -> AnyProc {
        AnyProc::Injury(Injury {
            id,
            crowd: true,
            fouler: None,
            mighty_blow: false,
        })
    }
    pub fn new_pure(id: PlayerID) -> Injury {
        Injury {
            id,
            crowd: false,
            fouler: None,
            mighty_blow: false,
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
                let (mut ko, mut cas) = (Sum2D6Target::EightPlus, Sum2D6Target::TenPlus);
                if self.mighty_blow {
                    ko.add_modifer(1);
                    cas.add_modifer(1);
                }
                return ProcState::NeedRoll(RequestedRoll::Sum2D6ThreeOutcomes(ko, cas));
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

        state.fix_d6(2);

        state.step_positional(PosAT::Move, move_to);

        state.fix_d6(1); //armor
        state.fix_d6(5); //armor
        state.fix_d8(d8_fix as u8);

        state.step_simple(SimpleAT::DontUseReroll);

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

        state.fix_d6(5); //armor
        state.fix_d6(5); //armor
        state.fix_d6(2); //injury
        state.fix_d6(1); //injury

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

        state.fix_d6(5); //armor
        state.fix_d6(6); //armor
        state.fix_d6(2); //injury
        state.fix_d6(2); //injury

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
    fn ejection_without_ball_causes_turnover() {
        let start_pos = Position::new((5, 5));
        let foul_pos = start_pos + (2, 0);
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(foul_pos)
            .build();

        let victim_id = state.get_player_id_at(foul_pos).unwrap();
        state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;

        assert!(state.home_to_act());

        state.step_positional(PosAT::StartFoul, start_pos);

        state.fix_d6(1); //armor: doubles -> ejected, sum too low to break armor
        state.fix_d6(1); //armor

        state.step_positional(PosAT::Foul, foul_pos);

        assert!(
            matches!(
                state.get_dugout().next(),
                Some(DugoutPlayer {
                    place: DugoutPlace::Ejected,
                    ..
                })
            ),
            "fouler should have been ejected"
        );
        assert!(state.away_to_act(), "ejection must always cause a turnover");
    }

    #[test]
    fn ejection_of_ball_carrier_causes_turnover_after_bounce() {
        let start_pos = Position::new((5, 5));
        let foul_pos = start_pos + (2, 0);
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(foul_pos)
            .add_ball_pos(start_pos)
            .build();

        let fouler_id = state.get_player_id_at(start_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(fouler_id));

        let victim_id = state.get_player_id_at(foul_pos).unwrap();
        state.get_mut_player_unsafe(victim_id).status = PlayerStatus::Down;

        state.step_positional(PosAT::StartFoul, start_pos);

        state.fix_d6(1); //armor: doubles -> ejected, sum too low to break armor
        state.fix_d6(1); //armor
        state.fix_d8(D8::Four as u8); //bounce direction once the ball is dropped, away from the downed victim

        state.step_positional(PosAT::Foul, foul_pos);

        assert!(
            matches!(
                state.get_dugout().next(),
                Some(DugoutPlayer {
                    place: DugoutPlace::Ejected,
                    ..
                })
            ),
            "fouler should have been ejected"
        );
        assert!(
            matches!(state.ball, BallState::OnGround(_)),
            "ball should have bounced free of the ejected carrier, got {:?}",
            state.ball
        );
        assert!(
            state.away_to_act(),
            "ejecting the ball carrier must still cause a turnover"
        );
    }
}

use crate::core::model::ProcInput;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::core::dices::{D6Target, RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction, DugoutPlace, PlayerID,
    PlayerStatus, Position, ProcState, Procedure, TeamType, Weather,
};
use crate::core::procedures::ball_procs;
use crate::core::table::*;

use crate::core::gamestate::GameState;

use super::setup_procs::{
    build_rearrange_actions, step_rearrange_end_setup, step_rearrange_position,
    SetupRearrangeConfig, SetupRearrangeState,
};
use super::AnyProc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Kickoff {
    aim: Position,
}
impl Kickoff {
    pub fn new() -> AnyProc {
        AnyProc::Kickoff(Kickoff {
            aim: Position::new((0, 0)),
        })
    }
}
impl Procedure for Kickoff {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let (len_roll, dir_roll) = match input {
            ProcInput::Nothing => {
                let mut aa = AvailableActions::new(game_state.info.kicking_this_drive);
                aa.insert_simple(SimpleAT::KickoffAimMiddle);
                return ProcState::NeedAction(aa);
            }
            ProcInput::Action(Action::Simple(SimpleAT::KickoffAimMiddle)) => {
                self.aim = game_state.get_best_kickoff_aim_for(game_state.info.kicking_this_drive);
                return ProcState::NeedRoll(RequestedRoll::Deviate);
            }
            ProcInput::Roll(RollResult::Deviate(len_roll, dir_roll)) => (len_roll, dir_roll),
            _ => panic!("Unexpected input {:?}", input),
        };

        let ball_pos = self.aim + Direction::from(dir_roll) * (len_roll as Coord);
        game_state.ball = BallState::InAir(ball_pos);
        ProcState::DoneNewProcs(vec![LandKickoff::new(), KickoffTable::new()])
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KickoffTable {}
impl KickoffTable {
    pub fn new() -> AnyProc {
        AnyProc::KickoffTable(KickoffTable {})
    }
}
impl Procedure for KickoffTable {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let kickoff_roll = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::Sum2D6);
            }
            ProcInput::Roll(RollResult::Sum2D6(kickoff_roll)) => kickoff_roll,
            _ => panic!("Unexpected input {:?}", input),
        };
        let mut procs: Vec<AnyProc> = Vec::new();
        match kickoff_roll {
            Sum2D6::Two => {
                //get the ref
                game_state.home.bribes += 1;
                game_state.away.bribes += 1;
            }
            Sum2D6::Three => {
                //Timeout
                if game_state.info.home_turn <= 5 {
                    game_state.info.away_turn += 1;
                    game_state.info.home_turn += 1;
                } else {
                    game_state.info.away_turn -= 1;
                    game_state.info.home_turn -= 1;
                }
            }
            Sum2D6::Four => {
                procs.push(SolidDefence::new());
            }
            Sum2D6::Five => {
                procs.push(HighKick::new());
            }
            Sum2D6::Six => {
                // todo: Cheering fans implement. The rules for cheering fans are:
                // Both coaches roll a D6 and add the number ofcheerleaders on their Team Draft list.
                // The coach with the highest total may immediately roll once on the Prayers to Nuffle table.
                // In the case of a tie, neither coach rolls on the Prayers to Nuffle table.
                // Note that if you roll a result that is currently in effect, you must re-roll it.
                // However,if you roll a result that has been rolled previously but has since expired, there is no need to re-roll
            }
            Sum2D6::Seven => {
                procs.push(BrilliantCoaching::new());
            }
            Sum2D6::Eight => {
                procs.push(ChangingWeather::new());
            }
            Sum2D6::Nine => {
                procs.push(QuickSnap::new());
            }
            Sum2D6::Ten => {
                // todo: Blitz! implementation. The rules for Blitz! are:
                // D3+3 Open players on the kicking team may immediately activate to perform a Move action.
                // One may perform a Blitz action and one may perform a Throw Team-mate action.
                // If a player Falls Over or is Knocked Down, no further players can be activated and the Blitz ends immediately
            }
            Sum2D6::Eleven => {
                procs.push(OfficiousRef::new());
            }
            Sum2D6::Twelve => {
                procs.push(PitchInvasion::new());
            }
        }

        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChangingWeather {}
impl ChangingWeather {
    pub fn new() -> AnyProc {
        AnyProc::ChangingWeather(ChangingWeather {})
    }
}
impl Procedure for ChangingWeather {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::Sum2D6),
            ProcInput::Roll(RollResult::Sum2D6(roll)) => {
                game_state.info.weather = Weather::from(roll);
                let ball_pos = game_state.get_ball_position().unwrap();
                if game_state.info.weather == Weather::Nice && !ball_pos.is_out() {
                    ProcState::NeedRoll(RequestedRoll::D8)
                } else {
                    ProcState::Done
                }
            }
            ProcInput::Roll(RollResult::D8(d8)) => {
                game_state.ball =
                    BallState::InAir(game_state.get_ball_position().unwrap() + Direction::from(d8));
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HighKick {}
impl HighKick {
    pub fn new() -> AnyProc {
        AnyProc::HighKick(HighKick {})
    }
}
impl Procedure for HighKick {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let BallState::InAir(ball_position) = game_state.ball else {
            return ProcState::Done;
        };
        let receiving_team = other_team(game_state.info.kicking_this_drive);

        if ball_position.is_out()
            || !ball_position.is_on_team_side(receiving_team)
            || game_state.get_player_id_at(ball_position).is_some()
        {
            return ProcState::Done;
        }

        match input {
            ProcInput::Nothing => {
                let positions: Vec<Position> = game_state
                    .get_open_player_ids_on_pitch(receiving_team)
                    .into_iter()
                    .map(|id| game_state.get_player_unsafe(id).position)
                    .collect();

                if positions.is_empty() {
                    return ProcState::Done;
                }

                let mut aa = AvailableActions::new(receiving_team);
                aa.insert_positional(PosAT::SelectPosition, positions);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let player_id = game_state.get_player_id_at(pos).unwrap();
                game_state.move_player(player_id, ball_position).unwrap();
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum QuickSnapState {
    Init,
    SelectPlayers,
    SelectMoveTarget,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuickSnap {
    state: QuickSnapState,
    team: TeamType,
    max_selected: usize,
    selected_ids: Vec<PlayerID>,
    active_player: Option<PlayerID>,
}
impl QuickSnap {
    pub fn new() -> AnyProc {
        AnyProc::QuickSnap(QuickSnap {
            state: QuickSnapState::Init,
            team: TeamType::Home,
            max_selected: 0,
            selected_ids: Vec::new(),
            active_player: None,
        })
    }

    fn build_selection_actions(&self, game_state: &GameState) -> ProcState {
        let allow_new_selection = self.selected_ids.len() < self.max_selected;
        let positions = game_state
            .get_open_player_ids_on_pitch(self.team)
            .into_iter()
            .filter(|id| self.selected_ids.contains(id) || allow_new_selection)
            .map(|id| game_state.get_player_unsafe(id).position)
            .collect();

        let mut aa = AvailableActions::new(self.team);
        aa.insert_simple(SimpleAT::EndSetup);
        aa.insert_positional(PosAT::SelectPosition, positions);
        ProcState::NeedAction(aa)
    }

    fn build_player_selection_actions(&self, game_state: &GameState) -> ProcState {
        let positions = self
            .selected_ids
            .iter()
            .copied()
            .map(|id| game_state.get_player_unsafe(id).position)
            .collect();

        let mut aa = AvailableActions::new(self.team);
        aa.insert_positional(PosAT::SelectPosition, positions);
        ProcState::NeedAction(aa)
    }

    fn build_move_target_actions(&self, game_state: &GameState, player_id: PlayerID) -> ProcState {
        let player_pos = game_state.get_player_unsafe(player_id).position;
        let positions = Direction::all_directions_iter()
            .map(|dir| player_pos + *dir)
            .filter(|pos| !pos.is_out())
            .filter(|&pos| game_state.get_player_id_at(pos).is_none())
            .collect();

        let mut aa = AvailableActions::new(self.team);
        aa.insert_positional(PosAT::SelectPosition, positions);
        ProcState::NeedAction(aa)
    }
}
impl Procedure for QuickSnap {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            QuickSnapState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D3),
                ProcInput::Roll(RollResult::D3(roll)) => {
                    self.team = other_team(game_state.info.kicking_this_drive);
                    self.max_selected = roll + 3;
                    self.selected_ids.clear();
                    self.active_player = None;
                    self.state = QuickSnapState::SelectPlayers;
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            QuickSnapState::SelectPlayers => match input {
                ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => {
                    self.active_player = None;
                    if self.selected_ids.is_empty() {
                        ProcState::Done
                    } else {
                        self.state = QuickSnapState::SelectMoveTarget;
                        self.build_player_selection_actions(game_state)
                    }
                }
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                    let id = game_state.get_player_id_at(pos).unwrap();
                    assert_eq!(game_state.get_player_unsafe(id).stats.team, self.team);
                    assert_eq!(game_state.get_player_unsafe(id).status, PlayerStatus::Up);
                    assert_eq!(game_state.get_tz_on(id), 0);

                    if let Some(index) = self.selected_ids.iter().position(|&pid| pid == id) {
                        self.selected_ids.swap_remove(index);
                    } else if self.selected_ids.len() < self.max_selected {
                        self.selected_ids.push(id);
                    }
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            QuickSnapState::SelectMoveTarget => {
                let active_player = self.active_player;
                match input {
                    ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos))
                        if active_player.is_none() =>
                    {
                        let id = game_state.get_player_id_at(pos).unwrap();
                        assert!(self.selected_ids.contains(&id));
                        self.active_player = Some(id);
                        self.build_move_target_actions(game_state, id)
                    }
                    ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                        let id = active_player.unwrap();
                        let current_pos = game_state.get_player_unsafe(id).position;
                        assert_eq!(current_pos.distance_to(&pos), 1);
                        assert!(!pos.is_out());
                        assert!(game_state.get_player_id_at(pos).is_none());

                        game_state.move_player(id, pos).unwrap();
                        self.active_player = None;
                        self.selected_ids.retain(|&selected_id| selected_id != id);

                        if self.selected_ids.is_empty() {
                            ProcState::Done
                        } else {
                            self.build_player_selection_actions(game_state)
                        }
                    }
                    _ => panic!("Unexpected input {:?}", input),
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum SolidDefenceState {
    Init,
    SelectPlayers,
    RearrangePlayers,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SolidDefence {
    state: SolidDefenceState,
    team: TeamType,
    max_rearrange: usize,
    selected_fielded_ids: Vec<PlayerID>,
    rearrange_cfg: SetupRearrangeConfig,
    rearrange_state: SetupRearrangeState,
}
impl SolidDefence {
    pub fn new() -> AnyProc {
        AnyProc::SolidDefence(SolidDefence {
            state: SolidDefenceState::Init,
            team: TeamType::Away,
            max_rearrange: 0,
            selected_fielded_ids: Vec::new(),
            rearrange_cfg: SetupRearrangeConfig {
                team: TeamType::Away,
                target_on_pitch: 0,
                min_los: 0,
                end_requires_pending_empty: true,
            },
            rearrange_state: SetupRearrangeState {
                selected_fielded_player: None,
                pending_reserve_ids: Vec::new(),
                controlled_fielded_ids: HashSet::new(),
            },
        })
    }

    fn open_selectable_positions(&self, game_state: &GameState) -> Vec<Position> {
        let allow_new_selection = self.selected_fielded_ids.len() < self.max_rearrange;
        game_state
            .get_open_player_ids_on_pitch(self.team)
            .into_iter()
            .filter(|id| {
                self.selected_fielded_ids.contains(id)
                    || (allow_new_selection && !self.selected_fielded_ids.contains(id))
            })
            .map(|id| game_state.get_player_unsafe(id).position)
            .collect()
    }

    fn build_selection_actions(&self, game_state: &GameState) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        aa.insert_simple(SimpleAT::EndSetup);
        aa.insert_positional(
            PosAT::SelectPosition,
            self.open_selectable_positions(game_state),
        );
        ProcState::NeedAction(aa)
    }

    fn start_rearrange_phase(&mut self, game_state: &mut GameState) {
        let target_on_pitch = game_state.get_players_on_pitch_in_team(self.team).count();
        let selected_reserve_ids = self
            .selected_fielded_ids
            .iter()
            .copied()
            .map(|id| {
                game_state
                    .unfield_player_to_reserves_and_get_dugout_id(id)
                    .unwrap()
            })
            .collect();
        self.rearrange_cfg = SetupRearrangeConfig {
            team: self.team,
            target_on_pitch,
            min_los: 3.min(target_on_pitch),
            end_requires_pending_empty: true,
        };
        self.rearrange_state = SetupRearrangeState {
            selected_fielded_player: None,
            pending_reserve_ids: selected_reserve_ids,
            controlled_fielded_ids: HashSet::new(),
        };
        self.state = SolidDefenceState::RearrangePlayers;
    }
}
impl Procedure for SolidDefence {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            SolidDefenceState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D3),
                ProcInput::Roll(RollResult::D3(roll)) => {
                    self.team = game_state.info.kicking_this_drive;
                    self.max_rearrange = roll + 3;
                    self.selected_fielded_ids.clear();
                    self.rearrange_cfg = SetupRearrangeConfig {
                        team: self.team,
                        target_on_pitch: 0,
                        min_los: 0,
                        end_requires_pending_empty: true,
                    };
                    self.rearrange_state = SetupRearrangeState {
                        selected_fielded_player: None,
                        pending_reserve_ids: Vec::new(),
                        controlled_fielded_ids: HashSet::new(),
                    };
                    self.state = SolidDefenceState::SelectPlayers;
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            SolidDefenceState::SelectPlayers => match input {
                ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => {
                    self.start_rearrange_phase(game_state);
                    build_rearrange_actions(game_state, self.rearrange_cfg, &self.rearrange_state)
                }
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                    let id = game_state.get_player_id_at(pos).unwrap();
                    assert_eq!(game_state.get_player_unsafe(id).stats.team, self.team);
                    assert_eq!(game_state.get_player_unsafe(id).status, PlayerStatus::Up);
                    assert_eq!(game_state.get_tz_on(id), 0);
                    if let Some(index) = self.selected_fielded_ids.iter().position(|&pid| pid == id)
                    {
                        self.selected_fielded_ids.swap_remove(index);
                    } else if self.selected_fielded_ids.len() < self.max_rearrange {
                        self.selected_fielded_ids.push(id);
                    }
                    self.build_selection_actions(game_state)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            SolidDefenceState::RearrangePlayers => match input {
                ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => step_rearrange_end_setup(
                    game_state,
                    self.rearrange_cfg,
                    &mut self.rearrange_state,
                ),
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                    step_rearrange_position(
                        game_state,
                        self.rearrange_cfg,
                        &mut self.rearrange_state,
                        pos,
                    )
                }
                _ => panic!("Unexpected input {:?}", input),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LandKickoff {}
impl LandKickoff {
    pub fn new() -> AnyProc {
        AnyProc::LandKickoff(LandKickoff {})
    }
}
impl Procedure for LandKickoff {
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
        let BallState::InAir(ball_position) = game_state.ball else {
            unreachable!()
        };

        if ball_position.is_out()
            || !ball_position.is_on_team_side(other_team(game_state.info.kicking_this_drive))
        {
            return ProcState::DoneNew(ball_procs::Touchback::new());
        }

        match game_state.get_player_id_at(ball_position) {
            Some(id) => ProcState::DoneNew(ball_procs::Catch::new_with_kick_arg(
                id,
                game_state.get_catch_target(id).unwrap(),
                true,
            )),
            None => ProcState::DoneNew(ball_procs::Bounce::new_with_kick_arg(true)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum BrilliantCoachingState {
    Init,
    AwaitAwayRoll { home_total: i8 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrilliantCoaching {
    state: BrilliantCoachingState,
}
impl BrilliantCoaching {
    pub fn new() -> AnyProc {
        AnyProc::BrilliantCoaching(BrilliantCoaching {
            state: BrilliantCoachingState::Init,
        })
    }
}
impl Procedure for BrilliantCoaching {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            BrilliantCoachingState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
                ProcInput::Roll(RollResult::D6(roll)) => {
                    self.state = BrilliantCoachingState::AwaitAwayRoll {
                        home_total: game_state.home.brilliant_coaching_total(roll),
                    };
                    ProcState::NeedRoll(RequestedRoll::D6)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            BrilliantCoachingState::AwaitAwayRoll { home_total } => match input {
                ProcInput::Roll(RollResult::D6(roll)) => {
                    let away_total = game_state.away.brilliant_coaching_total(roll);
                    match home_total.cmp(&away_total) {
                        std::cmp::Ordering::Greater => {
                            game_state.home.grant_temporary_reroll();
                        }
                        std::cmp::Ordering::Less => {
                            game_state.away.grant_temporary_reroll();
                        }
                        std::cmp::Ordering::Equal => {}
                    }
                    ProcState::Done
                }
                _ => panic!("Unexpected input {:?}", input),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum OfficiousRefState {
    Init,
    AwaitAwayRoll { home_total: i8 },
    ResolveSelectedPlayers { pending: Vec<PlayerID> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OfficiousRef {
    state: OfficiousRefState,
}
impl OfficiousRef {
    fn new() -> AnyProc {
        AnyProc::OfficiousRef(OfficiousRef {
            state: OfficiousRefState::Init,
        })
    }
}
impl Procedure for OfficiousRef {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match &mut self.state {
            OfficiousRefState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
                ProcInput::Roll(RollResult::D6(roll)) => {
                    self.state = OfficiousRefState::AwaitAwayRoll {
                        home_total: game_state.home.officious_ref_total(roll),
                    };
                    ProcState::NeedRoll(RequestedRoll::D6)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            OfficiousRefState::AwaitAwayRoll { home_total } => match input {
                ProcInput::Roll(RollResult::D6(roll)) => {
                    let away_total = game_state.away.officious_ref_total(roll);
                    let mut pending = Vec::new();

                    match (*home_total).cmp(&away_total) {
                        std::cmp::Ordering::Less => {
                            pending.extend(
                                game_state.get_random_player_ids_on_pitch_in_team(TeamType::Home, 1),
                            );
                        }
                        std::cmp::Ordering::Greater => {
                            pending.extend(
                                game_state.get_random_player_ids_on_pitch_in_team(TeamType::Away, 1),
                            );
                        }
                        std::cmp::Ordering::Equal => {
                            pending.extend(
                                game_state.get_random_player_ids_on_pitch_in_team(TeamType::Home, 1),
                            );
                            pending.extend(
                                game_state.get_random_player_ids_on_pitch_in_team(TeamType::Away, 1),
                            );
                        }
                    }

                    if pending.is_empty() {
                        ProcState::Done
                    } else {
                        self.state = OfficiousRefState::ResolveSelectedPlayers { pending };
                        ProcState::NeedRoll(RequestedRoll::D6PassFail(D6Target::TwoPlus))
                    }
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            OfficiousRefState::ResolveSelectedPlayers { pending } => match input {
                ProcInput::Roll(RollResult::Pass) => {
                    let id = pending.remove(0);
                    game_state.get_mut_player_unsafe(id).status = PlayerStatus::Stunned;

                    if pending.is_empty() {
                        ProcState::Done
                    } else {
                        ProcState::NeedRoll(RequestedRoll::D6PassFail(D6Target::TwoPlus))
                    }
                }
                ProcInput::Roll(RollResult::Fail) => {
                    let id = pending.remove(0);
                    game_state.unfield_player(id, DugoutPlace::Ejected).unwrap();

                    if pending.is_empty() {
                        ProcState::Done
                    } else {
                        ProcState::NeedRoll(RequestedRoll::D6PassFail(D6Target::TwoPlus))
                    }
                }
                _ => panic!("Unexpected input {:?}", input),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum PitchInvasionState {
    Init,
    AwaitAwayRoll { home_total: i8 },
    RollVictimCounts {
        pending_teams: Vec<TeamType>,
        selected_ids: Vec<PlayerID>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PitchInvasion {
    state: PitchInvasionState,
}
impl PitchInvasion {
    fn new() -> AnyProc {
        AnyProc::PitchInvasion(PitchInvasion {
            state: PitchInvasionState::Init,
        })
    }
}
impl Procedure for PitchInvasion {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match &mut self.state {
            PitchInvasionState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
                ProcInput::Roll(RollResult::D6(roll)) => {
                    self.state = PitchInvasionState::AwaitAwayRoll {
                        home_total: game_state.home.pitch_invasion_total(roll),
                    };
                    ProcState::NeedRoll(RequestedRoll::D6)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            PitchInvasionState::AwaitAwayRoll { home_total } => match input {
                ProcInput::Roll(RollResult::D6(roll)) => {
                    let away_total = game_state.away.pitch_invasion_total(roll);
                    let pending_teams = match (*home_total).cmp(&away_total) {
                        std::cmp::Ordering::Less => vec![TeamType::Home],
                        std::cmp::Ordering::Greater => vec![TeamType::Away],
                        std::cmp::Ordering::Equal => vec![TeamType::Home, TeamType::Away],
                    };

                    self.state = PitchInvasionState::RollVictimCounts {
                        pending_teams,
                        selected_ids: Vec::new(),
                    };
                    ProcState::NeedRoll(RequestedRoll::D3)
                }
                _ => panic!("Unexpected input {:?}", input),
            },
            PitchInvasionState::RollVictimCounts {
                pending_teams,
                selected_ids,
            } => match input {
                ProcInput::Roll(RollResult::D3(roll)) => {
                    let team = pending_teams.remove(0);
                    selected_ids.extend(
                        game_state.get_random_player_ids_on_pitch_in_team(team, roll as usize),
                    );

                    if pending_teams.is_empty() {
                        for &id in selected_ids.iter() {
                            game_state.get_mut_player_unsafe(id).status = PlayerStatus::Stunned;
                        }
                        ProcState::Done
                    } else {
                        ProcState::NeedRoll(RequestedRoll::D3)
                    }
                }
                _ => panic!("Unexpected input {:?}", input),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::gamestate::{BuilderState, GameState, GameStateBuilder};
    use crate::core::model::*;
    use crate::core::table::*;
    use std::collections::HashSet;

    #[test]
    fn kickoff_get_the_ref() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);

        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert_eq!(state.home.bribes, 1);
        assert_eq!(state.away.bribes, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 0);

        // todo: this assertion should be a in more general test
        //assert_eq!(state.info.home_turn, 1, "home turn counter should be 1");
        assert!(state.home_to_act());
        assert_eq!(
            (state.info.home_turn, state.info.away_turn),
            (1, 0),
            "turn counter (home, away) is wrong!"
        );
    }

    #[test]
    fn kickoff_timeout_step_clock_forward() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(2);
        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert!(state.home_to_act());
        assert_eq!(state.info.home_turn, 2);
        assert_eq!(state.info.away_turn, 1);
    }

    #[test]
    fn kickoff_timeout_step_clock_backwards() {
        let mut state: GameState = GameStateBuilder::new()
            .set_state(BuilderState::Kickoff { turn: 7 })
            .build();
        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 6);
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(2);
        state.fixes.fix_d8_direction(Direction::up()); // bounce dice

        state.step_simple(SimpleAT::KickoffAimMiddle);
        assert!(state.home_to_act());

        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 5);
    }

    #[test]
    fn kickoff_changing_weather_lands_after_gust() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        state.fixes.fix_d6(1); // scatter length
        state.fixes.fix_d8_direction(Direction::down()); // scatter direction

        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4); // kickoff table: changing weather

        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4); // weather: nice

        state.fixes.fix_d8_direction(Direction::right()); // gust of wind
        state.fixes.fix_d8_direction(Direction::right()); // bounce

        state.step_simple(SimpleAT::KickoffAimMiddle);

        assert_eq!(state.ball, BallState::OnGround(Position::new((23, 8))));
    }

    mod kickoff_quick_snap {
        use super::*;

        fn quick_snap_state(home_players: &[Position], away_players: &[Position]) -> GameState {
            let mut state: GameState = GameStateBuilder::new()
                .set_state(BuilderState::Kickoff { turn: 1 })
                .build();
            state.clear_all_players().unwrap();

            for position in home_players {
                state
                    .add_new_player_to_field(PlayerStats::new_lineman(TeamType::Home), *position)
                    .unwrap();
            }
            for position in away_players {
                state
                    .add_new_player_to_field(PlayerStats::new_lineman(TeamType::Away), *position)
                    .unwrap();
            }

            state.fixes.fix_d8_direction(Direction::up()); // scatter direction
            state.fixes.fix_d6(5); // scatter length
            state.fixes.fix_d6(4);
            state.fixes.fix_d6(5); // kickoff table: quick snap
            state
        }

        fn advance_quick_snap_to_selection(state: &mut GameState, d3_roll: u8) {
            state.fixes.fix_d3(d3_roll);
            state.step_simple(SimpleAT::KickoffAimMiddle);
            assert_eq!(state.available_actions.team, Some(TeamType::Home));
        }

        #[test]
        fn can_deselect_and_replace_before_confirm() {
            let mut state = quick_snap_state(
                &[
                    Position::new((10, 5)),
                    Position::new((10, 7)),
                    Position::new((10, 9)),
                    Position::new((10, 11)),
                    Position::new((12, 5)),
                    Position::new((12, 7)),
                ],
                &[],
            );
            advance_quick_snap_to_selection(&mut state, 1); // D3+3 => 4

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(selectable.len() > 4);

            let selected_for_cap: Vec<Position> = selectable.iter().copied().take(4).collect();
            let replacement_pos = selectable[4];
            for pos in &selected_for_cap {
                state.step_positional(PosAT::SelectPosition, *pos);
            }

            assert!(
                !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, replacement_pos)),
                "unselected open players should be blocked when at cap"
            );

            let deselected_pos = selected_for_cap[0];
            state.step_positional(PosAT::SelectPosition, deselected_pos);
            assert!(
                state.is_legal_action(&Action::Positional(PosAT::SelectPosition, replacement_pos)),
                "after deselecting, another player can be selected"
            );

            state.step_positional(PosAT::SelectPosition, replacement_pos);
            assert!(
                !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, deselected_pos)),
                "after replacement, deselected player should stay out of the capped selection"
            );

            state.step_simple(SimpleAT::EndSetup);
            let move_stage_positions = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert_eq!(move_stage_positions.len(), 4);
            assert!(move_stage_positions.contains(&replacement_pos));
            assert!(!move_stage_positions.contains(&deselected_pos));
        }

        #[test]
        fn can_confirm_with_zero_selected() {
            let mut state =
                quick_snap_state(&[Position::new((10, 5)), Position::new((12, 5))], &[]);
            let positions_before: HashSet<Position> = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.position)
                .collect();

            advance_quick_snap_to_selection(&mut state, 1);

            assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)));
            state.fixes.fix_d8_direction(Direction::up()); // bounce after quick snap resolves
            state.step_simple(SimpleAT::EndSetup);

            let positions_after: HashSet<Position> = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.position)
                .collect();
            assert_eq!(positions_after, positions_before);
            assert_eq!(state.available_actions.team, Some(TeamType::Home));
        }

        #[test]
        fn should_be_possible_to_select_less_than_rolled_nr_of_players() {
            let mut state = quick_snap_state(
                &[
                    Position::new((10, 5)),
                    Position::new((10, 7)),
                    Position::new((10, 9)),
                    Position::new((10, 11)),
                    Position::new((12, 5)),
                ],
                &[],
            );
            advance_quick_snap_to_selection(&mut state, 1); // D3+3 => 4

            let selected: Vec<Position> = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .into_iter()
                .take(2)
                .collect();
            assert_eq!(selected.len(), 2);

            for pos in &selected {
                state.step_positional(PosAT::SelectPosition, *pos);
            }

            assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)));
            state.step_simple(SimpleAT::EndSetup);

            let move_stage_positions: HashSet<Position> = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .into_iter()
                .collect();
            assert_eq!(
                move_stage_positions,
                selected.into_iter().collect(),
                "only the confirmed subset should enter the move stage"
            );
        }

        #[test]
        fn not_open_players_should_not_be_selectable() {
            let marked_pos = Position::new((10, 10));
            let down_pos = Position::new((16, 10));
            let open_pos = Position::new((14, 10));
            let mut state = quick_snap_state(
                &[marked_pos, open_pos, down_pos],
                &[Position::new((11, 10))],
            );

            let down_id = state.get_player_id_at(down_pos).unwrap();
            state.get_mut_player_unsafe(down_id).status = PlayerStatus::Down;

            advance_quick_snap_to_selection(&mut state, 1);

            let selectable: HashSet<Position> = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .into_iter()
                .collect();
            let expected_open: HashSet<Position> = state
                .get_open_player_ids_on_pitch(TeamType::Home)
                .into_iter()
                .map(|id| state.get_player_unsafe(id).position)
                .collect();

            assert_eq!(selectable, expected_open);
            assert!(!selectable.contains(&marked_pos));
            assert!(!selectable.contains(&down_pos));
            assert!(selectable.contains(&open_pos));
        }

        #[test]
        fn quick_snap_moves_should_not_follow_setup_rules() {
            let cross_los_start = Position::new((14, 8));
            let north_wing_start = Position::new((15, 5));
            let mut state = quick_snap_state(
                &[
                    cross_los_start,
                    north_wing_start,
                    Position::new((16, 1)),
                    Position::new((16, 2)),
                ],
                &[],
            );
            advance_quick_snap_to_selection(&mut state, 1);

            state.step_positional(PosAT::SelectPosition, cross_los_start);
            state.step_positional(PosAT::SelectPosition, north_wing_start);
            state.step_simple(SimpleAT::EndSetup);

            state.step_positional(PosAT::SelectPosition, cross_los_start);
            let cross_los_target = Position::new((13, 8));
            assert!(
                state.is_legal_action(&Action::Positional(PosAT::SelectPosition, cross_los_target)),
                "quick snap should allow movement across the line of scrimmage"
            );
            state.step_positional(PosAT::SelectPosition, cross_los_target);

            state.step_positional(PosAT::SelectPosition, north_wing_start);
            let north_wing_target = Position::new((15, 4));
            assert!(
                state.is_legal_action(&Action::Positional(
                    PosAT::SelectPosition,
                    north_wing_target
                )),
                "quick snap should ignore setup wing caps"
            );
            state.fixes.fix_d8_direction(Direction::up()); // bounce after final quick snap move
            state.step_positional(PosAT::SelectPosition, north_wing_target);
        }

        #[test]
        fn selected_player_is_only_allowed_one_square_of_movement() {
            let start = Position::new((10, 10));
            let mut state = quick_snap_state(&[start], &[]);
            advance_quick_snap_to_selection(&mut state, 1);

            state.step_positional(PosAT::SelectPosition, start);
            state.step_simple(SimpleAT::EndSetup);

            state.step_positional(PosAT::SelectPosition, start);
            let legal_targets = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(!legal_targets.is_empty());
            assert!(legal_targets
                .iter()
                .all(|target| target.distance_to(&start) == 1));
            assert!(
                state.is_legal_action(&Action::Positional(
                    PosAT::SelectPosition,
                    Position::new((11, 10))
                )),
                "an adjacent square should be legal"
            );
            assert!(
                !state.is_legal_action(&Action::Positional(
                    PosAT::SelectPosition,
                    Position::new((12, 10))
                )),
                "quick snap should not allow moving more than one square"
            );
        }
    }

    mod kickoff_solid_defence {
        use super::*;

        #[test]
        fn cap_can_be_reached_and_no_marked_player_selectable() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);
            // Remove all opposing tackle zones so kicking team has many open players (> 4 cap).
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }
            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(selectable.len() > 4);

            let marked_positions: Vec<Position> = state
                .get_players_on_pitch_in_team(kicking_team)
                .filter(|player| player.status == PlayerStatus::Up)
                .filter(|player| state.get_tz_on(player.id) > 0)
                .map(|player| player.position)
                .collect();
            for pos in marked_positions {
                assert!(!selectable.contains(&pos));
            }

            let reserves_before_cap = state.get_reserve_count_for_team(kicking_team);
            let pitch_before_cap = state.get_players_on_pitch_in_team(kicking_team).count();
            let selected_for_cap: Vec<Position> = selectable.iter().copied().take(4).collect();
            let unselected_open: Vec<Position> = selectable.iter().copied().skip(4).collect();
            for pos in &selected_for_cap {
                state.step_positional(PosAT::SelectPosition, *pos);
            }
            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before_cap,
                "selection should not move players before explicit confirmation"
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before_cap
            );
            for pos in &selected_for_cap {
                assert!(
                    state.is_legal_action(&Action::Positional(PosAT::SelectPosition, *pos)),
                    "selected players should stay selectable so they can be deselected"
                );
            }
            for pos in unselected_open {
                assert!(
                    !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, pos)),
                    "cannot keep choosing more than the fixed number"
                );
            }
            assert!(
                state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "selection should only end on explicit confirmation"
            );

            state.step_simple(SimpleAT::EndSetup);

            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before_cap + 4,
                "only confirmed selections are moved to reserves"
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before_cap - 4
            );
        }

        #[test]
        fn can_deselect_and_replace_before_confirm() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);
            let reserves_before = state.get_reserve_count_for_team(kicking_team);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(selectable.len() > 4);

            let selected_for_cap: Vec<Position> = selectable.iter().copied().take(4).collect();
            let selected_for_cap_ids: Vec<PlayerID> = selected_for_cap
                .iter()
                .map(|pos| state.get_player_id_at(*pos).unwrap())
                .collect();
            let replacement_pos = selectable[4];
            for pos in &selected_for_cap {
                state.step_positional(PosAT::SelectPosition, *pos);
            }

            assert!(
                !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, replacement_pos)),
                "unselected open players should be blocked when at cap"
            );

            let deselected_pos = selected_for_cap[0];
            let deselected_id = selected_for_cap_ids[0];
            state.step_positional(PosAT::SelectPosition, deselected_pos);

            assert!(
                state.is_legal_action(&Action::Positional(PosAT::SelectPosition, replacement_pos)),
                "after deselecting, another player can be selected"
            );
            state.step_positional(PosAT::SelectPosition, replacement_pos);

            assert!(
                !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, deselected_pos)),
                "after replacement, deselected player should stay out of the capped selection"
            );
            let final_selected_positions = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert_eq!(
                final_selected_positions.len(),
                4,
                "at cap only currently selected players should remain selectable"
            );
            assert!(final_selected_positions.contains(&replacement_pos));
            assert!(!final_selected_positions.contains(&deselected_pos));

            state.step_simple(SimpleAT::EndSetup);

            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before + 4,
                "exactly four players should move to reserves after confirming"
            );
            for selected_pos in final_selected_positions {
                assert!(
                    state.get_player_id_at(selected_pos).is_none(),
                    "confirmed selected player should move to reserves"
                );
            }
            assert_eq!(state.get_player_id_at(deselected_pos), Some(deselected_id));
        }

        #[test]
        fn can_confirm_with_zero_selected() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let reserves_before = state.get_reserve_count_for_team(kicking_team);
            let pitch_before = state.get_players_on_pitch_in_team(kicking_team).count();

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)));
            state.step_simple(SimpleAT::EndSetup);

            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before,
                "no players should move when nothing was selected"
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before
            );
            assert!(
                state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "rearrange should be immediately confirmable with zero selected players"
            );
        }

        #[test]
        fn should_be_possible_to_select_less_than_rolled_nr_of_players() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);

            let selected: Vec<Position> = selectable.into_iter().take(2).collect();
            let reserves_before = state.get_reserve_count_for_team(kicking_team);
            let pitch_before = state.get_players_on_pitch_in_team(kicking_team).count();
            for pos in selected {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before,
                "selection should not move players before explicit confirmation"
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before
            );
            assert!(
                state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "should be possible to choose fewer players than the fixed number"
            );

            state.step_simple(SimpleAT::EndSetup);

            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before + 2,
                "all selected players must be in reserves before any placement"
            );

            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before - 2
            );
        }

        #[test]
        fn downed_player_unselectable() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let down_id = state
                .get_players_on_pitch_in_team(kicking_team)
                .filter(|player| player.status == PlayerStatus::Up)
                .filter(|player| state.get_tz_on(player.id) == 0)
                .map(|player| player.id)
                .next()
                .unwrap();
            let down_pos = state.get_player_unsafe(down_id).position;
            state.get_mut_player_unsafe(down_id).status = PlayerStatus::Down;

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(
                !selectable.contains(&down_pos),
                "open player must be standing and not marked"
            );
        }

        #[test]
        fn occupied_positions_are_excluded_from_legal_setup_positions() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);

            let selected: Vec<Position> = selectable.into_iter().take(4).collect();
            for pos in selected {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            state.step_simple(SimpleAT::EndSetup);

            let anchored_positions: HashSet<Position> = state
                .get_players_on_pitch_in_team(kicking_team)
                .map(|player| player.position)
                .collect();
            let legal_placements = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            for pos in &anchored_positions {
                assert!(
                    !legal_placements.contains(pos),
                    "anchored players should make their occupied squares illegal setup positions"
                );
            }
        }

        #[test]
        fn selected_occupied_square_is_legal_during_rearrange_placement() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selectable = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            let selected: Vec<Position> = selectable.into_iter().take(2).collect();
            assert_eq!(selected.len(), 2);
            for pos in selected {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            state.step_simple(SimpleAT::EndSetup);

            let first_placements = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            let first_target = *first_placements
                .iter()
                .find(|&&pos| state.get_player_id_at(pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, first_target);

            let second_placements = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition);
            assert!(
                second_placements.contains(&first_target),
                "already placed selected player square should remain legal for swapping"
            );

            let anchored_pos = state
                .get_players_on_pitch_in_team(kicking_team)
                .map(|player| player.position)
                .find(|&pos| pos != first_target)
                .unwrap();
            assert!(
                !state.is_legal_action(&Action::Positional(PosAT::SelectPosition, anchored_pos)),
                "occupied squares of non-selected players must stay illegal"
            );
        }

        #[test]
        fn reserve_placement_can_swap_with_controlled_player() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let selected: Vec<Position> = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .into_iter()
                .take(2)
                .collect();
            assert_eq!(selected.len(), 2);
            for pos in selected {
                state.step_positional(PosAT::SelectPosition, pos);
            }
            state.step_simple(SimpleAT::EndSetup);

            let first_target = *state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .iter()
                .find(|&&pos| state.get_player_id_at(pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, first_target);

            let reserves_before_swap = state.get_reserve_count_for_team(kicking_team);
            assert!(state.is_legal_action(&Action::Positional(PosAT::SelectPosition, first_target)));
            state.step_positional(PosAT::SelectPosition, first_target);
            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before_swap,
                "swapping placed selected players should keep reserves count unchanged"
            );
            assert!(
                !state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "cannot end rearrange while one selected player is still in reserves"
            );

            let last_target = *state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .iter()
                .find(|&&pos| state.get_player_id_at(pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, last_target);
            assert!(
                state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)),
                "after placing the final selected player, setup should be endable"
            );
        }

        #[test]
        fn happy_path() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let kicking_team = state.info.kicking_this_drive;
            let receiving_team = other_team(kicking_team);

            // Make many kicking players open/selectable.
            let receiving_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(receiving_team)
                .map(|player| player.id)
                .collect();
            for id in receiving_ids {
                state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
            }

            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(3);
            state.fixes.fix_d3(1); // D3+3 => 4
            state.step_simple(SimpleAT::KickoffAimMiddle);

            let reserves_before = state.get_reserve_count_for_team(kicking_team);
            let pitch_before = state.get_players_on_pitch_in_team(kicking_team).count();

            let selected_positions: Vec<Position> = state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .into_iter()
                .take(2)
                .collect();
            assert_eq!(selected_positions.len(), 2);
            let selected_ids: HashSet<PlayerID> = selected_positions
                .iter()
                .map(|&pos| state.get_player_id_at(pos).unwrap())
                .collect();
            for pos in &selected_positions {
                state.step_positional(PosAT::SelectPosition, *pos);
            }

            state.step_simple(SimpleAT::EndSetup);
            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before + 2
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before - 2
            );
            for pos in &selected_positions {
                assert_eq!(state.get_player_id_at(*pos), None);
            }

            let first_target = *state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .iter()
                .find(|&&pos| state.get_player_id_at(pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, first_target);

            let second_target = *state
                .available_actions
                .get_positions_for_action(PosAT::SelectPosition)
                .iter()
                .find(|&&pos| state.get_player_id_at(pos).is_none())
                .unwrap();
            state.step_positional(PosAT::SelectPosition, second_target);

            assert_eq!(
                state.get_reserve_count_for_team(kicking_team),
                reserves_before
            );
            assert_eq!(
                state.get_players_on_pitch_in_team(kicking_team).count(),
                pitch_before
            );
            let placed_ids: HashSet<PlayerID> = [first_target, second_target]
                .into_iter()
                .map(|pos| state.get_player_id_at(pos).unwrap())
                .collect();
            assert_eq!(placed_ids, selected_ids);

            assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndSetup)));
            state.fixes.fix_d8_direction(Direction::up()); // ball bounce after Solid Defence resolves
            state.step_simple(SimpleAT::EndSetup);
        }
    }

    #[test]
    fn kickoff_high_kick() {
        let mut state: GameState = GameStateBuilder::new_at_kickoff();
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(4);

        state.step_simple(SimpleAT::KickoffAimMiddle);

        let ball_pos = state.get_ball_position().unwrap();
        assert!(matches!(state.ball, BallState::InAir(_)));

        assert!(state.home_to_act());
        let receiving_team = other_team(state.info.kicking_this_drive);
        let legal_positions: Vec<Position> = state
            .get_players_on_pitch_in_team(receiving_team)
            .filter(|p| p.status == PlayerStatus::Up)
            .filter(|p| state.get_tz_on(p.id) == 0)
            .map(|p| p.position)
            .collect();
        assert!(!legal_positions.is_empty());
        for pos in &legal_positions {
            let action = Action::Positional(PosAT::SelectPosition, *pos);
            assert!(state.available_actions.is_legal_action(action));
        }

        let catcher_start_pos = legal_positions[0];
        let catcher_id = state.get_player_id_at(catcher_start_pos).unwrap();

        state.fixes.fix_d6(6); // fix the roll for the catch
        state.step_positional(PosAT::SelectPosition, legal_positions[0]);

        assert_eq!(state.get_player_id_at(ball_pos).unwrap(), catcher_id);
        assert_eq!(state.get_player_id_at(catcher_start_pos), None);

        match state.ball {
            BallState::Carried(id) => {
                assert_eq!(id, catcher_id);
            }
            _ => panic!("ball should be carried"),
        }

        assert!(state.home_to_act());
    }
    //
    // #[test]
    // fn kickoff_cheering_fans() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(5);
    //     // TODO: Implement prayers to nuffle...
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    // }
    //

    mod kickoff_brilliant_coaching {
        use super::*;

        fn fix_brilliant_coaching_kickoff_rolls(state: &mut GameState) {
            state.fixes.fix_d8_direction(Direction::up()); // scatter direction
            state.fixes.fix_d6(5); // scatter length
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(6); // kickoff table: brilliant coaching
        }

        #[test]
        fn kickoff_brilliant_coaching() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            state.home.set_assistant_coaches(3);
            state.away.set_assistant_coaches(0);
            fix_brilliant_coaching_kickoff_rolls(&mut state);
            state.fixes.fix_d6(2); // home roll
            state.fixes.fix_d6(4); // away roll
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(state.home.temporary_rerolls, 1);
            assert_eq!(state.away.temporary_rerolls, 0);
        }

        #[test]
        fn brilliant_coaching_teams_roll_equal_no_team_should_get_reroll() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            state.home.set_assistant_coaches(2);
            state.away.set_assistant_coaches(0);
            fix_brilliant_coaching_kickoff_rolls(&mut state);
            state.fixes.fix_d6(3); // home roll => 5 total
            state.fixes.fix_d6(5); // away roll => 5 total
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(state.home.temporary_rerolls, 0);
            assert_eq!(state.away.temporary_rerolls, 0);
        }

        #[test]
        fn brilliant_coaching_reroll_not_used_at_end_of_half_should_be_lost() {
            let mut state: GameState = GameStateBuilder::new()
                .set_state(BuilderState::Kickoff { turn: 8 })
                .build();
            state.away.set_assistant_coaches(2);
            fix_brilliant_coaching_kickoff_rolls(&mut state);
            state.fixes.fix_d6(3); // home roll
            state.fixes.fix_d6(4); // away roll => away wins with coaches
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(state.home.temporary_rerolls, 0);
            assert_eq!(state.away.temporary_rerolls, 1);

            assert!(state.home_to_act());
            state.step_simple(SimpleAT::EndTurn);
            assert!(state.away_to_act());
            state.step_simple(SimpleAT::EndTurn);

            assert_eq!(state.info.half, 2);
            assert_eq!(state.home.temporary_rerolls, 0);
            assert_eq!(state.away.temporary_rerolls, 0);
        }

        #[test]
        fn brilliant_coaching_applies_minus_one_for_ejected_coach() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            state.home.set_assistant_coaches(2);
            state.away.set_assistant_coaches(2);
            state.home.eject_coach();
            fix_brilliant_coaching_kickoff_rolls(&mut state);
            state.fixes.fix_d6(4); // home total => 5 after modifier
            state.fixes.fix_d6(5); // away total => 7
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(state.home.temporary_rerolls, 0);
            assert_eq!(state.away.temporary_rerolls, 1);
        }
    }

    mod kickoff_officious_ref {
        use super::*;

        fn fix_officious_ref_kickoff_rolls(state: &mut GameState) {
            state.fixes.fix_d8_direction(Direction::up()); // scatter direction
            state.fixes.fix_d6(5); // scatter length
            state.fixes.fix_d6(5);
            state.fixes.fix_d6(6); // kickoff table: officious ref
        }

        #[test]
        fn both_coaches_randomly_select_player_in_case_of_a_tie() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let selected_home_id = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.id)
                .next()
                .unwrap();
            let selected_away_id = state
                .get_players_on_pitch_in_team(TeamType::Away)
                .map(|player| player.id)
                .next()
                .unwrap();

            fix_officious_ref_kickoff_rolls(&mut state);
            state.fixes.fix_d6(3); // home coach total => 4
            state.fixes.fix_d6(3); // away coach total => 4
            state.fixes.fix_d16(1); // home selected player
            state.fixes.fix_d16(1); // away selected player
            state.fixes.fix_d6(2); // home selected player stays
            state.fixes.fix_d6(2); // away selected player stays
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(
                state.get_player_unsafe(selected_home_id).status,
                PlayerStatus::Down
            );
            assert_eq!(
                state.get_player_unsafe(selected_away_id).status,
                PlayerStatus::Stunned
            );
            assert_eq!(
                state
                    .get_players_on_pitch_in_team(TeamType::Home)
                    .filter(|player| player.status == PlayerStatus::Down)
                    .count(),
                1
            );
            assert_eq!(
                state
                    .get_players_on_pitch_in_team(TeamType::Away)
                    .filter(|player| player.status == PlayerStatus::Stunned)
                    .count(),
                1
            );
            assert!(!state
                .get_dugout()
                .any(|player| player.place == DugoutPlace::Ejected));
        }

        #[test]
        fn randomly_selected_player_must_be_on_the_pitch() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let home_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.id)
                .collect();
            let lone_home_id = home_ids[0];
            for id in home_ids.iter().skip(1).copied() {
                state.unfield_player(id, DugoutPlace::Reserves).unwrap();
            }

            fix_officious_ref_kickoff_rolls(&mut state);
            state.fixes.fix_d6(1); // home coach total => 2
            state.fixes.fix_d6(6); // away coach total => 7
            state.fixes.fix_d16(1); // only player on pitch must be selected
            state.fixes.fix_d6(2); // selected player stays
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(
                state.get_players_on_pitch_in_team(TeamType::Home).count(),
                1,
                "no reserve player should be pulled onto the pitch"
            );
            assert_eq!(
                state.get_player_unsafe(lone_home_id).status,
                PlayerStatus::Down
            );
            assert!(!state
                .get_dugout()
                .any(|player| player.place == DugoutPlace::Ejected));
        }

        #[test]
        fn rolls_two_plus_should_place_prone_and_stun_selected_player() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            state.home.set_fan_factor(3);
            state.away.set_fan_factor(1);
            let selected_away = state
                .get_players_on_pitch_in_team(TeamType::Away)
                .next()
                .unwrap()
                .clone();

            fix_officious_ref_kickoff_rolls(&mut state);
            state.fixes.fix_d6(2); // home raw roll lower, but fan factor wins
            state.fixes.fix_d6(3); // away raw roll higher, but lower total after fan factor
            state.fixes.fix_d16(1); // away selected player
            state.fixes.fix_d6(2); // selected player stays
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(
                state.get_player_id_at(selected_away.position),
                Some(selected_away.id)
            );
            assert_eq!(
                state.get_player_unsafe(selected_away.id).status,
                PlayerStatus::Stunned
            );
            assert!(!state
                .get_dugout()
                .any(|player| player.place == DugoutPlace::Ejected));
        }

        #[test]
        fn rolls_one_should_send_selected_player_off() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let selected_home = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .next()
                .unwrap()
                .clone();

            fix_officious_ref_kickoff_rolls(&mut state);
            state.fixes.fix_d6(1); // home coach total => 2
            state.fixes.fix_d6(6); // away coach total => 7
            state.fixes.fix_d16(1); // home selected player
            state.fixes.fix_d6(1); // selected player is sent off
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(state.get_player_id_at(selected_home.position), None);
            assert!(
                state
                    .get_players_on_pitch()
                    .all(|player| player.id != selected_home.id),
                "selected player should no longer be fielded"
            );
            assert!(state.get_dugout().any(|player| {
                player.place == DugoutPlace::Ejected && player.stats.team == TeamType::Home
            }));
        }
    }

    mod kickoff_pitch_invasion {
        use super::*;

        fn fix_pitch_invasion_kickoff_rolls(state: &mut GameState) {
            state.fixes.fix_d8_direction(Direction::up()); // scatter direction
            state.fixes.fix_d6(5); // scatter length
            state.fixes.fix_d6(6);
            state.fixes.fix_d6(6); // kickoff table: pitch invasion
        }

        #[test]
        fn both_coaches_randomly_select_players_in_case_of_a_tie() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let home_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.id)
                .collect();
            let away_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(TeamType::Away)
                .map(|player| player.id)
                .collect();
            let selected_home = vec![home_ids[0], *home_ids.last().unwrap()];
            let selected_away = vec![away_ids[0], *away_ids.last().unwrap()];

            fix_pitch_invasion_kickoff_rolls(&mut state);
            state.fixes.fix_d6(3); // home coach total => 4
            state.fixes.fix_d6(3); // away coach total => 4
            state.fixes.fix_d3(2); // home selected players
            state.fixes.fix_d16(1);
            state.fixes.fix_d16(1);
            state.fixes.fix_d3(2); // away selected players
            state.fixes.fix_d16(1);
            state.fixes.fix_d16(1);
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            for id in selected_home {
                assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
            }
            for id in selected_away {
                assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Stunned);
            }
            assert_eq!(
                state
                    .get_players_on_pitch_in_team(TeamType::Home)
                    .filter(|player| player.status == PlayerStatus::Down)
                    .count(),
                2
            );
            assert_eq!(
                state
                    .get_players_on_pitch_in_team(TeamType::Away)
                    .filter(|player| player.status == PlayerStatus::Stunned)
                    .count(),
                2
            );
        }

        #[test]
        fn randomly_selected_player_must_be_on_the_pitch() {
            let mut state: GameState = GameStateBuilder::new_at_kickoff();
            let home_ids: Vec<PlayerID> = state
                .get_players_on_pitch_in_team(TeamType::Home)
                .map(|player| player.id)
                .collect();
            let lone_home_id = home_ids[0];
            for id in home_ids.iter().skip(1).copied() {
                state.unfield_player(id, DugoutPlace::Reserves).unwrap();
            }

            fix_pitch_invasion_kickoff_rolls(&mut state);
            state.fixes.fix_d6(1); // home coach total => 2
            state.fixes.fix_d6(6); // away coach total => 7
            state.fixes.fix_d3(3); // capped to lone player on pitch
            state.fixes.fix_d16(1); // only player on pitch must be selected
            state.fixes.fix_d8_direction(Direction::up()); // bounce

            state.step_simple(SimpleAT::KickoffAimMiddle);

            assert_eq!(
                state.get_players_on_pitch_in_team(TeamType::Home).count(),
                1,
                "no reserve player should be pulled onto the pitch"
            );
            assert_eq!(
                state.get_player_unsafe(lone_home_id).status,
                PlayerStatus::Down
            );
        }
    }
    // #[test]
    // fn kickoff_changing_weather() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(1);
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    // }
    // #[test]
    // fn kickoff_after_td() {
    //     let start_pos = Position::new((2, 5));
    //     let mut state = GameStateBuilder::new()
    //         .add_home_player(start_pos)
    //         .add_ball_pos(start_pos)
    //         .build();
    //
    //     state.step_positional(PosAT::StartMove, start_pos);
    //     state.step_positional(PosAT::Move, Position::new((1, 5)));
    //
    //     assert_eq!(state.home.score, 1);
    //     assert_eq!(state.away.score, 0);
    //
    //     assert!(state.home_to_act());
    //     state.step_simple(SimpleAT::SetupLine);
    //     state.step_simple(SimpleAT::EndSetup);
    //
    //     assert!(state.away_to_act());
    //     state.step_simple(SimpleAT::SetupLine);
    //     state.step_simple(SimpleAT::EndSetup);
    //}
}

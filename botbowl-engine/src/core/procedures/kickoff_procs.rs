use crate::core::model::ProcInput;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::core::dices::{RequestedRoll, RollResult, Sum2D6};
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction, PlayerID, PlayerStatus,
    Position, ProcState, Procedure, TeamType, Weather,
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
                //Cheering fans
            }
            Sum2D6::Seven => {
                //Brilliant coaching
            }
            Sum2D6::Eight => {
                procs.push(ChangingWeather::new());
            }
            Sum2D6::Nine => {
                //Quick snap
            }
            Sum2D6::Ten => {
                //Blitz!
            }
            Sum2D6::Eleven => {
                //Officious ref
            }
            Sum2D6::Twelve => {
                //Pitch invasion
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

    fn open_player_ids(&self, game_state: &GameState) -> Vec<PlayerID> {
        game_state.get_open_player_ids_on_pitch(self.team)
    }

    fn open_selectable_positions(&self, game_state: &GameState) -> Vec<Position> {
        let allow_new_selection = self.selected_fielded_ids.len() < self.max_rearrange;
        self.open_player_ids(game_state)
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

    fn max_rearrange_from_d6(roll: u8) -> usize {
        usize::from((roll + 1) / 2 + 3)
    }
}
impl Procedure for SolidDefence {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match self.state {
            SolidDefenceState::Init => match input {
                ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
                ProcInput::Roll(RollResult::D6(roll)) => {
                    self.team = game_state.info.kicking_this_drive;
                    self.max_rearrange = Self::max_rearrange_from_d6(roll as u8);
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
                ProcInput::Nothing => self.build_selection_actions(game_state),
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
                ProcInput::Nothing => {
                    build_rearrange_actions(game_state, self.rearrange_cfg, &self.rearrange_state)
                }
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
            state.fixes.fix_d6(1); // D3+3 => 4
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
    // #[test]
    // fn kickoff_brilliant_coaching() {
    //     let mut state: GameState = GameStateBuilder::new_at_kickoff();
    //     // ball fixes
    //     state.fixes.fix_d8_direction(Direction::up()); // scatter direction
    //     state.fixes.fix_d6(5); // scatter length
    //
    //     // kickoff event fix
    //     state.fixes.fix_d6(1);
    //     state.fixes.fix_d6(1);
    //
    //     state.fixes.fix_d6(5); //fix home brilliant coaching roll
    //     state.fixes.fix_d6(6); //fix away brilliant coaching roll
    //
    //     state.step_simple(SimpleAT::KickoffAimMiddle);
    //
    //     assert_eq!(state.away.rerolls, 4);
    //     assert_eq!(state.home.rerolls, 3);
    // }
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

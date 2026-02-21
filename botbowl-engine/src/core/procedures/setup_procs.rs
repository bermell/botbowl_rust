use super::AnyProc;
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{
    Action, AvailableActions, Coord, DugoutPlace, PlayerID, Position, ProcState, Procedure, Result,
    TeamType, HEIGHT_, LINE_OF_SCRIMMAGE_Y_RANGE,
};
use crate::core::table::{PlayerRole, PosAT, SimpleAT};

use serde::{Deserialize, Serialize};

use rand::Rng;
use std::collections::HashSet;
use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SetupLegalConfig {
    pub(super) team: TeamType,
    pub(super) line_x: Coord,
    pub(super) target_on_pitch: usize,
    pub(super) min_los: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SetupCounts {
    pub(super) on_pitch: usize,
    pub(super) los: usize,
    pub(super) north: usize,
    pub(super) south: usize,
}

pub(super) fn count_setup_positions_with_removed(
    state: &GameState,
    cfg: SetupLegalConfig,
    removed_pos: Option<Position>,
) -> SetupCounts {
    let mut counts = SetupCounts {
        on_pitch: 0,
        los: 0,
        north: 0,
        south: 0,
    };

    for pos in state
        .get_players_on_pitch_in_team(cfg.team)
        .map(|p| p.position)
    {
        counts.on_pitch += 1;
        if pos.is_los_position(cfg.line_x) {
            counts.los += 1;
        } else if pos.is_north_wing_position() {
            counts.north += 1;
        } else if pos.is_south_wing_position() {
            counts.south += 1;
        }
    }

    if let Some(pos) = removed_pos {
        counts.on_pitch = counts.on_pitch.saturating_sub(1);
        if pos.is_los_position(cfg.line_x) {
            counts.los = counts.los.saturating_sub(1);
        } else if pos.is_north_wing_position() {
            counts.north = counts.north.saturating_sub(1);
        } else if pos.is_south_wing_position() {
            counts.south = counts.south.saturating_sub(1);
        }
    }

    counts
}

pub(super) fn legal_setup_positions_with_removed(
    game_state: &GameState,
    cfg: SetupLegalConfig,
    removed_pos: Option<Position>,
) -> Vec<Position> {
    let counts = count_setup_positions_with_removed(game_state, cfg, removed_pos);
    let remaining_to_place = cfg.target_on_pitch.saturating_sub(counts.on_pitch);

    if remaining_to_place == 0 {
        return Vec::new();
    }

    let empty_los_squares = LINE_OF_SCRIMMAGE_Y_RANGE
        .clone()
        .filter(|&y| {
            let pos = Position::new((cfg.line_x, y));
            !pos.is_out()
                && (game_state.get_player_id_at(pos).is_none() || removed_pos == Some(pos))
        })
        .count();

    let mut candidates = Vec::new();
    for pos in Position::all_positions() {
        if pos.is_out()
            || !pos.is_on_team_half(cfg.team, cfg.line_x)
            || (game_state.get_player_id_at(pos).is_some() && removed_pos != Some(pos))
        {
            continue;
        }

        let is_los = pos.is_los_position(cfg.line_x);
        let new_los = counts.los + usize::from(is_los);
        let new_north = counts.north + usize::from(pos.is_north_wing_position());
        let new_south = counts.south + usize::from(pos.is_south_wing_position());

        if new_north > 2 || new_south > 2 {
            continue;
        }

        let remaining_players_after = remaining_to_place.saturating_sub(1);
        let remaining_los_needed = cfg.min_los.saturating_sub(new_los);
        let remaining_los_squares = if is_los {
            empty_los_squares.saturating_sub(1)
        } else {
            empty_los_squares
        };

        if remaining_players_after < remaining_los_needed
            || remaining_los_squares < remaining_los_needed
        {
            continue;
        }

        candidates.push(pos);
    }

    candidates
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SetupRearrangeConfig {
    pub(super) team: TeamType,
    pub(super) target_on_pitch: usize,
    pub(super) min_los: usize,
    pub(super) end_requires_pending_empty: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SetupRearrangeState {
    pub(super) selected_fielded_player: Option<PlayerID>,
    pub(super) pending_reserve_ids: Vec<usize>,
    pub(super) controlled_fielded_ids: HashSet<PlayerID>,
}

impl SetupRearrangeState {
    fn empty() -> Self {
        Self {
            selected_fielded_player: None,
            pending_reserve_ids: Vec::new(),
            controlled_fielded_ids: HashSet::new(),
        }
    }
}

fn can_end_rearrange(
    game_state: &GameState,
    cfg: SetupRearrangeConfig,
    state: &SetupRearrangeState,
) -> bool {
    if game_state.get_players_on_pitch_in_team(cfg.team).count() != cfg.target_on_pitch {
        return false;
    }
    if cfg.end_requires_pending_empty && !state.pending_reserve_ids.is_empty() {
        // Todo: no test enters here. Investigate whether if condition can ever be true and cover with test in that case
        return false;
    }
    game_state.is_setup_legal_for_team(cfg.team)
}

pub(super) fn build_rearrange_actions(
    game_state: &GameState,
    cfg: SetupRearrangeConfig,
    state: &SetupRearrangeState,
) -> ProcState {
    let legal_cfg = SetupLegalConfig {
        team: cfg.team,
        line_x: game_state.get_line_of_scrimage_x(cfg.team),
        target_on_pitch: cfg.target_on_pitch,
        min_los: cfg.min_los,
    };
    let mut aa = AvailableActions::new(cfg.team);

    if let Some(source_id) = state.selected_fielded_player {
        let source_pos = game_state.get_player_unsafe(source_id).position;
        let mut positions =
            legal_setup_positions_with_removed(game_state, legal_cfg, Some(source_pos));
        positions.extend(
            state
                .controlled_fielded_ids
                .iter()
                .copied()
                .filter(|id| *id != source_id)
                .map(|id| game_state.get_player_unsafe(id).position),
        );
        positions.sort_unstable_by_key(|pos| (pos.x, pos.y));
        positions.dedup();
        aa.insert_positional(PosAT::SelectPosition, positions);
        return ProcState::NeedAction(aa);
    }

    if !state.pending_reserve_ids.is_empty() {
        let mut positions: Vec<Position> =
            legal_setup_positions_with_removed(game_state, legal_cfg, None)
                .into_iter()
                .filter(|pos| game_state.get_player_id_at(*pos).is_none())
                .collect();
        positions.extend(
            state
                .controlled_fielded_ids
                .iter()
                .copied()
                .map(|id| game_state.get_player_unsafe(id).position),
        );
        positions.sort_unstable_by_key(|pos| (pos.x, pos.y));
        positions.dedup();
        aa.insert_positional(PosAT::SelectPosition, positions);
        if can_end_rearrange(game_state, cfg, state) {
            aa.insert_simple(SimpleAT::EndSetup);
        }
        return ProcState::NeedAction(aa);
    }

    let positions: Vec<Position> = state
        .controlled_fielded_ids
        .iter()
        .copied()
        .map(|id| game_state.get_player_unsafe(id).position)
        .collect();
    if !positions.is_empty() {
        aa.insert_positional(PosAT::SelectPosition, positions);
    }
    if can_end_rearrange(game_state, cfg, state) {
        aa.insert_simple(SimpleAT::EndSetup);
    }
    ProcState::NeedAction(aa)
}

pub(super) fn step_rearrange_position(
    game_state: &mut GameState,
    cfg: SetupRearrangeConfig,
    state: &mut SetupRearrangeState,
    pos: Position,
) -> ProcState {
    if let Some(source_id) = state.selected_fielded_player.take() {
        if let Some(target_id) = game_state.get_player_id_at(pos) {
            assert!(state.controlled_fielded_ids.contains(&target_id));
            if source_id != target_id {
                game_state
                    .swap_players_positions(source_id, target_id)
                    .unwrap();
            }
        } else {
            game_state.move_player(source_id, pos).unwrap();
        }
        return build_rearrange_actions(game_state, cfg, state);
    }

    if let Some(reserve_id) = state.pending_reserve_ids.pop() {
        if let Some(displaced_id) = game_state.get_player_id_at(pos) {
            assert!(state.controlled_fielded_ids.contains(&displaced_id));
            let displaced_reserve_id = game_state
                .unfield_player_to_reserves_and_get_dugout_id(displaced_id)
                .unwrap();
            game_state.field_dugout_player(reserve_id, pos);
            let placed_id = game_state.get_player_id_at(pos).unwrap();
            state.pending_reserve_ids.push(displaced_reserve_id);
            state.controlled_fielded_ids.remove(&displaced_id);
            state.controlled_fielded_ids.insert(placed_id);
        } else {
            game_state.field_dugout_player(reserve_id, pos);
            state
                .controlled_fielded_ids
                .insert(game_state.get_player_id_at(pos).unwrap());
        }
        return build_rearrange_actions(game_state, cfg, state);
    }

    if let Some(id) = game_state.get_player_id_at(pos) {
        assert!(state.controlled_fielded_ids.contains(&id));
        state.selected_fielded_player = Some(id);
    }
    build_rearrange_actions(game_state, cfg, state)
}

pub(super) fn step_rearrange_end_setup(
    game_state: &GameState,
    cfg: SetupRearrangeConfig,
    state: &mut SetupRearrangeState,
) -> ProcState {
    state.selected_fielded_player = None;
    assert!(can_end_rearrange(game_state, cfg, state));
    ProcState::Done
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Setup {
    team: TeamType,
    rearrange_state: SetupRearrangeState,
}
impl Setup {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::Setup(Setup {
            team,
            rearrange_state: SetupRearrangeState::empty(),
        })
    }

    fn setup_legal_config(&self, game_state: &GameState) -> SetupLegalConfig {
        let total_on_pitch = game_state.get_players_on_pitch_in_team(self.team).count();
        let num_players_on_bench = game_state.get_reserve_count_for_team(self.team);
        let num_available_players = total_on_pitch + num_players_on_bench;
        SetupLegalConfig {
            team: self.team,
            line_x: game_state.get_line_of_scrimage_x(self.team),
            target_on_pitch: 11.min(num_available_players),
            min_los: 3.min(num_available_players),
        }
    }

    fn rearrange_config(&self, game_state: &GameState) -> SetupRearrangeConfig {
        let legal_cfg = self.setup_legal_config(game_state);
        SetupRearrangeConfig {
            team: self.team,
            target_on_pitch: legal_cfg.target_on_pitch,
            min_los: legal_cfg.min_los,
            end_requires_pending_empty: false,
        }
    }

    fn reset_rearrange_state(&mut self, game_state: &GameState) {
        let cfg = self.rearrange_config(game_state);
        let controlled_fielded_ids: HashSet<PlayerID> = game_state
            .get_players_on_pitch_in_team(self.team)
            .map(|p| p.id)
            .collect();
        let pending_count = cfg
            .target_on_pitch
            .saturating_sub(controlled_fielded_ids.len());
        let pending_reserve_ids = game_state
            .reserve_ids_for_team(self.team)
            .into_iter()
            .take(pending_count)
            .collect();
        self.rearrange_state = SetupRearrangeState {
            selected_fielded_player: None,
            pending_reserve_ids,
            controlled_fielded_ids,
        };
    }

    fn ensure_rearrange_state_initialized(&mut self, game_state: &GameState) {
        if self.rearrange_state.selected_fielded_player.is_none()
            && self.rearrange_state.pending_reserve_ids.is_empty()
            && self.rearrange_state.controlled_fielded_ids.is_empty()
        {
            self.reset_rearrange_state(game_state);
        }
    }

    fn get_empty_pos_in_box(
        game_state: &GameState,
        x_range: RangeInclusive<Coord>,
        y_range: RangeInclusive<Coord>,
    ) -> Position {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(x_range.clone());
            let y = rng.gen_range(y_range.clone());
            if game_state.get_player_id_at_coord(x, y).is_none() {
                return Position { x, y };
            }
        }
    }

    #[cfg(test)]
    fn get_legal_setup_positions_with_removed(
        &self,
        game_state: &GameState,
        removed_pos: Option<Position>,
    ) -> Vec<Position> {
        legal_setup_positions_with_removed(
            game_state,
            self.setup_legal_config(game_state),
            removed_pos,
        )
    }

    #[cfg(test)]
    fn get_legal_setup_positions(&self, game_state: &GameState) -> Vec<Position> {
        self.get_legal_setup_positions_with_removed(game_state, None)
    }

    fn build_available_actions(&mut self, game_state: &GameState) -> ProcState {
        self.ensure_rearrange_state_initialized(game_state);
        let cfg = self.rearrange_config(game_state);
        let ProcState::NeedAction(mut aa) =
            build_rearrange_actions(game_state, cfg, &self.rearrange_state)
        else {
            unreachable!();
        };
        if game_state
            .get_players_on_pitch_in_team(self.team)
            .next()
            .is_none()
        {
            aa.insert_simple(SimpleAT::SetupLine);
        }
        ProcState::NeedAction(aa)
    }

    pub fn random_setup(&self, game_state: &mut GameState) {
        #[allow(clippy::needless_collect)]
        let players: Vec<PlayerID> = game_state
            .get_dugout()
            .take(11)
            .filter(|dplayer| dplayer.stats.team == self.team)
            .map(|p| p.id)
            .collect();

        let mut ids = players.into_iter();
        let los_x = game_state.get_line_of_scrimage_x(self.team);
        let los_x_range = los_x..=los_x;
        let x_range = match self.team {
            TeamType::Home => los_x..=crate::core::model::WIDTH_ - 2,
            TeamType::Away => 1..=los_x,
        };
        for _ in 0..3 {
            if let Some(id) = ids.next() {
                let p = Setup::get_empty_pos_in_box(
                    game_state,
                    los_x_range.clone(),
                    LINE_OF_SCRIMMAGE_Y_RANGE.clone(),
                );
                game_state.field_dugout_player(id, p);
            }
        }
        for id in ids {
            let p = Setup::get_empty_pos_in_box(
                game_state,
                x_range.clone(),
                LINE_OF_SCRIMMAGE_Y_RANGE.clone(),
            );
            game_state.field_dugout_player(id, p);
        }
    }

    fn setup_line(&self, game_state: &mut GameState) -> Result<()> {
        //unfield all players
        let player_ids = game_state
            .get_players_on_pitch_in_team(self.team)
            .map(|p| p.id)
            .collect::<Vec<_>>();
        for id in player_ids {
            game_state.unfield_player(id, DugoutPlace::Reserves)?;
        }
        let mut linemen_pos = vec![(0, 0), (0, -1), (0, 1), (0, -3), (0, 3)];
        let mut blitzer_pos = vec![(0, -2), (0, 2)];
        let mut catcher_pos = vec![(2, 2), (2, -2)];
        let mut thrower_pos = vec![(6, 3), (6, -3)];
        #[allow(clippy::needless_collect)]
        let players: Vec<PlayerID> = game_state
            .get_dugout()
            .filter(|dplayer| dplayer.stats.team == self.team)
            .filter(|dplayer| dplayer.place == DugoutPlace::Reserves)
            .map(|p| p.id)
            .collect();
        let x_delta_sign = if self.team == TeamType::Home { 1 } else { -1 };
        let middle_x = game_state.get_line_of_scrimage_x(self.team);
        let middle_y = HEIGHT_ / 2;
        for id in players {
            let player = game_state.get_dugout_player(id).unwrap();
            let (dx, dy) = {
                match player.stats.role {
                    PlayerRole::Blitzer if !blitzer_pos.is_empty() => blitzer_pos.pop().unwrap(),
                    PlayerRole::Thrower if !thrower_pos.is_empty() => thrower_pos.pop().unwrap(),
                    PlayerRole::Catcher if !catcher_pos.is_empty() => catcher_pos.pop().unwrap(),
                    PlayerRole::Lineman if !linemen_pos.is_empty() => linemen_pos.pop().unwrap(),
                    _ => continue,
                }
            };
            let position = Position::new((middle_x + dx * x_delta_sign, middle_y + dy));
            game_state.log(format!(
                "fielding {:?} {:?} at {:?}",
                player.stats.role, player.stats.team, position
            ));
            game_state.field_dugout_player(id, position)
        }
        Ok(())
    }
}

impl Procedure for Setup {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        if input == ProcInput::Nothing {
            return self.build_available_actions(game_state);
        }

        match input {
            ProcInput::Action(Action::Simple(SimpleAT::SetupLine)) => {
                self.setup_line(game_state).unwrap();
                self.reset_rearrange_state(game_state);
                self.build_available_actions(game_state)
            }

            ProcInput::Action(Action::Simple(SimpleAT::EndSetup)) => step_rearrange_end_setup(
                game_state,
                self.rearrange_config(game_state),
                &mut self.rearrange_state,
            ),
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                self.ensure_rearrange_state_initialized(game_state);
                step_rearrange_position(
                    game_state,
                    self.rearrange_config(game_state),
                    &mut self.rearrange_state,
                    pos,
                )
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{gamestate::GameStateBuilder, model::PlayerStats};
    use std::collections::HashSet;
    use std::iter::zip;

    fn add_reserves(state: &mut GameState, team: TeamType, count: usize) {
        for _ in 0..count {
            state.dugout_add_new_player(PlayerStats::new_lineman(team), DugoutPlace::Reserves);
        }
    }

    fn first_empty_position(positions: &[Position], state: &GameState) -> Position {
        *positions
            .iter()
            .find(|&&pos| state.get_player_id_at(pos).is_none())
            .unwrap()
    }

    fn setup_for(team: TeamType) -> Setup {
        Setup {
            team,
            rearrange_state: SetupRearrangeState::empty(),
        }
    }

    #[test]
    fn half_constraint_for_both_teams() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        add_reserves(&mut state, TeamType::Away, 11);

        for team in [TeamType::Home, TeamType::Away] {
            let setup = setup_for(team);
            let positions = setup.get_legal_setup_positions(&state);
            assert!(!positions.is_empty());
            let line_x = state.get_line_of_scrimage_x(team);
            for pos in positions {
                assert!(!pos.is_out());
                match team {
                    TeamType::Home => assert!(pos.x >= line_x),
                    TeamType::Away => assert!(pos.x <= line_x),
                }
            }
        }
    }

    #[test]
    fn setup_specific_player_from_dugout_reserve() {
        let mut state: GameState = GameStateBuilder::new_at_setup();
        let team: TeamType = TeamType::Away;
        let mut setup: Setup = setup_for(team);

        let middle_x = state.get_line_of_scrimage_x(team);
        let middle_y = HEIGHT_ / 2;
        let pitch_pos: Position = Position {
            x: middle_x,
            y: middle_y,
        };

        let reserves_before = state
            .get_dugout()
            .filter(|p| p.stats.team == team && p.place == DugoutPlace::Reserves)
            .count();

        // build legal action squares
        let _ = setup.step(&mut state, ProcInput::Nothing);

        // should be able to select select_player to place at pitch_pos
        let _ = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pitch_pos)),
        );

        let placed_player = state.get_player_at(pitch_pos).unwrap();
        assert_eq!(placed_player.stats.team, team);
        let reserves_after = state
            .get_dugout()
            .filter(|p| p.stats.team == team && p.place == DugoutPlace::Reserves)
            .count();
        assert_eq!(reserves_before - 1, reserves_after);
    }

    #[test]
    fn change_setup_square_for_already_fielded_player() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        let mut setup = setup_for(team);
        let line_x = state.get_line_of_scrimage_x(team);
        let source_pos = Position::new((line_x, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));
        let player_id = state
            .add_new_player_to_field(PlayerStats::new_lineman(team), source_pos)
            .unwrap();

        let ProcState::NeedAction(aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction before selecting source player.");
        };
        assert!(aa.get_positions_for_action(PosAT::SelectPosition).contains(&source_pos));

        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, source_pos)),
        ) else {
            panic!("Expected NeedAction after selecting source player.");
        };
        let destination =
            first_empty_position(
                &aa.get_positions_for_action(PosAT::SelectPosition),
                &state
            );

        let _ = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, destination)),
        );

        assert_eq!(state.get_player_id_at(source_pos), None);
        assert_eq!(state.get_player_id_at(destination), Some(player_id));
        assert_eq!(state.get_players_on_pitch_in_team(team).count(), 1);
    }

    #[test]
    fn during_setup_swap_position_of_dugout_reserve_player_with_fielded_player() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        let mut setup = setup_for(team);
        add_reserves(&mut state, team, 1);

        let line_x = state.get_line_of_scrimage_x(team);
        let occupied_pos = Position::new((line_x + 1, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));
        state
            .add_new_player_to_field(PlayerStats::new_lineman(team), occupied_pos)
            .unwrap();
        let reserves_before = state
            .get_dugout()
            .filter(|player| player.stats.team == team && player.place == DugoutPlace::Reserves)
            .count();

        let ProcState::NeedAction(aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction before reserve swap.");
        };
        assert!(aa.get_positions_for_action(PosAT::SelectPosition).contains(&occupied_pos));

        let _ = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, occupied_pos)),
        );

        assert_eq!(state.get_players_on_pitch_in_team(team).count(), 1);
        assert!(state.get_player_id_at(occupied_pos).is_some());
        let reserves_after = state
            .get_dugout()
            .filter(|player| player.stats.team == team && player.place == DugoutPlace::Reserves)
            .count();
        assert_eq!(reserves_before, reserves_after);
    }

    #[test]
    fn during_setup_swap_positions_of_already_fielded_players() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        let mut setup = setup_for(team);
        let line_x = state.get_line_of_scrimage_x(team);
        let first_pos = Position::new((line_x + 1, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));
        let second_pos = Position::new((line_x + 2, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));
        let first_id = state
            .add_new_player_to_field(PlayerStats::new_lineman(team), first_pos)
            .unwrap();
        let second_id = state
            .add_new_player_to_field(PlayerStats::new_lineman(team), second_pos)
            .unwrap();

        let _ = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, first_pos)),
        );
        let _ = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, second_pos)),
        );

        assert_eq!(state.get_player_id_at(first_pos), Some(second_id));
        assert_eq!(state.get_player_id_at(second_pos), Some(first_id));
        assert_eq!(state.get_players_on_pitch_in_team(team).count(), 2);
    }

    #[test]
    fn wing_cap_blocks_north_wing_positions() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let north_y = *crate::core::model::NORTH_WING_Y_RANGE.start();

        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x + 1, north_y)),
            )
            .unwrap();
        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x + 1, north_y + 1)),
            )
            .unwrap();

        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();
        for offset in 0..3 {
            let offset = offset as Coord;
            state
                .add_new_player_to_field(
                    PlayerStats::new_lineman(TeamType::Home),
                    Position::new((line_x, los_y + offset)),
                )
                .unwrap();
        }

        let setup = setup_for(TeamType::Home);
        let positions = setup.get_legal_setup_positions(&state);
        assert!(positions.iter().all(|pos| !pos.is_north_wing_position()));
    }

    #[test]
    fn los_feasibility_requires_los_squares() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 2);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();

        state
            .add_new_player_to_field(
                PlayerStats::new_lineman(TeamType::Home),
                Position::new((line_x, los_y)),
            )
            .unwrap();

        let setup = setup_for(TeamType::Home);
        let positions = setup.get_legal_setup_positions(&state);
        assert!(!positions.is_empty());
        assert!(positions.iter().all(|pos| pos.is_los_position(line_x)));
    }

    #[test]
    fn placed_player_who_change_position_during_setup_keeps_same_id() {
        let mut state: GameState = GameStateBuilder::empty_state();
        let team: TeamType = TeamType::Home;
        let mut setup: Setup = setup_for(team);

        //place player
        let middle_x = state.get_line_of_scrimage_x(team);
        let middle_y = HEIGHT_ / 2;
        let pos: Position = Position {
            x: middle_x,
            y: middle_y,
        };
        let player_id: PlayerID = state
            .add_new_player_to_field(PlayerStats::new_lineman(team), pos)
            .unwrap();

        // First click selects source player.
        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)),
        ) else {
            panic!("Expected NeedAction after selecting source player.");
        };
        let destination = aa.get_positions_for_action(PosAT::SelectPosition)
            .into_iter()
            .find(|&candidate| candidate != pos && state.get_player_id_at(candidate).is_none())
            .unwrap();
        let ProcState::NeedAction(_aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, destination)),
        ) else {
            panic!("Expected NeedAction after moving selected player.");
        };
        assert_eq!(state.get_players_on_pitch_in_team(team).count(), 1);
        assert_eq!(state.get_player_id_at(pos), None);
        assert_eq!(state.get_player_id_at(destination), Some(player_id));
    }

    #[test]
    fn los_requirement_satisfied_allows_non_los_positions() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 1);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let los_y = *LINE_OF_SCRIMMAGE_Y_RANGE.start();

        for offset in 0..3 {
            let offset = offset as Coord;
            state
                .add_new_player_to_field(
                    PlayerStats::new_lineman(TeamType::Home),
                    Position::new((line_x, los_y + offset)),
                )
                .unwrap();
        }

        let setup = setup_for(TeamType::Home);
        let positions = setup.get_legal_setup_positions(&state);
        assert!(positions.iter().any(|pos| !pos.is_los_position(line_x)));
    }

    #[test]
    fn shared_setup_helper_matches_setup_logic_for_moving_a_fielded_player() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        let line_x = state.get_line_of_scrimage_x(team);

        let fielded_positions = [
            Position::new((line_x, 5)),
            Position::new((line_x, 6)),
            Position::new((line_x, 7)),
            Position::new((line_x + 1, 8)),
            Position::new((line_x + 2, 9)),
            Position::new((line_x + 3, 10)),
            Position::new((line_x + 4, 11)),
            Position::new((line_x + 5, 3)),
            Position::new((line_x + 5, 13)),
            Position::new((line_x + 2, 2)),
            Position::new((line_x + 2, 14)),
        ];
        for pos in fielded_positions {
            state
                .add_new_player_to_field(PlayerStats::new_lineman(team), pos)
                .unwrap();
        }

        let setup = setup_for(team);
        let removed_pos = Position::new((line_x + 2, 9));
        let from_setup: HashSet<Position> = setup
            .get_legal_setup_positions_with_removed(&state, Some(removed_pos))
            .into_iter()
            .collect();

        let cfg = SetupLegalConfig {
            team,
            line_x,
            target_on_pitch: 11,
            min_los: 3,
        };
        let from_shared: HashSet<Position> =
            legal_setup_positions_with_removed(&state, cfg, Some(removed_pos))
                .into_iter()
                .collect();

        assert_eq!(from_setup, from_shared);
    }

    #[test]
    fn occupied_positions_are_excluded() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 11);
        let line_x = state.get_line_of_scrimage_x(TeamType::Home);
        let occupied_pos = Position::new((line_x + 1, *LINE_OF_SCRIMMAGE_Y_RANGE.start()));

        state
            .add_new_player_to_field(PlayerStats::new_lineman(TeamType::Home), occupied_pos)
            .unwrap();

        let setup = setup_for(TeamType::Home);
        let positions = setup.get_legal_setup_positions(&state);
        assert!(!positions.contains(&occupied_pos));
    }

    #[test]
    fn test_setup_preconfigured_formations() {
        let mut state: GameState = GameStateBuilder::new_at_setup();
        //away as defense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);
        //home as offense
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);

        for team in [TeamType::Home, TeamType::Away] {
            let middle_x = state.get_line_of_scrimage_x(team);
            let middle_y = HEIGHT_ / 2;

            let linemen_pos = vec![(0, 0), (0, -1), (0, 1), (0, -3), (0, 3)];
            let blitzer_pos = vec![(0, -2), (0, 2)];
            let catcher_pos = vec![(2, 2), (2, -2)];
            let thrower_pos = vec![(6, 3), (6, -3)];
            let stats_types = vec![
                PlayerStats::new_lineman(team),
                PlayerStats::new_blitzer(team),
                PlayerStats::new_catcher(team),
                PlayerStats::new_thrower(team),
            ];
            let stats_positions = vec![linemen_pos, blitzer_pos, catcher_pos, thrower_pos];

            let expected_count = stats_positions.iter().map(|x| x.len()).sum::<usize>();
            let actual_count = state.get_players_on_pitch_in_team(team).count();
            assert_eq!(
                actual_count, expected_count,
                "Team {:?} has {:?} players,",
                team, actual_count
            );

            let x_delta_sign = if team == TeamType::Home { 1 } else { -1 };

            for (stats, positions) in zip(stats_types, stats_positions) {
                for (dx, dy) in positions {
                    let (x, y) = (middle_x + dx * x_delta_sign, middle_y + dy);
                    match state.get_player_at_coord(x, y) {
                    Some(correct_player) if correct_player.stats == stats => (),
                    Some(wrong_player) => panic!(
                        "Wrong player at ({:?}, {:?}), found a {:?} ({:?}) but expected a {:?} ({:?})",
                        x, y, wrong_player.stats.role, wrong_player.stats.team, stats.role, stats.team
                    ),
                    None => panic!(
                        "No player at ({:?}, {:?}), expected a {:?} ({:?})",
                        x, y, stats.role, stats.team
                    ),
                }
                }
            }
        }
    }

    #[test]
    fn manual_setup_places_players_until_done() {
        let mut state = GameStateBuilder::empty_state();
        add_reserves(&mut state, TeamType::Home, 3);

        let mut setup = setup_for(TeamType::Home);

        let ProcState::NeedAction(aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction on initial setup prompt.");
        };

        let positions = aa.get_positions_for_action( PosAT::SelectPosition);
        assert!(!positions.is_empty());
        assert!(aa.get_simple().contains(&SimpleAT::SetupLine));
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, positions[0])),
        ) else {
            panic!("Expected NeedAction after placing first player.");
        };

        assert_eq!(
            state.get_players_on_pitch_in_team(TeamType::Home).count(),
            1
        );
        assert!(!aa.get_simple().contains(&SimpleAT::SetupLine));
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let positions = aa.get_positions_for_action(PosAT::SelectPosition);
        let next_pos = first_empty_position(&positions, &state);
        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, next_pos)),
        ) else {
            panic!("Expected NeedAction after placing second player.");
        };

        assert_eq!(
            state.get_players_on_pitch_in_team(TeamType::Home).count(),
            2
        );
        assert!(!aa.get_simple().contains(&SimpleAT::EndSetup));

        let positions = aa.get_positions_for_action(PosAT::SelectPosition);
        let next_pos = first_empty_position(&positions, &state);
        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, next_pos)),
        ) else {
            panic!("Expected NeedAction after placing third player.");
        };

        assert_eq!(
            state.get_players_on_pitch_in_team(TeamType::Home).count(),
            3
        );
        assert!(aa.get_simple().contains(&SimpleAT::EndSetup));
        assert!(aa.get_positions_for_action(PosAT::SelectPosition)
            .iter()
            .all(|pos| state.get_player_id_at(*pos).is_some()));

        let reserves_left = state
            .get_dugout()
            .filter(|player| player.stats.team == TeamType::Home)
            .filter(|player| player.place == DugoutPlace::Reserves)
            .count();
        assert_eq!(reserves_left, 0);
    }

    #[test]
    fn setup_pending_flow_can_swap_with_controlled_occupied_square() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        add_reserves(&mut state, team, 3);
        let mut setup = setup_for(team);

        let ProcState::NeedAction(aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction on initial setup prompt.");
        };
        let first_target =
            first_empty_position(&aa.get_positions_for_action(PosAT::SelectPosition), &state);

        let ProcState::NeedAction(aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, first_target)),
        ) else {
            panic!("Expected NeedAction after first placement.");
        };
        let reserves_before_swap = state.get_reserve_count_for_team(team);
        assert!(
            aa.get_positions_for_action(PosAT::SelectPosition).contains(&first_target),
            "already placed controlled square should remain legal for reserve swap"
        );

        let ProcState::NeedAction(_aa) = setup.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, first_target)),
        ) else {
            panic!("Expected NeedAction after reserve swap.");
        };
        assert_eq!(
            state.get_reserve_count_for_team(team),
            reserves_before_swap,
            "swapping pending reserve with a controlled player should not change reserve count"
        );
        assert!(state.get_player_id_at(first_target).is_some());
    }

    #[test]
    fn setup_can_end_with_extra_reserves_remaining() {
        let mut state = GameStateBuilder::empty_state();
        let team = TeamType::Home;
        add_reserves(&mut state, team, 12);
        let mut setup = setup_for(team);

        let ProcState::NeedAction(mut aa) = setup.step(&mut state, ProcInput::Nothing) else {
            panic!("Expected NeedAction on initial setup prompt.");
        };

        for _ in 0..11 {
            let target =
                first_empty_position(&aa.get_positions_for_action(PosAT::SelectPosition), &state);
            let ProcState::NeedAction(next_aa) = setup.step(
                &mut state,
                ProcInput::Action(Action::Positional(PosAT::SelectPosition, target)),
            ) else {
                panic!("Expected NeedAction while placing mandatory setup players.");
            };
            aa = next_aa;
        }

        assert!(aa.get_simple().contains(&SimpleAT::EndSetup));
        assert_eq!(state.get_reserve_count_for_team(team), 1);

        let done = setup.step(
            &mut state,
            ProcInput::Action(Action::Simple(SimpleAT::EndSetup)),
        );
        assert!(matches!(done, ProcState::Done));
    }

    #[test]
    fn random_setup() {
        // Todo: implement random_setup happy path test
    }
}

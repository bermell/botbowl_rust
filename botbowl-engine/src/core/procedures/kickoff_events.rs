//! The S3 kick-off events `KickoffTable` resolves, and Sweltering Heat's end-of-drive faint.
//!
//! A D3 is a D6 read as 1-2 / 3-4 / 5-6 (`d3`), as on the tabletop, so no event needs a roll type
//! of its own; "randomly selects a player" is `RequestedRoll::Pick` over the candidates in id
//! order. The decisions reuse existing actions: `SelectPosition` on a player picks them,
//! `PlacePlayer` puts the picked player on a square, `EndPlayerTurn` declines or stops picking,
//! and Charge! is a turn of `StartMove` / `StartBlitz` / `EndTurn`.

use serde::{Deserialize, Serialize};

use crate::core::dices::{RequestedRoll, RollResult, D6};
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, DugoutPlace, PlayerID, PlayerStatus, Position, ProcInput,
    ProcState, Procedure, TeamType,
};
use crate::core::procedures::kickoff_procs::{role_rank, staging_squares, Setup};
use crate::core::procedures::movement_procs::MoveAction;
use crate::core::procedures::team_mate_procs;
use crate::core::table::*;

use super::AnyProc;

#[cfg(test)]
#[path = "kickoff_events_tests.rs"]
mod tests;

/// A D3 read off a D6.
pub fn d3(d6: D6) -> u8 {
    (d6 as u8).div_ceil(2)
}

/// `team`'s players on the pitch, in id order: the candidates of a random pick.
fn on_pitch(game_state: &GameState, team: TeamType) -> Vec<PlayerID> {
    game_state.get_players_on_pitch_in_team(team).map(|p| p.id).collect()
}

/// `team`'s Open players (Standing, not Marked), in id order.
fn open_players(game_state: &GameState, team: TeamType) -> Vec<PlayerID> {
    game_state
        .get_players_on_pitch_in_team(team)
        .filter(|p| game_state.is_open(p))
        .map(|p| p.id)
        .collect()
}

fn positions_of(game_state: &GameState, ids: &[PlayerID]) -> Vec<Position> {
    ids.iter()
        .map(|&id| game_state.get_player_unsafe(id).position)
        .collect()
}

/// Which roll-off a `RollOff` settles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RollOffEvent {
    /// D6 + Cheerleaders, highest wins (ties: both).
    CheeringFans,
    /// D6 + Assistant Coaches, highest wins (ties: both).
    BrilliantCoaching,
    /// D6, lowest loses (ties: both).
    DodgySnack,
    /// D6 + Fan Factor, lowest loses (ties: both).
    PitchInvasion,
}

/// Both coaches roll a D6 (Home first) and add the event's bonus; the event then goes to the
/// higher or lower total, or to both teams on a tie.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RollOff {
    event: RollOffEvent,
    home: Option<u8>,
}
impl RollOff {
    pub fn new(event: RollOffEvent) -> AnyProc {
        AnyProc::RollOff(RollOff { event, home: None })
    }
    fn bonus(&self, game_state: &GameState, team: TeamType) -> u8 {
        let t = game_state.get_team(team);
        match self.event {
            RollOffEvent::CheeringFans => t.cheerleaders,
            RollOffEvent::BrilliantCoaching => t.assistant_coaches,
            RollOffEvent::DodgySnack => 0,
            RollOffEvent::PitchInvasion => t.fan_factor,
        }
    }
}
impl Procedure for RollOff {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let roll = match input {
            ProcInput::Nothing => return ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(d6)) => d6 as u8,
            _ => panic!("Unexpected input {:?}", input),
        };
        let Some(home_roll) = self.home else {
            self.home = Some(roll);
            return ProcState::NeedRoll(RequestedRoll::D6);
        };
        let home = home_roll + self.bonus(game_state, TeamType::Home);
        let away = roll + self.bonus(game_state, TeamType::Away);
        let highest_wins = matches!(self.event, RollOffEvent::CheeringFans | RollOffEvent::BrilliantCoaching);
        let teams: Vec<TeamType> = match home.cmp(&away) {
            std::cmp::Ordering::Equal => vec![TeamType::Home, TeamType::Away],
            std::cmp::Ordering::Greater if highest_wins => vec![TeamType::Home],
            std::cmp::Ordering::Less if !highest_wins => vec![TeamType::Home],
            _ => vec![TeamType::Away],
        };
        crate::game_log!(game_state, "{:?} roll-off {home} vs {away}: {:?}", self.event, teams);
        let mut procs = Vec::new();
        for team in teams {
            match self.event {
                RollOffEvent::CheeringFans => game_state.get_mut_team(team).cheering_fans = true,
                RollOffEvent::BrilliantCoaching => game_state.get_mut_team(team).drive_rerolls += 1,
                RollOffEvent::DodgySnack => procs.push(DodgySnack::new(team)),
                RollOffEvent::PitchInvasion => procs.push(PitchInvasion::new(team)),
            }
        }
        ProcState::DoneNewProcs(procs)
    }
}

/// Dodgy Snack for one team: a random player on the pitch rolls a D6. On a 2+ they have -1 MA and
/// -1 AV for the drive (`FieldedPlayer::snack`); on a 1 they go to the Reserves box for the drive
/// (`DugoutPlace::Heated`, released when the next drive starts).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DodgySnack {
    team: TeamType,
    player: Option<PlayerID>,
}
impl DodgySnack {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::DodgySnack(DodgySnack { team, player: None })
    }
}
impl Procedure for DodgySnack {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let candidates = on_pitch(game_state, self.team);
        match input {
            ProcInput::Nothing if candidates.is_empty() => ProcState::Done,
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::Pick(candidates.len() as u8)),
            ProcInput::Roll(RollResult::Pick(i)) => {
                self.player = Some(candidates[i as usize]);
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Roll(RollResult::D6(D6::One)) => {
                game_state
                    .unfield_player(self.player.unwrap(), DugoutPlace::Heated)
                    .unwrap();
                ProcState::Done
            }
            ProcInput::Roll(RollResult::D6(_)) => {
                let player = game_state.get_mut_player_unsafe(self.player.unwrap());
                // Undone by `unfield_player`. Rosters have MA >= 1 and AV >= 2+.
                player.stats.ma = player.stats.ma.saturating_sub(1);
                player.stats.armour = player.stats.armour.saturating_sub(1);
                player.snack = true;
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

/// Pitch Invasion for one team: D3 random players on the pitch are Placed Prone and Stunned.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PitchInvasion {
    team: TeamType,
    left: Option<u8>,
}
impl PitchInvasion {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::PitchInvasion(PitchInvasion { team, left: None })
    }
}
impl Procedure for PitchInvasion {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => return ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(d6)) => self.left = Some(d3(d6)),
            ProcInput::Roll(RollResult::Pick(i)) => {
                let id = self.candidates(game_state)[i as usize];
                game_state.get_mut_player_unsafe(id).status = PlayerStatus::Stunned;
                *self.left.as_mut().unwrap() -= 1;
            }
            _ => panic!("Unexpected input {:?}", input),
        }
        let candidates = self.candidates(game_state);
        if self.left == Some(0) || candidates.is_empty() {
            ProcState::Done
        } else {
            ProcState::NeedRoll(RequestedRoll::Pick(candidates.len() as u8))
        }
    }
}
impl PitchInvasion {
    /// Not yet selected: a player is selected once, and is Stunned from then on.
    fn candidates(&self, game_state: &GameState) -> Vec<PlayerID> {
        game_state
            .get_players_on_pitch_in_team(self.team)
            .filter(|p| p.status != PlayerStatus::Stunned)
            .map(|p| p.id)
            .collect()
    }
}

/// Sweltering Heat at the end of a drive: one D3, and each team sends that many random players of
/// those on the pitch to the Reserves box for the next drive (`DugoutPlace::Heated`). Then it
/// clears the pitch, which `Half::do_kickoff` would otherwise have done.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SwelteringHeat {
    count: Option<u8>,
    team: TeamType,
    left: u8,
}
impl SwelteringHeat {
    pub fn new() -> AnyProc {
        AnyProc::SwelteringHeat(SwelteringHeat {
            count: None,
            team: TeamType::Home,
            left: 0,
        })
    }
}
impl Procedure for SwelteringHeat {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => return ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(d6)) => {
                self.count = Some(d3(d6));
                self.left = d3(d6);
            }
            ProcInput::Roll(RollResult::Pick(i)) => {
                let id = on_pitch(game_state, self.team)[i as usize];
                game_state.unfield_player(id, DugoutPlace::Heated).unwrap();
                self.left -= 1;
            }
            _ => panic!("Unexpected input {:?}", input),
        }
        loop {
            let candidates = on_pitch(game_state, self.team);
            if self.left > 0 && !candidates.is_empty() {
                return ProcState::NeedRoll(RequestedRoll::Pick(candidates.len() as u8));
            }
            if self.team == TeamType::Away {
                game_state.unfield_all_players().unwrap();
                return ProcState::Done;
            }
            self.team = TeamType::Away;
            self.left = self.count.unwrap();
        }
    }
}

/// High Kick: the receiving coach may put one Open player on the square the ball will land in —
/// if that square is empty and in the receiving half. `SelectPosition` on the player, or
/// `EndPlayerTurn` to decline.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HighKick {}
impl HighKick {
    pub fn new() -> AnyProc {
        AnyProc::HighKick(HighKick {})
    }
}
impl Procedure for HighKick {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let BallState::InAir(target) = game_state.ball else {
            unreachable!("High Kick with the ball not in the air")
        };
        let receiving = other_team(game_state.info.kicking_this_drive);
        match input {
            ProcInput::Nothing => {
                if game_state.is_out(target)
                    || !game_state.is_on_team_side(target, receiving)
                    || game_state.get_player_at(target).is_some()
                {
                    return ProcState::Done;
                }
                let players = open_players(game_state, receiving);
                if players.is_empty() {
                    return ProcState::Done;
                }
                let mut aa = AvailableActions::new(receiving);
                aa.insert_positional(PosAT::SelectPosition, positions_of(game_state, &players));
                aa.insert_simple(SimpleAT::EndPlayerTurn);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let id = game_state.get_player_id_at(pos).unwrap();
                game_state.move_player(id, target).unwrap();
                ProcState::Done
            }
            ProcInput::Action(Action::Simple(SimpleAT::EndPlayerTurn)) => ProcState::Done,
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

/// Quick Snap!: up to D3+3 of the receiving team's Open players (Open when the event happens) may
/// each move one square in any direction, across the line of scrimmage too. Per player:
/// `SelectPosition` on them, then `PlacePlayer` on an empty adjacent square; `EndPlayerTurn`
/// instead of a selection ends the event.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct QuickSnap {
    left: u8,
    eligible: Vec<PlayerID>,
    started: bool,
}
impl QuickSnap {
    pub fn new() -> AnyProc {
        AnyProc::QuickSnap(QuickSnap {
            left: 0,
            eligible: Vec::new(),
            started: false,
        })
    }
    fn free_adjacent(game_state: &GameState, id: PlayerID) -> Vec<Position> {
        let pos = game_state.get_player_unsafe(id).position;
        game_state
            .get_adj_positions(pos)
            .filter(|&p| !game_state.is_out(p) && game_state.get_player_at(p).is_none())
            .collect()
    }
    fn offer_selection(&self, game_state: &mut GameState) -> ProcState {
        game_state.info.active_player = None;
        let movable: Vec<PlayerID> = self
            .eligible
            .iter()
            .copied()
            .filter(|&id| !Self::free_adjacent(game_state, id).is_empty())
            .collect();
        if self.left == 0 || movable.is_empty() {
            return ProcState::Done;
        }
        let mut aa = AvailableActions::new(other_team(game_state.info.kicking_this_drive));
        aa.insert_positional(PosAT::SelectPosition, positions_of(game_state, &movable));
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        ProcState::NeedAction(aa)
    }
}
impl Procedure for QuickSnap {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let receiving = other_team(game_state.info.kicking_this_drive);
        match input {
            ProcInput::Nothing if !self.started => {
                self.started = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Roll(RollResult::D6(d6)) => {
                self.left = d3(d6) + 3;
                self.eligible = open_players(game_state, receiving);
                self.offer_selection(game_state)
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let id = game_state.get_player_id_at(pos).unwrap();
                game_state.info.active_player = Some(id);
                let mut aa = AvailableActions::new(receiving);
                aa.insert_positional(PosAT::PlacePlayer, Self::free_adjacent(game_state, id));
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::PlacePlayer, pos)) => {
                let id = game_state.info.active_player.unwrap();
                game_state.move_player(id, pos).unwrap();
                self.eligible.retain(|&e| e != id);
                self.left -= 1;
                self.offer_selection(game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                game_state.info.active_player = None;
                ProcState::Done
            }
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

/// Solid Defence: the kicking coach picks up to D3+3 of their Open players (Open when the event
/// happens) — `SelectPosition` on each, `EndPlayerTurn` to stop — and sets them up again. A picked
/// player is staged in the team's own endzone as `Setup` stages a reserve, and `Setup::resetup`
/// places them under the usual restrictions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SolidDefence {
    left: u8,
    eligible: Vec<PlayerID>,
    lifted: Vec<PlayerID>,
    started: bool,
}
impl SolidDefence {
    pub fn new() -> AnyProc {
        AnyProc::SolidDefence(SolidDefence {
            left: 0,
            eligible: Vec::new(),
            lifted: Vec::new(),
            started: false,
        })
    }
    fn offer_or_setup(&mut self, game_state: &mut GameState) -> ProcState {
        let kicking = game_state.info.kicking_this_drive;
        if self.left > 0 && !self.eligible.is_empty() {
            let mut aa = AvailableActions::new(kicking);
            aa.insert_positional(PosAT::SelectPosition, positions_of(game_state, &self.eligible));
            aa.insert_simple(SimpleAT::EndPlayerTurn);
            return ProcState::NeedAction(aa);
        }
        self.setup(game_state)
    }
    fn setup(&mut self, game_state: &mut GameState) -> ProcState {
        if self.lifted.is_empty() {
            return ProcState::Done;
        }
        let mut queue = std::mem::take(&mut self.lifted);
        // Setup's fixed order (role rank, then id), so equal choices recombine.
        queue.sort_by_key(|&id| (role_rank(game_state.get_player_unsafe(id).stats.role), id));
        let kicking = game_state.info.kicking_this_drive;
        ProcState::DoneNew(Setup::resetup(game_state, kicking, queue))
    }
}
impl Procedure for SolidDefence {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let kicking = game_state.info.kicking_this_drive;
        match input {
            ProcInput::Nothing if !self.started => {
                self.started = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Roll(RollResult::D6(d6)) => {
                self.left = d3(d6) + 3;
                self.eligible = open_players(game_state, kicking);
                self.offer_or_setup(game_state)
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => {
                let id = game_state.get_player_id_at(pos).unwrap();
                let staging = staging_squares(game_state.board_dims, kicking)
                    .find(|p| game_state.get_player_id_at(*p).is_none())
                    .expect("no free square to stage a player on");
                game_state.move_player(id, staging).unwrap();
                game_state.get_mut_player_unsafe(id).used = true;
                self.eligible.retain(|&e| e != id);
                self.lifted.push(id);
                self.left -= 1;
                self.offer_or_setup(game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::EndPlayerTurn)) => self.setup(game_state),
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

/// Charge!: up to D3+3 of the kicking team's Open players (Open when the event happens) are
/// activated one at a time, exactly as in a turn of theirs, for a free Move Action — one of them
/// may Blitz instead, one may Throw a Team-mate and one Kick a Team-mate (hidden declarations,
/// `team_mate_procs`: only a human takes them). The activated player Falling
/// Over or being Knocked Down ends the Charge, with no turnover; nothing else does (a team-mate
/// pushed into the crowd is no turnover here, see `GameState::in_charge`). `EndTurn` ends it
/// early. The ball is in the air, so nobody can pick it up, hand it off or pass it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Charge {
    team: TeamType,
    left: u8,
    eligible: Vec<PlayerID>,
    started: bool,
    /// The player activated last, until their activation has ended.
    current: Option<PlayerID>,
    /// `GameInfo::team_turn` before the Charge, put back after it.
    team_turn: TeamType,
}
impl Charge {
    pub fn new(team: TeamType) -> AnyProc {
        AnyProc::Charge(Charge {
            team,
            left: 0,
            eligible: Vec::new(),
            started: false,
            current: None,
            team_turn: team,
        })
    }
    pub fn team(&self) -> TeamType {
        self.team
    }
    fn end(&self, game_state: &mut GameState) -> ProcState {
        let info = &mut game_state.info;
        info.team_turn = self.team_turn;
        info.turnover = false;
        info.active_player = None;
        info.player_action_type = None;
        info.blitz_this_activation = false;
        info.pickup_this_activation = false;
        info.blitz_available = false;
        info.throw_team_mate_available = false;
        info.kick_team_mate_available = false;
        info.kick_team_mate_this_activation = false;
        info.cheering_assist = false;
        ProcState::Done
    }
    fn offer(&mut self, game_state: &mut GameState) -> ProcState {
        let info = &mut game_state.info;
        info.active_player = None;
        info.player_action_type = None;
        info.pickup_this_activation = false;
        info.cheering_assist = false;
        // One pushed into the crowd is gone.
        self.eligible.retain(|&id| game_state.get_player(id).is_ok());
        if self.left == 0 || self.eligible.is_empty() {
            return self.end(game_state);
        }
        let positions = positions_of(game_state, &self.eligible);
        let mut aa = AvailableActions::new(self.team);
        if game_state.info.blitz_available {
            aa.insert_positional(PosAT::StartBlitz, positions.clone());
        }
        // "... one may perform a free Throw Team-mate Action, and one may perform a free Kick
        // Team-mate Action": hidden, a human's only.
        team_mate_procs::offer_declarations(game_state, &positions, &mut aa);
        aa.insert_positional(PosAT::StartMove, positions);
        aa.insert_simple(SimpleAT::EndTurn);
        ProcState::NeedAction(aa)
    }
}
impl Procedure for Charge {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing if !self.started => {
                self.started = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Roll(RollResult::D6(d6)) => {
                self.left = d3(d6) + 3;
                self.eligible = open_players(game_state, self.team);
                let info = &mut game_state.info;
                self.team_turn = info.team_turn;
                info.team_turn = self.team;
                info.turnover = false;
                info.blitz_available = true;
                info.throw_team_mate_available = true;
                info.kick_team_mate_available = true;
                info.handoff_available = false;
                info.pass_available = false;
                info.foul_available = false;
                self.offer(game_state)
            }
            // An activation has ended. "If a selected player Falls Over or is Knocked Down during
            // their activation ... the Charge ends": a turnover that left them off their feet (or
            // the pitch). Placed Prone (Wrestle) is no turnover and goes on.
            ProcInput::Nothing => {
                let floored = self
                    .current
                    .take()
                    .is_some_and(|id| game_state.get_player(id).map_or(true, |p| p.status != PlayerStatus::Up));
                if game_state.info.turnover && floored {
                    return self.end(game_state);
                }
                game_state.info.turnover = false;
                self.offer(game_state)
            }
            ProcInput::Action(Action::Positional(at @ (PosAT::StartMove | PosAT::StartBlitz | PosAT::ThrowTeamMate), pos)) => {
                let id = game_state.get_player_id_at(pos).unwrap();
                game_state.set_active_player(id);
                if at == PosAT::ThrowTeamMate {
                    team_mate_procs::declare(game_state, id);
                }
                let info = &mut game_state.info;
                info.player_action_type = Some(at);
                if at == PosAT::StartBlitz {
                    info.blitz_available = false;
                    info.blitz_this_activation = true;
                }
                self.eligible.retain(|&e| e != id);
                self.left -= 1;
                self.current = Some(id);
                ProcState::NotDoneNew(MoveAction::new(id))
            }
            ProcInput::Action(Action::Simple(SimpleAT::EndTurn)) => self.end(game_state),
            _ => panic!("Unexpected input {:?}", input),
        }
    }
}

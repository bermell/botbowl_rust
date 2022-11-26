use core::panic;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashSet, VecDeque};

use crate::core::{bb_errors::EmptyProcStackError, model};

use model::*;

use super::{
    bb_errors::{IllegalActionError, IllegalMovePosition, InvalidPlayerId},
    dices::{D6, D8},
    procedures::Turn,
    table::PosAT,
};

pub const DIRECTIONS: [(Coord, Coord); 8] = [
    (1, 1),
    (0, 1),
    (-1, 1),
    (1, 0),
    (-1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

pub struct GameStateBuilder {
    home_players: Vec<Position>,
    away_players: Vec<Position>,
    ball_pos: Option<Position>,
}

impl GameStateBuilder {
    pub fn new(
        home_players: &[(Coord, Coord)],
        away_players: &[(Coord, Coord)],
    ) -> GameStateBuilder {
        let mut builder = GameStateBuilder {
            home_players: Vec::new(),
            away_players: Vec::new(),
            ball_pos: None,
        };
        home_players
            .iter()
            .for_each(|(x, y)| builder.home_players.push(Position::new((*x, *y))));
        away_players
            .iter()
            .for_each(|(x, y)| builder.away_players.push(Position::new((*x, *y))));

        builder
    }

    pub fn add_ball(&mut self, xy: (Coord, Coord)) -> &mut GameStateBuilder {
        self.ball_pos = Some(Position::new((xy.0, xy.1)));
        self
    }

    pub fn add_ball_pos(&mut self, position: Position) -> &mut GameStateBuilder {
        self.ball_pos = Some(position);
        self
    }

    pub fn build(&mut self) -> GameState {
        let mut state = GameState {
            fielded_players: Default::default(),
            home: TeamState::new(),
            away: TeamState::new(),
            board: Default::default(),
            ball: BallState::OffPitch,
            half: 1,
            turn: 1,
            active_player: None,
            game_over: false,
            dugout_players: Vec::new(),
            proc_stack: Vec::new(),
            new_procs: VecDeque::new(),
            available_actions: AvailableActions::new_empty(),
            rng: ChaCha8Rng::from_entropy(),
            d6_fixes: VecDeque::new(),
            d8_fixes: VecDeque::new(),
            rng_enabled: false,
            weather: Weather::Nice,
        };

        for position in self.home_players.iter() {
            let player_stats = PlayerStats::new(TeamType::Home);
            _ = state.field_player(player_stats, *position)
        }

        for position in self.away_players.iter() {
            let player_stats = PlayerStats::new(TeamType::Away);
            _ = state.field_player(player_stats, *position)
        }

        if let Some(pos) = self.ball_pos {
            state.ball = match state.get_player_at(pos) {
                None => BallState::OnGround(pos),
                Some(p) if p.status == PlayerStatus::Up => BallState::Carried(p.id),
                _ => panic!(),
            }
        }
        let mut proc = Turn {
            team: TeamType::Home,
        };
        state.available_actions = proc.available_actions(&state);
        state.proc_stack.push(Box::new(proc));

        state
    }
}

#[allow(dead_code)]
pub struct GameState {
    pub home: TeamState,
    pub away: TeamState,
    fielded_players: [Option<FieldedPlayer>; 22],
    dugout_players: Vec<DugoutPlayer>,
    board: FullPitch<Option<PlayerID>>,
    pub ball: BallState,
    pub half: u8,
    pub turn: u8,
    pub active_player: Option<PlayerID>,
    pub game_over: bool,
    pub weather: Weather,
    proc_stack: Vec<Box<dyn Procedure>>,
    new_procs: VecDeque<Box<dyn Procedure>>,
    available_actions: AvailableActions,
    pub rng_enabled: bool,
    rng: ChaCha8Rng,
    pub d6_fixes: VecDeque<D6>,
    pub d8_fixes: VecDeque<D8>,
    //rerolled_procs: ???? //TODO!!!
}

impl GameState {
    pub fn set_seed(&mut self, state: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(state);
    }

    pub fn get_d6_roll(&mut self) -> D6 {
        match self.d6_fixes.pop_front() {
            Some(roll) => roll,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }
    pub fn get_d8_roll(&mut self) -> D8 {
        match self.d8_fixes.pop_front() {
            Some(roll) => roll,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }

    pub fn get_team_from_player(&self, id: PlayerID) -> Result<&TeamState> {
        self.get_player(id)
            .map(|player| player.stats.team)
            .map(|team| self.get_team(team))
    }

    pub fn get_mut_team_from_player(&mut self, id: PlayerID) -> Result<&mut TeamState> {
        self.get_player(id)
            .map(|player| player.stats.team)
            .map(|team| self.get_mut_team(team))
    }

    pub fn get_team(&self, team: TeamType) -> &TeamState {
        match team {
            TeamType::Home => &self.home,
            TeamType::Away => &self.away,
        }
    }

    pub fn get_mut_team(&mut self, team: TeamType) -> &mut TeamState {
        match team {
            TeamType::Home => &mut self.home,
            TeamType::Away => &mut self.away,
        }
    }
    pub fn get_active_teamtype(&self) -> Option<TeamType> {
        self.available_actions.get_team()
    }

    pub fn get_active_team(&self) -> Option<&TeamState> {
        self.get_active_teamtype()
            .map(|team_type| self.get_team(team_type))
    }

    pub fn get_active_team_mut(&mut self) -> Option<&mut TeamState> {
        self.get_active_teamtype()
            .map(|team_type| self.get_mut_team(team_type))
    }

    pub fn get_player_id_at(&self, p: Position) -> Option<PlayerID> {
        self.get_player_id_at_coord(p.x, p.y)
    }
    pub fn get_player_at(&self, p: Position) -> Option<&FieldedPlayer> {
        self.get_player_at_coord(p.x, p.y)
    }

    pub fn get_player_id_at_coord(&self, x: Coord, y: Coord) -> Option<PlayerID> {
        //unwrap is OK here because if you're requesting negative indicies, you want the program to crash!
        let xx = usize::try_from(x).unwrap();
        let yy = usize::try_from(y).unwrap();
        self.board[xx][yy]
    }
    pub fn get_player_at_coord(&self, x: Coord, y: Coord) -> Option<&FieldedPlayer> {
        match self.get_player_id_at_coord(x, y) {
            None => None,
            Some(id) => Some(self.get_player(id).unwrap()),
            //above unwrap is safe for bad input. If it panics it's an interal logical error!
        }
    }

    pub fn get_player(&self, id: PlayerID) -> Result<&FieldedPlayer> {
        match &self.fielded_players[id] {
            Some(player) => Ok(player),
            None => Err(Box::new(InvalidPlayerId { id })),
        }
    }

    pub fn get_adj_positions(&self, p: Position) -> impl Iterator<Item = Position> {
        match p {
            position if position.is_out() => panic!(),
            Position { x, y } => DIRECTIONS
                .iter()
                .map(move |(dx, dy)| Position::new((x + dx, y + dy))),
        }
    }

    pub fn get_adj_players(&self, p: Position) -> impl Iterator<Item = &FieldedPlayer> + '_ {
        self.get_adj_positions(p)
            .filter_map(|adj_pos| self.get_player_at(adj_pos))
    }

    pub fn get_mut_player(&mut self, id: PlayerID) -> Result<&mut FieldedPlayer> {
        match &mut self.fielded_players[id] {
            Some(player) => Ok(player),
            None => Err(Box::new(InvalidPlayerId { id })),
        }
    }
    pub fn get_catch_modifers(&self, id: PlayerID) -> Result<i8> {
        let player = self.get_player(id)?;
        let team = player.stats.team;
        let mut modifier = -(self
            .get_adj_players(player.position)
            .filter(|player_| player_.stats.team != team && player_.has_tackle_zone())
            .count() as i8);

        if let Weather::Rain = self.weather {
            modifier -= 1;
        }
        Ok(modifier)
    }

    pub fn get_ball_position(&self) -> Option<Position> {
        match self.ball {
            BallState::OffPitch => None,
            BallState::OnGround(pos) => Some(pos),
            BallState::Carried(id) => Some(self.get_player(id).unwrap().position),
            BallState::InAir(pos) => Some(pos),
        }
    }

    pub fn move_player(&mut self, id: PlayerID, new_pos: Position) -> Result<()> {
        let (old_x, old_y) = self.get_player(id)?.position.to_usize()?;
        let (new_x, new_y) = new_pos.to_usize()?;
        if let Some(occupied_id) = self.board[new_x][new_y] {
            panic!(
                "Tried to move {}, to {:?} but it was already occupied by {}",
                id, new_pos, occupied_id
            );
            //return Err(Box::new(IllegalMovePosition{position: new_pos} ))
        }
        self.board[old_x][old_y] = None;
        self.get_mut_player(id)?.position = new_pos;
        self.board[new_x][new_y] = Some(id);
        Ok(())
    }
    pub fn get_players_on_pitch(&self) -> impl Iterator<Item = &FieldedPlayer> {
        self.fielded_players.iter().filter_map(|x| x.as_ref())
    }
    pub fn get_players_on_pitch_in_team(
        &self,
        team: TeamType,
    ) -> impl Iterator<Item = &FieldedPlayer> {
        self.get_players_on_pitch()
            .filter(move |p| p.stats.team == team)
    }
    pub fn field_player(
        &mut self,
        player_stats: PlayerStats,
        position: Position,
    ) -> Result<PlayerID> {
        let (new_x, new_y) = position.to_usize()?;
        if self.board[new_x][new_y].is_some() {
            return Err(Box::new(IllegalMovePosition { position }));
        }

        let id = match self
            .fielded_players
            .iter()
            .enumerate()
            .find(|(_, player)| player.is_none())
        {
            Some((id, _)) => id,
            None => panic!("Not room in gamestate of another fielded player!"),
        };

        self.board[new_x][new_y] = Some(id);
        self.fielded_players[id] = Some(FieldedPlayer {
            id,
            stats: player_stats,
            position,
            status: PlayerStatus::Up,
            used: false,
            moves: 0,
            used_skills: HashSet::new(),
        });
        Ok(id)
    }

    pub fn unfield_player(&mut self, id: PlayerID, place: DogoutPlace) -> Result<()> {
        if let BallState::Carried(carrier_id) = self.ball {
            assert_ne!(carrier_id, id);
            //if carrier_id == id {
            //return Err(Box::new(InvalidPlayerId{id: 4}))
            //}
        }

        let player = self.get_player(id)?;
        let (x, y) = player.position.to_usize()?;

        let dugout_player = DugoutPlayer {
            stats: player.stats.clone(),
            place,
        };
        self.dugout_players.push(dugout_player);

        self.board[x][y] = None;
        self.fielded_players[id] = None;
        Ok(())
    }

    pub fn push_proc(&mut self, proc: Box<dyn Procedure>) {
        self.new_procs.push_back(proc);
    }

    pub fn step(&mut self, action: Action) -> Result<()> {
        if !self.is_legal_action(&action) {
            return Err(Box::new(IllegalActionError { action }));
        }

        let mut top_proc = self
            .proc_stack
            .pop()
            .ok_or_else(|| Box::new(EmptyProcStackError {}))?;

        let mut top_proc_is_finished = top_proc.step(self, Some(action));

        loop {
            if self.game_over {
                break;
            }
            top_proc = match (self.new_procs.pop_back(), top_proc_is_finished) {
                (Some(new_top_proc), false) => {
                    self.proc_stack.push(top_proc);
                    new_top_proc
                }
                (Some(new_top_proc), true) => new_top_proc,
                (None, false) => top_proc,
                (None, true) => self
                    .proc_stack
                    .pop()
                    .ok_or_else(|| Box::new(EmptyProcStackError {}))?,
            };

            while let Some(new_proc) = self.new_procs.pop_front() {
                self.proc_stack.push(new_proc);
            }

            self.available_actions = top_proc.available_actions(self);
            if !self.available_actions.is_empty() {
                self.proc_stack.push(top_proc);
                break;
            }

            top_proc_is_finished = top_proc.step(self, None);
        }
        Ok(())
    }

    pub fn is_legal_action(&self, action: &Action) -> bool {
        /*let mut top_proc = self.proc_stack.pop().unwrap(); //TODO: remove these three lines at some point!
        debug_assert_eq!(top_proc.available_actions(self), self.available_actions);
        self.proc_stack.push(top_proc);
         */
        self.available_actions.is_legal_action(*action)
    }

    pub fn get_legal_positions(&self, at: PosAT) -> Vec<Position> {
        match self.available_actions.positional.get(&at) {
            Some(positions) => positions.clone(),
            None => Vec::new(),
        }
    }
}

use core::panic;
use derivative::Derivative;
use itertools::Itertools;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{max, min},
    collections::{HashSet, VecDeque},
};

use crate::core::{bb_errors::EmptyProcStackError, model, procedures::CoinToss};

use model::*;

use super::{
    bb_errors::{IllegalActionError, IllegalMovePosition, InvalidPlayerId, MissingActionError},
    dices::{BlockDice, Coin, D6Target, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6, D8},
    procedures::{AnyProc, GameOver, Half},
    table::{NumBlockDices, PosAT, SimpleAT},
};

pub enum BuilderState {
    Turn { turn: u8 },
    Setup { turn: u8 },
    Kickoff { turn: u8 },
    CoinToss,
}

pub struct GameStateBuilder {
    home_players: Vec<Position>,
    away_players: Vec<Position>,
    ball_pos: Option<Position>,
    state: BuilderState,
}

impl GameStateBuilder {
    /// creates a gamestate where away won the coin toss and choose to kick
    /// next is setup for away as defence and home as offence
    pub fn new_at_setup() -> GameState {
        let mut state: GameState = GameStateBuilder::new_start_of_game();

        state.fixes.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away
        state
    }

    /// creates a gamestate where away won the coin toss and choose to kick, and
    /// both teams setup their line of scrimmage
    pub fn new_at_kickoff() -> GameState {
        let mut state: GameState = GameStateBuilder::new_start_of_game();

        state.fixes.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away

        state.step_simple(SimpleAT::SetupLine); //Away
        state.step_simple(SimpleAT::EndSetup); //Away

        state.step_simple(SimpleAT::SetupLine); //Home
        state.step_simple(SimpleAT::EndSetup); //Home
        state
    }
    ///creates a gamestate with two human teams at very beginning of a gamestate
    ///which right now is the coin toss. (but later should be pregame which does weather roll abd
    ///such)
    pub fn new_start_of_game() -> GameState {
        let mut state = GameStateBuilder::empty_state();

        // Dugout
        let place = DugoutPlace::Reserves;
        for team in [TeamType::Home, TeamType::Away] {
            for _ in 0..6 {
                state.dugout_add_new_player(PlayerStats::new_lineman(team), place);
            }
            for _ in 0..2 {
                state.dugout_add_new_player(PlayerStats::new_blitzer(team), place);
            }
            for _ in 0..2 {
                state.dugout_add_new_player(PlayerStats::new_catcher(team), place);
            }
            for _ in 0..2 {
                state.dugout_add_new_player(PlayerStats::new_thrower(team), place);
            }
        }

        state.proc_stack = vec![GameOver::new(), Half::new(2), Half::new(1), CoinToss::new()];
        state.step_simple(SimpleAT::EndTurn);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::Heads)));
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::Tails)));
        // available_actions: AvailableActions::new_empty(),
        state
    }
    pub fn new() -> GameStateBuilder {
        GameStateBuilder {
            home_players: Vec::new(),
            away_players: Vec::new(),
            ball_pos: None,
            state: BuilderState::Turn { turn: 1 },
        }
    }
    pub fn add_str(&mut self, start_pos: Position, s: &str) -> &mut GameStateBuilder {
        let mut pos = start_pos;
        let start_x = pos.x;
        let mut newline = false;
        for c in s.chars() {
            assert!(!pos.is_out());
            match c {
                'a' => self.away_players.push(pos),
                'h' => self.home_players.push(pos),
                'H' => {
                    self.home_players.push(pos);
                    self.ball_pos = Some(pos);
                }
                'A' => {
                    self.away_players.push(pos);
                    self.ball_pos = Some(pos);
                }
                '\n' => newline = true,
                _ => (),
            }
            if newline {
                pos.y += 1;
                pos.x = start_x;
                newline = false;
            } else {
                pos.x += 1;
            }
        }
        self
    }
    pub fn add_home_player(&mut self, position: Position) -> &mut GameStateBuilder {
        self.home_players.push(position);
        self
    }
    pub fn set_state(&mut self, state: BuilderState) -> &mut GameStateBuilder {
        self.state = state;
        self
    }

    pub fn add_away_player(&mut self, position: Position) -> &mut GameStateBuilder {
        self.away_players.push(position);
        self
    }

    pub fn add_home_players(&mut self, players: &[(Coord, Coord)]) -> &mut GameStateBuilder {
        players
            .iter()
            .for_each(|(x, y)| self.home_players.push(Position::new((*x, *y))));
        self
    }

    pub fn add_away_players(&mut self, players: &[(Coord, Coord)]) -> &mut GameStateBuilder {
        players
            .iter()
            .for_each(|(x, y)| self.away_players.push(Position::new((*x, *y))));
        self
    }

    pub fn add_ball(&mut self, xy: (Coord, Coord)) -> &mut GameStateBuilder {
        self.ball_pos = Some(Position::new((xy.0, xy.1)));
        self
    }

    pub fn add_ball_pos(&mut self, position: Position) -> &mut GameStateBuilder {
        self.ball_pos = Some(position);
        self
    }

    pub fn empty_state() -> GameState {
        GameState {
            fielded_players: Default::default(),
            home: TeamState::new(),
            away: TeamState::new(),
            board: Default::default(),
            ball: BallState::OffPitch,
            dugout_players: Default::default(),
            proc_stack: Vec::new(),
            //new_procs: VecDeque::new(),
            available_actions: AvailableActions::new_empty(),
            rng: ChaCha8Rng::from_entropy(),
            rng_enabled: false,
            info: GameInfo::new(),
            fixes: Default::default(),
            log: Vec::new(),
            print_log: false,
            next_input: None,
        }
    }
    pub fn build(&mut self) -> GameState {
        let mut state = GameStateBuilder::new_start_of_game();

        let user_turn = match self.state {
            BuilderState::CoinToss => return state,
            BuilderState::Kickoff { turn } => turn,
            BuilderState::Setup { turn } => turn,
            BuilderState::Turn { turn } => turn,
        };
        assert!(user_turn > 0, "turn must be positive");
        assert_eq!(state.info.home_turn, 0);
        assert_eq!(state.info.away_turn, 0);

        state.fixes.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away

        //increase turn counter according to user wish
        state.info.home_turn += user_turn - 1;
        state.info.away_turn += user_turn - 1;

        state.step_simple(SimpleAT::SetupLine); //Away
        state.step_simple(SimpleAT::EndSetup); //Away

        state.step_simple(SimpleAT::SetupLine); //Home
        state.step_simple(SimpleAT::EndSetup); //Home

        if let BuilderState::Kickoff { .. } = self.state {
            return state;
        }
        // ball fixes
        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        // kickoff event fix - changing Weather
        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4);

        //changing weather - fair
        state.fixes.fix_d6(4);
        state.fixes.fix_d6(4);

        state.fixes.fix_d8_direction(Direction::down()); // gust of wind
        state.fixes.fix_d8_direction(Direction::down()); // bounce

        state.step_simple(SimpleAT::KickoffAimMiddle);
        state.clear_all_players().unwrap();
        state.ball = BallState::OffPitch;

        for position in self.home_players.iter() {
            let player_stats = PlayerStats::new_lineman(TeamType::Home);
            _ = state.add_new_player_to_field(player_stats, *position)
        }

        for position in self.away_players.iter() {
            let player_stats = PlayerStats::new_lineman(TeamType::Away);
            _ = state.add_new_player_to_field(player_stats, *position)
        }

        if let Some(pos) = self.ball_pos {
            state.ball = match state.get_player_at(pos) {
                None => BallState::OnGround(pos),
                Some(p) if p.status == PlayerStatus::Up => BallState::Carried(p.id),
                _ => panic!(),
            }
        }
        // decrease turn counter before calling endturn twice
        // (need to call end turn here to refresh available actions)
        state.step_simple(SimpleAT::EndTurn);
        state.step_simple(SimpleAT::EndTurn);
        state.info.home_turn -= 1;
        state.info.away_turn -= 1;

        state.set_logging_state(true);
        state
    }
}

impl Default for GameStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct GameInfo {
    pub half: u8,
    pub home_turn: u8,
    pub away_turn: u8,
    pub winner: Option<TeamType>,
    pub turnover: bool,
    pub active_player: Option<PlayerID>,
    pub player_action_type: Option<PosAT>,
    pub team_turn: TeamType,
    pub game_over: bool,
    pub weather: Weather,
    pub kicking_first_half: TeamType,
    pub kickoff_by_team: Option<TeamType>,
    pub kicking_this_drive: TeamType,
    pub handoff_available: bool,
    pub foul_available: bool,
    pub pass_available: bool,
    pub blitz_available: bool,
    pub handle_td_by: Option<PlayerID>,
}
impl GameInfo {
    fn new() -> GameInfo {
        GameInfo {
            half: 0,
            active_player: None,
            team_turn: TeamType::Away,
            game_over: false,
            winner: None,
            weather: Weather::Nice,
            kicking_first_half: TeamType::Away,
            home_turn: 0,
            away_turn: 0,
            player_action_type: None,
            handoff_available: true,
            pass_available: true,
            foul_available: true,
            blitz_available: true,
            handle_td_by: None,
            kickoff_by_team: None,
            kicking_this_drive: TeamType::Away,
            turnover: false,
        }
    }
}
#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct FixedDice {
    d3_fixes: VecDeque<D3>,
    d6_fixes: VecDeque<D6>,
    blockdice_fixes: VecDeque<BlockDice>,
    d8_fixes: VecDeque<D8>,
    coin_fixes: VecDeque<Coin>,
}
impl FixedDice {
    pub fn fix_coin(&mut self, value: Coin) {
        self.coin_fixes.push_back(value);
    }
    pub fn fix_d3(&mut self, value: u8) {
        self.d3_fixes.push_back(D3::try_from(value).unwrap());
    }
    pub fn fix_d6(&mut self, value: u8) {
        self.d6_fixes.push_back(D6::try_from(value).unwrap());
    }
    pub fn fix_d8(&mut self, value: u8) {
        self.d8_fixes.push_back(D8::try_from(value).unwrap());
    }
    pub fn fix_d8_direction(&mut self, direction: Direction) {
        self.d8_fixes.push_back(D8::from(direction));
    }
    pub fn fix_blockdice(&mut self, value: BlockDice) {
        self.blockdice_fixes.push_back(value);
    }
    pub fn is_empty(&self) -> bool {
        self.d6_fixes.is_empty() && self.d8_fixes.is_empty() && self.blockdice_fixes.is_empty()
    }
    pub fn assert_is_empty(&self) {
        assert!(
            self.is_empty(),
            "fixed dices are not empty: d6:{:?}, d8: {:?}, blockdice: {:?}",
            self.d6_fixes,
            self.d8_fixes,
            self.blockdice_fixes
        );
    }
}
#[derive(Derivative)]
#[derivative(PartialEq)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GameState {
    pub info: GameInfo,
    pub home: TeamState,
    pub away: TeamState,

    fielded_players: [Option<FieldedPlayer>; 22],
    dugout_players: [Option<DugoutPlayer>; 32],
    board: FullPitch<Option<PlayerID>>,
    pub ball: BallState,
    proc_stack: Vec<AnyProc>,
    pub available_actions: Box<AvailableActions>,
    pub rng_enabled: bool,
    pub fixes: FixedDice,

    #[serde(skip)]
    #[derivative(PartialEq = "ignore")]
    next_input: Option<ProcInput>,

    #[serde(skip, default = "ChaCha8Rng::from_entropy")]
    #[derivative(PartialEq = "ignore")]
    rng: ChaCha8Rng,

    log: Vec<String>,
    print_log: bool,
}

impl GameState {
    pub fn log(&mut self, s: String) {
        if self.print_log {
            println!("{}", s);
        }
        self.log.push(s);
    }
    pub fn get_log(&self) -> &Vec<String> {
        &self.log
    }
    pub fn set_logging_state(&mut self, state: bool) {
        self.print_log = state;
    }
    pub fn get_dugout(&self) -> impl Iterator<Item = &DugoutPlayer> {
        self.dugout_players.iter().flatten()
    }
    pub fn get_dugout_mut(&mut self) -> impl Iterator<Item = &mut DugoutPlayer> {
        self.dugout_players.iter_mut().flatten()
    }
    pub fn dugout_add_new_player(&mut self, player_stats: PlayerStats, place: DugoutPlace) {
        let id = match self
            .dugout_players
            .iter()
            .enumerate()
            .find(|(_, player)| player.is_none())
        {
            Some((id, _)) => id,
            None => panic!("Not room in gamestate of another dugout player!"),
        };
        self.dugout_players[id] = Some(DugoutPlayer {
            stats: player_stats,
            place,
            id,
        })
    }
    pub fn get_dugout_player(&self, id: DugoutPlayerID) -> Option<&DugoutPlayer> {
        self.dugout_players[id].as_ref()
    }
    pub fn get_dugout_player_mut(&mut self, id: DugoutPlayerID) -> Option<&mut DugoutPlayer> {
        self.dugout_players[id].as_mut()
    }

    pub fn field_dugout_player(&mut self, dugout_id: DugoutPlayerID, position: Position) {
        let DugoutPlayer { stats, place, .. } = self.dugout_players[dugout_id].take().unwrap();
        assert_eq!(place, DugoutPlace::Reserves, "Must field from reserves_box");
        self.add_new_player_to_field(stats, position).unwrap();
    }
    pub fn get_available_actions(&self) -> &AvailableActions {
        &self.available_actions
    }
    pub fn home_to_act(&self) -> bool {
        self.get_available_actions()
            .get_team()
            .map(|team| team == TeamType::Home)
            .unwrap_or(false)
    }
    pub fn away_to_act(&self) -> bool {
        self.get_available_actions()
            .get_team()
            .map(|team| team == TeamType::Away)
            .unwrap_or(false)
    }

    pub fn set_active_player(&mut self, id: PlayerID) {
        debug_assert!(self.get_player(id).is_ok());
        self.info.active_player = Some(id);
    }

    pub fn get_active_player(&self) -> Option<&FieldedPlayer> {
        self.info
            .active_player
            .and_then(|id| self.get_player(id).ok())
    }
    pub fn get_active_player_mut(&mut self) -> Option<&mut FieldedPlayer> {
        self.info
            .active_player
            .and_then(|id| self.get_mut_player(id).ok())
    }
    pub fn get_endzone_x(&self, team: TeamType) -> Coord {
        match team {
            TeamType::Home => 1,
            TeamType::Away => WIDTH_ - 2,
        }
    }
    pub fn set_seed(&mut self, state: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(state);
    }

    fn get_d3_roll(&mut self) -> D3 {
        match self.fixes.d3_fixes.pop_front() {
            Some(roll) => roll,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }
    fn get_d6_roll(&mut self) -> D6 {
        match self.fixes.d6_fixes.pop_front() {
            Some(roll) => roll,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }
    fn get_2d6_roll(&mut self) -> Sum2D6 {
        self.get_d6_roll() + self.get_d6_roll()
    }

    fn get_d8_roll(&mut self) -> D8 {
        match self.fixes.d8_fixes.pop_front() {
            Some(roll) => roll,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }

    fn get_coin_toss(&mut self) -> Coin {
        match self.fixes.coin_fixes.pop_front() {
            Some(fixed_toss) => fixed_toss,
            None => {
                assert!(self.rng_enabled);
                self.rng.gen()
            }
        }
    }

    pub fn get_best_kickoff_aim_for(&self, team: TeamType) -> Position {
        match team {
            TeamType::Home => Position::new((WIDTH_ / 4, HEIGHT_ / 2 - 1)),
            TeamType::Away => Position::new((WIDTH_ * 3 / 4, HEIGHT_ / 2 - 1)),
        }
    }
    fn get_block_dice_roll(&mut self) -> BlockDice {
        match self.fixes.blockdice_fixes.pop_front() {
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
    pub fn get_active_players_team(&self) -> Option<&TeamState> {
        self.info
            .active_player
            .and_then(|id| self.get_player(id).ok())
            .map(|player| self.get_team(player.stats.team))
    }

    pub fn get_active_players_team_mut(&mut self) -> Option<&mut TeamState> {
        self.get_mut_team_from_player(self.info.active_player.unwrap())
            .ok()
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
        // let xx = usize::try_from(x).unwrap();
        // let yy = usize::try_from(y).unwrap();
        // self.board[xx][yy]
        self.board[Position::new((x, y))]
    }
    pub fn get_player_at_coord(&self, x: Coord, y: Coord) -> Option<&FieldedPlayer> {
        match self.get_player_id_at_coord(x, y) {
            None => None,
            Some(id) => Some(self.get_player(id).unwrap()),
            //above unwrap is safe for bad input. If it panics it's an interal logical error!
        }
    }

    pub fn get_player_unsafe(&self, id: PlayerID) -> &FieldedPlayer {
        self.fielded_players[id].as_ref().unwrap()
    }

    pub fn get_mut_player_unsafe(&mut self, id: PlayerID) -> &mut FieldedPlayer {
        self.fielded_players[id].as_mut().unwrap()
    }

    pub fn get_player(&self, id: PlayerID) -> Result<&FieldedPlayer> {
        match &self.fielded_players[id] {
            Some(player) => Ok(player),
            None => Err(Box::new(InvalidPlayerId { id })),
        }
    }

    pub fn get_adj_positions(&self, position: Position) -> impl Iterator<Item = Position> {
        debug_assert!(!position.is_out());
        Direction::all_directions_iter().map(move |&direction| position + direction)
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
    pub fn get_catch_target(&self, id: PlayerID) -> Result<D6Target> {
        let player = self.get_player(id)?;
        let mut target = player.ag_target();
        let team = player.stats.team;
        target.add_modifer(
            -(self
                .get_adj_players(player.position)
                .filter(|player_| player_.stats.team != team && player_.has_tackle_zone())
                .count() as i8),
        );

        if let Weather::Rain = self.info.weather {
            target.add_modifer(-1);
        }
        Ok(target)
    }

    pub fn get_ball_position(&self) -> Option<Position> {
        match self.ball {
            BallState::OffPitch => None,
            BallState::OnGround(pos) => Some(pos),
            BallState::Carried(id) => Some(self.get_player(id).unwrap().position),
            BallState::InAir(pos) => Some(pos),
        }
    }

    pub fn get_tz_on_except_from_id(&self, id: PlayerID, except_from_id: PlayerID) -> u8 {
        let player = self.get_player_unsafe(id);
        let team = player.stats.team;

        self.get_adj_players(player.position)
            .filter(|adj_player| {
                adj_player.stats.team != team
                    && adj_player.has_tackle_zone()
                    && adj_player.id != except_from_id
            })
            .count() as u8
    }

    pub fn get_tz_on(&self, id: PlayerID) -> u8 {
        let player = self.get_player_unsafe(id);
        let team = player.stats.team;

        self.get_adj_players(player.position)
            .filter(|adj_player| adj_player.stats.team != team && adj_player.has_tackle_zone())
            .count() as u8
    }

    pub fn get_blockdices(&self, attacker: PlayerID, defender: PlayerID) -> NumBlockDices {
        let attacker_pos = self.get_player_unsafe(attacker).position;
        self.get_blockdices_from(attacker, attacker_pos, defender)
    }

    pub fn get_blockdices_from(
        &self,
        attacker: PlayerID,
        attacker_pos: Position,
        defender: PlayerID,
    ) -> NumBlockDices {
        let attr = self.get_player_unsafe(attacker);
        let defr = self.get_player_unsafe(defender);

        debug_assert_ne!(attr.stats.team, defr.stats.team);
        debug_assert_eq!(attacker_pos.distance_to(&defr.position), 1);
        // debug_assert!(attr.has_tackle_zone());
        debug_assert_eq!(defr.status, PlayerStatus::Up);

        let mut attr_str = attr.stats.str_;
        let mut defr_str = defr.stats.str_;

        attr_str += self
            .get_adj_players(defr.position)
            .filter(|attr_assister| {
                attr_assister.id != attr.id
                    && attr_assister.stats.team == attr.stats.team
                    && attr_assister.has_tackle_zone()
                    && self.get_tz_on_except_from_id(attr_assister.id, defr.id) == 0
                //what is guard anyway?
            })
            .count() as u8;

        defr_str += self
            .get_adj_players(attacker_pos)
            .filter(|defr_assister| {
                defr_assister.id != defr.id
                    && defr_assister.stats.team == defr.stats.team
                    && defr_assister.has_tackle_zone()
                    && self.get_tz_on_except_from_id(defr_assister.id, attr.id) == 0
                //what is guard anyway?
            })
            .count() as u8;

        if attr_str > 2 * defr_str {
            NumBlockDices::Three
        } else if attr_str > defr_str {
            NumBlockDices::Two
        } else if attr_str == defr_str {
            NumBlockDices::One
        } else if 2 * attr_str < defr_str {
            NumBlockDices::ThreeUphill
        } else {
            NumBlockDices::TwoUphill
        }
    }
    pub fn get_line_of_scrimage_x(&self, team: TeamType) -> Coord {
        match team {
            TeamType::Home => LINE_OF_SCRIMMAGE_HOME_X,
            TeamType::Away => LINE_OF_SCRIMMAGE_AWAY_X,
        }
    }
    pub fn move_player(&mut self, id: PlayerID, new_pos: Position) -> Result<()> {
        let old_pos = self.get_player(id)?.position;
        if let Some(occupied_id) = self.board[new_pos] {
            panic!(
                "Tried to move {}, to {:?} but it was already occupied by {}",
                id, new_pos, occupied_id
            );
            //return Err(Box::new(IllegalMovePosition{position: new_pos} ))
        }
        self.board[old_pos] = None;
        self.get_mut_player(id)?.position = new_pos;
        self.board[new_pos] = Some(id);
        Ok(())
    }
    pub fn get_players_on_pitch(&self) -> impl Iterator<Item = &FieldedPlayer> {
        self.fielded_players.iter().filter_map(|x| x.as_ref())
    }
    pub fn get_players_on_pitch_mut(&mut self) -> impl Iterator<Item = &mut FieldedPlayer> {
        self.fielded_players.iter_mut().filter_map(|x| x.as_mut())
    }
    pub fn get_players_on_pitch_in_team(
        &self,
        team: TeamType,
    ) -> impl Iterator<Item = &FieldedPlayer> {
        self.get_players_on_pitch()
            .filter(move |p| p.stats.team == team)
    }
    pub fn add_new_player_to_field(
        &mut self,
        player_stats: PlayerStats,
        position: Position,
    ) -> Result<PlayerID> {
        if self.board[position].is_some() {
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

        self.board[position] = Some(id);
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

    pub fn unfield_player(&mut self, id: PlayerID, place: DugoutPlace) -> Result<()> {
        if let BallState::Carried(carrier_id) = self.ball {
            assert_ne!(carrier_id, id);
        }
        if matches!(self.info.active_player, Some(active_id) if active_id == id) {
            self.info.active_player = None;
        }

        let FieldedPlayer {
            stats, position, ..
        } = self.fielded_players[id].take().unwrap();

        self.dugout_add_new_player(stats, place);

        self.board[position] = None;
        Ok(())
    }

    pub fn unfield_all_players(&mut self) -> Result<()> {
        #[allow(clippy::needless_collect)]
        let player_id_on_pitch: Vec<PlayerID> = self
            .get_players_on_pitch()
            .map(|player| player.id)
            .collect();

        player_id_on_pitch
            .into_iter()
            .for_each(|id| self.unfield_player(id, DugoutPlace::Reserves).unwrap());
        Ok(())
    }
    pub fn clear_all_players(&mut self) -> Result<()> {
        self.unfield_all_players().unwrap();
        self.dugout_players = Default::default();
        Ok(())
    }
    pub fn micro_step(&mut self, action: Option<Action>) -> Result<()> {
        let proc_input: ProcInput = {
            if self.available_actions.is_empty() {
                debug_assert!(action.is_none());
                self.next_input.take().unwrap_or(ProcInput::Nothing)
            } else {
                match action {
                    None => return Err(Box::new(MissingActionError {})),
                    Some(action) if !self.is_legal_action(&action) => {
                        return Err(Box::new(IllegalActionError { action }))
                    }
                    Some(action) => ProcInput::Action(action),
                }
            }
        };
        let mut top_proc = self
            .proc_stack
            .pop()
            .ok_or_else(|| Box::new(EmptyProcStackError {}))?;

        let proc_return = top_proc.step(self, proc_input);
        self.available_actions = Default::default();
        self.next_input = match proc_return {
            ProcState::NotDoneNewProcs(new_procs) => {
                self.proc_stack.push(top_proc);
                self.proc_stack.extend(new_procs);
                Some(ProcInput::Nothing)
            }
            ProcState::DoneNewProcs(new_procs) => {
                self.proc_stack.extend(new_procs);
                Some(ProcInput::Nothing)
            }
            ProcState::NotDoneNew(new_proc) => {
                self.proc_stack.push(top_proc);
                self.proc_stack.push(new_proc);
                Some(ProcInput::Nothing)
            }
            ProcState::DoneNew(new_proc) => {
                self.proc_stack.push(new_proc);
                Some(ProcInput::Nothing)
            }
            ProcState::NotDone => {
                self.proc_stack.push(top_proc);
                Some(ProcInput::Nothing)
            }
            ProcState::Done => Some(ProcInput::Nothing),
            ProcState::NeedAction(aa) => {
                self.available_actions = aa;
                self.proc_stack.push(top_proc);
                None
            }
            ProcState::NeedRoll(requested_roll) => {
                self.proc_stack.push(top_proc);
                let result = self.get_roll_result(requested_roll);
                Some(ProcInput::Roll(result))
            }
        };

        Ok(())
    }

    pub fn step(&mut self, action: Action) -> Result<()> {
        let mut top_proc = self
            .proc_stack
            .pop()
            .ok_or_else(|| Box::new(EmptyProcStackError {}))?;

        let mut proc_input: ProcInput = {
            if self.available_actions.is_empty() {
                self.log(format!("STEPPING: {:?}", top_proc));
                ProcInput::Nothing
            } else if !self.is_legal_action(&action) {
                return Err(Box::new(IllegalActionError { action }));
            } else {
                self.log(format!("STEPPING: {:?}\n  action={:?}", top_proc, action));
                ProcInput::Action(action)
            }
        };

        let mut top_proc_state: ProcState = top_proc.step(self, proc_input);

        loop {
            if self.info.game_over {
                break;
            }
            self.log(format!("  result:   {:?}", top_proc_state));
            proc_input = ProcInput::Nothing;
            match top_proc_state {
                ProcState::NotDoneNewProcs(mut new_procs) => {
                    self.proc_stack.push(top_proc);
                    top_proc = new_procs.pop().unwrap();
                    self.proc_stack.extend(new_procs);
                }
                ProcState::DoneNewProcs(mut new_procs) => {
                    top_proc = new_procs.pop().unwrap();
                    self.proc_stack.extend(new_procs);
                }
                ProcState::NotDoneNew(new_proc) => {
                    self.proc_stack.push(top_proc);
                    top_proc = new_proc;
                }
                ProcState::DoneNew(new_proc) => {
                    top_proc = new_proc;
                }
                ProcState::NotDone => (),
                ProcState::Done => {
                    top_proc = self
                        .proc_stack
                        .pop()
                        .ok_or_else(|| Box::new(EmptyProcStackError {}))?;
                }
                ProcState::NeedAction(aa) => {
                    self.available_actions = aa;
                    self.proc_stack.push(top_proc);
                    break;
                }
                ProcState::NeedRoll(requested_roll) => {
                    let result = self.get_roll_result(requested_roll);
                    proc_input = ProcInput::Roll(result);
                }
            };

            if matches!(proc_input, ProcInput::Roll(_)) {
                self.log(format!("STEPPING again with:   {:?}", proc_input));
            } else {
                self.log(format!("STEPPING: {:?}", top_proc));
            }

            top_proc_state = top_proc.step(self, proc_input);
        }
        debug_assert!(!self.available_actions.is_empty() || self.info.game_over);
        Ok(())
    }

    fn get_roll_result(&mut self, requested_roll: RequestedRoll) -> RollResult {
        match requested_roll {
            RequestedRoll::D6 => RollResult::D6(self.get_d6_roll()),
            RequestedRoll::D6PassFail(target) => {
                if target.is_success(self.get_d6_roll()) {
                    RollResult::Pass
                } else {
                    RollResult::Fail
                }
            }
            RequestedRoll::D6ThreeOutcomes(low_target, high_target) => {
                let roll = self.get_d6_roll();
                if high_target.is_success(roll) {
                    RollResult::Pass
                } else if low_target.is_success(roll) {
                    RollResult::MiddleOutcome
                } else {
                    RollResult::Fail
                }
            }
            RequestedRoll::Sum2D6 => RollResult::Sum2D6(self.get_2d6_roll()),
            RequestedRoll::Sum2D6PassFail(target) => {
                if target.is_success(self.get_2d6_roll()) {
                    RollResult::Pass
                } else {
                    RollResult::Fail
                }
            }
            RequestedRoll::Sum2D6ThreeOutcomes(low_target, high_target) => {
                let roll = self.get_2d6_roll();
                if high_target.is_success(roll) {
                    RollResult::Pass
                } else if low_target.is_success(roll) {
                    RollResult::MiddleOutcome
                } else {
                    RollResult::Fail
                }
            }
            RequestedRoll::D8 => RollResult::D8(self.get_d8_roll()),
            RequestedRoll::Coin => RollResult::Coin(self.get_coin_toss()),
            RequestedRoll::Deviate => RollResult::Deviate(self.get_d6_roll(), self.get_d8_roll()),
            RequestedRoll::FoulArmor(target) => {
                let roll1 = self.get_d6_roll();
                let roll2 = self.get_d6_roll();
                RollResult::FoulArmor {
                    broken: target.is_success(roll1 + roll2),
                    ejected: roll1 == roll2,
                }
            }
            RequestedRoll::FoulInjury(ko_target, cas_target) => {
                let roll1 = self.get_d6_roll();
                let roll2 = self.get_d6_roll();
                let outcome = if cas_target.is_success(roll1 + roll2) {
                    InjuryOutcome::Casualty
                } else if ko_target.is_success(roll1 + roll2) {
                    InjuryOutcome::KO
                } else {
                    InjuryOutcome::Stunned
                };
                RollResult::FoulInjury {
                    outcome,
                    ejected: roll1 == roll2,
                }
            }
            RequestedRoll::ThrowIn => RollResult::ThrowIn {
                direction: self.get_d3_roll(),
                distance: self.get_2d6_roll(),
            },
            RequestedRoll::BlockDice(num_dices) => {
                let mut dices: [Option<BlockDice>; 3] = [None, None, None];
                for dice in dices.iter_mut().take(u8::from(num_dices) as usize) {
                    *dice = Some(self.get_block_dice_roll());
                }
                RollResult::BlockDice(dices)
            }
            RequestedRoll::Scatter => {
                RollResult::Scatter(self.get_d8_roll(), self.get_d8_roll(), self.get_d8_roll())
            }
        }
    }

    pub fn is_legal_action(&self, action: &Action) -> bool {
        /*let mut top_proc = self.proc_stack.pop().unwrap(); //TODO: remove these three lines at some point!
        debug_assert_eq!(top_proc.available_actions(self), self.available_actions);
        self.proc_stack.push(top_proc);
         */
        self.available_actions.is_legal_action(*action)
    }

    pub fn is_setup_legal(&self, team: TeamType) -> bool {
        let mut north_wing = 0;
        let mut south_wing = 0;
        let mut line_of_scrimage = 0;
        let num_players_on_pitch = self.get_players_on_pitch_in_team(team).count();
        let num_players_on_bench = self
            .get_dugout()
            .filter(|player| player.stats.team == team && player.place == DugoutPlace::Reserves)
            .count();
        let num_available_players = num_players_on_bench + num_players_on_pitch;
        let min_people_on_pitch = 11.min(num_available_players);
        let min_people_on_scrimage = 3.min(num_available_players);

        if num_players_on_pitch < min_people_on_pitch || num_players_on_pitch > 11 {
            return false;
        }
        let line_of_scrimage_x = self.get_line_of_scrimage_x(team);

        for pos in self.get_players_on_pitch_in_team(team).map(|p| p.position) {
            if pos.is_out()
                || (team == TeamType::Home && pos.x < line_of_scrimage_x)
                || (team == TeamType::Away && pos.x > line_of_scrimage_x)
            {
                return false;
            }

            if pos.x == line_of_scrimage_x && LINE_OF_SCRIMMAGE_Y_RANGE.contains(&pos.y) {
                line_of_scrimage += 1;
            } else if SOUTH_WING_Y_RANGE.contains(&pos.y) {
                south_wing += 1;
            } else if NORTH_WING_Y_RANGE.contains(&pos.y) {
                north_wing += 1;
            }
        }
        north_wing <= 2 && south_wing <= 2 && line_of_scrimage >= min_people_on_scrimage
    }

    pub fn step_simple(&mut self, action: SimpleAT) {
        self.step(Action::Simple(action)).unwrap();
        self.fixes.assert_is_empty();
    }

    pub fn step_positional(&mut self, action: PosAT, position: Position) {
        self.step(Action::Positional(action, position)).unwrap();
        self.fixes.assert_is_empty();
    }

    pub fn get_interception_positions(
        from: Position,
        to: Position,
    ) -> impl Iterator<Item = Position> {
        let fr_y = from.y as i16;
        let to_y = to.y as i16;
        let fr_x = from.x as i16;
        let to_x = to.x as i16;
        let max_x = max(fr_x, to_x) + 1;
        let min_x = min(fr_x, to_x) - 1;
        let max_y = max(fr_y, to_y) + 1;
        let min_y = min(fr_y, to_y) - 1;
        let dx = to_x - fr_x;
        let dy = to_y - fr_y;
        let distance_squared = dx * dx + dy * dy;
        let distance = (distance_squared as f32).sqrt();

        ((min_x)..=(max_x))
            .cartesian_product((min_y)..=(max_y))
            .filter(move |(x, y)| ((*x - fr_x).pow(2) + (*y - fr_y).pow(2)) <= distance_squared)
            .filter(move |(x, y)| ((*x - to_x).pow(2) + (*y - to_y).pow(2)) <= distance_squared)
            .filter(move |(x, y)| {
                1.2 >= ((to_x - x) * dy - (to_y - y) * dx).abs() as f32 / distance
            })
            .map(|(x, y)| Position::new((x as i8, y as i8)))
            .filter(|pos| !pos.is_out())
            .filter(move |pos| *pos != from && *pos != to)
    }

    pub fn get_intercepters(
        &self,
        team: TeamType,
        from: Position,
        to: Position,
    ) -> Vec<(Position, D6Target)> {
        // TODO: thos function needs tests!!!
        GameState::get_interception_positions(from, to)
            .filter_map(|pos| {
                self.get_player_at(pos)
                    .filter(|p| p.stats.team == team && p.can_catch())
                    .and_then(|p| self.get_catch_target(p.id).ok())
                    .map(|target| (pos, target))
            })
            .collect::<Vec<(Position, D6Target)>>()
    }

    pub fn get_pass_modifier(&self, id: usize, from: Position, to: Position) -> Option<i8> {
        // TODO: move this whole function to pathing and cache it there.
        const MATRIX: [[i8; 14]; 14] = [
            [8, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 9],
            [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 9],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 9],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 9, 9],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 9, 9],
            [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 9, 9, 9],
            [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 9, 9, 9],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 9, 9, 9, 9],
            [2, 2, 2, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9],
            [3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9],
            [3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ];
        // 8 - passing to oneself, not possible
        // 9 - hail mary pass - skill not implemented
        // 0 - quick pass
        // 1 - short pass
        // 2 - long pass
        // 3 - long bomb

        let delta = to - from;
        let (dx, dy) = (
            delta.dx.unsigned_abs() as usize,
            delta.dy.unsigned_abs() as usize,
        );
        let distance_modifier = MATRIX.get(dx)?.get(dy)?;
        if *distance_modifier == 8 {
            panic!(
                "Passing to oneself is not possible: from {} to {}",
                from, to
            );
        } else if *distance_modifier == 9 {
            return None;
        }

        let Some(team) = self.get_player(id).ok().map(|p| p.stats.team) else {
            //return None;
            panic!("Player not found");
        };
        let tackle_zones = self
            .get_adj_players(from)
            .filter(|adj_p| adj_p.stats.team != team && adj_p.has_tackle_zone())
            .count() as i8;
        // TODO: weather effect
        let sum_modifiers = -tackle_zones - *distance_modifier;
        Some(sum_modifiers)
    }
    pub fn get_pass_target(&self, id: usize, from: Position, to: Position) -> Option<D6Target> {
        let Some(player) = self.get_player(id).ok() else {
            //return None;
            panic!("Player not found");
        };
        let mut target = player.pass_target();
        let modifiers = self.get_pass_modifier(id, from, to)?;
        target.add_modifer(modifiers);
        Some(target)
    }
}

#[cfg(test)]
mod gamestate_tests {
    use itertools::Itertools;
    use serde_json;

    use crate::{
        core::{
            dices::D6,
            gamestate::{BuilderState, GameState},
            model::{
                BallState, DugoutPlace, PlayerStats, Position, Result, TeamType, HEIGHT_, WIDTH,
                WIDTH_,
            },
        },
        standard_state,
    };
    use std::{collections::HashSet, io::Write, iter::repeat_with};

    use super::GameStateBuilder;

    #[test]
    fn symmetric_interception_positions() {
        for (dx, dy) in (0..14).cartesian_product(0..14) {
            let from = Position::new((3, 3));
            let to = from + (dx, dy);
            let to_from = GameState::get_interception_positions(from, to);
            let from_to = GameState::get_interception_positions(to, from);
            assert_eq!(
                to_from.collect::<HashSet<_>>(),
                from_to.collect::<HashSet<_>>()
            );
        }
    }
    #[test]
    fn interception_positions() {
        let correct_thing = [
            [
                ".......................",
                ".......................",
                ".....oo.................",
                "....XooX...............",
                ".....oo................",
                ".......................",
                ".......................",
            ],
            [
                ".......................",
                ".......................",
                ".......................",
                "....oooX................",
                "....Xooo..............",
                ".....................",
                ".......................",
            ],
            [
                ".......................",
                ".......................",
                ".....ooX...............",
                "....oooo................",
                "....Xoo...............",
                ".....................",
                ".......................",
            ],
            [
                ".......................",
                "......oX...............",
                ".....ooo...............",
                "....ooo.................",
                "....Xo................",
                ".....................",
                ".......................",
            ],
        ];
        for s in correct_thing {
            let mut to_from = Vec::new();
            let mut correct_intercepters = HashSet::new();
            for (y, line) in s.iter().enumerate() {
                for (x, c) in line.chars().enumerate() {
                    let p = Position::new((x as i8 + 2, y as i8 + 2));
                    if c == 'X' {
                        to_from.push(p);
                    } else if c == 'o' {
                        correct_intercepters.insert(p);
                    } else {
                        assert_eq!(c, '.');
                    }
                }
            }

            assert!(to_from.len() == 2);
            let from = to_from.pop().unwrap();
            let to = to_from.pop().unwrap();

            let calc_intercepters =
                GameState::get_interception_positions(from, to).collect::<HashSet<Position>>();
            // assert_eq!(calc_intercepters, correct_intercepters);
            if calc_intercepters != correct_intercepters {
                // create ss, a clone of s.
                println!("from: {:?}, to: {:?}", from, to);
                println!("calc_intercepters: {:?}", calc_intercepters);
                let mut ss: Vec<Vec<char>> = (0..HEIGHT_).map(|_| vec!['.'; WIDTH]).collect();
                let correctly_addeds = correct_intercepters.intersection(&calc_intercepters);
                println!("correctly_addeds: {:?}", correctly_addeds);
                let wrongly_addeds = calc_intercepters.difference(&correct_intercepters);
                println!("wrongly_addeds: {:?}", wrongly_addeds);
                for wrongly_added in wrongly_addeds {
                    assert!(!wrongly_added.is_out());
                    let (x, y) = wrongly_added.to_usize().unwrap();
                    ss[y][x] = 'a';
                }
                let wrongly_missings = correct_intercepters.difference(&calc_intercepters);
                println!("wrongly_missings: {:?}", wrongly_missings);
                for wrongly_missing in wrongly_missings {
                    assert!(!wrongly_missing.is_out());
                    let (x, y) = wrongly_missing.to_usize().unwrap();
                    ss[y][x] = 'm';
                }
                let (x, y) = from.to_usize().unwrap();
                ss[y][x] = 'X';
                let (x, y) = to.to_usize().unwrap();
                ss[y][x] = 'X';

                let error_strs: Vec<String> = ss
                    .iter()
                    .enumerate()
                    .map(|(i, line)| {
                        format!("{}: {}", i, line.iter().collect::<String>()).to_string()
                    })
                    .filter(|s| s.contains('a') || s.contains('m') || s.contains('X'))
                    .collect();
                let error_str: String = error_strs.join("\n");
                assert_eq!(
                    calc_intercepters,
                    correct_intercepters,
                    "\n{}\n\nshould have been\n{}\n",
                    error_str,
                    s.iter().join("\n")
                );
            }
        }
    }

    #[test]
    fn kickoff_position() {
        let state = GameStateBuilder::new().build();
        assert_eq!(
            state.get_best_kickoff_aim_for(crate::core::model::TeamType::Home),
            Position::new((7, 7))
        );
        assert_eq!(
            state.get_best_kickoff_aim_for(crate::core::model::TeamType::Away),
            Position::new((21, 7))
        );
    }

    #[test]
    fn test_unfield_all_players() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(1, 2), (2, 2), (3, 1)])
            .add_away_players(&[(5, 2), (5, 5), (2, 3)])
            .add_ball((3, 2))
            .build();
        assert_eq!(state.get_players_on_pitch().count(), 6);
        state.unfield_all_players().unwrap();

        assert_eq!(state.get_players_on_pitch().count(), 0);
    }
    #[test]
    fn test_clear_all_players() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(1, 2), (2, 2), (3, 1)])
            .add_away_players(&[(5, 2), (5, 5), (2, 3)])
            .add_ball((3, 2))
            .build();
        assert_eq!(state.get_dugout().count(), 0);
        assert_eq!(state.get_players_on_pitch().count(), 6);

        state.clear_all_players().unwrap();

        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert_eq!(state.get_dugout().count(), 0);
    }
    #[test]
    fn test_build_game_custom_turn() {
        let state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 3 })
            .build();
        assert_eq!(state.info.home_turn, 3);
        assert_eq!(state.info.away_turn, 2);
    }

    #[test]
    fn test_kickoff_game_custom_turn() {
        let state = GameStateBuilder::new()
            .set_state(BuilderState::Kickoff { turn: 7 })
            .build();
        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 6);
    }
    #[test]
    fn state_from_str() {
        let mut field = "".to_string();
        field += " aa\n";
        field += " Aa\n";
        field += "h  \n";
        let first_pos = Position::new((5, 1));
        let state = GameStateBuilder::new().add_str(first_pos, &field).build();
        assert_eq!(
            state
                .get_player_at(Position::new((5, 3)))
                .unwrap()
                .stats
                .team,
            TeamType::Home
        );

        assert_eq!(
            state
                .get_player_at(Position::new((6, 2)))
                .unwrap()
                .stats
                .team,
            TeamType::Away
        );

        let id = state.get_player_id_at_coord(6, 2).unwrap();
        assert_eq!(state.ball, BallState::Carried(id));
    }

    #[test]
    fn player_unique_id_and_correct_positions() {
        let state = standard_state();

        let mut ids = HashSet::new();
        for x in 0..WIDTH_ {
            for y in 0..HEIGHT_ {
                let pos = Position::new((x, y));
                if let Some(player) = state.get_player_at(pos) {
                    assert_eq!(player.position, pos);
                    assert!(ids.insert(player.id));
                }
            }
        }
        assert_eq!(0, ids.into_iter().filter(|id| *id >= 22).count());
    }

    #[test]
    fn adjescent() {
        let state = standard_state();
        let num_adj = state.get_adj_players(Position::new((2, 2))).count();
        assert_eq!(num_adj, 3);
    }

    #[test]
    fn mutate_player() {
        let mut state = standard_state();

        assert!(!(state.get_player(0).unwrap().used));
        state.get_mut_player(0).unwrap().used = true;
        assert!(state.get_player(0).unwrap().used);
    }

    #[test]
    fn move_player() -> Result<()> {
        let mut state = standard_state();
        let id = 1;
        let old_pos = Position::new((2, 2));
        let new_pos = Position::new((10, 10));

        assert_eq!(state.get_player_id_at(old_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, old_pos);
        assert!(state.get_player_id_at(new_pos).is_none());

        state.move_player(id, new_pos)?;

        assert!(state.get_player_id_at(old_pos).is_none());
        assert_eq!(state.get_player_id_at(new_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, new_pos);
        Ok(())
    }
    #[test]
    fn field_a_player() -> Result<()> {
        let mut state = standard_state();
        let player_stats = PlayerStats::new_lineman(TeamType::Home);
        let position = Position::new((10, 10));

        assert!(state.get_player_id_at(position).is_none());

        let id = state
            .add_new_player_to_field(player_stats, position)
            .unwrap();

        assert_eq!(state.get_player_id_at(position), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, position);

        state.unfield_player(id, DugoutPlace::Reserves)?;

        assert!(state.get_player_id_at(position).is_none());
        Ok(())
    }
    #[test]
    fn rng_seed_in_gamestate() -> Result<()> {
        let mut state = standard_state();
        state.rng_enabled = true;
        let seed = 5;
        state.set_seed(seed);

        fn get_random_rolls(state: &mut GameState) -> Vec<D6> {
            repeat_with(|| state.get_d6_roll()).take(200).collect()
        }

        let numbers: Vec<D6> = get_random_rolls(&mut state);
        let different_numbers = get_random_rolls(&mut state);
        assert_ne!(numbers, different_numbers);

        state.set_seed(seed);
        let same_numbers = get_random_rolls(&mut state);

        assert_eq!(numbers, same_numbers);

        Ok(())
    }
    #[test]
    fn serialize_gamestate() {
        let state = standard_state();

        let serialized = serde_json::to_string(&state).unwrap();
        let mut file = std::fs::File::create("serialized_test.json").unwrap();
        file.write_all(serialized.as_bytes()).unwrap();

        let json_str = std::fs::read_to_string("serialized_test.json").unwrap();
        let deserialized: GameState = serde_json::from_str(&json_str).unwrap();
        assert_eq!(state, deserialized);
        std::fs::remove_file("serialized_test.json").unwrap();
    }
}

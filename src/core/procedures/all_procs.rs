use crate::core::{dices::D6, model};
use model::*;
use rand::Rng;
use std::{iter::repeat_with, ops::RangeInclusive};

use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::procedures::{ball_procs, block_procs};
use crate::core::table::*;

use crate::core::{
    dices::{BlockDice, D6Target, RollTarget, Sum2D6, Sum2D6Target},
    gamestate::GameState,
    pathing::{event_ends_player_action, CustomIntoIter, NodeIterator, PathFinder, PathingEvent},
};

#[derive(Debug)]
pub struct Half {
    pub half: u8,
    pub started: bool,
    pub kicking_this_half: TeamType,
    pub kickoff: Option<TeamType>,
}
impl Half {
    pub fn new(half: u8) -> Box<Half> {
        debug_assert!(half == 1 || half == 2);
        Box::new(Half {
            half,
            started: false,
            kicking_this_half: TeamType::Home,
            kickoff: None,
        })
    }
    fn do_kickoff(&mut self, kicking_team: TeamType, game_state: &mut GameState) -> ProcState {
        //SCORING IN THE OPPONENT’S TURN
        // In some rare cases a team will score a touchdown in the
        // opponent’s turn. For example, a player holding the ball could be
        // pushed into the End Zone by a block. If one of your players is
        // holding the ball in the opposing team's End Zone at any point
        // during your opponent's turn then your team scores a touchdown
        // immediately, but must move their Turn marker one space along
        // the Turn track to represent the extra time the players spend
        // celebrating this unusual method of scoring!

        game_state.info.kicking_this_drive = kicking_team;

        let procs: Vec<Box<dyn Procedure>> = vec![
            Kickoff::new(),
            Setup::new(kicking_team),
            Setup::new(other_team(kicking_team)),
            KOWakeUp::new(),
        ];

        game_state.ball = BallState::OffPitch;

        #[allow(clippy::needless_collect)]
        let player_id_on_pitch: Vec<PlayerID> = game_state
            .get_players_on_pitch()
            .map(|player| player.id)
            .collect();

        player_id_on_pitch.into_iter().for_each(|id| {
            game_state
                .unfield_player(id, DugoutPlace::Reserves)
                .unwrap()
        });
        ProcState::NotDoneNewProcs(procs)
    }
}

impl Procedure for Half {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let info = &mut game_state.info;
        if !self.started {
            self.started = true;
            info.half = self.half;
            info.home_turn = 0;
            info.away_turn = 0;
            self.kicking_this_half = {
                if self.half == 1 {
                    info.kicking_first_half
                } else {
                    other_team(info.kicking_first_half)
                }
            };
            self.kickoff = Some(self.kicking_this_half);
        } else {
            self.kickoff = info.kickoff_by_team.take();
        }

        if info.home_turn == 8 && info.away_turn == 8 {
            return ProcState::Done;
        }

        if let Some(team) = self.kickoff {
            self.kickoff = None;
            return self.do_kickoff(team, game_state);
        }

        let next_team: TeamType = if info.home_turn == info.away_turn {
            self.kicking_this_half
        } else {
            other_team(self.kicking_this_half)
        };

        match next_team {
            TeamType::Home => info.home_turn += 1,
            TeamType::Away => info.away_turn += 1,
        }

        info.team_turn = next_team;
        info.handoff_available = true;
        info.blitz_available = true;
        info.foul_available = true;
        info.pass_available = true;
        info.turnover = false;

        game_state
            .get_players_on_pitch_mut()
            .filter(|p| p.stats.team == next_team && p.status != PlayerStatus::Stunned)
            .for_each(|p| p.used = false);

        ProcState::NotDoneNewProcs(vec![TurnStunned::new(), Turn::new(next_team)])
    }
}

#[derive(Debug)]
pub struct TurnStunned {}
impl TurnStunned {
    pub fn new() -> Box<TurnStunned> {
        Box::new(TurnStunned {})
    }
}
impl Procedure for TurnStunned {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let team = game_state.info.team_turn;
        let active_id = game_state.info.active_player.unwrap_or(999); // shall not turn active id, since they stunned themselves
        game_state
            .get_players_on_pitch_mut()
            .filter(|p| {
                p.stats.team == team && p.status == PlayerStatus::Stunned && p.id != active_id
            })
            .for_each(|p| p.status = PlayerStatus::Down);
        ProcState::Done
    }
}

#[derive(Debug)]
pub struct Turn {
    pub team: TeamType,
}
impl Turn {
    pub fn new(team: TeamType) -> Box<Turn> {
        Box::new(Turn { team })
    }
    fn available_actions(&mut self, game_state: &GameState) -> Box<AvailableActions> {
        let mut aa = AvailableActions::new(self.team);

        let positions: Vec<Position> = game_state
            .get_players_on_pitch_in_team(self.team)
            .filter(|p| !p.used)
            .map(|p| p.position)
            .collect();

        let block_positions: Vec<Position> = positions
            .iter()
            .filter(|&&pos| {
                game_state.get_adj_players(pos).any(|adj_player| {
                    adj_player.status == PlayerStatus::Up && adj_player.stats.team != self.team
                })
            })
            .copied()
            .collect();
        aa.insert_positional(PosAT::StartBlock, block_positions);
        if game_state.info.handoff_available {
            aa.insert_positional(PosAT::StartHandoff, positions.clone());
        }

        if game_state.info.blitz_available {
            aa.insert_positional(PosAT::StartBlitz, positions.clone());
        }

        if game_state.info.foul_available {
            aa.insert_positional(PosAT::StartFoul, positions.clone());
        }

        if game_state.info.pass_available {
            aa.insert_positional(PosAT::StartPass, positions.clone());
        }

        aa.insert_positional(PosAT::StartMove, positions);
        aa.insert_simple(SimpleAT::EndTurn);
        aa
    }
}
impl Procedure for Turn {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if let Some(id) = game_state.info.handle_td_by {
            //todo, set internal state to kickoff next (or if it was the last turn return done )
            game_state.info.handle_td_by = None;
            return ProcState::NotDoneNew(ball_procs::Touchdown::new(id));
        }

        if game_state.info.kickoff_by_team.is_some() || game_state.info.turnover {
            return ProcState::Done;
        }

        game_state.info.active_player = None;
        game_state.info.player_action_type = None;
        if action.is_none() {
            return ProcState::NeedAction(self.available_actions(game_state));
        }

        if let Some(Action::Positional(at, position)) = action {
            game_state.set_active_player(game_state.get_player_id_at(position).unwrap());
            let info = &mut game_state.info;
            info.player_action_type = Some(at);
            match at {
                PosAT::StartMove => (),
                PosAT::StartHandoff => info.handoff_available = false,
                PosAT::StartFoul => info.foul_available = false,
                PosAT::StartBlitz => info.blitz_available = false,
                PosAT::StartBlock => {
                    return ProcState::NotDoneNew(block_procs::BlockAction::new());
                }
                _ => unreachable!(),
            }
            ProcState::NotDoneNew(MoveAction::new(info.active_player.unwrap()))
        } else if let Some(Action::Simple(SimpleAT::EndTurn)) = action {
            ProcState::Done
        } else {
            unreachable!()
        }
    }
}

fn proc_from_roll(roll: PathingEvent, active_player: PlayerID) -> Box<dyn Procedure> {
    match roll {
        PathingEvent::Dodge(target) => DodgeProc::new(active_player, target),
        PathingEvent::GFI(target) => GfiProc::new(active_player, target),
        PathingEvent::Pickup(target) => ball_procs::PickupProc::new(active_player, target),
        PathingEvent::Block(id, dices) => block_procs::Block::new(dices, id),
        PathingEvent::Handoff(id, target) => ball_procs::Catch::new(id, target),
        PathingEvent::Touchdown(id) => ball_procs::Touchdown::new(id),
        PathingEvent::Foul(victim, target) => {
            block_procs::Armor::new_foul(victim, target, active_player)
        }
        PathingEvent::StandUp => StandUp::new(active_player),
    }
}

#[derive(Debug)]
enum MoveActionState {
    Init,
    ActivePath(NodeIterator),
    SelectPath,
}

#[derive(Debug)]
pub struct MoveAction {
    player_id: PlayerID,
    state: MoveActionState,
}
impl MoveAction {
    pub fn new(id: PlayerID) -> Box<MoveAction> {
        Box::new(MoveAction {
            state: MoveActionState::Init,
            player_id: id,
        })
    }
    fn continue_along_path(path: &mut NodeIterator, game_state: &mut GameState) -> ProcState {
        let player_id = game_state.info.active_player.unwrap();

        for next_event in path.by_ref() {
            match next_event {
                itertools::Either::Left(position) => {
                    game_state.move_player(player_id, position).unwrap();
                    game_state.get_mut_player_unsafe(player_id).add_move(1);
                }
                itertools::Either::Right(roll) => {
                    if event_ends_player_action(&roll) {
                        game_state.get_mut_player_unsafe(player_id).used = true;
                    }
                    return ProcState::NotDoneNew(proc_from_roll(roll, player_id));
                }
            }
        }
        ProcState::NotDone
    }
    fn available_actions(&self, game_state: &GameState) -> Box<AvailableActions> {
        let player = game_state.get_player_unsafe(self.player_id);
        let mut aa = AvailableActions::new(player.stats.team);
        aa.insert_paths(PathFinder::player_paths(game_state, self.player_id).unwrap());
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }
}
impl Procedure for MoveAction {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if game_state.info.handle_td_by.is_some() || game_state.info.turnover {
            // game_state.get_mut_player_unsafe(self.player_id).used = true;
            return ProcState::Done;
        }

        match game_state.get_player(self.player_id) {
            Ok(player) if player.used => return ProcState::Done,
            Err(_) => return ProcState::Done, // player not on field anymore
            _ => (),
        }

        match (action, &mut self.state) {
            (None, MoveActionState::Init) => {
                self.state = MoveActionState::SelectPath;
                ProcState::NeedAction(self.available_actions(game_state))
            }
            (None, MoveActionState::ActivePath(path)) => {
                let proc_state = MoveAction::continue_along_path(path, game_state);
                if path.is_empty() {
                    self.state = MoveActionState::Init;
                }
                proc_state
            }
            (Some(Action::Positional(_, position)), MoveActionState::SelectPath) => {
                let mut path = game_state
                    .available_actions
                    .take_path(position)
                    .unwrap()
                    .iter();
                let proc_state = MoveAction::continue_along_path(&mut path, game_state);
                if path.is_empty() {
                    self.state = MoveActionState::Init;
                } else {
                    self.state = MoveActionState::ActivePath(path);
                }
                proc_state
            }
            (Some(Action::Simple(SimpleAT::EndPlayerTurn)), _) => {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
                ProcState::Done
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct DodgeProc {
    target: D6Target,
    id: PlayerID,
}
impl DodgeProc {
    fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<DodgeProc>> {
        SimpleProcContainer::new(DodgeProc { target, id })
    }
}
impl SimpleProc for DodgeProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Dodge)
    }

    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

#[derive(Debug)]
struct GfiProc {
    target: D6Target,
    id: PlayerID,
}
impl GfiProc {
    fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<GfiProc>> {
        SimpleProcContainer::new(GfiProc { target, id })
    }
}
impl SimpleProc for GfiProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::SureFeet)
    }

    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.info.turnover = true;
        vec![block_procs::KnockDown::new(self.id)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

#[derive(Debug)]
pub struct GameOver;
impl GameOver {
    pub fn new() -> Box<GameOver> {
        Box::new(GameOver {})
    }
}
impl Procedure for GameOver {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        game_state.info.winner = match game_state.home.score.cmp(&game_state.away.score) {
            std::cmp::Ordering::Less => Some(TeamType::Away),
            std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => Some(TeamType::Home),
        };
        game_state.info.game_over = true;

        let mut aa = AvailableActions::new(TeamType::Home);
        aa.insert_simple(SimpleAT::EndSetup);
        aa.insert_simple(SimpleAT::DontUseReroll);
        ProcState::NeedAction(aa)
    }
}
#[derive(Debug)]
pub struct Kickoff {}
impl Kickoff {
    pub fn new() -> Box<Kickoff> {
        Box::new(Kickoff {})
    }
    fn changing_weather(&self, game_state: &mut GameState) {
        let roll = game_state.get_2d6_roll();
        game_state.info.weather = Weather::from(roll);
        let ball_pos = game_state.get_ball_position().unwrap();
        if game_state.info.weather == Weather::Nice && !ball_pos.is_out() {
            let d8 = game_state.get_d8_roll();
            let gust_of_wind = Direction::from(d8);
            game_state.ball = BallState::InAir(ball_pos + gust_of_wind);
        }
    }
}
impl Procedure for Kickoff {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        let team = game_state.info.kicking_this_drive;
        if action.is_none() {
            let mut aa = AvailableActions::new(team);
            aa.insert_simple(SimpleAT::KickoffAimMiddle);
            return ProcState::NeedAction(aa);
        }
        let mut ball_pos: Position = match action {
            Some(Action::Simple(SimpleAT::KickoffAimMiddle)) => {
                game_state.get_best_kickoff_aim_for(team)
            }
            _ => unreachable!(),
        };

        let dir_roll = game_state.get_d8_roll();
        let len_roll = game_state.get_d6_roll();
        ball_pos = ball_pos + Direction::from(dir_roll) * (len_roll as Coord);
        game_state.ball = BallState::InAir(ball_pos);

        let kickoff_roll = game_state.get_2d6_roll();
        let procs: Vec<Box<dyn Procedure>> = vec![LandKickoff::new()];
        match kickoff_roll {
            Sum2D6::Two => {
                //get the ref
            }
            Sum2D6::Three => {
                //Timeout
            }
            Sum2D6::Four => {
                //solid defense
            }
            Sum2D6::Five => {
                //High Kick
            }
            Sum2D6::Six => {
                //Cheering fans
            }
            Sum2D6::Seven => {
                //Brilliant coaching
            }
            Sum2D6::Eight => {
                self.changing_weather(game_state);
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
#[derive(Debug)]
pub struct Setup {
    team: TeamType,
}
impl Setup {
    pub fn new(team: TeamType) -> Box<Setup> {
        Box::new(Setup { team })
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
            TeamType::Home => los_x..=WIDTH_ - 2,
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
}
impl Procedure for Setup {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        let mut aa = AvailableActions::new(self.team);
        if action.is_none() {
            aa.insert_simple(SimpleAT::SetupLine);
            return ProcState::NeedAction(aa);
        }

        match action {
            Some(Action::Simple(SimpleAT::SetupLine)) => {
                self.random_setup(game_state);
                aa.insert_simple(SimpleAT::EndSetup);
                ProcState::NeedAction(aa)
            }

            Some(Action::Simple(SimpleAT::EndSetup)) => ProcState::Done,
            _ => unreachable!(),
        }
    }
}
#[derive(Debug)]
pub struct KOWakeUp {}
impl KOWakeUp {
    pub fn new() -> Box<KOWakeUp> {
        Box::new(KOWakeUp {})
    }
}
impl Procedure for KOWakeUp {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let target = D6Target::FourPlus;
        let num_kos = game_state
            .get_dugout()
            .filter(|player| player.place == DugoutPlace::KnockOut)
            .count();

        #[allow(clippy::needless_collect)]
        let rolls: Vec<D6> = repeat_with(|| game_state.get_d6_roll())
            .take(num_kos)
            .collect();

        game_state
            .get_dugout_mut()
            .filter(|player| player.place == DugoutPlace::KnockOut)
            .zip(rolls.into_iter())
            .filter(|(_, roll)| target.is_success(*roll))
            .for_each(|(player, _)| {
                player.place = DugoutPlace::Reserves;
            });

        ProcState::Done
    }
}
#[derive(Debug)]
pub struct CoinToss {
    coin_toss_winner: TeamType,
}
impl CoinToss {
    pub fn new() -> Box<CoinToss> {
        Box::new(CoinToss {
            coin_toss_winner: TeamType::Home,
        })
    }
}
impl Procedure for CoinToss {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if action.is_none() {
            let mut aa = AvailableActions::new(TeamType::Away);
            aa.insert_simple(SimpleAT::Heads);
            aa.insert_simple(SimpleAT::Tails);
            return ProcState::NeedAction(aa);
        }

        let Some(Action::Simple(simple_action)) = action else {unreachable!()};

        match simple_action {
            SimpleAT::Heads | SimpleAT::Tails => {
                let toss = game_state.get_coin_toss();
                self.coin_toss_winner = if simple_action == SimpleAT::from(toss) {
                    TeamType::Away
                } else {
                    TeamType::Home
                };

                let mut aa = AvailableActions::new(self.coin_toss_winner);
                aa.insert_simple(SimpleAT::Receive);
                aa.insert_simple(SimpleAT::Kick);
                ProcState::NeedAction(aa)
            }
            SimpleAT::Receive => {
                game_state.info.kicking_first_half = other_team(self.coin_toss_winner);
                ProcState::Done
            }
            SimpleAT::Kick => {
                game_state.info.kicking_first_half = self.coin_toss_winner;
                ProcState::Done
            }

            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct LandKickoff {}
impl LandKickoff {
    pub fn new() -> Box<LandKickoff> {
        Box::new(LandKickoff {})
    }
}
impl Procedure for LandKickoff {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let BallState::InAir(ball_position) = game_state.ball else { unreachable!() };

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

#[derive(Debug)]
pub struct TurnoverIfPossessionLost {}
impl TurnoverIfPossessionLost {
    pub fn new() -> Box<TurnoverIfPossessionLost> {
        Box::new(TurnoverIfPossessionLost {})
    }
}
impl Procedure for TurnoverIfPossessionLost {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        match game_state.ball {
            BallState::OnGround(_) | BallState::InAir(_) => {
                game_state.info.turnover = true;
            }
            BallState::Carried(id)
                if game_state.get_player_unsafe(id).stats.team != game_state.info.team_turn =>
            {
                game_state.info.turnover = true;
            }
            _ => unreachable!(),
        }
        ProcState::Done
    }
}

#[derive(Debug)]
pub struct StandUp {
    id: PlayerID,
}
impl StandUp {
    pub fn new(id: PlayerID) -> Box<StandUp> {
        Box::new(StandUp { id })
    }
}
impl Procedure for StandUp {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        debug_assert_eq!(
            game_state.get_player_unsafe(self.id).status,
            PlayerStatus::Down
        );
        game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Up;
        game_state.get_mut_player_unsafe(self.id).add_move(3);

        ProcState::Done
    }
}

use std::ops::RangeInclusive;

use rand::Rng;

use crate::core::dices::Sum2D6;
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, Coord, Direction, DugoutPlace, PlayerID,
    PlayerStatus, Position, ProcState, Procedure, Result, TeamType, Weather, HEIGHT_,
    LINE_OF_SCRIMMAGE_Y_RANGE,
};
use crate::core::pathing::{
    event_ends_player_action, CustomIntoIter, NodeIterator, PathFinder, PathingEvent,
};
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::procedures::{ball_procs, block_procs};
use crate::core::table::*;

use crate::core::{dices::D6Target, gamestate::GameState};
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
            .map(|p| p.id)
            .collect();
        let x_delta_sign = if self.team == TeamType::Home { 1 } else { -1 };
        let middle_x = game_state.get_line_of_scrimage_x(self.team);
        let middle_y = HEIGHT_ / 2;
        for id in players {
            let player = game_state.get_dugout_player(id).unwrap();
            let (dx, dy) = {
                match player.stats.role {
                    PlayerRole::Blitzer if blitzer_pos.len() > 0 => blitzer_pos.pop().unwrap(),
                    PlayerRole::Thrower if thrower_pos.len() > 0 => thrower_pos.pop().unwrap(),
                    PlayerRole::Catcher if catcher_pos.len() > 0 => catcher_pos.pop().unwrap(),
                    PlayerRole::Lineman if linemen_pos.len() > 0 => linemen_pos.pop().unwrap(),
                    _ => continue,
                }
            };
            let position = Position::new((middle_x + dx * x_delta_sign, middle_y + dy));
            println!(
                "fielding {:?} {:?} at {:?}",
                player.stats.role, player.stats.team, position
            );
            game_state.field_dugout_player(id, position)
        }
        Ok(())
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
                self.setup_line(game_state).unwrap();
                aa.insert_simple(SimpleAT::EndSetup);
                ProcState::NeedAction(aa)
            }

            Some(Action::Simple(SimpleAT::EndSetup)) => ProcState::Done,
            _ => unreachable!(),
        }
    }
}

use std::iter::repeat_with;

use crate::core::dices::D6;
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, DugoutPlace, PlayerStatus, Position,
    ProcState, Procedure, TeamType,
};
use crate::core::procedures::{ball_procs, block_procs, kickoff_procs, movement_procs};
use crate::core::table::*;

use crate::core::{
    dices::{D6Target, RollTarget},
    gamestate::GameState,
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
            kickoff_procs::Kickoff::new(),
            kickoff_procs::Setup::new(kicking_team),
            kickoff_procs::Setup::new(other_team(kicking_team)),
            KOWakeUp::new(),
        ];

        game_state.ball = BallState::OffPitch;
        game_state.unfield_all_players().unwrap();

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
            other_team(self.kicking_this_half)
        } else {
            self.kicking_this_half
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
            .for_each(|p| p.reset_skills_and_moves());

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

        if !positions.is_empty() {
            let block_positions: Vec<Position> = positions
                .iter()
                .filter(|&&pos| game_state.get_player_at(pos).unwrap().status == PlayerStatus::Up)
                .filter(|&&pos| {
                    game_state.get_adj_players(pos).any(|adj_player| {
                        adj_player.status == PlayerStatus::Up && adj_player.stats.team != self.team
                    })
                })
                .copied()
                .collect();
            if !block_positions.is_empty() {
                aa.insert_positional(PosAT::StartBlock, block_positions);
            }
            if game_state.info.handoff_available {
                aa.insert_positional(PosAT::StartHandoff, positions.clone());
            }

            if game_state.info.blitz_available {
                aa.insert_positional(PosAT::StartBlitz, positions.clone());
            }

            if game_state.info.foul_available {
                aa.insert_positional(PosAT::StartFoul, positions.clone());
            }

            // if game_state.info.pass_available {
            //     aa.insert_positional(PosAT::StartPass, positions.clone());
            // }

            aa.insert_positional(PosAT::StartMove, positions);
        }
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
            ProcState::NotDoneNew(movement_procs::MoveAction::new(info.active_player.unwrap()))
        } else if let Some(Action::Simple(SimpleAT::EndTurn)) = action {
            ProcState::Done
        } else {
            unreachable!()
        }
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
            .zip(rolls)
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

        let Some(Action::Simple(simple_action)) = action else {
            unreachable!()
        };

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

#[cfg(test)]
mod tests {

    use crate::core::dices::BlockDice;
    use crate::core::dices::Coin;
    use crate::core::dices::D8;
    use crate::core::model::*;
    use crate::core::table::*;
    use crate::core::{
        gamestate::{GameState, GameStateBuilder},
        model::{Position, TeamType},
        table::PosAT,
    };
    use crate::standard_state;

    #[test]
    fn turnover() {
        let h1_pos = Position::new((5, 5));
        let h2_pos = Position::new((5, 6));
        let a1_pos = Position::new((6, 5));
        let a2_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(h1_pos)
            .add_home_player(h2_pos)
            .add_away_player(a1_pos)
            .add_away_player(a2_pos)
            .build();

        let id_h1 = state.get_player_id_at(h1_pos).unwrap();
        // let id_h2 = state.get_player_id_at(h2_pos).unwrap();
        // let id_a1 = state.get_player_id_at(a1_pos).unwrap();
        // let id_a2 = state.get_player_id_at(a2_pos).unwrap();

        state.home.rerolls = 0;
        state.away.rerolls = 0;

        state.step_positional(PosAT::StartMove, h2_pos);
        state.step_simple(SimpleAT::EndPlayerTurn);

        state.step_positional(PosAT::StartMove, h1_pos);
        state.fixes.fix_d6(1); //dodge fail
        state.fixes.fix_d6(6); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d6(1); //injury
        state.fixes.fix_d6(1); //injury
        state.step_positional(PosAT::Move, h1_pos + (-1, -1));

        assert!(state.away_to_act());
        assert_eq!(state.get_player_unsafe(id_h1).status, PlayerStatus::Stunned);

        state.step_simple(SimpleAT::EndTurn);

        assert!(state.home_to_act());
        assert_eq!(state.get_player_unsafe(id_h1).status, PlayerStatus::Stunned);

        state.step_simple(SimpleAT::EndTurn);
        assert_eq!(state.get_player_unsafe(id_h1).status, PlayerStatus::Down);
    }

    #[test]
    fn clear_used() {
        let start_pos = Position::new((2, 5));
        let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

        let id = state.get_player_id_at(start_pos).unwrap();

        assert!(state.home_to_act());
        state.step_positional(PosAT::StartMove, start_pos);
        state.step_simple(SimpleAT::EndPlayerTurn);
        assert!(state.get_player_unsafe(id).used);

        state.step_simple(SimpleAT::EndTurn);

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.home_to_act());
        assert!(!state.get_player_unsafe(id).used);
        state.step_positional(PosAT::StartMove, start_pos);
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    #[test]
    fn turn_stunned() {
        let start_pos = Position::new((2, 5));
        let mut state = GameStateBuilder::new().add_home_player(start_pos).build();

        let id = state.get_player_id_at(start_pos).unwrap();

        assert!(state.home_to_act());
        state.get_mut_player_unsafe(id).status = PlayerStatus::Stunned;
        state.get_mut_player_unsafe(id).used = true;
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.away_to_act());
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
        state.step_simple(SimpleAT::EndTurn);

        assert!(state.home_to_act());
        assert!(!state.get_player_unsafe(id).used);
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
        state.step_positional(PosAT::StartMove, start_pos);
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    #[test]
    fn start_of_game() {
        let mut state: GameState = GameStateBuilder::new_start_of_game();

        assert!(state.away_to_act());
        state.fixes.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads);

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::Kick);

        assert!(state.home_to_act());
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::SetupLine);
        state.step_simple(SimpleAT::EndSetup);

        state.fixes.fix_d8_direction(Direction::up()); // scatter direction
        state.fixes.fix_d6(5); // scatter length

        state.fixes.fix_d6(4); // fix changing whether kickoff result
        state.fixes.fix_d6(4); // fix changing weather kickoff result

        state.fixes.fix_d6(2); // Nice weather
        state.fixes.fix_d6(5); // nice weather

        state.fixes.fix_d8_direction(Direction::right()); // gust of wind
        state.fixes.fix_d8_direction(Direction::right()); // bounce

        assert!(state.away_to_act());
        state.step_simple(SimpleAT::KickoffAimMiddle);

        let ball_pos = state.get_ball_position().unwrap();
        assert!(matches!(state.ball, BallState::OnGround(_)));
        assert_eq!(ball_pos, Position::new((23, 2)));
    }
    #[test]
    fn turn_order() -> Result<()> {
        let mut state = standard_state();
        assert!(state.home_to_act());
        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 0);
        assert_eq!(state.info.team_turn, TeamType::Home);

        state.step_simple(SimpleAT::EndTurn);

        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 1);
        assert_eq!(state.info.away_turn, 1);
        assert_eq!(state.info.team_turn, TeamType::Away);

        state.step_simple(SimpleAT::EndTurn);

        assert_eq!(state.info.half, 1);
        assert_eq!(state.info.home_turn, 2);
        assert_eq!(state.info.away_turn, 1);
        assert_eq!(state.info.team_turn, TeamType::Home);

        Ok(())
    }

    #[test]
    fn touchdown() {
        let start_pos = Position::new((2, 1));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(start_pos)
            .build();

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        state.step_positional(PosAT::StartMove, start_pos);
        state.step_positional(PosAT::Move, td_pos);

        assert_eq!(state.home.score, 1);
        assert_eq!(state.away.score, 0);
        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
    }

    #[test]
    fn failed_gfi_touchdown() {
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(start_pos)
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();
        let ma = state.get_player_unsafe(id).stats.ma;
        state.get_mut_player_unsafe(id).moves = ma;
        assert_eq!(state.get_player_unsafe(id).moves_left(), 0);
        assert_eq!(state.get_player_unsafe(id).total_movement_left(), 2);

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        state.step_positional(PosAT::StartMove, start_pos);
        state.fixes.fix_d6(1);
        state.step_positional(PosAT::Move, td_pos);

        state.fixes.fix_d8(4);
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);
        state.step_simple(SimpleAT::DontUseReroll);

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
        assert_eq!(state.ball, BallState::OnGround(td_pos + (1, 0)));
        assert_eq!(state.get_player_unsafe(id).position, td_pos);
    }

    #[test]
    fn pushed_to_touchdown() {
        let carrier_pos = Position::new((2, 5));
        let blocker_pos = Position::new((3, 5));
        let td_pos = carrier_pos + (carrier_pos - blocker_pos);
        let mut state = GameStateBuilder::new()
            .add_home_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .add_away_player(blocker_pos)
            .build();

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);
        state.step_simple(SimpleAT::EndTurn);
        state.step_positional(PosAT::StartBlock, blocker_pos);
        state.fixes.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, carrier_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, td_pos);
        state.step_positional(PosAT::FollowUp, carrier_pos);

        assert_eq!(state.home.score, 1);
        assert_eq!(state.away.score, 0);
        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
    }

    #[test]
    fn no_td_when_knocked_down_with_ball() {
        let carrier_pos = Position::new((2, 5));
        let blocker_pos = Position::new((3, 5));
        let td_pos = carrier_pos + (carrier_pos - blocker_pos);
        let mut state = GameStateBuilder::new()
            .add_home_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .add_away_player(blocker_pos)
            .build();

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);
        state.step_simple(SimpleAT::EndTurn);
        state.step_positional(PosAT::StartBlock, blocker_pos);
        state.fixes.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, carrier_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, td_pos);
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);
        state.fixes.fix_d8(4);
        state.step_positional(PosAT::FollowUp, blocker_pos);
        // state.step_simple(SimpleAT::EndPlayerTurn);
        state.step_simple(SimpleAT::EndTurn);
        // state.step_positional(PosAT::StartMove, td_pos);
        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);
        assert_eq!(state.ball, BallState::OnGround(td_pos + (1, 0)));
    }

    #[test]
    fn follow_up_to_touchdown() {
        let carrier_pos = Position::new((2, 5));
        let victim_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .add_away_player(victim_pos)
            .build();

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);
        state.step_positional(PosAT::StartBlock, carrier_pos);
        state.fixes.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, victim_pos);
        state.step_simple(SimpleAT::SelectPush);
        //no need to select push position because crowd
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);
        state.step_positional(PosAT::FollowUp, victim_pos);

        assert_eq!(state.home.score, 1);
        assert_eq!(state.away.score, 0);
        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
    }

    #[test]
    fn touchdown_pickup_in_endzone() {
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        state.step_positional(PosAT::StartMove, start_pos);

        state.fixes.fix_d6(3);
        state.step_positional(PosAT::Move, td_pos);

        assert_eq!(state.home.score, 1);
        assert_eq!(state.away.score, 0);
        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
    }
    #[test]
    fn no_td_when_failed_pickup_in_endzone() {
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        state.step_positional(PosAT::StartMove, start_pos);

        state.fixes.fix_d6(2);
        state.step_positional(PosAT::Move, td_pos);

        state.fixes.fix_d8(4);
        state.step_simple(SimpleAT::DontUseReroll);

        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);
    }
    #[test]
    fn touchdown_when_catching_bouncing_ball() {
        let mut field = "".to_string();
        field += "hh \n";
        field += " A \n";
        field += "  h\n";
        let td_pos = Position::new((1, 3));
        let carrier_pos = td_pos + (1, 1);
        let blocker_pos = carrier_pos + (1, 1);
        let push_pos = carrier_pos + (-1, 0);
        let mut state = GameStateBuilder::new().add_str(td_pos, &field).build();
        assert_eq!(state.home.score, 0);
        assert_eq!(state.away.score, 0);

        state.step_positional(PosAT::StartBlock, blocker_pos);

        state.fixes.fix_blockdice(BlockDice::Pow);
        state.fixes.fix_blockdice(BlockDice::Pow);

        state.step_positional(PosAT::Block, carrier_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, push_pos);

        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d6(2); //armor
        state.fixes.fix_d6(4); //catch
        state
            .fixes
            .fix_d8(D8::from(Direction { dx: 0, dy: -1 }) as u8); // bounce direction up
        state.step_positional(PosAT::FollowUp, blocker_pos);

        assert_eq!(state.home.score, 1);
        assert_eq!(state.away.score, 0);

        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::SetupLine)));
    }

    #[test]
    fn ball_carrier_casualty() {
        color_backtrace::install();
        let push_pos = Position::new((4, 4));
        let away_pos = Position::new((5, 5));
        let home_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .add_ball_pos(away_pos)
            .build();
        state.step_positional(PosAT::StartBlock, home_pos);
        state.fixes.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, push_pos);
        state.fixes.fix_d6(6); //armor
        state.fixes.fix_d6(5); //armor
        state.fixes.fix_d6(6); //injury
        state.fixes.fix_d6(5); //injury
        let d8_fix = D8::Two;
        state.fixes.fix_d8(d8_fix as u8);
        state.step_positional(PosAT::FollowUp, away_pos);

        // let direction = Direction::from(d8_fix);
    }
}

use serde::{Deserialize, Serialize};

use crate::core::dices::{BlockDice, RequestedRoll, RollResult};
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, Direction, PlayerStatus, Position, ProcState, Procedure,
};
use crate::core::model::{BallState, PlayerID, ProcInput};
use crate::core::procedures::ball_procs;
use crate::core::procedures::casualty_procs;
use crate::core::table::{NumBlockDices, PosAT, SimpleAT, Skill};

use super::AnyProc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum PushSquares {
    Crowd(Position),
    ChainPush(Vec<Position>),
    FreeSquares(Vec<Position>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Push {
    from: Position,
    on: Position,
    knockdown_proc: Option<KnockDown>,
    moves_to_make: Vec<(Position, Position)>,
    follow_up_pos: Position,
}

impl Push {
    pub fn new(from: Position, on: Position) -> AnyProc {
        AnyProc::Push(Push {
            from,
            on,
            moves_to_make: Vec::with_capacity(1),
            knockdown_proc: None,
            follow_up_pos: on,
        })
    }
    pub fn new_pure(from: Position, on: Position) -> Push {
        Push {
            from,
            on,
            moves_to_make: Vec::with_capacity(1),
            knockdown_proc: None,
            follow_up_pos: on,
        }
    }

    /// Would pushing the player standing `on` away from `from` send them
    /// into the crowd? True exactly when no push square is free and at
    /// least one is out of bounds — the case `calculate_next_state`
    /// resolves without asking for a square. Read-only; used by the
    /// MCTS block-outcome model to fold "push into the crowd" into the
    /// defender-removed outcome.
    pub fn is_crowd_push(from: Position, on: Position, game_state: &GameState) -> bool {
        matches!(Push::get_push_squares(on, from, game_state), PushSquares::Crowd(_))
    }

    fn get_push_squares(on: Position, from: Position, game_state: &GameState) -> PushSquares {
        let direction = on - from;
        let opposite_pos = on + direction;
        let mut push_squares = match direction {
            Direction { dx: 0, dy: _ } => vec![opposite_pos + (1, 0), opposite_pos + (-1, 0)],
            Direction { dx: _, dy: 0 } => vec![opposite_pos + (0, 1), opposite_pos + (0, -1)],
            Direction { dx, dy } => vec![opposite_pos + (-dx, 0), opposite_pos + (0, -dy)],
        };
        push_squares.push(on + direction);
        let free_squares: Vec<Position> = push_squares
            .iter()
            .filter(|&pos| !game_state.is_out(*pos) && game_state.get_player_at(*pos).is_none())
            .copied()
            .collect();

        if !free_squares.is_empty() {
            PushSquares::FreeSquares(free_squares)
        } else if let Some(&oob) = push_squares.iter().rev().find(|&&pos| game_state.is_out(pos)) {
            // The victim goes into the crowd through a square that is
            // actually out of bounds — preferring the straight-ahead square
            // (last in the list). The straight square can be in bounds but
            // occupied while only a diagonal is out; blindly popping the
            // last candidate would move the victim onto an occupied square.
            PushSquares::Crowd(oob)
        } else {
            PushSquares::ChainPush(push_squares)
        }
    }
    fn do_moves(&self, game_state: &mut GameState) {
        self.moves_to_make.iter().rev().for_each(|(from, to)| {
            let id = game_state.get_player_id_at(*from).unwrap();
            game_state.move_player(id, *to).unwrap();
            if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == id && to.x == game_state.get_endzone_x(game_state.get_player_unsafe(id).stats.team)) {
                game_state.info.handle_td_by = Some(id);
            }
        });
    }

    fn handle_aftermath(&mut self, game_state: &mut GameState) -> ProcState {
        let mut procs: Vec<AnyProc> = Vec::with_capacity(3);
        let (last_push_from, last_push_to) = self.moves_to_make.pop().unwrap();
        if game_state.is_out(last_push_to) {
            let id = game_state.get_player_id_at(last_push_to).unwrap();
            if matches!(game_state.ball, BallState::Carried(carrier) if carrier == id) {
                game_state.set_ball(BallState::InAir(last_push_from));
                procs.push(ball_procs::ThrowIn::new(last_push_from));
            }
            procs.push(casualty_procs::Injury::new_crowd(id));
            if self.moves_to_make.is_empty() {
                //Means there was only one push which was the already handled crowd push, so we can forget about any knockdown proc
                self.knockdown_proc = None;
            }
        } else if matches!(game_state.ball, BallState::OnGround(ball_pos) if ball_pos == last_push_to) {
            // A player shoved onto a loose ball dislodges it — the ball may
            // never come to rest under a player. Only the *last* square of a
            // chain push can hold a loose ball; every earlier one was occupied.
            // Queued before the knockdown so it resolves after it (procs run
            // last-in-first-out), matching the reference implementation.
            procs.push(ball_procs::Bounce::new());
        }
        if let Some(proc) = self.knockdown_proc.take() {
            procs.push(AnyProc::KnockDown(proc));
        }
        ProcState::from(procs)
    }

    fn calculate_next_state(&mut self, game_state: &mut GameState) -> ProcState {
        let mut aa = AvailableActions::new(game_state.info.team_turn);
        match Push::get_push_squares(self.on, self.from, game_state) {
            PushSquares::Crowd(position_in_crowd) => {
                self.moves_to_make.push((self.on, position_in_crowd));
                self.do_moves(game_state);
                ProcState::NotDoneNew(FollowUp::new(self.follow_up_pos))
            }
            PushSquares::ChainPush(positions) | PushSquares::FreeSquares(positions) => {
                aa.insert_positional(PosAT::Push, positions);
                ProcState::NeedAction(aa)
            }
        }
    }
}

impl Procedure for Push {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing if self.moves_to_make.is_empty() => self.calculate_next_state(game_state),
            ProcInput::Nothing => self.handle_aftermath(game_state),
            ProcInput::Action(Action::Positional(PosAT::Push, position_to))
                if game_state.get_player_at(position_to).is_some() =>
            {
                self.moves_to_make.push((self.on, position_to));
                self.from = self.on;
                self.on = position_to;
                self.calculate_next_state(game_state)
            }
            ProcInput::Action(Action::Positional(PosAT::Push, position)) => {
                self.moves_to_make.push((self.on, position));
                self.do_moves(game_state);
                ProcState::NotDoneNew(FollowUp::new(self.follow_up_pos))
            }
            _ => panic!("very wrong!"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FollowUp {
    to: Position,
    //from is active player,
}
impl FollowUp {
    pub fn new(to: Position) -> AnyProc {
        AnyProc::FollowUp(FollowUp { to })
    }
}
impl Procedure for FollowUp {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let player = game_state.get_active_player().unwrap();
        match input {
            ProcInput::Nothing => {
                let mut aa = AvailableActions::new(player.stats.team);
                aa.insert_positional(PosAT::FollowUp, vec![player.position, self.to]);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::FollowUp, position)) => {
                if player.position != position {
                    let id = player.id;
                    let team = player.stats.team;

                    game_state.move_player(player.id, position).unwrap();

                    if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == id)
                        && game_state.get_endzone_x(team) == position.x
                    {
                        game_state.info.handle_td_by = Some(id)
                    }
                }
                ProcState::Done
            }
            _ => panic!("very wrong!"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct KnockDown {
    id: Option<PlayerID>,
    second_id: Option<PlayerID>,
}
impl KnockDown {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::KnockDown(KnockDown {
            id: Some(id),
            second_id: None,
        })
    }
    pub fn new_pure(id: PlayerID) -> KnockDown {
        KnockDown {
            id: Some(id),
            second_id: None,
        }
    }
    pub fn new_empty() -> AnyProc {
        AnyProc::KnockDown(KnockDown {
            id: None,
            second_id: None,
        })
    }
}
fn knock_down_player(game_state: &mut GameState, id: PlayerID) -> (bool, bool) {
    let player = match game_state.get_mut_player(id) {
        Ok(player_) => player_,
        Err(_) => return (false, false), //Means the player is already off the pitch, most likely crowd push
    };
    debug_assert!(matches!(player.status, PlayerStatus::Up));
    player.status = PlayerStatus::Down;
    player.used = true;
    let player_position = player.position;
    // let armor_proc = casualty_procs::Armor::new(id);

    if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == id) {
        game_state.set_ball(BallState::InAir(player_position));
        (true, true)
    } else {
        (true, false)
    }
}
impl Procedure for KnockDown {
    fn step(&mut self, game_state: &mut GameState, _input: ProcInput) -> ProcState {
        let mut procs: Vec<AnyProc> = Vec::with_capacity(3);
        let mut armor_procs: Vec<AnyProc> = Vec::with_capacity(2);
        let mut ball_in_air = false;
        for id in [self.id, self.second_id].iter().flatten() {
            let (knocked_down, p_ball_in_air) = knock_down_player(game_state, *id);
            ball_in_air |= p_ball_in_air;
            if knocked_down {
                armor_procs.push(casualty_procs::Armor::new(*id))
            }
        }
        if ball_in_air {
            procs.push(ball_procs::Bounce::new());
        }
        for armor_proc in armor_procs {
            procs.push(armor_proc);
        }
        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BlockAction {}

impl BlockAction {
    pub fn new() -> AnyProc {
        AnyProc::BlockAction(BlockAction {})
    }
    fn fill_available_actions(&mut self, game_state: &mut GameState) {
        let player = game_state.get_active_player().unwrap();
        let team = player.stats.team;
        let attacker_id = player.id;
        let attacker_pos = player.position;

        // Collect first to release the &GameState borrow before mutating below.
        let victims: smallvec::SmallVec<[(Position, NumBlockDices); 4]> = game_state
            .get_adj_players(attacker_pos)
            .filter(|adj_player| {
                !adj_player.used && adj_player.stats.team != team && adj_player.status == PlayerStatus::Up
            })
            .map(|defender| (defender.position, game_state.get_blockdices(attacker_id, defender.id)))
            .collect();

        // Reset AA + buffer, then place the block offerings.
        let mut buf = game_state.take_path_buffer();
        for slot in buf.iter_mut() {
            *slot = None;
        }
        for (pos, dice) in victims.iter() {
            buf[*pos] = Some(std::sync::Arc::new(crate::core::pathing::Node::new_direct_block_node(
                *dice, *pos,
            )));
        }
        game_state.install_path_buffer(buf);

        *game_state.available_actions = AvailableActions::default();
        game_state.available_actions.team = Some(team);
        game_state.available_actions.has_paths = true;
        game_state.available_actions.insert_simple(SimpleAT::EndPlayerTurn);
    }
}
impl Procedure for BlockAction {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => {
                self.fill_available_actions(game_state);
                ProcState::NeedActionInPlace
            }
            ProcInput::Action(Action::Positional(PosAT::Block, position)) => {
                let block_path = game_state.take_path(position).unwrap();
                let num_dice = block_path.get_block_dice().unwrap();
                let defender_id = game_state.get_player_id_at(position).unwrap();
                game_state.get_active_player_mut().unwrap().used = true;
                ProcState::DoneNew(Block::new(num_dice, defender_id))
            }
            ProcInput::Action(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                game_state.get_active_player_mut().unwrap().used = true;
                ProcState::Done
            }
            _ => panic!("Invalid input {:?}", input),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Block {
    dices: NumBlockDices,
    defender: PlayerID,
    state: BlockProcState,
    roll: [Option<BlockDice>; 3],
    is_uphill: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
enum BlockProcState {
    Init,               //step shall roll first dice
    SelectDice,         //attacker (or defender if uphill) to choose dice
    SelectDiceOrReroll, // Attacker may choose dice or reroll
    UphillSelectReroll, // In uphill, attacker may choose to reroll
}

impl Block {
    pub fn new(dices: NumBlockDices, defender: PlayerID) -> AnyProc {
        // the point is that number of dices has already been calculated, so this proc doesn't need to redo it.
        AnyProc::Block(Block {
            dices,
            defender,
            state: BlockProcState::Init,
            roll: Default::default(),
            is_uphill: matches!(dices, NumBlockDices::TwoUphill | NumBlockDices::ThreeUphill),
        })
    }

    /// The player being blocked.
    pub fn defender(&self) -> PlayerID {
        self.defender
    }

    /// The dice count (and uphill-ness) this block was set up with.
    pub fn num_dices(&self) -> NumBlockDices {
        self.dices
    }

    fn add_aa(&self, aa: &mut AvailableActions) {
        self.roll
            .iter()
            .filter_map(|&r| r.map(SimpleAT::from))
            .for_each(|at| aa.insert_simple(at));
    }
    fn available_actions(&mut self, game_state: &GameState) -> Box<AvailableActions> {
        let mut aa = AvailableActions::new_empty();
        let team = game_state.get_active_player().unwrap().stats.team;
        match self.state {
            BlockProcState::SelectDice => {
                aa.team = Some(if self.is_uphill { other_team(team) } else { team });
                self.add_aa(&mut aa);
            }
            BlockProcState::SelectDiceOrReroll => {
                aa.team = Some(team);
                self.add_aa(&mut aa);
                aa.insert_simple(SimpleAT::UseReroll);
            }
            BlockProcState::UphillSelectReroll => {
                aa.team = Some(team);
                aa.insert_simple(SimpleAT::UseReroll);
                aa.insert_simple(SimpleAT::DontUseReroll);
            }
            BlockProcState::Init => panic!("should not happen!"),
        }
        aa
    }
}
impl Procedure for Block {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        if game_state.info.player_action_type.unwrap() == PosAT::StartBlitz {
            game_state.info.player_action_type = Some(PosAT::StartMove); //to preven the player from blitzing again
            game_state.get_active_player_mut().unwrap().add_move(1);
        }
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::BlockDice(self.dices)),
            ProcInput::Roll(RollResult::BlockDice(rolls)) => {
                self.roll = rolls;
                let reroll_available = game_state.get_active_players_team().unwrap().can_use_reroll();
                self.state = match (reroll_available, self.is_uphill) {
                    (true, true) => BlockProcState::UphillSelectReroll,
                    (true, false) => BlockProcState::SelectDiceOrReroll,
                    (false, _) => BlockProcState::SelectDice,
                };
                ProcState::NeedAction(self.available_actions(game_state))
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseReroll)) => {
                game_state.get_active_players_team_mut().unwrap().use_reroll();
                ProcState::NeedRoll(RequestedRoll::BlockDice(self.dices))
            }
            ProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)) => {
                self.state = BlockProcState::SelectDice;
                // ProcState::NotDone //I think it should be available_actions here...
                ProcState::NeedAction(self.available_actions(game_state))
            }
            ProcInput::Action(Action::Simple(dice_action_type)) => {
                let attacker_id = game_state.info.active_player.unwrap();
                let mut push = false;
                let mut knockdown_proc: KnockDown = KnockDown {
                    id: None,
                    second_id: None,
                };

                match dice_action_type {
                    SimpleAT::SelectBothDown => {
                        if !game_state.get_active_player().unwrap().has_skill(Skill::Block) {
                            knockdown_proc.second_id = Some(attacker_id);
                        }
                        if !game_state.get_player_unsafe(self.defender).has_skill(Skill::Block) {
                            knockdown_proc.id = Some(self.defender);
                        }
                    }
                    SimpleAT::SelectPow => {
                        knockdown_proc.id = Some(self.defender);
                        push = true;
                    }
                    SimpleAT::SelectPush => {
                        push = true;
                    }
                    SimpleAT::SelectPowPush => {
                        if !game_state.get_player_unsafe(self.defender).has_skill(Skill::Dodge) {
                            knockdown_proc.id = Some(self.defender);
                        }
                        push = true;
                    }

                    SimpleAT::SelectSkull => knockdown_proc.second_id = Some(attacker_id),
                    _ => panic!("very wrong!"),
                }

                let mut procs: Vec<AnyProc> = Vec::with_capacity(3);

                //if attacker is knocked down it's a turnover
                if knockdown_proc.second_id.is_some() {
                    game_state.info.turnover = true;
                }

                // if any player is knocked down we add the knockdown proc making it the last proc
                // to be executed of the returned ones
                if knockdown_proc.id.is_some() || knockdown_proc.second_id.is_some() {
                    procs.push(AnyProc::KnockDown(knockdown_proc));
                }

                if push {
                    procs.push(Push::new(
                        game_state.get_player_unsafe(attacker_id).position,
                        game_state.get_player_unsafe(self.defender).position,
                    ));
                }
                ProcState::from(procs)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::dices::{BlockDice, D8};
    use crate::core::model::*;
    use crate::core::table::*;
    use crate::core::{
        gamestate::GameStateBuilder,
        model::{DugoutPlace, PlayerStats, Position, TeamType},
        table::PosAT,
    };

    #[test]
    fn crowd_chain_push() {
        let mut field = "".to_string();
        field += " aa\n";
        field += " aa\n";
        field += "h  \n";
        let first_pos = Position::new((5, 1));
        let mut state = GameStateBuilder::new().add_str(first_pos, &field).build();

        state.step_positional(PosAT::StartBlock, Position::new((5, 3)));
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, Position::new((6, 2)));
        state.step_simple(SimpleAT::SelectPush);

        state.step_positional(PosAT::Push, Position::new((6, 1)));
        state.fix_d6(1);
        state.fix_d6(1);

        state.step_positional(PosAT::FollowUp, Position::new((6, 2)));

        state.step_simple(SimpleAT::EndTurn);

        assert!(matches!(
            state.get_dugout().next(),
            Some(DugoutPlayer {
                place: DugoutPlace::Reserves,
                stats: PlayerStats {
                    team: TeamType::Away,
                    ..
                },
                ..
            })
        ));
    }
    /// Sideline sandwich: victim pushed along the sideline into a standing
    /// player, with the straight-ahead square occupied and only a *diagonal*
    /// square out of bounds. The crowd push must send the victim through
    /// the OOB square — not "the last candidate in the list" (the occupied
    /// straight square), which panics `move_player`. Constant on small
    /// boards (3 playable rows), reachable on the full pitch as here.
    #[test]
    fn crowd_push_picks_the_oob_square_not_the_occupied_straight_one() {
        let attacker_pos = Position::new((5, 1));
        let victim_pos = Position::new((6, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(victim_pos)
            .add_away_player(Position::new((7, 1))) // straight push square: occupied
            .add_home_player(Position::new((7, 2))) // diagonal in-bounds square: occupied
            .build();

        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, victim_pos);
        state.step_simple(SimpleAT::SelectPush);

        state.fix_d6(1); // crowd injury roll...
        state.fix_d6(1); // ...stunned → reserves box
        state.step_positional(PosAT::FollowUp, victim_pos);
        state.step_simple(SimpleAT::EndTurn);

        assert!(
            state.get_player_at(Position::new((7, 1))).is_some(),
            "the straight-ahead blocker must not be displaced"
        );
        assert!(matches!(
            state.get_dugout().next(),
            Some(DugoutPlayer {
                place: DugoutPlace::Reserves,
                stats: PlayerStats {
                    team: TeamType::Away,
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn blitz() {
        let start_pos = Position::new((2, 1));
        let target_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(target_pos)
            .build();
        state.step_positional(PosAT::StartBlitz, start_pos);

        state.fix_blockdice(BlockDice::Skull);
        state.step_positional(PosAT::Block, target_pos);
    }

    #[test]
    fn test_block_2d_bothdown_casualty() -> Result<()> {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(Position::new((5, 6)))
            .add_away_player(away_pos)
            .build();

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, away_pos);
        state.fix_d6(5); //home armor
        state.fix_d6(6); //home armor
        state.fix_d6(6); //home injury
        state.fix_d6(6); //home injury
        state.fix_d6(1); //away armor
        state.fix_d6(1); //away armor
        state.step_simple(SimpleAT::SelectBothDown);

        assert!(state.get_player_at(home_pos).is_none());
        assert!(matches!(
            state.get_dugout().next(),
            Some(DugoutPlayer {
                place: DugoutPlace::Injuried,
                stats: PlayerStats {
                    team: TeamType::Home,
                    ..
                },
                ..
            })
        ));
        assert_eq!(state.get_player_at(away_pos).unwrap().status, PlayerStatus::Down);

        assert!(state.fixes_is_empty());
        Ok(())
    }

    #[test]
    fn single_dice_block() -> Result<()> {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let push_pos = Position::new((6, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, push_pos);
        state.fix_d6(1);
        state.fix_d6(1);
        state.step_positional(PosAT::FollowUp, away_pos);

        assert_eq!(state.get_player_at(push_pos).unwrap().status, PlayerStatus::Down);
        assert!(state.fixes_is_empty());
        assert!(state.get_player_at(away_pos).unwrap().used);

        assert!(state.get_paths().is_none());
        {
            let aa = state.get_available_actions();
            assert!(
                aa.get_positional().is_none()
                    || aa.get_positional().clone().unwrap().iter().all(|pa| { pa.is_empty() }),
            );
            assert_eq!(aa.get_simple().len(), 1);
        }
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));

        Ok(())
    }
    #[test]
    fn end_player_turn_instead_of_block() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();

        state.step_positional(PosAT::StartBlock, home_pos);
        state.step_simple(SimpleAT::EndPlayerTurn);

        assert!(state.get_player_at(home_pos).unwrap().used);
        assert!(state.get_paths().is_none());
        {
            let aa = state.get_available_actions();
            assert!(
                aa.get_positional().is_none()
                    || aa.get_positional().clone().unwrap().iter().all(|pa| { pa.is_empty() }),
            );
            assert_eq!(aa.get_simple().len(), 1);
        }
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));
    }

    #[test]
    fn available_block_action_adjescent_to_downed_player() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let away_pos_down = Position::new((4, 4));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .add_away_player(away_pos_down)
            .build();
        let downed_id = state.get_player_id_at(away_pos_down).unwrap();
        state.get_mut_player_unsafe(downed_id).status = PlayerStatus::Down;

        state.step_positional(PosAT::StartBlock, home_pos);
        let block_paths = state.get_paths().expect("BlockAction should have populated paths");

        assert!(block_paths.get_pos(away_pos_down).is_none());
    }
    #[test]
    fn prune_players_cant_startblock_but_can_startblitz() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        state
            .get_mut_player_unsafe(state.get_player_id_at(away_pos).unwrap())
            .status = PlayerStatus::Down;
        state.step_simple(SimpleAT::EndTurn);
        let aa = state.get_available_actions();
        assert!(aa.team.unwrap() == TeamType::Away);
        let aa_pos = aa.get_positional().clone().unwrap().get_pos(away_pos).clone();
        println!("{:?}", aa_pos);
        assert!(!aa_pos.contains(&PosAT::StartBlock));
        state.step_positional(PosAT::StartBlitz, away_pos);
        state.fix_blockdice(BlockDice::Skull);
        state.step_positional(PosAT::Block, home_pos);
    }
    #[test]
    fn attacker_knockdown_causes_turnover() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_home_player(Position::new((3, 3)))
            .add_away_player(away_pos)
            .build();

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Skull);
        state.step_positional(PosAT::Block, away_pos);

        state.fix_d6(1); //home armor
        state.fix_d6(1); //home armor
        state.step_simple(SimpleAT::SelectSkull);

        assert!(state.get_player_at(home_pos).unwrap().status == PlayerStatus::Down);
        assert!(state.get_player_at(away_pos).unwrap().status == PlayerStatus::Up);

        assert_eq!(state.available_actions.team.unwrap(), TeamType::Away);
        state.step_positional(PosAT::StartMove, away_pos);
    }

    /// A player pushed onto a square holding a loose ball must make the ball
    /// bounce - a loose ball may never come to rest under a player. Recorded
    /// in `data/web-games/web-gamestunned_ball_carrier.json`: a blocked
    /// catcher was pushed onto the loose ball and stunned on top of it, and
    /// the ball sat under them for the rest of the game.
    #[test]
    fn push_onto_loose_ball_bounces_it() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let ball_pos = Position::new((7, 7)); // straight-ahead push square
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .add_ball_pos(ball_pos)
            .build();

        assert_eq!(state.ball, BallState::OnGround(ball_pos));

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, ball_pos);
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.fix_d8(d8_fix as u8); //bounce
        state.step_positional(PosAT::FollowUp, home_pos);

        assert_eq!(state.get_player_at(ball_pos).unwrap().status, PlayerStatus::Down);
        assert_eq!(state.ball, BallState::OnGround(ball_pos + direction));
        assert!(state.fixes_is_empty());
    }

    /// Same rule without a knockdown: the push alone dislodges the ball.
    #[test]
    fn push_without_knockdown_onto_loose_ball_bounces_it() {
        let home_pos = Position::new((5, 5));
        let away_pos = Position::new((6, 6));
        let ball_pos = Position::new((7, 7)); // straight-ahead push square
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .add_ball_pos(ball_pos)
            .build();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, ball_pos);
        state.fix_d8(d8_fix as u8); //bounce
        state.step_positional(PosAT::FollowUp, home_pos);

        assert_eq!(state.get_player_at(ball_pos).unwrap().status, PlayerStatus::Up);
        assert_eq!(state.ball, BallState::OnGround(ball_pos + direction));
        assert!(state.fixes_is_empty());
    }
}

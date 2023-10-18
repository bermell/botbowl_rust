use crate::core::dices::{BlockDice, RollTarget, Sum2D6Target};
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, Direction, DugoutPlace, PlayerStatus, Position,
    ProcState, Procedure,
};
use crate::core::model::{BallState, PlayerID};
use crate::core::procedures::ball_procs;
use crate::core::table::{NumBlockDices, PosAT, SimpleAT, Skill};

#[derive(Debug)]
enum PushSquares {
    Crowd(Position),
    ChainPush(Vec<Position>),
    FreeSquares(Vec<Position>),
}
#[derive(Debug)]
pub struct Push {
    from: Position,
    on: Position,
    knockdown_proc: Option<Box<KnockDown>>,
    moves_to_make: Vec<(Position, Position)>,
    follow_up_pos: Position,
}

impl Push {
    pub fn new(from: Position, on: Position) -> Box<Push> {
        Box::new(Push {
            from,
            on,
            moves_to_make: Vec::with_capacity(1),
            knockdown_proc: None,
            follow_up_pos: on,
        })
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
            .filter(|&pos| !pos.is_out() && game_state.get_player_at(*pos).is_none())
            .copied()
            .collect();

        if !free_squares.is_empty() {
            PushSquares::FreeSquares(free_squares)
        } else if push_squares.iter().any(|&pos| pos.is_out()) {
            PushSquares::Crowd(push_squares.pop().unwrap())
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
        let mut procs: Vec<Box<dyn Procedure>> = Vec::with_capacity(2);
        let (last_push_from, last_push_to) = self.moves_to_make.pop().unwrap();
        if last_push_to.is_out() {
            let id = game_state.get_player_id_at(last_push_to).unwrap();
            if matches!(game_state.ball, BallState::Carried(carrier) if carrier == id) {
                game_state.ball = BallState::InAir(last_push_from);
                procs.push(ball_procs::ThrowIn::new(last_push_from));
            }
            procs.push(Injury::new_crowd(id));
            if self.moves_to_make.is_empty() {
                //Means there was only one push which was the already handled crowd push, so we can forget about any knockdown proc
                self.knockdown_proc = None;
            }
        }
        if let Some(proc) = self.knockdown_proc.take() {
            procs.push(proc);
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        match action {
            None if self.moves_to_make.is_empty() => self.calculate_next_state(game_state),
            None => self.handle_aftermath(game_state),
            Some(Action::Positional(PosAT::Push, position_to))
                if game_state.get_player_at(position_to).is_some() =>
            {
                self.moves_to_make.push((self.on, position_to));
                self.from = self.on;
                self.on = position_to;
                self.calculate_next_state(game_state)
            }
            Some(Action::Positional(PosAT::Push, position)) => {
                self.moves_to_make.push((self.on, position));
                self.do_moves(game_state);
                ProcState::NotDoneNew(FollowUp::new(self.follow_up_pos))
            }
            _ => panic!("very wrong!"),
        }
    }
}

#[derive(Debug)]
pub struct FollowUp {
    to: Position,
    //from is active player,
}
impl FollowUp {
    pub fn new(to: Position) -> Box<FollowUp> {
        Box::new(FollowUp { to })
    }
}
impl Procedure for FollowUp {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        let player = game_state.get_active_player().unwrap();
        match action {
            None => {
                let mut aa = AvailableActions::new(player.stats.team);
                aa.insert_positional(PosAT::FollowUp, vec![player.position, self.to]);
                ProcState::NeedAction(aa)
            }
            Some(Action::Positional(PosAT::FollowUp, position)) => {
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

#[derive(Debug)]
pub struct KnockDown {
    id: PlayerID,
}
impl KnockDown {
    pub fn new(id: PlayerID) -> Box<KnockDown> {
        Box::new(KnockDown { id })
    }
}
impl Procedure for KnockDown {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let mut player = match game_state.get_mut_player(self.id) {
            Ok(player_) => player_,
            Err(_) => return ProcState::Done, //Means the player is already off the pitch, most likely crowd push
        };
        debug_assert!(matches!(player.status, PlayerStatus::Up));
        player.status = PlayerStatus::Down;
        player.used = true;
        let armor_proc = Armor::new(self.id);
        if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
            ProcState::DoneNewProcs(vec![ball_procs::Bounce::new(), armor_proc])
        } else {
            ProcState::DoneNew(armor_proc)
        }
    }
}

#[derive(Debug)]
pub struct Armor {
    id: PlayerID,
    foul_target: Option<(PlayerID, Sum2D6Target)>,
}
impl Armor {
    pub fn new(id: PlayerID) -> Box<Armor> {
        Box::new(Armor {
            id,
            foul_target: None,
        })
    }
    pub fn new_foul(id: PlayerID, target: Sum2D6Target, fouler_id: PlayerID) -> Box<Armor> {
        Box::new(Armor {
            id,
            foul_target: Some((fouler_id, target)),
        })
    }
}
impl Procedure for Armor {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let roll1 = game_state.get_d6_roll();
        let roll2 = game_state.get_d6_roll();
        let roll = roll1 + roll2;
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();
        let mut injury_proc = Injury::new(self.id);

        let target = if let Some((fouler_id, foul_target)) = self.foul_target {
            if roll1 == roll2 {
                procs.push(Ejection::new(fouler_id));
            } else {
                injury_proc.fouler = Some(fouler_id);
            }
            foul_target
        } else {
            game_state.get_player_unsafe(self.id).armor_target()
        };

        if target.is_success(roll) {
            procs.push(injury_proc);
        }

        ProcState::from(procs)
    }
}

#[derive(Debug)]
pub struct Ejection {
    id: PlayerID,
}
impl Ejection {
    pub fn new(id: PlayerID) -> Box<Ejection> {
        Box::new(Ejection { id })
    }
}
impl Procedure for Ejection {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let position = game_state.get_player_unsafe(self.id).position;
        game_state
            .unfield_player(self.id, DugoutPlace::Ejected)
            .unwrap();

        if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
            game_state.ball = BallState::InAir(position);
            ProcState::DoneNew(ball_procs::Bounce::new())
        } else {
            ProcState::Done
        }
    }
}

#[derive(Debug)]
pub struct Injury {
    id: PlayerID,
    crowd: bool,
    fouler: Option<PlayerID>,
}
impl Injury {
    pub fn new(id: PlayerID) -> Box<Injury> {
        Box::new(Injury {
            id,
            crowd: false,
            fouler: None,
        })
    }

    pub fn new_crowd(id: PlayerID) -> Box<Injury> {
        Box::new(Injury {
            id,
            crowd: true,
            fouler: None,
        })
    }
}
impl Procedure for Injury {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let cas_target = Sum2D6Target::TenPlus;
        let ko_target = Sum2D6Target::EightPlus;
        let roll1 = game_state.get_d6_roll();
        let roll2 = game_state.get_d6_roll();
        let roll = roll1 + roll2;
        let mut procs: Vec<Box<dyn Procedure>> = Vec::new();

        if self.fouler.is_some() && roll1 == roll2 {
            procs.push(Ejection::new(self.fouler.unwrap()))
        }

        if cas_target.is_success(roll) {
            game_state
                .unfield_player(self.id, DugoutPlace::Injuried)
                .unwrap();
        } else if ko_target.is_success(roll) {
            game_state
                .unfield_player(self.id, DugoutPlace::KnockOut)
                .unwrap();
        } else if self.crowd {
            game_state
                .unfield_player(self.id, DugoutPlace::Reserves)
                .unwrap();
        } else {
            game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Stunned;
        }
        ProcState::from(procs)
    }
}

#[derive(Debug)]
pub struct BlockAction {}

impl BlockAction {
    pub fn new() -> Box<BlockAction> {
        Box::new(BlockAction {})
    }
    fn available_actions(&mut self, game_state: &GameState) -> Box<AvailableActions> {
        let player = game_state.get_active_player().unwrap();
        let mut aa = AvailableActions::new(player.stats.team);

        game_state
            .get_adj_players(player.position)
            .filter(|adj_player| !adj_player.used && adj_player.stats.team != player.stats.team)
            .for_each(|block_victim| {
                aa.insert_block(
                    block_victim.position,
                    game_state.get_blockdices(player.id, block_victim.id),
                )
            });

        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }
}
impl Procedure for BlockAction {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        match action {
            None => ProcState::NeedAction(self.available_actions(game_state)),
            Some(Action::Positional(PosAT::Block, position)) => {
                let block_path = game_state.available_actions.take_path(position).unwrap();
                let num_dice = block_path.get_block_dice().unwrap();
                let defender_id = game_state.get_player_id_at(position).unwrap();
                ProcState::DoneNew(Block::new(num_dice, defender_id))
            }
            Some(Action::Simple(SimpleAT::EndPlayerTurn)) => todo!("EndPlayerTurn not implemented"),
            _ => panic!("Invalid action {:?}", action),
        }
    }
}

#[derive(Debug)]
pub struct Block {
    dices: NumBlockDices,
    defender: PlayerID,
    state: BlockProcState,
    roll: [Option<BlockDice>; 3],
    is_uphill: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockProcState {
    Init,               //step shall roll first dice
    SelectDice,         //attacker (or defender if uphill) to choose dice
    SelectDiceOrReroll, // Attacker may choose dice or reroll
    UphillSelectReroll, // In uphill, attacker may choose to reroll
}

impl Block {
    pub fn new(dices: NumBlockDices, defender: PlayerID) -> Box<Block> {
        // the point is that number of dices has already been calculated, so this proc doesn't need to redo it.
        Box::new(Block {
            dices,
            defender,
            state: BlockProcState::Init,
            roll: Default::default(),
            is_uphill: matches!(dices, NumBlockDices::TwoUphill | NumBlockDices::ThreeUphill),
        })
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
                aa.team = Some(if self.is_uphill {
                    other_team(team)
                } else {
                    team
                });
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        match action {
            None => {
                if game_state.info.player_action_type.unwrap() == PosAT::StartBlitz {
                    game_state.info.player_action_type = Some(PosAT::StartMove); //to preven the player from blitzing again
                    game_state.get_active_player_mut().unwrap().add_move(1);
                }
                for i in 0..u8::from(self.dices) {
                    self.roll[i as usize] = Some(game_state.get_block_dice_roll());
                }
                let reroll_available = game_state
                    .get_active_players_team()
                    .unwrap()
                    .can_use_reroll();
                self.state = match (reroll_available, self.is_uphill) {
                    (true, true) => BlockProcState::UphillSelectReroll,
                    (true, false) => BlockProcState::SelectDiceOrReroll,
                    (false, _) => BlockProcState::SelectDice,
                };
                ProcState::NeedAction(self.available_actions(game_state))
            }
            Some(Action::Simple(SimpleAT::UseReroll)) => {
                game_state
                    .get_active_players_team_mut()
                    .unwrap()
                    .use_reroll();
                ProcState::NotDone
            }
            Some(Action::Simple(SimpleAT::DontUseReroll)) => {
                self.state = BlockProcState::SelectDice;
                ProcState::NotDone
            }
            Some(Action::Simple(dice_action_type)) => {
                let attacker_id = game_state.info.active_player.unwrap();
                let mut knockdown_attacker = false;
                let mut knockdown_defender = false;
                let mut push = false;

                match dice_action_type {
                    SimpleAT::SelectBothDown => {
                        if !game_state
                            .get_active_player()
                            .unwrap()
                            .has_skill(Skill::Block)
                        {
                            knockdown_attacker = true;
                        }
                        if !game_state
                            .get_player_unsafe(self.defender)
                            .has_skill(Skill::Block)
                        {
                            knockdown_defender = true;
                        }
                    }
                    SimpleAT::SelectPow => {
                        knockdown_defender = true;
                        push = true;
                    }
                    SimpleAT::SelectPush => {
                        push = true;
                    }
                    SimpleAT::SelectPowPush => {
                        if !game_state
                            .get_player_unsafe(self.defender)
                            .has_skill(Skill::Dodge)
                        {
                            knockdown_defender = true;
                        }
                        push = true;
                    }

                    SimpleAT::SelectSkull => knockdown_attacker = true,
                    _ => panic!("very wrong!"),
                }
                let mut procs: Vec<Box<dyn Procedure>> = Vec::with_capacity(3);
                if knockdown_attacker {
                    procs.push(KnockDown::new(attacker_id));
                }
                if push {
                    let mut push_proc = Push::new(
                        game_state.get_player_unsafe(attacker_id).position,
                        game_state.get_player_unsafe(self.defender).position,
                    );
                    if knockdown_defender {
                        push_proc.knockdown_proc = Some(KnockDown::new(self.defender));
                    }
                    procs.push(push_proc);
                } else if knockdown_defender {
                    procs.push(KnockDown::new(self.defender));
                }
                ProcState::from(procs)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::dices::BlockDice;
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
        state.fixes.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, Position::new((6, 2)));
        state.step_simple(SimpleAT::SelectPush);

        state.step_positional(PosAT::Push, Position::new((6, 1)));
        state.fixes.fix_d6(1);
        state.fixes.fix_d6(1);

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
    #[test]
    fn blitz() {
        let start_pos = Position::new((2, 1));
        let target_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_away_player(target_pos)
            .build();
        state.step_positional(PosAT::StartBlitz, start_pos);

        state.fixes.fix_blockdice(BlockDice::Skull);
        state.step_positional(PosAT::Block, target_pos);
    }
}

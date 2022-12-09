use crate::core::{dices::D6, model};
use model::*;

use crate::core::table::*;

use super::{
    dices::{BlockDice, D6Target, RollTarget, Sum2D6Target},
    gamestate::GameState,
    pathing::{Path, PathFinder, Roll},
};

#[allow(unused_variables)]
trait SimpleProc {
    fn d6_target(&self) -> D6Target; //called immidiately before
    fn reroll_skill(&self) -> Option<Skill>;
    fn apply_success(&self, game_state: &mut GameState);
    fn apply_failure(&self, game_state: &mut GameState);
    fn player_id(&self) -> PlayerID;
}

pub struct Turn {
    pub team: TeamType,
}
impl Procedure for Turn {
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
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

        aa.insert_positional(PosAT::StartMove, positions.clone());
        aa.insert_positional(PosAT::StartHandoff, positions);
        aa.insert_simple(SimpleAT::EndTurn);
        aa
    }

    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(at, position)) => {
                game_state.set_active_player(game_state.get_player_id_at(position).unwrap());
                match at {
                    PosAT::StartMove => {
                        game_state.push_proc(MoveAction::new(game_state.active_player.unwrap()));
                    }
                    PosAT::StartBlock => {
                        game_state.push_proc(BlockAction::new());
                    }
                    _ => todo!(),
                }
                false
            }

            Some(Action::Simple(SimpleAT::EndTurn)) => true,
            _ => panic!("Action not allowed: {:?}", action),
        }
    }
}

fn proc_from_roll(roll: Roll, move_action: &MoveAction) -> Box<dyn Procedure> {
    match roll {
        Roll::Dodge(target) => DodgeProc::new(move_action.player_id, target),
        Roll::GFI(target) => GfiProc::new(move_action.player_id, target),
        Roll::Pickup(target) => PickupProc::new(move_action.player_id, target),
        Roll::Block(id, dices) => Block::new(dices, id),
    }
}

pub struct MoveAction {
    player_id: PlayerID,
    paths: FullPitch<Option<Path>>,
    active_path: Option<Path>,
    rolls: Option<Vec<Roll>>,
}
impl MoveAction {
    pub fn new(id: PlayerID) -> Box<MoveAction> {
        Box::new(MoveAction {
            player_id: id,
            paths: Default::default(),
            active_path: None,
            rolls: None,
        })
    }
    fn consolidate_active_path(&mut self) {
        if let Some(rolls) = &self.rolls {
            if !rolls.is_empty() {
                return;
            }
        }
        if let Some(path) = &self.active_path {
            if !path.steps.is_empty() {
                return;
            }
        }
        self.rolls = None;
        self.active_path = None;
    }

    fn continue_active_path(&mut self, game_state: &mut GameState) {
        let debug_roll_len_before = self.rolls.as_ref().map_or(0, |rolls| rolls.len());

        //are the rolls left to handle?
        if let Some(next_roll) = self.rolls.as_mut().and_then(|rolls| rolls.pop()) {
            let new_proc = proc_from_roll(next_roll, self);
            game_state.push_proc(new_proc);

            let debug_roll_len_after = self.rolls.as_ref().map_or(0, |rolls| rolls.len());
            assert_eq!(debug_roll_len_before - 1, debug_roll_len_after);

            return;
        }

        let path = self.active_path.as_mut().unwrap();

        // check if any rolls left to handle, if not then just move to end of path
        if path.steps.iter().all(|(_, rolls)| rolls.is_empty()) {
            //check if already there
            if let Some(id) = game_state.get_player_id_at(path.target) {
                debug_assert_eq!(id, self.player_id);
            } else {
                game_state.move_player(self.player_id, path.target).unwrap();
                game_state
                    .get_mut_player_unsafe(self.player_id)
                    .add_move(u8::try_from(path.steps.len()).unwrap())
            }
            path.steps.clear();
            return;
        }
        while let Some((position, mut rolls)) = path.steps.pop() {
            game_state.move_player(self.player_id, position).unwrap();
            game_state.get_mut_player_unsafe(self.player_id).add_move(1);

            if let Some(next_roll) = rolls.pop() {
                let new_proc = proc_from_roll(next_roll, self);
                game_state.push_proc(new_proc);
                if !rolls.is_empty() {
                    self.rolls = Some(rolls);
                }
                return;
            }
        }
    }
}
impl Procedure for MoveAction {
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let player = match game_state.get_player(self.player_id) {
            Ok(player) if !player.used => player,
            Ok(_) => return AvailableActions::new_empty(), // Player is used
            Err(_) => return AvailableActions::new_empty(), // Player not on field anymore
        };

        if self.active_path.is_some() {
            return AvailableActions::new_empty();
        }

        let mut aa = AvailableActions::new(player.stats.team);
        if player.total_movement_left() > 0 {
            self.paths = PathFinder::player_paths(game_state, self.player_id).unwrap();
            let move_positions = gimmi_iter(&self.paths)
                .flatten()
                .map(|path| path.target)
                .collect();

            aa.insert_positional(PosAT::Move, move_positions);
        }
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }

    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::Move, position)) => {
                if game_state.get_player_at(position).is_some() {
                    panic!("Very wrong!");
                }
                let (x, y) = position.to_usize().unwrap();
                self.active_path = self.paths[x][y].clone();
                debug_assert!(self.active_path.is_some());
                self.paths = Default::default();
                self.continue_active_path(game_state);
                self.consolidate_active_path();
                false
            }
            Some(Action::Simple(SimpleAT::EndPlayerTurn)) => {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
                true
            }
            None => {
                match game_state.get_player(self.player_id) {
                    Ok(player) if !player.used => (),
                    Ok(_) => return true,  // Player is used
                    Err(_) => return true, // Player not on field anymore
                }

                self.continue_active_path(game_state);
                self.consolidate_active_path();
                false
            }

            _ => panic!("very wrong!"),
        }
    }
}

#[allow(dead_code)]
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

    fn apply_success(&self, _game_state: &mut GameState) {}

    fn apply_failure(&self, game_state: &mut GameState) {
        game_state.push_proc(KnockDown::new(self.id))
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

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

    fn apply_success(&self, _game_state: &mut GameState) {}

    fn apply_failure(&self, game_state: &mut GameState) {
        game_state.push_proc(KnockDown::new(self.id));
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

struct PickupProc {
    target: D6Target,
    id: PlayerID,
}
impl PickupProc {
    fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<PickupProc>> {
        SimpleProcContainer::new(PickupProc { target, id })
    }
}
impl SimpleProc for PickupProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::SureHands)
    }

    fn apply_success(&self, game_state: &mut GameState) {
        game_state.ball = BallState::Carried(self.id);
    }

    fn apply_failure(&self, game_state: &mut GameState) {
        game_state.get_mut_player(self.id).unwrap().used = true;
        game_state.push_proc(Box::new(Bounce {}));
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RollProcState {
    Init,
    WaitingForTeamReroll,
    RerollUsed,
    //WaitingForSkillReroll,
}
struct SimpleProcContainer<T: SimpleProc> {
    proc: T,
    state: RollProcState,
}
impl<T: SimpleProc> SimpleProcContainer<T> {
    pub fn new(proc: T) -> Box<Self> {
        Box::new(SimpleProcContainer {
            proc,
            state: RollProcState::Init,
        })
    }
    pub fn id(&self) -> PlayerID {
        self.proc.player_id()
    }
}

impl<T> Procedure for SimpleProcContainer<T>
where
    T: SimpleProc,
{
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        // if action is DON*T REROLL, apply failure, return true
        match action {
            Some(Action::Simple(SimpleAT::DontUseReroll)) => {
                self.proc.apply_failure(game_state);
                return true;
            }
            Some(Action::Simple(SimpleAT::UseReroll)) => {
                game_state.get_active_team_mut().unwrap().use_reroll();
                self.state = RollProcState::RerollUsed;
            }
            _ => (),
        }

        loop {
            let roll = game_state.get_d6_roll();
            if self.proc.d6_target().is_success(roll) {
                self.proc.apply_success(game_state);
                return true;
            }
            if self.state == RollProcState::RerollUsed {
                break;
            }
            match self.proc.reroll_skill() {
                Some(skill) if game_state.get_player_unsafe(self.id()).can_use_skill(skill) => {
                    game_state.get_mut_player_unsafe(self.id()).use_skill(skill);
                    self.state = RollProcState::RerollUsed;
                    continue;
                }
                _ => (),
            }

            if game_state
                .get_team_from_player(self.id())
                .unwrap()
                .can_use_reroll()
            {
                self.state = RollProcState::WaitingForTeamReroll;
                return false;
            }
        }
        self.proc.apply_failure(game_state);
        true
    }
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        match self.state {
            RollProcState::Init => AvailableActions::new_empty(),
            RollProcState::WaitingForTeamReroll => {
                let mut aa =
                    AvailableActions::new(game_state.get_player_unsafe(self.id()).stats.team);
                aa.insert_simple(SimpleAT::UseReroll);
                aa.insert_simple(SimpleAT::DontUseReroll);
                aa
            }
            _ => panic!("Illegal state!"),
        }
    }
}

struct KnockDown {
    id: PlayerID,
}
impl KnockDown {
    pub fn new(id: PlayerID) -> Box<KnockDown> {
        Box::new(KnockDown { id })
    }
}
impl Procedure for KnockDown {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        debug_assert_eq!(action, None);
        let mut player = match game_state.get_mut_player(self.id) {
            Ok(player_) => player_,
            Err(_) => return true, //Means the player is already off the pitch, most likely crowd push
        };
        debug_assert!(matches!(player.status, PlayerStatus::Up));
        player.status = PlayerStatus::Down;
        player.used = true;
        if matches!(game_state.ball, BallState::Carried(carrier_id) if carrier_id == self.id) {
            game_state.push_proc(Bounce::new());
        }
        game_state.push_proc(Armor::new(self.id));
        true
    }
}

struct Armor {
    id: PlayerID,
}
impl Armor {
    pub fn new(id: PlayerID) -> Box<Armor> {
        Box::new(Armor { id })
    }
}
impl Procedure for Armor {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> bool {
        let player = game_state.get_player_unsafe(self.id);
        let target = player.armor_target();
        let roll = game_state.get_2d6_roll();
        if target.is_success(roll) {
            game_state.push_proc(Injury::new(self.id));
        }
        true
    }
}

struct Injury {
    id: PlayerID,
    crowd: bool,
}
impl Injury {
    pub fn new(id: PlayerID) -> Box<Injury> {
        Box::new(Injury { id, crowd: false })
    }

    pub fn new_crowd(id: PlayerID) -> Box<Injury> {
        Box::new(Injury { id, crowd: true })
    }
}
impl Procedure for Injury {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> bool {
        let cas_target = Sum2D6Target::TenPlus;
        let ko_target = Sum2D6Target::EightPlus;
        let roll = game_state.get_2d6_roll();
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
        true
    }
}

struct Bounce;
impl Bounce {
    pub fn new() -> Box<Bounce> {
        Box::new(Bounce {})
    }
}
impl Procedure for Bounce {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> bool {
        let current_ball_pos = game_state.get_ball_position().unwrap();
        let new_pos = current_ball_pos + Direction::from(game_state.get_d8_roll());

        if let Some(player) = game_state.get_player_at(new_pos) {
            if player.can_catch() {
                game_state.push_proc(Catch::new(
                    player.id,
                    game_state.get_catch_modifers(player.id).unwrap(),
                ));
                true
            } else {
                false //will run bounce again
            }
        } else if new_pos.is_out() {
            game_state.push_proc(Box::new(ThrowIn {
                from_position: current_ball_pos,
            }));
            true
        } else {
            game_state.ball = BallState::OnGround(new_pos);
            true
        }
    }
}
struct ThrowIn {
    from_position: Position,
}
impl Procedure for ThrowIn {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> bool {
        const MAX_X: Coord = HEIGHT_ - 2;
        const MAX_Y: Coord = WIDTH_ - 2;
        let directions: [(Coord, Coord); 3] = match self.from_position {
            Position { x: 1, y: 1 } => [(1, 0), (1, 1), (0, 1)],
            Position { x: 1, y: MAX_Y } => [(1, 0), (1, -1), (0, -1)],
            Position { x: MAX_X, y: 1 } => [(-1, 0), (-1, 1), (0, 1)],
            Position { x: MAX_X, y: MAX_Y } => [(-1, 0), (-1, -1), (0, -1)],
            Position { x: 1, .. } => [(1, 1), (1, 0), (1, -1)],
            Position { x: MAX_X, .. } => [(-1, 1), (-1, 0), (-1, -1)],
            Position { y: 1, .. } => [(1, 1), (0, 1), (-1, 1)],
            Position { y: MAX_Y, .. } => [(1, -1), (0, -1), (-1, -1)],
            _ => panic!("very wrong!"),
        };
        let direction = Direction::from(match game_state.get_d6_roll() {
            D6::One | D6::Two => directions[0],
            D6::Three | D6::Four => directions[1],
            D6::Five | D6::Six => directions[2],
        });

        let length = game_state.get_2d6_roll() as i8;
        let target: Position = self.from_position + direction * length;

        if target.is_out() {
            self.from_position = target - direction;

            while self.from_position.is_out() {
                self.from_position -= direction;
            }

            return false;
        }

        match game_state.get_player_at(target) {
            Some(player) if player.can_catch() => game_state.push_proc(Catch::new(
                player.id,
                game_state.get_catch_modifers(player.id).unwrap(),
            )),
            _ => {
                game_state.ball = BallState::InAir(target);
                game_state.push_proc(Bounce::new());
            }
        }
        true
    }
}
struct Catch {
    id: PlayerID,
    target: D6Target,
}
impl Catch {
    pub fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<Catch>> {
        SimpleProcContainer::new(Catch { id, target })
    }
}
impl SimpleProc for Catch {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Catch)
    }

    fn apply_success(&self, game_state: &mut GameState) {
        game_state.ball = BallState::Carried(self.id);
    }

    fn apply_failure(&self, game_state: &mut GameState) {
        game_state.push_proc(Box::new(Bounce {}));
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

struct BlockAction {}

impl BlockAction {
    fn new() -> Box<BlockAction> {
        Box::new(BlockAction {})
    }
}
impl Procedure for BlockAction {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::Block, position)) => {
                let ac = game_state
                    .get_available_actions()
                    .blocks
                    .iter()
                    .find(|ac| ac.position == position)
                    .unwrap();
                let defender_id = game_state.get_player_id_at(position).unwrap();
                game_state.push_proc(Block::new(ac.num_dices, defender_id));
            }
            _ => todo!(),
        }
        true
    }

    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let player = game_state.get_active_player().unwrap();
        let mut aa = AvailableActions::new(player.stats.team);

        let ac: Vec<BlockActionChoice> = game_state
            .get_adj_players(player.position)
            .filter(|adj_player| !adj_player.used && adj_player.stats.team != player.stats.team)
            .map(|block_victim| BlockActionChoice {
                num_dices: game_state.get_blockdices(player.id, block_victim.id),
                position: block_victim.position,
            })
            .collect();

        aa.insert_block(ac);
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }
}

struct Block {
    dices: NumBlockDices,
    defender: PlayerID,
    state: BlockProcState,
    roll: [Option<BlockDice>; 3],
    is_uphill: bool,
    //attacker is game_state.active_player()
    //is_blitz: bool //prepare for Horns, Juggernaught, etc..
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockProcState {
    Init,               //step shall roll first dice
    SelectDice,         //attacker (or defender if uphill) to choose dice
    SelectDiceOrReroll, // Attacker may choose dice or reroll
    UphillSelectReroll, // In uphill, attacker may choose to reroll
}

impl Block {
    fn new(dices: NumBlockDices, defender: PlayerID) -> Box<Block> {
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
}
impl Procedure for Block {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            None => {
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
                false
            }
            Some(Action::Simple(SimpleAT::UseReroll)) => {
                self.state = BlockProcState::Init;
                game_state
                    .get_active_players_team_mut()
                    .unwrap()
                    .use_reroll();
                false
            }
            Some(Action::Simple(SimpleAT::DontUseReroll)) => {
                self.state = BlockProcState::SelectDice;
                false
            }
            Some(Action::Simple(dice_action_type)) => {
                let attacker_id = game_state.active_player.unwrap();
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
                if knockdown_attacker {
                    game_state.push_proc(KnockDown::new(attacker_id));
                }
                if knockdown_defender {
                    game_state.push_proc(KnockDown::new(self.defender));
                }
                if push {
                    game_state.push_proc(FollowUp::new(
                        game_state.get_player_unsafe(self.defender).position,
                    ));
                    game_state.push_proc(Push::new(
                        game_state.get_player_unsafe(attacker_id).position,
                        game_state.get_player_unsafe(self.defender).position,
                    ));
                }
                true
            }
            _ => panic!("very wrong!"),
        }
    }
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let mut aa = AvailableActions::new_empty();
        let team = game_state.get_active_player().unwrap().stats.team;
        match self.state {
            BlockProcState::Init => {}
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
        }
        aa
    }
}

enum PushState {
    NotDecided,
    Square(Position),
    Crowd,
}
struct Push {
    from: Position,
    on: Position,
    target: PushState,
}

impl Push {
    fn new(from: Position, on: Position) -> Box<Push> {
        Box::new(Push {
            from,
            on,
            target: PushState::NotDecided,
        })
    }
    fn get_push_squares(&self, game_state: &GameState) -> Vec<Position> {
        let direction = self.on - self.from;
        let mut push_squares = match direction {
            Direction { dx: 0, dy: _ } => vec![self.on + (1, 0), self.on + (-1, 0)],
            Direction { dx: _, dy: 0 } => vec![self.on + (0, 1), self.on + (0, -1)],
            Direction { dx, dy } => vec![self.on + (dx, 0), self.on + (0, dy)],
        };
        push_squares.push(self.on + direction);
        let free_squares: Vec<Position> = push_squares
            .iter()
            .filter(|&pos| !pos.is_out() && game_state.get_player_at(*pos).is_none())
            .copied()
            .collect();

        if !free_squares.is_empty() {
            free_squares
        } else if push_squares.iter().any(|&pos| pos.is_out()) {
            Vec::new()
        } else {
            push_squares
        }
    }
    fn do_move(&self, game_state: &mut GameState) {
        let id = game_state.get_player_id_at(self.on).unwrap();
        match self.target {
            PushState::Square(to) => game_state.move_player(id, to).unwrap(),
            PushState::Crowd => game_state.push_proc(Injury::new_crowd(id)),
            _ => panic!("very wrong!"),
        }
    }
    fn handle_selected_square(&mut self, game_state: &mut GameState, position: Position) -> bool {
        self.target = PushState::Square(position);
        match game_state.get_player_id_at(position) {
            Some(_chain_push_id) => {
                game_state.push_proc(Push::new(self.on, position));
                false
            }
            None => {
                self.do_move(game_state);
                true
            }
        }
    }
}
impl Procedure for Push {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        match action {
            Some(Action::Positional(PosAT::Push, position)) => {
                self.handle_selected_square(game_state, position)
            }
            None => {
                self.do_move(game_state);
                true
            }
            _ => panic!("very wrong!"),
        }
    }

    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        match self.target {
            PushState::NotDecided => {
                let push_squares = self.get_push_squares(game_state);
                if push_squares.is_empty() {
                    self.target = PushState::Crowd;
                    AvailableActions::new_empty()
                } else {
                    let mut aa = AvailableActions::new(game_state.team_turn);
                    aa.insert_positional(PosAT::Push, push_squares);
                    aa
                }
            }
            _ => AvailableActions::new_empty(),
        }
    }
}

struct FollowUp {
    to: Position,
    //from is active player,
}
impl FollowUp {
    fn new(to: Position) -> Box<FollowUp> {
        Box::new(FollowUp { to })
    }
}
impl Procedure for FollowUp {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        let player = game_state.get_active_player().unwrap();
        match action {
            Some(Action::Positional(PosAT::FollowUp, position)) => {
                if player.position != position {
                    game_state.move_player(player.id, position).unwrap();
                }
                true
            }
            _ => panic!("very wrong!"),
        }
    }

    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let player = game_state.get_active_player().unwrap();
        let mut aa = AvailableActions::new(player.stats.team);
        aa.insert_positional(PosAT::FollowUp, vec![player.position, self.to]);
        aa
    }
}

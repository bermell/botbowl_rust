use crate::core::model;
use model::*;

use crate::core::table::*;

use super::{
    dices::{D6Target, RollTarget, Sum2D6Target},
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
            .map(|&pos| pos)
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
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> bool {
        let mut player = game_state.get_mut_player_unsafe(self.id);
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
}
impl Injury {
    pub fn new(id: PlayerID) -> Box<Injury> {
        Box::new(Injury { id })
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
        let new_pos = current_ball_pos + Position::from(game_state.get_d8_roll());

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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        todo!()
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
        // find the corresponding BlockActionChoice here, push block procedure
        todo!()
    }

    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let player = game_state.get_active_player().unwrap();
        let aa = AvailableActions::new(player.stats.team);
        // construct the Vec<BlockActionChoice> here
        aa
    }
}

struct Block {
    dices: NumBlockDices,
    //is_blitz: bool //prepare for Horns, Juggernaught, etc..
}

impl Block {
    fn new(dices: NumBlockDices) -> Box<Block> {
        // the point is that number of dices has already been calculated, so this proc doesn't need to redo it.
        Box::new(Block { dices })
    }
}
impl Procedure for Block {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> bool {
        todo!()
    }
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        AvailableActions::new_empty()
    }
}

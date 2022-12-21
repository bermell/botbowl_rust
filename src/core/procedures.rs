use crate::core::{dices::D6, model};
use model::*;

use crate::core::table::*;

use super::{
    dices::{BlockDice, D6Target, RollTarget, Sum2D6Target},
    gamestate::GameState,
    pathing::{FixedQueue, Path, PathFinder, PathingEvent},
};

#[allow(unused_variables)]
trait SimpleProc {
    fn d6_target(&self) -> D6Target; //called immidiately before
    fn reroll_skill(&self) -> Option<Skill>;
    fn apply_success(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        Vec::new()
    }
    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>>;
    fn player_id(&self) -> PlayerID;
}
impl From<Vec<Box<dyn Procedure>>> for ProcState {
    fn from(procs: Vec<Box<dyn Procedure>>) -> Self {
        match procs.len() {
            0 => ProcState::Done,
            // 1 => ProcState::DoneNew(procs.pop().unwrap()),
            _ => ProcState::DoneNewProcs(procs),
        }
    }
}

pub struct Half {
    half: u8,
    started: bool,
}
impl Half {
    pub fn new(half: u8) -> Box<Half> {
        debug_assert!(half == 1 || half == 2);
        Box::new(Half {
            half,
            started: false,
        })
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
        }

        if info.home_turn == 8 && info.away_turn == 8 {
            return ProcState::Done;
        }

        let kicking_this_half = if self.half == 1 {
            info.kicking_first_half
        } else {
            other_team(info.kicking_first_half)
        };

        let next_team: TeamType = if info.home_turn == info.away_turn {
            kicking_this_half
        } else {
            other_team(kicking_this_half)
        };

        match next_team {
            TeamType::Home => info.home_turn += 1,
            TeamType::Away => info.away_turn += 1,
        }

        info.team_turn = next_team;
        info.handoff_available = true;
        info.blitz_available = true;
        info.pass_available = true;

        ProcState::NotDoneNew(Turn::new(next_team))
    }
}

pub struct Turn {
    pub team: TeamType,
}
impl Turn {
    pub fn new(team: TeamType) -> Box<Turn> {
        Box::new(Turn { team })
    }
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
        if game_state.info.handoff_available {
            aa.insert_positional(PosAT::StartHandoff, positions.clone());
        }

        if game_state.info.blitz_available {
            aa.insert_positional(PosAT::StartBlitz, positions.clone());
        }

        if game_state.info.pass_available {
            aa.insert_positional(PosAT::StartPass, positions.clone());
        }

        aa.insert_positional(PosAT::StartMove, positions);
        aa.insert_simple(SimpleAT::EndTurn);
        aa
    }

    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if let Some(id) = game_state.info.handle_td_by {
            //todo, set internal state to kickoff next (or if it was the last turn return done )
            game_state.info.handle_td_by = None;
            return ProcState::NotDoneNew(Touchdown::new(id));
        }

        game_state.info.active_player = None;
        game_state.info.player_action_type = None;
        if action.is_none() {
            return ProcState::NeedAction(self.available_actions(game_state));
        }

        if let Some(Action::Positional(_, position)) = action {
            game_state.set_active_player(game_state.get_player_id_at(position).unwrap());
        }

        match action.unwrap() {
            Action::Positional(PosAT::StartMove, _) => {
                game_state.info.player_action_type = Some(PlayerActionType::MoveAction);
                ProcState::NotDoneNew(MoveAction::new(game_state.info.active_player.unwrap()))
            }
            Action::Positional(PosAT::StartHandoff, _) => {
                game_state.info.player_action_type = Some(PlayerActionType::HandoffAction);
                game_state.info.handoff_available = false;
                ProcState::NotDoneNew(MoveAction::new(game_state.info.active_player.unwrap()))
            }
            Action::Positional(PosAT::StartBlitz, _) => {
                game_state.info.player_action_type = Some(PlayerActionType::BlitzAction);
                game_state.info.blitz_available = false;
                ProcState::NotDoneNew(MoveAction::new(game_state.info.active_player.unwrap()))
            }
            Action::Positional(PosAT::StartBlock, _) => {
                game_state.info.player_action_type = Some(PlayerActionType::BlockAction);
                ProcState::NotDoneNew(BlockAction::new())
            }

            Action::Simple(SimpleAT::EndTurn) => ProcState::Done,
            _ => panic!("Action not allowed: {:?}", action),
        }
    }
}

fn proc_from_roll(roll: PathingEvent, move_action: &MoveAction) -> Box<dyn Procedure> {
    match roll {
        PathingEvent::Dodge(target) => DodgeProc::new(move_action.player_id, target),
        PathingEvent::GFI(target) => GfiProc::new(move_action.player_id, target),
        PathingEvent::Pickup(target) => PickupProc::new(move_action.player_id, target),
        PathingEvent::Block(id, dices) => Block::new(dices, id),
        PathingEvent::Handoff(id, target) => Catch::new(id, target),
        PathingEvent::Touchdown(id) => Touchdown::new(id),
    }
}

#[allow(clippy::large_enum_variant)]
enum MoveActionState {
    Init,
    ActivePath(Path),
    SelectPath(FullPitch<Option<Path>>),
}
pub struct MoveAction {
    player_id: PlayerID,
    state: MoveActionState,
    // paths: FullPitch<Option<Path>>,
    // active_path: Option<Path>,
    rolls: Option<FixedQueue<PathingEvent>>,
}
impl MoveAction {
    pub fn new(id: PlayerID) -> Box<MoveAction> {
        Box::new(MoveAction {
            state: MoveActionState::Init,
            player_id: id,
            // paths: Default::default(),
            // active_path: None,
            rolls: None,
        })
    }
    fn consolidate_active_path(&mut self) {
        if let Some(rolls) = &self.rolls {
            if !rolls.is_empty() {
                return;
            }
        }
        if let MoveActionState::ActivePath(path) = &self.state {
            if !path.steps.is_empty() {
                return;
            }
        }
        self.rolls = None;
        self.state = MoveActionState::Init;
    }

    fn continue_active_path(&mut self, game_state: &mut GameState) -> ProcState {
        let debug_roll_len_before = self.rolls.as_ref().map_or(0, |rolls| rolls.len());

        //are the rolls left to handle?
        if let Some(next_roll) = self.rolls.as_mut().and_then(|rolls| rolls.pop()) {
            let new_proc = proc_from_roll(next_roll, self);

            let debug_roll_len_after = self.rolls.as_ref().map_or(0, |rolls| rolls.len());
            assert_eq!(debug_roll_len_before - 1, debug_roll_len_after);

            self.consolidate_active_path();
            return ProcState::NotDoneNew(new_proc);
        }

        let path = if let MoveActionState::ActivePath(path) = &mut self.state {
            path
        } else {
            panic!()
        };

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
            self.consolidate_active_path();
            return ProcState::NotDone;
        }

        //todo: need to consolidate the roll handling below and above to avoid code duplication.
        //      no 100% sure it's needed

        while let Some((position, mut rolls)) = path.steps.pop() {
            if let Some(PathingEvent::Handoff(_, _)) = rolls.last() {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
            } else if let Some(PathingEvent::Block(_, _)) = rolls.last() {
                game_state.get_mut_player_unsafe(self.player_id).add_move(1);
            } else {
                game_state.move_player(self.player_id, position).unwrap();
                game_state.get_mut_player_unsafe(self.player_id).add_move(1);
            }
            if let Some(next_roll) = rolls.pop() {
                let new_proc = proc_from_roll(next_roll, self);
                if !rolls.is_empty() {
                    self.rolls = Some(rolls);
                }
                self.consolidate_active_path();
                return ProcState::NotDoneNew(new_proc);
            }
        }
        panic!("Should not get here!");
    }
}
impl Procedure for MoveAction {
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
        let player = game_state.get_player_unsafe(self.player_id);

        let mut aa = AvailableActions::new(player.stats.team);
        if player.total_movement_left() > 0 {
            let paths = PathFinder::player_paths(game_state, self.player_id).unwrap();
            gimmi_iter(&paths)
                .flatten()
                .for_each(|path| aa.insert_path(path));

            self.state = MoveActionState::SelectPath(paths);
        }
        aa.insert_simple(SimpleAT::EndPlayerTurn);
        aa
    }

    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if game_state.info.handle_td_by.is_some() {
            game_state.get_mut_player_unsafe(self.player_id).used = true;
            return ProcState::Done;
        }

        match game_state.get_player(self.player_id) {
            Ok(player) if player.used => {
                return ProcState::Done;
            }
            Err(_) => {
                // Player not on field anymore
                return ProcState::Done;
            }
            _ => (),
        }

        match (action, &mut self.state) {
            (Some(Action::Positional(_, position)), MoveActionState::SelectPath(all_paths)) => {
                self.state =
                    MoveActionState::ActivePath(all_paths.get_mut(position).take().unwrap());
                self.continue_active_path(game_state)
            }
            (Some(Action::Simple(SimpleAT::EndPlayerTurn)), _) => {
                game_state.get_mut_player_unsafe(self.player_id).used = true;
                ProcState::Done
            }
            (None, MoveActionState::ActivePath(_)) => self.continue_active_path(game_state),
            (None, MoveActionState::Init) => {
                ProcState::NeedAction(self.available_actions(game_state))
            }

            _ => panic!("very wrong!"),
        }
    }
}

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

    fn apply_failure(&self, _game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        vec![KnockDown::new(self.id)]
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

    fn apply_failure(&self, _game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        vec![KnockDown::new(self.id)]
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

    fn apply_success(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.ball = BallState::Carried(self.id);
        let player = game_state.get_player_unsafe(self.id);
        if player.position.x == game_state.get_endzone_x(player.stats.team) {
            game_state.info.handle_td_by = Some(self.id);
        }
        Vec::new()
    }

    fn apply_failure(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.get_mut_player(self.id).unwrap().used = true;
        vec![Box::new(Bounce {})]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RollProcState {
    Init,
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        // if action is DON*T REROLL, apply failure, return true
        match action {
            Some(Action::Simple(SimpleAT::DontUseReroll)) => {
                return ProcState::from(self.proc.apply_failure(game_state));
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
                return ProcState::from(self.proc.apply_success(game_state));
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
                let team = game_state.get_player_unsafe(self.id()).stats.team;
                let mut aa = AvailableActions::new(team);
                aa.insert_simple(SimpleAT::UseReroll);
                aa.insert_simple(SimpleAT::DontUseReroll);
                return ProcState::NeedAction(aa);
            } else {
                break;
            }
        }
        ProcState::from(self.proc.apply_failure(game_state))
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
            ProcState::DoneNewProcs(vec![Bounce::new(), armor_proc])
        } else {
            ProcState::DoneNew(armor_proc)
        }
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
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let target = game_state.get_player_unsafe(self.id).armor_target();
        let roll = game_state.get_2d6_roll();
        if target.is_success(roll) {
            ProcState::DoneNew(Injury::new(self.id))
        } else {
            ProcState::Done
        }
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
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
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
        ProcState::Done
    }
}

struct Bounce;
impl Bounce {
    pub fn new() -> Box<Bounce> {
        Box::new(Bounce {})
    }
}
impl Procedure for Bounce {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let current_ball_pos = game_state.get_ball_position().unwrap();
        let dice = game_state.get_d8_roll();
        let dir = Direction::from(dice);
        let new_pos = current_ball_pos + dir;

        if let Some(player) = game_state.get_player_at(new_pos) {
            if player.can_catch() {
                ProcState::DoneNew(Catch::new(
                    player.id,
                    game_state.get_catch_modifers(player.id).unwrap(),
                ))
            } else {
                //will run bounce again
                game_state.ball = BallState::InAir(new_pos);
                ProcState::NotDone
            }
        } else if new_pos.is_out() {
            ProcState::DoneNew(Box::new(ThrowIn {
                from: current_ball_pos,
            }))
        } else {
            game_state.ball = BallState::OnGround(new_pos);
            ProcState::Done
        }
    }
}
struct ThrowIn {
    from: Position,
}
impl ThrowIn {
    pub fn new(from: Position) -> Box<ThrowIn> {
        Box::new(ThrowIn { from })
    }
}
impl Procedure for ThrowIn {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        const MAX_X: Coord = HEIGHT_ - 2;
        const MAX_Y: Coord = WIDTH_ - 2;
        let directions: [(Coord, Coord); 3] = match self.from {
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
        let target: Position = self.from + direction * length;

        if target.is_out() {
            self.from = target - direction;

            while self.from.is_out() {
                self.from -= direction;
            }

            ProcState::NotDone
        } else {
            match game_state.get_player_at(target) {
                Some(player) if player.can_catch() => ProcState::DoneNew(Catch::new(
                    player.id,
                    game_state.get_catch_modifers(player.id).unwrap(),
                )),
                _ => {
                    game_state.ball = BallState::InAir(target);
                    ProcState::DoneNew(Bounce::new())
                }
            }
        }
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

    fn apply_success(&self, game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        game_state.ball = BallState::Carried(self.id);
        let player = game_state.get_player_unsafe(self.id);
        if player.position.x == game_state.get_endzone_x(player.stats.team) {
            game_state.info.handle_td_by = Some(self.id);
        }
        Vec::new()
    }

    fn apply_failure(&self, _game_state: &mut GameState) -> Vec<Box<dyn Procedure>> {
        vec![Box::new(Bounce {})]
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        match action {
            None => ProcState::NeedAction(self.available_actions(game_state)),
            Some(Action::Positional(PosAT::Block, position)) => {
                let ac = game_state
                    .get_available_actions()
                    .blocks
                    .iter()
                    .find(|ac| ac.position == position)
                    .unwrap();
                let defender_id = game_state.get_player_id_at(position).unwrap();
                ProcState::DoneNew(Block::new(ac.num_dices, defender_id))
            }
            _ => todo!(),
        }
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
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
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
            _ => panic!("very wrong!"),
        }
    }
    fn available_actions(&mut self, game_state: &GameState) -> AvailableActions {
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

enum PushSquares {
    Crowd(Position),
    ChainPush(Vec<Position>),
    FreeSquares(Vec<Position>),
}
struct Push {
    from: Position,
    on: Position,
    knockdown_proc: Option<Box<dyn Procedure>>,
    moves_to_make: Vec<(Position, Position)>,
    follow_up_pos: Position,
}

impl Push {
    fn new(from: Position, on: Position) -> Box<Push> {
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
                procs.push(ThrowIn::new(last_push_from));
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

struct Touchdown {
    id: PlayerID,
}
impl Touchdown {
    fn new(id: PlayerID) -> Box<Touchdown> {
        Box::new(Touchdown { id })
    }
}
impl Procedure for Touchdown {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        if let BallState::Carried(carrier_id) = game_state.ball {
            if carrier_id == self.id {
                game_state.get_mut_team_from_player(self.id).unwrap().score += 1;
                game_state.get_mut_player_unsafe(self.id).used = true;
            }
        }

        ProcState::Done
    }
}

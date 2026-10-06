use serde::{Deserialize, Serialize};

use crate::core::dices::{BlockDice, D6Target, RequestedRoll, RollResult, RollTarget};
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, Direction, PlayerStatus, Position, ProcState, Procedure,
};
use crate::core::model::{BallState, FieldedPlayer, PlayerID, ProcInput, TeamType};
use crate::core::procedures::ball_procs;
use crate::core::procedures::casualty_procs;
use crate::core::procedures::movement_procs;
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::table::{NumBlockDices, PosAT, SimpleAT, Skill};

use super::AnyProc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum PushSquares {
    Crowd(Position),
    ChainPush(Vec<Position>),
    FreeSquares(Vec<Position>),
}

/// An optional skill of the player being pushed, asked about before the push square is picked.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum PushSkill {
    StandFirm,
    SideStep,
}

/// What the player currently being pushed (`Push::on`) has been asked. Starts over for each
/// player a chain push reaches.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct PushQuestions {
    stand_firm_asked: bool,
    /// `Some(used)` once Sidestep has been asked about.
    sidestep: Option<bool>,
    /// The question waiting for an answer.
    pending: Option<PushSkill>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Push {
    from: Position,
    on: Position,
    knockdown_proc: Option<KnockDown>,
    moves_to_make: Vec<(Position, Position)>,
    follow_up_pos: Position,
    #[serde(default)]
    questions: PushQuestions,
    /// Strip Ball: this player, if carrying the ball, drops it in the square they are pushed
    /// into.
    #[serde(default)]
    strip_ball: Option<PlayerID>,
    /// Fend: this player, the one blocked, may stop the blocker following up.
    #[serde(default)]
    fend: Option<PlayerID>,
    /// Grab: the blocker's target can't Sidestep, and outside a Blitz may be pushed into any
    /// free square next to them. Only the target: cleared when a chain push moves on.
    #[serde(default)]
    grab: bool,
    /// Juggernaut in a Blitz: the blocker's target can't Stand Firm. Cleared when a chain push
    /// moves on.
    #[serde(default)]
    juggernaut: bool,
}

impl Push {
    pub fn new(from: Position, on: Position) -> AnyProc {
        AnyProc::Push(Push {
            from,
            on,
            moves_to_make: Vec::with_capacity(1),
            knockdown_proc: None,
            follow_up_pos: on,
            questions: PushQuestions::default(),
            strip_ball: None,
            fend: None,
            grab: false,
            juggernaut: false,
        })
    }
    pub fn new_pure(from: Position, on: Position) -> Push {
        Push {
            from,
            on,
            moves_to_make: Vec::with_capacity(1),
            knockdown_proc: None,
            follow_up_pos: on,
            questions: PushQuestions::default(),
            strip_ball: None,
            fend: None,
            grab: false,
            juggernaut: false,
        }
    }

    /// Would pushing the player standing `on` away from `from` send them
    /// into the crowd? True exactly when no push square is free and at
    /// least one is out of bounds — the case `calculate_next_state`
    /// resolves without asking for a square. A Sidestep player with a free
    /// square next to them is assumed to use it: nobody picks the crowd over
    /// a free square. Read-only; used by the MCTS block-outcome model to fold
    /// "push into the crowd" into the defender-removed outcome.
    pub fn is_crowd_push(from: Position, on: Position, game_state: &GameState) -> bool {
        let grab = game_state.get_player_at(from).is_some_and(|p| p.has_skill(Skill::Grab));
        if !grab && Push::sidestep_squares(on, game_state).is_some() {
            return false;
        }
        matches!(Push::get_push_squares(on, from, game_state), PushSquares::Crowd(_))
    }

    /// The free squares a Sidestep player standing `on` may step into, if they have the skill
    /// and any square next to them is free.
    fn sidestep_squares(on: Position, game_state: &GameState) -> Option<Vec<Position>> {
        if !game_state.get_player_at(on)?.has_skill(Skill::SideStep) {
            return None;
        }
        let free = Push::free_squares_next_to(on, game_state);
        (!free.is_empty()).then_some(free)
    }

    fn free_squares_next_to(on: Position, game_state: &GameState) -> Vec<Position> {
        game_state
            .get_adj_positions(on)
            .filter(|&pos| !game_state.is_out(pos) && game_state.get_player_at(pos).is_none())
            .collect()
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
        } else if let Some(id) = self
            .strip_ball
            .filter(|&id| matches!(game_state.ball, BallState::Carried(c) if c == id))
        {
            // Strip Ball: the carrier drops the ball where they were pushed to, and it bounces.
            let position = game_state.get_player_unsafe(id).position;
            game_state.set_ball(BallState::InAir(position));
            procs.push(ball_procs::Bounce::new());
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

    /// Stand Firm stopped the push. Strip Ball still works: the carrier drops the ball in their
    /// own square, and it bounces.
    fn strip_in_place(&self, game_state: &mut GameState) -> ProcState {
        match self.strip_ball {
            Some(id) if game_state.ball == BallState::Carried(id) => {
                let position = game_state.get_player_unsafe(id).position;
                game_state.set_ball(BallState::InAir(position));
                ProcState::DoneNew(ball_procs::Bounce::new())
            }
            _ => ProcState::Done,
        }
    }

    /// Ask the pushed player's coach about `skill`.
    fn ask(&mut self, skill: PushSkill, game_state: &GameState) -> ProcState {
        self.questions.pending = Some(skill);
        let mut aa = AvailableActions::new(game_state.get_player_at(self.on).unwrap().stats.team);
        aa.insert_simple(SimpleAT::UseSkill);
        aa.insert_simple(SimpleAT::DontUseSkill);
        ProcState::NeedAction(aa)
    }

    fn calculate_next_state(&mut self, game_state: &mut GameState) -> ProcState {
        let pushed = game_state.get_player_at(self.on).unwrap();
        // Stand Firm: the pushed player's coach may refuse the push — the blocked player's, or
        // any a chain push reaches. Then nobody is pushed at all, so there is no follow-up; a
        // knockdown (queued under this proc) still happens, in place.
        if !self.questions.stand_firm_asked {
            self.questions.stand_firm_asked = true;
            if pushed.has_skill(Skill::StandFirm) && !self.juggernaut {
                return self.ask(PushSkill::StandFirm, game_state);
            }
        }
        // Sidestep: the pushed player's coach may pick any free adjacent square instead.
        let sidestep = Push::sidestep_squares(self.on, game_state).filter(|_| !self.grab);
        if let Some(squares) = sidestep {
            match self.questions.sidestep {
                None => return self.ask(PushSkill::SideStep, game_state),
                Some(true) => return Push::pick(game_state.get_player_at(self.on).unwrap().stats.team, squares),
                Some(false) => (),
            }
        }
        let mut aa = AvailableActions::new(game_state.info.team_turn);
        match Push::get_push_squares(self.on, self.from, game_state) {
            PushSquares::Crowd(position_in_crowd) => {
                self.moves_to_make.push((self.on, position_in_crowd));
                self.do_moves(game_state);
                ProcState::NotDoneNew(FollowUp::new(self.follow_up_pos, self.fend))
            }
            PushSquares::ChainPush(mut positions) | PushSquares::FreeSquares(mut positions) => {
                if self.grab && !game_state.info.blitz_this_activation {
                    for square in Push::free_squares_next_to(self.on, game_state) {
                        if !positions.contains(&square) {
                            positions.push(square);
                        }
                    }
                }
                aa.insert_positional(PosAT::Push, positions);
                ProcState::NeedAction(aa)
            }
        }
    }

    fn pick(team: TeamType, squares: Vec<Position>) -> ProcState {
        let mut aa = AvailableActions::new(team);
        aa.insert_positional(PosAT::Push, squares);
        ProcState::NeedAction(aa)
    }
}

impl Procedure for Push {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Action(Action::Simple(answer @ (SimpleAT::UseSkill | SimpleAT::DontUseSkill))) => {
                let used = answer == SimpleAT::UseSkill;
                match self.questions.pending.take() {
                    Some(PushSkill::StandFirm) if used => self.strip_in_place(game_state),
                    Some(PushSkill::StandFirm) => self.calculate_next_state(game_state),
                    Some(PushSkill::SideStep) => {
                        self.questions.sidestep = Some(used);
                        self.calculate_next_state(game_state)
                    }
                    None => panic!("{answer:?} with no skill question pending"),
                }
            }
            ProcInput::Nothing if self.moves_to_make.is_empty() => self.calculate_next_state(game_state),
            ProcInput::Nothing => self.handle_aftermath(game_state),
            ProcInput::Action(Action::Positional(PosAT::Push, position_to))
                if game_state.get_player_at(position_to).is_some() =>
            {
                self.moves_to_make.push((self.on, position_to));
                self.from = self.on;
                self.on = position_to;
                self.questions = PushQuestions::default();
                self.grab = false;
                self.juggernaut = false;
                self.calculate_next_state(game_state)
            }
            ProcInput::Action(Action::Positional(PosAT::Push, position)) => {
                self.moves_to_make.push((self.on, position));
                self.do_moves(game_state);
                ProcState::NotDoneNew(FollowUp::new(self.follow_up_pos, self.fend))
            }
            _ => panic!("very wrong!"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FollowUp {
    to: Position,
    //from is active player,
    /// The pushed player with Fend, until their coach has answered.
    #[serde(default)]
    fend: Option<PlayerID>,
}
impl FollowUp {
    pub fn new(to: Position, fend: Option<PlayerID>) -> AnyProc {
        AnyProc::FollowUp(FollowUp { to, fend })
    }
}
impl Procedure for FollowUp {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let player = game_state.get_active_player().unwrap();
        match input {
            ProcInput::Nothing if self.fend.is_some() => {
                let fender = game_state.get_player_unsafe(self.fend.unwrap());
                let mut aa = AvailableActions::new(fender.stats.team);
                aa.insert_simple(SimpleAT::UseSkill);
                aa.insert_simple(SimpleAT::DontUseSkill);
                ProcState::NeedAction(aa)
            }
            // Fend: no follow-up, even for Frenzy.
            ProcInput::Action(Action::Simple(SimpleAT::UseSkill)) => ProcState::Done,
            ProcInput::Action(Action::Simple(SimpleAT::DontUseSkill)) => {
                self.fend = None;
                self.step(game_state, ProcInput::Nothing)
            }
            ProcInput::Nothing => {
                let mut aa = AvailableActions::new(player.stats.team);
                if player.has_skill(Skill::Frenzy) {
                    // Frenzy must follow up.
                    aa.insert_positional(PosAT::FollowUp, vec![self.to]);
                } else {
                    aa.insert_positional(PosAT::FollowUp, vec![player.position, self.to]);
                }
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
    /// `id` was knocked down by a block from a player with Mighty Blow.
    #[serde(default)]
    mighty_blow: bool,
    /// `second_id` (the blocker) was knocked down by a block against a player with Mighty Blow.
    #[serde(default)]
    second_mighty_blow: bool,
}
impl KnockDown {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::KnockDown(KnockDown {
            id: Some(id),
            second_id: None,
            mighty_blow: false,
            second_mighty_blow: false,
        })
    }
    pub fn new_pure(id: PlayerID) -> KnockDown {
        KnockDown {
            id: Some(id),
            second_id: None,
            mighty_blow: false,
            second_mighty_blow: false,
        }
    }
    pub fn new_empty() -> AnyProc {
        AnyProc::KnockDown(KnockDown {
            id: None,
            second_id: None,
            mighty_blow: false,
            second_mighty_blow: false,
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

    match game_state.ball {
        BallState::Carried(carrier_id) if carrier_id == id => {
            game_state.set_ball(BallState::InAir(player_position));
            (true, true)
        }
        // A player who falls onto a loose ball dislodges it — the ball may
        // never come to rest under a player, same invariant as
        // `Push::handle_aftermath`. Unlike the carried case, the ball is
        // already `OnGround` at this position, so no state change is needed
        // before `Bounce` picks it up.
        BallState::OnGround(ball_pos) if ball_pos == player_position => (true, true),
        _ => (true, false),
    }
}
impl Procedure for KnockDown {
    fn step(&mut self, game_state: &mut GameState, _input: ProcInput) -> ProcState {
        let mut procs: Vec<AnyProc> = Vec::with_capacity(3);
        let mut armor_procs: Vec<AnyProc> = Vec::with_capacity(2);
        let mut should_bounce_ball = false;
        for id in [self.id, self.second_id].iter().flatten() {
            let (knocked_down, p_should_bounce_ball) = knock_down_player(game_state, *id);
            should_bounce_ball |= p_should_bounce_ball;
            if knocked_down {
                let mighty_blow = if Some(*id) == self.id {
                    self.mighty_blow
                } else {
                    self.second_mighty_blow
                };
                armor_procs.push(casualty_procs::Armor::new_block(*id, mighty_blow))
            }
        }
        if should_bounce_ball {
            procs.push(ball_procs::Bounce::new());
        }
        for armor_proc in armor_procs {
            procs.push(armor_proc);
        }
        ProcState::from(procs)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BlockAction {
    /// The player jumped up to make this block (Jump Up), so they must block: no
    /// `EndPlayerTurn`.
    #[serde(default)]
    must_block: bool,
}

impl BlockAction {
    pub fn new() -> AnyProc {
        AnyProc::BlockAction(BlockAction { must_block: false })
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
        if !self.must_block {
            game_state.available_actions.insert_simple(SimpleAT::EndPlayerTurn);
        }
    }
}

/// Jump Up: a prone player declaring a Block first makes an Agility test at +1. On a pass they
/// stand up (for free) and must block; on a fail they stay prone and their activation ends —
/// not a turnover.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct JumpUp {
    id: PlayerID,
    target: D6Target,
}
impl JumpUp {
    pub fn new(game_state: &GameState, id: PlayerID) -> AnyProc {
        let target = *game_state.get_player_unsafe(id).ag_target().add_modifer(1);
        AnyProc::JumpUp(SimpleProcContainer::new(JumpUp { id, target }))
    }
}
impl SimpleProc for JumpUp {
    fn d6_target(&self) -> D6Target {
        self.target
    }
    fn reroll_skill(&self) -> Option<Skill> {
        None
    }
    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.get_mut_player_unsafe(self.id).status = PlayerStatus::Up;
        vec![AnyProc::BlockAction(BlockAction { must_block: true })]
    }
    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.get_mut_player_unsafe(self.id).used = true;
        Vec::new()
    }
    fn player_id(&self) -> PlayerID {
        self.id
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

/// Does the defender's Dodge turn a Stumble (`PowPush`) into a plain push? Not against Tackle,
/// assuming the blocker uses it (the MCTS block model and the scripted picks; the engine asks).
pub fn dodge_saves_from_stumble(attacker: &FieldedPlayer, defender: &FieldedPlayer) -> bool {
    defender.has_skill(Skill::Dodge) && !attacker.has_skill(Skill::Tackle)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Block {
    dices: NumBlockDices,
    defender: PlayerID,
    state: BlockProcState,
    roll: [Option<BlockDice>; 3],
    is_uphill: bool,
    /// Frenzy's second block, which does not chain into a third.
    #[serde(default)]
    frenzy_second: bool,
    /// The blocker's answer on Tackle, once asked (a Stumble against a Dodge defender).
    #[serde(default)]
    tackle: Option<bool>,
    /// The blocker has been asked about Juggernaut (a Both Down in a Blitz).
    #[serde(default)]
    juggernaut_asked: bool,
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
            frenzy_second: false,
            tackle: None,
            juggernaut_asked: false,
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
impl Block {
    /// A Both Down, unless someone may Wrestle.
    fn both_down_or_wrestle(&self, game_state: &mut GameState) -> ProcState {
        if [game_state.info.active_player.unwrap(), self.defender]
            .iter()
            .any(|&id| game_state.get_player_unsafe(id).has_skill(Skill::Wrestle))
        {
            ProcState::DoneNew(Wrestle::new(self.defender))
        } else {
            both_down(game_state, self.defender)
        }
    }

    /// Apply the chosen die (any but a Both Down, which `step` routes itself).
    fn resolve(&mut self, game_state: &mut GameState, dice_action_type: SimpleAT) -> ProcState {
        let attacker_id = game_state.info.active_player.unwrap();
        let mut push = false;
        let mut knockdown_proc: KnockDown = KnockDown {
            id: None,
            second_id: None,
            mighty_blow: game_state.get_player_unsafe(attacker_id).has_skill(Skill::MightyBlow),
            second_mighty_blow: game_state.get_player_unsafe(self.defender).has_skill(Skill::MightyBlow),
        };

        match dice_action_type {
            SimpleAT::SelectPow => {
                knockdown_proc.id = Some(self.defender);
                push = true;
            }
            SimpleAT::SelectPush => {
                push = true;
            }
            SimpleAT::SelectPowPush => {
                // Dodge turns a Stumble into a push, unless the blocker used Tackle.
                let dodges = game_state.get_player_unsafe(self.defender).has_skill(Skill::Dodge);
                if !dodges || self.tackle == Some(true) {
                    knockdown_proc.id = Some(self.defender);
                }
                push = true;
            }

            SimpleAT::SelectSkull => knockdown_proc.second_id = Some(attacker_id),
            _ => panic!("very wrong!"),
        }

        let mut procs: Vec<AnyProc> = Vec::with_capacity(4);

        // Frenzy: once the push, follow-up and any knockdown have resolved, block again.
        if push && !self.frenzy_second && game_state.get_player_unsafe(attacker_id).has_skill(Skill::Frenzy) {
            procs.push(FrenzyBlock::new(
                self.defender,
                game_state.get_player_unsafe(self.defender).position,
            ));
        }

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
            let mut push = Push::new_pure(
                game_state.get_player_unsafe(attacker_id).position,
                game_state.get_player_unsafe(self.defender).position,
            );
            // Strip Ball: a carrier pushed back drops the ball (one knocked down drops it anyway,
            // from the same square).
            if game_state.get_player_unsafe(attacker_id).has_skill(Skill::StripBall)
                && !game_state.get_player_unsafe(self.defender).has_skill(Skill::SureHands)
            {
                push.strip_ball = Some(self.defender);
            }
            push.juggernaut = juggernaut_blitz(game_state);
            if game_state.get_player_unsafe(self.defender).has_skill(Skill::Fend) && !push.juggernaut {
                push.fend = Some(self.defender);
            }
            push.grab = game_state.get_player_unsafe(attacker_id).has_skill(Skill::Grab);
            procs.push(AnyProc::Push(push));
        }
        ProcState::from(procs)
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
            // Juggernaut: in a Blitz the blocker's coach may play a Both Down as a Push.
            ProcInput::Action(Action::Simple(SimpleAT::SelectBothDown))
                if juggernaut_blitz(game_state) && !self.juggernaut_asked =>
            {
                self.juggernaut_asked = true;
                let mut aa = AvailableActions::new(game_state.get_active_player().unwrap().stats.team);
                aa.insert_simple(SimpleAT::UseSkill);
                aa.insert_simple(SimpleAT::DontUseSkill);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseSkill)) if self.juggernaut_asked => {
                self.resolve(game_state, SimpleAT::SelectPush)
            }
            ProcInput::Action(Action::Simple(SimpleAT::DontUseSkill)) if self.juggernaut_asked => {
                self.both_down_or_wrestle(game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::SelectBothDown)) => self.both_down_or_wrestle(game_state),
            // Tackle: on a Stumble against a Dodge defender the blocker's coach is asked whether to
            // use it (knocking the defender down) or let Dodge turn it into a push.
            ProcInput::Action(Action::Simple(SimpleAT::SelectPowPush))
                if self.tackle.is_none()
                    && game_state.get_player_unsafe(self.defender).has_skill(Skill::Dodge)
                    && game_state.get_active_player().unwrap().has_skill(Skill::Tackle) =>
            {
                let mut aa = AvailableActions::new(game_state.get_active_player().unwrap().stats.team);
                aa.insert_simple(SimpleAT::UseSkill);
                aa.insert_simple(SimpleAT::DontUseSkill);
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Simple(answer @ (SimpleAT::UseSkill | SimpleAT::DontUseSkill))) => {
                self.tackle = Some(answer == SimpleAT::UseSkill);
                self.resolve(game_state, SimpleAT::SelectPowPush)
            }
            ProcInput::Action(Action::Simple(dice_action_type)) => self.resolve(game_state, dice_action_type),
            _ => unreachable!(),
        }
    }
}

/// The active player blocks with Juggernaut in a Blitz: their target can't use Fend, Stand Firm
/// or Wrestle, and a Both Down may be played as a Push.
pub fn juggernaut_blitz(game_state: &GameState) -> bool {
    game_state.info.blitz_this_activation && game_state.get_active_player().unwrap().has_skill(Skill::Juggernaut)
}

/// A Both Down, played as normal: whoever lacks Block is knocked down, and the blocker going
/// down is a turnover.
fn both_down(game_state: &mut GameState, defender: PlayerID) -> ProcState {
    let attacker = game_state.get_active_player().unwrap();
    let attacker_id = attacker.id;
    let mut knockdown_proc = KnockDown {
        id: None,
        second_id: None,
        mighty_blow: attacker.has_skill(Skill::MightyBlow),
        second_mighty_blow: game_state.get_player_unsafe(defender).has_skill(Skill::MightyBlow),
    };
    if !attacker.has_skill(Skill::Block) {
        knockdown_proc.second_id = Some(attacker_id);
        game_state.info.turnover = true;
    }
    if !game_state.get_player_unsafe(defender).has_skill(Skill::Block) {
        knockdown_proc.id = Some(defender);
    }
    if knockdown_proc.id.is_some() || knockdown_proc.second_id.is_some() {
        ProcState::DoneNew(AnyProc::KnockDown(knockdown_proc))
    } else {
        ProcState::Done
    }
}

/// Wrestle on a Both Down: the blocker's coach is asked first if the blocker has it, then the
/// defender's. If either uses it, both players are placed prone — no armour rolls, the blocker's
/// activation ends, never a turnover; a carried ball bounces. If neither does, the Both Down
/// plays as normal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Wrestle {
    defender: PlayerID,
    stage: WrestleStage,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum WrestleStage {
    Start,
    AttackerAsked,
    DefenderAsked,
}
impl Wrestle {
    pub fn new(defender: PlayerID) -> AnyProc {
        AnyProc::Wrestle(Wrestle {
            defender,
            stage: WrestleStage::Start,
        })
    }
    fn ask(team: TeamType) -> ProcState {
        let mut aa = AvailableActions::new(team);
        aa.insert_simple(SimpleAT::UseSkill);
        aa.insert_simple(SimpleAT::DontUseSkill);
        ProcState::NeedAction(aa)
    }
    fn place_both_prone(&self, game_state: &mut GameState) -> ProcState {
        let attacker_id = game_state.info.active_player.unwrap();
        let mut bounce = false;
        for id in [attacker_id, self.defender] {
            let player = game_state.get_mut_player_unsafe(id);
            player.status = PlayerStatus::Down;
            player.used = true;
            let position = player.position;
            if matches!(game_state.ball, BallState::Carried(carrier) if carrier == id) {
                game_state.set_ball(BallState::InAir(position));
                bounce = true;
            }
        }
        if bounce {
            ProcState::DoneNew(ball_procs::Bounce::new())
        } else {
            ProcState::Done
        }
    }
}
impl Procedure for Wrestle {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Action(Action::Simple(SimpleAT::UseSkill)) => return self.place_both_prone(game_state),
            ProcInput::Action(Action::Simple(SimpleAT::DontUseSkill)) | ProcInput::Nothing => (),
            _ => panic!("Unexpected input {:?}", input),
        }
        if self.stage == WrestleStage::Start {
            self.stage = WrestleStage::AttackerAsked;
            let attacker = game_state.get_active_player().unwrap();
            if attacker.has_skill(Skill::Wrestle) {
                return Wrestle::ask(attacker.stats.team);
            }
        }
        if self.stage == WrestleStage::AttackerAsked {
            self.stage = WrestleStage::DefenderAsked;
            let defender = game_state.get_player_unsafe(self.defender);
            if defender.has_skill(Skill::Wrestle) && !juggernaut_blitz(game_state) {
                return Wrestle::ask(defender.stats.team);
            }
        }
        both_down(game_state, self.defender)
    }
}

/// Frenzy's second block against the same target, once the first block has fully resolved.
/// Only if the target was pushed and is still standing next to a still-standing blocker. In a Blitz it costs a square of movement, or a Rush when none is left; a failed
/// Rush knocks the blocker over and there is no second block.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FrenzyBlock {
    defender: PlayerID,
    /// Where the defender stood when blocked: no second block unless they were pushed off it.
    from: Position,
    rushed: bool,
}
impl FrenzyBlock {
    pub fn new(defender: PlayerID, from: Position) -> AnyProc {
        AnyProc::FrenzyBlock(FrenzyBlock {
            defender,
            from,
            rushed: false,
        })
    }
}
impl Procedure for FrenzyBlock {
    fn step(&mut self, game_state: &mut GameState, _input: ProcInput) -> ProcState {
        let attacker = game_state.get_active_player().unwrap();
        let Ok(defender) = game_state.get_player(self.defender) else {
            return ProcState::Done; // pushed off the pitch
        };
        // One who stood firm was not pushed at all; one who used Fend is no longer adjacent.
        if attacker.status != PlayerStatus::Up
            || defender.status != PlayerStatus::Up
            || defender.position == self.from
            || attacker.position.distance_to(&defender.position) > 1
        {
            return ProcState::Done;
        }
        let attacker_id = attacker.id;
        if game_state.info.blitz_this_activation && !self.rushed {
            if attacker.total_movement_left() == 0 {
                return ProcState::Done;
            }
            let needs_rush = attacker.moves_left() == 0;
            game_state.get_mut_player_unsafe(attacker_id).add_move(1);
            if needs_rush {
                self.rushed = true;
                return ProcState::NotDoneNew(movement_procs::GfiProc::new_rush(game_state, attacker_id));
            }
        }
        let dices = game_state.get_blockdices(attacker_id, self.defender);
        let AnyProc::Block(mut block) = Block::new(dices, self.defender) else {
            unreachable!()
        };
        block.frenzy_second = true;
        ProcState::DoneNew(AnyProc::Block(block))
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

    /// A prone away player next to a standing home player, with no team re-rolls in play. Home
    /// ends its turn so Away's actions are computed with the player already down.
    fn prone_blocker(jump_up: bool) -> (crate::core::gamestate::GameState, Position, Position) {
        let home_pos = Position::new((5, 3));
        let away_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        state.get_mut_team(TeamType::Away).rerolls = 0;
        let id = state.get_player_id_at(away_pos).unwrap();
        state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
        if jump_up {
            state.get_mut_player_unsafe(id).stats.give_skill(Skill::JumpUp);
        }
        state.step_simple(SimpleAT::EndTurn);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        (state, away_pos, home_pos)
    }

    #[test]
    fn a_prone_player_without_jump_up_cannot_block() {
        let (state, blocker, _) = prone_blocker(false);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::StartBlock, blocker)));
    }

    /// Jump Up: a prone player may declare a Block, standing up on an Agility test at +1 (AG 3
    /// needs a 3+ instead of a 4+). Once up they must block — the action cannot be abandoned.
    #[test]
    fn jump_up_blocks_from_prone_on_an_agility_test() {
        let (mut state, blocker, target) = prone_blocker(true);
        let id = state.get_player_id_at(blocker).unwrap();
        assert_eq!(state.get_player_unsafe(id).stats.ag, 3);

        state.fix_d6(3);
        state.step_positional(PosAT::StartBlock, blocker);
        assert!(state.is_legal_action(&Action::Positional(PosAT::Block, target)));
        assert!(
            !state.is_legal_action(&Action::Simple(SimpleAT::EndPlayerTurn)),
            "a player who jumped up must block"
        );
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);

        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, target);
        state.step_simple(SimpleAT::SelectPush);
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
    }

    /// A failed Jump Up leaves the player prone and ends their activation — not a turnover.
    #[test]
    fn a_failed_jump_up_ends_the_activation_without_a_turnover() {
        let (mut state, blocker, _) = prone_blocker(true);
        let id = state.get_player_id_at(blocker).unwrap();

        state.fix_d6(2);
        state.step_positional(PosAT::StartBlock, blocker);

        let player = state.get_player_unsafe(id);
        assert_eq!(player.status, PlayerStatus::Down);
        assert!(player.used);
        assert!(!state.info.turnover);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));
    }

    /// Sidestep: a pushed player's own coach picks the square, from every free square adjacent
    /// to them — not just the three away from the blocker.
    #[test]
    fn sidestep_lets_the_pushed_player_choose_any_free_adjacent_square() {
        let home_pos = Position::new((5, 3));
        let away_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        let defender = state.get_player_id_at(away_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SideStep);

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPush);

        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);
        let sideways = Position::new((6, 2));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, sideways)));
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Push, home_pos)));
        state.step_positional(PosAT::Push, sideways);
        assert_eq!(state.get_player_unsafe(defender).position, sideways);
    }

    /// Sidestep keeps a player on the pitch while any square next to them is free: a push that
    /// would go into the crowd becomes the player's own choice of square.
    #[test]
    fn sidestep_avoids_the_crowd() {
        let home_pos = Position::new((6, 2));
        let away_pos = Position::new((6, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        assert!(
            super::Push::is_crowd_push(home_pos, away_pos, &state),
            "the plain push goes into the crowd"
        );
        let defender = state.get_player_id_at(away_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SideStep);
        assert!(!super::Push::is_crowd_push(home_pos, away_pos, &state));

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPush);

        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);
        state.step_positional(PosAT::Push, Position::new((5, 1)));
        assert_eq!(state.get_player_unsafe(defender).position, Position::new((5, 1)));
    }

    /// Sidestep is the pushed player's to use: declined, the blocker picks the square as usual.
    #[test]
    fn declining_sidestep_is_a_normal_push() {
        let home_pos = Position::new((5, 3));
        let away_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        let defender = state.get_player_id_at(away_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SideStep);

        state.step_positional(PosAT::StartBlock, home_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, away_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_simple(SimpleAT::DontUseSkill);

        assert_eq!(state.get_available_actions().team, Some(TeamType::Home));
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Push, Position::new((6, 2)))));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, away_pos + (1, 0))));
    }

    /// Stand Firm in a chain push: if the player pushed into uses it, nobody is pushed.
    #[test]
    fn stand_firm_stops_a_chain_push() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_away_players(&[(7, 2), (7, 3), (7, 4)])
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        let firm = state.get_player_id_at(Position::new((7, 3))).unwrap();
        state.get_mut_player_unsafe(firm).stats.give_skill(Skill::StandFirm);

        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, Position::new((7, 3)));
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);

        assert_eq!(state.get_player_unsafe(attacker).position, attacker_pos);
        assert_eq!(state.get_player_unsafe(defender).position, defender_pos);
        assert_eq!(state.get_player_unsafe(firm).position, Position::new((7, 3)));
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "no follow-up"
        );
    }

    /// Guard: a marked player still assists a block. The home assister next to the defender is
    /// also marked by a second away player, so without Guard it does not count.
    #[test]
    fn guard_gives_an_offensive_assist_while_marked() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let assister_pos = Position::new((7, 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_home_player(assister_pos)
            .add_away_player(defender_pos)
            .add_away_player(Position::new((8, 1)))
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        assert_eq!(state.get_blockdices(attacker, defender), NumBlockDices::One);

        let assister = state.get_player_id_at(assister_pos).unwrap();
        state.get_mut_player_unsafe(assister).stats.give_skill(Skill::Guard);
        assert_eq!(state.get_blockdices(attacker, defender), NumBlockDices::Two);
    }

    #[test]
    fn guard_gives_a_defensive_assist_while_marked() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let assister_pos = Position::new((4, 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_home_player(Position::new((3, 1)))
            .add_away_player(defender_pos)
            .add_away_player(assister_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        assert_eq!(state.get_blockdices(attacker, defender), NumBlockDices::One);

        let assister = state.get_player_id_at(assister_pos).unwrap();
        state.get_mut_player_unsafe(assister).stats.give_skill(Skill::Guard);
        assert_eq!(state.get_blockdices(attacker, defender), NumBlockDices::TwoUphill);
    }

    /// Pow on an AV 8 defender (armour breaks on 9+), with the given armour and injury dice.
    /// Returns where the defender ended up: on the pitch with a status, or in the dugout.
    fn pow_on_av8(mighty_blow: bool, armour: (u8, u8), injury: Option<(u8, u8)>) -> Option<PlayerStatus> {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.av = 8;
        if mighty_blow {
            state
                .get_mut_player_unsafe(attacker)
                .stats
                .give_skill(Skill::MightyBlow);
        }
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        state.fix_d6(armour.0);
        state.fix_d6(armour.1);
        if let Some((a, b)) = injury {
            state.fix_d6(a);
            state.fix_d6(b);
        }
        state.step_positional(PosAT::FollowUp, defender_pos);
        state.get_player(defender).ok().map(|p| p.status)
    }

    /// Mighty Blow: +1 to the armour roll or the injury roll. An 8 holds against AV 8; with the
    /// +1 it breaks — and the +1 is then spent, so a 7 on injury stays stunned.
    #[test]
    fn mighty_blow_breaks_armour_that_would_hold() {
        assert_eq!(pow_on_av8(false, (5, 3), None), Some(PlayerStatus::Down));
        assert_eq!(pow_on_av8(true, (5, 3), Some((4, 3))), Some(PlayerStatus::Stunned));
    }

    /// Armour that breaks on its own leaves the +1 for the injury roll: a 7 becomes an 8, KO.
    #[test]
    fn mighty_blow_adds_to_injury_when_armour_breaks_anyway() {
        assert_eq!(pow_on_av8(false, (5, 5), Some((4, 3))), Some(PlayerStatus::Stunned));
        assert_eq!(pow_on_av8(true, (5, 5), Some((4, 3))), None, "knocked out");
    }

    /// Mighty Blow is for the player the block knocks down, never the blocker's own fall: on a
    /// Both Down against a Block defender, an 8 still holds the attacker's AV 8.
    #[test]
    fn mighty_blow_does_not_help_against_the_attacker() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.av = 8;
        state
            .get_mut_player_unsafe(attacker)
            .stats
            .give_skill(Skill::MightyBlow);
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Block);

        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, defender_pos);
        state.fix_d6(5);
        state.fix_d6(3);
        state.step_simple(SimpleAT::SelectBothDown);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
    }

    /// The defender's Mighty Blow counts against a blocker the block knocks down: an 8 breaks
    /// the attacker's AV 8, on a Skull and on a Both Down alike.
    #[test]
    fn a_defenders_mighty_blow_hits_a_fallen_blocker() {
        for die in [BlockDice::Skull, BlockDice::BothDown] {
            let attacker_pos = Position::new((5, 3));
            let defender_pos = Position::new((6, 3));
            let mut state = GameStateBuilder::new()
                .add_home_player(attacker_pos)
                .add_away_player(defender_pos)
                .build();
            let attacker = state.get_player_id_at(attacker_pos).unwrap();
            let defender = state.get_player_id_at(defender_pos).unwrap();
            state.get_mut_player_unsafe(attacker).stats.av = 8;
            state
                .get_mut_player_unsafe(defender)
                .stats
                .give_skill(Skill::MightyBlow);
            state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Block);

            state.step_positional(PosAT::StartBlock, attacker_pos);
            state.fix_blockdice(die);
            state.step_positional(PosAT::Block, defender_pos);
            state.fix_d6(5); //armour: 8 breaks only with Mighty Blow
            state.fix_d6(3);
            state.fix_d6(1); //injury: stunned
            state.fix_d6(2);
            state.step_simple(SimpleAT::from(die));
            assert_eq!(
                state.get_player_unsafe(attacker).status,
                PlayerStatus::Stunned,
                "{die:?}"
            );
        }
    }

    /// A home Frenzy blocker next to an away defender, with plenty of room behind the defender.
    fn frenzy_block() -> (crate::core::gamestate::GameState, PlayerID, PlayerID, Position) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Frenzy);
        (state, attacker, defender, defender_pos)
    }

    /// Frenzy: the blocker must follow up a push.
    #[test]
    fn frenzy_must_follow_up() {
        let (mut state, attacker, _, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert!(state.is_legal_action(&Action::Positional(PosAT::FollowUp, defender_pos)));
        assert!(!state.is_legal_action(&Action::Positional(PosAT::FollowUp, attacker_pos)));
    }

    /// Frenzy: a target still standing after the push is blocked again, once.
    #[test]
    fn frenzy_blocks_a_pushed_target_a_second_time() {
        let (mut state, attacker, defender, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        let pushed_to = defender_pos + (1, 0);
        state.step_positional(PosAT::Push, pushed_to);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::FollowUp, defender_pos);

        assert_eq!(state.proc_stack_top(), Some("Block"), "the second block's dice are up");
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, pushed_to + (1, 0));
        state.step_positional(PosAT::FollowUp, pushed_to);
        assert_eq!(state.get_player_unsafe(defender).position, pushed_to + (1, 0));
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "two blocks, not three"
        );
    }

    /// No second block once the target is down.
    #[test]
    fn frenzy_stops_when_the_target_goes_down() {
        let (mut state, attacker, _, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPow);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.step_positional(PosAT::FollowUp, defender_pos);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));
    }

    /// In a Blitz the second block costs a square of movement like the first ...
    #[test]
    fn frenzy_second_block_in_a_blitz_costs_a_square() {
        let (mut state, attacker, _, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.get_mut_player_unsafe(attacker).stats.ma = 2;
        state.step_positional(PosAT::StartBlitz, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        let pushed_to = defender_pos + (1, 0);
        state.step_positional(PosAT::Push, pushed_to);
        assert_eq!(state.get_player_unsafe(attacker).moves_left(), 1);
        state.fix_blockdice(BlockDice::Pow);
        state.step_positional(PosAT::FollowUp, defender_pos);
        assert_eq!(state.get_player_unsafe(attacker).moves_left(), 0);
        assert_eq!(state.proc_stack_top(), Some("Block"));
    }

    /// ... and with none left, a Rush. Failing it knocks the blocker over: turnover, no block.
    #[test]
    fn frenzy_second_block_in_a_blitz_may_need_a_rush() {
        let (mut state, attacker, _, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.get_mut_player_unsafe(attacker).stats.ma = 1;
        state.get_mut_team(TeamType::Home).rerolls = 0;
        state.step_positional(PosAT::StartBlitz, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert_eq!(state.get_player_unsafe(attacker).moves_left(), 0);
        state.fix_d6(1); //rush
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.step_positional(PosAT::FollowUp, defender_pos);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away), "a turnover");
    }

    /// A Blitz with no movement and no Rushes left has no second block.
    #[test]
    fn frenzy_has_no_second_block_without_movement_left() {
        let (mut state, attacker, _, defender_pos) = frenzy_block();
        let attacker_pos = state.get_player_unsafe(attacker).position;
        state.get_mut_player_unsafe(attacker).stats.ma = 0;
        state.get_mut_player_unsafe(attacker).moves = 1; // one Rush already spent
        state.step_positional(PosAT::StartBlitz, attacker_pos);
        state.fix_d6(6); //rush into the block
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert_eq!(state.get_player_unsafe(attacker).total_movement_left(), 0);
        state.step_positional(PosAT::FollowUp, defender_pos);
        assert_ne!(state.proc_stack_top(), Some("Block"));
    }

    /// A target pushed into the crowd has left the pitch: no second block.
    #[test]
    fn frenzy_has_no_second_block_after_a_crowd_push() {
        let attacker_pos = Position::new((6, 2));
        let defender_pos = Position::new((6, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Frenzy);
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.fix_d6(1); //crowd injury
        state.fix_d6(2); //crowd injury
        state.step_positional(PosAT::FollowUp, defender_pos);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));
    }

    /// Tackle: a defender with Dodge still goes down on a Stumble from a Tackle blocker — if the
    /// blocker's coach uses it; they are asked once the Stumble is picked.
    #[test]
    fn tackle_may_beat_dodge_on_a_stumble() {
        // `tackle`: None = the blocker has no Tackle, Some(used) = it has, and is (not) used.
        let stumble_with = |tackle: Option<bool>| {
            let attacker_pos = Position::new((5, 3));
            let defender_pos = Position::new((6, 3));
            let mut state = GameStateBuilder::new()
                .add_home_player(attacker_pos)
                .add_away_player(defender_pos)
                .build();
            let attacker = state.get_player_id_at(attacker_pos).unwrap();
            let defender = state.get_player_id_at(defender_pos).unwrap();
            state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Dodge);
            if tackle.is_some() {
                state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Tackle);
            }
            state.step_positional(PosAT::StartBlock, attacker_pos);
            state.fix_blockdice(BlockDice::PowPush);
            state.step_positional(PosAT::Block, defender_pos);
            state.step_simple(SimpleAT::SelectPowPush);
            if let Some(used) = tackle {
                assert_eq!(state.get_available_actions().team, Some(TeamType::Home));
                state.step_simple(if used {
                    SimpleAT::UseSkill
                } else {
                    SimpleAT::DontUseSkill
                });
            }
            state.step_positional(PosAT::Push, defender_pos + (1, 0));
            if tackle == Some(true) {
                state.fix_d6(1); //armor
                state.fix_d6(1); //armor
            }
            state.step_positional(PosAT::FollowUp, defender_pos);
            state.get_player_unsafe(defender).status
        };
        assert_eq!(stumble_with(None), PlayerStatus::Up);
        assert_eq!(stumble_with(Some(false)), PlayerStatus::Up);
        assert_eq!(stumble_with(Some(true)), PlayerStatus::Down);
    }

    /// A Both Down between a home blocker and an away defender, either of them with Wrestle.
    fn both_down_with(
        attacker_wrestle: bool,
        defender_wrestle: bool,
    ) -> (crate::core::gamestate::GameState, PlayerID, PlayerID) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        if attacker_wrestle {
            state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Wrestle);
        }
        if defender_wrestle {
            state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Wrestle);
        }
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectBothDown);
        (state, attacker, defender)
    }

    /// Wrestle: instead of the Both Down, both players are placed prone — no armour rolls, and
    /// the blocker going prone is not a turnover.
    #[test]
    fn wrestle_places_both_players_prone_without_a_turnover() {
        let (mut state, attacker, defender) = both_down_with(true, false);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Home));
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::DontUseSkill)));
        state.step_simple(SimpleAT::UseSkill);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
        assert_eq!(state.get_player_unsafe(defender).status, PlayerStatus::Down);
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "still Home's turn"
        );
    }

    /// Declining Wrestle plays the Both Down as normal: the blocker (no Block) falls, turnover.
    #[test]
    fn declining_wrestle_plays_the_both_down() {
        let (mut state, attacker, defender) = both_down_with(true, false);
        state.fix_d6(1); //attacker armour
        state.fix_d6(1);
        state.fix_d6(1); //defender armour
        state.fix_d6(1);
        state.step_simple(SimpleAT::DontUseSkill);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
        assert_eq!(state.get_player_unsafe(defender).status, PlayerStatus::Down);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away), "a turnover");
    }

    /// In a Blitz, going prone ends the blocker's activation: no more movement.
    #[test]
    fn wrestle_ends_a_blitz() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Wrestle);
        state.step_positional(PosAT::StartBlitz, attacker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectBothDown);
        state.step_simple(SimpleAT::UseSkill);
        assert!(state.get_player_unsafe(attacker).used);
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "back at the turn"
        );
    }

    /// The defender may use it too: their coach is asked.
    #[test]
    fn the_defender_may_wrestle() {
        let (mut state, attacker, defender) = both_down_with(false, true);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
        assert_eq!(state.get_player_unsafe(defender).status, PlayerStatus::Down);
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "still Home's turn"
        );
    }

    /// A ball carrier placed prone drops the ball, and still no turnover.
    #[test]
    fn a_wrestled_ball_carrier_drops_the_ball_without_a_turnover() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_ball_pos(attacker_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Wrestle);
        assert_eq!(state.ball, BallState::Carried(attacker));
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectBothDown);
        state.fix_d8(1); //bounce
        state.step_simple(SimpleAT::UseSkill);
        assert!(matches!(state.ball, BallState::OnGround(_)));
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "still Home's turn"
        );
    }

    /// A home blocker against an away Stand Firm defender, after the given die is selected.
    fn block_stand_firm(
        die: BlockDice,
        frenzy: bool,
    ) -> (crate::core::gamestate::GameState, PlayerID, PlayerID, Position) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::StandFirm);
        if frenzy {
            state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Frenzy);
        }
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(die);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::from(die));
        (state, attacker, defender, defender_pos)
    }

    /// Stand Firm: the defender's coach may refuse the push; nobody moves, no follow-up.
    #[test]
    fn stand_firm_refuses_the_push() {
        let (mut state, attacker, defender, defender_pos) = block_stand_firm(BlockDice::Push, false);
        let attacker_pos = state.get_player_unsafe(attacker).position;
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);
        assert_eq!(state.get_player_unsafe(defender).position, defender_pos);
        assert_eq!(state.get_player_unsafe(attacker).position, attacker_pos);
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "no follow-up"
        );
    }

    /// Declined, the push is the blocker's as usual.
    #[test]
    fn declining_stand_firm_is_a_normal_push() {
        let (mut state, _, _, defender_pos) = block_stand_firm(BlockDice::Push, false);
        state.step_simple(SimpleAT::DontUseSkill);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Home));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (1, 0))));
    }

    /// A knockdown still happens, in the defender's own square.
    #[test]
    fn stand_firm_goes_down_in_place() {
        let (mut state, _, defender, defender_pos) = block_stand_firm(BlockDice::Pow, false);
        state.fix_d6(1); //armour
        state.fix_d6(1);
        state.step_simple(SimpleAT::UseSkill);
        let player = state.get_player_unsafe(defender);
        assert_eq!((player.position, player.status), (defender_pos, PlayerStatus::Down));
    }

    /// Frenzy blocks again only after a push: a defender who stood firm was not pushed.
    #[test]
    fn frenzy_does_not_block_again_when_the_target_stands_firm() {
        let (mut state, _, _, _) = block_stand_firm(BlockDice::Push, true);
        state.step_simple(SimpleAT::UseSkill);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)));
    }

    /// A home blocker pushes an away Fend defender one square straight back.
    fn push_a_fend_defender(frenzy: bool) -> (crate::core::gamestate::GameState, Position, Position) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Fend);
        if frenzy {
            state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Frenzy);
        }
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        (state, attacker_pos, defender_pos)
    }

    /// Fend: the pushed defender's coach may stop the blocker following up.
    #[test]
    fn fend_stops_the_follow_up() {
        let (mut state, attacker_pos, _) = push_a_fend_defender(false);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away));
        state.step_simple(SimpleAT::UseSkill);
        assert!(state.get_player_at(attacker_pos).is_some(), "the blocker stays put");
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "no follow-up"
        );
    }

    /// Declined, the blocker chooses as usual.
    #[test]
    fn declining_fend_allows_the_follow_up() {
        let (mut state, attacker_pos, defender_pos) = push_a_fend_defender(false);
        state.step_simple(SimpleAT::DontUseSkill);
        assert!(state.is_legal_action(&Action::Positional(PosAT::FollowUp, defender_pos)));
        assert!(state.is_legal_action(&Action::Positional(PosAT::FollowUp, attacker_pos)));
    }

    /// Fend beats Frenzy: no forced follow-up, so no second block.
    #[test]
    fn fend_stops_a_frenzy_follow_up_and_second_block() {
        let (mut state, attacker_pos, _) = push_a_fend_defender(true);
        state.step_simple(SimpleAT::UseSkill);
        assert!(state.get_player_at(attacker_pos).is_some(), "the blocker stays put");
        assert!(
            state.is_legal_action(&Action::Simple(SimpleAT::EndTurn)),
            "no second block"
        );
    }

    /// A home Grab blocker pushes an away defender; `blockers` are away players behind the
    /// defender. Returns the state at the push-square choice (or the defender's Sidestep
    /// question), and the defender's square.
    fn grab_push(blitz: bool, sidestep: bool, blockers: &[(i8, i8)]) -> (crate::core::gamestate::GameState, Position) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut builder = GameStateBuilder::new();
        builder.add_home_player(attacker_pos).add_away_player(defender_pos);
        for &square in blockers {
            builder.add_away_player(Position::new(square));
        }
        let mut state = builder.build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Grab);
        if sidestep {
            state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SideStep);
        }
        let start = if blitz { PosAT::StartBlitz } else { PosAT::StartBlock };
        state.step_positional(start, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        (state, defender_pos)
    }

    /// Grab: the blocker may push the target into any free square next to them.
    #[test]
    fn grab_pushes_into_any_free_square_next_to_the_target() {
        let (mut state, defender_pos) = grab_push(false, false, &[]);
        let beside = defender_pos + (0, -1);
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (1, 0))));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, beside)));
        state.step_positional(PosAT::Push, beside);
        assert!(state.get_player_at(beside).is_some());
    }

    /// Grab may push into a free square instead of a chain push.
    #[test]
    fn grab_may_avoid_a_chain_push() {
        let (state, defender_pos) = grab_push(false, false, &[(7, 2), (7, 3), (7, 4)]);
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (1, 0))));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (0, 1))));
    }

    /// In a Blitz, Grab picks no squares ...
    #[test]
    fn grab_picks_no_squares_in_a_blitz() {
        let (state, defender_pos) = grab_push(true, false, &[]);
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (1, 0))));
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (0, -1))));
    }

    /// ... but in a Blitz too the target can't Sidestep.
    #[test]
    fn grab_stops_sidestep() {
        let (state, defender_pos) = grab_push(true, true, &[]);
        assert_eq!(
            state.get_available_actions().team,
            Some(TeamType::Home),
            "no Sidestep question"
        );
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (0, -1))));
    }

    /// ... so a Sidestep target on the sideline goes into the crowd.
    #[test]
    fn grab_sends_a_sidestep_target_on_the_sideline_into_the_crowd() {
        let home_pos = Position::new((6, 2));
        let away_pos = Position::new((6, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(home_pos)
            .add_away_player(away_pos)
            .build();
        let attacker = state.get_player_id_at(home_pos).unwrap();
        let defender = state.get_player_id_at(away_pos).unwrap();
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SideStep);
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::Grab);
        assert!(super::Push::is_crowd_push(home_pos, away_pos, &state));
    }

    /// Grab picks the square of its target only, not of a player a chain push reaches.
    #[test]
    fn grab_does_not_pick_the_square_of_a_chain_pushed_player() {
        let (mut state, defender_pos) = grab_push(false, false, &[(7, 2), (7, 3), (7, 4)]);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, Position::new((8, 3)))));
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Push, Position::new((6, 2)))));
    }

    /// A home Juggernaut blocker against an away defender with `defender_skills`, by Blitz or
    /// Block; `others` are away players. Returns the state after `die` is selected.
    fn juggernaut_block(
        blitz: bool,
        die: BlockDice,
        defender_skills: &[Skill],
        others: &[(i8, i8)],
    ) -> (crate::core::gamestate::GameState, PlayerID, PlayerID, Position) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut builder = GameStateBuilder::new();
        builder.add_home_player(attacker_pos).add_away_player(defender_pos);
        for &square in others {
            builder.add_away_player(Position::new(square));
        }
        let mut state = builder.build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state
            .get_mut_player_unsafe(attacker)
            .stats
            .give_skill(Skill::Juggernaut);
        for &skill in defender_skills {
            state.get_mut_player_unsafe(defender).stats.give_skill(skill);
        }
        let start = if blitz { PosAT::StartBlitz } else { PosAT::StartBlock };
        state.step_positional(start, attacker_pos);
        state.fix_blockdice(die);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::from(die));
        (state, attacker, defender, defender_pos)
    }

    /// Juggernaut: in a Blitz, the blocker may play a Both Down as a Push.
    #[test]
    fn juggernaut_may_push_on_a_both_down_in_a_blitz() {
        let (mut state, attacker, _, defender_pos) = juggernaut_block(true, BlockDice::BothDown, &[], &[]);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Home));
        state.step_simple(SimpleAT::UseSkill);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Up);
        assert!(state.is_legal_action(&Action::Positional(PosAT::FollowUp, defender_pos)));
    }

    /// Declined, the Both Down is played: the blocker without Block falls, a turnover.
    #[test]
    fn declining_juggernaut_plays_the_both_down() {
        let (mut state, attacker, _, _) = juggernaut_block(true, BlockDice::BothDown, &[], &[]);
        for _ in 0..4 {
            state.fix_d6(1); // both armour rolls
        }
        state.step_simple(SimpleAT::DontUseSkill);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
        assert_eq!(state.get_available_actions().team, Some(TeamType::Away), "a turnover");
    }

    /// Outside a Blitz there is no question: the Both Down is played.
    #[test]
    fn juggernaut_does_nothing_in_a_block() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        state
            .get_mut_player_unsafe(attacker)
            .stats
            .give_skill(Skill::Juggernaut);
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, defender_pos);
        for _ in 0..4 {
            state.fix_d6(1); // both armour rolls
        }
        state.step_simple(SimpleAT::SelectBothDown);
        assert_eq!(state.get_player_unsafe(attacker).status, PlayerStatus::Down);
    }

    /// The Blitz target can't Stand Firm ...
    #[test]
    fn juggernaut_beats_stand_firm() {
        let (state, _, _, defender_pos) = juggernaut_block(true, BlockDice::Push, &[Skill::StandFirm], &[]);
        assert_eq!(
            state.get_available_actions().team,
            Some(TeamType::Home),
            "no Stand Firm question"
        );
        assert!(state.is_legal_action(&Action::Positional(PosAT::Push, defender_pos + (1, 0))));
    }

    /// ... though a player a chain push reaches still may.
    #[test]
    fn juggernaut_lets_a_chain_pushed_player_stand_firm() {
        let (mut state, _, _, defender_pos) = juggernaut_block(true, BlockDice::Push, &[], &[(7, 2), (7, 3), (7, 4)]);
        let behind = state.get_player_id_at(defender_pos + (1, 0)).unwrap();
        state.get_mut_player_unsafe(behind).stats.give_skill(Skill::StandFirm);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert_eq!(
            state.get_available_actions().team,
            Some(TeamType::Away),
            "the Stand Firm question"
        );
    }

    /// ... nor Fend ...
    #[test]
    fn juggernaut_beats_fend() {
        let (mut state, _, _, defender_pos) = juggernaut_block(true, BlockDice::Push, &[Skill::Fend], &[]);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        assert!(state.is_legal_action(&Action::Positional(PosAT::FollowUp, defender_pos)));
    }

    /// ... nor Wrestle.
    #[test]
    fn juggernaut_beats_wrestle() {
        let (mut state, attacker, _, _) = juggernaut_block(true, BlockDice::BothDown, &[Skill::Wrestle], &[]);
        assert_eq!(
            state.get_available_actions().team,
            Some(TeamType::Home),
            "the Juggernaut question"
        );
        for _ in 0..4 {
            state.fix_d6(1); // both armour rolls
        }
        state.step_simple(SimpleAT::DontUseSkill);
        assert_eq!(
            state.get_player_unsafe(attacker).status,
            PlayerStatus::Down,
            "no Wrestle question"
        );
    }

    /// A Push on an away ball carrier; returns who holds the ball afterwards.
    fn push_the_carrier(strip_ball: bool, sure_hands: bool) -> (BallState, PlayerID) {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_ball_pos(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        assert_eq!(state.ball, BallState::Carried(defender));
        if strip_ball {
            state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::StripBall);
        }
        if sure_hands {
            state.get_mut_player_unsafe(defender).stats.give_skill(Skill::SureHands);
        }
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, defender_pos + (1, 0));
        if strip_ball && !sure_hands {
            state.fix_d8_direction(Direction::up()); //bounce to an empty square
        }
        state.step_positional(PosAT::FollowUp, attacker_pos);
        assert_eq!(state.get_player_unsafe(defender).status, PlayerStatus::Up);
        (state.ball, defender)
    }

    /// Strip Ball: a carrier pushed back by the Strip Ball player drops the ball, which bounces
    /// from the square they were pushed into. Sure Hands keeps it.
    #[test]
    fn strip_ball_knocks_the_ball_loose_on_a_push() {
        let (ball, carrier) = push_the_carrier(false, false);
        assert_eq!(ball, BallState::Carried(carrier));
        let (ball, _) = push_the_carrier(true, false);
        assert_eq!(ball, BallState::OnGround(Position::new((7, 2))));
        let (ball, carrier) = push_the_carrier(true, true);
        assert_eq!(ball, BallState::Carried(carrier), "Sure Hands");
    }

    /// Strip Ball works on a Stand Firm carrier too: they are not pushed, but still drop the
    /// ball in their own square, and it bounces.
    #[test]
    fn strip_ball_works_on_a_carrier_who_stands_firm() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_ball_pos(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::StripBall);
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::StandFirm);
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.fix_d8_direction(Direction::up()); //bounce to an empty square
        state.step_simple(SimpleAT::UseSkill);
        assert_eq!(state.get_player_unsafe(defender).position, defender_pos);
        assert_eq!(state.ball, BallState::OnGround(defender_pos + Direction::up()));
    }

    /// ... and when a player further down a chain push stands firm: nobody moves, but the
    /// carrier still loses the ball.
    #[test]
    fn strip_ball_works_when_a_chain_push_is_stopped_by_stand_firm() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_away_players(&[(7, 2), (7, 3), (7, 4)])
            .add_ball_pos(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        let firm = state.get_player_id_at(Position::new((7, 3))).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::StripBall);
        state.get_mut_player_unsafe(firm).stats.give_skill(Skill::StandFirm);
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::Push);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPush);
        state.step_positional(PosAT::Push, Position::new((7, 3)));
        state.fix_d8_direction(Direction::up()); //bounce to an empty square
        state.step_simple(SimpleAT::UseSkill);
        assert_eq!(state.get_player_unsafe(defender).position, defender_pos);
        assert_eq!(state.ball, BallState::OnGround(defender_pos + Direction::up()));
    }

    /// A Stumble the defender's Dodge turns into a push is a push: the carrier loses the ball.
    #[test]
    fn strip_ball_works_on_a_dodged_stumble() {
        let attacker_pos = Position::new((5, 3));
        let defender_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .add_home_player(attacker_pos)
            .add_away_player(defender_pos)
            .add_ball_pos(defender_pos)
            .build();
        let attacker = state.get_player_id_at(attacker_pos).unwrap();
        let defender = state.get_player_id_at(defender_pos).unwrap();
        state.get_mut_player_unsafe(attacker).stats.give_skill(Skill::StripBall);
        state.get_mut_player_unsafe(defender).stats.give_skill(Skill::Dodge);
        state.step_positional(PosAT::StartBlock, attacker_pos);
        state.fix_blockdice(BlockDice::PowPush);
        state.step_positional(PosAT::Block, defender_pos);
        state.step_simple(SimpleAT::SelectPowPush);
        let pushed_to = defender_pos + (1, 0);
        state.step_positional(PosAT::Push, pushed_to);
        state.fix_d8_direction(Direction::up()); //bounce to an empty square
        state.step_positional(PosAT::FollowUp, attacker_pos);
        assert_eq!(state.get_player_unsafe(defender).status, PlayerStatus::Up);
        assert_eq!(state.ball, BallState::OnGround(pushed_to + Direction::up()));
    }

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

use serde::{Deserialize, Serialize};

use crate::core::dices::{D6Target, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6};
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{
    other_team, Action, AvailableActions, BoardDims, Coord, Direction, Position, ProcState, Procedure,
};
use crate::core::model::{BallState, PlayerID, TeamType};
use crate::core::procedures::game_procs::TURNS_PER_HALF;
use crate::core::table::{PosAT, SimpleAT, Skill};

use crate::core::procedures::any_proc::AnyProc;

use super::procedure_tools::{SimpleProc, SimpleProcContainer};
use super::TurnoverIfPossessionLost;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PickupProc {
    target: D6Target,
    id: PlayerID,
}
impl PickupProc {
    pub fn new(id: PlayerID, target: D6Target) -> AnyProc {
        AnyProc::PickupProc(SimpleProcContainer::new(PickupProc { target, id }))
    }
}
impl SimpleProc for PickupProc {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::SureHands)
    }

    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.set_ball(BallState::Carried(self.id));
        // The active player just picked up the ball this activation; they are
        // owed one follow-up move action (the pathfinder stops a non-carrier's
        // path at the ball, so they need a fresh carrier-routed path to run).
        // See the field doc on `GameInfo::pickup_this_activation`.
        game_state.info.pickup_this_activation = true;
        let player = game_state.get_player_unsafe(self.id);
        if player.position.x == game_state.get_endzone_x(player.stats.team) {
            game_state.info.handle_td_by = Some(self.id);
        }
        Vec::new()
    }

    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.get_mut_player(self.id).unwrap().used = true;
        game_state.info.turnover = true;
        vec![Bounce::new()]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Bounce {
    kick: bool,
}
impl Bounce {
    pub fn new() -> AnyProc {
        AnyProc::Bounce(Bounce { kick: false })
    }
    pub fn new_with_kick_arg(kick: bool) -> AnyProc {
        AnyProc::Bounce(Bounce { kick })
    }
    /// A kickoff bounce: landing out of bounds or on the kicking team's half is a touchback.
    /// Public so MCTS's chance model can tell the touchback squares apart (plan 061).
    pub fn is_kick(&self) -> bool {
        self.kick
    }
}
impl Procedure for Bounce {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let dice = match input {
            ProcInput::Nothing => return ProcState::NeedRoll(RequestedRoll::D8),
            ProcInput::Roll(RollResult::D8(dice)) => dice,
            _ => panic!("Unexpected input {:?} for Bounce", input),
        };
        let current_ball_pos = game_state.get_ball_position().unwrap();
        let new_pos = current_ball_pos + Direction::from(dice);

        if self.kick
            && (game_state.is_out(new_pos) || game_state.is_on_team_side(new_pos, game_state.info.kicking_this_drive))
        {
            return ProcState::DoneNew(Touchback::new());
        }

        // Out-of-bounds must be checked *before* occupancy: `new_pos` can
        // be past the physical board when the ball bounces outward from a
        // square on the border ring (deviating kicks / scatter sequences
        // leave it there transiently), and `get_player_at` indexes the
        // array directly. Order is otherwise equivalent — border and OOB
        // squares never hold players.
        if game_state.is_out(new_pos) {
            // The throw-in origin must be on the pitch; if the ball was
            // itself on the ring, walk back along the bounce direction
            // (mirrors ThrowIn's own OOB re-request handling).
            let mut from = current_ball_pos;
            let direction = Direction::from(dice);
            while game_state.is_out(from) {
                from -= direction;
            }
            ProcState::DoneNew(ThrowIn::new(from))
        } else if let Some((catcher_id, can_catch)) = game_state.get_player_at(new_pos).map(|p| (p.id, p.can_catch())) {
            if can_catch {
                // The ball is in the catcher's square while they attempt
                // the catch — a failed catch bounces on from *their*
                // square. (Also records the square in `bounce_squares`,
                // which keeps recurring bounce states distinct for the
                // MCTS search graph.)
                game_state.set_ball(BallState::InAir(new_pos));
                // -1: catching a ball that has Bounced.
                ProcState::DoneNew(Catch::new_with_kick_arg(
                    catcher_id,
                    *game_state.get_catch_target(catcher_id).unwrap().add_modifer(-1),
                    self.kick,
                ))
            } else {
                //will run bounce again
                game_state.set_ball(BallState::InAir(new_pos));
                ProcState::NotDone
            }
        } else {
            game_state.set_ball(BallState::OnGround(new_pos));
            // debug assert there's no player on that square
            debug_assert!(game_state.get_player_at(new_pos).is_none());
            ProcState::Done
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ThrowIn {
    from: Position,
}
impl ThrowIn {
    pub fn new(from: Position) -> AnyProc {
        AnyProc::ThrowIn(ThrowIn { from })
    }
    /// The on-pitch square the ball is thrown in from.
    pub fn origin(&self) -> Position {
        self.from
    }
    /// The board direction a throw-in `D3` maps to, given which edge the
    /// ball left by. Public so MCTS's chance-outcome model can pick a
    /// *mirror-invariant* representative rather than one in `D3` order
    /// (plan 023, H-c).
    pub fn get_throw_in_direction(&self, dice: D3, dims: BoardDims) -> Direction {
        // Last playable column/row of the logical board (index 0 and width-1 /
        // height-1 are the OOB border). Runtime dims → can't be `const` match arms.
        let max_x: Coord = dims.width - 2;
        let max_y: Coord = dims.height - 2;
        let Position { x, y } = self.from;
        let directions: [(Coord, Coord); 3] = if x == 1 && y == 1 {
            [(1, 0), (1, 1), (0, 1)]
        } else if x == 1 && y == max_y {
            [(1, 0), (1, -1), (0, -1)]
        } else if x == max_x && y == 1 {
            [(-1, 0), (-1, 1), (0, 1)]
        } else if x == max_x && y == max_y {
            [(-1, 0), (-1, -1), (0, -1)]
        } else if x == 1 {
            [(1, 1), (1, 0), (1, -1)]
        } else if x == max_x {
            [(-1, 1), (-1, 0), (-1, -1)]
        } else if y == 1 {
            [(1, 1), (0, 1), (-1, 1)]
        } else if y == max_y {
            [(1, -1), (0, -1), (-1, -1)]
        } else {
            panic!("very wrong!")
        };
        Direction::from(directions[dice as usize - 1])
    }

    /// The square this throw-in roll would land on (before occupancy is
    /// considered): `from + direction * throw_in_length(distance)`. Pure, and must mirror
    /// `step`'s arithmetic exactly.
    /// Used by the MCTS scripted-outcome picker to choose a roll that keeps
    /// the ball in bounds — an out-of-bounds landing re-requests the roll,
    /// which under a deterministic scripted pick can loop forever on small
    /// boards.
    pub fn target_square(&self, direction: D3, distance: Sum2D6, dims: BoardDims) -> Position {
        let dir = self.get_throw_in_direction(direction, dims);
        self.from + dir * Self::throw_in_length(distance, dims)
    }

    /// How many squares from `from` the ball lands. S3: the ball travels 2D6 squares "counting
    /// the square underneath the Blood Bowl logo" — `from`, the last square it occupied — "as
    /// the first square", so it lands 2D6 - 1 squares away. Scaled down on narrow boards (a
    /// no-op on the full pitch) so a throw-in rarely lands out of bounds again, then capped at
    /// half the board width as a final safety net.
    fn throw_in_length(distance: Sum2D6, dims: BoardDims) -> Coord {
        ((distance as Coord - 1) / dims.scatter_divisor()).min(dims.max_scatter())
    }
}
impl Procedure for ThrowIn {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let (direction, length) = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::ThrowIn);
            }
            ProcInput::Roll(RollResult::ThrowIn { direction, distance }) => {
                let dims = game_state.board_dims;
                (
                    self.get_throw_in_direction(direction, dims),
                    Self::throw_in_length(distance, dims),
                )
            }
            _ => panic!("Unexpected input {:?} for ThrowIn", input),
        };
        let target: Position = self.from + direction * length;

        if game_state.is_out(target) {
            self.from = target - direction;

            while game_state.is_out(self.from) {
                self.from -= direction;
            }

            ProcState::NeedRoll(RequestedRoll::ThrowIn)
        } else {
            match game_state.get_player_at(target).map(|p| (p.id, p.can_catch())) {
                Some((catcher_id, true)) => {
                    // Same as the Bounce → Catch hand-off: the ball is in
                    // the catcher's square for the attempt, so a failed
                    // catch bounces from there.
                    game_state.set_ball(BallState::InAir(target));
                    // -1: catching a ball that has been Thrown-in.
                    let target = *game_state.get_catch_target(catcher_id).unwrap().add_modifer(-1);
                    ProcState::DoneNew(Catch::new(catcher_id, target))
                }
                _ => {
                    game_state.set_ball(BallState::InAir(target));
                    ProcState::DoneNew(Bounce::new())
                }
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Catch {
    id: PlayerID,
    target: D6Target,
    kick: bool,
}
impl Catch {
    pub fn new(id: PlayerID, target: D6Target) -> AnyProc {
        AnyProc::Catch(SimpleProcContainer::new(Catch {
            id,
            target,
            kick: false,
        }))
    }
    pub fn new_with_kick_arg(id: PlayerID, target: D6Target, kick: bool) -> AnyProc {
        AnyProc::Catch(SimpleProcContainer::new(Catch { id, target, kick }))
    }
}
impl SimpleProc for Catch {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        Some(Skill::Catch)
    }

    /// The ball is in the catcher's square for the attempt — a failed
    /// catch bounces from *there*. Bounce/ThrowIn place it before
    /// pushing this proc (and then this is a no-op); pass, handoff and
    /// deflect paths reach here with the ball still tracked at its
    /// origin, so this is the single point that enforces the invariant
    /// for every catch site.
    fn on_start(&self, game_state: &mut GameState) {
        let pos = game_state.get_player_unsafe(self.id).position;
        if game_state.ball != BallState::InAir(pos) {
            game_state.set_ball(BallState::InAir(pos));
        }
    }

    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.set_ball(BallState::Carried(self.id));
        let player = game_state.get_player_unsafe(self.id);
        if player.position.x == game_state.get_endzone_x(player.stats.team) {
            game_state.info.handle_td_by = Some(self.id);
        }
        Vec::new()
    }

    fn apply_failure(&mut self, _game_state: &mut GameState) -> Vec<AnyProc> {
        vec![Bounce::new_with_kick_arg(self.kick)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Touchback {}
impl Touchback {
    pub fn new() -> AnyProc {
        AnyProc::Touchback(Touchback {})
    }
}
impl Procedure for Touchback {
    fn step(&mut self, game_state: &mut GameState, action: ProcInput) -> ProcState {
        if let ProcInput::Action(Action::Positional(_, position)) = action {
            game_state.set_ball(match game_state.get_player_id_at(position) {
                Some(id) => BallState::Carried(id),
                None => BallState::OnGround(position),
            });
            ProcState::Done
        } else {
            // S3, Touchbacks: the receiving coach gives the ball to one of their Standing players,
            // or, with none, places it on any unoccupied square in their half.
            let team = other_team(game_state.info.kicking_this_drive);
            let mut positions: Vec<_> = game_state
                .get_players_on_pitch_in_team(team)
                .filter(|p| p.status == crate::core::model::PlayerStatus::Up)
                .map(|p| p.position)
                .collect();
            if positions.is_empty() {
                let dims = game_state.board_dims;
                positions = Position::all_positions()
                    .filter(|&pos| !dims.is_out(pos) && dims.is_on_team_side(pos, team))
                    .filter(|&pos| game_state.get_player_at(pos).is_none())
                    .collect();
            }
            let mut aa = AvailableActions::new(team);
            aa.insert_positional(PosAT::SelectPosition, positions);
            ProcState::NeedAction(aa)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Touchdown {
    id: PlayerID,
}
impl Touchdown {
    pub fn new(id: PlayerID) -> AnyProc {
        AnyProc::Touchdown(Touchdown { id })
    }
}
impl Procedure for Touchdown {
    fn step(&mut self, game_state: &mut GameState, _action: ProcInput) -> ProcState {
        if let BallState::Carried(carrier_id) = game_state.ball {
            if carrier_id == self.id {
                game_state.get_mut_team_from_player(self.id).unwrap().score += 1;
                game_state.get_mut_player_unsafe(self.id).used = true;
                // The scoring team kicks off to its opponent. Setting this to
                // the conceding team makes scoring self-reinforcing (the scorer
                // receives again), which compounds any per-kickoff edge into a
                // large win-rate gap — plan 023 B2.
                let scorer = game_state.get_player_unsafe(self.id).stats.team;
                game_state.info.kickoff_by_team = Some(scorer);
                // S3, Scoring in the Opponent's Turn: the scoring team skips its next turn, so the
                // conceding team plays first after the kick-off. `Half` gives the next turn to the
                // team that is behind on the turn track; a skip past the half's last turn ends it.
                if game_state.turn_team() != Some(scorer) {
                    let info = &mut game_state.info;
                    let turn = match scorer {
                        TeamType::Home => &mut info.home_turn,
                        TeamType::Away => &mut info.away_turn,
                    };
                    *turn = (*turn + 1).min(TURNS_PER_HALF);
                }
            }
        }

        ProcState::Done
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PassResult {
    Accurate,
    Inaccurate,
    /// Pre-S3 (a modified 1 deviated D8 x D6 from the passer). Never produced any more; kept so
    /// stored states that hold it still load.
    WildlyInaccurate,
    Fumble,
}

/// S3, Pass Actions: a Passing Ability Test (the target with the range, marking, weather and
/// skill modifiers already applied, `modifier` their sum). Success or a natural 6 is
/// Accurate: the ball heads for the target square. A natural 1, or a 1 after modifiers, is a
/// Fumble: it bounces from the passer and the turn is over. Anything else is Inaccurate: the
/// ball Scatters (3) from the target square. A failed test (Inaccurate or Fumble) may be
/// re-rolled once, with the Pass skill or a team re-roll.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Pass {
    pos: Position,
    pass: D6Target,
    modifier: i8,
    /// The test has been re-rolled: its next result stands.
    #[serde(default)]
    rerolled: bool,
    /// A failed result waiting on the passer's coach's team re-roll decision.
    #[serde(default)]
    failed: Option<PassResult>,
}
impl Pass {
    pub fn new(pos: Position, pass: D6Target, modifier: i8) -> AnyProc {
        AnyProc::Pass(Pass {
            pos,
            pass,
            modifier,
            rerolled: false,
            failed: None,
        })
    }
    /// The square the pass is aimed at: an inaccurate pass scatters from here. Public so MCTS's
    /// chance model can enumerate where the ball lands (plan 061).
    pub fn target(&self) -> Position {
        self.pos
    }

    fn result_of(&self, roll: D6) -> PassResult {
        if roll == D6::One || (roll as i8) + self.modifier <= 1 {
            PassResult::Fumble
        } else if self.pass.is_success(roll) {
            PassResult::Accurate
        } else {
            PassResult::Inaccurate
        }
    }

    fn resolve(&self, result: PassResult, game_state: &mut GameState) -> ProcState {
        match result {
            PassResult::Accurate => {
                let from = game_state.get_ball_position().unwrap();
                ProcState::DoneNewProcs(vec![
                    TurnoverIfPossessionLost::new(),
                    DeflectOrResolve::new(from, self.pos, PassResult::Accurate, None),
                ])
            }
            PassResult::Fumble => {
                game_state.info.turnover = true;
                ProcState::DoneNew(Bounce::new())
            }
            PassResult::Inaccurate => ProcState::NeedRoll(RequestedRoll::Scatter),
            PassResult::WildlyInaccurate => unreachable!("not an S3 result"),
        }
    }
}
impl Procedure for Pass {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(roll)) => {
                let result = self.result_of(roll);                if result != PassResult::Accurate && !self.rerolled {
                    let passer = game_state.get_active_player().unwrap();
                    let team = passer.stats.team;
                    if passer.has_skill(Skill::Pass) {
                        self.rerolled = true;
                        return ProcState::NeedRoll(RequestedRoll::D6);
                    }
                    if game_state.may_use_team_reroll(team) {
                        self.failed = Some(result);
                        let mut aa = AvailableActions::new(team);
                        aa.insert_simple(SimpleAT::UseReroll);
                        aa.insert_simple(SimpleAT::DontUseReroll);
                        return ProcState::NeedAction(aa);
                    }
                }
                self.resolve(result, game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseReroll)) if self.failed.is_some() => {
                let team = game_state.get_active_player().unwrap().stats.team;
                game_state.get_mut_team(team).use_reroll();
                self.failed = None;
                self.rerolled = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)) if self.failed.is_some() => {
                let result = self.failed.take().unwrap();
                self.resolve(result, game_state)
            }
            ProcInput::Roll(RollResult::Scatter(r1, r2, r3)) => {
                let from = game_state.get_ball_position().unwrap(); //or just acive plater...
                let mut target = self.pos;
                let mut throwin_pos = None;
                for d in [r1, r2, r3].iter().map(|r| Direction::from(*r)) {
                    let new_target = target + d;
                    if game_state.is_out(new_target) {
                        throwin_pos = Some(target);
                        break;
                    }
                    target = new_target;
                }
                ProcState::DoneNewProcs(vec![
                    TurnoverIfPossessionLost::new(),
                    DeflectOrResolve::new(from, target, PassResult::Inaccurate, throwin_pos),
                ])
            }
            _ => panic!("Unexpected input {:?} for Pass", input),
        }
    }
}
/// S3, Interceptions: the Agility Test is -3 against an Accurate Pass, -2 against an Inaccurate
/// one (on top of the marking and weather modifiers in `GameState::get_intercepters`).
pub fn interception_modifier(result: PassResult) -> i8 {
    match result {
        PassResult::Accurate => -3,
        _ => -2,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DeflectOrResolve {
    from: Position,
    to: Position,
    throw_in_pos: Option<Position>,
    result: PassResult,
    intercepters: Vec<(Position, D6Target)>,
}
impl DeflectOrResolve {
    pub fn new(from: Position, to: Position, result: PassResult, throw_in_pos: Option<Position>) -> AnyProc {
        AnyProc::DeflectOrResolve(DeflectOrResolve {
            from,
            to,
            throw_in_pos,
            result,
            intercepters: Vec::new(),
        })
    }
}
impl Procedure for DeflectOrResolve {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let active_team = game_state.get_active_player().unwrap().stats.team;
        let deflect_team = other_team(active_team);
        let interceptor: Option<(Position, D6Target)> = match input {
            ProcInput::Nothing => {
                self.intercepters = game_state.get_intercepters(deflect_team, self.from, self.to);
                if self.intercepters.is_empty() {
                    None
                } else if self.intercepters.len() == 1 {
                    Some(self.intercepters[0])
                } else {
                    let mut aa = AvailableActions::new(deflect_team);
                    aa.insert_positional(
                        PosAT::SelectPosition,
                        self.intercepters.iter().map(|(pos, _)| *pos).collect(),
                    );
                    return ProcState::NeedAction(aa);
                }
            }
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, pos)) => self
                .intercepters
                .iter()
                .find(|(p, _)| *p == pos)
                .map(|(p, target)| (*p, *target)),
            _ => panic!("Unexpected input {:?} for Interception", input),
        };
        let failed_deflect_proc: AnyProc = {
            if let Some(throw_in_pos) = self.throw_in_pos {
                debug_assert!(!game_state.is_out(throw_in_pos));
                ThrowIn::new(throw_in_pos)
            } else {
                // The catch has no modifier for the pass (S3, Catching the Ball). A Prone or
                // Stunned player in the square fails it automatically: the ball bounces from
                // there, as from an empty square.
                match game_state.get_player_at(self.to) {
                    Some(player) if player.can_catch() => {
                        Catch::new(player.id, game_state.get_catch_target(player.id).unwrap())
                    }
                    _ => Bounce::new(),
                }
            }
        };
        if let Some((pos, mut target)) = interceptor {
            // S3, Interceptions: -3 against an Accurate Pass, -2 against an Inaccurate one.
            target.add_modifer(interception_modifier(self.result));
            let id = game_state.get_player_id_at(pos).unwrap();
            ProcState::DoneNew(Deflect::new(id, target, failed_deflect_proc))
        } else {
            game_state.set_ball(BallState::InAir(self.to));
            ProcState::DoneNew(failed_deflect_proc)
        }
        //PASSING INTERFERENCE
        // If the pass was not fumbled, a single player from the opposing team may be able
        // to attempt to interfere with the pass, hoping to 'Deflect' the pass or, in some
        // rare cases, to 'Intercept' the pass. To determine if any opposition players are
        // able to attempt passing interference, place the range ruler so that the circle
        // at the end is over the centre of the square occupied by the player performing
        // the Pass action. Position the other end so that the ruler covers the square in
        // which the ball will land. Note that, depending upon the Passing Ability test,
        // this may not be the target square!
        //
        // To attempt to interfere with a pass, an opposition player must be:
        //
        // A Standing player that has not lost their Tackle Zone (as described on page 26).
        // Occupying a square that is between the square occupied by the player performing the Pass action and the square in which the ball will land.
        // In a square that is at least partially beneath the range ruler when placed as described above.
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Deflect {
    id: PlayerID,
    target: D6Target,
    failed_deflect_proc: Option<Box<AnyProc>>,
}

impl Deflect {
    pub fn new(id: PlayerID, target: D6Target, failed_deflect_proc: AnyProc) -> AnyProc {
        AnyProc::Deflect(SimpleProcContainer::new(Deflect {
            id,
            target,
            failed_deflect_proc: Some(Box::new(failed_deflect_proc)),
        }))
    }
}
impl SimpleProc for Deflect {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        None
    }

    fn apply_failure(&mut self, _game_state: &mut GameState) -> Vec<AnyProc> {
        vec![*self.failed_deflect_proc.take().unwrap()]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }

    /// S3: a successful interception gains possession at once — no catch follows. The pass
    /// below (`TurnoverIfPossessionLost`) turns it over. An interceptor standing in the end zone
    /// they score in scores in the opponent's turn.
    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.set_ball(BallState::Carried(self.id));
        let player = game_state.get_player_unsafe(self.id);
        if player.position.x == game_state.get_endzone_x(player.stats.team) {
            game_state.info.handle_td_by = Some(self.id);
        }
        Vec::new()
    }
}
#[cfg(test)]
mod tests {

    use crate::core::dices::BlockDice;
    use crate::core::dices::D8;
    use crate::core::model::*;
    use crate::core::table::*;
    use crate::core::{gamestate::GameStateBuilder, model::Position, table::PosAT};

    /// A bounce outward from a square on the border ring must not index
    /// past the physical board: `is_out` has to be checked before
    /// occupancy (`board[new_pos]`). The ball sits on the ring
    /// transiently (deviating kicks, scatter sequences); bouncing outward
    /// from there computed `board[y+1]` one past the array — caught on
    /// the 14x7 training board under NN-guided search (gamestate.rs:903).
    /// The resulting throw-in origin must also be walked back onto the
    /// pitch, or `get_throw_in_direction` panics on the ring square.
    #[test]
    fn bounce_off_border_square_throws_in_without_indexing_oob() {
        use crate::core::dices::{RollResult, Sum2D6, D3};
        use crate::core::procedures::ball_procs::Bounce;
        use crate::core::procedures::AnyProc;

        let mut state = GameStateBuilder::new().add_home_player(Position::new((2, 2))).build();
        let dims = state.board_dims;
        let border_pos = Position::new((5, dims.height - 1)); // bottom border ring
        state.set_ball(BallState::InAir(border_pos));

        let AnyProc::Bounce(mut proc) = Bounce::new() else {
            unreachable!()
        };
        let result = proc.step(&mut state, ProcInput::Roll(RollResult::D8(D8::from(Direction::down()))));
        let ProcState::DoneNew(AnyProc::ThrowIn(mut throw_in)) = result else {
            panic!("outward bounce from the border must resolve to a throw-in, got {result:?}");
        };

        // The throw-in origin must be on the pitch: resolving a roll walks
        // `get_throw_in_direction`, which panics on a ring square.
        let _ = throw_in.step(
            &mut state,
            ProcInput::Roll(RollResult::ThrowIn {
                direction: D3::One,
                distance: Sum2D6::Two,
            }),
        );
    }

    /// The squares a touchback offers.
    fn touchback_offers(state: &mut crate::core::gamestate::GameState) -> (Option<TeamType>, Vec<Position>) {
        use crate::core::procedures::ball_procs::Touchback;
        let ProcState::NeedAction(aa) = (Touchback {}).step(state, ProcInput::Nothing) else {
            panic!("a touchback asks the receiving coach");
        };
        let mut actions = Vec::new();
        aa.collect_non_path_actions(&mut actions);
        let squares = actions
            .into_iter()
            .map(|action| match action {
                Action::Positional(PosAT::SelectPosition, pos) => pos,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        (aa.get_team(), squares)
    }

    /// S3, Touchbacks: with no Standing receiver, the receiving coach places the ball on any
    /// unoccupied square in their half, where it stays (no bounce). This also keeps a touchback
    /// with nobody on the pitch (routine in 2-player small-board games) from emitting an empty
    /// `NeedAction`, which would deadlock every consumer.
    #[test]
    fn touchback_with_no_standing_receivers_places_the_ball_in_the_receiving_half() {
        use crate::core::procedures::ball_procs::Touchback;

        // Home kicks, Away receives — and Away's only player is prone.
        let prone_pos = Position::new((20, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((2, 2)))
            .add_away_player(prone_pos)
            .build();
        state.info.kicking_this_drive = TeamType::Home;
        let prone = state.get_player_id_at(prone_pos).unwrap();
        state.get_mut_player_unsafe(prone).status = PlayerStatus::Down;

        let (team, offered) = touchback_offers(&mut state);
        assert_eq!(team, Some(TeamType::Away));
        assert!(!offered.is_empty());
        assert!(offered
            .iter()
            .all(|&pos| state.board_dims.is_on_team_side(pos, TeamType::Away) && state.get_player_at(pos).is_none()));

        let square = offered[0];
        let result = (Touchback {}).step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, square)),
        );
        assert!(matches!(result, ProcState::Done));
        assert_eq!(state.ball, BallState::OnGround(square));
    }

    /// S3, Catching the Ball: "If a Prone or Stunned player ... is required to Catch a ball, they
    /// will automatically fail ... and the ball will Bounce" — for a kick-off landing on them ...
    #[test]
    fn a_kickoff_landing_on_a_prone_player_bounces() {
        use crate::core::procedures::kickoff_procs::LandKickoff;
        use crate::core::procedures::AnyProc;

        let prone_pos = Position::new((20, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((2, 2)))
            .add_away_player(prone_pos)
            .build();
        // The kick lands in the receiving half: kicked by the team whose half it is not.
        state.info.kicking_this_drive = if state.board_dims.is_on_team_side(prone_pos, TeamType::Away) {
            TeamType::Home
        } else {
            TeamType::Away
        };
        let prone = state.get_player_id_at(prone_pos).unwrap();
        state.get_mut_player_unsafe(prone).status = PlayerStatus::Down;
        state.set_ball(BallState::InAir(prone_pos));

        let result = (LandKickoff {}).step(&mut state, ProcInput::Nothing);
        assert!(matches!(result, ProcState::DoneNew(AnyProc::Bounce(_))), "got {result:?}");
    }

    /// ... and for a pass landing on them.
    #[test]
    fn a_pass_landing_on_a_prone_player_bounces() {
        use crate::core::procedures::ball_procs::{DeflectOrResolve, PassResult};
        use crate::core::procedures::AnyProc;

        let passer_pos = Position::new((5, 5));
        let prone_pos = Position::new((8, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(passer_pos)
            .add_home_player(prone_pos)
            .add_ball_pos(passer_pos)
            .build();
        state.info.active_player = state.get_player_id_at(passer_pos);
        let prone = state.get_player_id_at(prone_pos).unwrap();
        state.get_mut_player_unsafe(prone).status = PlayerStatus::Down;

        let AnyProc::DeflectOrResolve(mut proc) =
            DeflectOrResolve::new(passer_pos, prone_pos, PassResult::Inaccurate, None)
        else {
            unreachable!()
        };
        let result = proc.step(&mut state, ProcInput::Nothing);
        assert!(matches!(result, ProcState::DoneNew(AnyProc::Bounce(_))), "got {result:?}");
        assert_eq!(state.ball, BallState::InAir(prone_pos));
    }

    /// A Prone or Stunned receiver is not offered the touchback ball.
    #[test]
    fn touchback_goes_to_a_standing_receiver_only() {
        let standing_pos = Position::new((20, 5));
        let prone_pos = Position::new((20, 7));
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((2, 2)))
            .add_away_player(standing_pos)
            .add_away_player(prone_pos)
            .build();
        state.info.kicking_this_drive = TeamType::Home;
        let prone = state.get_player_id_at(prone_pos).unwrap();
        state.get_mut_player_unsafe(prone).status = PlayerStatus::Stunned;

        assert_eq!(touchback_offers(&mut state).1, vec![standing_pos]);
    }

    #[test]
    fn pickup_fail_and_bounce() -> Result<()> {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);

        state.step_positional(PosAT::StartMove, start_pos);
        state.fix_d6(2); //fail pickup (3+)
        state.step_positional(PosAT::Move, ball_pos);
        state.fix_d8(d8_fix as u8);
        state.step_simple(SimpleAT::DontUseReroll);

        assert!(matches!(state.ball, BallState::OnGround(pos) if pos == ball_pos + direction));
        // The ball settled on the ground, so the bounce-square record is cleared.
        assert!(state.bounce_squares.is_empty());

        Ok(())
    }

    #[test]
    fn set_ball_tracks_only_in_air_squares() {
        let mut state = GameStateBuilder::new().add_home_player(Position::new((1, 1))).build();
        let carrier = state.get_player_id_at(Position::new((1, 1))).unwrap();
        assert!(state.bounce_squares.is_empty());

        // Each in-air transition appends the square it moved to.
        let a = Position::new((5, 5));
        let b = Position::new((5, 6));
        state.set_ball(BallState::InAir(a));
        state.set_ball(BallState::InAir(b));
        assert_eq!(state.bounce_squares.as_slice(), &[a, b]);

        // Settling on the ground clears the record...
        state.set_ball(BallState::OnGround(b));
        assert!(state.bounce_squares.is_empty());

        // ...and so does being caught / off the pitch.
        state.set_ball(BallState::InAir(a));
        assert_eq!(state.bounce_squares.as_slice(), &[a]);
        state.set_ball(BallState::Carried(carrier));
        assert!(state.bounce_squares.is_empty());

        state.set_ball(BallState::InAir(a));
        state.set_ball(BallState::OffPitch);
        assert!(state.bounce_squares.is_empty());
    }

    #[test]
    fn bounce_through_occupied_square_is_recorded_then_cleared() {
        // Ball fails to be picked up and bounces onto a prone (can't-catch)
        // team-mate's square before bouncing again onto empty ground. The
        // intermediate occupied square must be recorded while the ball is
        // in the air, then cleared once it lands.
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let occupied = ball_pos + Direction::from(D8::One); // first bounce lands here
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(occupied)
            .add_ball_pos(ball_pos)
            .build();
        // Knock the occupant down so it can't catch — forces a second bounce.
        let occupant = state.get_player_id_at(occupied).unwrap();
        state.get_mut_player_unsafe(occupant).status = PlayerStatus::Down;

        state.step_positional(PosAT::StartMove, start_pos);
        state.fix_d6(2); // fail pickup (3+)
        state.step_positional(PosAT::Move, ball_pos);
        state.fix_d8(D8::One as u8); // bounce onto the prone player's square
        state.fix_d8(D8::Two as u8); // bounce again onto empty ground
        state.step_simple(SimpleAT::DontUseReroll);

        // Landed on the ground → record cleared.
        assert!(matches!(state.ball, BallState::OnGround(_)));
        assert!(state.bounce_squares.is_empty());
    }

    /// A ball that bounces onto a standing player is momentarily in that
    /// player's square while they attempt the catch — so a *failed* catch
    /// bounces on from the catcher's square, not from wherever the ball
    /// bounced in from. (Also load-bearing for MCTS: without the ball
    /// moving, a throw-in ⇄ failed-catch circuit reproduces a byte-equal
    /// GameState and recombination turns the search graph cyclic.)
    #[test]
    fn failed_catch_bounces_from_catcher_square() {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let dir = Direction::from(D8::One);
        let catcher_pos = ball_pos + dir;
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(catcher_pos)
            .add_ball_pos(ball_pos)
            .build();

        state.step_positional(PosAT::StartMove, start_pos);
        state.fix_d6(2); // fail pickup (3+)
        state.step_positional(PosAT::Move, ball_pos);

        state.fix_d8(D8::One as u8); // bounce onto the standing team-mate
        state.fix_d6(1); // they fail the catch
        state.step_simple(SimpleAT::DontUseReroll); // decline the pickup reroll

        state.fix_d8(D8::One as u8); // bounce on from the catcher's square
        state.step_simple(SimpleAT::DontUseReroll); // decline the catch reroll

        assert!(
            matches!(state.ball, BallState::OnGround(pos) if pos == catcher_pos + dir),
            "failed catch must bounce from the catcher's square, got {:?}",
            state.ball
        );
    }

    /// S3, Catching the Ball: an AG 3 player catches on a 3+; a hand-off has no modifier.
    #[test]
    fn a_handoff_catch_has_no_modifier() {
        let start_pos = Position::new((2, 1));
        let target_pos = Position::new((3, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(target_pos)
            .add_ball_pos(start_pos)
            .build();
        let receiver = state.get_player_id_at(target_pos).unwrap();
        state.step_positional(PosAT::StartHandoff, start_pos);
        state.fix_d6(3);
        state.step_positional(PosAT::Handoff, target_pos);
        assert_eq!(state.ball, BallState::Carried(receiver));
    }

    /// S3, Catch: "may re-roll any failed Agility Test when attempting to Catch" — not once per
    /// turn like Dodge. A Catch player who fails, sees the ball bounce off a team-mate and back,
    /// re-rolls the second catch too.
    #[test]
    fn the_catch_skill_rerolls_every_catch() {
        let ball_pos = Position::new((5, 5));
        let catcher_pos = ball_pos + Direction::up();
        let team_mate_pos = catcher_pos + Direction::right();
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((1, 1)))
            .add_home_player(catcher_pos)
            .add_home_player(team_mate_pos)
            .add_ball_pos(ball_pos)
            .build();
        state.get_mut_team(TeamType::Home).rerolls = 0;
        let catcher = state.get_player_id_at(catcher_pos).unwrap();
        state.get_mut_player_unsafe(catcher).stats.give_skill(Skill::Catch);

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));
        state.fix_d6(2); // fail pickup (3+)
        state.fix_d8_direction(Direction::up()); // bounce onto the Catch player
        state.fix_d6(1); // fail
        state.fix_d6(1); // Catch re-roll fails
        state.fix_d8_direction(Direction::right()); // bounce onto the team-mate
        state.fix_d6(1); // fail
        state.fix_d8_direction(Direction::left()); // back onto the Catch player
        state.fix_d6(1); // fail
        state.fix_d6(6); // Catch re-roll again
        state.step_positional(PosAT::Move, ball_pos);
        assert_eq!(state.ball, BallState::Carried(catcher));
    }

    /// ... but catching a ball that has Bounced is -1: an AG 3 player then needs a 4+.
    #[test]
    fn a_bounced_catch_is_minus_one() {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let dir = Direction::from(D8::One);
        let catcher_pos = ball_pos + dir;
        let catch_with = |roll: u8| {
            let mut state = GameStateBuilder::new()
                .add_home_player(start_pos)
                .add_home_player(catcher_pos)
                .add_ball_pos(ball_pos)
                .build();
            let catcher = state.get_player_id_at(catcher_pos).unwrap();
            state.get_mut_team(TeamType::Home).rerolls = 0;
            state.step_positional(PosAT::StartMove, start_pos);
            state.fix_d6(2); // fail pickup (3+)
            state.fix_d8(D8::One as u8); // bounce onto the standing team-mate
            state.fix_d6(roll);
            if roll < 4 {
                state.fix_d8(D8::One as u8); // the failed catch bounces on
            }
            state.step_positional(PosAT::Move, ball_pos);
            state.ball == BallState::Carried(catcher)
        };
        assert!(!catch_with(3));
        assert!(catch_with(4));
    }

    /// Same rule for a throw-in that lands on a player: the ball is in
    /// their square for the catch attempt, and a failed catch bounces
    /// from there.
    #[test]
    fn throw_in_onto_catcher_bounces_from_catcher_square() {
        let ball_pos = Position::new((5, 1)); // on the top edge
        let start_pos = Position::new((5, 4));
        let up = Direction::up();
        // ThrowIn from y==1 with D3::One goes direction (1,1); 2D6 - 1 squares
        // (S3 counts the origin as the first) — fix 1+2 = 3 → lands at ball_pos + (2,2).
        let catcher_pos = ball_pos + (2, 2);
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(catcher_pos)
            .add_ball_pos(ball_pos)
            .build();

        state.step_positional(PosAT::StartMove, start_pos);
        state.fix_d6(2); // fail pickup (3+)
        state.step_positional(PosAT::Move, ball_pos);

        state.fix_d8(D8::from(up) as u8); // bounce over the edge — out of bounds
        state.fix_d3(1); // throw-in direction (1,1)
        state.fix_d6(1); // throw-in distance...
        state.fix_d6(2); // ...1+2 = 3 → lands on the catcher
        state.fix_d6(1); // catcher fails the catch
        state.step_simple(SimpleAT::DontUseReroll); // decline the pickup reroll

        state.fix_d8(D8::from(up) as u8); // bounce on from the catcher's square
        state.step_simple(SimpleAT::DontUseReroll); // decline the catch reroll

        assert!(
            matches!(state.ball, BallState::OnGround(pos) if pos == catcher_pos + up),
            "throw-in catch failure must bounce from the catcher's square, got {:?}",
            state.ball
        );
    }

    /// A throw-in scales the 2D6 distance roll down on a narrow board, so it
    /// can't carry the ball back out of bounds as easily as an unscaled roll
    /// would — the dice rolled (D3 direction, 2D6 distance) are unchanged,
    /// only how far the distance carries the ball.
    #[test]
    fn throw_in_distance_is_scaled_down_on_a_narrow_board() {
        use crate::core::dices::{RollResult, Sum2D6, D3};
        use crate::core::gamestate::BuilderState;
        use crate::core::procedures::ball_procs::ThrowIn;
        use crate::core::procedures::AnyProc;

        let dims = BoardDims::new(16, 9, 3); // 14x7 playable
        assert_eq!(dims.scatter_divisor(), 2, "test assumes a divisor of 2");
        let mut state = GameStateBuilder::new()
            .with_board_dims(dims)
            .set_state(BuilderState::CoinToss)
            .build();

        // Corner origin: D3::One maps to straight along +x (see
        // `get_throw_in_direction`), so the landing square is easy to predict.
        let from = Position::new((1, 1));
        let AnyProc::ThrowIn(mut throw_in) = ThrowIn::new(from) else {
            unreachable!()
        };

        throw_in.step(
            &mut state,
            ProcInput::Roll(RollResult::ThrowIn {
                direction: D3::One,
                distance: Sum2D6::Twelve,
            }),
        );

        let BallState::InAir(pos) = state.ball else {
            panic!("ball should be airborne after landing in bounds, got {:?}", state.ball)
        };
        assert_eq!(
            pos,
            from + Direction::right() * 5,
            "(raw distance 12 - 1) / scatter_divisor 2 == 5, not the unscaled 11"
        );
    }

    /// S3, Throw-ins: 2D6 squares counting the square the ball left from as the first, so a 2
    /// lands on the next square in.
    #[test]
    fn a_throw_in_counts_its_origin_as_the_first_square() {
        use crate::core::dices::{RollResult, Sum2D6, D3};
        use crate::core::procedures::ball_procs::ThrowIn;
        use crate::core::procedures::AnyProc;

        // Unscaled distances: the full pitch only.
        crate::skip_if_board_smaller_than!(28, 17);
        let mut state = GameStateBuilder::new().build();
        assert_eq!(state.board_dims.scatter_divisor(), 1);
        let from = Position::new((1, 1));
        let AnyProc::ThrowIn(mut throw_in) = ThrowIn::new(from) else {
            unreachable!()
        };
        throw_in.step(
            &mut state,
            ProcInput::Roll(RollResult::ThrowIn {
                direction: D3::One,
                distance: Sum2D6::Two,
            }),
        );
        assert_eq!(state.ball, BallState::InAir(from + Direction::right()));
    }

    #[test]
    fn pickup_success() -> Result<()> {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();
        assert!(state.home_to_act());

        let id = state.get_player_id_at(start_pos).unwrap();

        assert_eq!(state.ball, BallState::OnGround(ball_pos));

        state.get_mut_player(id).unwrap().stats.give_skill(Skill::SureHands);

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));

        state.fix_d6(2); //fail first (3+)
        state.fix_d6(3); //succeed on reroll (3+)
        state.step_positional(PosAT::Move, Position::new((5, 5)));

        // S3: Sure Hands is not once per turn (unlike Dodge and Sure Feet).
        assert!(state.get_player(id).unwrap().can_use_skill(Skill::SureHands));

        match state.ball {
            BallState::Carried(id_carrier) if id_carrier == id => (),
            _ => panic!("wrong ball carried"),
        }

        Ok(())
    }

    #[test]
    fn pickup_sets_and_clears_activation_flag() -> Result<()> {
        // Drives the `GameInfo::pickup_this_activation` lifecycle that the
        // MCTS P8 pruning rule depends on: cleared at activation, set when the
        // ball is picked up, and cleared again the moment the next move action
        // is selected (consuming the one-move pickup bonus).
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();
        assert!(state.home_to_act());
        let id = state.get_player_id_at(start_pos).unwrap();

        state.step_positional(PosAT::StartMove, start_pos);
        assert!(
            !state.info.pickup_this_activation,
            "flag must be clear right after activation"
        );

        state.fix_d6(6); // succeed the pickup (3+ for AG3)
        state.step_positional(PosAT::Move, ball_pos);
        assert!(matches!(state.ball, BallState::Carried(c) if c == id));
        assert!(
            state.info.pickup_this_activation,
            "flag must be set once the ball is picked up"
        );

        // Selecting the follow-up move consumes the bonus.
        state.step_positional(PosAT::Move, Position::new((6, 5)));
        assert!(
            !state.info.pickup_this_activation,
            "flag must clear when the follow-up move action is selected"
        );
        Ok(())
    }

    #[test]
    fn crowd_surf_ball_carrier() {
        let carrier_pos = Position::new((5, 1));
        let blocker_pos = Position::new((5, 2));
        let mut state = GameStateBuilder::new()
            .add_home_player(blocker_pos)
            .add_away_player(carrier_pos)
            .add_ball_pos(carrier_pos)
            .build();

        state.step_positional(PosAT::StartBlock, blocker_pos);

        state.fix_blockdice(BlockDice::Pow);

        state.step_positional(PosAT::Block, carrier_pos);
        state.step_simple(SimpleAT::SelectPow);

        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.fix_d3(2); //throw in direction down
        state.fix_d6(1); //throw in length
        state.fix_d6(2); //throw in length: 2D6 = 3 lands 2 squares in (S3)
        state.fix_d8(2); //bounce direction down

        state.step_positional(PosAT::FollowUp, carrier_pos);

        assert_eq!(state.ball, BallState::OnGround(Position::new((5, 4))));

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
    fn knockdown_both_and_bounce_ball_around() {
        let carrier_pos = Position::new((5, 1));
        let blocker_pos = Position::new((5, 2));
        let bonus_downed_player_pos = Position::new((5, 3));

        let mut state = GameStateBuilder::new()
            .add_home_player(blocker_pos)
            .add_away_player(carrier_pos)
            .add_away_player(bonus_downed_player_pos)
            .add_ball_pos(carrier_pos)
            .build();

        state.get_mut_player_at_unsafe(bonus_downed_player_pos).status = PlayerStatus::Down;

        state.step_positional(PosAT::StartBlock, blocker_pos);
        state.fix_blockdice(BlockDice::BothDown);
        state.step_positional(PosAT::Block, carrier_pos);

        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor
        state.fix_d6(1); //armor

        // bouncing
        state.fix_d8_direction(Direction::down()); // to home player
        state.fix_d8_direction(Direction::up()); //back to away players
        state.fix_d8_direction(Direction::up()); // out of bounds

        // out of bounce throw-in
        state.fix_d3(2); // throw-in direction down
        state.fix_d6(1); // throw in length
        state.fix_d6(2); // throw in length: 2D6 = 3 lands 2 squares in (S3)
        state.fix_d8_direction(Direction::right()); // final bounce

        state.step_simple(SimpleAT::SelectBothDown);
        let expected_ball = BallState::OnGround(bonus_downed_player_pos + Direction::right());
        assert_eq!(state.ball, expected_ball);
    }

    #[test]
    fn handoff() {
        let start_pos = Position::new((2, 1));
        let target_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(target_pos)
            .add_ball_pos(start_pos)
            .build();
        let start_id = state.get_player_id_at(start_pos).unwrap();
        let carrier_id = state.get_player_id_at(target_pos).unwrap();

        state.step_positional(PosAT::StartHandoff, start_pos);

        state.fix_d6(6);
        state.step_positional(PosAT::Handoff, target_pos);

        // state.fix_d6(3);
        // state.step_simple(SimpleAT::UseReroll);

        assert!(state.get_player_unsafe(start_id).used);
        assert_eq!(state.ball, BallState::Carried(carrier_id));
    }
    #[test]
    fn can_only_handoff_when_carrying_the_ball() {
        let start_pos = Position::new((2, 1));
        let target_pos = Position::new((5, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_home_player(target_pos)
            .build();
        state.step_positional(PosAT::StartHandoff, start_pos);
        assert!(!state.is_legal_action(&Action::Positional(PosAT::Handoff, target_pos)));
    }
}

use serde::{Deserialize, Serialize};

use crate::core::dices::{D6Target, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6};
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{
    other_team, Action, AvailableActions, BoardDims, Coord, Direction, Position, ProcState, Procedure,
};
use crate::core::model::{BallState, PlayerID};
use crate::core::table::{PosAT, Skill};

use crate::core::procedures::any_proc::AnyProc;

use super::procedure_tools::{SimpleProc, SimpleProcContainer};
use super::TurnoverIfPossessionLost;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
                ProcState::DoneNew(Catch::new_with_kick_arg(
                    catcher_id,
                    game_state.get_catch_target(catcher_id).unwrap(),
                    self.kick,
                ))
            } else {
                //will run bounce again
                game_state.set_ball(BallState::InAir(new_pos));
                ProcState::NotDone
            }
        } else {
            game_state.set_ball(BallState::OnGround(new_pos));
            ProcState::Done
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThrowIn {
    from: Position,
}
impl ThrowIn {
    pub fn new(from: Position) -> AnyProc {
        AnyProc::ThrowIn(ThrowIn { from })
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
    /// considered): `from + direction * min(distance, max_scatter)`. Pure.
    /// Used by the MCTS scripted-outcome picker to choose a roll that keeps
    /// the ball in bounds — an out-of-bounds landing re-requests the roll,
    /// which under a deterministic scripted pick can loop forever on small
    /// boards.
    pub fn target_square(&self, direction: D3, distance: Sum2D6, dims: BoardDims) -> Position {
        let dir = self.get_throw_in_direction(direction, dims);
        let length = (distance as i8).min(dims.max_scatter());
        self.from + dir * length
    }
}
impl Procedure for ThrowIn {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let (direction, length) = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::ThrowIn);
            }
            ProcInput::Roll(RollResult::ThrowIn { direction, distance }) => {
                // Cap distance at half the board width so a throw-in can't fling
                // the ball clear across a narrow board (no-op on the full pitch).
                let dims = game_state.board_dims;
                (
                    self.get_throw_in_direction(direction, dims),
                    (distance as i8).min(dims.max_scatter()),
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
                    ProcState::DoneNew(Catch::new(catcher_id, game_state.get_catch_target(catcher_id).unwrap()))
                }
                _ => {
                    game_state.set_ball(BallState::InAir(target));
                    ProcState::DoneNew(Bounce::new())
                }
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Touchback {}
impl Touchback {
    pub fn new() -> AnyProc {
        AnyProc::Touchback(Touchback {})
    }
}
impl Procedure for Touchback {
    fn step(&mut self, game_state: &mut GameState, action: ProcInput) -> ProcState {
        if let ProcInput::Action(Action::Positional(_, position)) = action {
            game_state.set_ball(BallState::Carried(game_state.get_player_id_at(position).unwrap()));
            ProcState::Done
        } else {
            let team = other_team(game_state.info.kicking_this_drive);
            let positions: Vec<_> = game_state
                .get_players_on_pitch_in_team(team)
                .map(|p| p.position)
                .collect();
            if positions.is_empty() {
                // Nobody can take the touchback (all receivers off the
                // pitch — routine in 2-player small-board games). An empty
                // `NeedAction` would deadlock every consumer, so drop the
                // ball in the middle of the receiving half and let it
                // bounce from there.
                // `get_best_kickoff_aim_for` takes the *kicking* team; `team` here is
                // the receiver, so passing it aimed at the wrong half (plan 023 B1).
                let aim = game_state.get_best_kickoff_aim_for(game_state.info.kicking_this_drive);
                game_state.set_ball(BallState::InAir(aim));
                return ProcState::DoneNew(Bounce::new());
            }
            let mut aa = AvailableActions::new(team);
            aa.insert_positional(PosAT::SelectPosition, positions);
            ProcState::NeedAction(aa)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
                game_state.info.kickoff_by_team = Some(game_state.get_player_unsafe(self.id).stats.team);
            }
        }

        ProcState::Done
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PassResult {
    Accurate,
    Inaccurate,
    WildlyInaccurate,
    Fumble,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Pass {
    pos: Position,
    pass: D6Target,
    modifier: i8,
}
impl Pass {
    pub fn new(pos: Position, pass: D6Target, modifier: i8) -> AnyProc {
        AnyProc::Pass(Pass { pos, pass, modifier })
    }
}
impl Procedure for Pass {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing => ProcState::NeedRoll(RequestedRoll::D6),
            ProcInput::Roll(RollResult::D6(roll)) if self.pass.is_success(roll) => {
                // ACCURATE PASS
                let from = game_state.get_ball_position().unwrap();
                ProcState::DoneNewProcs(vec![
                    TurnoverIfPossessionLost::new(),
                    DeflectOrResolve::new(from, self.pos, PassResult::Accurate, None),
                ])
            }
            ProcInput::Roll(RollResult::D6(D6::One)) => {
                // FUMBLE
                game_state.info.turnover = true;
                ProcState::DoneNew(Bounce::new())
            }
            ProcInput::Roll(RollResult::D6(roll)) if roll + self.modifier == D6::One => {
                // WILDLY INACCURATE PASSES
                //  deviate (d8 * d6) from the square occupied by the player performing the Pass
                ProcState::NeedRoll(RequestedRoll::Deviate)
            }
            ProcInput::Roll(RollResult::D6(_)) => {
                //INACCURATE PASSES
                // scatter (d8 + d8 + d8) from the target square before landing.
                ProcState::NeedRoll(RequestedRoll::Scatter)
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
            ProcInput::Roll(RollResult::Deviate(distance, direction)) => {
                let from = game_state.get_ball_position().unwrap();
                let mut target = from; // + Direction::from(direction) * distance as i8;
                let mut throwin_pos = None;
                let dir = Direction::from(direction);
                for _ in 0..(distance as i8) {
                    let new_target = target + dir;
                    if game_state.is_out(new_target) {
                        throwin_pos = Some(target);
                        break;
                    }
                    target = new_target;
                }

                ProcState::DoneNewProcs(vec![
                    TurnoverIfPossessionLost::new(),
                    DeflectOrResolve::new(from, target, PassResult::WildlyInaccurate, throwin_pos),
                ])
            }
            ProcInput::Action(_) => todo!(),
            _ => panic!("Unexpected input {:?} for Pass", input),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
                    println!("no intercepters");
                    None
                } else if self.intercepters.len() == 1 {
                    println!("only one intercepter");
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
                match game_state.get_player_at(self.to) {
                    Some(player) => {
                        let mut target = game_state.get_catch_target(player.id).unwrap();
                        target.add_modifer(match self.result {
                            PassResult::Accurate => 0,
                            PassResult::Inaccurate => -1,
                            PassResult::WildlyInaccurate => -2,
                            PassResult::Fumble => -3,
                        });
                        Catch::new(player.id, target)
                    }
                    None => Bounce::new(),
                }
            }
        };
        if let Some((pos, mut target)) = interceptor {
            target.add_modifer(match self.result {
                PassResult::Accurate => 0,
                PassResult::Inaccurate => -1,
                PassResult::WildlyInaccurate => -2,
                PassResult::Fumble => -3,
            });
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.set_ball(BallState::InAir(game_state.get_player_unsafe(self.id).position));
        let mut catch_target = game_state.get_catch_target(self.id).unwrap();
        vec![Catch::new(self.id, *catch_target.add_modifer(-1))]
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

        let AnyProc::Bounce(mut proc) = Bounce::new() else { unreachable!() };
        let result = proc.step(
            &mut state,
            ProcInput::Roll(RollResult::D8(D8::from(Direction::down()))),
        );
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

    /// A touchback with no receiving player on the pitch (all injured/KO'd
    /// — routine in 2-player small-board games) must not emit an empty
    /// `NeedAction`: an action request with zero legal actions deadlocks
    /// every consumer (bots, MCTS marks the node terminal). The ball is
    /// dropped in the receiving half and bounces instead.
    #[test]
    fn touchback_with_no_receivers_drops_ball_in_receiving_half() {
        use crate::core::procedures::ball_procs::{Bounce, Touchback};
        use crate::core::procedures::AnyProc;

        // Home kicks, Away receives — and Away has nobody on the pitch.
        let mut state = GameStateBuilder::new().add_home_player(Position::new((2, 2))).build();
        state.info.kicking_this_drive = TeamType::Home;

        let mut proc = Touchback {};
        let result = proc.step(&mut state, ProcInput::Nothing);

        // Assert the *property* the test is named for. This previously computed
        // its expectation as `get_best_kickoff_aim_for(TeamType::Away)` — the
        // receiving team — which is the same argument mix-up the production code
        // had (the function takes the *kicking* team), so it asserted the ball
        // landed in the KICKING team's half while claiming to check the
        // receiving one. See plan 023 B1.
        let expected = state.get_best_kickoff_aim_for(TeamType::Home); // Home kicks
        assert!(
            matches!(state.ball, BallState::InAir(pos) if pos == expected),
            "ball must be dropped at the receiving half's aim square ({expected:?}), got {:?}",
            state.ball
        );
        assert!(
            matches!(state.ball, BallState::InAir(pos) if state.board_dims.is_on_team_side(pos, TeamType::Away)),
            "Away receives, so the ball must land in Away's half, got {:?}",
            state.ball
        );
        assert!(
            matches!(result, ProcState::DoneNew(AnyProc::Bounce(Bounce { .. }))),
            "touchback must resolve into a bounce, got {:?}",
            result
        );
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

    /// Same rule for a throw-in that lands on a player: the ball is in
    /// their square for the catch attempt, and a failed catch bounces
    /// from there.
    #[test]
    fn throw_in_onto_catcher_bounces_from_catcher_square() {
        let ball_pos = Position::new((5, 1)); // on the top edge
        let start_pos = Position::new((5, 4));
        let up = Direction::up();
        // ThrowIn from y==1 with D3::One goes direction (1,1); distance is
        // capped 2d6 — fix 1+1 = 2 → lands at ball_pos + (2,2).
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
        state.fix_d6(1); // ...1+1 = 2 → lands on the catcher
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

        assert!(!state.get_player(id).unwrap().can_use_skill(Skill::SureHands));

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
        state.fix_d6(1); //throw in length
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

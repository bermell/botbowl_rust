use serde::{Deserialize, Serialize};

use crate::core::dices::{D6Target, RequestedRoll, RollResult, RollTarget, D3, D6};
use crate::core::gamestate::GameState;
use crate::core::model::ProcInput;
use crate::core::model::{
    other_team, Action, AvailableActions, Coord, Direction, Position, ProcState, Procedure,
    HEIGHT_, WIDTH_,
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
        game_state.ball = BallState::Carried(self.id);
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
            && (new_pos.is_out() || new_pos.is_on_team_side(game_state.info.kicking_this_drive))
        {
            return ProcState::DoneNew(Touchback::new());
        }

        if let Some(player) = game_state.get_player_at(new_pos) {
            if player.can_catch() {
                ProcState::DoneNew(Catch::new_with_kick_arg(
                    player.id,
                    game_state.get_catch_target(player.id).unwrap(),
                    self.kick,
                ))
            } else {
                //will run bounce again
                game_state.ball = BallState::InAir(new_pos);
                ProcState::NotDone
            }
        } else if new_pos.is_out() {
            ProcState::DoneNew(ThrowIn::new(current_ball_pos))
        } else {
            game_state.ball = BallState::OnGround(new_pos);
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
    fn get_throw_in_direction(&self, dice: D3) -> Direction {
        const MAX_X: Coord = WIDTH_ - 2;
        const MAX_Y: Coord = HEIGHT_ - 2;
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
        Direction::from(match dice {
            D3::One => directions[0],
            D3::Two => directions[1],
            D3::Three => directions[2],
        })
    }
}
impl Procedure for ThrowIn {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let (direction, length) = match input {
            ProcInput::Nothing => {
                return ProcState::NeedRoll(RequestedRoll::ThrowIn);
            }
            ProcInput::Roll(RollResult::ThrowIn {
                direction,
                distance,
            }) => (self.get_throw_in_direction(direction), distance as i8),
            _ => panic!("Unexpected input {:?} for ThrowIn", input),
        };
        let target: Position = self.from + direction * length;

        if target.is_out() {
            self.from = target - direction;

            while self.from.is_out() {
                self.from -= direction;
            }

            ProcState::NeedRoll(RequestedRoll::ThrowIn)
        } else {
            match game_state.get_player_at(target) {
                Some(player) if player.can_catch() => ProcState::DoneNew(Catch::new(
                    player.id,
                    game_state.get_catch_target(player.id).unwrap(),
                )),
                _ => {
                    game_state.ball = BallState::InAir(target);
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

    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        game_state.ball = BallState::Carried(self.id);
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
            game_state.ball = BallState::Carried(game_state.get_player_id_at(position).unwrap());
            ProcState::Done
        } else {
            let team = other_team(game_state.info.kicking_this_drive);
            let mut aa = AvailableActions::new(team);
            let positions: Vec<_> = game_state
                .get_players_on_pitch_in_team(team)
                .map(|p| p.position)
                .collect();
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
                game_state.info.kickoff_by_team =
                    Some(other_team(game_state.get_player_unsafe(self.id).stats.team));
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
        AnyProc::Pass(Pass {
            pos,
            pass,
            modifier,
        })
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
                    if new_target.is_out() {
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
                    if new_target.is_out() {
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
    pub fn new(
        from: Position,
        to: Position,
        result: PassResult,
        throw_in_pos: Option<Position>,
    ) -> AnyProc {
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
        let deflect_team = other_team(game_state.get_active_teamtype().unwrap());
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
                debug_assert!(!throw_in_pos.is_out());
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
            game_state.ball = BallState::InAir(self.to);
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
        game_state.ball = BallState::InAir(game_state.get_player_unsafe(self.id).position);
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

    #[test]
    fn pickup_fail_and_bounce() -> Result<()> {
        let ball_pos = Position::new((5, 5));
        let start_pos = Position::new((1, 1));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();

        let id = state.get_player_id_at(start_pos).unwrap();

        let d8_fix = D8::One;
        let direction = Direction::from(d8_fix);

        state.step_positional(PosAT::StartMove, start_pos);
        state.fixes.fix_d6(2); //fail pickup (3+)
        state.step_positional(PosAT::Move, ball_pos);
        state.fixes.fix_d8(d8_fix as u8);
        state.step_simple(SimpleAT::DontUseReroll);

        let player = state.get_player(id).unwrap();
        assert!(player.used);
        assert!(matches!(state.ball, BallState::OnGround(pos) if pos == ball_pos + direction));

        Ok(())
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

        state
            .get_mut_player(id)
            .unwrap()
            .stats
            .give_skill(Skill::SureHands);

        state.step_positional(PosAT::StartMove, Position::new((1, 1)));

        state.fixes.fix_d6(2); //fail first (3+)
        state.fixes.fix_d6(3); //succeed on reroll (3+)
        state.step_positional(PosAT::Move, Position::new((5, 5)));

        assert!(!state
            .get_player(id)
            .unwrap()
            .can_use_skill(Skill::SureHands));

        match state.ball {
            BallState::Carried(id_carrier) if id_carrier == id => (),
            _ => panic!("wrong ball carried"),
        }

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

        state.fixes.fix_blockdice(BlockDice::Pow);

        state.step_positional(PosAT::Block, carrier_pos);
        state.step_simple(SimpleAT::SelectPow);

        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d6(1); //armor
        state.fixes.fix_d3(2); //throw in direction down
        state.fixes.fix_d6(1); //throw in length
        state.fixes.fix_d6(1); //throw in length
        state.fixes.fix_d8(2); //bounce direction down

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

        state.fixes.fix_d6(6);
        state.step_positional(PosAT::Handoff, target_pos);

        // state.fixes.fix_d6(3);
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

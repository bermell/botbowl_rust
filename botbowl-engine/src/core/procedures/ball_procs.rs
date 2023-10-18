use crate::core::dices::{D6Target, D6};
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, Coord, Direction, Position, ProcState, Procedure,
    HEIGHT_, WIDTH_,
};
use crate::core::model::{BallState, PlayerID};
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::table::{PosAT, Skill};

#[derive(Debug)]
pub struct PickupProc {
    target: D6Target,
    id: PlayerID,
}
impl PickupProc {
    pub fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<PickupProc>> {
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
        game_state.info.turnover = true;
        vec![Bounce::new()]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Debug)]
pub struct Bounce {
    kick: bool,
}
impl Bounce {
    pub fn new() -> Box<Bounce> {
        Box::new(Bounce { kick: false })
    }
    pub fn new_with_kick_arg(kick: bool) -> Box<Bounce> {
        Box::new(Bounce { kick })
    }
}
impl Procedure for Bounce {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
        let current_ball_pos = game_state.get_ball_position().unwrap();
        let dice = game_state.get_d8_roll();
        let dir = Direction::from(dice);
        let new_pos = current_ball_pos + dir;

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
#[derive(Debug)]
pub struct ThrowIn {
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
#[derive(Debug)]
pub struct Catch {
    id: PlayerID,
    target: D6Target,
    kick: bool,
}
impl Catch {
    pub fn new(id: PlayerID, target: D6Target) -> Box<SimpleProcContainer<Catch>> {
        SimpleProcContainer::new(Catch {
            id,
            target,
            kick: false,
        })
    }
    pub fn new_with_kick_arg(
        id: PlayerID,
        target: D6Target,
        kick: bool,
    ) -> Box<SimpleProcContainer<Catch>> {
        SimpleProcContainer::new(Catch { id, target, kick })
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
        vec![Bounce::new_with_kick_arg(self.kick)]
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}
#[derive(Debug)]
pub struct Touchback {}
impl Touchback {
    pub fn new() -> Box<Touchback> {
        Box::new(Touchback {})
    }
}
impl Procedure for Touchback {
    fn step(&mut self, game_state: &mut GameState, action: Option<Action>) -> ProcState {
        if let Some(Action::Positional(_, position)) = action {
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

#[derive(Debug)]
pub struct Touchdown {
    id: PlayerID,
}
impl Touchdown {
    pub fn new(id: PlayerID) -> Box<Touchdown> {
        Box::new(Touchdown { id })
    }
}
impl Procedure for Touchdown {
    fn step(&mut self, game_state: &mut GameState, _action: Option<Action>) -> ProcState {
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

#[cfg(test)]
mod tests {

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
}

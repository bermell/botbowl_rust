//! S3 Throw Team-mate and Kick Team-mate.
//!
//! Every decision is the **hidden** `PosAT::ThrowTeamMate` (legal, never listed, so bots, the
//! search and the NN never see it; a human UI offers it):
//!
//! 1. declaring the action on the thrower (`Turn`, `Charge`) — a Throw Team-mate if the player
//!    has the trait and the team has not declared one this turn, else a Kick Team-mate;
//! 2. after the free Move Action, picking up an adjacent Right Stuff team-mate on their square
//!    (`MoveAction`), which pushes [`ThrowTeamMate`];
//! 3. the target square, in Quick or Short range ([`ThrowTeamMate`]).
//!
//! Then the Passing Ability Test (Superb / Subpar / Fumbled), the Scatter (3) or the fumble's
//! Bounce, and the landing ([`ThrownPlayerFlight`], [`TeamMateLanding`]).

use serde::{Deserialize, Serialize};

use crate::core::dices::{D6Target, RequestedRoll, RollResult, RollTarget, D6};
use crate::core::gamestate::GameState;
use crate::core::model::{
    Action, AvailableActions, BallState, Direction, FieldedPlayer, PlayerID, PlayerStatus, Position, ProcInput,
    ProcState, Procedure, Weather,
};
use crate::core::procedures::procedure_tools::{SimpleProc, SimpleProcContainer};
use crate::core::procedures::{ball_procs, casualty_procs};
use crate::core::table::{PosAT, SimpleAT, Skill};

use super::AnyProc;

/// The action `player` may declare with `PosAT::ThrowTeamMate` now: `Some(false)` a Throw
/// Team-mate (the trait, and the team's one per turn is unused), `Some(true)` a Kick Team-mate
/// (likewise, and it "does not count as a team's Throw Team-mate Action"), `None` neither. A
/// player with both traits declares the Throw while it is available.
pub fn team_mate_declaration(game_state: &GameState, player: &FieldedPlayer) -> Option<bool> {
    let info = &game_state.info;
    if player.has_skill(Skill::ThrowTeamMate) && info.throw_team_mate_available {
        Some(false)
    } else if player.has_skill(Skill::KickTeamMate) && info.kick_team_mate_available {
        Some(true)
    } else {
        None
    }
}

/// Offer the declaration (hidden) on every square in `positions` whose player may declare.
pub fn offer_declarations(game_state: &GameState, positions: &[Position], aa: &mut AvailableActions) {
    for &pos in positions {
        let player = game_state.get_player_at(pos).unwrap();
        if team_mate_declaration(game_state, player).is_some() {
            aa.insert_hidden_positional(PosAT::ThrowTeamMate, pos);
        }
    }
}

/// `id` declares a Throw / Kick Team-mate: sets the activation's action type and spends the
/// team's one per turn. The caller has made `id` the active player and pushes the `MoveAction`.
pub fn declare(game_state: &mut GameState, id: PlayerID) {
    let kick = team_mate_declaration(game_state, game_state.get_player_unsafe(id))
        .expect("declared Throw / Kick Team-mate without being able to");
    let info = &mut game_state.info;
    info.player_action_type = Some(PosAT::ThrowTeamMate);
    info.kick_team_mate_this_activation = kick;
    if kick {
        info.kick_team_mate_available = false;
    } else {
        info.throw_team_mate_available = false;
    }
}

/// The squares of the team-mates `thrower` may pick up: adjacent, with Right Stuff, Standing or
/// Prone ("can be thrown ... even if this player is Prone"; a Stunned one cannot). The thrower
/// must be Standing.
pub fn throwable_team_mates(game_state: &GameState, thrower: &FieldedPlayer) -> Vec<Position> {
    if thrower.status != PlayerStatus::Up {
        return Vec::new();
    }
    game_state
        .get_adj_players(thrower.position)
        .filter(|p| p.stats.team == thrower.stats.team && p.has_skill(Skill::RightStuff))
        .filter(|p| p.status != PlayerStatus::Stunned)
        .map(|p| p.position)
        .collect()
}

/// S3, Declare Target Square: "wholly underneath the first two sections of the Range Ruler" —
/// a Quick or a Short Throw. Any square on the pitch (occupied or not) but the thrower's and
/// the thrown player's own.
pub fn target_squares(game_state: &GameState, from: Position, thrown_from: Position) -> Vec<Position> {
    let dims = game_state.board_dims;
    Position::all_positions()
        .filter(|&to| !dims.is_out(to) && to != from && to != thrown_from)
        .filter(|&to| matches!(GameState::pass_range(from, to), Some(0 | 1)))
        .collect()
}

/// The three results of the Passing Ability Test of a Throw Team-mate.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThrowQuality {
    Superb,
    Subpar,
    Fumbled,
}

/// The throw itself, once the thrower has picked up `thrown`: the target square (a hidden
/// `PosAT::ThrowTeamMate`), the Passing Ability Test (-1 Short, -1 per opponent Marking the
/// thrower, -1 Very Sunny; a team re-roll may re-roll a failed one), then the flight.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ThrowTeamMate {
    thrower: PlayerID,
    thrown: PlayerID,
    kick: bool,
    target: Option<Position>,
    modifier: i8,
    rerolled: bool,
    /// A failed result waiting on the team re-roll decision.
    failed: Option<ThrowQuality>,
    /// The result that stands, while the Scatter is rolled.
    quality: Option<ThrowQuality>,
}
impl ThrowTeamMate {
    pub fn new(thrower: PlayerID, thrown: PlayerID, kick: bool) -> AnyProc {
        AnyProc::ThrowTeamMate(ThrowTeamMate {
            thrower,
            thrown,
            kick,
            target: None,
            modifier: 0,
            rerolled: false,
            failed: None,
            quality: None,
        })
    }

    /// The declared target square, once declared.
    pub fn target(&self) -> Option<Position> {
        self.target
    }

    fn modifier(game_state: &GameState, thrower: &FieldedPlayer, to: Position) -> i8 {
        let range = GameState::pass_range(thrower.position, to).expect("target out of range");
        let mut modifier = -range - game_state.marking_opponents(thrower) as i8;
        // Very Sunny: "Whenever a player makes a Passing Ability Test, apply a -1 modifier".
        if game_state.info.weather == Weather::Sunny {
            modifier -= 1;
        }
        modifier
    }

    /// S3, Test for Accuracy: a natural 1 or a 1 after modifiers is Fumbled; a pass of the test
    /// or a natural 6 is Superb; anything else Subpar.
    fn quality_of(&self, game_state: &GameState, roll: D6) -> ThrowQuality {
        let thrower = game_state.get_player_unsafe(self.thrower);
        if roll == D6::One || (roll as i8) + self.modifier <= 1 {
            ThrowQuality::Fumbled
        } else if roll == D6::Six || thrower.pa_test(self.modifier).is_success(roll) {
            ThrowQuality::Superb
        } else {
            ThrowQuality::Subpar
        }
    }

    fn resolve(&mut self, quality: ThrowQuality, game_state: &mut GameState) -> ProcState {
        self.quality = Some(quality);
        let thrown_prone = game_state.get_player_unsafe(self.thrown).status != PlayerStatus::Up;
        match quality {
            ThrowQuality::Fumbled if self.kick => kicked_team_mate_fumbled(game_state, self.thrown),
            // "The thrown player is dropped and will Bounce from the throwing player's square and
            // must attempt to land."
            ThrowQuality::Fumbled => {
                let from = game_state.get_player_unsafe(self.thrower).position;
                ProcState::DoneNew(ThrownPlayerFlight::new(self.thrown, from, quality, thrown_prone))
            }
            ThrowQuality::Superb | ThrowQuality::Subpar => ProcState::NeedRoll(RequestedRoll::Scatter),
        }
    }
}
impl Procedure for ThrowTeamMate {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        let team = game_state.get_player_unsafe(self.thrower).stats.team;
        match input {
            ProcInput::Nothing if self.target.is_none() => {
                let from = game_state.get_player_unsafe(self.thrower).position;
                let thrown_from = game_state.get_player_unsafe(self.thrown).position;
                let mut aa = AvailableActions::new(team);
                for square in target_squares(game_state, from, thrown_from) {
                    aa.insert_hidden_positional(PosAT::ThrowTeamMate, square);
                }
                ProcState::NeedAction(aa)
            }
            ProcInput::Action(Action::Positional(PosAT::ThrowTeamMate, to)) if self.target.is_none() => {
                self.target = Some(to);
                let thrower = game_state.get_player_unsafe(self.thrower);
                self.modifier = Self::modifier(game_state, thrower, to);
                // The thrower may not move on after the throw.
                game_state.get_mut_player_unsafe(self.thrower).used = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Roll(RollResult::D6(roll)) => {
                let quality = self.quality_of(game_state, roll);
                if quality != ThrowQuality::Superb && !self.rerolled && game_state.may_use_team_reroll(team) {
                    self.failed = Some(quality);
                    let mut aa = AvailableActions::new(team);
                    aa.insert_simple(SimpleAT::UseReroll);
                    aa.insert_simple(SimpleAT::DontUseReroll);
                    return ProcState::NeedAction(aa);
                }
                self.resolve(quality, game_state)
            }
            ProcInput::Action(Action::Simple(SimpleAT::UseReroll)) if self.failed.is_some() => {
                game_state.get_mut_team(team).use_reroll();
                self.failed = None;
                self.rerolled = true;
                ProcState::NeedRoll(RequestedRoll::D6)
            }
            ProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)) if self.failed.is_some() => {
                let quality = self.failed.take().unwrap();
                self.resolve(quality, game_state)
            }
            // Superb and Subpar both Scatter (3) from the target square ("Should a thrown player
            // leave the pitch as a result of a throw, they will risk Injury by the Crowd").
            ProcInput::Roll(RollResult::Scatter(r1, r2, r3)) => {
                let quality = self.quality.unwrap();
                let thrown_prone = game_state.get_player_unsafe(self.thrown).status != PlayerStatus::Up;
                let mut square = self.target.unwrap();
                for d in [r1, r2, r3].map(Direction::from) {
                    let next = square + d;
                    if game_state.is_out(next) {
                        return ProcState::from(thrown_into_crowd(game_state, self.thrown, square, Vec::new()));
                    }
                    square = next;
                }
                ProcState::DoneNew(ThrownPlayerFlight::landing_at(
                    self.thrown,
                    square,
                    quality,
                    thrown_prone,
                ))
            }
            _ => panic!("Unexpected input {:?} for ThrowTeamMate", input),
        }
    }
}
/// Kick Team-mate's Fumbled Throw: an Injury Roll for the kicked team-mate, Stunned counting as
/// Knocked Out; a ball they held Bounces from their square (they have left the pitch with it:
/// a turnover).
fn kicked_team_mate_fumbled(game_state: &mut GameState, id: PlayerID) -> ProcState {
    let mut procs = Vec::new();
    if game_state.ball == BallState::Carried(id) {
        let pos = game_state.get_player_unsafe(id).position;
        game_state.set_ball(BallState::InAir(pos));
        game_state.info.turnover = true;
        procs.push(ball_procs::Bounce::new());
    }
    procs.push(casualty_procs::Injury::new_stunned_is_ko(id));
    ProcState::from(procs)
}

/// The thrown player leaves the pitch from `last`: "they will risk Injury by the Crowd and a
/// Turnover will be caused. If that player was also holding the ball, then a Throw-in will be
/// required" (from `last`, the last square on the pitch). `knocked` are players crashed into on
/// the way, Knocked Down first.
fn thrown_into_crowd(game_state: &mut GameState, id: PlayerID, last: Position, knocked: Vec<PlayerID>) -> Vec<AnyProc> {
    game_state.info.turnover = true;
    let mut procs = Vec::new();
    if game_state.ball == BallState::Carried(id) {
        game_state.set_ball(BallState::InAir(last));
        procs.push(ball_procs::ThrowIn::new(last));
    }
    procs.push(casualty_procs::Injury::new_crowd(id));
    procs.extend(crash_knock_downs(game_state, &knocked));
    procs
}

/// "The player in the occupied square is automatically Knocked Down even if they are already
/// Prone or Stunned": each falls (a Standing one goes Prone, a Stunned one stays Stunned), makes
/// an Armour Roll, and drops a ball they held. Knocking down a player of the active team is a
/// turnover. Returned in stack order (the last runs first), so the first knocked resolves first.
fn crash_knock_downs(game_state: &mut GameState, knocked: &[PlayerID]) -> Vec<AnyProc> {
    let mut procs = Vec::new();
    for &id in knocked.iter().rev() {
        let player = game_state.get_mut_player_unsafe(id);
        if player.status == PlayerStatus::Up {
            player.status = PlayerStatus::Down;
        }
        player.used = true;
        let (team, pos) = (player.stats.team, player.position);
        if game_state.turn_team() == Some(team) {
            game_state.info.turnover = true;
        }
        if game_state.ball == BallState::Carried(id) {
            game_state.set_ball(BallState::InAir(pos));
            procs.push(ball_procs::Bounce::new());
        }
        procs.push(casualty_procs::Armor::new(id));
    }
    procs
}

/// The thrown player Falls Over where they are: Prone, an Armour Roll, and a ball they held
/// Bounces — "this will only cause a Turnover if the thrown player was holding the ball". A
/// loose ball in the square Bounces too (it never rests under a player). In stack order.
fn thrown_player_falls(game_state: &mut GameState, id: PlayerID) -> Vec<AnyProc> {
    let player = game_state.get_mut_player_unsafe(id);
    player.status = PlayerStatus::Down;
    let pos = player.position;
    let mut procs = Vec::new();
    match game_state.ball {
        BallState::Carried(carrier) if carrier == id => {
            game_state.set_ball(BallState::InAir(pos));
            game_state.info.turnover = true;
            procs.push(ball_procs::Bounce::new());
        }
        BallState::OnGround(ball) if ball == pos => procs.push(ball_procs::Bounce::new()),
        _ => (),
    }
    procs.push(casualty_procs::Armor::new(id));
    procs
}

/// Where the thrown player comes down. It starts either at the square the throw scattered to
/// (`landing_at`) or with a Bounce from a square (`new`, the fumble from the thrower's). Each
/// occupied square they come down in is a crash landing: its player is Knocked Down and the
/// thrown player Bounces on from it; out of the pitch is the crowd. In an unoccupied square they
/// try to land — after a crash they simply Fall Over.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ThrownPlayerFlight {
    id: PlayerID,
    /// The square the next Bounce starts from, or (`bounce` false) the square to come down in.
    at: Position,
    bounce: bool,
    quality: ThrowQuality,
    /// Prone (or otherwise not Standing) when thrown: the landing fails automatically.
    prone: bool,
    /// Crashed into, in order. Knocked Down once the thrown player is down.
    knocked: Vec<PlayerID>,
}
impl ThrownPlayerFlight {
    /// Bounce from `from` first (the Fumbled Throw).
    pub fn new(id: PlayerID, from: Position, quality: ThrowQuality, prone: bool) -> AnyProc {
        AnyProc::ThrownPlayerFlight(ThrownPlayerFlight {
            id,
            at: from,
            bounce: true,
            quality,
            prone,
            knocked: Vec::new(),
        })
    }
    /// Come down in `square`.
    pub fn landing_at(id: PlayerID, square: Position, quality: ThrowQuality, prone: bool) -> AnyProc {
        AnyProc::ThrownPlayerFlight(ThrownPlayerFlight {
            id,
            at: square,
            bounce: false,
            quality,
            prone,
            knocked: Vec::new(),
        })
    }

    fn come_down(&mut self, game_state: &mut GameState, square: Position, last: Position) -> ProcState {
        if game_state.is_out(square) {
            let knocked = std::mem::take(&mut self.knocked);
            return ProcState::from(thrown_into_crowd(game_state, self.id, last, knocked));
        }
        match game_state.get_player_id_at(square) {
            // The thrown player's own square is empty: they were picked up from it.
            Some(occupant) if occupant != self.id => {
                if !self.knocked.contains(&occupant) {
                    self.knocked.push(occupant);
                }
                self.at = square;
                self.bounce = true;
                ProcState::NeedRoll(RequestedRoll::D8)
            }
            _ => {
                if game_state.get_player_unsafe(self.id).position != square {
                    game_state.move_player(self.id, square).unwrap();
                }
                let knocked = std::mem::take(&mut self.knocked);
                let crashed = !knocked.is_empty();
                // Crashed players first, then the thrown player: stack order puts them last.
                let mut procs = if crashed || self.prone {
                    // "... will Bounce from the occupied square and will Fall Over"; "Players that
                    // were Prone ... when they were thrown will automatically fail the Agility
                    // Test to land."
                    thrown_player_falls(game_state, self.id)
                } else {
                    vec![TeamMateLanding::new(game_state, self.id, self.quality)]
                };
                procs.extend(crash_knock_downs(game_state, &knocked));
                ProcState::from(procs)
            }
        }
    }
}
impl Procedure for ThrownPlayerFlight {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState {
        match input {
            ProcInput::Nothing if self.bounce => ProcState::NeedRoll(RequestedRoll::D8),
            ProcInput::Nothing => {
                let square = self.at;
                self.come_down(game_state, square, square)
            }
            ProcInput::Roll(RollResult::D8(d8)) => {
                let from = self.at;
                self.come_down(game_state, from + Direction::from(d8), from)
            }
            _ => panic!("Unexpected input {:?} for ThrownPlayerFlight", input),
        }
    }
}

/// S3, Landing: an Agility Test, -1 after a Subpar Throw, -1 after a Fumbled Throw, -1 per
/// opponent Marking the square. Passed: the player lands on their feet (and, holding the ball
/// in the End Zone, scores); they keep their activation if they had not used it. Failed: they
/// Fall Over. A team re-roll may re-roll it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TeamMateLanding {
    id: PlayerID,
    target: D6Target,
}
impl TeamMateLanding {
    pub fn new(game_state: &GameState, id: PlayerID, quality: ThrowQuality) -> AnyProc {
        let player = game_state.get_player_unsafe(id);
        let mut modifier = -(game_state.marking_opponents(player) as i8);
        if quality != ThrowQuality::Superb {
            modifier -= 1;
        }
        AnyProc::TeamMateLanding(SimpleProcContainer::new(TeamMateLanding {
            id,
            target: player.ag_test(modifier),
        }))
    }
}
impl SimpleProc for TeamMateLanding {
    fn d6_target(&self) -> D6Target {
        self.target
    }

    fn reroll_skill(&self) -> Option<Skill> {
        None
    }

    fn apply_success(&self, game_state: &mut GameState) -> Vec<AnyProc> {
        let player = game_state.get_player_unsafe(self.id);
        let pos = player.position;
        match game_state.ball {
            BallState::Carried(carrier) if carrier == self.id => {
                if pos.x == game_state.get_endzone_x(player.stats.team) {
                    game_state.info.handle_td_by = Some(self.id);
                }
                Vec::new()
            }
            // Landing on a loose ball: it may not rest under a player.
            BallState::OnGround(ball) if ball == pos => vec![ball_procs::Bounce::new()],
            _ => Vec::new(),
        }
    }

    fn apply_failure(&mut self, game_state: &mut GameState) -> Vec<AnyProc> {
        thrown_player_falls(game_state, self.id)
    }

    fn player_id(&self) -> PlayerID {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gamestate::GameStateBuilder;
    use crate::core::model::{DugoutPlace, TeamType};
    use crate::core::procedures::kickoff_events::Charge;

    const THROWER: Position = Position { x: 8, y: 5 };
    const MATE: Position = Position { x: 9, y: 5 };

    fn give(state: &mut GameState, pos: Position, skill: Skill) {
        state.get_mut_player_at_unsafe(pos).stats.skills.insert(skill);
    }

    /// Home's turn: a Throw Team-mate thrower at `THROWER`, a Right Stuff team-mate next to them
    /// at `MATE`, `more` home players and `away` opponents. No team re-rolls (so a failed test
    /// is not a re-roll question), the turn decision re-asked with the new skills.
    fn throw_state(more: &[(i8, i8)], away: &[(i8, i8)]) -> GameState {
        let mut state = GameStateBuilder::new()
            .add_home_player(THROWER)
            .add_home_player(MATE)
            .add_home_players(more)
            .add_away_players(away)
            .build();
        give(&mut state, THROWER, Skill::ThrowTeamMate);
        give(&mut state, MATE, Skill::RightStuff);
        state.get_mut_team(TeamType::Home).rerolls = 0;
        state.reask_turn_decision().unwrap();
        state
    }

    fn ttm(pos: Position) -> Action {
        Action::Positional(PosAT::ThrowTeamMate, pos)
    }

    fn listed(state: &GameState) -> bool {
        state
            .get_all_actions()
            .iter()
            .any(|a| matches!(a, Action::Positional(PosAT::ThrowTeamMate, _)))
    }

    /// Declare at `THROWER` and pick up the team-mate at `MATE`: the target is asked next.
    fn pick_up(state: &mut GameState) {
        state.step_positional(PosAT::ThrowTeamMate, THROWER);
        state.step_positional(PosAT::ThrowTeamMate, MATE);
    }

    fn id_at(state: &GameState, pos: Position) -> PlayerID {
        state.get_player_id_at(pos).unwrap()
    }

    fn scatter(state: &mut GameState, dirs: [Direction; 3]) {
        for d in dirs {
            state.fix_d8_direction(d);
        }
    }

    /// Make the thrower at `THROWER` a Kick Team-mate kicker instead.
    fn kicker_instead(state: &mut GameState) {
        let kicker = id_at(state, THROWER);
        let skills = &mut state.get_mut_player_unsafe(kicker).stats.skills;
        skills.remove(&Skill::ThrowTeamMate);
        skills.insert(Skill::KickTeamMate);
        state.reask_turn_decision().unwrap();
    }

    /// S3, Throw Team-mate Action: "A player may only declare a Throw Team-mate Action if they
    /// have the Throw Team-mate Trait". A hidden action: legal, never listed, so bots never see
    /// it.
    #[test]
    fn the_declaration_is_legal_for_the_trait_only_and_hidden_from_bots() {
        let state = throw_state(&[(8, 7)], &[]);
        assert!(state.is_legal_action(&ttm(THROWER)));
        assert!(!state.is_legal_action(&ttm(MATE)), "no trait");
        assert!(!state.is_legal_action(&ttm(Position::new((8, 7)))));
        assert!(!listed(&state));
    }

    /// "may only throw a team-mate that has the Right Stuff Trait", an "adjacent team-mate";
    /// Right Stuff: "can be thrown ... even if this player is Prone" — not Stunned. Offered after
    /// the declaration (and the free Move Action), hidden.
    #[test]
    fn who_may_be_thrown() {
        // (9,4) Right Stuff Prone, (9,6) Right Stuff Stunned, (7,5) adjacent without the trait,
        // (10,5) Right Stuff but not adjacent; the opponent at (7,4) has Right Stuff.
        let mut state = throw_state(&[(9, 4), (9, 6), (7, 5), (10, 5)], &[(7, 4)]);
        for (x, y) in [(9, 4), (9, 6), (10, 5), (7, 4)] {
            give(&mut state, Position::new((x, y)), Skill::RightStuff);
        }
        state.get_mut_player_at_unsafe(Position::new((9, 4))).status = PlayerStatus::Down;
        state.get_mut_player_at_unsafe(Position::new((9, 6))).status = PlayerStatus::Stunned;
        state.reask_turn_decision().unwrap();
        state.step_positional(PosAT::ThrowTeamMate, THROWER);
        let offered: Vec<Position> = Position::all_positions()
            .filter(|&p| state.is_legal_action(&ttm(p)))
            .collect();
        assert_eq!(offered, vec![Position::new((9, 4)), MATE]);
        assert!(!listed(&state));
        assert!(state.is_legal_action(&Action::Positional(PosAT::Move, Position::new((8, 6)))));
    }

    /// The thrower must be Standing to pick a team-mate up: a Prone one declares, but is offered
    /// nobody until they stand.
    #[test]
    fn a_prone_thrower_cannot_throw_without_standing_up() {
        let mut state = throw_state(&[], &[]);
        state.get_mut_player_at_unsafe(THROWER).status = PlayerStatus::Down;
        state.reask_turn_decision().unwrap();
        state.step_positional(PosAT::ThrowTeamMate, THROWER);
        assert!(!state.is_legal_action(&ttm(MATE)));
    }

    /// Declare Target Square: "wholly underneath the first two sections of the Range Ruler" —
    /// Quick or Short, not Long.
    #[test]
    fn the_target_is_in_quick_or_short_range() {
        let mut state = throw_state(&[], &[]);
        pick_up(&mut state);
        assert!(state.is_legal_action(&ttm(THROWER + (-3, 0))), "Quick");
        assert!(state.is_legal_action(&ttm(THROWER + (-6, 0))), "Short");
        assert!(!state.is_legal_action(&ttm(THROWER + (-7, 0))), "Long");
        assert!(!state.is_legal_action(&ttm(THROWER)));
        assert!(state.get_all_actions().is_empty(), "every target is hidden");
    }

    /// Superb Throw: "The thrown player will Scatter (3) from the target square before landing";
    /// an Agility Test passed lands them on their feet, and "If they have not yet been activated
    /// that Turn then they can still be activated." The thrower is done.
    #[test]
    fn a_superb_throw_scatters_and_lands_on_its_feet() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        let target = THROWER + (-4, 0); // Short: PA 4+ at -1 passes on a 5
        state.fix_d6(5); // PA test: Superb
        scatter(&mut state, [Direction::up(), Direction::up(), Direction::left()]);
        state.fix_d6(3); // landing: AG 3+
        state.step_positional(PosAT::ThrowTeamMate, target);
        let mate = state.get_player_unsafe(id).clone();
        assert_eq!(mate.position, target + (-1, -2));
        assert_eq!(mate.status, PlayerStatus::Up);
        assert!(!mate.used, "may still be activated");
        assert!(state.get_player_id_at(MATE).is_none());
        assert!(state.get_player_unsafe(id_at(&state, THROWER)).used);
        assert!(state.home_to_act() && !state.info.turnover);
        assert!(state.is_legal_action(&Action::Positional(PosAT::StartMove, mate.position)));
    }

    /// Subpar Throw (the test failed): Scatter (3), and the landing is at -1; failing it the
    /// thrown player Falls Over — "this will only cause a Turnover if the thrown player was
    /// holding the ball".
    #[test]
    fn a_subpar_throw_lands_at_minus_one_and_a_fall_is_no_turnover() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        let target = THROWER + (-2, 0); // Quick: PA 4+
        state.fix_d6(3); // PA test: Subpar
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(3); // landing: AG 3+ at -1 needs a 4
        state.fix_d6(1); // armour
        state.fix_d6(1);
        state.step_positional(PosAT::ThrowTeamMate, target);
        let mate = state.get_player_unsafe(id);
        assert_eq!(mate.position, target + (-1, 0));
        assert_eq!(mate.status, PlayerStatus::Down);
        assert!(state.home_to_act() && !state.info.turnover);
    }

    /// The same landing roll passes after a Superb Throw: the -1 is the Subpar Throw's.
    #[test]
    fn the_landing_roll_has_no_modifier_after_a_superb_throw() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        state.fix_d6(4); // PA test: Superb (Quick)
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(3); // landing: AG 3+
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
    }

    /// Landing: "-1 for each opposition player Marking the square the thrown player will land
    /// in".
    #[test]
    fn the_landing_roll_counts_the_opponents_marking_the_square() {
        let landing = THROWER + (-3, 0);
        let mut state = throw_state(&[], &[(landing.x - 1, landing.y)]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        state.fix_d6(6); // Superb
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(3); // landing: AG 3+ at -1 needs a 4
        state.fix_d6(1); // armour
        state.fix_d6(1);
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    }

    /// Test for Accuracy: "-1 for each opposition player Marking the player performing the
    /// Throw Team-mate Action" — a 4 is Subpar on a Quick Throw with one marker.
    #[test]
    fn a_marked_thrower_throws_at_minus_one() {
        let mut state = throw_state(&[], &[(THROWER.x, THROWER.y - 1)]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        state.fix_d6(4); // PA 4+ at -1: Subpar
        scatter(&mut state, [Direction::down(), Direction::down(), Direction::left()]);
        state.fix_d6(3); // landing at -1: fails
        state.fix_d6(1); // armour
        state.fix_d6(1);
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    }

    /// Fumbled Throw: "The thrown player is dropped and will Bounce from the throwing player's
    /// square and must attempt to land" (at -1).
    #[test]
    fn a_fumbled_throw_bounces_from_the_thrower() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        state.fix_d6(1); // PA test: Fumbled
        state.fix_d8_direction(Direction::up());
        state.fix_d6(4); // landing: AG 3+ at -1
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-3, 0));
        let mate = state.get_player_unsafe(id);
        assert_eq!(mate.position, THROWER + Direction::up());
        assert_eq!(mate.status, PlayerStatus::Up);
        assert!(state.home_to_act() && !state.info.turnover);
    }

    /// A failed PA test may be re-rolled with a team re-roll.
    #[test]
    fn a_failed_throw_may_use_a_team_re_roll() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        state.get_mut_team(TeamType::Home).rerolls = 1;
        pick_up(&mut state);
        state.fix_d6(2); // Subpar
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
        state.fix_d6(6); // Superb
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(3); // landing: AG 3+, no Subpar -1
        state.step_simple(SimpleAT::UseReroll);
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
        assert_eq!(state.get_team(TeamType::Home).rerolls, 0);
    }

    /// "Players that were Prone ... when they were thrown will automatically fail the Agility
    /// Test to land": no landing roll, they Fall Over.
    #[test]
    fn a_prone_team_mate_is_thrown_and_cannot_land() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        state.get_mut_player_unsafe(id).status = PlayerStatus::Down;
        state.reask_turn_decision().unwrap();
        pick_up(&mut state);
        state.fix_d6(6); // Superb
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(1); // armour
        state.fix_d6(1);
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        let mate = state.get_player_unsafe(id);
        assert_eq!(mate.position, THROWER + (-3, 0));
        assert_eq!(mate.status, PlayerStatus::Down);
        assert!(state.home_to_act());
    }

    /// Landing in an Occupied Square: "the player in the occupied square is automatically
    /// Knocked Down ... The thrown player will then Bounce from the occupied square and will
    /// Fall Over" — no landing roll, and no turnover without the ball.
    #[test]
    fn a_crash_landing_knocks_the_occupant_down_and_bounces() {
        let occupant = THROWER + (-3, 0);
        let mut state = throw_state(&[], &[(occupant.x, occupant.y)]);
        let id = id_at(&state, MATE);
        let away = id_at(&state, occupant);
        pick_up(&mut state);
        state.fix_d6(6); // Superb
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]); // onto them
        state.fix_d8_direction(Direction::down()); // the thrown player Bounces
        for _ in 0..4 {
            state.fix_d6(1); // both armour rolls hold
        }
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(away).status, PlayerStatus::Down);
        let mate = state.get_player_unsafe(id);
        assert_eq!(mate.position, occupant + Direction::down());
        assert_eq!(mate.status, PlayerStatus::Down);
        assert!(state.home_to_act() && !state.info.turnover);
    }

    /// "If after the Bounce the thrown player lands in another occupied square, repeat the above
    /// process": two players Knocked Down, then the fall.
    #[test]
    fn a_crash_landing_bounces_on_through_occupied_squares() {
        let first = THROWER + (-3, 0);
        let second = first + Direction::down();
        let mut state = throw_state(&[], &[(first.x, first.y), (second.x, second.y)]);
        let id = id_at(&state, MATE);
        pick_up(&mut state);
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d8_direction(Direction::down()); // onto the second
        state.fix_d8_direction(Direction::down()); // free
        for _ in 0..6 {
            state.fix_d6(1); // three armour rolls hold
        }
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(id_at(&state, first)).status, PlayerStatus::Down);
        assert_eq!(
            state.get_player_unsafe(id_at(&state, second)).status,
            PlayerStatus::Down
        );
        assert_eq!(state.get_player_unsafe(id).position, second + Direction::down());
    }

    /// Crashing into a team-mate Knocks Down a player of the active team: a turnover (S3, The
    /// Turnover: "A player on the active team is Knocked Down during their own team's turn").
    #[test]
    fn crashing_into_a_team_mate_is_a_turnover() {
        let occupant = THROWER + (-3, 0);
        let mut state = throw_state(&[(occupant.x, occupant.y)], &[]);
        pick_up(&mut state);
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d8_direction(Direction::down());
        for _ in 0..4 {
            state.fix_d6(1);
        }
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert!(state.away_to_act(), "turnover");
    }

    /// "Should a thrown player leave the pitch as a result of a throw, they will risk Injury by
    /// the Crowd and a Turnover will be caused."
    #[test]
    fn a_throw_into_the_crowd_is_an_injury_and_a_turnover() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        let stats = state.get_player_unsafe(id).stats.clone();
        pick_up(&mut state);
        let target = Position::new((THROWER.x - 2, 1)); // the top row of the pitch
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::up(), Direction::up()]);
        state.fix_d6(1); // injury: Stunned, so to the Reserves
        state.fix_d6(1);
        state.step_positional(PosAT::ThrowTeamMate, target);
        assert!(state.get_players_on_pitch().all(|p| p.stats != stats));
        assert!(state
            .get_dugout()
            .any(|p| p.stats == stats && p.place == DugoutPlace::Reserves));
        assert!(state.away_to_act(), "turnover");
    }

    /// The thrown player holding the ball lands in the End Zone on their feet: a touchdown.
    #[test]
    fn a_thrown_ball_carrier_landing_in_the_end_zone_scores() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        let thrower = Position::new((4, 5));
        state.move_player(id_at(&state, THROWER), thrower).unwrap();
        state.move_player(id, thrower + (1, 0)).unwrap();
        state.set_ball(BallState::Carried(id));
        state.reask_turn_decision().unwrap();
        assert_eq!(state.get_endzone_x(TeamType::Home), 1);
        state.step_positional(PosAT::ThrowTeamMate, thrower);
        state.step_positional(PosAT::ThrowTeamMate, thrower + (1, 0));
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::up()]);
        state.fix_d6(3); // lands
        state.step_positional(PosAT::ThrowTeamMate, Position::new((1, 5)));
        assert_eq!(state.home.score, 1);
    }

    /// "A player on the active team that is in possession of the ball is thrown by a team-mate
    /// and fails to land safely" — a turnover, and the ball Bounces.
    #[test]
    fn a_thrown_ball_carrier_failing_to_land_is_a_turnover() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        state.set_ball(BallState::Carried(id));
        state.reask_turn_decision().unwrap();
        pick_up(&mut state);
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(2); // landing fails
        state.fix_d6(1); // armour
        state.fix_d6(1);
        state.fix_d8_direction(Direction::up()); // the ball Bounces
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        let landed = THROWER + (-3, 0);
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
        assert_eq!(state.ball, BallState::OnGround(landed + Direction::up()));
        assert!(state.away_to_act(), "turnover");
    }

    /// "Only a single Throw Team-mate Action can be declared each Turn"; Kick Team-mate "does
    /// not count as a team's Throw Team-mate Action", and is one per turn of its own.
    #[test]
    fn one_throw_and_one_kick_per_turn() {
        let other = Position::new((8, 8));
        let kicker = Position::new((8, 2));
        let second_kicker = Position::new((8, 1));
        let mut state = throw_state(&[(other.x, other.y), (kicker.x, kicker.y), (8, 1)], &[]);
        give(&mut state, other, Skill::ThrowTeamMate);
        give(&mut state, kicker, Skill::KickTeamMate);
        give(&mut state, second_kicker, Skill::KickTeamMate);
        state.reask_turn_decision().unwrap();
        state.step_positional(PosAT::ThrowTeamMate, THROWER);
        state.step_simple(SimpleAT::EndPlayerTurn);
        assert!(!state.is_legal_action(&ttm(other)), "the Throw is spent");
        assert!(state.is_legal_action(&ttm(kicker)), "the Kick is not");
        state.step_positional(PosAT::ThrowTeamMate, kicker);
        assert!(state.info.kick_team_mate_this_activation);
        state.step_simple(SimpleAT::EndPlayerTurn);
        assert!(!state.is_legal_action(&ttm(second_kicker)), "the Kick is spent");
        // Both are back in the team's next turn.
        state.step_simple(SimpleAT::EndTurn);
        state.step_simple(SimpleAT::EndTurn);
        assert!(state.home_to_act());
        assert!(state.is_legal_action(&ttm(other)) && state.is_legal_action(&ttm(kicker)));
    }

    /// Kick Team-mate "works exactly the same as a Throw Team-mate Action" — here a Superb kick.
    #[test]
    fn a_kick_team_mate_lands_like_a_throw() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        kicker_instead(&mut state);
        pick_up(&mut state);
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::left()]);
        state.fix_d6(3);
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert_eq!(state.get_player_unsafe(id).position, THROWER + (-3, 0));
        assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
        assert!(state.info.throw_team_mate_available, "a Kick is not the team's Throw");
    }

    /// Kick Team-mate: "if a Kick Team-mate Special Action results in a Fumbled Throw,
    /// immediately make an Injury Roll for the team-mate being kicked, treating any result of
    /// Stunned as Knocked Out. If the kicked player was holding the ball, it will Bounce from the
    /// square they were in."
    #[test]
    fn a_fumbled_kick_injures_the_kicked_team_mate() {
        let mut state = throw_state(&[], &[]);
        let id = id_at(&state, MATE);
        let stats = state.get_player_unsafe(id).stats.clone();
        state.set_ball(BallState::Carried(id));
        kicker_instead(&mut state);
        pick_up(&mut state);
        state.fix_d6(1); // Fumbled
        state.fix_d6(1); // injury 2: Stunned, read as Knocked Out
        state.fix_d6(1);
        state.fix_d8_direction(Direction::down()); // the ball Bounces from their square
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (-2, 0));
        assert!(state
            .get_dugout()
            .any(|p| p.stats == stats && p.place == DugoutPlace::KnockOut));
        assert_eq!(state.ball, BallState::OnGround(MATE + Direction::down()));
        assert!(state.away_to_act(), "turnover: the ball carrier is off the pitch");
    }

    /// Charge!: "one may perform a free Throw Team-mate Action, and one may perform a free Kick
    /// Team-mate Action" — offered, hidden, to the selected players with the trait.
    #[test]
    fn charge_offers_throw_and_kick_team_mate() {
        let kicker = Position::new((8, 2));
        let mut state = GameStateBuilder::new()
            .add_away_player(THROWER)
            .add_away_player(MATE)
            .add_away_player(kicker)
            .build();
        give(&mut state, THROWER, Skill::ThrowTeamMate);
        give(&mut state, MATE, Skill::RightStuff);
        give(&mut state, kicker, Skill::KickTeamMate);
        state.set_ball(BallState::InAir(Position::new((2, 2))));
        state.fix_d6(1); // D3 + 3 players
        state.start_proc(Charge::new(TeamType::Away));
        assert!(state.away_to_act());
        assert!(state.is_legal_action(&ttm(THROWER)) && state.is_legal_action(&ttm(kicker)));
        assert!(!state.is_legal_action(&ttm(MATE)));
        assert!(!listed(&state));
        let id = id_at(&state, MATE);
        state.step_positional(PosAT::ThrowTeamMate, THROWER);
        state.step_positional(PosAT::ThrowTeamMate, MATE);
        state.fix_d6(6);
        scatter(&mut state, [Direction::up(), Direction::down(), Direction::right()]);
        state.fix_d6(3);
        state.step_positional(PosAT::ThrowTeamMate, THROWER + (3, 0));
        assert_eq!(state.get_player_unsafe(id).position, THROWER + (4, 0));
        assert!(!state.is_legal_action(&ttm(THROWER)), "the throw is taken");
        assert!(state.is_legal_action(&ttm(kicker)), "the kick is still on offer");
    }
}

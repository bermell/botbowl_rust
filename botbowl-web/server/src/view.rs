//! `GameState -> ViewState`: the one place game logic turns into something the
//! client can draw.
//!
//! Decision 3 of plan 034 puts *all* of it here, server-side and pure, so the
//! wasm build stays trivial and this is unit-testable against
//! `GameStateBuilder` positions. The client never computes geometry, legality
//! or probability — it renders what arrives.

use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model as em;
use botbowl_engine::core::model::{BallState, Position, SomeProcInput};
use botbowl_engine::core::pathing::{CustomIntoIter, Node, PathingEvent, PositionOrEvent};
use botbowl_engine::core::procedures::{AnyProc, Formation};
use botbowl_engine::core::table as et;
use botbowl_web_proto::view as pv;

use crate::mirror;

/// Session facts that are not in the `GameState` but belong on the view.
///
/// `trail` is the one piece of *history* on the view, and it comes from the
/// session's step snapshots ([`trail`]), never from the state: a `GameState`
/// that remembered where its players had been would stop recombining with
/// one that reached the same position another way.
pub struct DeriveCtx {
    /// The sides played from the browser.
    pub humans: Vec<botbowl_web_proto::TeamType>,
    pub seq: u64,
    pub can_undo: bool,
    pub bot_thinking: bool,
    pub step_mode: botbowl_web_proto::msg::StepMode,
    pub paused: bool,
    /// A bot's chosen, not yet played move (set only while `paused`).
    pub pending_action: Option<botbowl_web_proto::Action>,
    /// Where the active player has been this activation — see [`trail`].
    pub trail: Vec<botbowl_web_proto::Position>,
    /// Each side's team, for the players' pictures.
    pub looks: std::sync::Arc<crate::teams::Looks>,
}

impl Default for DeriveCtx {
    fn default() -> Self {
        DeriveCtx {
            humans: vec![botbowl_web_proto::TeamType::Home],
            seq: 0,
            can_undo: false,
            bot_thinking: false,
            step_mode: botbowl_web_proto::msg::StepMode::default(),
            paused: false,
            pending_action: None,
            trail: Vec::new(),
            looks: Default::default(),
        }
    }
}

/// How a pathfinder roll reads in the move-probability tooltip.
fn event_label(event: &PathingEvent) -> String {
    match event {
        PathingEvent::Dodge(t, _) => format!("Dodge {}+", *t as u8),
        PathingEvent::GFI(t) => format!("GFI {}+", *t as u8),
        PathingEvent::Pickup(t) => format!("Pickup {}+", *t as u8),
        PathingEvent::Block(_, n) => format!("Block ({} dice)", mirror::num_block_dices_to_proto(*n).signed()),
        PathingEvent::Handoff(_, t) => format!("Catch {}+", *t as u8),
        PathingEvent::Pass { to, pass, modifer } => {
            format!("Pass {}+ ({modifer:+}) to ({}, {})", *pass as u8, to.x, to.y)
        }
        PathingEvent::Touchdown(_) => "Touchdown!".to_string(),
        PathingEvent::Foul(_, t) => format!("Armour {}+", *t as u8),
        PathingEvent::StandUp => "Stand up".to_string(),
    }
}

fn square_kind(state: &GameState, pos: Position) -> pv::SquareKind {
    let dims = state.board_dims;
    if state.is_out(pos) {
        return pv::SquareKind::OutOfBounds;
    }
    if pos.x == dims.endzone_x(em::TeamType::Home) {
        return pv::SquareKind::EndzoneHome;
    }
    if pos.x == dims.endzone_x(em::TeamType::Away) {
        return pv::SquareKind::EndzoneAway;
    }
    let on_los_column = pos.x == dims.los_x(em::TeamType::Home) || pos.x == dims.los_x(em::TeamType::Away);
    if on_los_column && dims.los_y_range().contains(&pos.y) {
        return pv::SquareKind::Scrimmage;
    }
    if dims.north_wing_y_range().contains(&pos.y) {
        return pv::SquareKind::WingNorth;
    }
    if dims.south_wing_y_range().contains(&pos.y) {
        return pv::SquareKind::WingSouth;
    }
    pv::SquareKind::Normal
}

fn player_view(state: &GameState, ctx: &DeriveCtx, p: &em::FieldedPlayer, has_ball: bool) -> pv::PlayerView {
    let team = p.stats.team;
    let role = mirror::role_to_proto(p.stats.role);
    // Every skill the engine knows, not a hand-picked six: the curriculum can
    // now field players with any of them (`Skill::good_skills`), and one the
    // list forgot would be invisible in the browser.
    let mut skills: Vec<String> = et::Skill::ALL
        .into_iter()
        .filter(|s| p.has_skill(*s))
        .map(|s| mirror::skill_label(s).to_string())
        .collect();
    skills.sort();

    let active = state.info.active_player == Some(p.id);
    pv::PlayerView {
        id: p.id,
        team: mirror::team_to_proto(team),
        role,
        status: mirror::status_to_proto(p.status),
        used: p.used,
        // The engine flags a player `used` the moment they are activated. The
        // one still acting keeps the "has not acted" picture: greying them out
        // mid-move reads as "this player is done", which is the opposite of
        // what is happening.
        sprite: ctx.looks.sprite(&p.stats, p.used && !active),
        st: p.stats.str_,
        ma: p.stats.ma,
        ag: p.stats.ag,
        av: p.stats.av,
        movement_left: p.total_movement_left(),
        has_ball,
        skills,
        active,
    }
}

/// The block being resolved, if one is: the `Block` procedure anywhere on the
/// stack (a `Push`, an `Armor` or a pending die may sit on top of it), its
/// attacker being the active player.
fn block_view(state: &GameState) -> Option<pv::BlockView> {
    let block = state.proc_stack_iter().find_map(|p| match p {
        AnyProc::Block(b) => Some(b),
        _ => None,
    })?;
    let attacker = state.get_active_player()?;
    // The defender may already be in the crowd (and about to be unfielded).
    let defender = state.get_player(block.defender()).ok()?;
    Some(pv::BlockView {
        attacker: mirror::position_to_proto(attacker.position),
        defender: mirror::position_to_proto(defender.position),
        dice: mirror::num_block_dices_to_proto(block.num_dices()),
    })
}

/// The squares the active player has walked this activation, oldest first,
/// excluding where they now stand — read off the session's step snapshots.
///
/// `MoveAction::continue_along_path` walks every roll-free square of a path
/// in one micro-step, so consecutive snapshots can be several squares apart;
/// the squares in between come from the pathfinder route to the arrival
/// square in the last snapshot that still had a path buffer (the pathfinder
/// builds a tree, so that route is the one walked).
pub fn trail(steps: &[GameState]) -> Vec<botbowl_web_proto::Position> {
    let Some(current) = steps.last() else { return Vec::new() };
    let Some(id) = current.info.active_player else {
        return Vec::new();
    };
    let Ok(player) = current.get_player(id) else {
        return Vec::new();
    };
    let team = player.stats.team;
    let turn = (current.info.half, current.info.home_turn, current.info.away_turn);

    // Newest first: this player's position at each snapshot of the activation.
    let mut visited: Vec<(usize, Position)> = vec![(steps.len() - 1, player.position)];
    for (i, s) in steps.iter().enumerate().rev().skip(1) {
        if s.info.active_player != Some(id) || (s.info.half, s.info.home_turn, s.info.away_turn) != turn {
            break;
        }
        let Ok(p) = s.get_player(id) else { break };
        if p.stats.team != team {
            break;
        }
        if visited.last().is_some_and(|(_, last)| *last != p.position) {
            visited.push((i, p.position));
        }
    }
    visited.reverse();

    let mut out = Vec::new();
    for pair in visited.windows(2) {
        let ((i, from), (_, to)) = (pair[0], pair[1]);
        out.push(from);
        if from.distance_to(&to) > 1 {
            out.extend(route_between(&steps[..=i], from, to));
        }
    }
    out.into_iter().map(mirror::position_to_proto).collect()
}

/// The squares strictly between `from` and `to` on the pathfinder route to
/// `to`, from the newest of `steps` that offers one. Empty when none does.
fn route_between(steps: &[GameState], from: Position, to: Position) -> Vec<Position> {
    for s in steps.iter().rev() {
        let Some(paths) = s.get_paths() else { continue };
        let Some(node) = paths.get_pos(to) else { continue };
        let squares: Vec<Position> = node
            .iter()
            .filter_map(|e| match e {
                PositionOrEvent::Position(p) => Some(p),
                PositionOrEvent::Event(_) => None,
            })
            .collect();
        // The route excludes its origin; `from` is either that origin or a
        // square along the way.
        let start = squares.iter().position(|p| *p == from).map_or(0, |i| i + 1);
        return squares[start..].iter().copied().take_while(|p| *p != to).collect();
    }
    Vec::new()
}

/// Tackle zones exerted on every square by each team. Computed square-first
/// (rather than via `get_tz_on`, which needs a player id) so the overlay also
/// covers empty squares — which is the whole point of a tackle-zone overlay.
fn tackle_zones(state: &GameState, squares: &mut [pv::SquareView]) {
    let dims = state.board_dims;
    for p in state.get_players_on_pitch() {
        // A player pushed into the crowd sits at an out-of-bounds position
        // between `Push` moving them there and the queued crowd injury
        // unfielding them — still "on pitch" the whole time. `get_adj_positions`
        // asserts its input is in bounds, so skip them rather than panic.
        if !p.has_tackle_zone() || dims.is_out(p.position) {
            continue;
        }
        for adj in state.get_adj_positions(p.position) {
            if dims.is_out(adj) {
                continue;
            }
            let Some(index) = index_of(dims, adj) else { continue };
            let sq = &mut squares[index];
            match p.stats.team {
                em::TeamType::Home => sq.tz_home += 1,
                em::TeamType::Away => sq.tz_away += 1,
            }
        }
    }
}

/// Row-major index of a position in the square grid, or `None` when the
/// position is not on the grid at all.
///
/// This has to be fallible: the engine legitimately puts the ball *outside*
/// the array while it is in flight — `Kickoff` sets
/// `BallState::InAir(aim + direction * len)` with `len` capped only at
/// `max_scatter()`, which on a narrow board reaches well past the border ring
/// into negative coordinates. `Position` is `i8`, so an unchecked
/// `pos.y as usize` wrapped to ~2^64 and the multiply overflowed, panicking
/// the session thread mid-kickoff.
fn index_of(dims: em::BoardDims, pos: Position) -> Option<usize> {
    if pos.x < 0 || pos.y < 0 || pos.x >= dims.width || pos.y >= dims.height {
        return None;
    }
    Some(pos.y as usize * dims.width as usize + pos.x as usize)
}

/// Success probability, block dice and route of one pathfinder node.
fn node_info(node: &std::sync::Arc<Node>) -> (f32, Option<botbowl_web_proto::dice::NumBlockDices>, pv::RouteView) {
    let mut steps = Vec::new();
    let mut rolls = Vec::new();
    for item in node.iter() {
        match item {
            PositionOrEvent::Position(p) => steps.push(mirror::position_to_proto(p)),
            PositionOrEvent::Event(e) => rolls.push(event_label(&e)),
        }
    }
    (
        node.prob,
        node.get_block_dice().map(mirror::num_block_dices_to_proto),
        pv::RouteView { steps, rolls },
    )
}

/// Annotate every square offered by the path buffer with its success
/// probability, block dice and route. The path buffer only exists while a
/// player action is active, and it holds one `Node` per reachable square.
fn annotate_paths(state: &GameState, squares: &mut [pv::SquareView]) {
    let Some(paths) = state.get_paths() else {
        return;
    };
    let dims = state.board_dims;
    for y in 0..dims.height {
        for x in 0..dims.width {
            let pos = Position::new((x, y));
            let Some(node) = paths.get_pos(pos) else {
                continue;
            };
            // `paths` is capacity-sized, so this index is in range by
            // construction — but go through the same checked path anyway.
            let Some(index) = index_of(dims, pos) else { continue };
            let sq = &mut squares[index];
            let (prob, dice, route) = node_info(node);
            sq.move_prob = Some(prob);
            sq.block_dice = dice;
            sq.route = Some(route);
        }
    }
}

/// The six declarations, in policy-channel order.
const STARTS: [et::PosAT; 6] = [
    et::PosAT::StartMove,
    et::PosAT::StartBlitz,
    et::PosAT::StartPass,
    et::PosAT::StartFoul,
    et::PosAT::StartHandoff,
    et::PosAT::StartBlock,
];

/// Preview every declaration the side to act may make on the player at
/// `pos`, and resolve each square to the one action a click there should
/// take (see [`pv::SelectionView`]). Pure: each declaration is stepped on a
/// clone. `None` when `pos` is not a player who may be declared.
pub fn selection(state: &GameState, pos: botbowl_web_proto::Position, seq: u64) -> Option<pv::SelectionView> {
    selection_now(state, pos, seq).or_else(|| {
        // Mid-activation nobody else can be declared; preview the position
        // after `EndPlayerTurn` instead, and say the chain must start with it.
        let ended = after_end_player_turn(state)?;
        let mut sel = selection_now(&ended, pos, seq)?;
        sel.end_first = true;
        Some(sel)
    })
}

/// `state` with the active player's turn ended, when that is the side to
/// act's own choice and lands straight on its next declaration. `None`
/// otherwise — no `EndPlayerTurn` on offer, or ending it rolls a die or hands
/// the decision to the other side.
fn after_end_player_turn(state: &GameState) -> Option<GameState> {
    let end = em::Action::Simple(et::SimpleAT::EndPlayerTurn);
    if state.info.active_player.is_none() || state.pending_roll.is_some() || !state.is_legal_action(&end) {
        return None;
    }
    let team = state.get_active_teamtype();
    let mut ended = state.clone();
    if !matches!(ended.dice_mode(), DiceMode::RegisterRolls) {
        ended.set_dice_mode(DiceMode::RegisterRolls);
    }
    ended.step_with_roll_or_action(SomeProcInput::Action(end));
    (ended.pending_roll.is_none() && ended.get_active_teamtype() == team && !ended.info.game_over).then_some(ended)
}

/// The team-mates a click could select while a player is mid-activation:
/// whoever may be declared once that player's turn is ended.
fn reselect(state: &GameState) -> Vec<botbowl_web_proto::Position> {
    let Some(ended) = after_end_player_turn(state) else {
        return Vec::new();
    };
    let mut out: Vec<_> = ended
        .get_all_actions()
        .into_iter()
        .filter_map(|a| match a {
            em::Action::Positional(at, pos) if STARTS.contains(&at) => Some(mirror::position_to_proto(pos)),
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn selection_now(state: &GameState, pos: botbowl_web_proto::Position, seq: u64) -> Option<pv::SelectionView> {
    let at = mirror::position_from_proto(pos);
    let player = state.get_player_at(at)?;
    let (team, id, carrier) = (player.stats.team, player.id, state.ball == BallState::Carried(player.id));
    let starts: Vec<et::PosAT> = STARTS
        .into_iter()
        .filter(|s| state.is_legal_action(&em::Action::Positional(*s, at)))
        .collect();
    if starts.is_empty() {
        return None;
    }

    // Every (declaration, square, intent) the declarations would offer next.
    let mut offered: Vec<(et::PosAT, Position, pv::IntentView)> = Vec::new();
    for &start in &starts {
        let mut preview = state.clone();
        // A declaration that rolls at once (Jump Up) pauses on the roll here
        // rather than drawing on anybody's dice: it then offers nothing.
        if !matches!(preview.dice_mode(), DiceMode::RegisterRolls) {
            preview.set_dice_mode(DiceMode::RegisterRolls);
        }
        preview.step_with_roll_or_action(SomeProcInput::Action(em::Action::Positional(start, at)));
        if preview.pending_roll.is_some() || preview.info.active_player != Some(id) {
            continue;
        }
        for action in preview.get_all_actions() {
            let em::Action::Positional(next, target) = action else { continue };
            if STARTS.contains(&next) {
                continue;
            }
            let node = preview.get_paths().and_then(|p| p.get_pos(target).as_ref());
            let (prob, block_dice, route) = match node {
                Some(n) => {
                    let (p, d, r) = node_info(n);
                    (Some(p), d, Some(r))
                }
                None => (None, None, None),
            };
            offered.push((
                start,
                target,
                pv::IntentView {
                    pos: mirror::position_to_proto(target),
                    start: mirror::pos_at_to_proto(start),
                    action: mirror::pos_at_to_proto(next),
                    prob,
                    block_dice,
                    route,
                },
            ));
        }
    }
    let find = |start: et::PosAT, target: Position| {
        offered
            .iter()
            .find(|(s, t, _)| *s == start && *t == target)
            .map(|(_, _, i)| i.clone())
    };

    let mut squares: Vec<Position> = offered.iter().map(|(_, t, _)| *t).collect();
    squares.sort_by_key(|p| (p.y, p.x));
    squares.dedup();
    let mut targets = Vec::new();
    for target in squares {
        let intent = match state.get_player_at(target) {
            // An empty square, or the loose ball: walk there.
            None => find(et::PosAT::StartMove, target),
            Some(other) if other.stats.team != team => {
                if other.status != em::PlayerStatus::Up {
                    find(et::PosAT::StartFoul, target)
                } else if target.distance_to(&at) == 1 {
                    find(et::PosAT::StartBlock, target).or_else(|| find(et::PosAT::StartBlitz, target))
                } else {
                    find(et::PosAT::StartBlitz, target)
                }
            }
            // A team-mate, from the ball carrier: the likelier of a handoff
            // and a pass. From anyone else a click on a team-mate selects them.
            Some(_) if carrier => {
                let handoff = find(et::PosAT::StartHandoff, target);
                let pass = find(et::PosAT::StartPass, target);
                match (handoff, pass) {
                    (Some(h), Some(p)) => Some(if p.prob.unwrap_or(0.0) > h.prob.unwrap_or(0.0) { p } else { h }),
                    (h, p) => h.or(p),
                }
            }
            Some(_) => None,
        };
        targets.extend(intent);
    }
    Some(pv::SelectionView {
        seq,
        player: pos,
        end_first: false,
        starts: starts.into_iter().map(mirror::pos_at_to_proto).collect(),
        targets,
    })
}

/// The square the open reroll / skill / block-dice question is about.
fn prompt(state: &GameState) -> Option<pv::PromptView> {
    use et::SimpleAT as S;
    let asks = state.get_all_actions().into_iter().any(|a| {
        matches!(
            a,
            em::Action::Simple(
                S::UseReroll
                    | S::DontUseReroll
                    | S::UseSkill
                    | S::DontUseSkill
                    | S::SelectBothDown
                    | S::SelectPow
                    | S::SelectPush
                    | S::SelectPowPush
                    | S::SelectSkull
            )
        )
    });
    if !asks {
        return None;
    }
    let top = state.proc_stack_iter().next()?;
    let player_pos = |id: em::PlayerID| state.get_player(id).ok().map(|p| p.position);
    let pos = match top {
        AnyProc::Catch(c) => player_pos(c.id()),
        AnyProc::Deflect(c) => player_pos(c.id()),
        AnyProc::DodgeProc(c) => player_pos(c.id()),
        AnyProc::GfiProc(c) => player_pos(c.id()),
        AnyProc::PickupProc(c) => player_pos(c.id()),
        AnyProc::JumpUp(c) => player_pos(c.id()),
        AnyProc::Push(p) => Some(p.on()),
        // The dice go over the defender: that is what they are about.
        AnyProc::Block(_) => block_view(state).map(|b| mirror::position_from_proto(b.defender)),
        // Any other block question (Wrestle, Frenzy) belongs to whichever of
        // the two players' coaches is being asked.
        _ => match (block_view(state), state.get_active_player()) {
            (Some(b), Some(attacker)) => {
                let asked = state.get_active_teamtype();
                Some(mirror::position_from_proto(if asked == Some(attacker.stats.team) {
                    b.attacker
                } else {
                    b.defender
                }))
            }
            (None, active) => active.map(|p| p.position),
            _ => None,
        },
    }?;
    (!state.is_out(pos)).then(|| pv::PromptView {
        pos: mirror::position_to_proto(pos),
        title: crate::dice::purpose(top.name()),
    })
}

fn scoreboard(state: &GameState) -> pv::Scoreboard {
    let info = &state.info;
    pv::Scoreboard {
        half: info.half,
        home_turn: info.home_turn,
        away_turn: info.away_turn,
        home_score: state.home.score,
        away_score: state.away.score,
        home_rerolls: state.home.rerolls,
        away_rerolls: state.away.rerolls,
        home_can_reroll: state.home.can_use_reroll(),
        away_can_reroll: state.away.can_use_reroll(),
        weather: mirror::weather_to_proto(&info.weather),
        team_turn: mirror::team_to_proto(info.team_turn),
        kicking_this_drive: mirror::team_to_proto(info.kicking_this_drive),
        game_over: info.game_over,
        winner: info.winner.map(mirror::team_to_proto),
    }
}

fn dugout(state: &GameState, ctx: &DeriveCtx, team: em::TeamType) -> pv::DugoutView {
    let mut players: Vec<pv::DugoutPlayerView> = state
        .get_dugout()
        .filter(|p| p.stats.team == team)
        .map(|p| {
            let role = mirror::role_to_proto(p.stats.role);
            pv::DugoutPlayerView {
                id: p.id,
                team: mirror::team_to_proto(team),
                role,
                place: mirror::dugout_place_to_proto(p.place),
                // Bench players have not acted, so they get the `an` sprite.
                sprite: ctx.looks.sprite(&p.stats, false),
            }
        })
        .collect();
    // Group the bench so the panel reads Reserves → KO → casualties.
    players.sort_by_key(|p| (place_order(p.place), p.role.label(), p.id));
    pv::DugoutView {
        team: mirror::team_to_proto(team),
        players,
    }
}

fn place_order(place: pv::DugoutPlace) -> u8 {
    match place {
        pv::DugoutPlace::Reserves => 0,
        pv::DugoutPlace::Heated => 1,
        pv::DugoutPlace::KnockOut => 2,
        pv::DugoutPlace::Injured => 3,
        pv::DugoutPlace::Ejected => 4,
    }
}

/// The whole derivation. Pure: no interior mutability, no RNG, no engine
/// stepping.
pub fn derive(state: &GameState, ctx: &DeriveCtx) -> pv::ViewState {
    let dims = state.board_dims;
    let mut squares: Vec<pv::SquareView> = Vec::with_capacity(dims.width as usize * dims.height as usize);
    for y in 0..dims.height {
        for x in 0..dims.width {
            let pos = Position::new((x, y));
            squares.push(pv::SquareView::empty(
                mirror::position_to_proto(pos),
                square_kind(state, pos),
            ));
        }
    }

    let ball_pos = state.get_ball_position();
    let carrier = match state.ball {
        BallState::Carried(id) => Some(id),
        _ => None,
    };

    for p in state.get_players_on_pitch() {
        if let Some(idx) = index_of(dims, p.position) {
            squares[idx].player = Some(player_view(state, ctx, p, carrier == Some(p.id)));
        }
    }

    // A ball off the grid is one still in flight past the touchline; it is
    // simply not drawn until the throw-in brings it back.
    if let Some(idx) = ball_pos.and_then(|pos| index_of(dims, pos)) {
        squares[idx].ball = Some(match state.ball {
            BallState::Carried(_) => pv::BallView::Carried,
            BallState::InAir(_) => pv::BallView::InAir,
            BallState::OnGround(_) => pv::BallView::OnGround,
            // `get_ball_position` returned Some, so OffPitch is unreachable;
            // treat it as on the ground rather than panicking on a view.
            BallState::OffPitch => pv::BallView::OnGround,
        });
    }

    tackle_zones(state, &mut squares);

    // The game-over procedure still offers an action (`DontUseReroll` to
    // Home) — check `game_over` first or the UI invites a click that ends
    // the session.
    let game_over = state.info.game_over;
    let mut simple_actions: Vec<pv::SimpleActionView> = Vec::new();

    if !game_over {
        for action in state.get_all_actions() {
            match action {
                em::Action::Positional(at, pos) => {
                    // An action outside the runtime board would be an engine
                    // bug, but a view must not panic on one.
                    if let Some(index) = index_of(dims, pos) {
                        squares[index].actions.push(mirror::pos_at_to_proto(at));
                    }
                }
                em::Action::Simple(at) => {
                    let at = mirror::simple_at_to_proto(at);
                    simple_actions.push(pv::SimpleActionView {
                        at,
                        label: at.label().to_string(),
                        img: at.block_die_face().map(str::to_string),
                    });
                }
            }
        }
        annotate_paths(state, &mut squares);
    }

    for sq in squares.iter_mut() {
        sq.actions.sort_by_key(|at| format!("{at:?}"));
        sq.actions.dedup();
    }

    let to_act = if game_over {
        None
    } else {
        state.get_active_teamtype().map(mirror::team_to_proto)
    };

    // Setup is one placement per player: everyone available for the drive is
    // staged on the pitch with `used == true` until placed, so the counts
    // fall out of the fielded players rather than the dugout.
    let setup = state.setup_team().filter(|_| !game_over).map(|team| {
        let (placed, waiting) = state
            .get_players_on_pitch_in_team(team)
            .fold((0, 0), |(placed, waiting), p| {
                if p.used {
                    (placed, waiting + 1)
                } else {
                    (placed + 1, waiting)
                }
            });
        pv::SetupView {
            team: mirror::team_to_proto(team),
            placed,
            waiting,
            team_size: dims.team_size,
            formations: Formation::available(&dims).map(|f| format!("{f:?}")).collect(),
        }
    });

    pv::ViewState {
        seq: ctx.seq,
        dims: pv::Dims {
            width: dims.width,
            height: dims.height,
            team_size: dims.team_size,
        },
        scoreboard: scoreboard(state),
        squares,
        dugouts: vec![
            dugout(state, ctx, em::TeamType::Home),
            dugout(state, ctx, em::TeamType::Away),
        ],
        simple_actions,
        to_act,
        humans: ctx.humans.clone(),
        proc: state.proc_stack_top().unwrap_or("-").to_string(),
        active_player: state.info.active_player,
        pending_roll: state.pending_roll.map(mirror::requested_roll_to_proto),
        can_undo: ctx.can_undo,
        setup,
        bot_thinking: ctx.bot_thinking,
        step_mode: ctx.step_mode,
        paused: ctx.paused,
        pending_action: ctx.pending_action,
        trail: ctx.trail.clone(),
        block: block_view(state),
        prompt: if game_over { None } else { prompt(state) },
        reselect: if game_over { Vec::new() } else { reselect(state) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::{BuilderState, GameStateBuilder};
    use botbowl_engine::core::procedures::auto_setup;
    use botbowl_engine::core::table::SimpleAT;
    use botbowl_web_proto::action as pa;
    use botbowl_web_proto::action::{PosAT, SimpleAT as PSimpleAT, TeamType as PTeam};

    /// The plan's first target board: 14x7 playable → 16x9 engine, 4 a side.
    const DIMS: (i8, i8, usize) = (16, 9, 4);

    fn dims() -> em::BoardDims {
        let (w, h, t) = DIMS;
        em::BoardDims::new(w, h, t)
    }

    /// Home's own half is the high-x half (it *attacks* x = 1), so Home
    /// players go at x >= width/2.
    fn a_position() -> GameState {
        GameStateBuilder::new()
            .with_board_dims(dims())
            .add_home_players(&[(8, 3), (8, 4), (9, 5), (10, 2)])
            .add_away_players(&[(7, 3), (7, 4), (6, 5), (5, 2)])
            .add_ball((8, 5))
            .build()
    }

    fn view_of(state: &GameState) -> pv::ViewState {
        derive(state, &DeriveCtx::default())
    }

    fn at(v: &pv::ViewState, x: i8, y: i8) -> &pv::SquareView {
        v.square(pa::Position::new(x, y)).expect("square in range")
    }

    #[test]
    fn the_square_grid_is_indexable_by_position() {
        let v = view_of(&a_position());
        assert_eq!(v.squares.len(), 16 * 9);
        for sq in &v.squares {
            assert_eq!(
                v.square(sq.pos).map(|s| s.pos),
                Some(sq.pos),
                "index mismatch at {:?}",
                sq.pos
            );
        }
    }

    #[test]
    fn square_kinds_follow_the_runtime_board_not_the_capacity() {
        let v = view_of(&a_position());
        // The border ring.
        for (x, y) in [(0, 0), (15, 8), (0, 4), (15, 4), (7, 0), (7, 8)] {
            assert_eq!(at(&v, x, y).kind, pv::SquareKind::OutOfBounds, "({x},{y})");
        }
        // End zones: Home attacks x = 1, Away attacks x = width - 2.
        assert_eq!(at(&v, 1, 4).kind, pv::SquareKind::EndzoneHome);
        assert_eq!(at(&v, 14, 4).kind, pv::SquareKind::EndzoneAway);
        // Scrimmage: the two centre columns, but only across the middle rows.
        assert_eq!(at(&v, 8, 4).kind, pv::SquareKind::Scrimmage);
        assert_eq!(at(&v, 7, 4).kind, pv::SquareKind::Scrimmage);
        assert_eq!(at(&v, 8, 1).kind, pv::SquareKind::WingNorth);
        assert_eq!(at(&v, 8, 7).kind, pv::SquareKind::WingSouth);
        // Nothing beyond the runtime board should look playable, even though
        // the compiled `FullPitch` is much larger.
        assert!(v.squares.iter().all(|s| s.pos.x < 16 && s.pos.y < 9));
    }

    #[test]
    fn players_land_on_their_squares_with_a_colour_and_an_acted_flag() {
        let v = view_of(&a_position());
        let home = at(&v, 8, 3).player.as_ref().expect("home player at (8,3)");
        assert_eq!(home.team, PTeam::Home);
        assert!(!home.used, "nobody has acted at the start of a turn");
        // `b` is the home colourway, `an` means "has not acted yet".
        assert!(home.sprite.ends_with("ban.gif"), "{}", home.sprite);
        let away = at(&v, 7, 3).player.as_ref().expect("away player at (7,3)");
        assert_eq!(away.team, PTeam::Away);
        assert!(away.sprite.ends_with("1an.gif"), "{}", away.sprite);
        assert!(!away.sprite.contains("ban"), "away must not get the home colour");

        // Eight players, and nobody anywhere else.
        assert_eq!(v.squares.iter().filter(|s| s.player.is_some()).count(), 8);
    }

    #[test]
    fn a_loose_ball_sits_on_its_square_and_a_carried_one_marks_its_carrier() {
        let loose = view_of(&a_position());
        assert_eq!(at(&loose, 8, 5).ball, Some(pv::BallView::OnGround));
        assert!(loose
            .squares
            .iter()
            .all(|s| s.player.as_ref().is_none_or(|p| !p.has_ball)));

        let carried = GameStateBuilder::new()
            .with_board_dims(dims())
            .add_home_players(&[(9, 5)])
            .add_ball((9, 5))
            .build();
        let v = view_of(&carried);
        let sq = at(&v, 9, 5);
        assert_eq!(sq.ball, Some(pv::BallView::Carried));
        assert!(sq.player.as_ref().unwrap().has_ball);
    }

    #[test]
    fn tackle_zones_are_counted_on_empty_squares_too() {
        // Away players at (7,3) and (7,4) both cover the empty square (8,3)?
        // No — (8,3) holds a Home player. Use the empty (6,3)/(6,4) side.
        let v = view_of(&a_position());
        // (8,3) is occupied by Home but still *inside* two Away tackle zones
        // (from (7,3) and (7,4)) — this is exactly what `get_tz_on` gives for
        // that player, and what the overlay needs per square.
        assert_eq!(at(&v, 8, 3).tz_away, 2);
        // An empty square between two Away markers is in both zones...
        assert_eq!(at(&v, 6, 2).tz_away, 2, "(6,2) is adjacent to Away (5,2) and (7,3)");
        // ...and one on the edge of the cluster is in exactly one.
        assert_eq!(at(&v, 4, 1).tz_away, 1, "(4,1) is adjacent to Away (5,2) only");
        // A square far from everyone.
        assert_eq!(at(&v, 13, 7).tz_home, 0);
        assert_eq!(at(&v, 13, 7).tz_away, 0);
    }

    #[test]
    fn a_prone_player_exerts_no_tackle_zone() {
        let mut state = a_position();
        let id = state.get_player_id_at(em::Position::new((7, 3))).unwrap();
        let before = view_of(&state);
        state.get_mut_player(id).unwrap().status = em::PlayerStatus::Down;
        let after = view_of(&state);
        assert!(
            after.square(pa::Position::new(8, 3)).unwrap().tz_away
                < before.square(pa::Position::new(8, 3)).unwrap().tz_away,
            "knocking a marker down must reduce the tackle zones it exerts"
        );
    }

    /// A player pushed into the crowd sits at a genuinely out-of-bounds
    /// square for the span between `Push::do_moves` (which moves them there)
    /// and the queued `Injury::new_crowd` actually unfielding them — still
    /// "on pitch" per `get_players_on_pitch` the whole time. `tackle_zones`
    /// used to call `get_adj_positions` on every fielded player unconditionally,
    /// which panics (`!position.is_out()`) on that transient square. Found by
    /// `derive_every_state.rs`'s full-random-game fuzz test.
    #[test]
    fn a_player_mid_crowd_push_does_not_panic_the_view() {
        let mut state = a_position();
        let id = state.get_player_id_at(em::Position::new((7, 3))).unwrap();
        state.get_mut_player(id).unwrap().position = em::Position::new((0, 3));
        let v = view_of(&state); // must not panic
        assert_eq!(
            v.square(pa::Position::new(1, 3)).unwrap().tz_away,
            0,
            "a player in the crowd exerts no tackle zone"
        );
    }

    #[test]
    fn positional_actions_are_attached_to_the_squares_they_target() {
        let state = a_position();
        let v = view_of(&state);
        // Turn start: every standing Home player can be activated.
        for (x, y) in [(8, 3), (8, 4), (9, 5), (10, 2)] {
            let actions = &at(&v, x, y).actions;
            assert!(
                actions.contains(&PosAT::StartMove),
                "({x},{y}) should offer StartMove, got {actions:?}"
            );
        }
        // ...and no Away player can be.
        assert!(at(&v, 7, 3).actions.is_empty(), "{:?}", at(&v, 7, 3).actions);
        // Every offered action really is legal, and nothing legal is missing.
        let offered: usize = v.squares.iter().map(|s| s.actions.len()).sum();
        let engine_positional = state
            .get_all_actions()
            .into_iter()
            .filter(|a| matches!(a, em::Action::Positional(..)))
            .count();
        assert_eq!(offered, engine_positional);
        assert_eq!(v.to_act, Some(PTeam::Home));
    }

    #[test]
    fn simple_actions_arrive_as_buttons_with_labels() {
        let v = view_of(&a_position());
        let end_turn = v
            .simple_actions
            .iter()
            .find(|a| a.at == PSimpleAT::EndTurn)
            .expect("EndTurn is always available at a turn start");
        assert_eq!(end_turn.label, "End turn");
        assert!(end_turn.img.is_none(), "only block dice carry a face sprite");
    }

    #[test]
    fn declaring_a_move_annotates_reachable_squares_with_probability_and_route() {
        let mut state = a_position();
        state
            .step(em::Action::Positional(
                botbowl_engine::core::table::PosAT::StartMove,
                em::Position::new((10, 2)),
            ))
            .unwrap();
        let v = view_of(&state);

        let reachable: Vec<&pv::SquareView> = v.squares.iter().filter(|s| s.move_prob.is_some()).collect();
        assert!(
            reachable.len() > 5,
            "a movement-6 lineman should reach many squares, got {}",
            reachable.len()
        );
        for sq in &reachable {
            let p = sq.move_prob.unwrap();
            assert!((0.0..=1.0).contains(&p), "{:?} has probability {p}", sq.pos);
            let route = sq.route.as_ref().expect("a path action carries its route");
            assert!(
                route.steps.last() == Some(&sq.pos) || !route.rolls.is_empty(),
                "{:?}: route {:?} does not end on the square",
                sq.pos,
                route
            );
        }
        // A square right next to the activated player is a certainty; a
        // square deep in Away tackle zones is not.
        let adjacent = at(&v, 10, 3);
        assert_eq!(adjacent.move_prob, Some(1.0), "an unmarked single step cannot fail");
        assert_eq!(v.active_player, state.info.active_player);
    }

    #[test]
    fn a_block_target_carries_its_dice_count() {
        // Home blitzer-less lineman at (8,4) is adjacent to Away at (7,4).
        let mut state = a_position();
        state
            .step(em::Action::Positional(
                botbowl_engine::core::table::PosAT::StartBlock,
                em::Position::new((8, 4)),
            ))
            .unwrap();
        let v = view_of(&state);
        let target = at(&v, 7, 4);
        assert!(
            target.actions.contains(&PosAT::Block),
            "the adjacent Away player must be blockable, got {:?}",
            target.actions
        );
        let dice = target.block_dice.expect("a block target carries its dice count");
        assert!(dice.count() >= 1 && dice.count() <= 3);
    }

    /// From the moment a defender is named until the block is resolved, the
    /// view carries the pair, so the client can draw the arrow — here with a
    /// die pending on top of the `Block` procedure.
    #[test]
    fn a_block_in_progress_names_attacker_and_defender() {
        use botbowl_engine::core::gamestate::DiceMode;
        use botbowl_engine::core::model::SomeProcInput;
        let mut state = a_position();
        assert_eq!(view_of(&state).block, None, "no block at a turn start");
        state.set_dice_mode(DiceMode::RegisterRolls);
        state.step_with_roll_or_action(SomeProcInput::Action(em::Action::Positional(
            et::PosAT::StartBlock,
            em::Position::new((8, 4)),
        )));
        assert_eq!(view_of(&state).block, None, "picking a defender is not yet a block");
        state.step_with_roll_or_action(SomeProcInput::Action(em::Action::Positional(
            et::PosAT::Block,
            em::Position::new((7, 4)),
        )));
        assert!(state.pending_roll.is_some(), "the block dice are pending");
        let block = view_of(&state).block.expect("a block in progress");
        assert_eq!(block.attacker, pa::Position::new(8, 4));
        assert_eq!(block.defender, pa::Position::new(7, 4));
        assert!(block.dice.count() >= 1);
    }

    /// The active player's trail is read off the step history, with the
    /// roll-free squares the engine walked in one micro-step filled in from
    /// the pathfinder route — and the player's current square left out.
    #[test]
    fn the_trail_is_reconstructed_from_the_steps() {
        let mut state = a_position();
        let mut steps = vec![state.clone()];
        fn go(state: &mut GameState, action: em::Action, steps: &mut Vec<GameState>) {
            state.step(action).unwrap();
            steps.push(state.clone());
        }
        assert!(trail(&steps).is_empty(), "nobody is active");
        go(
            &mut state,
            em::Action::Positional(et::PosAT::StartMove, em::Position::new((10, 2))),
            &mut steps,
        );
        assert!(trail(&steps).is_empty(), "activated, not yet moved");
        // Three squares in one action: (10,2) -> (11,3) -> (12,4) -> (13,5),
        // none of them marked, so the engine walks them in one micro-step.
        go(
            &mut state,
            em::Action::Positional(et::PosAT::Move, em::Position::new((13, 5))),
            &mut steps,
        );
        assert_eq!(
            trail(&steps),
            vec![
                pa::Position::new(10, 2),
                pa::Position::new(11, 3),
                pa::Position::new(12, 4)
            ]
        );
        go(
            &mut state,
            em::Action::Positional(et::PosAT::Move, em::Position::new((13, 6))),
            &mut steps,
        );
        assert_eq!(trail(&steps).len(), 4, "one more square, {:?}", trail(&steps));
        assert_eq!(trail(&steps).last(), Some(&pa::Position::new(13, 5)));
        // Ending the activation ends the trail.
        go(&mut state, em::Action::Simple(SimpleAT::EndPlayerTurn), &mut steps);
        assert!(trail(&steps).is_empty(), "{:?}", trail(&steps));
        assert!(derive(&state, &DeriveCtx::default()).trail.is_empty());
    }

    #[test]
    fn the_scoreboard_and_dugout_come_across() {
        let state = a_position();
        let v = view_of(&state);
        assert_eq!(v.scoreboard.home_score, 0);
        assert_eq!(v.scoreboard.home_rerolls, state.home.rerolls);
        assert!(v.scoreboard.home_can_reroll);
        assert!(!v.scoreboard.game_over);
        assert_eq!(v.dugouts.len(), 2);
        assert_eq!(v.dugouts[0].team, PTeam::Home);
        assert_eq!(v.dugouts[1].team, PTeam::Away);
        assert!(!v.proc.is_empty());

        // `GameStateBuilder::build()` with explicit placements calls
        // `clear_all_players()`, which empties the *dugout* as well as the
        // pitch — so a hand-built position has no bench at all. A real game
        // starts from the coin toss with the whole roster in reserves.
        assert_eq!(v.dugouts[0].players.len(), 0, "a hand-built position has no bench");
        let real = GameStateBuilder::new()
            .with_board_dims(dims())
            .set_state(BuilderState::CoinToss)
            .build();
        let v = view_of(&real);
        let roster = dims().roster_per_team();
        assert_eq!(v.dugouts[0].count(pv::DugoutPlace::Reserves), roster);
        assert_eq!(v.dugouts[1].count(pv::DugoutPlace::Reserves), roster);
        assert!(v.dugouts[0].players.iter().all(|p| p.team == PTeam::Home));
    }

    /// Walk the coin toss by hand to reach a real setup prompt.
    ///
    /// Note `BuilderState::Setup` does *not* stop at one — it falls through to
    /// the kickoff fast-forward exactly like `Turn` does.
    fn at_setup(dims: em::BoardDims) -> GameState {
        let mut state = GameStateBuilder::new_start_of_game_with(dims);
        state.set_logging_state(false);
        state.fix_coin(botbowl_engine::core::dices::Coin::Heads);
        state.step(em::Action::Simple(SimpleAT::Heads)).unwrap();
        state.step(em::Action::Simple(SimpleAT::Kick)).unwrap();
        state
    }

    #[test]
    fn a_setup_offers_per_player_placements_and_a_summary() {
        let mut state = at_setup(dims());
        let v = view_of(&state);
        // The engine's `Setup` procedure (`kickoff_procs.rs`) asks about one
        // player at a time: `PlacePlayer` squares on the own half, plus
        // `BenchPlayer` while the roster has a spare. The formation shortcuts
        // are no longer engine actions — they are `SetupView::formations`,
        // played out by the session on request.
        let team = state.setup_team().expect("a setup is open");
        let offered = v.simple_actions.iter().map(|a| a.at).collect::<Vec<_>>();
        let placements = v
            .squares
            .iter()
            .filter(|s| s.actions.contains(&PosAT::PlacePlayer))
            .count();
        assert!(placements > 0, "a setup offers PlacePlayer squares");
        assert!(
            v.squares.iter().all(|s| !s.actions.contains(&PosAT::SelectPosition)),
            "placement is PlacePlayer, not SelectPosition"
        );
        assert!(
            offered.iter().all(|at| *at == PSimpleAT::BenchPlayer),
            "setup offers nothing but BenchPlayer, got {offered:?}"
        );
        if dims().roster_per_team() > dims().team_size {
            assert!(offered.contains(&PSimpleAT::BenchPlayer), "a spare player may sit out");
        }
        assert!(
            v.active_player.is_some(),
            "the player being placed is the active player"
        );
        assert_eq!(v.active_player, state.info.active_player);

        let setup = v.setup.clone().expect("a setup summary while placing");
        assert_eq!(setup.team, mirror::team_to_proto(team));
        assert_eq!(setup.placed, 0, "nobody placed yet");
        assert!(setup.waiting > 0, "the staged players are waiting");
        assert_eq!(setup.team_size, dims().team_size);
        assert!(setup.formations.contains(&"Line".to_string()), "{:?}", setup.formations);
        assert_eq!(
            setup.formations,
            Formation::available(&dims())
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
        );

        // Place one where the engine offers it: the summary moves along.
        let pos = v
            .squares
            .iter()
            .find(|s| s.actions.contains(&PosAT::PlacePlayer))
            .map(|s| s.pos)
            .unwrap();
        state
            .step(em::Action::Positional(
                et::PosAT::PlacePlayer,
                em::Position::new((pos.x, pos.y)),
            ))
            .unwrap();
        let v = view_of(&state);
        let after = v.setup.clone().expect("still setting up");
        assert_eq!(after.placed, 1);
        assert_eq!(after.waiting, setup.waiting - 1);
        assert!(
            v.squares
                .iter()
                .filter(|s| s.player.as_ref().is_some_and(|p| !p.used))
                .count()
                >= 1,
            "a placed player is no longer staged"
        );
    }

    /// Was a pinned finding, now a regression guard: the auto-setups used to
    /// clamp hard-coded offsets, which on any board smaller than the compiled
    /// default put the front rank on the line-of-scrimmage *column* but at `y`
    /// values outside `los_y_range` — `line_of_scrimage` counted 0 against a
    /// required 3, so the engine's own formation failed the engine's own
    /// `is_setup_legal` at 16x9/4, 18x11/6 and 22x11/8.
    ///
    /// The formations are built from board-relative anchors (`Formation` in
    /// `kickoff_procs.rs`) and open with three LOS slots, so every formation
    /// the view offers plays out to a legal setup on every board.
    #[test]
    fn every_auto_setup_formation_is_legal_on_clamped_boards() {
        let formations: Vec<Formation> = Formation::available(&dims()).collect();
        assert!(!formations.is_empty());
        for formation in formations {
            let mut small = at_setup(dims());
            let team = small.setup_team().unwrap();
            auto_setup(&mut small, formation);
            assert_ne!(small.setup_team(), Some(team), "{formation:?} finishes the setup");
            let los_x = small.get_line_of_scrimage_x(team);
            let los_y = small.board_dims.los_y_range();
            let on_scrimmage = small
                .get_players_on_pitch_in_team(team)
                .filter(|p| p.position.x == los_x && los_y.contains(&p.position.y))
                .count();
            assert!(
                on_scrimmage >= 3,
                "{formation:?} puts {on_scrimmage} players on the LOS rows"
            );
            assert!(small.is_setup_legal(team), "{formation:?} is an illegal setup");
            assert_eq!(
                small.get_players_on_pitch_in_team(team).count(),
                dims().team_size,
                "{formation:?} fields a full team"
            );
        }
    }

    /// The board paints the tackle zones of whoever is *not* moving, so the
    /// same overlay reads correctly whether it is your turn or the bot's.
    #[test]
    fn the_threatened_team_is_the_movers_opponent() {
        let state = a_position();
        let v = view_of(&state);
        assert_eq!(v.mover(), PTeam::Home, "Home's turn, Home is being asked");
        assert!(v.moves_offered(), "turn start offers StartMove");
        assert_eq!(
            v.threat_team(),
            Some(PTeam::Away),
            "while Home moves, Away's zones are the ones that matter"
        );
        // Which is exactly the count the squares already carry.
        assert_eq!(at(&v, 8, 3).tz(v.threat_team().unwrap()), 2);

        // Away to move — the bot's turn, from the same board: the overlay
        // flips sides rather than staying pinned to whichever team the
        // browser plays.
        let mut bot_turn = v.clone();
        bot_turn.to_act = Some(PTeam::Away);
        assert_eq!(bot_turn.threat_team(), Some(PTeam::Home));

        // `to_act` wins over `team_turn`, because it does not always agree
        // with it — an uphill block's dice are picked by the defender.
        let mut mid_procedure = v.clone();
        mid_procedure.to_act = None;
        mid_procedure.scoreboard.team_turn = PTeam::Away;
        assert_eq!(mid_procedure.threat_team(), Some(PTeam::Home));
    }

    /// Nothing to paint when nobody is choosing where to put a player: a
    /// kickoff, a dice prompt, or after the whistle.
    #[test]
    fn no_tackle_zones_when_no_move_is_on_offer() {
        let mut over = a_position();
        over.info.game_over = true;
        assert_eq!(view_of(&over).threat_team(), None);

        let mut state = a_position();
        state.info.game_over = false;
        let v = view_of(&state);
        assert!(v.moves_offered());
        // Strip the offers the way a mid-procedure state does, and the layer
        // goes away with them.
        let mut bare = v.clone();
        for sq in bare.squares.iter_mut() {
            sq.actions.clear();
        }
        assert!(!bare.moves_offered());
        assert_eq!(bare.threat_team(), None);
    }

    #[test]
    fn a_finished_game_offers_nothing_even_though_the_engine_still_does() {
        let mut state = a_position();
        state.info.game_over = true;
        let v = view_of(&state);
        assert!(v.to_act.is_none(), "nobody acts after the whistle");
        assert!(v.simple_actions.is_empty());
        assert!(v.squares.iter().all(|s| s.actions.is_empty()));
        assert!(v.setup.is_none());
    }

    #[test]
    fn the_view_is_a_pure_function_of_the_state() {
        // Two derivations of the same state must agree, and deriving must not
        // change the state (the engine's path buffer is `take`-able, so this
        // is a real hazard).
        let state = a_position();
        let a = view_of(&state);
        let b = view_of(&state);
        assert_eq!(a, b);
    }

    fn intent_at(sel: &pv::SelectionView, x: i8, y: i8) -> Option<&pv::IntentView> {
        sel.target(pa::Position::new(x, y))
    }

    #[test]
    fn a_selected_player_moves_onto_empty_squares_and_the_loose_ball() {
        let state = a_position();
        let sel = selection(&state, pa::Position::new(10, 2), 7).expect("a Home player at turn start");
        assert_eq!(sel.seq, 7);
        assert!(sel.starts.contains(&PosAT::StartMove) && sel.starts.contains(&PosAT::StartBlitz));
        let walk = intent_at(&sel, 11, 2).expect("an empty neighbour square");
        assert_eq!((walk.start, walk.action), (PosAT::StartMove, PosAT::Move));
        assert!(walk.prob.is_some() && walk.route.is_some());
        let ball = intent_at(&sel, 8, 5).expect("the loose ball's square");
        assert_eq!((ball.start, ball.action), (PosAT::StartMove, PosAT::Move));
        assert!(
            ball.route.as_ref().unwrap().rolls.iter().any(|r| r.starts_with("Pickup")),
            "{:?}",
            ball.route
        );
        // Previewing changes nothing.
        assert_eq!(view_of(&state), view_of(&a_position()));
    }

    #[test]
    fn an_adjacent_opponent_is_a_block_and_a_distant_one_a_blitz() {
        let state = a_position();
        let sel = selection(&state, pa::Position::new(8, 3), 0).unwrap();
        let block = intent_at(&sel, 7, 3).expect("adjacent Away (7,3)");
        assert_eq!((block.start, block.action), (PosAT::StartBlock, PosAT::Block));
        assert!(block.block_dice.is_some());
        let blitz = intent_at(&sel, 6, 5).expect("Away (6,5) is two squares away");
        assert_eq!((blitz.start, blitz.action), (PosAT::StartBlitz, PosAT::Block));
        assert!(blitz.block_dice.is_some());
        // A team-mate, from a player without the ball, is not a target.
        assert!(intent_at(&sel, 8, 4).is_none());
    }

    #[test]
    fn a_prone_opponent_is_a_foul() {
        let mut state = a_position();
        let id = state.get_player_id_at(em::Position::new((6, 5))).unwrap();
        state.get_mut_player(id).unwrap().status = em::PlayerStatus::Down;
        let sel = selection(&state, pa::Position::new(9, 5), 0).unwrap();
        let foul = intent_at(&sel, 6, 5).expect("the prone Away player");
        assert_eq!((foul.start, foul.action), (PosAT::StartFoul, PosAT::Foul));
    }

    #[test]
    fn the_ball_carrier_hands_off_or_passes_to_a_team_mate_whichever_is_likelier() {
        let state = GameStateBuilder::new()
            .with_board_dims(dims())
            .add_home_players(&[(10, 4), (11, 4), (12, 7)])
            .add_away_players(&[(2, 1)])
            .add_ball((10, 4))
            .build();
        let sel = selection(&state, pa::Position::new(10, 4), 0).unwrap();
        for (x, y) in [(11, 4), (12, 7)] {
            let i = intent_at(&sel, x, y).unwrap_or_else(|| panic!("team-mate ({x},{y})"));
            assert!(
                matches!(
                    (i.start, i.action),
                    (PosAT::StartHandoff, PosAT::Handoff) | (PosAT::StartPass, PosAT::Pass)
                ),
                "{i:?}"
            );
        }
        // The adjacent team-mate: a handoff never needs a pass roll on top.
        assert_eq!(intent_at(&sel, 11, 4).unwrap().action, PosAT::Handoff);
    }

    #[test]
    fn mid_activation_a_team_mate_is_selected_by_ending_the_active_player_first() {
        let mut state = a_position();
        let mover = em::Position::new((10, 2));
        state.step_positional(et::PosAT::StartMove, mover);
        state.step_positional(et::PosAT::Move, em::Position::new((11, 2)));
        let v = view_of(&state);
        assert!(v.simple_actions.iter().any(|a| a.at == PSimpleAT::EndPlayerTurn));
        // Nobody else is declarable right now...
        assert!(at(&v, 9, 5).actions.is_empty());
        // ...but every other Home player is selectable, the active one is not.
        for (x, y) in [(8, 3), (8, 4), (9, 5)] {
            assert!(v.reselect.contains(&pa::Position::new(x, y)), "({x},{y}) in {:?}", v.reselect);
        }
        assert!(!v.reselect.contains(&pa::Position::new(11, 2)));

        let sel = selection(&state, pa::Position::new(9, 5), 0).expect("selectable after ending the mover");
        assert!(sel.end_first);
        let walk = intent_at(&sel, 10, 5).expect("an empty square next to (9,5)");
        assert_eq!(
            sel.chain([
                pa::Action::Positional(walk.start, sel.player),
                pa::Action::Positional(walk.action, walk.pos),
            ])[0],
            pa::Action::Simple(PSimpleAT::EndPlayerTurn)
        );
        // At a turn start there is nothing to end.
        assert!(!selection(&a_position(), pa::Position::new(9, 5), 0).unwrap().end_first);
        assert!(view_of(&a_position()).reselect.is_empty());
    }

    #[test]
    fn only_the_side_to_act_can_be_selected() {
        let state = a_position();
        assert!(selection(&state, pa::Position::new(7, 3), 0).is_none(), "an Away player");
        assert!(selection(&state, pa::Position::new(12, 6), 0).is_none(), "an empty square");
    }

    #[test]
    fn a_reroll_prompt_sits_over_the_player_who_rolled() {
        let start = em::Position::new((12, 4));
        let mut state = GameStateBuilder::new()
            .with_board_dims(dims())
            .add_home_player(start)
            .add_away_player(em::Position::new((2, 7)))
            .build();
        state.get_mut_team(em::TeamType::Home).rerolls = 2;
        assert!(view_of(&state).prompt.is_none());
        state.step_positional(et::PosAT::StartMove, start);
        state.fix_d6(1); // fail the GFI
        state.step_positional(et::PosAT::Move, em::Position::new((5, 4)));
        let v = view_of(&state);
        assert!(v.simple_actions.iter().any(|a| a.at == PSimpleAT::UseReroll));
        let prompt = v.prompt.expect("a reroll question");
        assert_eq!(prompt.title, "GFI");
        // The player stands on the square where the GFI was rolled.
        let player = v.squares.iter().find(|s| s.player.is_some()).unwrap();
        assert_eq!(prompt.pos, player.pos);
    }

    #[test]
    fn block_dice_are_asked_over_the_defender() {
        let mut state = a_position();
        state.get_mut_team(em::TeamType::Home).rerolls = 1;
        state.step_positional(et::PosAT::StartBlock, em::Position::new((8, 3)));
        // ST 3 against ST 3 with one assist each side: one die.
        state.fix_blockdice(botbowl_engine::core::dices::BlockDice::Push);
        state.step_positional(et::PosAT::Block, em::Position::new((7, 3)));
        let v = view_of(&state);
        assert!(
            v.simple_actions.iter().any(|a| a.at == PSimpleAT::SelectPush),
            "{:?}",
            v.simple_actions
        );
        assert_eq!(v.prompt.expect("a dice question").pos, pa::Position::new(7, 3));
    }
}

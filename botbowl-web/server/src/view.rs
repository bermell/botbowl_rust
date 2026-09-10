//! `GameState -> ViewState`: the one place game logic turns into something the
//! client can draw.
//!
//! Decision 3 of plan 034 puts *all* of it here, server-side and pure, so the
//! wasm build stays trivial and this is unit-testable against
//! `GameStateBuilder` positions. The client never computes geometry, legality
//! or probability — it renders what arrives.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model as em;
use botbowl_engine::core::model::{BallState, Position};
use botbowl_engine::core::pathing::{CustomIntoIter, PathingEvent, PositionOrEvent};
use botbowl_engine::core::table as et;
use botbowl_web_proto::view as pv;

use crate::mirror;

/// Session facts that are not in the `GameState` but belong on the view.
///
/// `log_tail` comes from the *session's* own event log, not
/// `GameState::get_log()`: engine logging both pushes to a Vec and `println!`s
/// every micro-step (`STEPPING: <proc> action=...`), which would flood the
/// server's stdout and is a rules-engine trace rather than anything a player
/// wants to read. The session keeps a short human-facing log instead.
pub struct DeriveCtx {
    pub human: em::TeamType,
    pub seq: u64,
    pub can_undo: bool,
    pub bot_thinking: bool,
    pub log_tail: Vec<String>,
}

impl Default for DeriveCtx {
    fn default() -> Self {
        DeriveCtx {
            human: em::TeamType::Home,
            seq: 0,
            can_undo: false,
            bot_thinking: false,
            log_tail: Vec::new(),
        }
    }
}

/// How a pathfinder roll reads in the move-probability tooltip.
fn event_label(event: &PathingEvent) -> String {
    match event {
        PathingEvent::Dodge(t) => format!("Dodge {}+", *t as u8),
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

fn player_view(state: &GameState, p: &em::FieldedPlayer, has_ball: bool) -> pv::PlayerView {
    let team = p.stats.team;
    let role = mirror::role_to_proto(p.stats.role);
    let mut skills: Vec<String> = [
        et::Skill::Block,
        et::Skill::Dodge,
        et::Skill::Catch,
        et::Skill::Throw,
        et::Skill::SureHands,
        et::Skill::SureFeet,
    ]
    .into_iter()
    .filter(|s| p.has_skill(*s))
    .map(|s| mirror::skill_label(s).to_string())
    .collect();
    skills.sort();

    pv::PlayerView {
        id: p.id,
        team: mirror::team_to_proto(team),
        role,
        status: mirror::status_to_proto(p.status),
        used: p.used,
        sprite: role.sprite(mirror::team_to_proto(team), p.used),
        st: p.stats.str_,
        ma: p.stats.ma,
        ag: p.stats.ag,
        av: p.stats.av,
        movement_left: p.total_movement_left(),
        has_ball,
        skills,
        active: state.info.active_player == Some(p.id),
    }
}

/// Tackle zones exerted on every square by each team. Computed square-first
/// (rather than via `get_tz_on`, which needs a player id) so the overlay also
/// covers empty squares — which is the whole point of a tackle-zone overlay.
fn tackle_zones(state: &GameState, squares: &mut [pv::SquareView]) {
    let dims = state.board_dims;
    for p in state.get_players_on_pitch() {
        if !p.has_tackle_zone() {
            continue;
        }
        for adj in state.get_adj_positions(p.position) {
            if dims.is_out(adj) {
                continue;
            }
            let sq = &mut squares[index_of(dims, adj)];
            match p.stats.team {
                em::TeamType::Home => sq.tz_home += 1,
                em::TeamType::Away => sq.tz_away += 1,
            }
        }
    }
}

fn index_of(dims: em::BoardDims, pos: Position) -> usize {
    pos.y as usize * dims.width as usize + pos.x as usize
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
            let sq = &mut squares[index_of(dims, pos)];
            sq.move_prob = Some(node.prob);
            sq.block_dice = node.get_block_dice().map(mirror::num_block_dices_to_proto);

            let mut steps = Vec::new();
            let mut rolls = Vec::new();
            for item in node.iter() {
                match item {
                    PositionOrEvent::Position(p) => steps.push(mirror::position_to_proto(p)),
                    PositionOrEvent::Event(e) => rolls.push(event_label(&e)),
                }
            }
            sq.route = Some(pv::RouteView { steps, rolls });
        }
    }
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

fn dugout(state: &GameState, team: em::TeamType) -> pv::DugoutView {
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
                sprite: role.sprite(mirror::team_to_proto(team), false),
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
        let idx = index_of(dims, p.position);
        squares[idx].player = Some(player_view(state, p, carrier == Some(p.id)));
    }

    if let Some(pos) = ball_pos {
        let idx = index_of(dims, pos);
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

    // The game-over procedure still offers actions (`EndSetup`/`DontUseReroll`
    // to Home) — check `game_over` first or the UI invites a click that ends
    // the session.
    let game_over = state.info.game_over;
    let mut simple_actions: Vec<pv::SimpleActionView> = Vec::new();

    if !game_over {
        for action in state.get_all_actions() {
            match action {
                em::Action::Positional(at, pos) => {
                    let sq = &mut squares[index_of(dims, pos)];
                    sq.actions.push(mirror::pos_at_to_proto(at));
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

    // Only meaningful while a setup is open. Gated on *either* setup action
    // being offered, not just `EndSetup`: the engine withholds `EndSetup`
    // until the placement is already legal, so gating on it alone would only
    // ever report `true` and the UI could never say "not legal yet".
    let in_setup = {
        let simple = state.get_available_actions().get_simple();
        simple.contains(&et::SimpleAT::EndSetup) || simple.contains(&et::SimpleAT::SetupLine)
    };
    let setup_legal = state
        .get_active_teamtype()
        .filter(|_| !game_over && in_setup)
        .map(|team| state.is_setup_legal(team));

    pv::ViewState {
        seq: ctx.seq,
        dims: pv::Dims {
            width: dims.width,
            height: dims.height,
            team_size: dims.team_size,
        },
        scoreboard: scoreboard(state),
        squares,
        dugouts: vec![dugout(state, em::TeamType::Home), dugout(state, em::TeamType::Away)],
        simple_actions,
        to_act,
        human: mirror::team_to_proto(ctx.human),
        proc: state.proc_stack_top().unwrap_or("-").to_string(),
        active_player: state.info.active_player,
        pending_roll: state.pending_roll.map(mirror::requested_roll_to_proto),
        log_tail: ctx.log_tail.clone(),
        can_undo: ctx.can_undo,
        setup_legal,
        bot_thinking: ctx.bot_thinking,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::{BuilderState, GameStateBuilder};
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
    fn a_setup_offers_the_engines_two_setup_actions_and_nothing_else() {
        let mut state = at_setup(dims());
        let v = view_of(&state);
        // The engine's `Setup` procedure (`kickoff_procs.rs`) offers *only*
        // `SetupLine`, then `EndSetup` — there is no per-square placement
        // action, so manual setup is not something a UI can expose today.
        // `is_setup_legal` exists but nothing in the engine enforces it.
        assert_eq!(
            v.simple_actions.iter().map(|a| a.at).collect::<Vec<_>>(),
            vec![PSimpleAT::SetupLine],
            "an empty setup should offer only the auto-setup shortcut"
        );
        assert!(
            v.squares.iter().all(|s| !s.actions.contains(&PosAT::SelectPosition)),
            "the engine does not offer per-square placement during setup"
        );
        assert_eq!(v.setup_legal, Some(false), "an empty pitch is not a legal setup");

        state.step(em::Action::Simple(SimpleAT::SetupLine)).unwrap();
        let v = view_of(&state);
        assert_eq!(
            v.simple_actions.iter().map(|a| a.at).collect::<Vec<_>>(),
            vec![PSimpleAT::EndSetup]
        );
        assert!(
            v.squares.iter().filter(|s| s.player.is_some()).count() > 0,
            "SetupLine fields players"
        );
    }

    /// A finding, pinned here rather than fixed: on any board smaller than the
    /// compiled default, the engine's own `SetupLine` formation fails the
    /// engine's own `is_setup_legal`.
    ///
    /// The clamped formation offsets put the front rank on the line-of-
    /// scrimmage *column* but at `y` values outside `los_y_range`, so
    /// `line_of_scrimage` counts 0 against a required 3. Measured at 16x9/4,
    /// 18x11/6 and 22x11/8; legal at the default 28x17/11.
    ///
    /// Nothing in the engine enforces `is_setup_legal`, so this has never
    /// affected play — but it *is* the formation every 14x7 model was trained
    /// against, so changing `SetupLine` would invalidate the trained nets and
    /// every measured result in `plans/032`. It belongs in the experiment
    /// queue, not in a UI change.
    #[test]
    fn the_auto_setup_formation_is_illegal_on_clamped_boards() {
        let mut small = at_setup(dims());
        small.step(em::Action::Simple(SimpleAT::SetupLine)).unwrap();
        let team = small.get_active_teamtype().unwrap();
        let los_x = small.get_line_of_scrimage_x(team);
        let los_y = small.board_dims.los_y_range();
        let on_scrimmage = small
            .get_players_on_pitch_in_team(team)
            .filter(|p| p.position.x == los_x && los_y.contains(&p.position.y))
            .count();
        assert_eq!(
            on_scrimmage, 0,
            "the clamped formation reaches the LOS column but not its rows"
        );
        assert_eq!(view_of(&small).setup_legal, Some(false));

        // The default board is fine, which is why this went unnoticed.
        let capacity = em::BoardDims::default();
        if capacity.width >= 28 && capacity.height >= 17 && capacity.team_size >= 11 {
            let mut big = at_setup(capacity);
            big.step(em::Action::Simple(SimpleAT::SetupLine)).unwrap();
            assert_eq!(view_of(&big).setup_legal, Some(true));
        }
    }

    #[test]
    fn a_finished_game_offers_nothing_even_though_the_engine_still_does() {
        let mut state = a_position();
        state.info.game_over = true;
        let v = view_of(&state);
        assert!(v.to_act.is_none(), "nobody acts after the whistle");
        assert!(v.simple_actions.is_empty());
        assert!(v.squares.iter().all(|s| s.actions.is_empty()));
        assert!(v.setup_legal.is_none());
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
}

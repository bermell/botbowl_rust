use super::*;
use crate::core::dices::BlockDice;
use crate::core::gamestate::GameStateBuilder;
use crate::core::model::{Direction, PlayerStats};
use crate::core::table::NumBlockDices;

/// A full-pitch kickoff (Away kicks, Home receives) that rolls `event` on the table, deviating
/// onto an empty receiving-half square whose square above is empty too: a landing bounce rolled
/// `up` (`land`) settles on the ground with no catch. `dice` queues the event's own dice.
fn kickoff_rolling(event: u8, dice: impl FnOnce(&mut GameState)) -> GameState {
    let mut state = GameStateBuilder::new_at_kickoff();
    assert_eq!(state.info.kicking_this_drive, TeamType::Away);
    let aim = state.get_best_kickoff_aim_for(TeamType::Away);
    let clear =
        |p: Position| !state.is_out(p) && state.is_on_team_side(p, TeamType::Home) && state.get_player_at(p).is_none();
    let (dir, len) = Direction::all_directions_iter()
        .flat_map(|&d| (1..=6i8).map(move |l| (d, l)))
        .find(|&(d, l)| clear(aim + d * l) && clear(aim + d * l + Direction::up()))
        .expect("an empty landing square");
    state.fix_d8_direction(dir);
    state.fix_d6(len as u8);
    let first = if event > 7 { 6 } else { 1 };
    state.fix_d6(first);
    state.fix_d6(event - first);
    dice(&mut state);
    state.step_simple(SimpleAT::KickoffAimMiddle);
    state
}

/// The landing bounce of a `kickoff_rolling` ball: one square up, onto an empty square.
fn land(state: &mut GameState) {
    state.fix_d8_direction(Direction::up());
}

fn selectable(state: &GameState, at: PosAT) -> Vec<Position> {
    state
        .get_all_actions()
        .into_iter()
        .filter_map(|a| match a {
            Action::Positional(a, pos) if a == at => Some(pos),
            _ => None,
        })
        .collect()
}

fn sorted(mut v: Vec<Position>) -> Vec<Position> {
    v.sort();
    v
}

fn open_positions(state: &GameState, team: TeamType) -> Vec<Position> {
    sorted(positions_of(state, &open_players(state, team)))
}

#[test]
fn d3_reads_a_d6_in_pairs() {
    let faces = [D6::One, D6::Two, D6::Three, D6::Four, D6::Five, D6::Six];
    assert_eq!(faces.map(d3), [1, 1, 2, 2, 3, 3]);
}

/// S3 Kick-off Event 4: "The Coach of the kicking team selects up to D3+3 Open players on their
/// team. The selected players are then removed from the pitch and can be set up again."
#[test]
fn solid_defence_sets_up_again_the_open_kicking_players_it_picks() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(4, |s| s.fix_d6(1)); // D3 = 1: up to four players
    assert!(state.away_to_act());
    let offered = sorted(selectable(&state, PosAT::SelectPosition));
    assert_eq!(offered, open_positions(&state, TeamType::Away));
    assert!(!offered.is_empty());
    // The line is Marked by the receivers: not Open, not offered.
    let los_x = state.board_dims.los_x(TeamType::Away);
    assert!(offered.iter().all(|p| p.x != los_x), "{offered:?}");

    let from = offered[0];
    let id = state.get_player_id_at(from).unwrap();
    state.step_positional(PosAT::SelectPosition, from);
    assert_eq!(state.get_player_id_at(from), None, "lifted off its square");
    state.step_simple(SimpleAT::EndPlayerTurn);

    // `Setup` asks about the lifted player alone, and offers its old square back.
    assert!(state.away_to_act());
    assert_eq!(state.info.active_player, Some(id));
    let squares = selectable(&state, PosAT::PlacePlayer);
    assert!(squares.contains(&from));
    let to = *squares
        .iter()
        .find(|&&p| p != from && state.get_player_at(p).is_none())
        .unwrap();
    land(&mut state);
    state.step_positional(PosAT::PlacePlayer, to);
    assert_eq!(state.get_player_id_at(to), Some(id));
    assert!(state.is_setup_legal(TeamType::Away));
    assert!(
        state.home_to_act(),
        "the ball has landed and the receivers' turn starts"
    );
}

/// The fourth pick of a D3 = 1 ends the selection and hands the picked to `Setup`.
#[test]
fn solid_defence_stops_at_its_d3_plus_3() {
    let dims = GameStateBuilder::new().build().board_dims;
    let x = dims.los_x(TeamType::Away) + 3;
    let mut builder = GameStateBuilder::new();
    for y in 1..=6 {
        builder.add_away_player(Position::new((x, y)));
    }
    let mut state = builder.build();
    state.info.kicking_this_drive = TeamType::Away;
    let open = open_positions(&state, TeamType::Away);
    assert_eq!(open.len(), 6);
    let AnyProc::SolidDefence(mut sd) = SolidDefence::new() else {
        unreachable!()
    };
    sd.step(&mut state, ProcInput::Nothing);
    let mut last = sd.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Two)));
    for pos in &open[..4] {
        assert!(matches!(last, ProcState::NeedAction(_)));
        last = sd.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, *pos)),
        );
    }
    let ProcState::DoneNew(AnyProc::Setup(setup)) = last else {
        panic!("expected the re-setup, got {last:?}")
    };
    assert_eq!(setup.queue().len(), 4);
    assert_eq!(setup.placed(), 2, "the two not picked stay where they are");
}

/// S3 Kick-off Event 5: "One Open player on the receiving team may immediately be placed in the
/// square the ball is going to land in."
#[test]
fn high_kick_puts_an_open_receiver_under_the_ball() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(5, |_| {});
    let BallState::InAir(target) = state.ball else {
        panic!("ball should be in the air")
    };
    assert!(state.home_to_act());
    let offered = sorted(selectable(&state, PosAT::SelectPosition));
    assert_eq!(offered, open_positions(&state, TeamType::Home));
    assert!(state
        .get_all_actions()
        .contains(&Action::Simple(SimpleAT::EndPlayerTurn)));

    let id = state.get_player_id_at(offered[0]).unwrap();
    state.fix_d6(6); // the catch
    state.step_positional(PosAT::SelectPosition, offered[0]);
    assert_eq!(state.get_player_id_at(target), Some(id));
    assert_eq!(state.ball, BallState::Carried(id), "caught on a 6");
}

#[test]
fn a_declined_high_kick_lands_the_ball_where_it_was() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(5, |_| {});
    let BallState::InAir(target) = state.ball else {
        panic!("ball should be in the air")
    };
    assert!(
        state.get_player_at(target).is_none(),
        "test assumes an empty landing square"
    );
    let before: Vec<Position> = state.get_players_on_pitch().map(|p| p.position).collect();
    land(&mut state);
    state.step_simple(SimpleAT::EndPlayerTurn);
    let after: Vec<Position> = state.get_players_on_pitch().map(|p| p.position).collect();
    assert_eq!(before, after);
    assert_eq!(state.ball, BallState::OnGround(target + Direction::up()), "bounced");
}

/// High Kick run directly: Away kicks to `target`, Home receives with an Open player at `r`.
fn high_kick_at(target: Position, extra: impl FnOnce(&mut GameStateBuilder)) -> (GameState, ProcState) {
    let dims = GameStateBuilder::new().build().board_dims;
    let r = Position::new((dims.los_x(TeamType::Home) + 3, dims.height / 2));
    let mut builder = GameStateBuilder::new();
    builder.add_home_player(r);
    extra(&mut builder);
    let mut state = builder.build();
    state.info.kicking_this_drive = TeamType::Away;
    state.set_ball(BallState::InAir(target));
    let AnyProc::HighKick(mut kick) = HighKick::new() else {
        unreachable!()
    };
    let out = kick.step(&mut state, ProcInput::Nothing);
    (state, out)
}

/// No High Kick onto an occupied square, into the kicking half or off the pitch, and none with
/// no Open receiver.
#[test]
fn high_kick_needs_an_empty_landing_square_in_the_receiving_half_and_an_open_receiver() {
    let dims = GameStateBuilder::new().build().board_dims;
    let home_half = Position::new((dims.los_x(TeamType::Home) + 6, 2));
    let (_, out) = high_kick_at(home_half, |_| {});
    assert!(matches!(out, ProcState::NeedAction(_)), "the baseline is offered");

    let (_, out) = high_kick_at(home_half, |b| {
        b.add_home_player(home_half);
    });
    assert!(matches!(out, ProcState::Done), "occupied");
    let away_half = Position::new((dims.los_x(TeamType::Away) - 2, 2));
    let (_, out) = high_kick_at(away_half, |_| {});
    assert!(matches!(out, ProcState::Done), "kicking half");
    let (_, out) = high_kick_at(Position::new((home_half.x, 0)), |_| {});
    assert!(matches!(out, ProcState::Done), "off the pitch: a touchback");

    // The only receiver Marked.
    let r = Position::new((dims.los_x(TeamType::Home) + 3, dims.height / 2));
    let (_, out) = high_kick_at(home_half, |b| {
        b.add_away_player(r + (1, 0));
    });
    assert!(matches!(out, ProcState::Done), "no Open receiver");
}

#[test]
fn high_kick_offers_open_receivers_only() {
    let dims = GameStateBuilder::new().build().board_dims;
    let r = Position::new((dims.los_x(TeamType::Home) + 3, dims.height / 2));
    let free = Position::new((r.x, 1));
    let target = Position::new((dims.los_x(TeamType::Home) + 6, 2));
    let (_, out) = high_kick_at(target, |b| {
        b.add_home_player(free);
        b.add_away_player(r + (-1, 0)); // marks r
    });
    let ProcState::NeedAction(aa) = out else {
        panic!("expected an offer")
    };
    let mut state = GameStateBuilder::new().build();
    state.available_actions = aa;
    assert!(state.is_legal_action(&Action::Positional(PosAT::SelectPosition, free)));
    assert!(!state.is_legal_action(&Action::Positional(PosAT::SelectPosition, r)));
}

/// S3 Kick-off Event 6: D6 + Cheerleaders; the highest roll's first Block Action next turn gets
/// an extra Offensive Assist.
#[test]
fn cheering_fans_go_to_the_higher_roll_plus_cheerleaders() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(6, |s| {
        s.get_mut_team(TeamType::Away).cheerleaders = 2;
        s.fix_d6(4); // Home
        s.fix_d6(3); // Away + 2 = 5
        land(s);
    });
    assert!(state.home_to_act());
    assert!(state.away.cheering_fans, "kept for the kicking team's next turn");
    assert!(!state.home.cheering_fans);
    assert!(!state.info.cheering_fans_this_turn, "not Home's");
}

#[test]
fn tied_cheering_fans_go_to_both_and_the_receivers_get_them_this_turn() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(6, |s| {
        s.fix_d6(3);
        s.fix_d6(3);
        land(s);
    });
    assert!(state.home_to_act());
    assert!(state.info.cheering_fans_this_turn, "Home's turn has started with it");
    assert!(!state.home.cheering_fans, "moved into the turn");
    assert!(state.away.cheering_fans);
}

/// Two 3-vs-3 pairs with no other help: one block die each, two with the Cheering Fans assist.
fn two_even_blocks() -> (GameState, [Position; 4]) {
    let a = Position::new((5, 5));
    let d = a + (1, 0);
    let a2 = Position::new((10, 5));
    let d2 = a2 + (1, 0);
    let mut state = GameStateBuilder::new()
        .add_home_player(a)
        .add_away_player(d)
        .add_home_player(a2)
        .add_away_player(d2)
        .build();
    state.info.cheering_fans_this_turn = true;
    (state, [a, d, a2, d2])
}

/// The dice of the `Block` being rolled.
fn rolled_block_dice(state: &GameState) -> NumBlockDices {
    state
        .proc_stack_iter()
        .find_map(|p| match p {
            AnyProc::Block(b) => Some(b.num_dices()),
            _ => None,
        })
        .expect("a Block on the stack")
}

/// Push `d` straight back with the rolled Push and follow up.
fn push_and_follow(state: &mut GameState, d: Position) {
    state.step_simple(SimpleAT::SelectPush);
    state.step_positional(PosAT::Push, d + (1, 0));
    state.step_positional(PosAT::FollowUp, d);
}

/// The extra assist: a 3-vs-3 block rolls two dice instead of one, on the turn's first Block
/// Action performed only.
#[test]
fn the_cheering_fans_assist_goes_to_the_first_block_performed_only() {
    let (mut state, [a, d, a2, d2]) = two_even_blocks();
    state.step_positional(PosAT::StartBlock, a);
    state.fix_blockdice(BlockDice::Push);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, d);
    assert_eq!(rolled_block_dice(&state), NumBlockDices::Two);
    push_and_follow(&mut state, d);
    if state.is_legal_action(&Action::Simple(SimpleAT::EndPlayerTurn)) {
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    state.step_positional(PosAT::StartBlock, a2);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, d2);
    assert_eq!(rolled_block_dice(&state), NumBlockDices::One);
}

/// A Block Action declared and given up before any block is rolled keeps the assist for the next.
#[test]
fn an_abandoned_block_action_keeps_the_cheering_fans_assist() {
    let (mut state, [a, _, a2, d2]) = two_even_blocks();
    state.step_positional(PosAT::StartBlock, a);
    state.step_simple(SimpleAT::EndPlayerTurn);
    state.step_positional(PosAT::StartBlock, a2);
    state.fix_blockdice(BlockDice::Push);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, d2);
    assert_eq!(rolled_block_dice(&state), NumBlockDices::Two);
}

/// S3, Declaring vs Performing: a rule triggering on a Block Action also triggers "when the Block
/// Action is performed as part of the Blitz Action".
#[test]
fn a_blitz_s_block_gets_the_cheering_fans_assist() {
    let (mut state, [a, d, ..]) = two_even_blocks();
    state.step_positional(PosAT::StartBlitz, a);
    state.fix_blockdice(BlockDice::Push);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, d);
    assert_eq!(rolled_block_dice(&state), NumBlockDices::Two);
    assert!(!state.info.cheering_fans_this_turn, "spent");
}

/// Both blocks of a Frenzy belong to the one Block Action.
#[test]
fn a_frenzy_second_block_keeps_the_cheering_fans_assist() {
    let (mut state, [a, d, ..]) = two_even_blocks();
    let id = state.get_player_id_at(a).unwrap();
    state.get_mut_player_unsafe(id).stats.give_skill(Skill::Frenzy);
    state.step_positional(PosAT::StartBlock, a);
    state.fix_blockdice(BlockDice::Push);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, d);
    state.step_simple(SimpleAT::SelectPush);
    state.step_positional(PosAT::Push, d + (1, 0));
    state.fix_blockdice(BlockDice::Push);
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::FollowUp, d);
    assert_eq!(state.proc_stack_top(), Some("Block"), "the second block's dice are up");
    assert_eq!(rolled_block_dice(&state), NumBlockDices::Two);
}

/// S3 Kick-off Event 7: the highest D6 + Assistant Coaches gains a free team re-roll for the
/// drive; a tie gives one to both.
#[test]
fn brilliant_coaching_gives_the_winner_a_re_roll_for_the_drive() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(7, |s| {
        s.get_mut_team(TeamType::Home).assistant_coaches = 1;
        s.fix_d6(3); // Home + 1 = 4
        s.fix_d6(4); // Away
        land(s);
    });
    assert_eq!((state.home.drive_rerolls, state.away.drive_rerolls), (1, 1));
    let state = kickoff_rolling(7, |s| {
        s.fix_d6(2);
        s.fix_d6(5);
        land(s);
    });
    assert_eq!((state.home.drive_rerolls, state.away.drive_rerolls), (0, 1));
}

#[test]
fn the_drive_re_roll_is_spent_first() {
    let mut team = crate::core::model::TeamState::new();
    team.rerolls = 1;
    team.drive_rerolls = 1;
    team.use_reroll();
    assert_eq!((team.rerolls, team.drive_rerolls), (1, 0));
    team.use_reroll();
    assert!(!team.can_use_reroll());
    team.drive_rerolls = 1;
    assert!(team.can_use_reroll(), "a drive re-roll alone is a team re-roll");
}

/// S3 Kick-off Event 9: "The Coach of the receiving team selects up to D3+3 Open players on their
/// team. The selected players may immediately move one square in any direction, even if this takes
/// them into the opposition's half."
#[test]
fn quick_snap_moves_open_receivers_one_square() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(9, |s| s.fix_d6(1)); // up to four
    assert!(state.home_to_act());
    let offered = sorted(selectable(&state, PosAT::SelectPosition));
    assert_eq!(offered, open_positions(&state, TeamType::Home));
    let from = offered[0];
    let id = state.get_player_id_at(from).unwrap();
    state.step_positional(PosAT::SelectPosition, from);
    assert_eq!(state.info.active_player, Some(id));
    let squares = selectable(&state, PosAT::PlacePlayer);
    assert!(!squares.is_empty());
    assert!(squares
        .iter()
        .all(|s| s.distance_to(&from) == 1 && state.get_player_at(*s).is_none()));
    state.step_positional(PosAT::PlacePlayer, squares[0]);
    assert_eq!(state.get_player_id_at(squares[0]), Some(id));

    // Moved once: no longer offered.
    assert!(!selectable(&state, PosAT::SelectPosition).contains(&squares[0]));
    land(&mut state);
    state.step_simple(SimpleAT::EndPlayerTurn);
    assert!(state.home_to_act());
    assert!(matches!(state.proc_stack_peek(), Some(AnyProc::Turn(_))));
}

#[test]
fn quick_snap_may_cross_the_line_of_scrimmage() {
    // A lone Open receiver right behind the line, the kicker well back.
    let dims = GameStateBuilder::new().build().board_dims;
    let r = Position::new((dims.los_x(TeamType::Home), dims.height / 2));
    let across = Position::new((dims.los_x(TeamType::Away), r.y));
    let mut state = GameStateBuilder::new()
        .add_home_player(r)
        .add_away_player(Position::new((dims.los_x(TeamType::Away) - 4, 1)))
        .build();
    assert!(state.is_on_team_side(r, TeamType::Home));
    assert!(
        !state.is_on_team_side(across, TeamType::Home),
        "{across:?} is the other half"
    );
    assert_eq!(across.distance_to(&r), 1);
    state.info.kicking_this_drive = TeamType::Away;
    state.set_ball(BallState::InAir(r + (2, 0)));
    let AnyProc::QuickSnap(mut snap) = QuickSnap::new() else {
        unreachable!()
    };
    snap.step(&mut state, ProcInput::Nothing);
    let ProcState::NeedAction(_) = snap.step(&mut state, ProcInput::Roll(RollResult::D6(D6::One))) else {
        panic!("expected a selection")
    };
    let ProcState::NeedAction(aa) = snap.step(
        &mut state,
        ProcInput::Action(Action::Positional(PosAT::SelectPosition, r)),
    ) else {
        panic!("expected a square")
    };
    state.available_actions = aa;
    assert!(state.is_legal_action(&Action::Positional(PosAT::PlacePlayer, across)));
}

/// A D3 of 1 moves four of six Open receivers, then Quick Snap ends by itself.
#[test]
fn quick_snap_stops_at_its_d3_plus_3() {
    let dims = GameStateBuilder::new().build().board_dims;
    let x = dims.los_x(TeamType::Home) + 3;
    let mut builder = GameStateBuilder::new();
    for y in [1, 3, 5, 7, 9, 11] {
        builder.add_home_player(Position::new((x, y)));
    }
    let mut state = builder.build();
    state.info.kicking_this_drive = TeamType::Away;
    let AnyProc::QuickSnap(mut snap) = QuickSnap::new() else {
        unreachable!()
    };
    snap.step(&mut state, ProcInput::Nothing);
    let mut out = snap.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Two)));
    for y in [1, 3, 5, 7] {
        assert!(matches!(out, ProcState::NeedAction(_)));
        let from = Position::new((x, y));
        snap.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::SelectPosition, from)),
        );
        out = snap.step(
            &mut state,
            ProcInput::Action(Action::Positional(PosAT::PlacePlayer, from + (1, 0))),
        );
    }
    assert!(matches!(out, ProcState::Done), "{out:?}");
}

/// Solid Defence run on a small-board setup of Away's: `lift` is picked up and set up again.
/// Returns the squares `Setup` offers for it.
fn solid_defence_replacing(away: &[Position], lift: Position) -> Vec<Position> {
    let mut builder = GameStateBuilder::new();
    for &p in away {
        builder.add_away_player(p);
    }
    let mut state = builder.build();
    state.info.kicking_this_drive = TeamType::Away;
    assert!(state.is_setup_legal(TeamType::Away), "the fixture is a legal setup");
    state.fix_d6(1);
    state.start_proc(SolidDefence::new());
    state.step_positional(PosAT::SelectPosition, lift);
    state.step_simple(SimpleAT::EndPlayerTurn);
    assert!(state.away_to_act());
    selectable(&state, PosAT::PlacePlayer)
}

/// "Set up again following all the usual restrictions": with only three on the line, a lifted
/// line player goes back on the line; with a wing full, a lifted player can't join it.
#[test]
fn solid_defence_re_places_under_the_line_and_wing_limits() {
    let dims = GameStateBuilder::new().build().board_dims;
    let lx = dims.los_x(TeamType::Away);
    let los: Vec<Position> = dims.los_y_range().take(3).map(|y| Position::new((lx, y))).collect();
    let wing: Vec<Position> = dims
        .north_wing_y_range()
        .take(dims.max_players_per_wing() as usize)
        .map(|y| Position::new((lx - 2, y)))
        .collect();
    let back = Position::new((lx - 3, dims.height / 2));
    let away: Vec<Position> = los.iter().chain(&wing).chain([&back]).copied().collect();

    let squares = solid_defence_replacing(&away, los[0]);
    assert!(!squares.is_empty());
    assert!(
        squares.iter().all(|p| p.x == lx && dims.los_y_range().contains(&p.y)),
        "back on the line: {squares:?}"
    );

    let squares = solid_defence_replacing(&away, back);
    assert!(!squares.is_empty());
    assert!(
        squares.iter().all(|p| !dims.north_wing_y_range().contains(&p.y)),
        "the north wing is full: {squares:?}"
    );
}

/// S3 Kick-off Event 10: up to D3+3 Open kicking players are activated for a free Move Action,
/// one of them may Blitz instead, as in their own turn.
#[test]
fn charge_activates_open_kickers_as_in_their_turn() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(10, |s| s.fix_d6(1)); // up to four
    assert!(state.away_to_act());
    assert_eq!(
        state.turn_team(),
        Some(TeamType::Away),
        "team re-rolls are the kickers'"
    );
    let open = open_positions(&state, TeamType::Away);
    assert_eq!(sorted(selectable(&state, PosAT::StartMove)), open);
    assert_eq!(sorted(selectable(&state, PosAT::StartBlitz)), open);
    for at in [
        PosAT::StartPass,
        PosAT::StartFoul,
        PosAT::StartBlock,
        PosAT::StartHandoff,
    ] {
        assert!(selectable(&state, at).is_empty(), "{at:?}");
    }

    let from = open[0];
    let id = state.get_player_id_at(from).unwrap();
    state.step_positional(PosAT::StartMove, from);
    let to = *selectable(&state, PosAT::Move)
        .iter()
        .find(|p| p.distance_to(&from) == 1)
        .unwrap();
    state.step_positional(PosAT::Move, to);
    if state.is_legal_action(&Action::Simple(SimpleAT::EndPlayerTurn)) {
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    assert_eq!(state.get_player_unsafe(id).position, to);
    assert!(state.away_to_act());
    let still = selectable(&state, PosAT::StartMove);
    assert!(!still.contains(&to), "activated once");
    assert_eq!(still.len(), open.len() - 1);

    land(&mut state);
    state.step_simple(SimpleAT::EndTurn);
    assert!(
        state.home_to_act(),
        "the ball has landed and the receivers' turn starts"
    );
    assert!(!state.info.turnover);
}

/// "If a selected player Falls Over or is Knocked Down during their activation, no further
/// selected players can be activated and the Charge ends" — and it is no turnover.
#[test]
fn a_kicker_falling_over_ends_the_charge_without_a_turnover() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(10, |s| s.fix_d6(6)); // up to six
    state.get_mut_team(TeamType::Away).rerolls = 0;
    let open = open_positions(&state, TeamType::Away);
    let from = open[0];
    let id = state.get_player_id_at(from).unwrap();
    let ma = state.get_player_unsafe(id).stats.ma;
    state.step_positional(PosAT::StartMove, from);
    // A square only a rush reaches, back towards the own endzone and away from every opponent.
    let to = selectable(&state, PosAT::Move)
        .into_iter()
        .filter(|p| p.distance_to(&from) == ma as i8 + 1)
        .max_by_key(|p| (p.x, -(p.y - from.y).abs()))
        .expect("a rush square");
    state.fix_d6(1); // the rush fails
    state.fix_d6(1); // armour holds
    state.fix_d6(1);
    land(&mut state);
    state.step_positional(PosAT::Move, to);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    assert!(state.home_to_act(), "Charge over, ball landed, receivers' turn");
    assert!(!state.info.turnover);
    assert_eq!(state.info.home_turn, 1);
}

/// A Charge! by Away on a small board, run from its first decision: `d6` is the D3's die.
fn charging(state: &mut GameState, d6: u8) {
    state.set_ball(BallState::InAir(Position::new((2, 2))));
    state.fix_d6(d6);
    state.start_proc(Charge::new(TeamType::Away));
    assert!(state.away_to_act());
}

fn in_charge(state: &GameState) -> bool {
    state.proc_stack_iter().any(|p| matches!(p, AnyProc::Charge(_)))
}

/// Where a receiver waits while the Charge picks its Open kickers.
const PARKED: Position = Position { x: 2, y: 8 };

/// The parked receiver steps next to a kicker once the Charge has picked its players.
fn step_in(state: &mut GameState, to: Position) {
    let id = state.get_player_id_at(PARKED).unwrap();
    state.move_player(id, to).unwrap();
}

/// Activate the player at `at` and end their activation at once.
fn activate_idle(state: &mut GameState, at: PosAT, pos: Position) {
    state.step_positional(at, pos);
    state.step_simple(SimpleAT::EndPlayerTurn);
}

/// "Up to D3+3": a D3 of 1 activates four players, then the Charge ends by itself; one Blitz in
/// all.
#[test]
fn charge_activates_up_to_d3_plus_3_players_and_blitzes_once() {
    let mut builder = GameStateBuilder::new();
    let squares: Vec<Position> = (1..=6).map(|y| Position::new((10, y))).collect();
    for &p in &squares {
        builder.add_away_player(p);
    }
    let mut state = builder.build();
    let team_turn = state.info.team_turn;
    charging(&mut state, 1);
    assert_eq!(selectable(&state, PosAT::StartBlitz).len(), 6);
    activate_idle(&mut state, PosAT::StartBlitz, squares[0]);
    assert!(
        selectable(&state, PosAT::StartBlitz).is_empty(),
        "the one Blitz is taken"
    );
    assert_eq!(selectable(&state, PosAT::StartMove).len(), 5);
    for &p in &squares[1..4] {
        assert!(in_charge(&state));
        activate_idle(&mut state, PosAT::StartMove, p);
    }
    assert!(!in_charge(&state), "the fourth activation was the last");
    assert_eq!(state.info.team_turn, team_turn, "the turn marker is back");
}

/// Team re-rolls are the kicking team's to spend during the Charge.
#[test]
fn a_charging_player_may_use_a_team_re_roll() {
    let from = Position::new((10, 5));
    let mut state = GameStateBuilder::new()
        .add_away_player(from)
        .add_home_player(Position::new((2, 8)))
        .build();
    state.away.rerolls = 1;
    charging(&mut state, 1);
    let id = state.get_player_id_at(from).unwrap();
    state.get_mut_player_unsafe(id).stats.ma = 1;
    state.step_positional(PosAT::StartMove, from);
    let to = from + (0, 2); // one square and one Rush
    state.fix_d6(1); // the Rush fails
    state.step_positional(PosAT::Move, to);
    assert!(state.is_legal_action(&Action::Simple(SimpleAT::UseReroll)));
    state.fix_d6(6);
    state.step_simple(SimpleAT::UseReroll);
    assert_eq!(state.away.rerolls, 0);
    assert_eq!(state.get_player_unsafe(id).position, to);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Up);
}

/// Knocked Down on a Both Down (no Block): the Charge ends, with no turnover.
#[test]
fn a_kicker_knocked_down_ends_the_charge() {
    let a = Position::new((10, 5));
    let h = a + (1, 0);
    let mut state = GameStateBuilder::new()
        .add_away_player(a)
        .add_away_player(Position::new((10, 1)))
        .add_home_player(PARKED)
        .build();
    state.away.rerolls = 0;
    charging(&mut state, 6);
    step_in(&mut state, h);
    state.step_positional(PosAT::StartBlitz, a);
    state.fix_blockdice(BlockDice::BothDown);
    state.step_positional(PosAT::Block, h);
    for _ in 0..4 {
        state.fix_d6(1); // both armour rolls hold
    }
    state.step_simple(SimpleAT::SelectBothDown);
    let id = state.get_player_id_at(a).unwrap();
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    assert!(!in_charge(&state));
    assert!(!state.info.turnover);
}

/// Placed Prone is not Knocked Down: a Wrestled kicker does not end the Charge.
#[test]
fn a_wrestled_kicker_does_not_end_the_charge() {
    let a = Position::new((10, 5));
    let h = a + (1, 0);
    let mut state = GameStateBuilder::new()
        .add_away_player(a)
        .add_away_player(Position::new((10, 1)))
        .add_home_player(PARKED)
        .build();
    let id = state.get_player_id_at(a).unwrap();
    state.get_mut_player_unsafe(id).stats.give_skill(Skill::Wrestle);
    charging(&mut state, 6);
    step_in(&mut state, h);
    state.step_positional(PosAT::StartBlitz, a);
    state.fix_blockdice(BlockDice::BothDown);
    state.step_positional(PosAT::Block, h);
    state.step_simple(SimpleAT::SelectBothDown);
    state.step_simple(SimpleAT::UseSkill);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Down);
    if state.is_legal_action(&Action::Simple(SimpleAT::EndPlayerTurn)) {
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    assert!(in_charge(&state), "the Charge goes on");
    assert!(!selectable(&state, PosAT::StartMove).is_empty());
}

/// A kicker chain-pushed into the crowd by a charging team-mate is no turnover and does not end
/// the Charge: only the activated player falling or being Knocked Down does. The one in the crowd
/// can no longer be activated.
#[test]
fn a_kicker_pushed_into_the_crowd_does_not_end_the_charge() {
    let a = Position::new((6, 3));
    let h = Position::new((6, 2));
    let mates = [Position::new((5, 1)), Position::new((6, 1)), Position::new((7, 1))];
    let mut builder = GameStateBuilder::new();
    builder.add_away_player(a).add_home_player(PARKED);
    for &m in &mates {
        builder.add_away_player(m);
    }
    builder.add_away_player(Position::new((12, 5)));
    let mut state = builder.build();
    let pushed = state.get_player_id_at(mates[1]).unwrap();
    charging(&mut state, 6);
    step_in(&mut state, h);
    state.step_positional(PosAT::StartBlitz, a);
    state.fix_blockdice(BlockDice::Push); // 3 + 3 assists vs 3: two dice
    state.fix_blockdice(BlockDice::Push);
    state.step_positional(PosAT::Block, h);
    state.step_simple(SimpleAT::SelectPush);
    // The chain push has only the crowd to go to.
    state.step_positional(PosAT::Push, mates[1]);
    state.fix_d6(1); // the crowd's injury roll
    state.fix_d6(1);
    state.step_positional(PosAT::FollowUp, a);
    assert!(state.get_player(pushed).is_err(), "in the crowd");
    if state.is_legal_action(&Action::Simple(SimpleAT::EndPlayerTurn)) {
        state.step_simple(SimpleAT::EndPlayerTurn);
    }
    assert!(in_charge(&state), "the Charge goes on");
    assert!(!state.info.turnover);
    assert!(!selectable(&state, PosAT::StartMove).is_empty());
}

/// S3 Kick-off Event 11: the lowest D6 picks a random player of theirs; on a 2+ MA and AV are one
/// lower for the drive, on a 1 the player sits the drive out in the Reserves box.
#[test]
fn dodgy_snack_weakens_a_random_player_of_the_lower_roll() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut victim = None;
    let mut stats_before: Option<PlayerStats> = None;
    let state = kickoff_rolling(11, |s| {
        s.fix_d6(2); // Home: lowest
        s.fix_d6(5);
        s.fix_pick(3);
        let id = on_pitch(s, TeamType::Home)[3];
        victim = Some(id);
        stats_before = Some(s.get_player_unsafe(id).stats.clone());
        s.fix_d6(4);
        land(s);
    });
    let p = state.get_player_unsafe(victim.unwrap());
    let before = stats_before.unwrap();
    assert!(p.snack);
    assert_eq!((p.stats.ma, p.stats.armour), (before.ma - 1, before.armour - 1));
    assert_eq!(
        state.get_players_on_pitch().filter(|p| p.snack).count(),
        1,
        "Away won the roll-off"
    );
}

#[test]
fn a_dodgy_snack_on_a_one_sends_the_player_to_the_reserves_for_the_drive() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut victim = None;
    let state = kickoff_rolling(11, |s| {
        s.fix_d6(5);
        s.fix_d6(2); // Away: lowest
        s.fix_pick(0);
        victim = Some(on_pitch(s, TeamType::Away)[0]);
        s.fix_d6(1);
        land(s);
    });
    assert!(state.get_player(victim.unwrap()).is_err(), "off the pitch");
    let heated: Vec<_> = state.get_dugout().filter(|p| p.place == DugoutPlace::Heated).collect();
    assert_eq!(heated.len(), 1);
    assert_eq!(heated[0].stats.team, TeamType::Away);
}

#[test]
fn a_tied_dodgy_snack_hits_both_teams() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(11, |s| {
        s.fix_d6(3);
        s.fix_d6(3);
        s.fix_pick(0); // Home
        s.fix_d6(4);
        s.fix_pick(0); // Away
        s.fix_d6(4);
        land(s);
    });
    let snacked = |team| state.get_players_on_pitch_in_team(team).filter(|p| p.snack).count();
    assert_eq!((snacked(TeamType::Home), snacked(TeamType::Away)), (1, 1));
}

/// -1 AV makes the armour easier to break: an AV 9+ player with the snack breaks on an 8.
#[test]
fn a_snacked_player_s_armour_breaks_one_lower() {
    let a = Position::new((5, 3));
    let d = a + (1, 0);
    let mut state = GameStateBuilder::new().add_home_player(a).add_away_player(d).build();
    let id = state.get_player_id_at(d).unwrap();
    state.get_mut_player_unsafe(id).stats.armour = 9; // AV 9+
    let AnyProc::DodgySnack(mut snack) = DodgySnack::new(TeamType::Away) else {
        unreachable!()
    };
    snack.step(&mut state, ProcInput::Roll(RollResult::Pick(0)));
    snack.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Three)));
    assert_eq!(state.get_player_unsafe(id).stats.armour, 8);

    state.step_positional(PosAT::StartBlock, a);
    state.fix_blockdice(BlockDice::Pow);
    state.step_positional(PosAT::Block, d);
    state.step_simple(SimpleAT::SelectPow);
    state.step_positional(PosAT::Push, d + (1, 0));
    state.fix_d6(4); // armour 8: broken on 8+
    state.fix_d6(4);
    state.fix_d6(1); // injury 2: Stunned
    state.fix_d6(1);
    state.step_positional(PosAT::FollowUp, d);
    assert_eq!(state.get_player_unsafe(id).status, PlayerStatus::Stunned);
}

#[test]
fn a_snack_wears_off_when_the_player_leaves_the_pitch() {
    let pos = Position::new((5, 5));
    let mut state = GameStateBuilder::new().add_home_player(pos).build();
    let id = state.get_player_id_at(pos).unwrap();
    let before = state.get_player_unsafe(id).stats.clone();
    let AnyProc::DodgySnack(mut snack) = DodgySnack::new(TeamType::Home) else {
        unreachable!()
    };
    snack.step(&mut state, ProcInput::Roll(RollResult::Pick(0)));
    snack.step(&mut state, ProcInput::Roll(RollResult::D6(D6::Three)));
    assert_ne!(state.get_player_unsafe(id).stats, before);
    state.unfield_player(id, DugoutPlace::Reserves).unwrap();
    assert_eq!(state.get_dugout().next().unwrap().stats, before);
}

/// S3 Kick-off Event 12: D6 + Fan Factor; the lowest has D3 random players on the pitch Placed
/// Prone and Stunned.
#[test]
fn pitch_invasion_stuns_d3_random_players_of_the_lower_roll() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(12, |s| {
        s.get_mut_team(TeamType::Home).fan_factor = 3;
        s.fix_d6(2); // Home + 3 = 5
        s.fix_d6(4); // Away: lowest
        s.fix_d6(3); // D3 = 2
        s.fix_pick(0);
        s.fix_pick(0); // the next one not yet picked
        land(s);
    });
    let stunned: Vec<_> = state
        .get_players_on_pitch()
        .filter(|p| p.status == PlayerStatus::Stunned)
        .map(|p| (p.stats.team, p.id))
        .collect();
    let away = on_pitch(&state, TeamType::Away);
    assert_eq!(stunned, vec![(TeamType::Away, away[0]), (TeamType::Away, away[1])]);
}

#[test]
fn a_tied_pitch_invasion_hits_both_teams() {
    crate::skip_if_board_smaller_than!(28, 17);
    let state = kickoff_rolling(12, |s| {
        s.fix_d6(4);
        s.fix_d6(4);
        s.fix_d6(1); // Home: D3 = 1
        s.fix_pick(0);
        s.fix_d6(1); // Away: D3 = 1
        s.fix_pick(0);
        land(s);
    });
    let stunned = |team| {
        state
            .get_players_on_pitch_in_team(team)
            .filter(|p| p.status == PlayerStatus::Stunned)
            .count()
    };
    assert_eq!((stunned(TeamType::Home), stunned(TeamType::Away)), (1, 1));
}

/// The Stunned receiver sits out the first turn; the Stunned kicker stays Stunned through it and
/// only turns over at the end of the kicking team's own next turn.
#[test]
fn pitch_invasion_stuns_last_into_the_drive() {
    crate::skip_if_board_smaller_than!(28, 17);
    let mut state = kickoff_rolling(12, |s| {
        s.get_mut_team(TeamType::Home).rerolls = 0;
        s.fix_d6(4);
        s.fix_d6(4);
        s.fix_d6(1); // Home: D3 = 1
        s.fix_pick(0);
        s.fix_d6(1); // Away: D3 = 1
        s.fix_pick(0);
        land(s);
    });
    let stunned_of = |state: &GameState, team| {
        state
            .get_players_on_pitch_in_team(team)
            .find(|p| p.status == PlayerStatus::Stunned)
            .map(|p| (p.id, p.position))
    };
    let (home_id, home_pos) = stunned_of(&state, TeamType::Home).unwrap();
    let (away_id, _) = stunned_of(&state, TeamType::Away).unwrap();
    assert!(state.home_to_act());
    assert!(!selectable(&state, PosAT::StartMove).contains(&home_pos));
    state.step_simple(SimpleAT::EndTurn);
    assert!(state.away_to_act());
    assert_eq!(
        state.get_player_unsafe(home_id).status,
        PlayerStatus::Down,
        "turned over at Home's turn end"
    );
    assert_eq!(state.get_player_unsafe(away_id).status, PlayerStatus::Stunned);
}

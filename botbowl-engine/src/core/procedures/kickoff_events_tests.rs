use super::*;
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

/// The extra assist: a 3-vs-3 block with no other help rolls two dice instead of one, on the
/// turn's first Block Action only.
#[test]
fn the_cheering_fans_assist_goes_to_the_first_block_action_only() {
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
    let dice = |state: &GameState, at: Position, def: Position| {
        let att = state.get_player_id_at(at).unwrap();
        let defender = state.get_player_id_at(def).unwrap();
        state.blockdices(att, at, defender, state.get_player_unsafe(att).stats.str_)
    };
    assert_eq!(dice(&state, a, d), NumBlockDices::One, "before the Block Action");
    state.step_positional(PosAT::StartBlock, a);
    assert_eq!(dice(&state, a, d), NumBlockDices::Two);
    state.step_simple(SimpleAT::EndPlayerTurn);
    state.step_positional(PosAT::StartBlock, a2);
    assert_eq!(dice(&state, a2, d2), NumBlockDices::One);
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

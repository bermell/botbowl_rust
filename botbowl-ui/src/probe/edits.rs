//! Controlled position edits. Each returns a new state that differs from the input in exactly
//! the stated way (the tests pin it through the encoder: one edit, one changed cell), or an error
//! naming why the edit does not apply. The result's offerings (`available_actions`) are *not*
//! regenerated — a value forward does not read them (the encoder recomputes the path plane) — so
//! a caller that wants to play on from an edited `Turn` decision calls
//! `GameState::reask_turn_decision` (see [`reasked`]).

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{BallState, DugoutPlace, PlayerID, PlayerStats, PlayerStatus, Position, TeamType};
use botbowl_engine::core::table::Skill;
use serde::Serialize;

pub type Edited = Result<GameState, String>;

/// A characteristic the encoder sees (PA is not encoded, so it is not probed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, PartialOrd, Ord)]
pub enum Stat {
    St,
    Ma,
    Ag,
    Av,
}

impl Stat {
    pub const ALL: [Stat; 4] = [Stat::St, Stat::Ma, Stat::Ag, Stat::Av];

    /// The range the random-start generator draws from (`botbowl-curriculum` `*_RANGE`): an
    /// improvement past it is an input the net has never seen, so it is not probed. Higher is
    /// better for all four in the engine's representation (AG: target = 7 - ag; AV: armour
    /// breaks on > av).
    pub fn training_range(self) -> (u8, u8) {
        match self {
            Stat::St => (1, 6),
            Stat::Ma => (1, 9),
            Stat::Ag => (1, PlayerStats::MAX_AG),
            Stat::Av => (3, 11),
        }
    }

    fn get(self, s: &PlayerStats) -> u8 {
        match self {
            Stat::St => s.str_,
            Stat::Ma => s.ma,
            Stat::Ag => s.ag,
            Stat::Av => s.av,
        }
    }

    fn set(self, s: &mut PlayerStats, v: u8) {
        match self {
            Stat::St => s.str_ = v,
            Stat::Ma => s.ma = v,
            Stat::Ag => s.ag = v,
            Stat::Av => s.av = v,
        }
    }
}

fn player_exists(s: &GameState, id: PlayerID) -> Result<(), String> {
    s.get_player(id)
        .map(|_| ())
        .map_err(|_| format!("no player {id} on the pitch"))
}

pub fn add_skill(s: &GameState, id: PlayerID, skill: Skill) -> Edited {
    player_exists(s, id)?;
    if s.get_player_unsafe(id).has_skill(skill) {
        return Err(format!("player {id} already has {skill:?}"));
    }
    let mut out = s.clone();
    out.get_mut_player_unsafe(id).stats.give_skill(skill);
    Ok(out)
}

pub fn remove_skill(s: &GameState, id: PlayerID, skill: Skill) -> Edited {
    player_exists(s, id)?;
    if !s.get_player_unsafe(id).has_skill(skill) {
        return Err(format!("player {id} does not have {skill:?}"));
    }
    let mut out = s.clone();
    out.get_mut_player_unsafe(id).stats.skills.remove(&skill);
    Ok(out)
}

/// `stat += delta`, refused outside the training range. A movement allowance cut below what the
/// player has already moved is refused too.
pub fn bump_stat(s: &GameState, id: PlayerID, stat: Stat, delta: i8) -> Edited {
    player_exists(s, id)?;
    let p = s.get_player_unsafe(id);
    let (lo, hi) = stat.training_range();
    let v = stat.get(&p.stats) as i16 + delta as i16;
    if v < lo as i16 || v > hi as i16 {
        return Err(format!(
            "{stat:?} {} {delta:+} leaves the training range {lo}..={hi}",
            stat.get(&p.stats)
        ));
    }
    if stat == Stat::Ma && (p.moves as i16) > v + 2 {
        return Err("the player has already moved further".into());
    }
    let mut out = s.clone();
    stat.set(&mut out.get_mut_player_unsafe(id).stats, v as u8);
    Ok(out)
}

/// A playable square nobody stands on and the ball does not lie on.
pub fn is_free(s: &GameState, p: Position) -> bool {
    !s.board_dims.is_out(p)
        && s.get_player_at(p).is_none()
        && !matches!(s.ball, BallState::OnGround(b) | BallState::InAir(b) if b == p)
}

/// Move player `id` to `to`, everything else unchanged. A carrier takes the ball along, and may
/// not be put in an endzone column (a touchdown the engine never awarded).
pub fn translate(s: &GameState, id: PlayerID, to: Position) -> Edited {
    player_exists(s, id)?;
    if s.get_player_unsafe(id).position == to {
        return Ok(s.clone());
    }
    if !is_free(s, to) {
        return Err(format!("{to:?} is not a free square"));
    }
    let dims = s.board_dims;
    if s.ball == BallState::Carried(id)
        && (to.x == dims.endzone_x(TeamType::Home) || to.x == dims.endzone_x(TeamType::Away))
    {
        return Err("a carrier in an endzone".into());
    }
    let mut out = s.clone();
    out.move_player(id, to).map_err(|e| e.to_string())?;
    Ok(out)
}

pub fn set_status(s: &GameState, id: PlayerID, status: PlayerStatus) -> Edited {
    player_exists(s, id)?;
    let p = s.get_player_unsafe(id);
    if p.status == status {
        return Err(format!("player {id} is already {status:?}"));
    }
    if status != PlayerStatus::Up && s.ball == BallState::Carried(id) {
        return Err("a carrier cannot be put down (the ball would have to bounce)".into());
    }
    let mut out = s.clone();
    out.get_mut_player_unsafe(id).status = status;
    Ok(out)
}

/// Set how far player `id` has moved this turn (`0..=ma+2`).
pub fn set_moves(s: &GameState, id: PlayerID, moves: u8) -> Edited {
    player_exists(s, id)?;
    let p = s.get_player_unsafe(id);
    if moves > p.stats.ma + 2 {
        return Err(format!("{moves} moves > MA {} + 2", p.stats.ma));
    }
    let mut out = s.clone();
    out.get_mut_player_unsafe(id).moves = moves;
    Ok(out)
}

/// Take player `id` off the pitch (to the KO box). Not the carrier, not the active player.
pub fn remove_player(s: &GameState, id: PlayerID) -> Edited {
    player_exists(s, id)?;
    if s.ball == BallState::Carried(id) {
        return Err("the carrier".into());
    }
    if s.info.active_player == Some(id) {
        return Err("the active player".into());
    }
    let mut out = s.clone();
    out.unfield_player(id, DugoutPlace::KnockOut)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// Field a plain standing lineman for `team` on the free square `at`, if the team is below its
/// team size on the pitch.
pub fn add_lineman(s: &GameState, team: TeamType, at: Position) -> Result<(GameState, PlayerID), String> {
    if !is_free(s, at) {
        return Err(format!("{at:?} is not a free square"));
    }
    if s.get_players_on_pitch_in_team(team).count() >= s.board_dims.team_size {
        return Err("the team is at full strength".into());
    }
    let mut out = s.clone();
    let id = out
        .add_new_player_to_field(PlayerStats::new_lineman(team), at)
        .map_err(|e| e.to_string())?;
    Ok((out, id))
}

/// `team`'s rerolls `+= delta`, kept in `0..=rerolls_start` (a drive starts at the full allowance).
pub fn add_rerolls(s: &GameState, team: TeamType, delta: i8) -> Edited {
    let t = s.get_team(team);
    let v = t.rerolls as i16 + delta as i16;
    if v < 0 || v > t.rerolls_start as i16 {
        return Err(format!(
            "rerolls {} {delta:+} outside 0..={}",
            t.rerolls, t.rerolls_start
        ));
    }
    let mut out = s.clone();
    out.get_mut_team(team).rerolls = v as u8;
    Ok(out)
}

/// Both teams' turn counters `-= turns` (more of the half left for both), kept `>= 1` for the
/// mover and `>= 0` for the other side.
pub fn rewind_clock(s: &GameState, mover: TeamType, turns: u8) -> Edited {
    let (m, o) = match mover {
        TeamType::Home => (s.info.home_turn, s.info.away_turn),
        TeamType::Away => (s.info.away_turn, s.info.home_turn),
    };
    if m <= turns || o < turns {
        return Err("too early in the half".into());
    }
    let mut out = s.clone();
    out.info.home_turn -= turns;
    out.info.away_turn -= turns;
    Ok(out)
}

/// `s` with its `Turn` offerings regenerated after an edit; the error names a non-`Turn` state.
pub fn reasked(mut s: GameState) -> Edited {
    s.reask_turn_decision()?;
    Ok(s)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use botbowl_curriculum::{generate_random_start, RandomStartConfig};
    use botbowl_engine::core::model::Action;
    use botbowl_nn::encode::{encode_raw, spatial_channel_names};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Fresh random-start `Turn` decisions on the env board.
    pub(crate) fn starts(n: u64) -> Vec<GameState> {
        (0..n)
            .map(|seed| {
                let mut st = generate_random_start(&RandomStartConfig::default(), &mut ChaCha8Rng::seed_from_u64(seed));
                st.set_logging_state(false);
                st
            })
            .collect()
    }

    fn team_of(s: &GameState) -> TeamType {
        botbowl_nn::perspective::mover_for(s)
    }

    /// Spatial cells `(channel name, flat index)` whose raw encoding differs, and whether any
    /// global feature does.
    fn encoding_diff(a: &GameState, b: &GameState) -> (Vec<(String, usize)>, bool) {
        let (ea, eb) = (encode_raw(a), encode_raw(b));
        let names = spatial_channel_names();
        let plane = ea.h * ea.w;
        let cells = ea
            .spatial
            .iter()
            .zip(&eb.spatial)
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| (names[i / plane].clone(), i % plane))
            .collect();
        (cells, ea.global != eb.global)
    }

    /// Adding a skill changes exactly one input: that skill's plane at that player's square.
    /// Removing it again gives back the very same state.
    #[test]
    fn adding_a_skill_changes_only_that_skill_on_that_player() {
        let mut n = 0;
        for s in starts(12) {
            for p in s.get_players_on_pitch() {
                for skill in [Skill::Block, Skill::Dodge, Skill::Guard, Skill::Claws] {
                    let Ok(e) = add_skill(&s, p.id, skill) else { continue };
                    let (cells, global) = encoding_diff(&s, &e);
                    assert!(!global);
                    assert_eq!(cells.len(), 1, "{skill:?}: {cells:?}");
                    assert_eq!(cells[0].0, format!("skill_{skill:?}").to_lowercase(), "{cells:?}");
                    assert_eq!(cells[0].1, {
                        let c = botbowl_nn::perspective::canonical_pos(p.position, s.board_dims, team_of(&s));
                        c.y as usize * s.board_dims.width as usize + c.x as usize
                    });
                    assert!(remove_skill(&e, p.id, skill).unwrap() == s);
                    n += 1;
                }
            }
        }
        assert!(n > 50, "{n}");
    }

    /// A stat bump changes that characteristic's plane at that square (and, for MA, the
    /// movement-left plane), nothing else; refused outside the training range.
    #[test]
    fn a_stat_bump_changes_only_that_stat() {
        for s in starts(8) {
            for p in s.get_players_on_pitch() {
                for stat in Stat::ALL {
                    match bump_stat(&s, p.id, stat, 1) {
                        Ok(e) => {
                            let (cells, global) = encoding_diff(&s, &e);
                            assert!(!global);
                            let want: &[&str] = match stat {
                                Stat::St => &["st"],
                                Stat::Ma => &["ma", "movement_left"],
                                Stat::Ag => &["ag"],
                                Stat::Av => &["av"],
                            };
                            let mut got: Vec<&str> = cells.iter().map(|(c, _)| c.as_str()).collect();
                            got.sort();
                            let mut want = want.to_vec();
                            want.sort();
                            assert_eq!(got, want, "{stat:?}");
                            assert!(cells.iter().all(|(_, i)| *i == cells[0].1), "one square");
                            assert!(bump_stat(&e, p.id, stat, -1).unwrap() == s);
                        }
                        Err(_) => assert_eq!(stat.get(&p.stats), stat.training_range().1),
                    }
                }
            }
        }
    }

    /// A translation moves one player, keeps every other player, the ball (unless carried) and
    /// the game info, and moving back restores the state exactly.
    #[test]
    fn translation_keeps_everything_else_equal() {
        let mut n = 0;
        for s in starts(10) {
            let dims = s.board_dims;
            for p in s.get_players_on_pitch() {
                let to = Position::all_positions()
                    .find(|q| is_free(&s, *q) && q.x > 1 && q.x < dims.width - 2)
                    .unwrap();
                let e = translate(&s, p.id, to).unwrap();
                assert_eq!(e.get_player_unsafe(p.id).position, to);
                assert_eq!(e.get_player_at(p.position).map(|x| x.id), None);
                for q in s.get_players_on_pitch().filter(|q| q.id != p.id) {
                    assert_eq!(e.get_player_unsafe(q.id), q);
                }
                assert_eq!(e.ball, s.ball);
                assert!(e.info == s.info && e.home == s.home && e.away == s.away);
                assert!(translate(&e, p.id, p.position).unwrap() == s);
                n += 1;
            }
        }
        assert!(n > 50);
    }

    /// The y-flip is exactly a y-reflection of the encoded tensor, path plane included (the
    /// pathfinder's probabilities are symmetric), with the globals unchanged — so any value
    /// difference the symmetry probe reports is the net's, not the encoder's.
    #[test]
    fn a_y_flipped_state_encodes_as_the_reflected_tensor() {
        let mut with_paths = 0;
        for s in starts(10) {
            let mut states = vec![s.clone()];
            // An activated mover, so the path plane is populated.
            let team = s.available_actions.team.unwrap();
            if let Some(p) = s.get_players_on_pitch_in_team(team).find(|p| {
                s.is_legal_action(&Action::Positional(
                    botbowl_engine::core::table::PosAT::StartMove,
                    p.position,
                ))
            }) {
                let mut m = s.clone();
                m.step(Action::Positional(
                    botbowl_engine::core::table::PosAT::StartMove,
                    p.position,
                ))
                .unwrap();
                if m.available_actions.has_paths() {
                    with_paths += 1;
                    states.push(m);
                }
            }
            for st in states {
                let (a, b) = (encode_raw(&st), encode_raw(&st.y_flipped()));
                assert_eq!(a.global, b.global);
                let (h, w) = (a.h, a.w);
                for c in 0..a.spatial.len() / (h * w) {
                    for y in 0..h {
                        for x in 0..w {
                            assert_eq!(
                                a.spatial[c * h * w + y * w + x],
                                b.spatial[c * h * w + (h - 1 - y) * w + x],
                                "channel {} at ({x},{y})",
                                spatial_channel_names()[c]
                            );
                        }
                    }
                }
            }
        }
        assert!(with_paths > 3, "{with_paths}");
    }

    /// An edited `Turn` decision re-asked is a legal state: its offerings follow the edit and
    /// every one of them steps.
    #[test]
    fn an_edited_turn_decision_reasked_is_playable() {
        for s in starts(6) {
            let team = s.available_actions.team.unwrap();
            let p = s.get_players_on_pitch_in_team(team).next().unwrap().clone();
            let to = Position::all_positions()
                .find(|q| is_free(&s, *q) && q.x > 2 && q.x < s.board_dims.width - 3)
                .unwrap();
            let e = reasked(translate(&s, p.id, to).unwrap()).unwrap();
            for a in e.get_all_actions() {
                let mut x = e.clone();
                x.step(a).unwrap();
            }
        }
    }

    #[test]
    fn edits_refuse_what_does_not_apply() {
        let s = &starts(1)[0];
        let carrier = match s.ball {
            BallState::Carried(id) => Some(id),
            _ => None,
        };
        let p = s.get_players_on_pitch().next().unwrap();
        assert!(translate(s, p.id, Position::new((0, 0))).is_err(), "out of bounds");
        let other = s.get_players_on_pitch().nth(1).unwrap();
        assert!(translate(s, p.id, other.position).is_err(), "occupied");
        if let Some(c) = carrier {
            assert!(remove_player(s, c).is_err());
            assert!(set_status(s, c, PlayerStatus::Down).is_err());
        }
        assert!(
            add_rerolls(s, TeamType::Home, 1).is_err(),
            "a drive starts at the full allowance"
        );
        assert!(add_rerolls(s, TeamType::Home, -1).is_ok());
    }
}

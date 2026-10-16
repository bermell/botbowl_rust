//! Edits that (nearly) dominate for one side, so the mover's value must not move against them.
//! Each row's `gain` is signed so that it should be `>= 0`:
//!
//! | edit | gain | strict? |
//! |---|---|---|
//! | `reroll_own+1` / `reroll_opp+1` | `dV` / `-dV` | strict: a re-roll is only ever an option |
//! | `remove_opp` / `remove_own` (non-carrier, non-active) | `dV` / `-dV` | near: a body can block a lane for its own side, rarely |
//! | `add_own` (a plain lineman on a square 2+ from everyone) | `dV` | near |
//! | `stand_up_own` (Down -> Up) / `knock_down_opp` (Up -> Down) | `dV` | near |
//! | `carrier_forward` (one square toward the scoring end, both squares unmarked) | `dV` | near |
//! | `clock_rewind` (both turn counters -1, the mover holds the ball) | `dV` | soft: the opponent gets the turn too |
//!
//! Sizes are read against `tau`, the 99th percentile of the net's `|ΔV|` for placebo skills (no
//! rules) on the same states.

use std::collections::BTreeMap;
use std::io;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{other_team, BallState, PlayerStatus, Position, TeamType};
use botbowl_nn::perspective::mover_for;

use super::common::{attack_dx, cheb, dist, frac, par_map, placebo_deltas, Base, Case, Dist, Kind, Net, TopK};
use super::edits::{add_lineman, add_rerolls, is_free, remove_player, rewind_clock, set_status, translate};
use super::{bases, SuiteOutput};
use crate::cli::ProbeArgs;

#[derive(Serialize, Clone)]
pub struct Row {
    pub base: String,
    pub kind: Kind,
    pub edit: &'static str,
    pub strict: bool,
    pub soft: bool,
    pub player: Option<usize>,
    pub v: f32,
    pub dv: f32,
    pub gain: f32,
}

/// Whether a standing player of `team` marks `p`.
fn marked_by(s: &GameState, p: Position, team: TeamType) -> bool {
    s.get_players_on_pitch_in_team(team)
        .any(|q| q.status == PlayerStatus::Up && cheb(q.position, p) == 1)
}

/// `(edit, strict, soft, player, state, sign)`: the edited states of one base.
type Edit = (
    &'static str,
    bool,
    bool,
    Option<usize>,
    GameState,
    f32,
    Vec<(Position, char)>,
);

fn edits(s: &GameState, seed: u64) -> Vec<Edit> {
    let m = mover_for(s);
    let o = other_team(m);
    let dims = s.board_dims;
    let mut out: Vec<Edit> = Vec::new();
    if let Ok(e) = add_rerolls(s, m, 1) {
        out.push(("reroll_own+1", true, false, None, e, 1.0, vec![]));
    }
    if let Ok(e) = add_rerolls(s, o, 1) {
        out.push(("reroll_opp+1", true, false, None, e, -1.0, vec![]));
    }
    for p in s.get_players_on_pitch() {
        let mark = vec![(p.position, '!')];
        let own = p.stats.team == m;
        if let Ok(e) = remove_player(s, p.id) {
            if own {
                out.push(("remove_own", false, false, Some(p.id), e, -1.0, mark.clone()));
            } else {
                out.push(("remove_opp", false, false, Some(p.id), e, 1.0, mark.clone()));
            }
        }
        if s.info.active_player == Some(p.id) {
            continue;
        }
        if own && p.status == PlayerStatus::Down {
            if let Ok(e) = set_status(s, p.id, PlayerStatus::Up) {
                out.push(("stand_up_own", false, false, Some(p.id), e, 1.0, mark.clone()));
            }
        }
        if !own && p.status == PlayerStatus::Up {
            if let Ok(e) = set_status(s, p.id, PlayerStatus::Down) {
                out.push(("knock_down_opp", false, false, Some(p.id), e, 1.0, mark.clone()));
            }
        }
        if own && s.ball == BallState::Carried(p.id) && p.status == PlayerStatus::Up {
            let to = Position::new((p.position.x + attack_dx(dims, m), p.position.y));
            if !marked_by(s, p.position, o) && !marked_by(s, to, o) {
                if let Ok(e) = translate(s, p.id, to) {
                    out.push((
                        "carrier_forward",
                        false,
                        false,
                        Some(p.id),
                        e,
                        1.0,
                        vec![(p.position, '<'), (to, '!')],
                    ));
                }
            }
            if let Ok(e) = rewind_clock(s, m, 1) {
                out.push(("clock_rewind", false, true, None, e, 1.0, vec![]));
            }
        }
    }
    // An extra lineman on a neutral square: 2+ from every player and the ball, off the endzones.
    let ball = s.get_ball_position();
    let mut neutral: Vec<Position> = Position::all_positions()
        .filter(|q| {
            is_free(s, *q)
                && q.x != dims.endzone_x(TeamType::Home)
                && q.x != dims.endzone_x(TeamType::Away)
                && s.get_players_on_pitch().all(|p| cheb(p.position, *q) >= 2)
                && ball.is_none_or(|b| cheb(b, *q) >= 2)
        })
        .collect();
    neutral.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    for q in neutral.into_iter().take(2) {
        if let Ok((e, id)) = add_lineman(s, m, q) {
            out.push(("add_own", false, false, Some(id), e, 1.0, vec![(q, '!')]));
        }
    }
    out
}

fn probe(net: &Net, b: &Base, seed: u64, top: usize) -> (Vec<Row>, Vec<f32>, Vec<Case>) {
    let s = &b.state;
    let v = net.value(s);
    let noise = placebo_deltas(net, s, v);
    let mut rows = Vec::new();
    let mut worst = TopK::new(top);
    for (edit, strict, soft, player, after, sign, marks) in edits(s, seed) {
        let dv = net.value(&after) - v;
        let row = Row {
            base: b.id.clone(),
            kind: b.kind,
            edit,
            strict,
            soft,
            player,
            v,
            dv,
            gain: sign * dv,
        };
        if !soft && row.gain < 0.0 {
            worst.push(Case {
                score: -row.gain,
                label: format!("{} {edit} gain {:+.3}", b.id, row.gain),
                row: serde_json::to_value(&row).expect("row"),
                before: s.clone(),
                after,
                marks,
            });
        }
        rows.push(row);
    }
    (rows, noise, worst.into_items())
}

#[derive(Serialize)]
struct EditSummary {
    edit: &'static str,
    strict: bool,
    soft: bool,
    n: usize,
    gain: Dist,
    violation: f32,
    violation_gt_tau: f32,
}

pub fn run(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let bases = bases(args, &|_| true)?;
    eprintln!("probe dominance: {} base states", bases.len());
    let results = par_map(bases.len(), args.parallel, |i| {
        probe(
            net,
            &bases[i],
            args.seed ^ (i as u64).wrapping_mul(0x9E37_79B9),
            args.top,
        )
    });
    let noise = dist(&results.iter().flat_map(|r| r.1.iter().copied()).collect::<Vec<_>>());
    let tau = noise.p99;
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    for (r, _, c) in results {
        rows.extend(r);
        worst.extend(c);
    }
    let mut by: BTreeMap<&'static str, Vec<&Row>> = BTreeMap::new();
    for r in &rows {
        by.entry(r.edit).or_default().push(r);
    }
    let summaries: Vec<EditSummary> = by
        .iter()
        .map(|(e, rs)| {
            let g: Vec<f32> = rs.iter().map(|r| r.gain).collect();
            EditSummary {
                edit: e,
                strict: rs[0].strict,
                soft: rs[0].soft,
                n: g.len(),
                gain: dist(&g),
                violation: frac(g.iter().map(|x| *x < 0.0)),
                violation_gt_tau: frac(g.iter().map(|x| *x < -tau)),
            }
        })
        .collect();
    let mut table = format!(
        "placebo |dV| (inert skills): p50 {:.4} p99 (tau) {:.4} over {} edits\n{:<16} {:>6} {:>6} {:>8} {:>8} {:>7} {:>7} {:>8}\n",
        noise.p50, tau, noise.n, "edit", "kind", "n", "mean", "median", "viol", ">tau", "worst"
    );
    for e in &summaries {
        table += &format!(
            "{:<16} {:>6} {:>6} {:>+8.4} {:>+8.4} {:>6.1}% {:>6.1}% {:>+8.4}\n",
            e.edit,
            if e.strict {
                "strict"
            } else if e.soft {
                "soft"
            } else {
                "near"
            },
            e.n,
            e.gain.mean,
            e.gain.p50,
            100.0 * e.violation,
            100.0 * e.violation_gt_tau,
            e.gain.min
        );
    }
    Ok(SuiteOutput {
        summary: serde_json::json!({ "tau": tau, "tau_rule": "p99 |dV| of inert-skill edits", "placebo_abs_dv": noise, "edits": summaries }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}

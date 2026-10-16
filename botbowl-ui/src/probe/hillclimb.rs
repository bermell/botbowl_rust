//! Adversarial search for skill-monotonicity violations the random sample misses. From a base
//! `Turn` decision pick the worst of a few (own player, in-distribution beneficial skill) pairs,
//! then greedily mutate the *rest* of the position — move a player one square, toggle a pooled
//! skill or nudge a characteristic on another player, stand one up or knock one down — keeping a
//! mutation only if it makes the violation `f = V(s) - V(s + skill)` (mover's frame) larger.
//! Every mutation stays inside the training distribution's ranges and leaves a legal position
//! (re-asked), so the final state is a plausible one where giving that player that skill costs
//! the most value.

use std::io;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{BallState, Direction, PlayerID, PlayerStatus, Position};
use botbowl_engine::core::table::Skill;
use botbowl_nn::perspective::mover_for;

use super::common::{ascii, at_turn_decision, dist, frac, par_map, placebo_deltas, Base, Case, Net, TopK};
use super::edits::{add_skill, bump_stat, reasked, remove_skill, set_status, translate, Stat};
use super::skills::{class_of, SkillClass};
use super::{bases, SuiteOutput};
use crate::cli::ProbeArgs;

const SCREEN_PAIRS: usize = 8;

fn violation(net: &Net, s: &GameState, id: PlayerID, skill: Skill) -> Option<f32> {
    let e = add_skill(s, id, skill).ok()?;
    Some(net.value(s) - net.value(&e))
}

fn pooled() -> Vec<Skill> {
    Skill::good_skills()
        .into_iter()
        .filter(|s| class_of(*s) == SkillClass::Beneficial)
        .collect()
}

/// One random mutation of everything but `(id, skill)`; `None` if the draw does not apply.
fn mutate(s: &GameState, id: PlayerID, skill: Skill, rng: &mut ChaCha8Rng) -> Option<(GameState, String)> {
    let players: Vec<_> = s.get_players_on_pitch().map(|p| (p.id, p.position, p.status)).collect();
    let (pid, pos, status) = *players.choose(rng)?;
    let carrier = s.ball == BallState::Carried(pid);
    let out = match rng.gen_range(0..4) {
        0 => {
            let d = *Direction::all_directions_iter().collect::<Vec<_>>().choose(rng)?;
            let to = Position::new((pos.x + d.dx, pos.y + d.dy));
            (translate(s, pid, to).ok()?, format!("move {pid} to {to:?}"))
        }
        1 if pid != id => {
            let k = *pooled().choose(rng)?;
            let e = if s.get_player_unsafe(pid).has_skill(k) {
                remove_skill(s, pid, k).ok()?
            } else {
                add_skill(s, pid, k).ok()?
            };
            (e, format!("toggle {k:?} on {pid}"))
        }
        2 if pid != id => {
            let st = *Stat::ALL.choose(rng)?;
            let d = if rng.gen::<bool>() { 1 } else { -1 };
            (bump_stat(s, pid, st, d).ok()?, format!("{st:?}{d:+} on {pid}"))
        }
        3 if pid != id && !carrier => {
            let to = if status == PlayerStatus::Up {
                PlayerStatus::Down
            } else {
                PlayerStatus::Up
            };
            (set_status(s, pid, to).ok()?, format!("{pid} {to:?}"))
        }
        _ => return None,
    };
    let _ = skill;
    Some((reasked(out.0).ok()?, out.1))
}

#[derive(Serialize, Clone)]
pub struct Row {
    pub base: String,
    pub player: PlayerID,
    pub skill: String,
    /// The violation at the base, and after the climb (mover's frame, `V(s) - V(s + skill)`).
    pub f0: f32,
    pub f: f32,
    pub accepted: usize,
    pub mutations: Vec<String>,
    /// The largest placebo `|ΔV|` at the base.
    pub noise: f32,
}

fn climb(net: &Net, b: &Base, steps: usize, seed: u64) -> Option<(Row, Case, Vec<f32>)> {
    let s0 = &b.state;
    let m = mover_for(s0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pairs: Vec<(PlayerID, Skill)> = s0
        .get_players_on_pitch_in_team(m)
        .flat_map(|p| {
            pooled()
                .into_iter()
                .filter(|k| !p.has_skill(*k))
                .map(move |k| (p.id, k))
        })
        .collect();
    pairs.shuffle(&mut rng);
    let (id, skill, f0) = pairs
        .into_iter()
        .take(SCREEN_PAIRS)
        .filter_map(|(id, k)| Some((id, k, violation(net, s0, id, k)?)))
        .max_by(|a, b| a.2.total_cmp(&b.2))?;
    let mut cur = s0.clone();
    let mut f = f0;
    let mut mutations = Vec::new();
    for _ in 0..steps {
        let Some((cand, what)) = mutate(&cur, id, skill, &mut rng) else {
            continue;
        };
        let Some(fc) = violation(net, &cand, id, skill) else {
            continue;
        };
        if fc > f {
            cur = cand;
            f = fc;
            mutations.push(what);
        }
    }
    let placebo = placebo_deltas(net, s0, net.value(s0));
    let noise = placebo.iter().copied().fold(0.0, f32::max);
    let row = Row {
        base: b.id.clone(),
        player: id,
        skill: format!("{skill:?}"),
        f0,
        f,
        accepted: mutations.len(),
        mutations,
        noise,
    };
    let pos = cur.get_player_unsafe(id).position;
    let after = add_skill(&cur, id, skill).expect("the climb never gives the player the skill");
    let mut value = serde_json::to_value(&row).expect("row");
    value["base_board"] = serde_json::json!(ascii(s0, &[(s0.get_player_unsafe(id).position, '!')]));
    let case = Case {
        score: f,
        label: format!("{} {skill:?} on player {id}: violation {f0:+.3} -> {f:+.3}", b.id),
        row: value,
        before: cur,
        after,
        marks: vec![(pos, '!')],
    };
    Some((row, case, placebo))
}

pub fn run(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let bases = bases(args, &at_turn_decision)?;
    eprintln!(
        "probe hillclimb: {} base states, {} proposals each",
        bases.len(),
        args.steps
    );
    let results = par_map(bases.len(), args.parallel, |i| {
        climb(
            net,
            &bases[i],
            args.steps,
            args.seed ^ (i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D),
        )
    });
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    let mut placebo = Vec::new();
    for (r, c, p) in results.into_iter().flatten() {
        rows.push(r);
        worst.push(c);
        placebo.extend(p);
    }
    let tau = dist(&placebo).p99;
    let f0 = dist(&rows.iter().map(|r| r.f0).collect::<Vec<_>>());
    let f = dist(&rows.iter().map(|r| r.f).collect::<Vec<_>>());
    let acc = dist(&rows.iter().map(|r| r.accepted as f32).collect::<Vec<_>>());
    let mut table = format!(
        "{} climbs, {} proposals each; tau (p99 placebo |dV| at the bases) {tau:.4}\n",
        rows.len(),
        args.steps
    );
    for (name, d) in [("violation at the base", &f0), ("violation after the climb", &f)] {
        table += &format!(
            "{name:<26} mean {:+.4}  p50 {:+.4}  p90 {:+.4}  max {:+.4}\n",
            d.mean, d.p50, d.p90, d.max
        );
    }
    table += &format!(
        "violation > tau: base {:.1}%, climbed {:.1}%; mutations kept: mean {:.1}, max {:.0}\n",
        100.0 * frac(rows.iter().map(|r| r.f0 > tau)),
        100.0 * frac(rows.iter().map(|r| r.f > tau)),
        acc.mean,
        acc.max
    );
    let mut by_skill = std::collections::BTreeMap::new();
    for r in &rows {
        by_skill.entry(r.skill.clone()).or_insert_with(Vec::new).push(r.f);
    }
    for (k, v) in &by_skill {
        let d = dist(v);
        table += &format!(
            "  {k:<10} n {:>3}  climbed mean {:+.4}  max {:+.4}\n",
            d.n, d.mean, d.max
        );
    }
    Ok(SuiteOutput {
        summary: serde_json::json!({ "tau": tau, "f0": f0, "f": f, "accepted": acc,
            "gt_tau_base": frac(rows.iter().map(|r| r.f0 > tau)), "gt_tau_climbed": frac(rows.iter().map(|r| r.f > tau)) }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}

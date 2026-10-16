//! Suite 2, the lone mover (the user's design). A player of the mover's stands in his own back
//! field, three or more squares from everyone and from the ball, which is in the opponent's half:
//! nothing presses him, and wherever he starts there should be one square the net (and the
//! search) thinks is best for him. Translate his start square over the back field, all else
//! equal, activate him (`StartMove`) and record where he ends the move. Changing the destination
//! with the start square, when the same square is reachable from both, is the inconsistency.
//!
//! **How a move is represented.** A `Move` action names a destination and the engine walks the
//! path there in one step. The search's pruning (`pruning.rs`, P1 + P8) forbids ending the
//! activation before moving and, after one move, re-moving, so the activation is exactly one
//! destination choice: the "destination" is that `Move`'s square (the policy's argmax, or the
//! search's pick), and the final square is where the player stands when the activation ends
//! (the same unless a GFI fails — one fixed dice seed per scenario).
//!
//! **Is it the policy or the value?** Every candidate destination X gets a start-independent
//! value: the base position with the player put on X and marked used (`V_X`, one forward each).
//! Per start, the value's best reachable square is `argmax V_X` over that start's move fan;
//! `follow_value` is how often the agent goes there, `regret` the value it leaves, and `v_range`
//! how much the value thinks the agent's different destinations differ. If the agent changes its
//! mind but `v_range` is under the noise level, the value is indifferent and the policy is
//! breaking ties; if `v_range` is large, the policy disagrees with its own value.

use std::collections::{BTreeMap, HashMap};
use std::io;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::{Action, PlayerID, PlayerStatus, Position};
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::SearchBudget;
use botbowl_nn::perspective::mover_for;
use botbowl_play::bots::{load_mcts_config, make_mcts, Evaluator, SearchConfig};
use botbowl_play::policy::{search_legal, PolicyBot};

use super::common::{at_turn_decision, attack_dx, cheb, frac, is_probe_decision, par_map, Base, Case, Net, TopK};
use super::edits::{reasked, translate};
use super::{bases, SuiteOutput};
use crate::cli::ProbeArgs;

/// The back field: the four columns in front of the mover's own endzone.
const BACK_COLUMNS: i8 = 4;
/// How far (Chebyshev) a start square stays from every other player and the ball.
const ISOLATION: i8 = 3;

pub struct Scenario {
    pub player: PlayerID,
    pub starts: Vec<Position>,
}

/// The scenario a `Turn` decision offers, if any: the ball in the opponent's half, a free own
/// player (standing, unused, not carrying, unmarked; the one furthest from the ball), and at
/// least four isolated back-field squares.
pub fn scenario_of(s: &GameState, max_starts: usize, seed: u64) -> Option<Scenario> {
    if !at_turn_decision(s) {
        return None;
    }
    let m = mover_for(s);
    let dims = s.board_dims;
    let ball = s.get_ball_position()?;
    let own_end = dims.own_endzone_x(m);
    let score_end = dims.endzone_x(m);
    if (ball.x - score_end).abs() >= (ball.x - own_end).abs() {
        return None;
    }
    let carrier = match s.ball {
        botbowl_engine::core::model::BallState::Carried(id) => Some(id),
        _ => None,
    };
    let unmarked = |p: Position| {
        !s.get_players_on_pitch()
            .any(|q| q.stats.team != m && q.status == PlayerStatus::Up && cheb(q.position, p) == 1)
    };
    let player = s
        .get_players_on_pitch_in_team(m)
        .filter(|p| p.status == PlayerStatus::Up && !p.used && Some(p.id) != carrier && unmarked(p.position))
        .max_by_key(|p| (cheb(p.position, ball), std::cmp::Reverse(p.id)))?
        .id;
    let dx = attack_dx(dims, m);
    let mut starts: Vec<Position> = Vec::new();
    for k in 1..=BACK_COLUMNS {
        for y in 1..dims.height - 1 {
            let q = Position::new((own_end + dx * k, y));
            if dims.is_out(q) || cheb(q, ball) < ISOLATION {
                continue;
            }
            let clear = s
                .get_players_on_pitch()
                .all(|o| o.id == player || cheb(o.position, q) >= ISOLATION);
            if clear {
                starts.push(q);
            }
        }
    }
    if starts.len() < 4 {
        return None;
    }
    starts.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    starts.truncate(max_starts);
    starts.sort();
    Some(Scenario { player, starts })
}

/// The player on `x`, marked used: the position "after he moved there", independent of where he
/// started (and of how far he walked).
fn landed(base: &GameState, id: PlayerID, x: Position) -> Option<GameState> {
    let mut t = translate(base, id, x).ok()?;
    t.get_mut_player_unsafe(id).used = true;
    reasked(t).ok()
}

/// The move decision for a start: the base with the player on `q`, activated.
fn activated(base: &GameState, id: PlayerID, q: Position, seed: u64) -> Option<GameState> {
    let mut s = reasked(translate(base, id, q).ok()?).ok()?;
    let act = Action::Positional(PosAT::StartMove, q);
    if !s.is_legal_action(&act) {
        return None;
    }
    s.set_dice_mode(DiceMode::RollDice);
    s.set_seed(seed);
    s.step(act).ok()?;
    (s.info.active_player == Some(id) && s.available_actions.has_paths()).then_some(s)
}

struct Played {
    first: Option<Position>,
    last: Position,
    fell: bool,
    end: GameState,
}

/// Let `agent` play the rest of the activation.
fn play_activation(agent: &mut dyn Bot, start: &GameState, id: PlayerID) -> Played {
    let mut st = start.clone();
    let mut first = None;
    for _ in 0..8 {
        if st.info.active_player != Some(id) || !is_probe_decision(&st) {
            break;
        }
        let a = agent.get_action(&st);
        if first.is_none() {
            if let Action::Positional(PosAT::Move, t) = a {
                first = Some(t);
            }
        }
        if st.step(a).is_err() {
            break;
        }
    }
    let (last, fell) = match st.get_player(id) {
        Ok(p) => (p.position, p.status != PlayerStatus::Up),
        Err(_) => (Position::new((0, 0)), true),
    };
    Played {
        first,
        last,
        fell,
        end: st,
    }
}

#[derive(Serialize, Clone)]
pub struct Row {
    pub scenario: String,
    pub start: (i8, i8),
    pub agent: String,
    pub first: Option<(i8, i8)>,
    pub last: (i8, i8),
    pub fell: bool,
    /// Squares the move fan offers from this start.
    pub fan: usize,
    /// The value's best square among them, and its `V_X`.
    pub value_best: (i8, i8),
    pub v_best: f32,
    /// `V_X` of where the agent went (`None` if he fell or left the board).
    pub v_last: Option<f32>,
    /// The net's value of the real position after the activation, mover's frame.
    pub v_end: f32,
}

#[derive(Serialize, Clone, Default)]
pub struct AgentScenario {
    pub agent: String,
    pub starts: usize,
    /// The most common destination, and in how many starts it was reachable.
    pub mode: (i8, i8),
    pub mode_reachable: usize,
    /// Share of the starts that reach the mode which went there; mean Chebyshev distance of the
    /// destination from the mode over the same starts; distinct destinations over all starts.
    pub agree: f32,
    pub spread: f32,
    pub distinct: usize,
    /// Share of starts that went to the value's best reachable square; the mean `V_best -
    /// V_last`; `max - min` of `V_X` over the agent's distinct destinations.
    pub follow_value: f32,
    pub regret: f32,
    pub v_range: f32,
    pub fell: usize,
}

fn key(p: Position) -> (i8, i8) {
    (p.x, p.y)
}

fn agent_summary(
    agent: &str,
    rows: &[&Row],
    reach: &HashMap<(i8, i8), Vec<Position>>,
    vx: &HashMap<Position, f32>,
) -> AgentScenario {
    let mut count: BTreeMap<(i8, i8), usize> = BTreeMap::new();
    for r in rows.iter().filter(|r| !r.fell) {
        *count.entry(r.last).or_default() += 1;
    }
    let vof = |k: (i8, i8)| vx.get(&Position::new(k)).copied().unwrap_or(f32::NEG_INFINITY);
    let Some(mode) = count
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(vof(*a.0).total_cmp(&vof(*b.0))))
        .map(|(k, _)| *k)
    else {
        return AgentScenario {
            agent: agent.into(),
            starts: rows.len(),
            fell: rows.len(),
            ..Default::default()
        };
    };
    let mode_pos = Position::new(mode);
    let reachable: Vec<&&Row> = rows
        .iter()
        .filter(|r| !r.fell && reach[&r.start].contains(&mode_pos))
        .collect();
    let lasts: Vec<f32> = count.keys().map(|k| vof(*k)).filter(|v| v.is_finite()).collect();
    AgentScenario {
        agent: agent.into(),
        starts: rows.len(),
        mode,
        mode_reachable: reachable.len(),
        agree: frac(reachable.iter().map(|r| r.last == mode)),
        spread: reachable
            .iter()
            .map(|r| cheb(Position::new(r.last), mode_pos) as f32)
            .sum::<f32>()
            / reachable.len().max(1) as f32,
        distinct: count.len(),
        follow_value: frac(rows.iter().map(|r| r.last == r.value_best)),
        regret: {
            let r: Vec<f32> = rows.iter().filter_map(|r| r.v_last.map(|v| r.v_best - v)).collect();
            r.iter().sum::<f32>() / r.len().max(1) as f32
        },
        v_range: lasts.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - lasts.iter().cloned().fold(f32::INFINITY, f32::min),
        fell: rows.iter().filter(|r| r.fell).count(),
    }
}

struct ScenarioResult {
    rows: Vec<Row>,
    agents: Vec<AgentScenario>,
    noise: Vec<f32>,
    case: Option<Case>,
}

fn run_scenario(net: &Net, b: &Base, sc: &Scenario, searches: &[(String, SearchConfig)], seed: u64) -> ScenarioResult {
    let base = &b.state;
    let m = mover_for(base);
    let noise = super::common::placebo_deltas(net, base, net.value(base));
    let mut decisions: Vec<(Position, GameState, Vec<Position>)> = Vec::new();
    for &q in &sc.starts {
        let Some(d) = activated(base, sc.player, q, seed) else {
            continue;
        };
        let fan: Vec<Position> = search_legal(&d)
            .into_iter()
            .filter_map(|a| match a {
                Action::Positional(PosAT::Move, t) => Some(t),
                _ => None,
            })
            .collect();
        if !fan.is_empty() {
            decisions.push((q, d, fan));
        }
    }
    let mut vx: HashMap<Position, f32> = HashMap::new();
    for (_, _, fan) in &decisions {
        for &x in fan {
            if let std::collections::hash_map::Entry::Vacant(e) = vx.entry(x) {
                if let Some(t) = landed(base, sc.player, x) {
                    e.insert(net.value_for(&t, m));
                }
            }
        }
    }
    let reach: HashMap<(i8, i8), Vec<Position>> = decisions.iter().map(|(q, _, f)| (key(*q), f.clone())).collect();
    let mut rows = Vec::new();
    let mut ends: HashMap<(String, (i8, i8)), GameState> = HashMap::new();
    let agent_names: Vec<String> = std::iter::once("policy".to_string())
        .chain(searches.iter().map(|(n, _)| n.clone()))
        .collect();
    for (q, d, fan) in &decisions {
        let (best, v_best) = fan
            .iter()
            .filter_map(|x| vx.get(x).map(|v| (*x, *v)))
            .fold((*q, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
        for name in &agent_names {
            let played = if name == "policy" {
                play_activation(&mut PolicyBot::new(net.nn.clone()), d, sc.player)
            } else {
                let cfg = &searches.iter().find(|(n, _)| n == name).expect("named search").1;
                let mut bot = make_mcts(cfg, Evaluator::Nn, Some(&net.nn));
                play_activation(&mut bot, d, sc.player)
            };
            let v_last = (!played.fell).then(|| vx.get(&played.last).copied()).flatten();
            rows.push(Row {
                scenario: b.id.clone(),
                start: key(*q),
                agent: name.clone(),
                first: played.first.map(key),
                last: key(played.last),
                fell: played.fell,
                fan: fan.len(),
                value_best: key(best),
                v_best,
                v_last,
                v_end: net.value_for(&played.end, m),
            });
            ends.insert((name.clone(), key(*q)), played.end);
        }
    }
    let agents: Vec<AgentScenario> = agent_names
        .iter()
        .map(|n| {
            let rs: Vec<&Row> = rows.iter().filter(|r| &r.agent == n).collect();
            agent_summary(n, &rs, &reach, &vx)
        })
        .collect();
    // The case to look at: the last agent's (largest budget) disagreements, drawn on the base.
    let case = agents.last().map(|a| {
        let mut marks: Vec<(Position, char)> = Vec::new();
        for r in rows.iter().filter(|r| r.agent == a.agent) {
            marks.push((Position::new(r.start), if r.last == a.mode { 's' } else { 'o' }));
        }
        let mut dests: Vec<(i8, i8)> = rows.iter().filter(|r| r.agent == a.agent).map(|r| r.last).collect();
        dests.sort();
        dests.dedup();
        for dpos in dests {
            marks.push((Position::new(dpos), if dpos == a.mode { 'M' } else { 'x' }));
        }
        let pairs: Vec<_> = rows
            .iter()
            .filter(|r| r.agent == a.agent)
            .map(|r| serde_json::json!({"start": r.start, "last": r.last, "value_best": r.value_best}))
            .collect();
        let worst_start = rows
            .iter()
            .filter(|r| r.agent == a.agent && r.last != a.mode)
            .map(|r| r.start)
            .next()
            .unwrap_or(a.mode);
        Case {
            score: 1.0 - a.agree,
            label: format!(
                "{} {}: agree {:.2} over {} starts, {} destinations",
                b.id, a.agent, a.agree, a.mode_reachable, a.distinct
            ),
            row: serde_json::json!({"summary": a, "pairs": pairs,
                "legend": "s/o: start squares going to the mode / elsewhere; M: the mode; x: other destinations"}),
            before: base.clone(),
            after: ends
                .get(&(a.agent.clone(), worst_start))
                .cloned()
                .unwrap_or_else(|| base.clone()),
            marks,
        }
    });
    ScenarioResult {
        rows,
        agents,
        noise,
        case,
    }
}

pub fn run(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let preset = load_mcts_config(&args.search_config)?;
    let searches: Vec<(String, SearchConfig)> = args
        .budgets
        .iter()
        .map(|&b| {
            (
                format!("search@{b}"),
                SearchConfig {
                    budget: SearchBudget::Iterations(b),
                    workers: 1,
                    puct: None,
                    horizon_turns: None,
                    fpu_reduction: None,
                    config: Some(preset.config),
                },
            )
        })
        .collect();
    let starts = args.starts;
    let seed = args.seed;
    let bases = bases(args, &|s| scenario_of(s, starts, seed).is_some())?;
    eprintln!(
        "probe lone-mover: {} scenarios, up to {starts} starts, agents policy + {:?} ({})",
        bases.len(),
        args.budgets,
        preset.name
    );
    let results = par_map(bases.len(), args.parallel, |i| {
        let sc = scenario_of(&bases[i].state, starts, seed).expect("filtered");
        let r = run_scenario(net, &bases[i], &sc, &searches, seed ^ i as u64);
        eprintln!("  scenario {}/{} ({})", i + 1, bases.len(), bases[i].id);
        r
    });
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    let mut per_agent: BTreeMap<String, Vec<AgentScenario>> = BTreeMap::new();
    let mut noise = Vec::new();
    for r in results {
        rows.extend(r.rows);
        noise.extend(r.noise);
        for a in r.agents {
            per_agent.entry(a.agent.clone()).or_default().push(a);
        }
        worst.extend(r.case);
    }
    let tau = super::common::dist(&noise).p99;
    let mean = |xs: &[AgentScenario], f: &dyn Fn(&AgentScenario) -> f32| {
        let v: Vec<f32> = xs.iter().map(f).filter(|x| x.is_finite()).collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    let mut table = format!(
        "{} scenarios; tau (p99 placebo |dV| at the bases) {tau:.4}\n{:<12} {:>7} {:>9} {:>7} {:>9} {:>8} {:>8} {:>8} {:>10}\n",
        per_agent.values().next().map_or(0, |v| v.len()),
        "agent",
        "agree",
        "all-agree",
        "spread",
        "distinct",
        "follow-V",
        "regret",
        "v_range",
        "v_range>tau"
    );
    let mut summary = serde_json::Map::new();
    for (agent, xs) in &per_agent {
        let all_agree = frac(xs.iter().map(|a| a.agree >= 0.999));
        let range_gt = frac(xs.iter().map(|a| a.v_range > tau));
        table += &format!(
            "{:<12} {:>6.1}% {:>8.1}% {:>7.2} {:>9.2} {:>7.1}% {:>8.4} {:>8.4} {:>9.1}%\n",
            agent,
            100.0 * mean(xs, &|a| a.agree),
            100.0 * all_agree,
            mean(xs, &|a| a.spread),
            mean(xs, &|a| a.distinct as f32),
            100.0 * mean(xs, &|a| a.follow_value),
            mean(xs, &|a| a.regret),
            mean(xs, &|a| a.v_range),
            100.0 * range_gt,
        );
        summary.insert(
            agent.clone(),
            serde_json::json!({
                "scenarios": xs.len(),
                "agree": mean(xs, &|a| a.agree),
                "all_agree": all_agree,
                "spread": mean(xs, &|a| a.spread),
                "distinct": mean(xs, &|a| a.distinct as f32),
                "follow_value": mean(xs, &|a| a.follow_value),
                "regret": mean(xs, &|a| a.regret),
                "v_range": mean(xs, &|a| a.v_range),
                "v_range_gt_tau": range_gt,
                "fell": xs.iter().map(|a| a.fell).sum::<usize>(),
                "per_scenario": xs,
            }),
        );
    }
    // Agents against each other: how often two agents pick the same destination from one start.
    let mut by_start: HashMap<(String, (i8, i8)), HashMap<String, (i8, i8)>> = HashMap::new();
    for r in &rows {
        by_start
            .entry((r.scenario.clone(), r.start))
            .or_default()
            .insert(r.agent.clone(), r.last);
    }
    let names: Vec<String> = per_agent.keys().cloned().collect();
    let mut pairs = serde_json::Map::new();
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let same = frac(
                by_start
                    .values()
                    .filter_map(|m| Some(m.get(&names[i])? == m.get(&names[j])?)),
            );
            table += &format!("same destination {} vs {}: {:.1}%\n", names[i], names[j], 100.0 * same);
            pairs.insert(format!("{} vs {}", names[i], names[j]), serde_json::json!(same));
        }
    }

    Ok(SuiteOutput {
        summary: serde_json::json!({ "tau": tau, "agents": summary, "same_destination": pairs }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::edits::tests::starts;

    /// A scenario's starts are isolated back-field squares, and every one of them activates into a
    /// move decision whose fan the value landscape can score.
    #[test]
    fn scenarios_are_isolated_back_field_starts_that_activate() {
        let mut found = 0;
        for s in starts(80) {
            let Some(sc) = scenario_of(&s, 8, 1) else { continue };
            found += 1;
            let m = mover_for(&s);
            let own_end = s.board_dims.own_endzone_x(m);
            for &q in &sc.starts {
                assert!((1..=BACK_COLUMNS).contains(&(q.x - own_end).abs()), "{q:?}");
                for o in s.get_players_on_pitch().filter(|o| o.id != sc.player) {
                    assert!(cheb(o.position, q) >= ISOLATION);
                }
                let d = activated(&s, sc.player, q, 7).expect("activates");
                assert_eq!(d.get_player_unsafe(sc.player).position, q);
                assert!(search_legal(&d)
                    .iter()
                    .all(|a| matches!(a, Action::Positional(PosAT::Move, _))));
                let x = landed(&s, sc.player, q).unwrap();
                assert!(x.get_player_unsafe(sc.player).used);
            }
        }
        assert!(found >= 3, "only {found} scenarios in 80 random starts");
    }
}

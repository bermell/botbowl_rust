//! Search stability: does the search's move change with the budget, and does more budget
//! converge? For each base decision with at least two moves, a fresh search at every
//! `--budgets` value (ascending), plus a second fresh search at the middle budget: with the
//! deterministic preset (`gumbel_scale = 0`) a repeat can differ only through `recon_mcts`'s
//! HashMap order, which is the search's irreducible run-to-run noise.
//!
//! A flip between two budgets is weighed by the largest budget's opinion: the Q gap (mover's
//! frame, TD units) between the two moves at the largest budget. A flip under 0.01 is a near tie.

use std::io;

use serde::Serialize;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action, TeamType};
use botbowl_mcts::SearchBudget;
use botbowl_nn::perspective::mover_for;
use botbowl_play::bots::{load_mcts_config, make_mcts, Evaluator, SearchConfig};
use botbowl_play::policy::{in_frame, search_legal, PolicyBot};

use super::common::{dist, frac, par_map, Base, Case, Kind, Net, TopK};
use super::{bases, SuiteOutput};
use crate::cli::ProbeArgs;

const NEAR_TIE: f32 = 0.01;

#[derive(Serialize, Clone)]
pub struct Pick {
    pub budget: usize,
    pub action: String,
    /// The pick's Q in its own search, mover's frame, TD units.
    pub q: Option<f32>,
    pub visits: u32,
    pub root_value: Option<f32>,
}

#[derive(Serialize, Clone)]
pub struct Row {
    pub base: String,
    pub kind: Kind,
    pub fan: usize,
    pub policy: String,
    pub picks: Vec<Pick>,
    /// The second search at the middle budget.
    pub repeat: Pick,
    /// For each pick, the Q its move has in the **largest** budget's search, mover's frame.
    pub q_at_max: Vec<Option<f32>>,
}

fn search(
    state: &GameState,
    cfg: &SearchConfig,
    net: &Net,
    budget: usize,
) -> (Action, Pick, Vec<(Action, Option<f32>)>) {
    let mut bot = make_mcts(cfg, Evaluator::Nn, Some(&net.nn));
    let (a, rec) = bot.get_action_with_record(state);
    let mover: TeamType = mover_for(state);
    let qs: Vec<(Action, Option<f32>)> = rec
        .children
        .iter()
        .map(|c| (c.action, c.q.map(|q| in_frame(mover, q as f32 / 1000.0))))
        .collect();
    let mine = rec.children.iter().find(|c| c.action == a);
    let pick = Pick {
        budget,
        action: format!("{a:?}"),
        q: mine.and_then(|c| c.q).map(|q| in_frame(mover, q as f32 / 1000.0)),
        visits: mine.map_or(0, |c| c.visits),
        root_value: rec.root_value.map(|q| in_frame(mover, q as f32 / 1000.0)),
    };
    (a, pick, qs)
}

fn probe(net: &Net, b: &Base, configs: &[(usize, SearchConfig)], mid: usize) -> (Row, Case) {
    let s = &b.state;
    let (legal, _, best) = PolicyBot::new(net.nn.clone()).priors(s);
    let mut picks = Vec::new();
    let mut actions = Vec::new();
    let mut last_qs = Vec::new();
    for (budget, cfg) in configs {
        let (a, p, qs) = search(s, cfg, net, *budget);
        actions.push(a);
        picks.push(p);
        last_qs = qs;
    }
    let (_, repeat, _) = search(s, &configs[mid].1, net, configs[mid].0);
    let q_at_max: Vec<Option<f32>> = actions
        .iter()
        .map(|a| last_qs.iter().find(|(x, _)| x == a).and_then(|(_, q)| *q))
        .collect();
    let row = Row {
        base: b.id.clone(),
        kind: b.kind,
        fan: legal.len(),
        policy: format!("{:?}", legal[best]),
        picks,
        repeat,
        q_at_max,
    };
    let gap = row.q_at_max.last().copied().flatten().unwrap_or(0.0)
        - row.q_at_max.iter().filter_map(|q| *q).fold(f32::INFINITY, f32::min);
    let distinct = {
        let mut v: Vec<&String> = row.picks.iter().map(|p| &p.action).collect();
        v.sort();
        v.dedup();
        v.len()
    };
    let case = Case {
        score: (distinct as f32 - 1.0) + gap.max(0.0) + (row.repeat.action != row.picks[mid].action) as u8 as f32,
        label: format!("{} {distinct} distinct picks, Q gap at max {gap:.3}", b.id),
        row: serde_json::to_value(&row).expect("row"),
        before: s.clone(),
        after: s.clone(),
        marks: vec![],
    };
    (row, case)
}

pub fn run(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let preset = load_mcts_config(&args.search_config)?;
    if preset.config.gumbel_scale > 0.0 {
        eprintln!(
            "warning: {} draws Gumbel noise; repeats then measure the noise too",
            preset.name
        );
    }
    let mut budgets = args.budgets.clone();
    budgets.sort();
    budgets.dedup();
    if budgets.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "stability needs two or more --budgets",
        ));
    }
    let configs: Vec<(usize, SearchConfig)> = budgets
        .iter()
        .map(|&b| {
            (
                b,
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
    let mid = (configs.len() - 1) / 2;
    let bases = bases(args, &|s| search_legal(s).len() >= 2)?;
    eprintln!(
        "probe stability: {} base decisions, budgets {budgets:?} ({}), repeat at {}",
        bases.len(),
        preset.name,
        budgets[mid]
    );
    let results = par_map(bases.len(), args.parallel, |i| {
        let r = probe(net, &bases[i], &configs, mid);
        if (i + 1) % 10 == 0 {
            eprintln!("  {}/{}", i + 1, bases.len());
        }
        r
    });
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    for (r, c) in results {
        rows.push(r);
        worst.push(c);
    }
    let k = budgets.len();
    let mut table = format!(
        "{} decisions, budgets {budgets:?}, near tie < {NEAR_TIE} Q at the largest budget\n",
        rows.len()
    );
    let mut pairs = serde_json::Map::new();
    for i in 0..k {
        for j in i + 1..k {
            let flips: Vec<&Row> = rows.iter().filter(|r| r.picks[i].action != r.picks[j].action).collect();
            let gaps: Vec<f32> = flips
                .iter()
                .filter_map(|r| Some((r.q_at_max[j]? - r.q_at_max[i]?).abs()))
                .collect();
            let near = frac(gaps.iter().map(|g| *g < NEAR_TIE));
            let rate = flips.len() as f32 / rows.len().max(1) as f32;
            table += &format!(
                "flip {:>5} vs {:>5}: {:>5.1}%   |Q gap| at max: median {:.4}, near ties {:.0}%\n",
                budgets[i],
                budgets[j],
                100.0 * rate,
                dist(&gaps).p50,
                100.0 * near
            );
            pairs.insert(
                format!("{}_{}", budgets[i], budgets[j]),
                serde_json::json!({"flip": rate, "gap": dist(&gaps), "near_tie": near}),
            );
        }
    }
    let repeat = frac(rows.iter().map(|r| r.repeat.action != r.picks[mid].action));
    table += &format!("repeat at {} flips: {:.1}%\n", budgets[mid], 100.0 * repeat);
    let mut vs_policy = serde_json::Map::new();
    for (i, b) in budgets.iter().enumerate() {
        let agree = frac(rows.iter().map(|r| r.picks[i].action == r.policy));
        table += &format!("search@{b} plays the policy's argmax: {:.1}%\n", 100.0 * agree);
        vs_policy.insert(b.to_string(), serde_json::json!(agree));
    }
    let mut by_kind = serde_json::Map::new();
    for kd in [Kind::TurnStart, Kind::TurnMid, Kind::Activation, Kind::Other] {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.kind == kd).collect();
        if !rs.is_empty() {
            let f = frac(rs.iter().map(|r| r.picks[0].action != r.picks[k - 1].action));
            table += &format!(
                "  {kd:?}: n {}, flip {} vs {} {:.1}%\n",
                rs.len(),
                budgets[0],
                budgets[k - 1],
                100.0 * f
            );
            by_kind.insert(format!("{kd:?}"), serde_json::json!({"n": rs.len(), "flip_min_max": f}));
        }
    }
    Ok(SuiteOutput {
        summary: serde_json::json!({ "budgets": budgets, "pairs": pairs, "repeat_flip": repeat,
            "agree_with_policy": vs_policy, "by_kind": by_kind }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}

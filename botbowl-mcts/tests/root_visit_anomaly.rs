//! Diagnostic (2026-09-30, `#[ignore]`d): a web-app search at a turn start spent most of its visits
//! on Hand-off / Pass children with ~0% prior and negative Q, while the 53%-prior Move that won got
//! fewer. Search turn-start positions with gen21's net and print every root child, under 1 and 8
//! workers and under both roll models, to see which knob produces it.
//!
//! cargo test --release -p botbowl-mcts --test root_visit_anomaly -- --ignored --nocapture

mod common;

use std::sync::Arc;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::{ChanceModel, MctsBot, MctsConfig, SearchBudget};
use botbowl_nn::eval::NnEvaluator;

const MODEL: &str = "../models/az_v7/bbnet_mix16x9_gen21.onnx";

fn has(state: &GameState, at: PosAT) -> bool {
    state
        .get_all_actions()
        .iter()
        .any(|a| matches!(a, EngineAction::Positional(t, _) if *t == at))
}

fn turn_starts(n: usize) -> Vec<GameState> {
    common::states(400, 5)
        .into_iter()
        .filter(|s| s.proc_stack_top() == Some("Turn") && has(s, PosAT::StartHandoff) && has(s, PosAT::StartMove))
        .take(n)
        .collect()
}

fn search(state: &GameState, nn: &Arc<NnEvaluator>, workers: usize, model: ChanceModel) {
    search_vl(state, nn, workers, model, None)
}

fn search_vl(state: &GameState, nn: &Arc<NnEvaluator>, workers: usize, model: ChanceModel, vl: Option<i32>) {
    let mut cfg = MctsConfig::new();
    cfg.workers = workers;
    if let Some(v) = vl {
        cfg.virtual_loss = v;
    }
    cfg.chance_model = model;
    cfg.trace_root_descents = true;
    let mut bot = MctsBot::with_budget_and_config(SearchBudget::Iterations(2000), cfg).with_evaluator(nn.clone());
    let played = botbowl_engine::bots::Bot::get_action(&mut bot, state);
    let s = bot.last_search().unwrap();
    let total: u32 = s.children.iter().map(|e| e.stats.visits).sum();
    println!(
        "  workers={workers} model={model:?} vl={} root visits {} (children sum {total}) root Q {:?} played {played:?}",
        cfg_vl(vl),
        s.root.visits,
        s.root.q_agent
    );
    let descents = s.root_descents.clone().unwrap_or_default();
    let dsum: u32 = descents.iter().map(|(_, n)| n).sum();
    println!("    root descents traced: {dsum}");
    for e in s.children.iter().take(8) {
        let d = match &e.action {
            botbowl_mcts::BbAction::Player { action, .. } => {
                descents.iter().find(|(a, _)| a == action).map_or(0, |x| x.1)
            }
            _ => 0,
        };
        println!(
            "    {:>28}  prior {:6.3}  descents {d:5}  visits {:5}  Q {:>7}  solved {}",
            match &e.action {
                botbowl_mcts::BbAction::Player { action, .. } => format!("{action:?}"),
                a => format!("{a:?}"),
            },
            e.prior().unwrap_or(0.0),
            e.stats.visits,
            e.stats.q_agent.map_or("-".into(), |q| format!("{q:+.3}")),
            e.stats.solved
        );
    }
}

#[test]
#[ignore]
fn where_do_the_root_visits_go() {
    let nn = Arc::new(NnEvaluator::from_path(MODEL).expect("gen21 net"));
    for (i, state) in turn_starts(3).iter().enumerate() {
        println!("== position {i}: {} legal", state.get_all_actions().len());
        for (workers, model) in [
            (8, ChanceModel::Exact),
            (1, ChanceModel::Exact),
            (8, ChanceModel::Legacy),
            (1, ChanceModel::Legacy),
        ] {
            search(state, &nn, workers, model);
        }
    }
}

/// Grow one tree 100 descents at a time (tree reuse on the same root) and print how each root
/// child's Q and visits evolve — whether the heavy-visit children were ever the best by Q.
#[test]
#[ignore]
fn how_the_visits_accumulate() {
    let nn = Arc::new(NnEvaluator::from_path(MODEL).expect("gen21 net"));
    let state = turn_starts(3).remove(2);
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.chance_model = ChanceModel::Legacy;
    let mut bot = MctsBot::with_budget_and_config(SearchBudget::Iterations(100), cfg).with_evaluator(nn.clone());
    let name = |e: &botbowl_mcts::Edge| match &e.action {
        botbowl_mcts::BbAction::Player { action, .. } => format!("{action:?}"),
        a => format!("{a:?}"),
    };
    let watch: Vec<String> = [
        "StartBlock (7,7)",
        "StartMove (7,7)",
        "StartBlitz (7,7)",
        "StartMove (2,5)",
        "StartBlitz (2,5)",
        "StartHandoff (2,5)",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    println!(
        "{:>6} {}",
        "iter",
        watch.iter().map(|w| format!("{w:>22}")).collect::<String>()
    );
    for step in 1..=20 {
        botbowl_engine::bots::Bot::get_action(&mut bot, &state);
        let s = bot.last_search().unwrap();
        let cells: String = watch
            .iter()
            .map(|w| match s.children.iter().find(|e| name(e) == *w) {
                Some(e) => format!(
                    "{:>22}",
                    format!(
                        "{:5}v {:>7}",
                        e.stats.visits,
                        e.stats.q_agent.map_or("-".into(), |q| format!("{q:+.3}"))
                    )
                ),
                None => format!("{:>22}", "?"),
            })
            .collect();
        println!(
            "{:>6} {cells}   root {:?} {:?}",
            step * 100,
            s.reuse.outcome,
            s.root.q_agent
        );
    }
}

/// One descent per call: which root child's visit count moves, and does each call run a descent?
#[test]
#[ignore]
fn one_descent_at_a_time() {
    let nn = Arc::new(NnEvaluator::from_path(MODEL).expect("gen21 net"));
    let state = turn_starts(3).remove(2);
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.chance_model = ChanceModel::Legacy;
    let mut bot = MctsBot::with_budget_and_config(SearchBudget::Iterations(1), cfg).with_evaluator(nn.clone());
    let name = |e: &botbowl_mcts::Edge| match &e.action {
        botbowl_mcts::BbAction::Player { action, .. } => format!("{action:?}"),
        a => format!("{a:?}"),
    };
    let mut prev: std::collections::HashMap<String, u32> = Default::default();
    let mut moved: std::collections::BTreeMap<String, (u32, i64)> = Default::default();
    let (mut no_change, mut calls) = (0, 0);
    let mut prev_root = 0u32;
    for i in 0..600 {
        let before_iters = bot.telemetry().iterations;
        botbowl_engine::bots::Bot::get_action(&mut bot, &state);
        let ran = bot.telemetry().iterations - before_iters;
        let s = bot.last_search().unwrap();
        let mut changed = vec![];
        for e in &s.children {
            let n = name(e);
            let v = e.stats.visits;
            let p = prev.insert(n.clone(), v).unwrap_or(0);
            if v != p && i > 0 {
                let m = moved.entry(n.clone()).or_default();
                m.0 += 1;
                m.1 += v as i64 - p as i64;
                changed.push(format!("{n}{:+}", v as i64 - p as i64));
            }
        }
        if i > 0 {
            calls += 1;
            if changed.is_empty() {
                no_change += 1;
            }
            if i < 40 || (i % 100 == 0) {
                println!(
                    "desc {i:3} ran {ran} root {}->{} changed {:?}",
                    prev_root, s.root.visits, changed
                );
            }
        }
        prev_root = s.root.visits;
    }
    println!("{no_change} of {calls} single descents moved no root child's visits");
    for (n, (k, d)) in &moved {
        println!("  {n:>24}: moved in {k:4} descents, net {d:+}");
    }
}

fn cfg_vl(vl: Option<i32>) -> String {
    vl.map_or("default".into(), |v| v.to_string())
}

/// The virtual-loss-leak hypothesis: with virtual loss off, does selection follow PUCT again?
#[test]
#[ignore]
fn virtual_loss_off() {
    let nn = Arc::new(NnEvaluator::from_path(MODEL).expect("gen21 net"));
    for (i, state) in turn_starts(3).iter().enumerate() {
        println!("== position {i}");
        search_vl(state, &nn, 1, ChanceModel::Exact, Some(30));
        search_vl(state, &nn, 1, ChanceModel::Exact, Some(0));
    }
}

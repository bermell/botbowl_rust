//! Invariances whose true effect is exactly zero:
//!
//! - **y-flip** (`GameState::y_flipped`, the top-bottom mirror). The rules are symmetric and the
//!   trainer augments with it; the encoder maps it to the exact reflected tensor (pinned in
//!   `edits.rs`), so `V(flip s) - V(s)` and the policy's total variation between `s` and the
//!   mirrored moves of `flip s` are purely the net's.
//! - **colour swap** (`GameState::mirrored`, x-mirror + Home/Away): the encoder canonicalises it
//!   away, so the mover-centric value must be bit-identical. A canary for the perspective code,
//!   measured on states without path offerings (the mirror drops them).
//! - **a used player's movement** (`moves` of a player who has finished his activation): nothing
//!   reads it until his next turn resets it, but the encoder shows it (`movement_left`).

use std::collections::BTreeMap;
use std::io;

use serde::Serialize;

use botbowl_engine::core::model::{Action, Position};
use botbowl_play::policy::search_legal;

use super::common::{dist, frac, par_map, Base, Case, Kind, Net, TopK};
use super::edits::set_moves;
use super::{bases, SuiteOutput};
use crate::cli::ProbeArgs;

#[derive(Serialize, Clone)]
pub struct Row {
    pub base: String,
    pub kind: Kind,
    pub v: f32,
    /// `V(y_flipped s) - V(s)`, mover's frame.
    pub dv_flip: f32,
    /// Policy legal-set size, total variation between `π(s)` and the flipped `π(flip s)`, and
    /// whether the argmax is the mirrored move.
    pub fan: usize,
    pub tv_flip: Option<f32>,
    pub argmax_agree: Option<bool>,
    /// `V(mirrored s) - V(s)` (mover-centric, so 0 exactly), states without path offerings only.
    pub dv_colour: Option<f32>,
    /// Largest `|ΔV|` over the used players' movement set to `0` or `MA + 2`.
    pub dv_used_moves: Option<f32>,
    pub used_players: usize,
}

fn flip(a: Action, h: i8) -> Action {
    match a {
        Action::Positional(at, p) => Action::Positional(at, Position::new((p.x, h - 1 - p.y))),
        s => s,
    }
}

fn probe(net: &Net, b: &Base) -> (Row, Option<Case>) {
    let s = &b.state;
    let v = net.value(s);
    let f = s.y_flipped();
    let dv_flip = net.value(&f) - v;
    let legal = search_legal(s);
    let (tv_flip, argmax_agree) = if legal.len() >= 2 {
        let h = s.board_dims.height;
        let flipped: Vec<Action> = legal.iter().map(|a| flip(*a, h)).collect();
        let (p, q) = (net.probs(s, &legal), net.probs(&f, &flipped));
        let tv = 0.5 * p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f32>();
        let am = |x: &[f32]| (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b });
        (Some(tv), Some(am(&p) == am(&q)))
    } else {
        (None, None)
    };
    let dv_colour = (!s.available_actions.has_paths()).then(|| net.value(&s.mirrored()) - v);
    let used: Vec<_> = s
        .get_players_on_pitch()
        .filter(|p| p.used && s.info.active_player != Some(p.id))
        .map(|p| (p.id, p.moves, p.stats.ma + 2))
        .collect();
    let mut dv_used_moves: Option<f32> = None;
    for &(id, moves, cap) in &used {
        for m in [0, cap] {
            if m == moves {
                continue;
            }
            if let Ok(e) = set_moves(s, id, m) {
                let d = (net.value(&e) - v).abs();
                dv_used_moves = Some(dv_used_moves.map_or(d, |x: f32| x.max(d)));
            }
        }
    }
    let row = Row {
        base: b.id.clone(),
        kind: b.kind,
        v,
        dv_flip,
        fan: legal.len(),
        tv_flip,
        argmax_agree,
        dv_colour,
        dv_used_moves,
        used_players: used.len(),
    };
    let case = Case {
        score: dv_flip.abs(),
        label: format!("{} y-flip dV {:+.3}", b.id, dv_flip),
        row: serde_json::to_value(&row).expect("row"),
        before: s.clone(),
        after: f,
        marks: vec![],
    };
    (row, Some(case))
}

pub fn run(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let bases = bases(args, &|_| true)?;
    eprintln!("probe symmetry: {} base states", bases.len());
    let results = par_map(bases.len(), args.parallel, |i| probe(net, &bases[i]));
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    for (r, c) in results {
        worst.extend(c);
        rows.push(r);
    }
    let abs = |xs: Vec<f32>| xs.into_iter().map(f32::abs).collect::<Vec<_>>();
    let flip_all = dist(&abs(rows.iter().map(|r| r.dv_flip).collect()));
    let mut by_kind = BTreeMap::new();
    for k in [Kind::TurnStart, Kind::TurnMid, Kind::Activation, Kind::Other] {
        let d = abs(rows.iter().filter(|r| r.kind == k).map(|r| r.dv_flip).collect());
        if !d.is_empty() {
            by_kind.insert(format!("{k:?}"), dist(&d));
        }
    }
    let tv = dist(&rows.iter().filter_map(|r| r.tv_flip).collect::<Vec<_>>());
    let agree = frac(rows.iter().filter_map(|r| r.argmax_agree));
    let colour = dist(&abs(rows.iter().filter_map(|r| r.dv_colour).collect()));
    let moves = dist(&rows.iter().filter_map(|r| r.dv_used_moves).collect::<Vec<_>>());
    let signed_mean = rows.iter().map(|r| r.dv_flip).sum::<f32>() / rows.len().max(1) as f32;
    let mut table = String::new();
    let line = |name: &str, d: &super::common::Dist| {
        format!(
            "{name:<26} n {:>5}  mean {:.4}  p50 {:.4}  p90 {:.4}  p99 {:.4}  max {:.4}\n",
            d.n, d.mean, d.p50, d.p90, d.p99, d.max
        )
    };
    table += &line("|dV| y-flip", &flip_all);
    for (k, d) in &by_kind {
        table += &line(&format!("  |dV| y-flip {k}"), d);
    }
    table += &format!("  signed mean dV y-flip {signed_mean:+.4}\n");
    table += &line("policy TV y-flip", &tv);
    table += &format!("policy argmax agrees under y-flip: {:.1}%\n", 100.0 * agree);
    table += &line("|dV| colour swap (canary)", &colour);
    table += &line("|dV| used player's moves", &moves);
    Ok(SuiteOutput {
        summary: serde_json::json!({
            "abs_dv_flip": flip_all, "abs_dv_flip_by_kind": by_kind, "signed_mean_dv_flip": signed_mean,
            "tv_flip": tv, "argmax_agree_flip": agree, "abs_dv_colour": colour, "abs_dv_used_moves": moves,
        }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}

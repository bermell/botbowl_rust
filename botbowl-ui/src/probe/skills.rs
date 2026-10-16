//! Suite 1, skill monotonicity: give one player one skill (or +1 on one characteristic) and read
//! the value again. For a skill that can only help its owner, the owner's side must not lose
//! value: `gain = ΔV` for a mover's player, `-ΔV` for an opponent's, and `gain < 0` is a
//! violation. Every edit is a single input cell (pinned in `edits.rs`), so a violation is the
//! net's response to exactly that cell.
//!
//! The size that matters is set against the net's own noise on the same states. Two references:
//! the *level* noise `|V(flip s) - V(s)|` (the top-bottom mirror, true difference zero), and the
//! *differential* noise, the `|ΔV|` the net shows for skills no engine rule reads (the `Inert`
//! class, a placebo edit of the same one cell). `tau` is the placebo's 99th percentile: a
//! violation larger than `tau` is one a meaningless edit almost never produces.

use std::collections::BTreeMap;

use serde::Serialize;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::BallState;
use botbowl_engine::core::table::Skill;
use botbowl_nn::perspective::mover_for;

use super::common::{dist, frac, Base, Case, Dist, Kind, Net, TopK};
use super::edits::{add_skill, bump_stat, Stat};

/// What adding a skill should do to its owner's value, given the engine's rules and what the net
/// was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SkillClass {
    /// Implemented, never hurts its owner (passive, or a use-or-not choice the owner controls),
    /// and in the random-start pool (`Skill::good_skills`), so the net has seen it.
    Beneficial,
    /// Implemented and never hurts, but **never fielded in training**: the random-start
    /// generator only hands out `good_skills` and every player is a lineman. The net has a plane
    /// for it that has only ever been zero.
    BeneficialOod,
    /// Implemented, but the encoder has no plane for it (`ENCODED_SKILLS = 39` stops before
    /// `StripBall`): the net cannot see it, so ΔV is exactly 0.
    Unencoded,
    /// Forced and sometimes harmful (Frenzy must follow up and block again).
    DoubleEdged,
    /// No engine rule reads it (Leap, Pass, Bone Head, ...): the true effect is zero.
    Inert,
}

pub fn class_of(skill: Skill) -> SkillClass {
    use Skill::*;
    match skill {
        StripBall => SkillClass::Unencoded,
        Frenzy => SkillClass::DoubleEdged,
        Catch | Dodge | JumpUp | SideStep | SureFeet | Block | Dauntless | Fend | SureHands | Tackle | Wrestle
        | Claws | ArmBar | Brawler | Grab | Guard | Juggernaut | MightyBlow | StandFirm => {
            if Skill::good_skills().contains(&skill) {
                SkillClass::Beneficial
            } else {
                SkillClass::BeneficialOod
            }
        }
        _ => SkillClass::Inert,
    }
}

/// One line per skill for the plan's table: what the engine does with it, and why it is in its
/// class.
pub fn rule_note(skill: Skill) -> &'static str {
    use Skill::*;
    match skill {
        Catch => "re-roll a failed catch",
        Dodge => "re-roll a failed dodge; a Stumble against it is a Push",
        JumpUp => "stand up for free; may block from prone",
        SideStep => "optional: choose the push square",
        SureFeet => "re-roll a failed GFI",
        Block => "Both Down does not knock its owner down",
        Dauntless => "roll to match a stronger target's ST",
        Fend => "the blocker cannot follow up",
        SureHands => "re-roll a failed pickup",
        Tackle => "optional: deny an opponent's Dodge",
        Wrestle => "optional: on Both Down, both go prone, no armour, no turnover",
        Claws => "armour breaks on 8+",
        ArmBar => "a player dodging out of its zone falls harder",
        Brawler => "optional: re-roll one Both Down",
        Grab => "pushes to any free square; no Sidestep",
        Guard => "assists while marked",
        Juggernaut => "optional on a blitz: Both Down is a Push; no Stand Firm/Fend/Wrestle",
        MightyBlow => "+1 to armour or injury",
        StandFirm => "optional: refuse to be pushed",
        StripBall => "a pushed carrier drops the ball (implemented, not encoded)",
        Frenzy => "must follow up and block again",
        _ => "no rule implemented",
    }
}

/// A probed edit: a skill, or +1 on a characteristic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkillEdit {
    Skill(Skill),
    Plus(Stat),
}

impl SkillEdit {
    pub fn label(self) -> String {
        match self {
            SkillEdit::Skill(s) => format!("{s:?}"),
            SkillEdit::Plus(s) => format!("{s:?}+1"),
        }
    }

    /// Characteristic improvements are beneficial and in distribution (inside the training range).
    pub fn class(self) -> SkillClass {
        match self {
            SkillEdit::Skill(s) => class_of(s),
            SkillEdit::Plus(_) => SkillClass::Beneficial,
        }
    }

    pub fn apply(self, s: &GameState, id: usize) -> Result<GameState, String> {
        match self {
            SkillEdit::Skill(k) => add_skill(s, id, k),
            SkillEdit::Plus(st) => bump_stat(s, id, st, 1),
        }
    }

    pub fn all() -> Vec<SkillEdit> {
        Skill::ALL
            .iter()
            .map(|&s| SkillEdit::Skill(s))
            .chain(Stat::ALL.iter().map(|&s| SkillEdit::Plus(s)))
            .collect()
    }
}

#[derive(Serialize, Clone)]
pub struct Row {
    pub base: String,
    pub kind: Kind,
    pub player: usize,
    /// `own`: the mover's player; `opp`: the opponent's.
    pub side: &'static str,
    pub carrier: bool,
    pub active: bool,
    pub edit: String,
    pub class: SkillClass,
    /// V(s), mover's frame.
    pub v: f32,
    pub dv: f32,
    /// The owner's change: `dv` for `own`, `-dv` for `opp`. Must be `>= 0` for a beneficial edit.
    pub gain: f32,
}

pub struct StateResult {
    pub rows: Vec<Row>,
    /// `|V(y-flip s) - V(s)|`: this state's noise reference.
    pub noise: f32,
    pub worst: Vec<Case>,
}

pub fn probe_state(net: &Net, b: &Base, top: usize) -> StateResult {
    let s = &b.state;
    let mover = mover_for(s);
    let v = net.value(s);
    let noise = (net.value(&s.y_flipped()) - v).abs();
    let mut rows = Vec::new();
    let mut worst = TopK::new(top);
    for p in s.get_players_on_pitch() {
        let own = p.stats.team == mover;
        for e in SkillEdit::all() {
            let Ok(after) = e.apply(s, p.id) else { continue };
            let dv = net.value(&after) - v;
            let gain = if own { dv } else { -dv };
            let row = Row {
                base: b.id.clone(),
                kind: b.kind,
                player: p.id,
                side: if own { "own" } else { "opp" },
                carrier: s.ball == BallState::Carried(p.id),
                active: s.info.active_player == Some(p.id),
                edit: e.label(),
                class: e.class(),
                v,
                dv,
                gain,
            };
            if e.class() == SkillClass::Beneficial && gain < 0.0 {
                worst.push(Case {
                    score: -gain,
                    label: format!("{} {} on {} player {}", b.id, e.label(), row.side, p.id),
                    row: serde_json::to_value(&row).expect("row"),
                    before: s.clone(),
                    after,
                    marks: vec![(p.position, '!')],
                });
            }
            rows.push(row);
        }
    }
    StateResult {
        rows,
        noise,
        worst: worst.into_items(),
    }
}

#[derive(Serialize)]
pub struct EditSummary {
    pub edit: String,
    pub class: SkillClass,
    pub note: &'static str,
    pub n: usize,
    pub gain: Dist,
    /// Share of edits where the owner lost value at all, and by more than `tau`.
    pub violation: f32,
    pub violation_gt_tau: f32,
    pub own_violation: f32,
    pub opp_violation: f32,
    /// Mean `|ΔV|` (the size of the response, signed or not).
    pub mean_abs: f32,
}

pub fn summarise(rows: &[Row], tau: f32) -> (Vec<EditSummary>, BTreeMap<String, serde_json::Value>) {
    let mut by_edit: BTreeMap<(SkillClass, String), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        by_edit.entry((r.class, r.edit.clone())).or_default().push(r);
    }
    let note_of = |e: &str| {
        Skill::ALL
            .iter()
            .find(|s| format!("{s:?}") == e)
            .map(|s| rule_note(*s))
            .unwrap_or("characteristic +1 inside the training range")
    };
    let edits: Vec<EditSummary> = by_edit
        .iter()
        .map(|((class, e), rs)| {
            let gains: Vec<f32> = rs.iter().map(|r| r.gain).collect();
            EditSummary {
                edit: e.clone(),
                class: *class,
                note: note_of(e),
                n: rs.len(),
                gain: dist(&gains),
                violation: frac(gains.iter().map(|g| *g < 0.0)),
                violation_gt_tau: frac(gains.iter().map(|g| *g < -tau)),
                own_violation: frac(rs.iter().filter(|r| r.side == "own").map(|r| r.gain < 0.0)),
                opp_violation: frac(rs.iter().filter(|r| r.side == "opp").map(|r| r.gain < 0.0)),
                mean_abs: gains.iter().map(|g| g.abs()).sum::<f32>() / gains.len().max(1) as f32,
            }
        })
        .collect();
    let mut by_class: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut classes: BTreeMap<SkillClass, Vec<&Row>> = BTreeMap::new();
    for r in rows {
        classes.entry(r.class).or_default().push(r);
    }
    for (c, rs) in &classes {
        let gains: Vec<f32> = rs.iter().map(|r| r.gain).collect();
        let mut by_kind = BTreeMap::new();
        for k in [Kind::TurnStart, Kind::TurnMid, Kind::Activation, Kind::Other] {
            let g: Vec<bool> = rs.iter().filter(|r| r.kind == k).map(|r| r.gain < -tau).collect();
            if !g.is_empty() {
                by_kind.insert(format!("{k:?}"), (g.len(), frac(g)));
            }
        }
        by_class.insert(
            serde_json::to_value(c)
                .expect("class")
                .as_str()
                .unwrap_or("?")
                .to_string(),
            serde_json::json!({
                "n": rs.len(),
                "gain": dist(&gains),
                "violation": frac(gains.iter().map(|g| *g < 0.0)),
                "violation_gt_tau": frac(gains.iter().map(|g| *g < -tau)),
                "mean_abs": gains.iter().map(|g| g.abs()).sum::<f32>() / gains.len().max(1) as f32,
                "violation_gt_tau_by_kind": by_kind,
            }),
        );
    }
    (edits, by_class)
}

pub fn table(edits: &[EditSummary], tau: f32, noise: &Dist, placebo: &Dist) -> String {
    let mut t = format!(
        "level noise |V(flip)-V|: p50 {:.4} p90 {:.4} p99 {:.4} over {} states\n\
         placebo |dV| (inert skills): p50 {:.4} p90 {:.4} p99 (tau) {tau:.4} over {} edits\n",
        noise.p50, noise.p90, noise.p99, noise.n, placebo.p50, placebo.p90, placebo.n
    );
    t += &format!(
        "{:<16} {:<14} {:>6} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>8}\n",
        "edit", "class", "n", "mean", "median", "viol", ">tau", "own", "opp", "worst"
    );
    let mut sorted: Vec<&EditSummary> = edits.iter().collect();
    sorted.sort_by(|a, b| {
        a.class
            .cmp(&b.class)
            .then(b.violation_gt_tau.total_cmp(&a.violation_gt_tau))
    });
    for e in sorted {
        t += &format!(
            "{:<16} {:<14} {:>6} {:>+8.4} {:>+8.4} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>+8.4}\n",
            e.edit,
            serde_json::to_value(e.class).unwrap().as_str().unwrap_or("?"),
            e.n,
            e.gain.mean,
            e.gain.p50,
            100.0 * e.violation,
            100.0 * e.violation_gt_tau,
            100.0 * e.own_violation,
            100.0 * e.opp_violation,
            e.gain.min
        );
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::edits::tests::starts;
    use botbowl_nn::encode::encode_raw;

    #[test]
    fn every_skill_has_a_class_and_the_pool_is_the_in_distribution_class() {
        for s in Skill::ALL {
            let c = class_of(s);
            let pooled = Skill::good_skills().contains(&s);
            match c {
                SkillClass::Beneficial => assert!(pooled, "{s:?}"),
                SkillClass::BeneficialOod => assert!(!pooled, "{s:?}"),
                _ => {}
            }
        }
        // Every pooled skill is probed as beneficial except the forced one.
        for s in Skill::good_skills() {
            assert!(
                matches!(class_of(s), SkillClass::Beneficial | SkillClass::DoubleEdged),
                "{s:?}"
            );
        }
    }

    /// The unencoded class really is invisible: the tensor does not change.
    #[test]
    fn an_unencoded_skill_leaves_the_encoding_unchanged() {
        for s in starts(4) {
            for p in s.get_players_on_pitch() {
                for k in Skill::ALL.into_iter().filter(|k| class_of(*k) == SkillClass::Unencoded) {
                    let e = add_skill(&s, p.id, k).unwrap();
                    let (a, b) = (encode_raw(&s), encode_raw(&e));
                    assert!(a.spatial == b.spatial && a.global == b.global, "{k:?}");
                }
            }
        }
    }
}

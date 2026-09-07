//! Training targets built from the raw per-child search stats.
//!
//! Because `recon_mcts` freezes a child's visit count the moment its
//! subtree is solved — and the fastest-solving child is often the *best*
//! move (a touchdown solves in ~10 descents while mediocre siblings keep
//! accruing) — `π ∝ visits` is actively wrong. We read the `solved` flag
//! and correct (plan 017 §caveat):
//!
//! 1. **Root solved** → the position is proven; emit a one-hot on the
//!    argmax mover-`Q` child (or [`SolvedRootPolicy::Skip`] the sample).
//! 2. **Root partially solved** → hybrid: unsolved children keep their
//!    visits; the argmax-`Q` solved child is floored at the max unsolved
//!    sibling visit count; other solved children keep their frozen count.
//! 3. **Nothing solved** → normalise visits.
//!
//! `Q` in the schema is **Home-centric**; every comparison here is done
//! in the *mover's* frame (`Home` maximises, `Away` minimises → negate),
//! and the value target is mover-signed to match the network's
//! mover-centric `v`.
//!
//! [`PolicyTargetKind::CompletedQ`] (plan 032 #7) is the alternative to
//! visit counts for unsolved roots: `softmax(ln prior + q_mover / τ)`, with
//! children the search never scored "completed" by the visit-weighted mean
//! `Q` of the scored ones (Gumbel-MuZero's completed-Q). At 1000 iterations
//! over a 30-100-square move fan the visit target is mostly the FPU sweep,
//! and plan 031 D2 measured it disagreeing with the move actually played
//! three times in four on those roots; the search's *value* information is
//! in `Q`, which this target reads directly.

use botbowl_data::{Sample, Team};

/// How to treat a fully-solved root when building the policy target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolvedRootPolicy {
    /// Emit a one-hot on the argmax mover-`Q` child.
    OneHot,
    /// Drop the sample from the policy dataset entirely (trivially decided).
    Skip,
}

/// Which statistic the policy target reads on an unsolved root. Solved
/// roots are one-hot on the exact argmax `Q` under every kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolicyTargetKind {
    /// Normalised visit counts with the solved-child floor (the original).
    Visits,
    /// `softmax(ln prior + q_mover / tau)`, unscored children completed
    /// with the visit-weighted mean `Q` of the scored ones. `tau` is in `Q`
    /// points (one touchdown = 1000).
    CompletedQ { tau: f32 },
}

/// Per-child policy target, aligned index-for-index to `sample.children`.
/// `probs` sums to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyTarget {
    pub probs: Vec<f32>,
}

/// Child `Q` in the *mover's* frame (`Home` maximises, `Away` negates the
/// Home-centric stored value). `None` if the child was never scored.
fn mover_q(q: Option<i64>, mover: Team) -> Option<i64> {
    q.map(|v| match mover {
        Team::Home => v,
        Team::Away => -v,
    })
}

/// Build the visit-count policy target for one decision, applying the
/// solved-count corrections. Returns `None` when the sample should be
/// dropped from the policy dataset (no children, all-zero counts, or a
/// solved root under [`SolvedRootPolicy::Skip`]).
pub fn policy_target(sample: &Sample, solved_root: SolvedRootPolicy) -> Option<PolicyTarget> {
    policy_target_of(sample, solved_root, PolicyTargetKind::Visits)
}

/// [`policy_target`] with the unsolved-root statistic selectable.
pub fn policy_target_of(
    sample: &Sample,
    solved_root: SolvedRootPolicy,
    kind: PolicyTargetKind,
) -> Option<PolicyTarget> {
    let n = sample.children.len();
    if n == 0 {
        return None;
    }
    let mover = sample.to_move;

    // Index of the child with the best mover-Q among a filtered set.
    let argmax_q = |filter_solved: Option<bool>| -> Option<usize> {
        sample
            .children
            .iter()
            .enumerate()
            .filter(|(_, c)| filter_solved.map_or(true, |s| c.solved == s))
            .filter_map(|(i, c)| mover_q(c.q, mover).map(|q| (i, q)))
            .max_by_key(|(_, q)| *q)
            .map(|(i, _)| i)
    };

    if sample.root_solved {
        match solved_root {
            SolvedRootPolicy::Skip => return None,
            SolvedRootPolicy::OneHot => {
                // Argmax over exact child Q; fall back to most-visited if
                // no child carries a Q (shouldn't happen for a solved root).
                let best = argmax_q(None).or_else(|| {
                    sample
                        .children
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, c)| c.visits)
                        .map(|(i, _)| i)
                })?;
                let mut probs = vec![0.0f32; n];
                probs[best] = 1.0;
                return Some(PolicyTarget { probs });
            }
        }
    }

    if let PolicyTargetKind::CompletedQ { tau } = kind {
        return completed_q_target(sample, tau);
    }

    // Partially / not solved: start from raw visit counts.
    let mut counts: Vec<f32> = sample.children.iter().map(|c| c.visits as f32).collect();

    let any_solved = sample.children.iter().any(|c| c.solved);
    if any_solved {
        let max_unsolved_visits = sample
            .children
            .iter()
            .filter(|c| !c.solved)
            .map(|c| c.visits)
            .max()
            .unwrap_or(0) as f32;
        // Floor the best solved child at the max unsolved sibling visits.
        if let Some(best_solved) = argmax_q(Some(true)) {
            counts[best_solved] = counts[best_solved].max(max_unsolved_visits);
        }
    }

    let total: f32 = counts.iter().sum();
    if total <= 0.0 {
        return None;
    }
    for c in &mut counts {
        *c /= total;
    }
    Some(PolicyTarget { probs: counts })
}

/// `softmax(ln prior + q_mover / tau)` over the children. A child counts as
/// scored when it has a `Q` *and* at least one visit; the rest are completed
/// with the visit-weighted mean `Q` of the scored children (plain mean of
/// the `Q`-bearing children if nothing was visited — the 0-visit `Q` a
/// solved/terminal child can carry). `None` when no child has a `Q`. A
/// missing prior is treated as `ln 1 = 0`; the prior's global scale
/// (softmax×len for the NN, unnormalised multipliers for the heuristic)
/// cancels in the softmax, so both evaluators' corpora are usable as-is.
fn completed_q_target(sample: &Sample, tau: f32) -> Option<PolicyTarget> {
    let mover = sample.to_move;
    let qs: Vec<Option<f64>> = sample
        .children
        .iter()
        .map(|c| mover_q(c.q, mover).map(|q| q as f64))
        .collect();
    let (mut w_sum, mut wq_sum) = (0.0f64, 0.0f64);
    let (mut n_scored, mut q_sum) = (0usize, 0.0f64);
    for (c, q) in sample.children.iter().zip(&qs) {
        if let Some(q) = q {
            n_scored += 1;
            q_sum += q;
            if c.visits > 0 {
                w_sum += f64::from(c.visits);
                wq_sum += f64::from(c.visits) * q;
            }
        }
    }
    if n_scored == 0 {
        return None;
    }
    let fill = if w_sum > 0.0 {
        wq_sum / w_sum
    } else {
        q_sum / n_scored as f64
    };
    let tau = f64::from(tau);
    let logits: Vec<f64> = sample
        .children
        .iter()
        .zip(&qs)
        .map(|(c, q)| {
            let q = match q {
                Some(q) if c.visits > 0 => *q,
                _ => fill,
            };
            let prior = c.prior.map_or(0.0, |p| f64::from(p.max(1e-6)).ln());
            prior + q / tau
        })
        .collect();
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = weights.iter().sum();
    Some(PolicyTarget {
        probs: weights.iter().map(|w| (w / z) as f32).collect(),
    })
}

/// Value target `v1`: the sample's backfilled drive outcome (Home-centric,
/// `[-1,1]`, see `Trajectory::backfill_outcome_value`) re-signed into the
/// mover's frame, matching the network's mover-centric `v`. `None` before
/// the outcome is backfilled.
pub fn value_target(sample: &Sample) -> Option<f32> {
    sample.outcome_value.map(|z| match sample.to_move {
        Team::Home => z,
        Team::Away => -z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_data::ChildStat;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{Action, Position};
    use botbowl_engine::core::table::PosAT;

    fn child(visits: u32, q: Option<i64>, solved: bool) -> ChildStat {
        ChildStat {
            action: Action::Positional(PosAT::Move, Position::new((1, 1))),
            visits,
            q,
            prior: Some(1.0),
            solved,
            terminal: solved,
        }
    }

    fn sample(children: Vec<ChildStat>, root_solved: bool, to_move: Team, outcome: Option<f32>) -> Sample {
        Sample {
            state: GameStateBuilder::new_start_of_game(),
            to_move,
            chosen_action: Action::Simple(botbowl_engine::core::table::SimpleAT::EndTurn),
            children,
            root_value: Some(0),
            root_visits: 100,
            root_solved,
            outcome_value: outcome,
        }
    }

    #[test]
    fn no_children_yields_none() {
        let s = sample(vec![], false, Team::Home, Some(1.0));
        assert!(policy_target(&s, SolvedRootPolicy::OneHot).is_none());
    }

    #[test]
    fn plain_visits_normalise() {
        let s = sample(
            vec![child(30, Some(10), false), child(70, Some(5), false)],
            false,
            Team::Home,
            Some(0.0),
        );
        let t = policy_target(&s, SolvedRootPolicy::OneHot).unwrap();
        assert!((t.probs[0] - 0.3).abs() < 1e-6);
        assert!((t.probs[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn solved_root_onehot_picks_best_home_q() {
        let s = sample(
            vec![child(5, Some(1000), true), child(90, Some(-200), true)],
            true,
            Team::Home,
            Some(1.0),
        );
        let t = policy_target(&s, SolvedRootPolicy::OneHot).unwrap();
        assert_eq!(t.probs, vec![1.0, 0.0]);
    }

    #[test]
    fn solved_root_onehot_flips_for_away() {
        // Same Home-centric Q; Away minimises, so the -200 child is best.
        let s = sample(
            vec![child(5, Some(1000), true), child(90, Some(-200), true)],
            true,
            Team::Away,
            Some(1.0),
        );
        let t = policy_target(&s, SolvedRootPolicy::OneHot).unwrap();
        assert_eq!(t.probs, vec![0.0, 1.0]);
    }

    #[test]
    fn solved_root_skip_drops_sample() {
        let s = sample(vec![child(5, Some(1000), true)], true, Team::Home, Some(1.0));
        assert!(policy_target(&s, SolvedRootPolicy::Skip).is_none());
    }

    #[test]
    fn partial_solve_floors_best_solved_child_at_max_unsolved() {
        // A solved TD child has frozen visits=8 but best Q; an unsolved
        // sibling has 90 visits. The solved child must be floored to 90.
        let s = sample(
            vec![
                child(8, Some(1000), true),  // best mover-Q, solved
                child(90, Some(100), false), // unsolved, most visits
                child(20, Some(-50), true),  // other solved child, keeps 20
            ],
            false,
            Team::Home,
            Some(1.0),
        );
        let t = policy_target(&s, SolvedRootPolicy::OneHot).unwrap();
        // counts: [90, 90, 20] → total 200
        assert!((t.probs[0] - 90.0 / 200.0).abs() < 1e-6);
        assert!((t.probs[1] - 90.0 / 200.0).abs() < 1e-6);
        assert!((t.probs[2] - 20.0 / 200.0).abs() < 1e-6);
    }

    #[test]
    fn partial_solve_away_picks_min_home_q_as_best_solved() {
        // Away minimises: the -300 solved child is "best solved" and gets
        // floored; the +50 solved child keeps its frozen visits.
        let s = sample(
            vec![
                child(4, Some(-300), true), // best for Away, solved
                child(60, Some(0), false),  // unsolved
                child(30, Some(50), true),  // worse for Away, keeps 30
            ],
            false,
            Team::Away,
            Some(-1.0),
        );
        let t = policy_target(&s, SolvedRootPolicy::OneHot).unwrap();
        // counts: [max(4,60)=60, 60, 30] → total 150
        assert!((t.probs[0] - 60.0 / 150.0).abs() < 1e-6);
        assert!((t.probs[1] - 60.0 / 150.0).abs() < 1e-6);
        assert!((t.probs[2] - 30.0 / 150.0).abs() < 1e-6);
    }

    fn cq(sample: &Sample, tau: f32) -> Vec<f32> {
        policy_target_of(sample, SolvedRootPolicy::OneHot, PolicyTargetKind::CompletedQ { tau })
            .unwrap()
            .probs
    }

    fn child_p(visits: u32, q: Option<i64>, prior: f32) -> ChildStat {
        ChildStat {
            prior: Some(prior),
            ..child(visits, q, false)
        }
    }

    #[test]
    fn completed_q_is_softmax_of_log_prior_plus_scaled_q() {
        // Equal priors: pure softmax(q / tau). q = 0, 100 at tau 100 →
        // e^0 : e^1.
        let s = sample(
            vec![child_p(10, Some(0), 1.0), child_p(10, Some(100), 1.0)],
            false,
            Team::Home,
            Some(0.0),
        );
        let p = cq(&s, 100.0);
        let e = std::f32::consts::E;
        assert!((p[0] - 1.0 / (1.0 + e)).abs() < 1e-6);
        assert!((p[1] - e / (1.0 + e)).abs() < 1e-6);
        // Equal q: the prior alone, ratio 1:3.
        let s = sample(
            vec![child_p(10, Some(50), 1.0), child_p(10, Some(50), 3.0)],
            false,
            Team::Home,
            Some(0.0),
        );
        let p = cq(&s, 100.0);
        assert!((p[0] - 0.25).abs() < 1e-6);
        assert!((p[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn completed_q_prior_scale_cancels() {
        let mk = |k: f32| {
            sample(
                vec![
                    child_p(5, Some(0), 1.0 * k),
                    child_p(5, Some(200), 0.5 * k),
                    child_p(0, None, 2.0 * k),
                ],
                false,
                Team::Home,
                Some(0.0),
            )
        };
        let a = cq(&mk(1.0), 100.0);
        let b = cq(&mk(37.0), 100.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn completed_q_fills_unvisited_with_visit_weighted_mean() {
        // Scored: q=0 (3 visits), q=400 (1 visit) → fill = 100. The
        // unvisited child (equal prior) must then match a visited child with
        // q=100 exactly.
        let s = sample(
            vec![
                child_p(3, Some(0), 1.0),
                child_p(1, Some(400), 1.0),
                child_p(0, None, 1.0),
            ],
            false,
            Team::Home,
            Some(0.0),
        );
        let s_ref = sample(
            vec![
                child_p(3, Some(0), 1.0),
                child_p(1, Some(400), 1.0),
                child_p(1, Some(100), 1.0),
            ],
            false,
            Team::Home,
            Some(0.0),
        );
        let (p, r) = (cq(&s, 100.0), cq(&s_ref, 100.0));
        for (x, y) in p.iter().zip(&r) {
            assert!((x - y).abs() < 1e-6, "{p:?} vs {r:?}");
        }
        // A 0-visit child that does carry a Q is still completed, not read.
        let s_q0 = sample(
            vec![
                child_p(3, Some(0), 1.0),
                child_p(1, Some(400), 1.0),
                child_p(0, Some(-9000), 1.0),
            ],
            false,
            Team::Home,
            Some(0.0),
        );
        let p2 = cq(&s_q0, 100.0);
        for (x, y) in p2.iter().zip(&r) {
            assert!((x - y).abs() < 1e-6, "{p2:?} vs {r:?}");
        }
    }

    #[test]
    fn completed_q_is_in_the_movers_frame() {
        // Home-centric q = +300 / -300. Home prefers the first, Away the
        // second, with mirrored probabilities.
        let kids = || vec![child_p(5, Some(300), 1.0), child_p(5, Some(-300), 1.0)];
        let home = cq(&sample(kids(), false, Team::Home, Some(0.0)), 100.0);
        let away = cq(&sample(kids(), false, Team::Away, Some(0.0)), 100.0);
        assert!(home[0] > 0.99);
        assert!(away[1] > 0.99);
        assert!((home[0] - away[1]).abs() < 1e-6);
    }

    #[test]
    fn completed_q_keeps_solved_root_onehot_and_drops_unscored() {
        let solved = sample(
            vec![child_p(5, Some(1000), 1.0), child_p(90, Some(-200), 1.0)],
            true,
            Team::Home,
            Some(1.0),
        );
        assert_eq!(cq(&solved, 100.0), vec![1.0, 0.0]);
        assert!(policy_target_of(
            &solved,
            SolvedRootPolicy::Skip,
            PolicyTargetKind::CompletedQ { tau: 100.0 }
        )
        .is_none());
        let unscored = sample(
            vec![child_p(0, None, 1.0), child_p(0, None, 2.0)],
            false,
            Team::Home,
            Some(0.0),
        );
        assert!(policy_target_of(
            &unscored,
            SolvedRootPolicy::OneHot,
            PolicyTargetKind::CompletedQ { tau: 100.0 }
        )
        .is_none());
    }

    #[test]
    fn completed_q_sums_to_one_on_a_wide_fan() {
        // Five visited children with Home q = 0, 50, .., 200 and 75 unvisited
        // ones, uniform priors. Away minimises Home q, so child 0 is the
        // best visited one; the unvisited are completed at the mean (-100
        // in Away's frame) and so sit strictly between child 0 and child 4.
        let kids: Vec<ChildStat> = (0..80)
            .map(|i| child_p(if i < 5 { 20 } else { 0 }, if i < 5 { Some(i * 50) } else { None }, 1.0))
            .collect();
        let p = cq(&sample(kids, false, Team::Away, Some(0.0)), 100.0);
        assert_eq!(p.len(), 80);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, 0);
        assert!(p[0] > p[40] && p[40] > p[4]);
        assert!((p[40] - p[79]).abs() < 1e-7);
    }

    #[test]
    fn value_target_signs_by_mover() {
        let home = sample(vec![child(1, Some(0), false)], false, Team::Home, Some(1.0));
        let away = sample(vec![child(1, Some(0), false)], false, Team::Away, Some(1.0));
        assert_eq!(value_target(&home), Some(1.0));
        assert_eq!(value_target(&away), Some(-1.0));
        let none = sample(vec![child(1, Some(0), false)], false, Team::Home, None);
        assert_eq!(value_target(&none), None);
    }
}

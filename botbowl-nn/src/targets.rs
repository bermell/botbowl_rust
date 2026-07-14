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

use botbowl_data::{Sample, Team};

/// How to treat a fully-solved root when building the policy target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolvedRootPolicy {
    /// Emit a one-hot on the argmax mover-`Q` child.
    OneHot,
    /// Drop the sample from the policy dataset entirely (trivially decided).
    Skip,
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

/// Build the policy target for one decision, applying the solved-count
/// corrections. Returns `None` when the sample should be dropped from the
/// policy dataset (no children, all-zero counts, or a solved root under
/// [`SolvedRootPolicy::Skip`]).
pub fn policy_target(sample: &Sample, solved_root: SolvedRootPolicy) -> Option<PolicyTarget> {
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

/// Value target `v1`: the trajectory outcome `z` (Home-centric, `[-1,1]`)
/// re-signed into the mover's frame, matching the network's mover-centric
/// `v`. `None` before the trajectory outcome is backfilled.
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

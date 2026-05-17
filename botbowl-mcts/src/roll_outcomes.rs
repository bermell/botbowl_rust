use botbowl_engine::core::dices::{RequestedRoll, RollTarget};

use crate::action::{BbAction, ChanceOutcome};

/// Enumerate the possible outcomes of a `RequestedRoll`, paired with the
/// probability that they occur. Used by the MCTS dynamics to expand a
/// chance node into one child per outcome.
///
/// The MVP handles `D6PassFail` and `Sum2D6PassFail` only. The engine's
/// `RollTarget::success_prob` knows the success probability for either;
/// the failure probability is its complement.
pub fn enumerate(req: &RequestedRoll) -> Vec<BbAction> {
    match req {
        RequestedRoll::D6PassFail(target) => {
            let p_pass = target.success_prob();
            vec![
                BbAction::chance(ChanceOutcome::Pass, p_pass),
                BbAction::chance(ChanceOutcome::Fail, 1.0 - p_pass),
            ]
        }
        RequestedRoll::Sum2D6PassFail(target) => {
            let p_pass = target.success_prob();
            vec![
                BbAction::chance(ChanceOutcome::Pass, p_pass),
                BbAction::chance(ChanceOutcome::Fail, 1.0 - p_pass),
            ]
        }
        other => panic!(
            "MCTS chance-node enumeration not implemented for {:?} \
             — extend roll_outcomes::enumerate when a lecture needs it",
            other
        ),
    }
}

/// Pre-load `state.fixes` with the dice values that will yield `outcome`
/// when the engine consumes its pending roll. The dynamics calls this
/// just before stepping a Chance action through `state.micro_step(None)`.
pub fn fix_for_outcome(
    state: &mut botbowl_engine::core::gamestate::GameState,
    outcome: ChanceOutcome,
) {
    let req = state
        .pending_roll
        .as_ref()
        .expect("fix_for_outcome called with no pending roll");
    match (req, outcome) {
        (RequestedRoll::D6PassFail(_), ChanceOutcome::Pass) => state.fixes.fix_d6(6),
        (RequestedRoll::D6PassFail(_), ChanceOutcome::Fail) => state.fixes.fix_d6(1),
        (RequestedRoll::Sum2D6PassFail(_), ChanceOutcome::Pass) => {
            state.fixes.fix_d6(6);
            state.fixes.fix_d6(6);
        }
        (RequestedRoll::Sum2D6PassFail(_), ChanceOutcome::Fail) => {
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(1);
        }
        (other, _) => panic!(
            "fix_for_outcome doesn't know how to encode outcome for {:?}",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::dices::{D6Target, Sum2D6Target};

    fn probs_sum_to_one(actions: &[BbAction]) -> bool {
        let total: f32 = actions.iter().filter_map(|a| a.prob_f32()).sum();
        (total - 1.0).abs() < 1e-5
    }

    #[test]
    fn d6_pass_fail_outcomes_sum_to_one() {
        for target in [
            D6Target::TwoPlus,
            D6Target::ThreePlus,
            D6Target::FourPlus,
            D6Target::FivePlus,
            D6Target::SixPlus,
        ] {
            let outcomes = enumerate(&RequestedRoll::D6PassFail(target));
            assert_eq!(outcomes.len(), 2, "expected pass+fail for {:?}", target);
            assert!(
                probs_sum_to_one(&outcomes),
                "probabilities for {:?} don't sum to 1.0: {:?}",
                target,
                outcomes
            );
        }
    }

    #[test]
    fn sum2d6_pass_fail_outcomes_sum_to_one() {
        for target in [
            Sum2D6Target::FourPlus,
            Sum2D6Target::SevenPlus,
            Sum2D6Target::NinePlus,
        ] {
            let outcomes = enumerate(&RequestedRoll::Sum2D6PassFail(target));
            assert!(
                probs_sum_to_one(&outcomes),
                "probabilities for {:?} don't sum to 1.0",
                target
            );
        }
    }

    #[test]
    fn three_plus_pass_probability_is_4_over_6() {
        let outcomes = enumerate(&RequestedRoll::D6PassFail(D6Target::ThreePlus));
        let pass = outcomes
            .iter()
            .find_map(|a| match a {
                BbAction::Chance {
                    outcome: ChanceOutcome::Pass,
                    prob_bits,
                } => Some(f32::from_bits(*prob_bits)),
                _ => None,
            })
            .unwrap();
        assert!(
            (pass - 4.0 / 6.0).abs() < 1e-5,
            "expected 4/6, got {}",
            pass
        );
    }
}

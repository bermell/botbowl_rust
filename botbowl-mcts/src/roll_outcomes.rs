use botbowl_engine::core::dices::{RequestedRoll, RollTarget};

use crate::action::{BbAction, ChanceOutcome};

/// Enumerate the possible outcomes of a `RequestedRoll`, paired with the
/// probability that they occur. Used by the MCTS dynamics to expand a
/// chance node into one child per outcome.
///
/// - `D6PassFail` / `Sum2D6PassFail` → two children (Pass + Fail) with
///   probabilities from `RollTarget::success_prob`.
/// - All other roll types → a single `ChanceOutcome::Advance` child.
///   `apply_action` resolves these by letting the engine consume the
///   `pending_roll` via its `DicePolicy` (or the RNG when no policy
///   applies). Single-advance keeps the search tree bounded; the trade-
///   off is no probabilistic averaging across the outcomes of those
///   rolls — fine for the deterministic-policy curriculum, revisit when
///   we get to genuinely probabilistic policies.
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
        _ => vec![BbAction::chance(ChanceOutcome::Advance, 1.0)],
    }
}

/// Pre-load `state.fixes` with the dice values that will yield `outcome`
/// when the engine consumes its pending roll. The dynamics calls this
/// just before stepping a Chance action through `state.micro_step(None)`.
///
/// `ChanceOutcome::Advance` queues no fix — the engine resolves the
/// pending roll via its `DicePolicy` (or the RNG when no policy
/// applies). That's the path for non-pass/fail roll types where MCTS
/// only models a single chance child.
pub fn fix_for_outcome(
    state: &mut botbowl_engine::core::gamestate::GameState,
    outcome: ChanceOutcome,
) {
    if matches!(outcome, ChanceOutcome::Advance) {
        return;
    }
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
            "fix_for_outcome: pass/fail outcome on a non-pass/fail roll {:?} \
             (should have been ChanceOutcome::Advance)",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::dices::{D6Target, Sum2D6Target};
    use botbowl_engine::core::table::NumBlockDices;

    fn probs_sum_to_one(actions: &[BbAction]) -> bool {
        let total: f32 = actions.iter().filter_map(|a| a.prob_f32()).sum();
        (total - 1.0).abs() < 1e-5
    }

    fn is_advance(a: &BbAction) -> bool {
        matches!(
            a,
            BbAction::Chance {
                outcome: ChanceOutcome::Advance,
                ..
            }
        )
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
    fn d8_returns_single_advance_outcome() {
        let outcomes = enumerate(&RequestedRoll::D8);
        assert_eq!(outcomes.len(), 1);
        assert!(is_advance(&outcomes[0]));
        assert!(probs_sum_to_one(&outcomes));
    }

    #[test]
    fn deviate_returns_single_advance_outcome() {
        let outcomes = enumerate(&RequestedRoll::Deviate);
        assert_eq!(outcomes.len(), 1);
        assert!(is_advance(&outcomes[0]));
    }

    #[test]
    fn scatter_returns_single_advance_outcome() {
        let outcomes = enumerate(&RequestedRoll::Scatter);
        assert_eq!(outcomes.len(), 1);
        assert!(is_advance(&outcomes[0]));
    }

    #[test]
    fn block_dice_returns_single_advance_outcome() {
        for n in [
            NumBlockDices::One,
            NumBlockDices::Two,
            NumBlockDices::Three,
            NumBlockDices::TwoUphill,
            NumBlockDices::ThreeUphill,
        ] {
            let outcomes = enumerate(&RequestedRoll::BlockDice(n));
            assert_eq!(outcomes.len(), 1, "expected single Advance for {:?}", n);
            assert!(is_advance(&outcomes[0]));
        }
    }

    #[test]
    fn throw_in_returns_single_advance_outcome() {
        let outcomes = enumerate(&RequestedRoll::ThrowIn);
        assert_eq!(outcomes.len(), 1);
        assert!(is_advance(&outcomes[0]));
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

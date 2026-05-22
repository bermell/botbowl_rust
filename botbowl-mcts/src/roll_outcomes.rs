use botbowl_engine::core::dices::{BlockDice, RequestedRoll, RollTarget};
use botbowl_engine::core::model::Direction;

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
/// Every `outcome` queues a deterministic dice pattern — including
/// `Advance`. Letting the engine resolve via RNG (the v2 behaviour)
/// made the same (parent, action) edge produce different child states
/// on different descent paths, breaking state-hash recombination and
/// fanning out the search tree.
pub fn fix_for_outcome(
    state: &mut botbowl_engine::core::gamestate::GameState,
    outcome: ChanceOutcome,
) {
    let req = state
        .pending_roll
        .as_ref()
        .expect("fix_for_outcome called with no pending roll");
    match (req, outcome) {
        // ----- Pass / Fail rolls: branch the search on the outcome -----
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

        // ----- Single-child Advance rolls: pick a deterministic value -----
        // D8 is used for ball bounces and ball-scatter directions. Any
        // constant direction is fine; we just need *a* deterministic
        // outcome so two paths to the same logical position recombine.
        (RequestedRoll::D8, ChanceOutcome::Advance) => {
            state.fixes.fix_d8_direction(Direction::up());
        }
        // Deviate = D6 (distance) + D8 (direction). Minimum scatter +
        // up — the ball barely moves.
        (RequestedRoll::Deviate, ChanceOutcome::Advance) => {
            state.fixes.fix_d6(1);
            state.fixes.fix_d8_direction(Direction::up());
        }
        // Scatter = three D8 directions. Pick the same direction each
        // time; the engine treats the sequence as separate bounces.
        (RequestedRoll::Scatter, ChanceOutcome::Advance) => {
            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d8_direction(Direction::up());
            state.fixes.fix_d8_direction(Direction::up());
        }
        // ThrowIn = D3 direction + 2D6 distance. Pick low values for a
        // short throw-in.
        (RequestedRoll::ThrowIn, ChanceOutcome::Advance) => {
            state.fixes.fix_d3(1);
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(1);
        }
        // FoulArmor: two D6s. (1, 2) means no break + no doubles —
        // doubles would trigger the foul send-off rule, which we want
        // to avoid until a foul scenario specifically calls for it.
        (RequestedRoll::FoulArmor(_), ChanceOutcome::Advance) => {
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(2);
        }
        // FoulInjury: same (1, 2) pattern — Stunned, no ejection.
        (RequestedRoll::FoulInjury(_, _), ChanceOutcome::Advance) => {
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(2);
        }
        // BlockDice: hand off to the engine's block-dice policy via a
        // deterministic Pow per die. Matches `BlockDicePolicy::
        // KnockdownAtAdvantage` semantics; the player-side die
        // selection that follows is collapsed by
        // `block_dice::scripted_pick` in the dynamics.
        (RequestedRoll::BlockDice(_), ChanceOutcome::Advance) => {
            state.fixes.fix_blockdice(BlockDice::Pow);
            state.fixes.fix_blockdice(BlockDice::Pow);
            state.fixes.fix_blockdice(BlockDice::Pow);
        }
        // Raw value rolls — pick low constants.
        (RequestedRoll::D6, ChanceOutcome::Advance) => state.fixes.fix_d6(1),
        (RequestedRoll::Sum2D6, ChanceOutcome::Advance) => {
            state.fixes.fix_d6(1);
            state.fixes.fix_d6(1);
        }
        (RequestedRoll::D6ThreeOutcomes(_, _), ChanceOutcome::Advance) => {
            // Pick a 6 so the highest target is met (= "Pass" outcome).
            state.fixes.fix_d6(6);
        }
        (RequestedRoll::Sum2D6ThreeOutcomes(_, _), ChanceOutcome::Advance) => {
            state.fixes.fix_d6(6);
            state.fixes.fix_d6(6);
        }
        (RequestedRoll::Coin, ChanceOutcome::Advance) => {
            state
                .fixes
                .fix_coin(botbowl_engine::core::dices::Coin::Heads);
        }

        (other, outcome) => panic!(
            "fix_for_outcome: unhandled (roll, outcome) = ({:?}, {:?}) — \
             extend roll_outcomes::fix_for_outcome when a lecture surfaces it",
            other, outcome
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

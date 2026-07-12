use botbowl_engine::core::dices::{BlockDice, Coin, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6, D8};
use botbowl_engine::core::model::Direction;

use crate::action::BbAction;

/// Enumerate the possible outcomes of a `RequestedRoll` as MCTS chance
/// children, each carrying the concrete engine `RollResult` that
/// `apply_action` will feed straight into `SomeProcInput::Roll`, paired
/// with the probability that it occurs.
///
/// - `D6PassFail` / `Sum2D6PassFail` → two children, `RollResult::Pass`
///   and `RollResult::Fail`, with probabilities from
///   `RollTarget::success_prob`.
/// - Every other roll type → a single child carrying the deterministic
///   `RollResult` from [`scripted_result`]. Single-child keeps the
///   search tree bounded; the trade-off is no probabilistic averaging
///   across the outcomes of those rolls — fine for the deterministic-
///   policy curriculum, revisit when we get to genuinely probabilistic
///   policies.
///
/// The result carried by each child is a pure function of `req` plus the
/// chosen branch. That determinism is load-bearing: two descent paths
/// reaching the same chance edge must produce identical post-roll states
/// or the search DAG silently splits and recombination breaks.
pub fn enumerate(req: &RequestedRoll) -> Vec<BbAction> {
    match req {
        RequestedRoll::D6PassFail(target) => {
            let p_pass = target.success_prob();
            vec![
                BbAction::chance(RollResult::Pass, p_pass),
                BbAction::chance(RollResult::Fail, 1.0 - p_pass),
            ]
        }
        RequestedRoll::Sum2D6PassFail(target) => {
            let p_pass = target.success_prob();
            vec![
                BbAction::chance(RollResult::Pass, p_pass),
                BbAction::chance(RollResult::Fail, 1.0 - p_pass),
            ]
        }
        _ => vec![BbAction::chance(scripted_result(req), 1.0)],
    }
}

/// The single deterministic `RollResult` used to collapse a non-pass/fail
/// roll to one chance child. Picking a fixed value (rather than letting
/// the engine resolve via RNG) is what makes the same (parent, roll) edge
/// yield the same child state on every descent path, so state-hash
/// recombination holds and the tree doesn't fan out.
///
/// The pass/fail rolls are handled directly by [`enumerate`] (they branch
/// into two outcomes), so they are unreachable here.
fn scripted_result(req: &RequestedRoll) -> RollResult {
    // The engine accepts a `D8` constructed from a `Direction`, so wrap
    // that here to keep the match arms tight.
    let d8_up = || D8::from(Direction::up());
    match req {
        // D8 is used for ball bounces and scatter directions. Any
        // constant direction is fine; we just need *a* deterministic
        // outcome so two paths to the same logical position recombine.
        RequestedRoll::D8 => RollResult::D8(d8_up()),
        // Deviate = D6 (distance) + D8 (direction). Minimum scatter +
        // up — the ball barely moves.
        RequestedRoll::Deviate => RollResult::Deviate(D6::One, d8_up()),
        // Scatter = three D8 directions. Pick the same direction each
        // time; the engine treats the sequence as separate bounces.
        RequestedRoll::Scatter => RollResult::Scatter(d8_up(), d8_up(), d8_up()),
        // ThrowIn = D3 direction + 2D6 distance. Pick low values for a
        // short throw-in.
        RequestedRoll::ThrowIn => RollResult::ThrowIn {
            direction: D3::One,
            distance: Sum2D6::Two,
        },
        RequestedRoll::FoulArmor(..) => RollResult::FoulArmor {
            broken: false,
            ejected: false,
        },
        RequestedRoll::FoulInjury(..) => RollResult::FoulInjury {
            outcome: botbowl_engine::core::model::InjuryOutcome::Stunned,
            ejected: false,
        },
        // BlockDice: a deterministic Pow per die. Matches
        // `BlockDicePolicy::KnockdownAtAdvantage` semantics; the
        // player-side die selection that follows is collapsed by
        // `block_dice::scripted_pick` in the dynamics. Plan 009:
        // push exactly `num_dices` fixes (not a constant 3) to
        // avoid stale fixes leaking into unrelated later block rolls.
        RequestedRoll::BlockDice(n) => {
            let mut dices: [Option<BlockDice>; 3] = [None, None, None];
            for slot in dices.iter_mut().take(u8::from(*n) as usize) {
                *slot = Some(BlockDice::Pow);
            }
            RollResult::BlockDice(dices)
        }
        // Raw value rolls — pick low constants.
        RequestedRoll::D6 => RollResult::D6(D6::One),
        RequestedRoll::Sum2D6 => RollResult::Sum2D6(Sum2D6::Two),
        RequestedRoll::D6ThreeOutcomes(_, _) => RollResult::Pass,
        RequestedRoll::Sum2D6ThreeOutcomes(_, _) => RollResult::Pass,
        RequestedRoll::Coin => RollResult::Coin(Coin::Heads),

        RequestedRoll::D6PassFail(_) | RequestedRoll::Sum2D6PassFail(_) => unreachable!(
            "scripted_result: pass/fail rolls are branched by enumerate, not scripted: {:?}",
            req
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

    /// Extract the `RollResult` a chance action carries (panics if it
    /// isn't a `Chance` action).
    fn result_of(a: &BbAction) -> RollResult {
        match a {
            BbAction::Chance { result, .. } => *result,
            other => panic!("expected a Chance action, got {:?}", other),
        }
    }

    /// A single-child roll collapses to exactly one chance outcome with
    /// probability 1.0.
    fn sole_result(req: &RequestedRoll) -> RollResult {
        let outcomes = enumerate(req);
        assert_eq!(outcomes.len(), 1, "expected a single outcome for {:?}", req);
        assert!(probs_sum_to_one(&outcomes), "probability must be 1.0 for {:?}", req);
        result_of(&outcomes[0])
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
            assert_eq!(result_of(&outcomes[0]), RollResult::Pass);
            assert_eq!(result_of(&outcomes[1]), RollResult::Fail);
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
        for target in [Sum2D6Target::FourPlus, Sum2D6Target::SevenPlus, Sum2D6Target::NinePlus] {
            let outcomes = enumerate(&RequestedRoll::Sum2D6PassFail(target));
            assert_eq!(result_of(&outcomes[0]), RollResult::Pass);
            assert_eq!(result_of(&outcomes[1]), RollResult::Fail);
            assert!(
                probs_sum_to_one(&outcomes),
                "probabilities for {:?} don't sum to 1.0",
                target
            );
        }
    }

    #[test]
    fn d8_returns_single_up_direction() {
        let up = D8::from(Direction::up());
        assert_eq!(sole_result(&RequestedRoll::D8), RollResult::D8(up));
    }

    #[test]
    fn deviate_returns_single_min_distance_up() {
        let up = D8::from(Direction::up());
        assert_eq!(sole_result(&RequestedRoll::Deviate), RollResult::Deviate(D6::One, up));
    }

    #[test]
    fn scatter_uses_three_up_directions() {
        let up = D8::from(Direction::up());
        assert_eq!(sole_result(&RequestedRoll::Scatter), RollResult::Scatter(up, up, up));
    }

    #[test]
    fn block_dice_has_exactly_num_dices_of_pow() {
        for n in [
            NumBlockDices::One,
            NumBlockDices::Two,
            NumBlockDices::Three,
            NumBlockDices::TwoUphill,
            NumBlockDices::ThreeUphill,
        ] {
            let RollResult::BlockDice(dices) = sole_result(&RequestedRoll::BlockDice(n)) else {
                panic!("expected BlockDice result for {:?}", n);
            };
            let count = dices.iter().filter(|d| d.is_some()).count();
            assert_eq!(count, u8::from(n) as usize, "wrong dice count for {:?}", n);
            assert!(
                dices.iter().flatten().all(|d| *d == BlockDice::Pow),
                "all dice should be Pow for {:?}",
                n
            );
        }
    }

    #[test]
    fn throw_in_uses_d3_one_and_min_distance() {
        assert_eq!(
            sole_result(&RequestedRoll::ThrowIn),
            RollResult::ThrowIn {
                direction: D3::One,
                distance: Sum2D6::Two,
            }
        );
    }

    // The remaining tests pin the scripted-chance behaviour: foul armour
    // stays intact against strong armour but breaks against weak armour,
    // and the injury roll collapses to Stunned. These scripts are
    // load-bearing for MCTS recombination — two paths to the same chance
    // outcome must produce identical post-roll states, or the DAG
    // silently splits. See the `Foul armor breaks` and `Ball bounce/
    // scatter` sections of plan 003.

    #[test]
    fn foul_armor_holds_for_high_av() {
        // SevenPlus target ~ AV 7. Roll-of-3 (the scripted constant) is
        // a fail; armour holds, no injury cascade triggered.
        let result = sole_result(&RequestedRoll::FoulArmor(Sum2D6Target::SevenPlus));
        match result {
            RollResult::FoulArmor { broken, ejected } => {
                assert!(!broken, "expected armour to hold at AV 7");
                assert!(!ejected, "fouler must not be ejected on the scripted path");
            }
            other => panic!("expected FoulArmor result, got {:?}", other),
        }
    }

    #[test]
    fn foul_armor_breaks_for_av_three() {
        // ThreePlus target — armour needing just 3+ to break (an
        // already-injured / shoeless target). Roll-of-3 succeeds
        // against ThreePlus → armour broken. Documents the asymmetry:
        // weak armour still cascades into the injury roll, which is
        // itself scripted to Stunned (see test below).
        let result = sole_result(&RequestedRoll::FoulArmor(Sum2D6Target::ThreePlus));
        match result {
            RollResult::FoulArmor { broken, .. } => assert!(broken, "roll-of-3 should beat ThreePlus"),
            other => panic!("expected FoulArmor result, got {:?}", other),
        }
    }

    #[test]
    fn foul_injury_collapses_to_stunned() {
        use botbowl_engine::core::model::InjuryOutcome;
        // Typical Blood Bowl injury thresholds: KO at 8+, Cas at 10+.
        // Roll-of-3 misses both → Stunned. Scripting this collapses
        // the injury sub-tree to a single deterministic outcome.
        let result = sole_result(&RequestedRoll::FoulInjury(
            Sum2D6Target::EightPlus,
            Sum2D6Target::TenPlus,
        ));
        match result {
            RollResult::FoulInjury { outcome, ejected } => {
                assert_eq!(outcome, InjuryOutcome::Stunned);
                assert!(!ejected);
            }
            other => panic!("expected FoulInjury result, got {:?}", other),
        }
    }

    #[test]
    fn three_plus_pass_probability_is_4_over_6() {
        let outcomes = enumerate(&RequestedRoll::D6PassFail(D6Target::ThreePlus));
        let pass = outcomes
            .iter()
            .find_map(|a| match a {
                BbAction::Chance {
                    result: RollResult::Pass,
                    prob_bits,
                } => Some(f32::from_bits(*prob_bits)),
                _ => None,
            })
            .unwrap();
        assert!((pass - 4.0 / 6.0).abs() < 1e-5, "expected 4/6, got {}", pass);
    }
}

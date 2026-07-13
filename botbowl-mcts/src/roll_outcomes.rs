use botbowl_engine::core::dices::{BlockDice, Coin, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6, D8};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Direction, Position};

use crate::action::BbAction;

/// Enumerate the possible outcomes of a `RequestedRoll` as MCTS chance
/// children, each carrying the concrete engine `RollResult` that
/// `apply_action` will feed straight into `SomeProcInput::Roll`, paired
/// with the probability that it occurs.
///
/// - `D6PassFail` / `Sum2D6PassFail` → two children, `RollResult::Pass`
///   and `RollResult::Fail`, with probabilities from
///   `RollTarget::success_prob`.
/// - `D8` while a `Bounce` is on top of the proc stack → the reduced
///   bounce enumeration in [`bounce_outcomes`] (settle squares + a
///   collapsed out-of-bounds child, or the surrounding player squares
///   minus already-visited ones).
/// - Every other roll type (including a `D8` that isn't a live ball
///   bounce) → a single child carrying the deterministic `RollResult`
///   from [`scripted_result`]. Single-child keeps the search tree
///   bounded; the trade-off is no probabilistic averaging across the
///   outcomes of those rolls — fine for the deterministic-policy
///   curriculum, revisit when we get to genuinely probabilistic policies.
///
/// The result carried by each child is a pure function of `state` plus
/// `req` plus the chosen branch. That determinism is load-bearing: two
/// descent paths reaching the same chance edge must produce identical
/// post-roll states or the search DAG silently splits and recombination
/// breaks. (`bounce_outcomes` reads only board occupancy, OOB geometry
/// and `state.bounce_squares` — all state fields, so it stays pure.)
pub fn enumerate(state: &GameState, req: &RequestedRoll) -> Vec<BbAction> {
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
        RequestedRoll::D8 => enumerate_d8(state),
        _ => vec![BbAction::chance(scripted_result(req), 1.0)],
    }
}

/// The single scripted D8 outcome (bounce/scatter direction "up"), used
/// whenever a D8 isn't a live ball bounce we want to reason about.
fn scripted_d8() -> BbAction {
    BbAction::chance(RollResult::D8(D8::from(Direction::up())), 1.0)
}

/// Dispatch a `D8` roll. Only when the proc-stack top is a `Bounce` do we
/// reason about *where* the ball is going and prune the fan-out; any other
/// D8 (e.g. kickoff scatter) collapses to the scripted single direction.
fn enumerate_d8(state: &GameState) -> Vec<BbAction> {
    if state.proc_stack_top() != Some("Bounce") {
        return vec![scripted_d8()];
    }
    bounce_outcomes(state)
}

/// Chance children for a live ball bounce, reduced to keep the tree small.
///
/// The 8 D8 directions off the ball's current square are classified by
/// what the ball hits:
/// - **empty in-bounds square** → the ball settles there; one child each.
/// - **out of bounds** → a throw-in. *All* OOB directions collapse into a
///   single child (one representative direction) whose probability is
///   weighted by how many rolls go OOB — we don't care *which* edge it
///   left by, only that a throw-in happens.
/// - **occupied square** → the ball bounces off and keeps going.
///
/// If at least one empty or OOB square exists the ball can come to rest,
/// so we present *only* those settling / throw-in outcomes and drop the
/// player bounces entirely (the search doesn't model the ball ricocheting
/// off players once it could have landed). If the ball is fully boxed in
/// by players, we instead present each player-bounce direction so the
/// search can follow the ball onward — excluding any square already in
/// `state.bounce_squares`, which the ball has bounced through this
/// sequence, to avoid revisiting states and looping.
///
/// Probabilities are renormalised to sum to 1 across the kept children
/// (we drop branches in both cases), so the chance-node backprop
/// expectation stays a proper distribution.
fn bounce_outcomes(state: &GameState) -> Vec<BbAction> {
    let Some(ball_pos) = state.get_ball_position() else {
        return vec![scripted_d8()];
    };

    let mut empty: Vec<D8> = Vec::new(); // settles here
    let mut oob: Vec<D8> = Vec::new(); // throw-in (collapsed)
    let mut onto_player: Vec<(D8, Position)> = Vec::new(); // keeps bouncing
    for dir in Direction::all_directions_as_array() {
        let target = ball_pos + dir;
        let d8 = D8::from(dir);
        if state.is_out(target) {
            oob.push(d8);
        } else if state.get_player_at(target).is_some() {
            onto_player.push((d8, target));
        } else {
            empty.push(d8);
        }
    }

    const P_EACH: f32 = 1.0 / 8.0;
    let mut outcomes: Vec<BbAction> = Vec::new();

    if !empty.is_empty() || !oob.is_empty() {
        // The ball can come to rest: settle on each empty square, or (once)
        // sail out of bounds. Player ricochets are dropped here.
        for d8 in empty {
            outcomes.push(BbAction::chance(RollResult::D8(d8), P_EACH));
        }
        if let Some(&rep) = oob.first() {
            outcomes.push(BbAction::chance(RollResult::D8(rep), P_EACH * oob.len() as f32));
        }
    } else {
        // Surrounded by players — follow the ball onto each of them, but
        // skip squares it has already bounced through this sequence.
        for (d8, target) in onto_player {
            if !state.bounce_squares.contains(&target) {
                outcomes.push(BbAction::chance(RollResult::D8(d8), P_EACH));
            }
        }
    }

    // Everything got pruned (e.g. fully boxed in by already-visited
    // squares): keep the chance node expandable with the scripted outcome
    // rather than emitting zero children.
    if outcomes.is_empty() {
        return vec![scripted_d8()];
    }

    renormalize(&mut outcomes);
    outcomes
}

/// Scale the probabilities of a set of chance children so they sum to 1,
/// preserving their relative weights. No-op when they already sum to 1.
fn renormalize(outcomes: &mut [BbAction]) {
    let total: f32 = outcomes.iter().filter_map(|a| a.prob_f32()).sum();
    if total <= 0.0 {
        return;
    }
    for a in outcomes.iter_mut() {
        if let BbAction::Chance { prob_bits, .. } = a {
            *prob_bits = (f32::from_bits(*prob_bits) / total).to_bits();
        }
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
    use botbowl_engine::core::gamestate::{DiceMode, GameStateBuilder};
    use botbowl_engine::core::model::{Action, BallState, Coord, SomeProcInput};
    use botbowl_engine::core::table::{NumBlockDices, PosAT, SimpleAT};

    /// A plain state that is *not* mid-bounce, so `enumerate` routes D8
    /// through the scripted single-outcome path. Enumeration of the
    /// non-D8 roll types ignores the state entirely.
    fn dummy_state() -> GameState {
        GameStateBuilder::new().add_home_player(Position::new((5, 5))).build()
    }

    /// Build a state paused on the D8 roll of a live ball `Bounce`: a home
    /// player moves onto a loose ball, fails the pickup, and the engine
    /// (in `RegisterRolls`) stops at the bounce's D8 request. The ball sits
    /// on `ball_pos`. `setup` runs on the built state before the drive, to
    /// place extra players / walls.
    fn state_paused_on_bounce(ball_pos: Position, setup: impl FnOnce(&mut GameState)) -> GameState {
        let start_pos = Position::new((ball_pos.x - 1, ball_pos.y));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();
        setup(&mut state);
        state.set_dice_mode(DiceMode::RegisterRolls);
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start_pos)));
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, ball_pos)));
        // Pickup is a D6PassFail chance node — resolve it as a failure, then
        // decline the offered reroll, so a Bounce is pushed and the engine
        // pauses on its D8.
        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
        assert_eq!(state.proc_stack_top(), Some("Bounce"), "expected to be mid-bounce");
        assert_eq!(state.pending_roll, Some(RequestedRoll::D8), "expected a pending D8");
        state
    }

    /// The set of directions (as target squares) the enumerated outcomes
    /// would send the ball, relative to `ball_pos`.
    fn target_squares(outcomes: &[BbAction], ball_pos: Position) -> Vec<Position> {
        outcomes
            .iter()
            .map(|a| match result_of(a) {
                RollResult::D8(d8) => ball_pos + Direction::from(d8),
                other => panic!("expected a D8 result, got {:?}", other),
            })
            .collect()
    }

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
        let outcomes = enumerate(&dummy_state(), req);
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
            let outcomes = enumerate(&dummy_state(), &RequestedRoll::D6PassFail(target));
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
            let outcomes = enumerate(&dummy_state(), &RequestedRoll::Sum2D6PassFail(target));
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
        // A D8 that isn't a live ball bounce (dummy_state is mid-turn, not
        // mid-bounce) collapses to the single scripted "up" direction.
        let up = D8::from(Direction::up());
        assert_eq!(sole_result(&RequestedRoll::D8), RollResult::D8(up));
    }

    /// End-to-end: a real failed-pickup bounce with all eight squares free
    /// fans out into one settling child per direction (each 1/8), proving
    /// `enumerate` routes a genuine `Bounce`'s D8 into `bounce_outcomes`.
    #[test]
    fn enumerate_bounce_settles_on_all_free_squares() {
        let ball_pos = Position::new((5, 5));
        let state = state_paused_on_bounce(ball_pos, |_| {});
        let outcomes = enumerate(&state, &RequestedRoll::D8);

        assert_eq!(outcomes.len(), 8, "all 8 neighbours free → 8 settling children");
        assert!(probs_sum_to_one(&outcomes));
        let targets = target_squares(&outcomes, ball_pos);
        for dir in Direction::all_directions_as_array() {
            assert!(
                targets.contains(&(ball_pos + dir)),
                "expected a child settling at {:?}",
                ball_pos + dir
            );
        }
    }

    /// All out-of-bounds directions collapse into a *single* throw-in
    /// child whose probability is weighted by how many rolls go OOB.
    #[test]
    fn bounce_collapses_out_of_bounds_into_one_child() {
        // Ball against the left wall (x == 1): the three left-ward
        // directions (x == 0) are OOB, the other five are empty pitch.
        let ball_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new().build();
        state.set_ball(BallState::InAir(ball_pos));

        let outcomes = bounce_outcomes(&state);
        let targets = target_squares(&outcomes, ball_pos);

        let oob: Vec<_> = targets.iter().filter(|p| state.is_out(**p)).collect();
        assert_eq!(oob.len(), 1, "3 OOB directions must collapse to 1 child, got {:?}", targets);
        assert_eq!(targets.len(), 6, "5 empty squares + 1 collapsed OOB");
        assert!(probs_sum_to_one(&outcomes));

        // The collapsed child carries 3/8 (three OOB rolls), the settling
        // children 1/8 each — no renormalisation needed since 5/8+3/8 = 1.
        let oob_prob = outcomes
            .iter()
            .find(|a| matches!(result_of(a), RollResult::D8(d8) if state.is_out(ball_pos + Direction::from(d8))))
            .and_then(|a| a.prob_f32())
            .unwrap();
        assert!((oob_prob - 3.0 / 8.0).abs() < 1e-5, "expected 3/8 for OOB, got {}", oob_prob);
    }

    /// When the ball can settle, squares occupied by players are dropped
    /// (the search doesn't model ricochets off players once it could land)
    /// and the surviving children are renormalised to a proper distribution.
    #[test]
    fn bounce_drops_player_squares_when_it_can_settle() {
        let ball_pos = Position::new((5, 5));
        let occupied = ball_pos + Direction::right();
        let state = {
            let mut s = GameStateBuilder::new().add_away_player(occupied).build();
            s.set_ball(BallState::InAir(ball_pos));
            s
        };

        let outcomes = bounce_outcomes(&state);
        let targets = target_squares(&outcomes, ball_pos);

        assert_eq!(targets.len(), 7, "7 free neighbours, the occupied one dropped");
        assert!(!targets.contains(&occupied), "must not bounce onto the occupied square");
        assert!(probs_sum_to_one(&outcomes), "kept children must renormalise to 1");
    }

    /// Fully boxed in by players: the ball must keep bouncing onto them, so
    /// every player direction is a child — except squares already recorded
    /// in `bounce_squares`, which are skipped to avoid revisiting / looping.
    #[test]
    fn bounce_surrounded_explores_players_minus_visited() {
        let ball_pos = Position::new((5, 5));
        let neighbours: Vec<Position> = Direction::all_directions_as_array()
            .iter()
            .map(|d| ball_pos + *d)
            .collect();

        let coords: Vec<(Coord, Coord)> = neighbours.iter().map(|p| (p.x, p.y)).collect();
        let mut state = GameStateBuilder::new().add_away_players(&coords).build();
        state.set_ball(BallState::InAir(ball_pos));

        // Pretend the ball has already bounced through two of the eight
        // surrounding squares this sequence.
        state.bounce_squares.clear();
        let visited = [neighbours[0], neighbours[3]];
        state.bounce_squares.extend(visited);

        let outcomes = bounce_outcomes(&state);
        let targets = target_squares(&outcomes, ball_pos);

        assert_eq!(targets.len(), 6, "8 player squares minus 2 already-visited");
        for v in visited {
            assert!(!targets.contains(&v), "visited square {:?} must be excluded", v);
        }
        for t in &targets {
            assert!(neighbours.contains(t), "every child must land on a surrounding player");
        }
        assert!(probs_sum_to_one(&outcomes));
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
        let outcomes = enumerate(&dummy_state(), &RequestedRoll::D6PassFail(D6Target::ThreePlus));
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

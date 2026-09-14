use botbowl_engine::core::dices::{BlockDice, Coin, RequestedRoll, RollResult, RollTarget, Sum2D6, D3, D6, D8};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Direction, Position};
use botbowl_engine::core::procedures::block_procs::Push;
use botbowl_engine::core::procedures::AnyProc;
use botbowl_engine::core::table::{NumBlockDices, Skill};

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
/// - `BlockDice` while a `Block` is on top of the proc stack → one child
///   per resolved block outcome in [`block_outcomes`] (defender down and
///   pushed / down in place / pushed / nothing / both down / attacker
///   down, plus the attacker's down-vs-down-and-push choice), weighted by
///   the exact probability the picker ends up with that outcome given
///   both players' skills and the push geometry.
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
        RequestedRoll::ThrowIn => vec![throw_in_outcome(state)],
        RequestedRoll::BlockDice(n) => block_outcomes(state, *n),
        _ => vec![BbAction::chance(scripted_result(req), 1.0)],
    }
}

/// What one block die *does* once skills and board position are applied,
/// ordered from the attacker's best to the attacker's worst. The picker
/// (attacker on a normal block, defender on an uphill one) chooses among
/// the outcomes the rolled dice offer, so this order is the preference
/// that resolves a multi-die roll — the defender's preference is the
/// reverse.
///
/// `DefDownPush` and `DefDownNoPush` are deliberately *not* ranked
/// against each other for the attacker: knocking the defender down in
/// place (Block skill turning a `BothDown` into a one-sided knockdown)
/// versus knocking them down *and* moving them is a genuine positional
/// choice, so a roll offering both becomes [`Resolved::DefDownChoice`]
/// and the search decides. The defender, picking uphill, takes the
/// no-push variant (no displacement, no crowd risk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BlockOutcome {
    /// Defender knocked down and pushed: `Pow`, `PowPush` against a
    /// defender without Dodge — or *any* push die when the only push
    /// square is the crowd (the defender leaves the pitch either way).
    DefDownPush,
    /// Defender knocked down where they stand: `BothDown` when only the
    /// attacker has Block.
    DefDownNoPush,
    /// Defender pushed back, nobody down: `Push`, or `PowPush` against Dodge.
    Push,
    /// Nobody moves, nobody falls: `BothDown` when both have Block.
    NothingHappens,
    /// Both players down, turnover: `BothDown` when neither has Block.
    AllDown,
    /// Attacker down, turnover: `Skull`, or `BothDown` when only the
    /// defender has Block.
    AttDown,
}

/// The picker's resolution of one rolled combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Resolved {
    Single(BlockOutcome),
    /// Attacker may take either `DefDownPush` or `DefDownNoPush`.
    DefDownChoice,
}

/// The six faces of a block die.
const BLOCK_FACES: [BlockDice; 6] = [
    BlockDice::Skull,
    BlockDice::BothDown,
    BlockDice::Push,
    BlockDice::Push,
    BlockDice::PowPush,
    BlockDice::Pow,
];

/// Skill / position context that decides what each die face does.
#[derive(Debug, Clone, Copy)]
struct BlockContext {
    attacker_block: bool,
    defender_block: bool,
    defender_dodge: bool,
    crowd_push: bool,
}

impl BlockContext {
    fn outcome_of(&self, die: BlockDice) -> BlockOutcome {
        use BlockOutcome::*;
        let push_effect = if self.crowd_push { DefDownPush } else { Push };
        match die {
            BlockDice::Skull => AttDown,
            BlockDice::BothDown => match (self.attacker_block, self.defender_block) {
                (true, true) => NothingHappens,
                (true, false) => DefDownNoPush,
                (false, true) => AttDown,
                (false, false) => AllDown,
            },
            BlockDice::Push => push_effect,
            BlockDice::PowPush if self.defender_dodge => push_effect,
            BlockDice::PowPush | BlockDice::Pow => DefDownPush,
        }
    }

    /// The die face that, rolled on every die, forces the engine to
    /// produce exactly `outcome` in this context (the engine then offers
    /// a single `Select*`, so no pick heuristic can steer it elsewhere).
    fn representative(&self, outcome: BlockOutcome) -> BlockDice {
        use BlockOutcome::*;
        match outcome {
            DefDownPush => BlockDice::Pow,
            DefDownNoPush | NothingHappens | AllDown => BlockDice::BothDown,
            Push => BlockDice::Push,
            AttDown => BlockDice::Skull,
        }
    }
}

/// Chance children for a block roll: one child per *resolved outcome*
/// rather than per die combination, each carrying a representative dice
/// array that makes the engine produce exactly that outcome once the
/// scripted die pick (`block_dice::scripted_pick`) runs in
/// `apply_action`'s quiescent loop.
///
/// All `6^n` face combinations (n ≤ 3) are enumerated and classified by
/// [`BlockContext::outcome_of`], then resolved by whoever picks — the
/// attacker on `One`/`Two`/`Three` (best outcome for them), the defender
/// on `TwoUphill`/`ThreeUphill` (worst for the attacker). Probabilities
/// are exact counts over `6^n`. The attacker-Block-only roll showing both
/// a knockdown-and-push die and a `BothDown` is emitted as its own child
/// carrying `[Pow, BothDown, ..]`, on which `scripted_pick` declines to
/// choose so the search sees both `SelectPow` and `SelectBothDown`.
///
/// Reads only the `Block` proc on top of the stack, the two players'
/// skills and the push geometry, so it is a pure function of `state`.
/// Falls back to the all-`Pow` script when not actually mid-block
/// (dummy states in tests).
fn block_outcomes(state: &GameState, n: NumBlockDices) -> Vec<BbAction> {
    let Some(AnyProc::Block(block)) = state.proc_stack_peek() else {
        return vec![BbAction::chance(scripted_result(&RequestedRoll::BlockDice(n)), 1.0)];
    };
    let (Some(attacker), Ok(defender)) = (state.get_active_player(), state.get_player(block.defender())) else {
        return vec![BbAction::chance(scripted_result(&RequestedRoll::BlockDice(n)), 1.0)];
    };
    let ctx = BlockContext {
        attacker_block: attacker.has_skill(Skill::Block),
        defender_block: defender.has_skill(Skill::Block),
        defender_dodge: defender.has_skill(Skill::Dodge),
        crowd_push: Push::is_crowd_push(attacker.position, defender.position, state),
    };
    let num_dice = u8::from(n) as usize;
    let defender_picks = matches!(n, NumBlockDices::TwoUphill | NumBlockDices::ThreeUphill);

    // Count combinations per resolved outcome. Fixed-order buckets keep
    // the child order deterministic (Vec, not HashMap).
    let mut counts: Vec<(Resolved, u32)> = Vec::new();
    let total = 6u32.pow(num_dice as u32);
    for combo in 0..total {
        let mut rest = combo;
        let mut best = None::<BlockOutcome>;
        let mut worst = None::<BlockOutcome>;
        for _ in 0..num_dice {
            let o = ctx.outcome_of(BLOCK_FACES[(rest % 6) as usize]);
            rest /= 6;
            best = Some(best.map_or(o, |b| b.min(o)));
            worst = Some(worst.map_or(o, |w| w.max(o)));
        }
        let (best, worst) = (best.unwrap(), worst.unwrap());
        let resolved = if defender_picks {
            Resolved::Single(worst)
        } else if best == BlockOutcome::DefDownPush && has_face(&ctx, combo, num_dice, BlockOutcome::DefDownNoPush) {
            Resolved::DefDownChoice
        } else {
            Resolved::Single(best)
        };
        match counts.iter_mut().find(|(r, _)| *r == resolved) {
            Some((_, c)) => *c += 1,
            None => counts.push((resolved, 1)),
        }
    }

    counts
        .into_iter()
        .map(|(resolved, count)| {
            let mut dices: [Option<BlockDice>; 3] = [None, None, None];
            for (i, slot) in dices.iter_mut().take(num_dice).enumerate() {
                *slot = Some(match resolved {
                    Resolved::Single(o) => ctx.representative(o),
                    Resolved::DefDownChoice if i == 0 => BlockDice::Pow,
                    Resolved::DefDownChoice => BlockDice::BothDown,
                });
            }
            BbAction::chance(RollResult::BlockDice(dices), count as f32 / total as f32)
        })
        .collect()
}

/// Does die combination `combo` (base-6 encoded) contain a die whose
/// effect is `outcome`?
fn has_face(ctx: &BlockContext, combo: u32, num_dice: usize, outcome: BlockOutcome) -> bool {
    let mut rest = combo;
    (0..num_dice).any(|_| {
        let o = ctx.outcome_of(BLOCK_FACES[(rest % 6) as usize]);
        rest /= 6;
        o == outcome
    })
}

/// The single scripted throw-in child, picked so the ball lands **in
/// bounds** and so that the choice *mirrors with the board*.
///
/// The in-bounds part is load-bearing for termination: a constant
/// (direction, distance) can land straight back out on small boards; the
/// engine then re-requests the roll from the new boundary square, and on a
/// 3-row pitch those re-request states oscillate between two positions —
/// identical states recur and the search DAG gets a genuine cycle
/// (recon_mcts panics).
///
/// The *direction* part is plan 023's H-c. The old pick walked the D3
/// values in order and took the first that stayed in bounds; the engine's
/// table (`ThrowIn::get_throw_in_direction`) lists `D3::One` first, which
/// on a y-sideline is `(1, ±1)` — so both bots believed sideline throw-ins
/// always travel toward +x, i.e. always toward Away's end zone. That is a
/// modelling assumption in *board* coordinates: it flatters one side's
/// plans and punishes the other's. Preferring the axis-aligned throw
/// instead (`dx == 0`, else `dy == 0`) is invariant under
/// `x -> width-1-x`, so the modelled throw-in mirrors with the position.
///
/// Reading the `ThrowIn` proc's data keeps this a pure function of
/// `state`, same as `bounce_outcomes`.
fn throw_in_outcome(state: &GameState) -> BbAction {
    let scripted = |direction, distance| BbAction::chance(RollResult::ThrowIn { direction, distance }, 1.0);
    let Some(AnyProc::ThrowIn(throw_in)) = state.proc_stack_peek() else {
        // Not actually mid-throw-in (dummy states in tests): keep the old
        // constant short throw.
        return scripted(D3::One, Sum2D6::Two);
    };
    // Shortest distance first; some direction at distance 2 always lands
    // in bounds on any legal board (straight-in exists in every direction
    // triple and the playable cross-axis is ≥ 3).
    for distance in [Sum2D6::Two, Sum2D6::Three, Sum2D6::Four] {
        let pick = [D3::One, D3::Two, D3::Three]
            .into_iter()
            .filter(|&direction| !state.is_out(throw_in.target_square(direction, distance, state.board_dims)))
            .min_by_key(|&direction| axis_rank(throw_in.get_throw_in_direction(direction, state.board_dims)));
        if let Some(direction) = pick {
            return scripted(direction, distance);
        }
    }
    scripted(D3::One, Sum2D6::Two)
}

/// Ranks a direction by how axis-aligned it is, preferring `dx == 0`.
/// Both `dx == 0` and `dy == 0` map onto themselves as *classes* under
/// `x -> width-1-x`, so a choice made with this key mirrors with the
/// board, unlike one made in `ALL_DIRECTIONS` / `D3` order (plan 023 H-c).
fn axis_rank(dir: Direction) -> u8 {
    match (dir.dx == 0, dir.dy == 0) {
        (true, _) => 0,
        (_, true) => 1,
        _ => 2,
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
        if let Some(rep) = oob_representative(&oob) {
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

/// The single direction that stands for "the ball went out here".
///
/// The collapse itself is deliberate (we only care that a throw-in
/// happens, not which edge it crossed) but the *choice* of representative
/// is not free: it fixes the square the throw-in is taken from. The old
/// `oob.first()` took it in `ALL_DIRECTIONS` order, which starts with
/// `dx = +1` — so a ball leaving by the left wall was modelled as exiting
/// diagonally while its mirror image at the right wall exited straight
/// out. That is a modelling assumption in *board* coordinates, and it
/// biases the search for one side (plan 023, H-c).
///
/// Preferring the axis-aligned exit fixes it: `dx == 0` (straight out over
/// a sideline) is invariant under `x -> width-1-x`, and `dy == 0` maps
/// onto itself under the same reflection, so the representative always
/// mirrors with the position. A rectangular board's out-of-bounds set
/// always contains one of the two, so the fallback is unreachable in
/// practice.
fn oob_representative(oob: &[D8]) -> Option<D8> {
    oob.iter().min_by_key(|d8| axis_rank(Direction::from(**d8))).copied()
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
        // Scripted as a fixed roll-of-3 against the target: armour holds
        // for any realistic AV, but a weak (already-broken) 3+ target
        // still cascades into the (scripted, Stunned) injury roll — see
        // the `foul_armor_*` tests below, which pin this asymmetry.
        RequestedRoll::FoulArmor(target) => RollResult::FoulArmor {
            broken: target.is_success(Sum2D6::Three),
            ejected: false,
        },
        RequestedRoll::FoulInjury(..) => RollResult::FoulInjury {
            outcome: botbowl_engine::core::model::InjuryOutcome::Stunned,
            ejected: false,
        },
        // BlockDice: only reached by `block_outcomes`' fallback when the
        // state is not actually mid-block (no `Block` proc on top). A
        // deterministic Pow per die; exactly `num_dices` of them (plan
        // 009) so no stale fixes leak into later block rolls.
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

        RequestedRoll::D6PassFail(_) | RequestedRoll::Sum2D6PassFail(_) | RequestedRoll::ThrowIn => unreachable!(
            "scripted_result: pass/fail and throw-in rolls are handled by enumerate, not scripted: {:?}",
            req
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::dices::{D6Target, Sum2D6Target};
    use botbowl_engine::core::gamestate::{DiceMode, GameStateBuilder};
    use botbowl_engine::core::model::{Action, BallState, Coord, PlayerStatus, SomeProcInput, TeamType};
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
    /// child whose probability is weighted by how many rolls go OOB — and
    /// the representative is the axis-aligned exit, not `ALL_DIRECTIONS`'
    /// first (plan 023, H-c: that one always leaned +x).
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
        assert_eq!(
            oob.len(),
            1,
            "3 OOB directions must collapse to 1 child, got {:?}",
            targets
        );
        assert_eq!(targets.len(), 6, "5 empty squares + 1 collapsed OOB");
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(
            *oob[0],
            Position::new((0, 5)),
            "the representative must be the straight-out exit"
        );

        // The collapsed child carries 3/8 (three OOB rolls), the settling
        // children 1/8 each — no renormalisation needed since 5/8+3/8 = 1.
        let oob_prob = outcomes
            .iter()
            .find(|a| matches!(result_of(a), RollResult::D8(d8) if state.is_out(ball_pos + Direction::from(d8))))
            .and_then(|a| a.prob_f32())
            .unwrap();
        assert!(
            (oob_prob - 3.0 / 8.0).abs() < 1e-5,
            "expected 3/8 for OOB, got {}",
            oob_prob
        );
    }

    /// The set of children must be mirror-invariant: reflecting the ball
    /// across the pitch's long axis must reflect the outcome set exactly.
    /// `oob.first()` failed this — it preferred +x at both walls.
    #[test]
    fn bounce_outcomes_are_mirror_invariant() {
        let mirror = |p: Position, w: Coord| Position::new((w - 1 - p.x, p.y));
        let mut state = GameStateBuilder::new().build();
        let w = state.board_dims.width;

        let left = Position::new((1, 5));
        state.set_ball(BallState::InAir(left));
        let mut left_targets: Vec<Position> = target_squares(&bounce_outcomes(&state), left)
            .into_iter()
            .map(|p| mirror(p, w))
            .collect();

        let right = mirror(left, w);
        state.set_ball(BallState::InAir(right));
        let mut right_targets = target_squares(&bounce_outcomes(&state), right);

        left_targets.sort_by_key(|p| (p.x, p.y));
        right_targets.sort_by_key(|p| (p.x, p.y));
        assert_eq!(left_targets, right_targets);
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

    // ---- block outcomes -------------------------------------------------

    /// Build a state paused on a block roll: home attacker at `att`
    /// blocks the away defender at `def`. `setup` runs before the block
    /// is declared (add assists, skills, sideline geometry). Asserts the
    /// engine computed `expected_dice` so a test can't silently model a
    /// different roll than it thinks.
    fn state_paused_on_block(
        att: Position,
        def: Position,
        extra: &[(Position, TeamType)],
        expected_dice: NumBlockDices,
        setup: impl FnOnce(&mut GameState),
    ) -> GameState {
        let mut builder = GameStateBuilder::new();
        builder.add_home_player(att).add_away_player(def);
        for (pos, team) in extra {
            match team {
                TeamType::Home => builder.add_home_player(*pos),
                TeamType::Away => builder.add_away_player(*pos),
            };
        }
        let mut state = builder.build();
        setup(&mut state);
        state.set_dice_mode(DiceMode::RegisterRolls);
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartBlock, att)));
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Block, def)));
        assert_eq!(state.proc_stack_top(), Some("Block"), "expected to be mid-block");
        assert_eq!(state.pending_roll, Some(RequestedRoll::BlockDice(expected_dice)));
        state
    }

    fn give_skill(state: &mut GameState, pos: Position, skill: Skill) {
        let id = state.get_player_id_at(pos).unwrap();
        state.get_mut_player(id).unwrap().stats.give_skill(skill);
    }

    /// The dice array a block chance child carries.
    fn dice_of(a: &BbAction) -> [Option<BlockDice>; 3] {
        match result_of(a) {
            RollResult::BlockDice(d) => d,
            other => panic!("expected a BlockDice result, got {:?}", other),
        }
    }

    /// Probability of the child whose dice array is exactly `dice`
    /// (`None` if no such child).
    fn prob_of(outcomes: &[BbAction], dice: &[BlockDice]) -> Option<f32> {
        let want: Vec<Option<BlockDice>> = (0..3).map(|i| dice.get(i).copied()).collect();
        outcomes
            .iter()
            .find(|a| dice_of(a).to_vec() == want)
            .map(|a| a.prob_f32().unwrap())
    }

    fn assert_prob(outcomes: &[BbAction], dice: &[BlockDice], numer: u32, denom: u32) {
        let got = prob_of(outcomes, dice).unwrap_or_else(|| panic!("no child with dice {:?} in {:?}", dice, outcomes));
        let want = numer as f32 / denom as f32;
        assert!(
            (got - want).abs() < 1e-5,
            "child {:?}: expected {}/{} = {}, got {}",
            dice,
            numer,
            denom,
            want,
            got
        );
    }

    const ATT: Position = Position { x: 5, y: 5 };
    const DEF: Position = Position { x: 6, y: 5 };
    /// A home assist adjacent to the defender only → 2-dice block.
    const HOME_ASSIST: &[(Position, TeamType)] = &[(Position { x: 7, y: 6 }, TeamType::Home)];
    /// An away assist adjacent to the attacker only → 2-dice uphill.
    const AWAY_ASSIST: &[(Position, TeamType)] = &[(Position { x: 4, y: 4 }, TeamType::Away)];

    #[test]
    fn block_one_die_no_skills_has_four_outcomes() {
        let state = state_paused_on_block(ATT, DEF, &[], NumBlockDices::One, |_| {});
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::One));
        assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        assert!(probs_sum_to_one(&outcomes));
        assert_prob(&outcomes, &[BlockDice::Pow], 2, 6); // Pow + PowPush
        assert_prob(&outcomes, &[BlockDice::Push], 2, 6);
        assert_prob(&outcomes, &[BlockDice::BothDown], 1, 6);
        assert_prob(&outcomes, &[BlockDice::Skull], 1, 6);
    }

    #[test]
    fn block_two_dice_attacker_picks_best() {
        let state = state_paused_on_block(ATT, DEF, HOME_ASSIST, NumBlockDices::Two, |_| {});
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::Two));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        // P(≥1 Pow-ish) = 1 − (4/6)²
        assert_prob(&outcomes, &[BlockDice::Pow, BlockDice::Pow], 20, 36);
        // P(≥1 Push, no Pow-ish) = (4/6)² − (2/6)²
        assert_prob(&outcomes, &[BlockDice::Push, BlockDice::Push], 12, 36);
        // P(≥1 BothDown, no Push/Pow-ish) = (2/6)² − (1/6)²
        assert_prob(&outcomes, &[BlockDice::BothDown, BlockDice::BothDown], 3, 36);
        assert_prob(&outcomes, &[BlockDice::Skull, BlockDice::Skull], 1, 36);
    }

    #[test]
    fn block_attacker_block_only_splits_choice_from_forced_outcomes() {
        let state = state_paused_on_block(ATT, DEF, HOME_ASSIST, NumBlockDices::Two, |s| {
            give_skill(s, ATT, Skill::Block);
        });
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::Two));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 5, "{outcomes:?}");
        // P(≥1 Pow-ish, ≥1 BothDown) = 2·2·1/36
        assert_prob(&outcomes, &[BlockDice::Pow, BlockDice::BothDown], 4, 36);
        // P(≥1 Pow-ish) − choice = (1 − (4/6)²) − 4/36
        assert_prob(&outcomes, &[BlockDice::Pow, BlockDice::Pow], 16, 36);
        // P(≥1 BothDown, no Pow-ish) = (4/6)² − (3/6)²
        assert_prob(&outcomes, &[BlockDice::BothDown, BlockDice::BothDown], 7, 36);
        // P(≥1 Push, no BothDown/Pow-ish) = (3/6)² − (1/6)²
        assert_prob(&outcomes, &[BlockDice::Push, BlockDice::Push], 8, 36);
        assert_prob(&outcomes, &[BlockDice::Skull, BlockDice::Skull], 1, 36);
    }

    /// Both have Block: a `BothDown` leaves both standing *in place*, which
    /// is not the same as a push — it gets its own child.
    #[test]
    fn block_both_have_block_keeps_nothing_happens_distinct_from_push() {
        let state = state_paused_on_block(ATT, DEF, &[], NumBlockDices::One, |s| {
            give_skill(s, ATT, Skill::Block);
            give_skill(s, DEF, Skill::Block);
        });
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::One));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        assert_prob(&outcomes, &[BlockDice::Pow], 2, 6);
        assert_prob(&outcomes, &[BlockDice::Push], 2, 6);
        assert_prob(&outcomes, &[BlockDice::BothDown], 1, 6);
        assert_prob(&outcomes, &[BlockDice::Skull], 1, 6);
    }

    /// Only the defender has Block: `BothDown` fells the attacker alone.
    #[test]
    fn block_defender_block_only_folds_both_down_into_attacker_down() {
        let state = state_paused_on_block(ATT, DEF, &[], NumBlockDices::One, |s| give_skill(s, DEF, Skill::Block));
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::One));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 3, "{outcomes:?}");
        assert_prob(&outcomes, &[BlockDice::Skull], 2, 6);
        assert!(prob_of(&outcomes, &[BlockDice::BothDown]).is_none());
    }

    #[test]
    fn block_defender_dodge_turns_pow_push_into_push() {
        let state = state_paused_on_block(ATT, DEF, &[], NumBlockDices::One, |s| give_skill(s, DEF, Skill::Dodge));
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::One));
        assert!(probs_sum_to_one(&outcomes));
        assert_prob(&outcomes, &[BlockDice::Pow], 1, 6);
        assert_prob(&outcomes, &[BlockDice::Push], 3, 6);
    }

    /// Defender on the sideline with the attacker pushing straight out:
    /// every push die surfs them, so there is no "nobody down" child.
    #[test]
    fn block_crowd_push_folds_push_into_defender_down() {
        let def = Position::new((6, 1));
        let att = Position::new((6, 2));
        let state = state_paused_on_block(att, def, &[], NumBlockDices::One, |_| {});
        assert!(Push::is_crowd_push(att, def, &state));
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::One));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 3, "{outcomes:?}");
        assert_prob(&outcomes, &[BlockDice::Pow], 4, 6);
        assert!(prob_of(&outcomes, &[BlockDice::Push]).is_none());
        assert_prob(&outcomes, &[BlockDice::BothDown], 1, 6);
        assert_prob(&outcomes, &[BlockDice::Skull], 1, 6);
    }

    /// Uphill: the defender picks, so the resolution order flips.
    #[test]
    fn block_uphill_defender_picks_worst_for_attacker() {
        let state = state_paused_on_block(ATT, DEF, AWAY_ASSIST, NumBlockDices::TwoUphill, |_| {});
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::TwoUphill));
        assert!(probs_sum_to_one(&outcomes));
        assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        // P(≥1 Skull) = 1 − (5/6)²
        assert_prob(&outcomes, &[BlockDice::Skull, BlockDice::Skull], 11, 36);
        // P(≥1 BothDown, no Skull) = (5/6)² − (4/6)²
        assert_prob(&outcomes, &[BlockDice::BothDown, BlockDice::BothDown], 9, 36);
        // P(≥1 Push, no Skull/BothDown) = (4/6)² − (2/6)²
        assert_prob(&outcomes, &[BlockDice::Push, BlockDice::Push], 12, 36);
        assert_prob(&outcomes, &[BlockDice::Pow, BlockDice::Pow], 4, 36);
    }

    /// Uphill with attacker-Block-only: no choice child — the defender
    /// takes the knockdown without the push.
    #[test]
    fn block_uphill_attacker_block_only_has_no_choice_child() {
        let state = state_paused_on_block(ATT, DEF, AWAY_ASSIST, NumBlockDices::TwoUphill, |s| {
            give_skill(s, ATT, Skill::Block);
        });
        let outcomes = enumerate(&state, &RequestedRoll::BlockDice(NumBlockDices::TwoUphill));
        assert!(probs_sum_to_one(&outcomes));
        assert!(prob_of(&outcomes, &[BlockDice::Pow, BlockDice::BothDown]).is_none());
        // BothDown (def down, no push) beats Pow-ish for the defender:
        // P(≥1 BothDown, no Skull/Push) = (3/6)² − (2/6)²
        assert_prob(&outcomes, &[BlockDice::BothDown, BlockDice::BothDown], 5, 36);
        assert_prob(&outcomes, &[BlockDice::Pow, BlockDice::Pow], 4, 36);
    }

    /// Every block child carries exactly `n` dice, for every dice count.
    #[test]
    fn block_children_carry_exactly_num_dice() {
        for (n, extra) in [
            (NumBlockDices::One, &[][..]),
            (NumBlockDices::Two, HOME_ASSIST),
            (NumBlockDices::TwoUphill, AWAY_ASSIST),
        ] {
            let state = state_paused_on_block(ATT, DEF, extra, n, |_| {});
            let outcomes = enumerate(&state, &RequestedRoll::BlockDice(n));
            assert!(probs_sum_to_one(&outcomes));
            for a in &outcomes {
                let count = dice_of(a).iter().filter(|d| d.is_some()).count();
                assert_eq!(count, u8::from(n) as usize, "wrong dice count for {:?}: {:?}", n, a);
            }
        }
    }

    /// End-to-end through the engine: the choice child really leaves the
    /// attacker a two-way decision, the forced children really produce
    /// the modelled outcome.
    #[test]
    fn block_children_drive_the_engine_to_the_modelled_outcome() {
        let build = || {
            state_paused_on_block(ATT, DEF, HOME_ASSIST, NumBlockDices::Two, |s| {
                give_skill(s, ATT, Skill::Block);
            })
        };
        let def_id = build().get_player_id_at(DEF).unwrap();

        // Choice child: engine offers both picks, the script declines.
        let mut state = build();
        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::BlockDice([
            Some(BlockDice::Pow),
            Some(BlockDice::BothDown),
            None,
        ])));
        let simple = state.available_actions.get_simple();
        assert!(simple.contains(&SimpleAT::SelectPow) && simple.contains(&SimpleAT::SelectBothDown));
        assert_eq!(crate::block_dice::scripted_pick(&state), None);

        // Forced DefDownNoPush child: defender down where they stand.
        let mut state = build();
        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::BlockDice([
            Some(BlockDice::BothDown),
            Some(BlockDice::BothDown),
            None,
        ])));
        let pick = crate::block_dice::scripted_pick(&state).expect("single die → scripted");
        assert_eq!(pick, Action::Simple(SimpleAT::SelectBothDown));
        state.step_with_roll_or_action(SomeProcInput::Action(pick));
        let defender = state.get_player_unsafe(def_id);
        assert_eq!(defender.position, DEF);
        assert_eq!(defender.status, PlayerStatus::Down);
        assert_eq!(state.get_player_at(ATT).unwrap().status, PlayerStatus::Up);
    }

    #[test]
    fn block_dice_fallback_has_exactly_num_dices_of_pow() {
        // `dummy_state` is not mid-block, so `block_outcomes` falls back
        // to the all-Pow script.
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

    /// Every throw-in child must resolve in one roll. If one lands out of
    /// bounds the engine re-requests the roll from the new boundary square,
    /// and those re-request states oscillate between two positions on a
    /// 3-row board — identical states recur and the search DAG gets a
    /// genuine cycle (recon_mcts panics). The children must also cover the
    /// whole in-bounds D3 triple uniformly, not just its first member:
    /// `D3::One` is `(1, ±1)` on a y-sideline, so preferring it made both
    /// bots believe throw-ins always travel toward +x (plan 023, H-c).
    #[test]
    fn throw_in_outcome_lands_in_bounds_and_mirrors() {
        use botbowl_engine::core::model::BoardDims;

        // Runtime 10x5 board (playable 8x3, the smallest training tier).
        // Ball on the bottom edge (max_y = 3); a failed pickup bounces it
        // straight off the pitch → ThrowIn from (7, 3). The old constant
        // (D3::One, distance 2) pick targets (9, 1) — out of bounds.
        let ball_pos = Position::new((7, 3));
        let start_pos = Position::new((6, 3));
        let mut state = GameStateBuilder::new()
            .with_board_dims(BoardDims::new(10, 5, 2))
            .add_home_player(start_pos)
            .add_ball_pos(ball_pos)
            .build();
        state.set_dice_mode(DiceMode::RegisterRolls);
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start_pos)));
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, ball_pos)));
        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));
        state.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::D8(D8::from(Direction::down()))));
        assert_eq!(state.proc_stack_top(), Some("ThrowIn"), "expected to be mid-throw-in");
        assert_eq!(state.pending_roll, Some(RequestedRoll::ThrowIn));

        let outcomes = enumerate(&state, &RequestedRoll::ThrowIn);
        assert_eq!(outcomes.len(), 1, "throw-in stays a single scripted child");
        assert!(probs_sum_to_one(&outcomes));

        // Mirror-invariance: the modelled throw-in must be the straight-in
        // one (dx == 0 off a y-sideline), not `D3::One`'s +x diagonal.
        let RollResult::ThrowIn { direction, .. } = result_of(&outcomes[0]) else {
            panic!("expected a ThrowIn result")
        };
        let Some(AnyProc::ThrowIn(throw_in)) = state.proc_stack_peek() else {
            panic!("expected to be mid-throw-in")
        };
        let dir = throw_in.get_throw_in_direction(direction, state.board_dims);
        assert_eq!(dir.dx, 0, "throw-in direction leans along x: {dir:?}");

        state.step_with_roll_or_action(SomeProcInput::Roll(result_of(&outcomes[0])));
        assert_ne!(
            state.pending_roll,
            Some(RequestedRoll::ThrowIn),
            "scripted throw-in landed out of bounds and re-requested the roll — cycle risk"
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

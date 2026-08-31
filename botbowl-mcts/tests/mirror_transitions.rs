//! Is the engine's *transition function* mirror-invariant? (plan 023)
//!
//! `mirror_symmetry.rs` asserts that the things a search reads off a single
//! state — leaf value, priors, pruning, the legal-action set — all mirror.
//! That is necessary but not sufficient: a search also cares about where
//! actions *lead*. This file asserts the stronger property
//!
//! ```text
//! mirror(step(s, a, r)) == step(mirror(s), mirror(a), mirror(r))
//! ```
//!
//! over every legal action of a few hundred random mid-drive states, with
//! the dice pinned on both sides (`DiceMode::RegisterRolls`, mirroring
//! direction-valued results in `x`). Any surviving x-asymmetry in the rules
//! — a formation, a scatter table, a push resolution, a "first legal"
//! fallback inside a procedure — shows up here as a fingerprint mismatch,
//! localised to the exact action that caused it.
//!
//! This is the property plan 023 wanted after `ScriptedBot`'s touchback
//! tie-break, the kickoff-aim off-by-one and the throw-in/bounce
//! representatives were each found by measurement rather than by reading.

mod common;

use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::{Action as EngineAction, InjuryOutcome, SomeProcInput};
use botbowl_engine::core::table::{NumBlockDices, SimpleAT};

use common::{fingerprint, mirror_action, mirror_fingerprint, mirror_playable, mirror_roll, states, tier};

/// Advance a mirrored pair in lockstep until both are back at a decision
/// (or the game is over), feeding mirrored die results. Returns `Ok(())`
/// when both stopped in mirrored states, `Err(reason)` when they diverged,
/// and `Ok(())` early (skipped) when a roll type has no mirror.
enum Step {
    Mirrored,
    Diverged(String),
    Skipped,
}

fn step_pair(s: &mut GameState, m: &mut GameState, a: EngineAction) -> Step {
    let dims = tier();
    s.step_with_roll_or_action(SomeProcInput::Action(a));
    m.step_with_roll_or_action(SomeProcInput::Action(mirror_action(dims, a)));
    // The engine can pause on an arbitrarily long chain of rolls before the
    // next decision; walk them together.
    for _ in 0..64 {
        match (s.pending_roll, m.pending_roll) {
            (None, None) => break,
            (Some(rs), Some(rm)) => {
                if rs != rm {
                    return Step::Diverged(format!("pending_roll {rs:?} vs {rm:?}"));
                }
                // A concrete result for this request, then its mirror.
                let result = match sample_result(rs) {
                    Some(r) => r,
                    None => return Step::Skipped,
                };
                let mirrored = match mirror_roll(result) {
                    Some(r) => r,
                    None => return Step::Skipped,
                };
                s.step_with_roll_or_action(SomeProcInput::Roll(result));
                m.step_with_roll_or_action(SomeProcInput::Roll(mirrored));
            }
            (a, b) => {
                return Step::Diverged(format!("one side paused on a roll: {a:?} vs {b:?}"));
            }
        }
    }
    Step::Mirrored
}

/// One representative result per request type. Deliberately not random: the
/// point is a reproducible mirrored pair, not coverage of the dice.
/// Direction rolls take a *diagonal* so a reflection actually changes them
/// (an axis-aligned direction mirrors onto itself and would prove nothing).
fn sample_result(req: botbowl_engine::core::dices::RequestedRoll) -> Option<botbowl_engine::core::dices::RollResult> {
    use botbowl_engine::core::dices::{BlockDice, RequestedRoll as Req, RollResult as Res, Sum2D6, D6, D8};
    use botbowl_engine::core::model::Direction;
    let diag = D8::from(Direction::from((1, 1)));
    Some(match req {
        Req::BlockDice(n) => {
            let mut dice = [None, None, None];
            let count = match n {
                NumBlockDices::One => 1,
                NumBlockDices::Two | NumBlockDices::TwoUphill => 2,
                NumBlockDices::Three | NumBlockDices::ThreeUphill => 3,
            };
            for slot in dice.iter_mut().take(count) {
                *slot = Some(BlockDice::Pow);
            }
            Res::BlockDice(dice)
        }
        Req::Coin => return None, // the toss has no board geometry to mirror
        Req::D6 => Res::D6(D6::Four),
        Req::D6PassFail(_) | Req::Sum2D6PassFail(_) => Res::Pass,
        Req::D6ThreeOutcomes(..) | Req::Sum2D6ThreeOutcomes(..) => Res::Pass,
        Req::D8 => Res::D8(diag),
        Req::FoulArmor(_) => Res::FoulArmor {
            broken: false,
            ejected: false,
        },
        Req::FoulInjury(..) => Res::FoulInjury {
            outcome: InjuryOutcome::Stunned,
            ejected: false,
        },
        Req::Deviate => Res::Deviate(D6::Three, diag),
        Req::Scatter => Res::Scatter(diag, diag, diag),
        Req::Sum2D6 => Res::Sum2D6(Sum2D6::Seven),
        Req::ThrowIn => return None, // D3 indexes a sideline-dependent table
    })
}

/// The headline property. Every legal action of every sampled state, taken
/// on both sides of the mirror, must land in mirrored states.
#[test]
fn every_transition_is_mirror_invariant() {
    let dims = tier();
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (i, s0) in states(120, 23_020).into_iter().enumerate() {
        let m0 = mirror_playable(&s0, dims);
        assert_eq!(
            mirror_fingerprint(&s0, dims),
            fingerprint(&m0),
            "state {i}: the rebuilt mirror is not the mirror before any action"
        );
        for a in s0.get_all_actions() {
            // EndTurn hands the turn over and is not interesting here; the
            // per-action geometry is.
            if a == EngineAction::Simple(SimpleAT::EndTurn) {
                continue;
            }
            let mut s = s0.clone();
            let mut m = m0.clone();
            s.set_dice_mode(DiceMode::RegisterRolls);
            m.set_dice_mode(DiceMode::RegisterRolls);
            match step_pair(&mut s, &mut m, a) {
                Step::Skipped => {
                    skipped += 1;
                    continue;
                }
                Step::Diverged(why) => {
                    failures.push(format!("state {i} action {a:?}: {why}"));
                    continue;
                }
                Step::Mirrored => {}
            }
            checked += 1;
            let expected = mirror_fingerprint(&s, dims);
            let got = fingerprint(&m);
            if expected != got {
                failures.push(format!(
                    "state {i} action {a:?}: transition is not mirror-invariant\n  \
                     mirror(step(s,a)) = {expected}\n  step(mirror(s),mirror(a)) = {got}"
                ));
            }
        }
    }
    assert!(checked > 500, "expected a few thousand transitions, checked {checked}");
    assert!(
        failures.is_empty(),
        "{} of {} mirrored transitions diverged ({skipped} skipped). First 3:\n{}",
        failures.len(),
        checked,
        failures.iter().take(3).cloned().collect::<Vec<_>>().join("\n"),
    );
}

/// Regression test for the plan-023 defect this file found: **the
/// pathfinder's route choice used not to be mirror-invariant.**
///
/// `Move(dest)` does not name a route, only a destination. `pathing.rs`
/// picks one, and when two routes tie on everything `Node::is_better_than`
/// compares (probability, block dice, foul target, remaining movement,
/// cumulative distance) the winner is whichever was *inserted first* —
/// i.e. whichever came first in `expand_node`'s
/// `Direction::all_directions_iter()`. `ALL_DIRECTIONS` starts
/// `(1,1), (0,1), (-1,1), (1,0), (-1,0), …`: reflecting it in `x` gives a
/// permutation of itself, not itself, and every `dx = +1` entry precedes
/// its `dx = -1` partner. So on a tie the pathfinder steps toward **+x** —
/// Away's attacking direction, for both teams.
///
/// The destination is unaffected, so a risk-free route change is invisible.
/// It becomes visible the moment the route crosses a tackle zone: the two
/// mirrored players dodge from *different squares*, so they are exposed to
/// different opponents and, on a failure, fall in different places. That is
/// what a search sees, and it is the same "decision made in absolute board
/// coordinates" family as `ScriptedBot`'s touchback tie-break (a measured
/// 0.113 Home share) and the throw-in/bounce representatives.
///
/// It was measured at 1080 of 25121 risky routes (4.3%) before the fix and
/// 0 after: `expand_node` now takes its direction order from
/// `Direction::all_directions_toward`, oriented by the mover's own
/// attacking direction, so the tie-break mirrors with the board.
#[test]
fn movement_routes_are_mirror_invariant() {
    use botbowl_engine::core::table::PosAT;

    let dims = tier();
    let mut compared = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (i, s0) in states(150, 23_021).into_iter().enumerate() {
        let m0 = mirror_playable(&s0, dims);
        // Activate each player in turn, then walk to each reachable square
        // and compare at the first roll the route asks for.
        for start in s0.get_all_actions() {
            let EngineAction::Positional(PosAT::StartMove, _) = start else {
                continue;
            };
            let mut s = s0.clone();
            let mut m = m0.clone();
            s.set_dice_mode(DiceMode::RegisterRolls);
            m.set_dice_mode(DiceMode::RegisterRolls);
            s.step_with_roll_or_action(SomeProcInput::Action(start));
            m.step_with_roll_or_action(SomeProcInput::Action(mirror_action(dims, start)));
            for mv in s.get_all_actions() {
                let EngineAction::Positional(PosAT::Move, _) = mv else {
                    continue;
                };
                let mut sm = s.clone();
                let mut mm = m.clone();
                sm.step_with_roll_or_action(SomeProcInput::Action(mv));
                mm.step_with_roll_or_action(SomeProcInput::Action(mirror_action(dims, mv)));
                // Only routes that ask for a die roll can differ observably;
                // a risk-free route change never leaves the destination.
                if sm.pending_roll.is_none() && mm.pending_roll.is_none() {
                    continue;
                }
                compared += 1;
                let want = mirror_fingerprint(&sm, dims);
                let got = fingerprint(&mm);
                if want != got {
                    failures.push(format!("state {i} {start:?} then {mv:?}:\n  want {want}\n  got  {got}"));
                }
            }
        }
    }
    assert!(compared > 100, "expected many risky routes, compared {compared}");
    assert!(
        failures.is_empty(),
        "{} of {compared} risky movement routes are not mirror-invariant. First:\n{}",
        failures.len(),
        failures.first().cloned().unwrap_or_default(),
    );
}

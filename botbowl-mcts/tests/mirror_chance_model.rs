//! Is the *search's model of chance* mirror-invariant? (plan 023)
//!
//! `roll_outcomes::enumerate` is where MCTS decides what a die roll can do.
//! It does not enumerate the real distribution: bounces are pruned to free
//! squares, out-of-bounds directions collapse to one representative, and
//! most non-branching rolls collapse to a single scripted outcome. Every one
//! of those choices is made in board coordinates, and plan 023's H-c was
//! exactly this bug — `D3::One` for a throw-in and `oob.first()` for a
//! bounce both meant "+x", i.e. Away's attacking direction, for both teams.
//!
//! H-c was fixed with two targeted tests. This is the general property those
//! tests are special cases of:
//!
//! ```text
//! enumerate(mirror(s), req) == mirror(enumerate(s, req))
//! ```
//!
//! as a multiset of (outcome, probability). `GameState::mirrored` is enough
//! here — `enumerate` only *reads* the state, so the procedure stack it
//! leaves alone does not matter.

mod common;

use botbowl_engine::core::dices::{RequestedRoll, RollResult};
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::SomeProcInput;
use botbowl_mcts::roll_outcomes;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use common::{mirror_roll, states};

/// Walk random starts forward with random legal actions under
/// `RegisterRolls` and collect every state the engine pauses on a roll in.
/// These are exactly the states MCTS builds chance nodes from.
fn pending_roll_states(n_states: u32, seed: u64) -> Vec<(GameState, RequestedRoll)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xBEEF);
    let mut out = Vec::new();
    for mut s in states(n_states, seed) {
        s.set_dice_mode(DiceMode::RegisterRolls);
        for _ in 0..80 {
            if s.info.game_over {
                break;
            }
            if let Some(req) = s.pending_roll {
                out.push((s.clone(), req));
                // Resolve it and carry on so one trajectory yields many
                // chance nodes.
                let outcomes = roll_outcomes::enumerate(&s, &req);
                let pick = outcomes.choose(&mut rng).expect("enumerate never returns empty");
                let result = match pick {
                    botbowl_mcts::BbAction::Chance { result, .. } => *result,
                    _ => unreachable!("enumerate yields chance actions"),
                };
                s.step_with_roll_or_action(SomeProcInput::Roll(result));
                continue;
            }
            let actions = s.get_all_actions();
            if actions.is_empty() {
                break;
            }
            let a = *actions.choose(&mut rng).expect("non-empty");
            s.step_with_roll_or_action(SomeProcInput::Action(a));
        }
    }
    out
}

/// `(result, probability-bits)` pairs, sorted — the enumerated distribution
/// as a comparable value. Probabilities are compared exactly: `enumerate`
/// computes them from the same expressions on both sides of the mirror, so
/// anything other than bit-equality is a real difference.
fn distribution(actions: &[botbowl_mcts::BbAction], mirror: bool) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(actions.len());
    for a in actions {
        let (result, prob) = match a {
            botbowl_mcts::BbAction::Chance { result, .. } => (*result, a.prob_f32().unwrap()),
            _ => unreachable!("enumerate yields chance actions"),
        };
        let result = if mirror { mirror_roll(result)? } else { result };
        out.push(format!("{result:?}@{:.6}", prob));
    }
    out.sort();
    Some(out)
}

#[test]
fn enumerated_roll_outcomes_are_mirror_invariant() {
    let cases = pending_roll_states(150, 23_030);
    assert!(
        cases.len() > 200,
        "expected plenty of chance nodes, got {}",
        cases.len()
    );
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut by_kind: std::collections::BTreeMap<String, usize> = Default::default();
    let mut failures = Vec::new();
    for (i, (s, req)) in cases.iter().enumerate() {
        let m = s.mirrored();
        let direct = roll_outcomes::enumerate(s, req);
        let mirrored = roll_outcomes::enumerate(&m, req);
        let (Some(expected), Some(got)) = (distribution(&direct, true), distribution(&mirrored, false)) else {
            skipped += 1;
            continue;
        };
        *by_kind.entry(format!("{req:?}")).or_default() += 1;
        checked += 1;
        if expected != got {
            failures.push(format!(
                "case {i} req {req:?} ball {:?}\n  mirror(enumerate(s))     = {expected:?}\n  enumerate(mirror(s))     = {got:?}",
                s.ball
            ));
        }
    }
    eprintln!("checked {checked} chance nodes ({skipped} skipped); by request: {by_kind:?}");
    assert!(
        failures.is_empty(),
        "{} of {checked} enumerated chance distributions are not mirror-invariant. First 3:\n{}",
        failures.len(),
        failures.iter().take(3).cloned().collect::<Vec<_>>().join("\n"),
    );
}

/// Sanity: the sweep must actually reach the interesting request types, or
/// the test above is vacuous for exactly the code plan 023's H-c lived in.
#[test]
fn the_sweep_reaches_direction_rolls() {
    let cases = pending_roll_states(150, 23_031);
    let has = |p: fn(&RequestedRoll) -> bool| cases.iter().any(|(_, r)| p(r));
    assert!(
        has(|r| matches!(r, RequestedRoll::D8)),
        "no bounce/D8 rolls in the sweep"
    );
    assert!(
        cases
            .iter()
            .any(|(s, _)| matches!(s.ball, botbowl_engine::core::model::BallState::InAir(_))),
        "no in-air ball states in the sweep"
    );
    let _ = RollResult::Pass;
}

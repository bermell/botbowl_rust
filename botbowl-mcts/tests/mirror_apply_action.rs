//! Is the *search's own* state transition mirror-invariant? (plan 023)
//!
//! `mirror_transitions.rs` tests the raw engine. But MCTS does not step the
//! engine directly: `BloodBowlDynamics::apply_action` wraps it in a
//! quiescent loop that also applies `scripted::scripted_player_pick` (block
//! dice, coin toss, kick/receive) and `sole_legal_action` (which consults
//! the pruning rules), and its chance actions come from
//! `roll_outcomes::enumerate`. Every one of those is a place a
//! board-coordinate decision could hide.
//!
//! This walks a mirrored pair of states down the search's own transition
//! function in lockstep — enumerate, pick, mirror the pick, apply to both,
//! compare — for the full depth of a search horizon. It is the last
//! unproperty-tested link between "the engine mirrors" and "the search's
//! root value does not".

mod common;

use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::TeamType;
use botbowl_mcts::dynamics::HorizonAnchor;
use botbowl_mcts::{BbAction, BbPlayer, BloodBowlDynamics, Evaluator, PuctMode};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use recon_mcts::GameDynamics;

use common::{fingerprint, mirror_action, mirror_fingerprint, mirror_playable, mirror_roll, states, tier};

fn mover(s: &GameState) -> TeamType {
    s.available_actions.team.unwrap_or(s.info.team_turn)
}

fn dynamics(s: &GameState) -> BloodBowlDynamics {
    BloodBowlDynamics {
        horizon: Some(HorizonAnchor::capture(s, mover(s))),
        virtual_loss: 0,
        evaluator: Evaluator::Heuristic,
        puct: PuctMode::default(),
        ..Default::default()
    }
}

fn mirror_bb_action(dims: botbowl_engine::core::model::BoardDims, a: &BbAction) -> Option<BbAction> {
    Some(match a {
        BbAction::Player { action, prior_bits } => BbAction::Player {
            action: mirror_action(dims, *action),
            prior_bits: *prior_bits,
        },
        BbAction::Chance { result, prob_bits } => BbAction::Chance {
            result: mirror_roll(*result)?,
            prob_bits: *prob_bits,
        },
    })
}

fn player_of(s: &GameState) -> BbPlayer {
    if s.pending_roll.is_some() {
        return BbPlayer::Chance;
    }
    match s.available_actions.team {
        Some(TeamType::Home) => BbPlayer::Home,
        Some(TeamType::Away) => BbPlayer::Away,
        None => BbPlayer::Chance,
    }
}

#[test]
fn search_transitions_are_mirror_invariant() {
    let dims = tier();
    let mut rng = ChaCha8Rng::seed_from_u64(23_040);
    let mut steps = 0usize;
    let mut walks = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // `BB_WALK_N` widens the sweep for a hunt; the committed default keeps
    // the test fast.
    let n: u32 = std::env::var("BB_WALK_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    for (i, s0) in states(n, 23_040).into_iter().enumerate() {
        let mut s = s0.clone();
        let mut m = mirror_playable(&s0, dims);
        s.set_dice_mode(DiceMode::RegisterRolls);
        m.set_dice_mode(DiceMode::RegisterRolls);
        let gd_s = dynamics(&s);
        let gd_m = dynamics(&m);
        assert_eq!(
            mirror_fingerprint(&s, dims),
            fingerprint(&m),
            "state {i}: pair is not mirrored before the walk"
        );
        walks += 1;

        // Walk to the depth a real search reaches (the horizon ends the
        // action list), comparing at every step.
        for depth in 0..40 {
            let acts = match gd_s.available_actions(&player_of(&s), &s) {
                Some(a) if !a.is_empty() => a,
                _ => break,
            };
            let acts_m = match gd_m.available_actions(&player_of(&m), &m) {
                Some(a) if !a.is_empty() => a,
                _ => {
                    failures.push(format!("state {i} depth {depth}: mirror ran out of actions first"));
                    break;
                }
            };
            // The enumerated sets themselves must mirror.
            let mut want: Vec<String> = acts
                .iter()
                .filter_map(|a| mirror_bb_action(dims, a).map(|b| format!("{b:?}")))
                .collect();
            let mut got: Vec<String> = acts_m.iter().map(|a| format!("{a:?}")).collect();
            if want.len() == got.len() {
                want.sort();
                got.sort();
                if want != got {
                    failures.push(format!(
                        "state {i} depth {depth}: enumerated action sets do not mirror\n  want {want:?}\n  got  {got:?}"
                    ));
                    break;
                }
            }

            let a = acts.choose(&mut rng).expect("non-empty");
            let Some(am) = mirror_bb_action(dims, a) else { break };
            let Some(next_s) = gd_s.apply_action(s.clone(), a) else {
                break;
            };
            let Some(next_m) = gd_m.apply_action(m.clone(), &am) else {
                failures.push(format!("state {i} depth {depth}: mirror rejected {am:?}"));
                break;
            };
            steps += 1;
            let want = mirror_fingerprint(&next_s, dims);
            let got = fingerprint(&next_m);
            if want != got {
                failures.push(format!(
                    "state {i} depth {depth} action {a:?}: search transition is not mirror-invariant\n  \
                     mirror(apply(s,a)) = {want}\n  apply(mirror(s),mirror(a)) = {got}"
                ));
                break;
            }
            s = next_s;
            m = next_m;
        }
    }

    assert!(
        steps > 500,
        "expected a deep sweep, took only {steps} steps over {walks} walks"
    );
    if !failures.is_empty() {
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for f in &failures {
            let kind = f
                .split(" action ")
                .nth(1)
                .map(|t| t.split_whitespace().take(2).collect::<Vec<_>>().join(" "))
                .unwrap_or_else(|| "<no action>".into());
            *kinds.entry(kind).or_default() += 1;
        }
        eprintln!("mismatch actions: {kinds:?}");
    }
    assert!(
        failures.is_empty(),
        "{} mismatches over {steps} mirrored search transitions ({walks} walks). First 2:\n{}",
        failures.len(),
        failures.iter().take(2).cloned().collect::<Vec<_>>().join("\n\n"),
    );
}

/// Plan 035 T3: `apply_action` is a pure, deterministic function of
/// `(state, action)`.
///
/// This is the one assumption behind plan 035's equivalence argument. The old
/// mover tag was `player_for_state(apply_action(parent_state, a))` computed
/// eagerly at enumeration time (`peek_mover`); the new one is
/// `player_for_state(child_state)` where `child_state` is what descent
/// computed via the *same* `apply_action`. Those are the same value iff
/// `apply_action` returns the same state every time it is called on equal
/// inputs. Recombination already requires this (two paths reaching the same
/// logical state must produce equal `GameState`s or the DAG silently splits),
/// so a red here is a pre-existing bug of that class, not a plan-035
/// regression.
///
/// Applies each legal action twice from independent clones and compares with
/// `GameState`'s own `PartialEq` — the same equality the transposition table
/// uses under `MemoryMode::StoreState` — plus the derived mover tag.
#[test]
fn apply_action_is_pure_and_deterministic() {
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (i, s0) in states(200, 35_030).into_iter().enumerate() {
        let mut s = s0.clone();
        s.set_dice_mode(DiceMode::RegisterRolls);
        s.set_logging_state(false);
        s.clear_log();
        let gd = dynamics(&s);
        let Some(acts) = gd.available_actions(&player_of(&s), &s) else {
            continue;
        };
        for a in acts.iter() {
            let first = gd.apply_action(s.clone(), a);
            let second = gd.apply_action(s.clone(), a);
            checked += 1;
            match (first, second) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    if x != y {
                        failures.push(format!("state {i} action {a:?}: two applies gave different states"));
                    } else if player_of(&x) != player_of(&y) {
                        failures.push(format!(
                            "state {i} action {a:?}: mover tag is not a function of the state"
                        ));
                    }
                }
                (f, sec) => failures.push(format!(
                    "state {i} action {a:?}: legality is not deterministic ({} vs {})",
                    f.is_some(),
                    sec.is_some()
                )),
            }
        }
    }

    assert!(checked > 500, "expected a broad sweep, checked only {checked} applies");
    assert!(
        failures.is_empty(),
        "{} impure applies over {checked} checked. First 3:\n{}",
        failures.len(),
        failures.iter().take(3).cloned().collect::<Vec<_>>().join("\n"),
    );
}

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

    for (i, s0) in states(150, 23_040).into_iter().enumerate() {
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
                .filter_map(|(_, a)| mirror_bb_action(dims, a).map(|b| format!("{b:?}")))
                .collect();
            let mut got: Vec<String> = acts_m.iter().map(|(_, a)| format!("{a:?}")).collect();
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

            let (_, a) = acts.choose(&mut rng).expect("non-empty");
            let Some(am) = mirror_bb_action(dims, a) else { break };
            let Some(next_s) = gd_s.apply_action(s.clone(), a) else { break };
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

    assert!(steps > 500, "expected a deep sweep, took only {steps} steps over {walks} walks");
    if !failures.is_empty() {
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for f in &failures {
            let kind = f.split(" action ").nth(1).map(|t| {
                t.split_whitespace().take(2).collect::<Vec<_>>().join(" ")
            }).unwrap_or_else(|| "<no action>".into());
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

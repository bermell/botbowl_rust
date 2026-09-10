//! Turning a resolved `(RequestedRoll, RollResult)` pair into something the
//! dice ticker can show.
//!
//! This exists only because the server rolls the dice itself
//! (`DiceMode::RegisterRolls`, decision 4). In `RollDice` mode the engine
//! resolves rolls internally and a UI never sees them.
//!
//! Note the deliberate gap: `RollResult::Pass`/`Fail` carry **no** die face —
//! the engine collapses pass/fail rolls to their outcome — so those events
//! show the target and the verdict but no pips. That is the engine's
//! representation, not a shortcut here.

use botbowl_engine::core::dices as ed;
use botbowl_web_proto::dice as pd;

use crate::mirror;

fn d6_face(v: u8) -> pd::DieFace {
    pd::DieFace {
        img: format!("dice/{v}.png"),
        label: format!("D6 {v}"),
    }
}

fn d8_face(v: u8) -> pd::DieFace {
    pd::DieFace {
        img: format!("dice/d8-{v}.png"),
        label: format!("D8 {v}"),
    }
}

fn block_face(f: pd::BlockDice) -> pd::DieFace {
    pd::DieFace {
        img: f.img().to_string(),
        label: f.label().to_string(),
    }
}

fn pass_fail_word(result: &pd::RollResult) -> &'static str {
    match result {
        pd::RollResult::Pass => "success",
        pd::RollResult::Fail => "failure",
        pd::RollResult::MiddleOutcome => "partial",
        _ => "?",
    }
}

/// Build the streamed event. `fixed` marks a value that came from the debug
/// "fix next roll" control rather than the session RNG.
pub fn event(requested: ed::RequestedRoll, result: ed::RollResult, fixed: bool) -> pd::DiceEvent {
    let requested = mirror::requested_roll_to_proto(requested);
    let result = mirror::roll_result_to_proto(result);
    let (text, faces) = render(&requested, &result);
    pd::DiceEvent {
        requested,
        result,
        text,
        faces,
        fixed,
    }
}

fn render(requested: &pd::RequestedRoll, result: &pd::RollResult) -> (String, Vec<pd::DieFace>) {
    let label = requested.label();
    match result {
        pd::RollResult::Coin(c) => (format!("{label}: {c:?}"), Vec::new()),
        pd::RollResult::Pass | pd::RollResult::Fail | pd::RollResult::MiddleOutcome => {
            (format!("{label} — {}", pass_fail_word(result)), Vec::new())
        }
        pd::RollResult::D6 { value } => (format!("{label}: {value}"), vec![d6_face(*value)]),
        pd::RollResult::D8 { value } => (format!("{label}: {value}"), vec![d8_face(*value)]),
        pd::RollResult::Sum2D6 { value } => (format!("{label}: {value}"), Vec::new()),
        pd::RollResult::Deviate { distance, direction } => (
            format!("{label}: {distance} squares, direction {direction}"),
            vec![d6_face(*distance), d8_face(*direction)],
        ),
        pd::RollResult::Scatter { first, second, third } => (
            format!("{label}: {first}, {second}, {third}"),
            vec![d8_face(*first), d8_face(*second), d8_face(*third)],
        ),
        pd::RollResult::ThrowIn { direction, distance } => (
            format!("{label}: direction {direction}, {distance} squares"),
            Vec::new(),
        ),
        pd::RollResult::BlockDice { faces } => {
            let names: Vec<&str> = faces.iter().map(|f| f.label()).collect();
            (
                format!("{label}: {}", names.join(", ")),
                faces.iter().map(|f| block_face(*f)).collect(),
            )
        }
        pd::RollResult::FoulArmor { broken, ejected } => {
            let mut text = format!("{label}: {}", if *broken { "broken" } else { "held" });
            if *ejected {
                text.push_str(" — sent off!");
            }
            (text, Vec::new())
        }
        pd::RollResult::FoulInjury { outcome, ejected } => {
            let mut text = format!("{label}: {outcome:?}");
            if *ejected {
                text.push_str(" — sent off!");
            }
            (text, Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Every roll the engine can request must render without panicking and
    /// with a non-empty ticker line — the ticker is the only place a human
    /// sees the dice, so a blank entry is a silent hole.
    #[test]
    fn every_roll_request_renders() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let requests = [
            ed::RequestedRoll::Coin,
            ed::RequestedRoll::D6,
            ed::RequestedRoll::D8,
            ed::RequestedRoll::Deviate,
            ed::RequestedRoll::Scatter,
            ed::RequestedRoll::Sum2D6,
            ed::RequestedRoll::ThrowIn,
            ed::RequestedRoll::D6PassFail(ed::D6Target::FourPlus),
            ed::RequestedRoll::D6ThreeOutcomes(ed::D6Target::TwoPlus, ed::D6Target::FivePlus),
            ed::RequestedRoll::Sum2D6PassFail(ed::Sum2D6Target::SevenPlus),
            ed::RequestedRoll::Sum2D6ThreeOutcomes(ed::Sum2D6Target::ThreePlus, ed::Sum2D6Target::TenPlus),
            ed::RequestedRoll::FoulArmor(ed::Sum2D6Target::NinePlus),
            ed::RequestedRoll::FoulInjury(ed::Sum2D6Target::EightPlus, ed::Sum2D6Target::TenPlus),
            ed::RequestedRoll::BlockDice(botbowl_engine::core::table::NumBlockDices::Two),
            ed::RequestedRoll::BlockDice(botbowl_engine::core::table::NumBlockDices::ThreeUphill),
        ];
        for request in requests {
            // A few draws each, so both branches of a pass/fail are covered.
            for _ in 0..20 {
                let result = ed::resolve_with_rng(request, &mut rng);
                let e = event(request, result, false);
                assert!(!e.text.is_empty(), "{request:?} rendered an empty ticker line");
                assert!(
                    !e.text.contains("?"),
                    "{request:?} -> {result:?} rendered as {:?}",
                    e.text
                );
            }
        }
    }

    #[test]
    fn block_dice_faces_match_the_rolled_count() {
        use botbowl_engine::core::table::NumBlockDices;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        for n in [
            NumBlockDices::One,
            NumBlockDices::Two,
            NumBlockDices::Three,
            NumBlockDices::TwoUphill,
            NumBlockDices::ThreeUphill,
        ] {
            let request = ed::RequestedRoll::BlockDice(n);
            let result = ed::resolve_with_rng(request, &mut rng);
            let e = event(request, result, false);
            assert_eq!(
                e.faces.len(),
                mirror::num_block_dices_to_proto(n).count(),
                "{n:?} produced {} faces",
                e.faces.len()
            );
        }
    }
}

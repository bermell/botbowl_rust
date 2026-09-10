//! Exhaustive conversions between the engine's types and the `proto` mirrors.
//!
//! `botbowl-web-proto` re-declares the engine's action and dice enums so the
//! wasm client can compile without the engine (decision 3). This module is the
//! price and the safety net: every match here is **exhaustive with no wildcard
//! arm**, so adding or renaming an engine variant is a compile error right
//! here, and [`tests`] round-trips every variant of every enum.

use botbowl_engine::core::dices as ed;
use botbowl_engine::core::model as em;
use botbowl_engine::core::table as et;
use botbowl_web_proto::action as pa;
use botbowl_web_proto::dice as pd;
use botbowl_web_proto::view as pv;

// ---------------------------------------------------------------- primitives

pub fn position_to_proto(p: em::Position) -> pa::Position {
    pa::Position { x: p.x, y: p.y }
}

pub fn position_from_proto(p: pa::Position) -> em::Position {
    em::Position::new((p.x, p.y))
}

pub fn team_to_proto(t: em::TeamType) -> pa::TeamType {
    match t {
        em::TeamType::Home => pa::TeamType::Home,
        em::TeamType::Away => pa::TeamType::Away,
    }
}

pub fn team_from_proto(t: pa::TeamType) -> em::TeamType {
    match t {
        pa::TeamType::Home => em::TeamType::Home,
        pa::TeamType::Away => em::TeamType::Away,
    }
}

// ------------------------------------------------------------------- actions

pub fn pos_at_to_proto(at: et::PosAT) -> pa::PosAT {
    match at {
        et::PosAT::StartMove => pa::PosAT::StartMove,
        et::PosAT::StartBlitz => pa::PosAT::StartBlitz,
        et::PosAT::StartPass => pa::PosAT::StartPass,
        et::PosAT::StartFoul => pa::PosAT::StartFoul,
        et::PosAT::SelectPosition => pa::PosAT::SelectPosition,
        et::PosAT::Push => pa::PosAT::Push,
        et::PosAT::FollowUp => pa::PosAT::FollowUp,
        et::PosAT::StartHandoff => pa::PosAT::StartHandoff,
        et::PosAT::Handoff => pa::PosAT::Handoff,
        et::PosAT::Pass => pa::PosAT::Pass,
        et::PosAT::Move => pa::PosAT::Move,
        et::PosAT::Foul => pa::PosAT::Foul,
        et::PosAT::StartBlock => pa::PosAT::StartBlock,
        et::PosAT::Block => pa::PosAT::Block,
    }
}

pub fn pos_at_from_proto(at: pa::PosAT) -> et::PosAT {
    match at {
        pa::PosAT::StartMove => et::PosAT::StartMove,
        pa::PosAT::StartBlitz => et::PosAT::StartBlitz,
        pa::PosAT::StartPass => et::PosAT::StartPass,
        pa::PosAT::StartFoul => et::PosAT::StartFoul,
        pa::PosAT::SelectPosition => et::PosAT::SelectPosition,
        pa::PosAT::Push => et::PosAT::Push,
        pa::PosAT::FollowUp => et::PosAT::FollowUp,
        pa::PosAT::StartHandoff => et::PosAT::StartHandoff,
        pa::PosAT::Handoff => et::PosAT::Handoff,
        pa::PosAT::Pass => et::PosAT::Pass,
        pa::PosAT::Move => et::PosAT::Move,
        pa::PosAT::Foul => et::PosAT::Foul,
        pa::PosAT::StartBlock => et::PosAT::StartBlock,
        pa::PosAT::Block => et::PosAT::Block,
    }
}

pub fn simple_at_to_proto(at: et::SimpleAT) -> pa::SimpleAT {
    match at {
        et::SimpleAT::SelectBothDown => pa::SimpleAT::SelectBothDown,
        et::SimpleAT::SelectPow => pa::SimpleAT::SelectPow,
        et::SimpleAT::SelectPush => pa::SimpleAT::SelectPush,
        et::SimpleAT::SelectPowPush => pa::SimpleAT::SelectPowPush,
        et::SimpleAT::SelectSkull => pa::SimpleAT::SelectSkull,
        et::SimpleAT::UseReroll => pa::SimpleAT::UseReroll,
        et::SimpleAT::DontUseReroll => pa::SimpleAT::DontUseReroll,
        et::SimpleAT::EndPlayerTurn => pa::SimpleAT::EndPlayerTurn,
        et::SimpleAT::EndTurn => pa::SimpleAT::EndTurn,
        et::SimpleAT::Heads => pa::SimpleAT::Heads,
        et::SimpleAT::Tails => pa::SimpleAT::Tails,
        et::SimpleAT::Kick => pa::SimpleAT::Kick,
        et::SimpleAT::Receive => pa::SimpleAT::Receive,
        et::SimpleAT::SetupLine => pa::SimpleAT::SetupLine,
        et::SimpleAT::EndSetup => pa::SimpleAT::EndSetup,
        et::SimpleAT::KickoffAimMiddle => pa::SimpleAT::KickoffAimMiddle,
    }
}

pub fn simple_at_from_proto(at: pa::SimpleAT) -> et::SimpleAT {
    match at {
        pa::SimpleAT::SelectBothDown => et::SimpleAT::SelectBothDown,
        pa::SimpleAT::SelectPow => et::SimpleAT::SelectPow,
        pa::SimpleAT::SelectPush => et::SimpleAT::SelectPush,
        pa::SimpleAT::SelectPowPush => et::SimpleAT::SelectPowPush,
        pa::SimpleAT::SelectSkull => et::SimpleAT::SelectSkull,
        pa::SimpleAT::UseReroll => et::SimpleAT::UseReroll,
        pa::SimpleAT::DontUseReroll => et::SimpleAT::DontUseReroll,
        pa::SimpleAT::EndPlayerTurn => et::SimpleAT::EndPlayerTurn,
        pa::SimpleAT::EndTurn => et::SimpleAT::EndTurn,
        pa::SimpleAT::Heads => et::SimpleAT::Heads,
        pa::SimpleAT::Tails => et::SimpleAT::Tails,
        pa::SimpleAT::Kick => et::SimpleAT::Kick,
        pa::SimpleAT::Receive => et::SimpleAT::Receive,
        pa::SimpleAT::SetupLine => et::SimpleAT::SetupLine,
        pa::SimpleAT::EndSetup => et::SimpleAT::EndSetup,
        pa::SimpleAT::KickoffAimMiddle => et::SimpleAT::KickoffAimMiddle,
    }
}

pub fn action_to_proto(a: em::Action) -> pa::Action {
    match a {
        em::Action::Positional(at, pos) => pa::Action::Positional(pos_at_to_proto(at), position_to_proto(pos)),
        em::Action::Simple(at) => pa::Action::Simple(simple_at_to_proto(at)),
    }
}

pub fn action_from_proto(a: pa::Action) -> em::Action {
    match a {
        pa::Action::Positional(at, pos) => em::Action::Positional(pos_at_from_proto(at), position_from_proto(pos)),
        pa::Action::Simple(at) => em::Action::Simple(simple_at_from_proto(at)),
    }
}

// ---------------------------------------------------------------------- dice

pub fn coin_to_proto(c: ed::Coin) -> pd::Coin {
    match c {
        ed::Coin::Heads => pd::Coin::Heads,
        ed::Coin::Tails => pd::Coin::Tails,
    }
}

pub fn coin_from_proto(c: pd::Coin) -> ed::Coin {
    match c {
        pd::Coin::Heads => ed::Coin::Heads,
        pd::Coin::Tails => ed::Coin::Tails,
    }
}

pub fn block_dice_to_proto(b: ed::BlockDice) -> pd::BlockDice {
    match b {
        ed::BlockDice::Skull => pd::BlockDice::Skull,
        ed::BlockDice::BothDown => pd::BlockDice::BothDown,
        ed::BlockDice::Push => pd::BlockDice::Push,
        ed::BlockDice::PowPush => pd::BlockDice::PowPush,
        ed::BlockDice::Pow => pd::BlockDice::Pow,
    }
}

pub fn block_dice_from_proto(b: pd::BlockDice) -> ed::BlockDice {
    match b {
        pd::BlockDice::Skull => ed::BlockDice::Skull,
        pd::BlockDice::BothDown => ed::BlockDice::BothDown,
        pd::BlockDice::Push => ed::BlockDice::Push,
        pd::BlockDice::PowPush => ed::BlockDice::PowPush,
        pd::BlockDice::Pow => ed::BlockDice::Pow,
    }
}

pub fn num_block_dices_to_proto(n: et::NumBlockDices) -> pd::NumBlockDices {
    match n {
        et::NumBlockDices::ThreeUphill => pd::NumBlockDices::ThreeUphill,
        et::NumBlockDices::TwoUphill => pd::NumBlockDices::TwoUphill,
        et::NumBlockDices::One => pd::NumBlockDices::One,
        et::NumBlockDices::Two => pd::NumBlockDices::Two,
        et::NumBlockDices::Three => pd::NumBlockDices::Three,
    }
}

pub fn num_block_dices_from_proto(n: pd::NumBlockDices) -> et::NumBlockDices {
    match n {
        pd::NumBlockDices::ThreeUphill => et::NumBlockDices::ThreeUphill,
        pd::NumBlockDices::TwoUphill => et::NumBlockDices::TwoUphill,
        pd::NumBlockDices::One => et::NumBlockDices::One,
        pd::NumBlockDices::Two => et::NumBlockDices::Two,
        pd::NumBlockDices::Three => et::NumBlockDices::Three,
    }
}

pub fn injury_to_proto(o: em::InjuryOutcome) -> pd::InjuryOutcome {
    match o {
        em::InjuryOutcome::Stunned => pd::InjuryOutcome::Stunned,
        em::InjuryOutcome::KO => pd::InjuryOutcome::KO,
        em::InjuryOutcome::Casualty => pd::InjuryOutcome::Casualty,
    }
}

pub fn injury_from_proto(o: pd::InjuryOutcome) -> em::InjuryOutcome {
    match o {
        pd::InjuryOutcome::Stunned => em::InjuryOutcome::Stunned,
        pd::InjuryOutcome::KO => em::InjuryOutcome::KO,
        pd::InjuryOutcome::Casualty => em::InjuryOutcome::Casualty,
    }
}

/// The engine's numeric dice and targets are `#[repr(u8)]` with `TryFrom<u8>`,
/// so the wire carries the face value and this is where it is validated.
fn d6(v: u8) -> Result<ed::D6, String> {
    ed::D6::try_from(v).map_err(|_| format!("{v} is not a D6 face"))
}
fn d8(v: u8) -> Result<ed::D8, String> {
    ed::D8::try_from(v).map_err(|_| format!("{v} is not a D8 face"))
}
fn d3(v: u8) -> Result<ed::D3, String> {
    ed::D3::try_from(v).map_err(|_| format!("{v} is not a D3 face"))
}
fn sum2d6(v: u8) -> Result<ed::Sum2D6, String> {
    ed::Sum2D6::try_from(v).map_err(|_| format!("{v} is not a 2D6 sum"))
}
fn d6_target(v: u8) -> Result<ed::D6Target, String> {
    ed::D6Target::try_from(v).map_err(|_| format!("{v} is not a D6 target"))
}
fn sum2d6_target(v: u8) -> Result<ed::Sum2D6Target, String> {
    ed::Sum2D6Target::try_from(v).map_err(|_| format!("{v} is not a 2D6 target"))
}

pub fn requested_roll_to_proto(r: ed::RequestedRoll) -> pd::RequestedRoll {
    match r {
        ed::RequestedRoll::BlockDice(n) => pd::RequestedRoll::BlockDice {
            count: num_block_dices_to_proto(n),
        },
        ed::RequestedRoll::Coin => pd::RequestedRoll::Coin,
        ed::RequestedRoll::D6 => pd::RequestedRoll::D6,
        ed::RequestedRoll::D6PassFail(t) => pd::RequestedRoll::D6PassFail { target: t as u8 },
        ed::RequestedRoll::D6ThreeOutcomes(lo, hi) => pd::RequestedRoll::D6ThreeOutcomes {
            low: lo as u8,
            high: hi as u8,
        },
        ed::RequestedRoll::D8 => pd::RequestedRoll::D8,
        ed::RequestedRoll::FoulArmor(t) => pd::RequestedRoll::FoulArmor { target: t as u8 },
        ed::RequestedRoll::FoulInjury(ko, cas) => pd::RequestedRoll::FoulInjury {
            ko_target: ko as u8,
            cas_target: cas as u8,
        },
        ed::RequestedRoll::Deviate => pd::RequestedRoll::Deviate,
        ed::RequestedRoll::Scatter => pd::RequestedRoll::Scatter,
        ed::RequestedRoll::Sum2D6 => pd::RequestedRoll::Sum2D6,
        ed::RequestedRoll::Sum2D6PassFail(t) => pd::RequestedRoll::Sum2D6PassFail { target: t as u8 },
        ed::RequestedRoll::Sum2D6ThreeOutcomes(lo, hi) => pd::RequestedRoll::Sum2D6ThreeOutcomes {
            low: lo as u8,
            high: hi as u8,
        },
        ed::RequestedRoll::ThrowIn => pd::RequestedRoll::ThrowIn,
    }
}

pub fn requested_roll_from_proto(r: pd::RequestedRoll) -> Result<ed::RequestedRoll, String> {
    Ok(match r {
        pd::RequestedRoll::BlockDice { count } => ed::RequestedRoll::BlockDice(num_block_dices_from_proto(count)),
        pd::RequestedRoll::Coin => ed::RequestedRoll::Coin,
        pd::RequestedRoll::D6 => ed::RequestedRoll::D6,
        pd::RequestedRoll::D6PassFail { target } => ed::RequestedRoll::D6PassFail(d6_target(target)?),
        pd::RequestedRoll::D6ThreeOutcomes { low, high } => {
            ed::RequestedRoll::D6ThreeOutcomes(d6_target(low)?, d6_target(high)?)
        }
        pd::RequestedRoll::D8 => ed::RequestedRoll::D8,
        pd::RequestedRoll::FoulArmor { target } => ed::RequestedRoll::FoulArmor(sum2d6_target(target)?),
        pd::RequestedRoll::FoulInjury { ko_target, cas_target } => {
            ed::RequestedRoll::FoulInjury(sum2d6_target(ko_target)?, sum2d6_target(cas_target)?)
        }
        pd::RequestedRoll::Deviate => ed::RequestedRoll::Deviate,
        pd::RequestedRoll::Scatter => ed::RequestedRoll::Scatter,
        pd::RequestedRoll::Sum2D6 => ed::RequestedRoll::Sum2D6,
        pd::RequestedRoll::Sum2D6PassFail { target } => ed::RequestedRoll::Sum2D6PassFail(sum2d6_target(target)?),
        pd::RequestedRoll::Sum2D6ThreeOutcomes { low, high } => {
            ed::RequestedRoll::Sum2D6ThreeOutcomes(sum2d6_target(low)?, sum2d6_target(high)?)
        }
        pd::RequestedRoll::ThrowIn => ed::RequestedRoll::ThrowIn,
    })
}

pub fn roll_result_to_proto(r: ed::RollResult) -> pd::RollResult {
    match r {
        ed::RollResult::BlockDice(faces) => pd::RollResult::BlockDice {
            faces: faces.iter().flatten().map(|f| block_dice_to_proto(*f)).collect(),
        },
        ed::RollResult::Coin(c) => pd::RollResult::Coin(coin_to_proto(c)),
        ed::RollResult::Pass => pd::RollResult::Pass,
        ed::RollResult::Fail => pd::RollResult::Fail,
        ed::RollResult::FoulArmor { broken, ejected } => pd::RollResult::FoulArmor { broken, ejected },
        ed::RollResult::FoulInjury { outcome, ejected } => pd::RollResult::FoulInjury {
            outcome: injury_to_proto(outcome),
            ejected,
        },
        ed::RollResult::MiddleOutcome => pd::RollResult::MiddleOutcome,
        ed::RollResult::D6(v) => pd::RollResult::D6 { value: v as u8 },
        ed::RollResult::D8(v) => pd::RollResult::D8 { value: v as u8 },
        ed::RollResult::Deviate(dist, dir) => pd::RollResult::Deviate {
            distance: dist as u8,
            direction: dir as u8,
        },
        ed::RollResult::Scatter(a, b, c) => pd::RollResult::Scatter {
            first: a as u8,
            second: b as u8,
            third: c as u8,
        },
        ed::RollResult::Sum2D6(v) => pd::RollResult::Sum2D6 { value: v as u8 },
        ed::RollResult::ThrowIn { direction, distance } => pd::RollResult::ThrowIn {
            direction: direction as u8,
            distance: distance as u8,
        },
    }
}

pub fn roll_result_from_proto(r: &pd::RollResult) -> Result<ed::RollResult, String> {
    Ok(match r {
        pd::RollResult::BlockDice { faces } => {
            if faces.is_empty() || faces.len() > 3 {
                return Err(format!("block roll needs 1..=3 faces, got {}", faces.len()));
            }
            let mut slots: [Option<ed::BlockDice>; 3] = [None; 3];
            for (slot, face) in slots.iter_mut().zip(faces.iter()) {
                *slot = Some(block_dice_from_proto(*face));
            }
            ed::RollResult::BlockDice(slots)
        }
        pd::RollResult::Coin(c) => ed::RollResult::Coin(coin_from_proto(*c)),
        pd::RollResult::Pass => ed::RollResult::Pass,
        pd::RollResult::Fail => ed::RollResult::Fail,
        pd::RollResult::FoulArmor { broken, ejected } => ed::RollResult::FoulArmor {
            broken: *broken,
            ejected: *ejected,
        },
        pd::RollResult::FoulInjury { outcome, ejected } => ed::RollResult::FoulInjury {
            outcome: injury_from_proto(*outcome),
            ejected: *ejected,
        },
        pd::RollResult::MiddleOutcome => ed::RollResult::MiddleOutcome,
        pd::RollResult::D6 { value } => ed::RollResult::D6(d6(*value)?),
        pd::RollResult::D8 { value } => ed::RollResult::D8(d8(*value)?),
        pd::RollResult::Deviate { distance, direction } => ed::RollResult::Deviate(d6(*distance)?, d8(*direction)?),
        pd::RollResult::Scatter { first, second, third } => {
            ed::RollResult::Scatter(d8(*first)?, d8(*second)?, d8(*third)?)
        }
        pd::RollResult::Sum2D6 { value } => ed::RollResult::Sum2D6(sum2d6(*value)?),
        pd::RollResult::ThrowIn { direction, distance } => ed::RollResult::ThrowIn {
            direction: d3(*direction)?,
            distance: sum2d6(*distance)?,
        },
    })
}

// ------------------------------------------------------------- board flavour

pub fn role_to_proto(r: et::PlayerRole) -> pv::PlayerRole {
    match r {
        et::PlayerRole::Lineman => pv::PlayerRole::Lineman,
        et::PlayerRole::Blitzer => pv::PlayerRole::Blitzer,
        et::PlayerRole::Thrower => pv::PlayerRole::Thrower,
        et::PlayerRole::Catcher => pv::PlayerRole::Catcher,
    }
}

pub fn status_to_proto(s: em::PlayerStatus) -> pv::PlayerStatus {
    match s {
        em::PlayerStatus::Up => pv::PlayerStatus::Up,
        em::PlayerStatus::Down => pv::PlayerStatus::Down,
        em::PlayerStatus::Stunned => pv::PlayerStatus::Stunned,
    }
}

pub fn dugout_place_to_proto(p: em::DugoutPlace) -> pv::DugoutPlace {
    match p {
        em::DugoutPlace::Reserves => pv::DugoutPlace::Reserves,
        em::DugoutPlace::Heated => pv::DugoutPlace::Heated,
        em::DugoutPlace::KnockOut => pv::DugoutPlace::KnockOut,
        em::DugoutPlace::Injuried => pv::DugoutPlace::Injured,
        em::DugoutPlace::Ejected => pv::DugoutPlace::Ejected,
    }
}

pub fn weather_to_proto(w: &em::Weather) -> pv::Weather {
    match w {
        em::Weather::Nice => pv::Weather::Nice,
        em::Weather::Sunny => pv::Weather::Sunny,
        em::Weather::Rain => pv::Weather::Rain,
        em::Weather::Blizzard => pv::Weather::Blizzard,
        em::Weather::Sweltering => pv::Weather::Sweltering,
    }
}

pub fn skill_label(s: et::Skill) -> &'static str {
    match s {
        et::Skill::Dodge => "Dodge",
        et::Skill::Throw => "Throw",
        et::Skill::Block => "Block",
        et::Skill::Catch => "Catch",
        et::Skill::SureHands => "Sure Hands",
        et::Skill::SureFeet => "Sure Feet",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mirrored enum round-trips. Combined with the exhaustive matches
    /// above (no wildcard arms), this is what pins `proto` to the engine.
    #[test]
    fn action_types_round_trip() {
        for at in pa::PosAT::ALL {
            assert_eq!(pos_at_to_proto(pos_at_from_proto(at)), at);
        }
        for at in pa::SimpleAT::ALL {
            assert_eq!(simple_at_to_proto(simple_at_from_proto(at)), at);
        }
        // ...and the engine side is covered by the same identity, because the
        // maps are bijections between two 14- and 16-variant enums.
        assert_eq!(pa::PosAT::ALL.len(), 14);
        assert_eq!(pa::SimpleAT::ALL.len(), 16);
    }

    #[test]
    fn actions_round_trip() {
        let pos = pa::Position::new(3, 5);
        for at in pa::PosAT::ALL {
            let a = pa::Action::Positional(at, pos);
            assert_eq!(action_to_proto(action_from_proto(a)), a);
        }
        for at in pa::SimpleAT::ALL {
            let a = pa::Action::Simple(at);
            assert_eq!(action_to_proto(action_from_proto(a)), a);
        }
    }

    #[test]
    fn positions_round_trip() {
        for x in 0..8i8 {
            for y in 0..8i8 {
                let p = pa::Position::new(x, y);
                assert_eq!(position_to_proto(position_from_proto(p)), p);
            }
        }
    }

    fn all_requested_rolls() -> Vec<pd::RequestedRoll> {
        use pd::RequestedRoll as R;
        let mut v = vec![
            R::Coin,
            R::D6,
            R::D8,
            R::Deviate,
            R::Scatter,
            R::Sum2D6,
            R::ThrowIn,
            R::D6PassFail { target: 4 },
            R::D6ThreeOutcomes { low: 2, high: 5 },
            R::FoulArmor { target: 9 },
            R::FoulInjury {
                ko_target: 8,
                cas_target: 10,
            },
            R::Sum2D6PassFail { target: 7 },
            R::Sum2D6ThreeOutcomes { low: 3, high: 11 },
        ];
        for count in [
            pd::NumBlockDices::ThreeUphill,
            pd::NumBlockDices::TwoUphill,
            pd::NumBlockDices::One,
            pd::NumBlockDices::Two,
            pd::NumBlockDices::Three,
        ] {
            v.push(R::BlockDice { count });
        }
        v
    }

    #[test]
    fn requested_rolls_round_trip() {
        for r in all_requested_rolls() {
            let engine = requested_roll_from_proto(r).expect("valid");
            assert_eq!(requested_roll_to_proto(engine), r, "{r:?}");
        }
    }

    fn all_roll_results() -> Vec<pd::RollResult> {
        use pd::RollResult as R;
        let mut v = vec![
            R::Pass,
            R::Fail,
            R::MiddleOutcome,
            R::Coin(pd::Coin::Heads),
            R::Coin(pd::Coin::Tails),
            R::FoulArmor {
                broken: true,
                ejected: false,
            },
            R::FoulArmor {
                broken: false,
                ejected: true,
            },
            R::D6 { value: 1 },
            R::D6 { value: 6 },
            R::D8 { value: 8 },
            R::Deviate {
                distance: 3,
                direction: 7,
            },
            R::Scatter {
                first: 1,
                second: 4,
                third: 8,
            },
            R::Sum2D6 { value: 12 },
            R::ThrowIn {
                direction: 2,
                distance: 9,
            },
        ];
        for outcome in [
            pd::InjuryOutcome::Stunned,
            pd::InjuryOutcome::KO,
            pd::InjuryOutcome::Casualty,
        ] {
            v.push(R::FoulInjury {
                outcome,
                ejected: false,
            });
        }
        for face in [
            pd::BlockDice::Skull,
            pd::BlockDice::BothDown,
            pd::BlockDice::Push,
            pd::BlockDice::PowPush,
            pd::BlockDice::Pow,
        ] {
            v.push(R::BlockDice { faces: vec![face] });
            v.push(R::BlockDice {
                faces: vec![face, pd::BlockDice::Pow],
            });
            v.push(R::BlockDice {
                faces: vec![face, pd::BlockDice::Pow, pd::BlockDice::Skull],
            });
        }
        v
    }

    #[test]
    fn roll_results_round_trip() {
        for r in all_roll_results() {
            let engine = roll_result_from_proto(&r).expect("valid");
            assert_eq!(roll_result_to_proto(engine), r, "{r:?}");
        }
    }

    #[test]
    fn every_engine_roll_request_is_answered_by_a_compatible_mirrored_result() {
        // The mirror is only useful if a pinned roll actually satisfies
        // `RequestedRoll::is_compatible` — that is what `micro_step` asserts.
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for r in all_requested_rolls() {
            let engine = requested_roll_from_proto(r).unwrap();
            let result = ed::resolve_with_rng(engine, &mut rng);
            let round_tripped = roll_result_from_proto(&roll_result_to_proto(result)).unwrap();
            assert!(
                engine.is_compatible(round_tripped),
                "{r:?}: mirrored {round_tripped:?} is not compatible"
            );
        }
    }

    #[test]
    fn invalid_face_values_are_rejected_not_panicked_on() {
        assert!(roll_result_from_proto(&pd::RollResult::D6 { value: 7 }).is_err());
        assert!(roll_result_from_proto(&pd::RollResult::D8 { value: 0 }).is_err());
        assert!(roll_result_from_proto(&pd::RollResult::Sum2D6 { value: 13 }).is_err());
        assert!(roll_result_from_proto(&pd::RollResult::BlockDice { faces: vec![] }).is_err());
        assert!(requested_roll_from_proto(pd::RequestedRoll::D6PassFail { target: 1 }).is_err());
    }
}

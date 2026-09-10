//! Mirrors of the engine's dice vocabulary.
//!
//! The server drives the engine in `DiceMode::RegisterRolls` (decision 4), so
//! it *sees* every `RequestedRoll`/`RollResult` pair and can both stream them
//! to the client and accept a pinned value back ([`crate::ClientMsg::FixNextRoll`]).
//!
//! The variant *structure* is mirrored one-for-one so the round trip is
//! lossless, but the engine's `#[repr(u8)]` numeric dice newtypes (`D6`, `D8`,
//! `D3`, `Sum2D6`, `D6Target`, `Sum2D6Target`) are carried as plain `u8` face
//! values. They all have `TryFrom<u8>`, so the server-side conversion is total
//! in both directions and the wire stays readable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Coin {
    Heads,
    Tails,
}

/// A single block-die face. Mirror of `dices::BlockDice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockDice {
    Skull,
    BothDown,
    Push,
    PowPush,
    Pow,
}

impl BlockDice {
    /// Sprite path relative to the `/img/` asset mount.
    pub fn img(self) -> &'static str {
        match self {
            BlockDice::Skull => "dice/attacker_down.png",
            BlockDice::BothDown => "dice/both_down.png",
            BlockDice::Push => "dice/push.png",
            BlockDice::PowPush => "dice/defender_stumbles.png",
            BlockDice::Pow => "dice/defender_down.png",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BlockDice::Skull => "Attacker Down",
            BlockDice::BothDown => "Both Down",
            BlockDice::Push => "Push",
            BlockDice::PowPush => "POW!/Push",
            BlockDice::Pow => "POW!",
        }
    }
}

/// How many block dice, and who picks. Mirror of `table::NumBlockDices`
/// (declaration order is worst → best, so `Ord` is meaningful).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NumBlockDices {
    ThreeUphill,
    TwoUphill,
    One,
    Two,
    Three,
}

impl NumBlockDices {
    /// Number of dice actually rolled.
    pub fn count(self) -> usize {
        match self {
            NumBlockDices::One => 1,
            NumBlockDices::Two | NumBlockDices::TwoUphill => 2,
            NumBlockDices::Three | NumBlockDices::ThreeUphill => 3,
        }
    }

    /// "Uphill" = the *defender* picks which die counts.
    pub fn defender_picks(self) -> bool {
        matches!(self, NumBlockDices::TwoUphill | NumBlockDices::ThreeUphill)
    }

    /// The block-count preview badge drawn on a block target (relative to `/img/`).
    pub fn badge(self) -> &'static str {
        match self {
            NumBlockDices::One => "icons/decorations/block1d.gif",
            NumBlockDices::Two => "icons/decorations/block2d.gif",
            NumBlockDices::Three => "icons/decorations/block3d.gif",
            NumBlockDices::TwoUphill => "icons/decorations/block2dagainst.gif",
            NumBlockDices::ThreeUphill => "icons/decorations/block3dagainst.gif",
        }
    }

    /// Signed die count as the UI labels it: `-2` for a two-dice uphill block.
    pub fn signed(self) -> i8 {
        let n = self.count() as i8;
        if self.defender_picks() {
            -n
        } else {
            n
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InjuryOutcome {
    Stunned,
    KO,
    Casualty,
}

/// Mirror of `dices::RequestedRoll`. Numeric targets are the engine's
/// `#[repr(u8)]` discriminants (`D6Target::FourPlus == 4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestedRoll {
    BlockDice { count: NumBlockDices },
    Coin,
    D6,
    D6PassFail { target: u8 },
    D6ThreeOutcomes { low: u8, high: u8 },
    D8,
    FoulArmor { target: u8 },
    FoulInjury { ko_target: u8, cas_target: u8 },
    Deviate,
    Scatter,
    Sum2D6,
    Sum2D6PassFail { target: u8 },
    Sum2D6ThreeOutcomes { low: u8, high: u8 },
    ThrowIn,
}

impl RequestedRoll {
    /// What the UI calls this roll while it is pending.
    pub fn label(self) -> String {
        match self {
            RequestedRoll::BlockDice { count } => format!("Block ({} dice)", count.signed()),
            RequestedRoll::Coin => "Coin toss".into(),
            RequestedRoll::D6 => "D6".into(),
            RequestedRoll::D6PassFail { target } => format!("D6 {target}+"),
            RequestedRoll::D6ThreeOutcomes { low, high } => format!("D6 {low}+/{high}+"),
            RequestedRoll::D8 => "D8 direction".into(),
            RequestedRoll::FoulArmor { target } => format!("Armour {target}+"),
            RequestedRoll::FoulInjury { ko_target, cas_target } => {
                format!("Injury (KO {ko_target}+, cas {cas_target}+)")
            }
            RequestedRoll::Deviate => "Deviate".into(),
            RequestedRoll::Scatter => "Scatter".into(),
            RequestedRoll::Sum2D6 => "2D6".into(),
            RequestedRoll::Sum2D6PassFail { target } => format!("2D6 {target}+"),
            RequestedRoll::Sum2D6ThreeOutcomes { low, high } => format!("2D6 {low}+/{high}+"),
            RequestedRoll::ThrowIn => "Throw-in".into(),
        }
    }
}

/// Mirror of `dices::RollResult`. `BlockDice` carries only the faces actually
/// rolled (the engine's `[Option<BlockDice>; 3]` is always a `Some` prefix).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RollResult {
    BlockDice {
        faces: Vec<BlockDice>,
    },
    Coin(Coin),
    Pass,
    Fail,
    FoulArmor {
        broken: bool,
        ejected: bool,
    },
    FoulInjury {
        outcome: InjuryOutcome,
        ejected: bool,
    },
    MiddleOutcome,
    D6 {
        value: u8,
    },
    D8 {
        value: u8,
    },
    /// `distance` is the D6, `direction` the D8.
    Deviate {
        distance: u8,
        direction: u8,
    },
    /// Three successive D8 directions.
    Scatter {
        first: u8,
        second: u8,
        third: u8,
    },
    Sum2D6 {
        value: u8,
    },
    /// `direction` is a D3, `distance` a 2D6 sum.
    ThrowIn {
        direction: u8,
        distance: u8,
    },
}

/// One resolved roll, as streamed to the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiceEvent {
    pub requested: RequestedRoll,
    pub result: RollResult,
    /// One line for the dice ticker, e.g. `"Dodge 3+: 5 — success"`.
    pub text: String,
    /// Faces to draw, in roll order.
    pub faces: Vec<DieFace>,
    /// True when the value came from the debug "fix next roll" control rather
    /// than the session RNG.
    pub fixed: bool,
}

/// A single die face to draw: a sprite under `/img/` plus its alt text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DieFace {
    /// Path relative to the asset mount, e.g. `"dice/5.png"`.
    pub img: String,
    pub label: String,
}

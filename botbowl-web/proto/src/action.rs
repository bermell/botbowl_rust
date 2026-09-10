//! Mirrors of the engine's action vocabulary.
//!
//! `proto` deliberately does **not** depend on `botbowl-engine`: the wasm
//! client must compile without the engine (decision 3 of plan 034), and the
//! engine is a large host-only crate. The price is that these enums are
//! hand-copied from `botbowl_engine::core::table`, so
//! `botbowl-web/server/src/mirror.rs` carries exhaustive `From`/`Into`
//! conversions plus a round-trip test that fails to compile the moment an
//! engine variant is added or removed.

use serde::{Deserialize, Serialize};

/// A board square. Coordinates are engine coordinates: the full grid
/// including the 2-cell out-of-bounds border, `x` across the pitch's long
/// axis, `y` across the short one, both 0-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Position {
    pub x: i8,
    pub y: i8,
}

impl Position {
    pub fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TeamType {
    Home,
    Away,
}

impl TeamType {
    pub fn other(self) -> Self {
        match self {
            TeamType::Home => TeamType::Away,
            TeamType::Away => TeamType::Home,
        }
    }
}

/// Positional action types — mirror of `botbowl_engine::core::table::PosAT`.
/// Order matches `botbowl_nn::actions::pos_at_index`, which is the schema
/// the trained nets use; keep it stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PosAT {
    StartMove,
    StartBlitz,
    StartPass,
    StartFoul,
    SelectPosition,
    Push,
    FollowUp,
    StartHandoff,
    Handoff,
    Pass,
    Move,
    Foul,
    StartBlock,
    Block,
}

impl PosAT {
    /// All variants, in policy-channel order.
    pub const ALL: [PosAT; 14] = [
        PosAT::StartMove,
        PosAT::StartBlitz,
        PosAT::StartPass,
        PosAT::StartFoul,
        PosAT::SelectPosition,
        PosAT::Push,
        PosAT::FollowUp,
        PosAT::StartHandoff,
        PosAT::Handoff,
        PosAT::Pass,
        PosAT::Move,
        PosAT::Foul,
        PosAT::StartBlock,
        PosAT::Block,
    ];

    /// Label for the action menu / tooltip.
    pub fn label(self) -> &'static str {
        match self {
            PosAT::StartMove => "Move",
            PosAT::StartBlitz => "Blitz",
            PosAT::StartPass => "Pass",
            PosAT::StartFoul => "Foul",
            PosAT::SelectPosition => "Place here",
            PosAT::Push => "Push here",
            PosAT::FollowUp => "Follow up",
            PosAT::StartHandoff => "Hand-off",
            PosAT::Handoff => "Hand off to",
            PosAT::Pass => "Pass to",
            PosAT::Move => "Step here",
            PosAT::Foul => "Foul",
            PosAT::StartBlock => "Block",
            PosAT::Block => "Block",
        }
    }

    /// Sprite path relative to the `/img/` asset mount, when one fits.
    pub fn icon(self) -> Option<&'static str> {
        match self {
            PosAT::StartMove | PosAT::Move => Some("icons/actions/move.gif"),
            PosAT::StartBlitz => Some("icons/actions/blitz.gif"),
            PosAT::StartBlock | PosAT::Block => Some("icons/actions/block.gif"),
            PosAT::StartPass | PosAT::Pass => Some("icons/actions/pass.gif"),
            PosAT::StartHandoff | PosAT::Handoff => Some("icons/actions/handoff.gif"),
            PosAT::StartFoul | PosAT::Foul => Some("icons/actions/foul.gif"),
            PosAT::Push | PosAT::FollowUp | PosAT::SelectPosition => None,
        }
    }

    /// True for the "declare a player action" family, i.e. the actions that
    /// pick *which* of your players acts rather than where a resolution goes.
    pub fn is_start(self) -> bool {
        matches!(
            self,
            PosAT::StartMove
                | PosAT::StartBlitz
                | PosAT::StartPass
                | PosAT::StartFoul
                | PosAT::StartHandoff
                | PosAT::StartBlock
        )
    }
}

/// Non-positional action types — mirror of
/// `botbowl_engine::core::table::SimpleAT`, in `simple_at_index` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimpleAT {
    SelectBothDown,
    SelectPow,
    SelectPush,
    SelectPowPush,
    SelectSkull,
    UseReroll,
    DontUseReroll,
    EndPlayerTurn,
    EndTurn,
    Heads,
    Tails,
    Kick,
    Receive,
    SetupLine,
    EndSetup,
    KickoffAimMiddle,
}

impl SimpleAT {
    pub const ALL: [SimpleAT; 16] = [
        SimpleAT::SelectBothDown,
        SimpleAT::SelectPow,
        SimpleAT::SelectPush,
        SimpleAT::SelectPowPush,
        SimpleAT::SelectSkull,
        SimpleAT::UseReroll,
        SimpleAT::DontUseReroll,
        SimpleAT::EndPlayerTurn,
        SimpleAT::EndTurn,
        SimpleAT::Heads,
        SimpleAT::Tails,
        SimpleAT::Kick,
        SimpleAT::Receive,
        SimpleAT::SetupLine,
        SimpleAT::EndSetup,
        SimpleAT::KickoffAimMiddle,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SimpleAT::SelectBothDown => "Both Down",
            SimpleAT::SelectPow => "POW!",
            SimpleAT::SelectPush => "Push",
            SimpleAT::SelectPowPush => "POW!/Push",
            SimpleAT::SelectSkull => "Attacker Down",
            SimpleAT::UseReroll => "Use reroll",
            SimpleAT::DontUseReroll => "No reroll",
            SimpleAT::EndPlayerTurn => "End player action",
            SimpleAT::EndTurn => "End turn",
            SimpleAT::Heads => "Heads",
            SimpleAT::Tails => "Tails",
            SimpleAT::Kick => "Kick",
            SimpleAT::Receive => "Receive",
            SimpleAT::SetupLine => "Auto setup",
            SimpleAT::EndSetup => "End setup",
            SimpleAT::KickoffAimMiddle => "Aim at middle",
        }
    }

    /// For the five block-die picks, the face sprite (relative to `/img/`).
    pub fn block_die_face(self) -> Option<&'static str> {
        match self {
            SimpleAT::SelectSkull => Some("dice/attacker_down.png"),
            SimpleAT::SelectBothDown => Some("dice/both_down.png"),
            SimpleAT::SelectPush => Some("dice/push.png"),
            SimpleAT::SelectPowPush => Some("dice/defender_stumbles.png"),
            SimpleAT::SelectPow => Some("dice/defender_down.png"),
            _ => None,
        }
    }
}

/// Mirror of `botbowl_engine::core::model::Action`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Positional(PosAT, Position),
    Simple(SimpleAT),
}

impl Action {
    pub fn position(self) -> Option<Position> {
        match self {
            Action::Positional(_, p) => Some(p),
            Action::Simple(_) => None,
        }
    }

    /// Short human-readable form for the log / candidate list.
    pub fn describe(self) -> String {
        match self {
            Action::Positional(at, p) => format!("{} ({},{})", at.label(), p.x, p.y),
            Action::Simple(at) => at.label().to_string(),
        }
    }
}

//! Bijection between engine [`Action`]s and policy-head cells.
//!
//! The policy head has `A = 30` channels: the 14 [`PosAT`] variants map
//! to channels `0..14`, the 16 [`SimpleAT`] variants to channels
//! `14..30`. The forward and inverse maps are **exhaustive matches** on
//! the engine enums — adding a variant there is a compile error here, a
//! deliberate trip-wire forcing a schema version bump (the manifest
//! records `A` and the channel names).
//!
//! A positional action targets a single board cell: `(channel, y, x)`
//! with `x` canonicalised to the mover-attacks-toward-`x=1` frame
//! ([`crate::perspective`]) so it lines up with the canonically-oriented
//! spatial tensor. A simple action has no cell; training/inference gather
//! its logit as the channel-wide max over the spatial grid — recorded
//! here via `is_simple`.

use botbowl_engine::core::model::{Action, BoardDims, Position, TeamType};
use botbowl_engine::core::table::{PosAT, SimpleAT};

use crate::perspective::canonical_x;

/// Number of positional action types → policy channels `0..NUM_POS_AT`.
pub const NUM_POS_AT: usize = 14;
/// Number of simple action types → policy channels `NUM_POS_AT..POLICY_CHANNELS`.
pub const NUM_SIMPLE_AT: usize = 16;
/// Policy-head channel count `A`.
pub const POLICY_CHANNELS: usize = NUM_POS_AT + NUM_SIMPLE_AT;

/// Where an [`Action`] lands in the policy head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionCell {
    /// Policy channel `0..POLICY_CHANNELS`.
    pub channel: usize,
    /// Canonical board row (y). `0` for simple actions.
    pub y: usize,
    /// Canonical board column (x). `0` for simple actions.
    pub x: usize,
    /// Simple actions have no cell — their logit is the channel-wide max.
    pub is_simple: bool,
}

/// Positional action type → channel `0..14`. Exhaustive by design.
pub fn pos_at_index(at: PosAT) -> usize {
    match at {
        PosAT::StartMove => 0,
        PosAT::StartBlitz => 1,
        PosAT::StartPass => 2,
        PosAT::StartFoul => 3,
        PosAT::SelectPosition => 4,
        PosAT::Push => 5,
        PosAT::FollowUp => 6,
        PosAT::StartHandoff => 7,
        PosAT::Handoff => 8,
        PosAT::Pass => 9,
        PosAT::Move => 10,
        PosAT::Foul => 11,
        PosAT::StartBlock => 12,
        PosAT::Block => 13,
    }
}

/// Simple action type → index `0..16` (policy channel is `NUM_POS_AT + this`).
pub fn simple_at_index(at: SimpleAT) -> usize {
    match at {
        SimpleAT::SelectBothDown => 0,
        SimpleAT::SelectPow => 1,
        SimpleAT::SelectPush => 2,
        SimpleAT::SelectPowPush => 3,
        SimpleAT::SelectSkull => 4,
        SimpleAT::UseReroll => 5,
        SimpleAT::DontUseReroll => 6,
        SimpleAT::EndPlayerTurn => 7,
        SimpleAT::EndTurn => 8,
        SimpleAT::Heads => 9,
        SimpleAT::Tails => 10,
        SimpleAT::Kick => 11,
        SimpleAT::Receive => 12,
        SimpleAT::SetupLine => 13,
        SimpleAT::EndSetup => 14,
        SimpleAT::KickoffAimMiddle => 15,
    }
}

/// Inverse of [`pos_at_index`]. Panics on an out-of-range channel.
pub fn pos_at_from_index(i: usize) -> PosAT {
    match i {
        0 => PosAT::StartMove,
        1 => PosAT::StartBlitz,
        2 => PosAT::StartPass,
        3 => PosAT::StartFoul,
        4 => PosAT::SelectPosition,
        5 => PosAT::Push,
        6 => PosAT::FollowUp,
        7 => PosAT::StartHandoff,
        8 => PosAT::Handoff,
        9 => PosAT::Pass,
        10 => PosAT::Move,
        11 => PosAT::Foul,
        12 => PosAT::StartBlock,
        13 => PosAT::Block,
        _ => panic!("pos_at channel {i} out of range 0..{NUM_POS_AT}"),
    }
}

/// Inverse of [`simple_at_index`]. Panics on an out-of-range index.
pub fn simple_at_from_index(i: usize) -> SimpleAT {
    match i {
        0 => SimpleAT::SelectBothDown,
        1 => SimpleAT::SelectPow,
        2 => SimpleAT::SelectPush,
        3 => SimpleAT::SelectPowPush,
        4 => SimpleAT::SelectSkull,
        5 => SimpleAT::UseReroll,
        6 => SimpleAT::DontUseReroll,
        7 => SimpleAT::EndPlayerTurn,
        8 => SimpleAT::EndTurn,
        9 => SimpleAT::Heads,
        10 => SimpleAT::Tails,
        11 => SimpleAT::Kick,
        12 => SimpleAT::Receive,
        13 => SimpleAT::SetupLine,
        14 => SimpleAT::EndSetup,
        15 => SimpleAT::KickoffAimMiddle,
        _ => panic!("simple_at index {i} out of range 0..{NUM_SIMPLE_AT}"),
    }
}

/// Map an engine [`Action`] to its policy cell, canonicalising the
/// position into the mover-centric frame.
pub fn action_cell(action: Action, mover: TeamType, dims: BoardDims) -> ActionCell {
    match action {
        Action::Positional(at, pos) => {
            let cx = canonical_x(pos.x, dims, mover);
            ActionCell {
                channel: pos_at_index(at),
                y: pos.y as usize,
                x: cx as usize,
                is_simple: false,
            }
        }
        Action::Simple(at) => ActionCell {
            channel: NUM_POS_AT + simple_at_index(at),
            y: 0,
            x: 0,
            is_simple: true,
        },
    }
}

/// Reconstruct the engine [`Action`] from a policy cell. `x`/`y` are
/// canonical coordinates (as produced by [`action_cell`]); the mirror is
/// its own inverse so passing `mover` back in undoes the canonicalisation.
/// Simple channels ignore `x`/`y`.
pub fn action_from_cell(cell: ActionCell, mover: TeamType, dims: BoardDims) -> Action {
    if cell.channel < NUM_POS_AT {
        let at = pos_at_from_index(cell.channel);
        let raw_x = canonical_x(cell.x as i8, dims, mover);
        Action::Positional(at, Position::new((raw_x, cell.y as i8)))
    } else {
        Action::Simple(simple_at_from_index(cell.channel - NUM_POS_AT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_channel_count_is_thirty() {
        assert_eq!(POLICY_CHANNELS, 30);
        assert_eq!(NUM_POS_AT, 14);
        assert_eq!(NUM_SIMPLE_AT, 16);
    }

    #[test]
    fn pos_at_indices_are_a_dense_bijection() {
        use PosAT::*;
        let all = [
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
        ];
        assert_eq!(all.len(), NUM_POS_AT);
        for (i, at) in all.into_iter().enumerate() {
            assert_eq!(pos_at_index(at), i);
            assert_eq!(pos_at_from_index(i), at);
        }
    }

    #[test]
    fn simple_at_indices_are_a_dense_bijection() {
        use SimpleAT::*;
        let all = [
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
        ];
        assert_eq!(all.len(), NUM_SIMPLE_AT);
        for (i, at) in all.into_iter().enumerate() {
            assert_eq!(simple_at_index(at), i);
            assert_eq!(simple_at_from_index(i), at);
        }
    }

    #[test]
    fn positional_cell_round_trips_both_perspectives() {
        let dims = BoardDims::default();
        for mover in [TeamType::Home, TeamType::Away] {
            for &(at, xy) in &[
                (PosAT::Move, (5i8, 7i8)),
                (PosAT::Block, (1, 1)),
                (PosAT::Handoff, (dims.width - 2, dims.height - 2)),
            ] {
                let action = Action::Positional(at, Position::new(xy));
                let cell = action_cell(action, mover, dims);
                assert!(!cell.is_simple);
                assert_eq!(action_from_cell(cell, mover, dims), action);
            }
        }
    }

    #[test]
    fn away_positional_cell_is_x_mirrored() {
        let dims = BoardDims::default();
        let action = Action::Positional(PosAT::Move, Position::new((3, 4)));
        let home = action_cell(action, TeamType::Home, dims);
        let away = action_cell(action, TeamType::Away, dims);
        assert_eq!(home.x, 3);
        assert_eq!(away.x, (dims.width - 1) as usize - 3);
        assert_eq!(home.y, away.y);
        assert_eq!(home.channel, away.channel);
    }

    #[test]
    fn simple_cell_round_trips() {
        let dims = BoardDims::default();
        let action = Action::Simple(SimpleAT::EndTurn);
        let cell = action_cell(action, TeamType::Away, dims);
        assert!(cell.is_simple);
        assert_eq!(cell.channel, NUM_POS_AT + simple_at_index(SimpleAT::EndTurn));
        assert_eq!(action_from_cell(cell, TeamType::Away, dims), action);
    }
}

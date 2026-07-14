//! Single authority for perspective + canonical board orientation.
//!
//! Every place that turns a `GameState` into tensor coordinates — the
//! encoder, the action↔cell map, the evaluator's value sign — routes
//! through here, so a perspective bug can only exist in one file.
//!
//! **Whose move it is** ([`mover_for`]): the team named by
//! `available_actions.team`, falling back to `info.team_turn` when the
//! engine has no team-scoped decision exposed (mid-procedure / chance
//! states the evaluator may still be asked to score).
//!
//! **Canonical view**: the mover always attacks toward `x = 1`. `Home`
//! already attacks toward its scoring endzone at `x = 1`
//! (`BoardDims::endzone_x(Home) == 1`), so `Home` states are encoded
//! verbatim; `Away` attacks toward `x = width-2`, so `Away` states are
//! **mirrored across the vertical axis** (`x → (width-1) - x`, no y-flip).
//! The mirror is its own inverse and maps border cells to border cells.
//!
//! **Value convention**: the network's scalar `v` is *mover-centric*
//! (`+1` = the team to move is winning). The Home-centric target/return
//! is recovered by flipping sign when the mover is `Away` — see
//! [`crate::eval`] and [`crate::targets`].

use botbowl_engine::core::model::{BoardDims, Coord, Position, TeamType};

/// The team to move at `state` — the canonical anchor for every
/// perspective-dependent transform. Uses the engine's exposed decision
/// team, falling back to `team_turn` for mid-procedure / chance states
/// (which the evaluator can still be asked to score).
pub fn mover_for(state: &botbowl_engine::core::gamestate::GameState) -> TeamType {
    state.available_actions.team.unwrap_or(state.info.team_turn)
}

/// Canonicalise a board x-coordinate into the mover-attacks-toward-`x=1`
/// frame. Identity for `Home`, mirror across the vertical axis for
/// `Away`. Involutive: applying it twice returns the original x.
#[inline]
pub fn canonical_x(x: Coord, dims: BoardDims, mover: TeamType) -> Coord {
    match mover {
        TeamType::Home => x,
        TeamType::Away => (dims.width - 1) - x,
    }
}

/// Canonicalise a whole position (only x is mirrored; y is preserved).
#[inline]
pub fn canonical_pos(pos: Position, dims: BoardDims, mover: TeamType) -> Position {
    Position::new((canonical_x(pos.x, dims, mover), pos.y))
}

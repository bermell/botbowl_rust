//! Play one game, return its record.
//!
//! This is the process-agnostic core that `botbowl-ui dataset` and
//! `botbowl-ui eval` used to carry inline, split out (plan 041 phase 0) so
//! a remote worker can run the exact same code path. Nothing here knows
//! about files, CLI flags, progress output or threads: a caller hands in a
//! config and a seed and gets back a [`botbowl_data::Trajectory`] or an
//! [`eval::EvalGameLine`]. Parallelism, output and provenance stamping
//! belong to the caller — `botbowl-ui` for the single-box path, the hub
//! and worker for the distributed one.
//!
//! - [`bots`] — evaluator/search configuration and `MctsBot` construction.
//! - [`generate`] — self-play, random-start and curriculum trajectories.
//! - [`eval`] — one ladder game, its per-game record, and the report rows
//!   the per-game records fold into.

pub mod bots;
pub mod eval;
pub mod generate;

/// Game workers only orchestrate — the search runs on `MctsBot`'s own
/// threads — but they do build a `GameState` on the stack, so match the
/// engine's generous convention rather than the 2 MB default.
pub const GAME_STACK_SIZE: usize = 16 * 1024 * 1024;

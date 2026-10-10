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
//! - [`board_sizes`] — weighted board-size distributions and the seed-keyed
//!   draw that decides which board a game is played on (plan 042).
//! - [`bots`] — evaluator/search configuration and `MctsBot` construction.
//! - [`drives`] — paired contested drives from frozen positions, a cheaper ladder rung (plan 051).
//! - [`generate`] — self-play, random-start and curriculum trajectories.
//! - [`mc_label`] — Monte Carlo value labels: replay a trajectory, average policy-only playouts
//!   per sample (plan 056; shared by `botbowl-ui mc-label` and the hub's `job label`, plan 062).
//! - [`policy`] — the bare policy as a bot, and policy-only drive playouts.
//! - [`eval`] — one ladder game, its per-game record, and the report rows
//!   the per-game records fold into.
//! - [`stats`] — paired-game scoring and the SPRT that stops a decided rung (plan 051).
//! - [`trace`] — the opt-in per-decision tree-reuse trace (plan 043).

pub mod board_sizes;
pub mod bots;
#[cfg(feature = "cli")]
pub mod cli_args;
pub mod drives;
pub mod eval;
pub mod generate;
pub mod mc_label;
pub mod policy;
pub mod stats;
pub mod trace;

/// Game workers only orchestrate — the search runs on `MctsBot`'s own
/// threads — but they do build a `GameState` on the stack, so match the
/// engine's generous convention rather than the 2 MB default.
pub const GAME_STACK_SIZE: usize = 16 * 1024 * 1024;

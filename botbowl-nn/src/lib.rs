//! Neural-network plumbing for the Blood Bowl MCTS bot (plan 017).
//!
//! All `GameState → tensor` encoding lives here in Rust so the offline
//! prepare step ([`bin/prepare`]) and the live evaluator ([`eval`]) share
//! **exactly** the same feature layout — there is no train/inference
//! encoding skew, and Python never parses a `GameState`.
//!
//! Module map:
//! - [`actions`] — `Action ↔ policy cell`. An exhaustive match pins the
//!   engine action enums to fixed policy channels (adding an engine
//!   variant is a compile error → a deliberate version bump).
//! - [`perspective`] — the single authority on whose move it is and the
//!   canonical (mover-centric) board orientation.
//! - [`encode`] — `encode(state) -> Encoded { spatial, global, h, w, mover }`.
//! - [`targets`] — policy/value training targets built from the raw
//!   search stats (plan 017 §caveat).
//! - [`npy`] — a minimal hand-rolled `.npy` v1 writer/reader.
//! - [`eval`] — `NnEvaluator`: tract-onnx inference for priors + value,
//!   with an optional batched remote backend.
//! - [`remote`] — client for the batching inference sidecar
//!   (`scripts/nn_server.py`, plan 024).

pub mod actions;
pub mod encode;
pub mod eval;
pub mod npy;
pub mod perspective;
pub mod remote;
pub mod targets;

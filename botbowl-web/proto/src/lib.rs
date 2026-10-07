//! Wire types shared by `botbowl-web-server` and the wasm client.
//!
//! This crate deliberately has **no engine dependency** (decision 1/3 of
//! `plans/completed/034-plan--web-play-ui.md`): it must compile for `wasm32-unknown-unknown`,
//! and the client is a pure renderer of a server-derived view. The engine
//! enums are therefore hand-mirrored here, with exhaustive conversions and a
//! round-trip test living in `botbowl-web-server/src/mirror.rs` — adding an
//! engine variant breaks *that* compile, which is the intended trip-wire.
//!
//! Module map:
//! - [`action`] — `Action`/`PosAT`/`SimpleAT`/`Position`/`TeamType` mirrors.
//! - [`dice`] — `RequestedRoll`/`RollResult` mirrors plus a rendered
//!   [`dice::DiceEvent`].
//! - [`log`] — the streamed game log, [`log::LogEntry`] (text and dice in one).
//! - [`view`] — [`view::ViewState`], the fully derived board.
//! - [`search`] — [`search::SearchReport`] and the tree explorer's
//!   [`search::NodeExpansion`].
//! - [`decision`] — the decision log's [`decision::DecisionRecord`] and the
//!   net's [`decision::NetReadout`].
//! - [`msg`] — [`msg::ClientMsg`]/[`msg::ServerMsg`] and the lobby's
//!   [`msg::GameSpec`].
//! - [`team`] — rosters: [`team::TeamDef`] and the built-in teams.

pub mod action;
pub mod decision;
pub mod dice;
pub mod log;
pub mod msg;
pub mod search;
pub mod team;
pub mod view;

pub use action::{Action, PosAT, Position, SimpleAT, TeamType};
pub use decision::{ActionPrior, Decider, DecisionRecord, NetReadout};
pub use dice::{DiceEvent, DieFace, RequestedRoll, RollResult};
pub use log::{LogEntry, LogKind};
pub use msg::{BoardSpec, BotSpec, Budget, ClientMsg, GameSpec, LobbyInfo, MctsSpec, Seat, ServerMsg};
pub use search::{ChildReport, NodeExpansion, SearchEdge, SearchReport};
pub use view::{Dims, PlayerView, SetupView, SquareView, ViewState};

/// Bumped whenever a wire type changes shape. The client refuses to render a
/// view from a server it does not match.
pub const WIRE_VERSION: u32 = 7;

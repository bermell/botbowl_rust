//! The web-play server: owns the `GameState` and the bots, streams a fully
//! derived view to the browser over one websocket per game.
//!
//! See `plans/034-plan--web-play-ui.md`. The shape in one paragraph: axum
//! serves the trunk-built wasm client and the sprite directory from a sibling
//! checkout; each websocket gets a `spawn_blocking` thread that owns one
//! [`session::GameSession`]; the session drives the engine in
//! `DiceMode::RegisterRolls` so it sees every die, and turns each resulting
//! `GameState` into a [`botbowl_web_proto::view::ViewState`] with
//! [`view::derive`].
//!
//! Module map:
//! - [`mirror`] — exhaustive engine ↔ `proto` conversions (the trip-wire that
//!   keeps the engine-free wire types honest).
//! - [`view`] — `GameState -> ViewState`, pure and unit-tested.
//! - [`dice`] — a resolved roll rendered for the dice ticker.
//! - [`bots`] — `BotSpec -> SessionBot`, plus the model catalogue.
//! - [`report`] — `botbowl_mcts::report` → the wire's search report.
//! - [`teams`] — built-in and saved rosters, and each player's picture.
//! - [`session`] — the per-socket game loop.
//! - [`ws`] — the axum handler that pumps JSON between the two.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::http::{header, HeaderValue};
use axum::routing::get;
use axum::Router;
use botbowl_engine::core::model::{HEIGHT, TEAM_SIZE, WIDTH};
use botbowl_web_proto::msg::{BoardSpec, StepMode};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;

pub mod bots;
pub mod config;
pub mod dice;
pub mod mirror;
pub mod report;
pub mod session;
pub mod teams;
pub mod view;
pub mod ws;

/// Everything a session needs that outlives it.
pub struct AppState {
    /// The largest board this binary can run — a compile-time property of the
    /// engine (`BOARD_SIZE_W`/`_H`/`BOARD_PLAYERS`), in *playable* terms.
    pub capacity: BoardSpec,
    pub models_dir: PathBuf,
    pub recordings_dir: PathBuf,
    pub model_cache: bots::ModelCache,
    pub server: String,
    pub opts: PlayOptions,
}

/// Knobs a binary sets on the app, with defaults safe to leave a tab open on.
#[derive(Debug, Clone)]
pub struct PlayOptions {
    /// The pacing a new connection starts with. Auto at 600 ms, so a bot-vs-bot game is
    /// watchable and does not run flat out.
    pub initial_step_mode: StepMode,
    /// The most search threads one MCTS bot may use, whatever the lobby asks. `None` = no cap.
    pub max_workers: Option<usize>,
    /// Saved teams and uploaded pictures. `None` = built-ins only, nothing saved.
    pub teams: teams::TeamStore,
    /// The sprite directory, for the team editor's picture list.
    pub assets_dir: Option<PathBuf>,
    /// Allow `StartFrom::Recording` (the server opens a path the browser names). Off on a hub,
    /// which listens on the network.
    pub allow_recording_paths: bool,
    /// More directories of nets after `AppState::models_dir` (`web.toml`'s `models_dirs`).
    pub extra_model_dirs: Vec<PathBuf>,
    /// A worker's model cache to offer as well, by the hub's names. `None` = none.
    pub worker_cache: Option<PathBuf>,
}

impl Default for PlayOptions {
    fn default() -> Self {
        PlayOptions {
            initial_step_mode: StepMode::Auto { ms: 600 },
            max_workers: None,
            teams: teams::TeamStore { dir: None },
            assets_dir: None,
            allow_recording_paths: true,
            extra_model_dirs: Vec::new(),
            worker_cache: None,
        }
    }
}

/// The compiled capacity, converted from engine dims (which include the
/// 2-cell out-of-bounds border) to the playable rectangle the lobby and the
/// model filenames use.
pub fn compiled_capacity() -> BoardSpec {
    BoardSpec::new(WIDTH as i8 - 2, HEIGHT as i8 - 2, TEAM_SIZE)
}

impl AppState {
    /// Every net the lobby offers: `models_dir`, the extra directories, then the worker cache.
    pub fn list_models(&self) -> Vec<botbowl_web_proto::msg::ModelInfo> {
        let mut dirs = vec![self.models_dir.clone()];
        dirs.extend(
            self.opts
                .extra_model_dirs
                .iter()
                .filter(|d| **d != self.models_dir)
                .cloned(),
        );
        bots::list_all_models(&dirs, self.opts.worker_cache.as_deref())
    }

    /// Board sizes the lobby offers: the tiers the project actually trains on,
    /// filtered to what this binary can run, plus the capacity itself.
    pub fn board_presets(&self) -> Vec<BoardSpec> {
        let mut presets: Vec<BoardSpec> = [
            BoardSpec::new(8, 3, 3),
            BoardSpec::new(12, 5, 4),
            BoardSpec::new(14, 7, 4),
            BoardSpec::new(16, 9, 6),
            BoardSpec::new(20, 9, 8),
            BoardSpec::new(26, 15, 11),
        ]
        .into_iter()
        .filter(|b| b.validate(self.capacity).is_ok())
        .collect();
        if !presets.contains(&self.capacity) && self.capacity.validate(self.capacity).is_ok() {
            presets.push(self.capacity);
        }
        presets
    }
}

/// The HTTP surface: the websocket, a health probe, the sprite mount and the
/// trunk-built client.
///
/// Built here rather than in `main` so the integration test drives exactly
/// the router the binary serves.
pub fn router(app: Arc<AppState>, assets_dir: Option<&Path>, dist_dir: Option<&Path>) -> Router {
    let mut router = Router::new()
        .route("/ws", get(ws::handler))
        // Exists so a caller can wait for the bind without opening a socket.
        .route("/health", get(|| async { "ok" }));

    // Uploaded team pictures sit under the sprite mount, so every picture path on the wire is
    // relative to `img/` alike.
    if let Some(custom) = app.opts.teams.img_dir() {
        router = router.nest_service("/img/custom", ServeDir::new(custom));
    }
    if let Some(assets) = assets_dir {
        router = router.nest_service("/img", ServeDir::new(assets));
    }
    if let Some(dist) = dist_dir {
        // Single-page app: unknown paths fall back to index.html.
        let index = dist.join("index.html");
        router = router.fallback_service(ServeDir::new(dist).fallback(ServeFile::new(index)));
    }
    // `no-cache` (revalidate, don't blindly reuse) on everything. Without it
    // a browser keeps serving the previous `index.html` after a
    // `trunk build`, so the page silently runs stale wasm — which costs an
    // hour the first time it happens and looks exactly like a code bug.
    router
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
        .with_state(app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_never_exceed_the_compiled_capacity() {
        let app = AppState {
            capacity: compiled_capacity(),
            models_dir: PathBuf::new(),
            recordings_dir: PathBuf::new(),
            model_cache: Default::default(),
            server: String::new(),
            opts: Default::default(),
        };
        let presets = app.board_presets();
        assert!(!presets.is_empty(), "at least one board must fit any capacity");
        for board in presets {
            board
                .validate(app.capacity)
                .unwrap_or_else(|e| panic!("offered an unusable board {board:?}: {e}"));
        }
    }

    #[test]
    fn the_default_build_can_play_the_trained_boards() {
        // The plan builds the server at the default capacity precisely so one
        // binary covers 8x3 and 14x7 (the trained tiers) and 26x15.
        let capacity = compiled_capacity();
        if capacity.width < 26 || capacity.height < 15 || capacity.team_size < 11 {
            eprintln!("skipped: this build is {capacity:?}, not the default 26x15/11");
            return;
        }
        for board in [
            BoardSpec::new(8, 3, 3),
            BoardSpec::new(14, 7, 4),
            BoardSpec::new(26, 15, 11),
        ] {
            board.validate(capacity).unwrap();
        }
    }
}

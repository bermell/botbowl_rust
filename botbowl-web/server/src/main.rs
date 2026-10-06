//! `botbowl-web-server` — play Blood Bowl against the workspace bots in a
//! browser, and inspect the search behind every bot move.
//!
//! ```sh
//! # once, to build the client:
//! (cd botbowl-web/client && trunk build --release)
//! cargo run -p botbowl-web-server -- \
//!     --assets-dir /Users/mattias/repos/blood/botbowl/botbowl/web/static/img
//! ```
//!
//! Binds `127.0.0.1` only. This is a local single-player POC with no auth,
//! and it exposes debug controls (pin the next die roll, load an arbitrary
//! recording) that have no business on a network.
//!
//! Paths come from the flags, then `~/.config/botbowl/web.toml` (created on first run, see
//! [`botbowl_web_server::config`]), then built-in defaults. The built-in
//! `--dist-dir` / `--models-dir` / `--recordings-dir` defaults are
//! resolved from this crate's own source path, **not** the working directory,
//! so `cargo run -p botbowl-web-server` behaves the same from the repo root
//! and from inside `botbowl-web/client`. A cwd-relative default served a 404
//! and "0 model(s)" instead, with no hint why.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use botbowl_web_server::{compiled_capacity, config, router, teams, AppState, PlayOptions};
use clap::Parser;

/// This crate's directory, baked in at compile time: `<repo>/botbowl-web/server`.
const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// A path relative to the repo root, made absolute against [`CRATE_DIR`].
/// Canonicalised when it exists, so the startup banner prints
/// `<repo>/models` rather than `<repo>/botbowl-web/server/../../models`.
fn from_repo_root(relative: &str) -> PathBuf {
    let path = PathBuf::from(CRATE_DIR).join("../..").join(relative);
    std::fs::canonicalize(&path).unwrap_or(path)
}

#[derive(Parser, Debug)]
#[command(
    name = "botbowl-web-server",
    about = "Blood Bowl in the browser — human or bot on either side — with a decision log and search-tree overlays."
)]
struct Args {
    /// Port on 127.0.0.1.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// The sprite directory from the sibling `botbowl` checkout, mounted at
    /// `/img/`. Nothing is copied into this repo: the player icons are
    /// explicitly not under the botbowl licence.
    #[arg(long)]
    assets_dir: Option<PathBuf>,

    /// `trunk build` output for the wasm client. Overrides `dist_dir` in `web.toml`; defaults to
    /// `<repo>/botbowl-web/client/dist`, independent of the working directory.
    #[arg(long)]
    dist_dir: Option<PathBuf>,

    /// Where the lobby looks for `.onnx` nets first, ahead of `models_dirs` in `web.toml` and the
    /// worker cache. Defaults to the first of `models_dirs`, else `<repo>/models`.
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Where `SaveRecording` writes. Only bare filenames are accepted.
    /// Defaults to `<repo>/data/web-games`.
    #[arg(long)]
    recordings_dir: Option<PathBuf>,

    /// Saved teams and uploaded pictures. Defaults to `teams_dir` in `~/.config/botbowl/web.toml`,
    /// else `~/.config/botbowl/teams`.
    #[arg(long)]
    teams_dir: Option<PathBuf>,

    /// Cap on one MCTS bot's search threads, whatever the lobby asks for.
    #[arg(long)]
    max_workers: Option<usize>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let capacity = compiled_capacity();

    // Flag, then ~/.config/botbowl/web.toml, then the built-in default.
    let paths = config::resolve(config::Flags {
        dist_dir: args.dist_dir.clone(),
        models_dir: args.models_dir.clone(),
        assets_dir: args.assets_dir.clone(),
        teams_dir: args.teams_dir.clone(),
    })
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let dist_dir = paths.dist_dir.clone();
    let recordings_dir = args.recordings_dir.unwrap_or_else(|| from_repo_root("data/web-games"));

    // Hard error, not a warning: without the client there is nothing to serve
    // and every request is a bare 404, which says nothing about the cause.
    let index = dist_dir.join("index.html");
    if !index.is_file() {
        eprintln!("no client build at {}", index.display());
        eprintln!("build it with:  cd botbowl-web/client && trunk build --release");
        std::process::exit(1);
    }

    let app = Arc::new(AppState {
        capacity,
        models_dir: paths.models_dir.clone(),
        recordings_dir: recordings_dir.clone(),
        model_cache: Default::default(),
        server: format!(
            "botbowl-web-server {} (capacity {}x{}/{})",
            env!("CARGO_PKG_VERSION"),
            capacity.width,
            capacity.height,
            capacity.team_size
        ),
        opts: PlayOptions {
            max_workers: args.max_workers,
            teams: teams::TeamStore {
                dir: paths.teams_dir.clone(),
            },
            assets_dir: paths.assets_dir.clone(),
            extra_model_dirs: paths.extra_model_dirs.clone(),
            worker_cache: paths.worker_cache.clone(),
            ..PlayOptions::default()
        },
    });

    let models = app.list_models();
    match &paths.assets_dir {
        Some(assets) if !assets.is_dir() => eprintln!(
            "warning: assets dir {} is not a directory; sprites will 404",
            assets.display()
        ),
        Some(_) => {}
        None => eprintln!(
            "warning: no sprite directory (assets_dir in {}, or --assets-dir); the pitch will render without sprites",
            paths.config.display()
        ),
    }
    if models.is_empty() {
        eprintln!(
            "warning: no .onnx models in {} or the worker cache — the MCTS bot will have nothing to offer",
            paths.models_dir.display()
        );
    }
    let router = router(app, paths.assets_dir.as_deref(), Some(&dist_dir));

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, args.port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("botbowl web play on http://{addr}");
    println!("  config {}", paths.config.display());
    println!(
        "  capacity {}x{}/{} · {} model(s) from {}{}{}",
        capacity.width,
        capacity.height,
        capacity.team_size,
        models.len(),
        paths.models_dir.display(),
        paths
            .extra_model_dirs
            .iter()
            .map(|d| format!(", {}", d.display()))
            .collect::<String>(),
        paths
            .worker_cache
            .as_ref()
            .map(|d| format!(" and the worker cache {}", d.display()))
            .unwrap_or_default()
    );
    println!("  client from {}", dist_dir.display());
    axum::serve(listener, router).await.unwrap();
}

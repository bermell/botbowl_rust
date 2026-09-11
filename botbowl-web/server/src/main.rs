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
//! The `--dist-dir` / `--models-dir` / `--recordings-dir` defaults are
//! resolved from this crate's own source path, **not** the working directory,
//! so `cargo run -p botbowl-web-server` behaves the same from the repo root
//! and from inside `botbowl-web/client`. A cwd-relative default served a 404
//! and "0 model(s)" instead, with no hint why.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use botbowl_web_server::{bots, compiled_capacity, router, AppState};
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
    about = "Human-vs-bot Blood Bowl in the browser, with search-tree overlays."
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

    /// `trunk build` output for the wasm client. Defaults to
    /// `<repo>/botbowl-web/client/dist`, independent of the working directory.
    #[arg(long)]
    dist_dir: Option<PathBuf>,

    /// Where the lobby looks for `.onnx` nets. Defaults to `<repo>/models`.
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Where `SaveRecording` writes. Only bare filenames are accepted.
    /// Defaults to `<repo>/data/web-games`.
    #[arg(long)]
    recordings_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let capacity = compiled_capacity();

    let dist_dir = args
        .dist_dir
        .unwrap_or_else(|| from_repo_root("botbowl-web/client/dist"));
    let models_dir = args.models_dir.unwrap_or_else(|| from_repo_root("models"));
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
        models_dir: models_dir.clone(),
        recordings_dir: recordings_dir.clone(),
        model_cache: Default::default(),
        server: format!(
            "botbowl-web-server {} (capacity {}x{}/{})",
            env!("CARGO_PKG_VERSION"),
            capacity.width,
            capacity.height,
            capacity.team_size
        ),
    });

    let models = bots::list_models(&models_dir);
    if let Some(assets) = &args.assets_dir {
        if !assets.is_dir() {
            eprintln!(
                "warning: --assets-dir {} is not a directory; sprites will 404",
                assets.display()
            );
        }
    } else {
        eprintln!("warning: no --assets-dir given; the pitch will render without sprites");
    }
    if models.is_empty() {
        eprintln!(
            "warning: no .onnx models in {} — the NN evaluators will have nothing to offer",
            models_dir.display()
        );
    }
    let router = router(app, args.assets_dir.as_deref(), Some(&dist_dir));

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, args.port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("botbowl web play on http://{addr}");
    println!(
        "  capacity {}x{}/{} · {} model(s) in {}",
        capacity.width,
        capacity.height,
        capacity.team_size,
        models.len(),
        models_dir.display()
    );
    println!("  client from {}", dist_dir.display());
    axum::serve(listener, router).await.unwrap();
}

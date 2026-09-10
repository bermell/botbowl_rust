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

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use botbowl_web_server::{bots, compiled_capacity, router, AppState};
use clap::Parser;

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

    /// `trunk build` output for the wasm client.
    #[arg(long, default_value = "botbowl-web/client/dist")]
    dist_dir: PathBuf,

    /// Where the lobby looks for `.onnx` nets.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Where `SaveRecording` writes. Only bare filenames are accepted.
    #[arg(long, default_value = "data/web-games")]
    recordings_dir: PathBuf,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let capacity = compiled_capacity();

    let app = Arc::new(AppState {
        capacity,
        models_dir: args.models_dir.clone(),
        recordings_dir: args.recordings_dir.clone(),
        model_cache: Default::default(),
        server: format!(
            "botbowl-web-server {} (capacity {}x{}/{})",
            env!("CARGO_PKG_VERSION"),
            capacity.width,
            capacity.height,
            capacity.team_size
        ),
    });

    let models = bots::list_models(&args.models_dir);
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
    if !args.dist_dir.join("index.html").is_file() {
        eprintln!(
            "warning: {} not found — run `cd botbowl-web/client && trunk build`",
            args.dist_dir.join("index.html").display()
        );
    }
    let router = router(app, args.assets_dir.as_deref(), Some(&args.dist_dir));

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
        args.models_dir.display()
    );
    axum::serve(listener, router).await.unwrap();
}

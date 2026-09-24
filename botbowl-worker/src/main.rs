use std::path::PathBuf;

use clap::Parser;

use botbowl_worker::{run, WorkerConfig, DEFAULT_MEM_FLOOR_MB};

/// Dial a botbowl-hub and play the games it hands out.
#[derive(Parser, Debug)]
#[command(name = "botbowl-worker")]
struct Cli {
    /// Hub websocket URL, e.g. ws://hub.example:7777/ws
    #[arg(long)]
    hub: String,
    /// Shared token (the hub prints it at startup; also `--token-file`).
    #[arg(long, conflicts_with = "token_file")]
    token: Option<String>,
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// Name shown on the hub status page. Defaults to the hostname.
    #[arg(long)]
    name: Option<String>,
    /// Concurrent games. Default: the hub sizes it from cores and RAM.
    #[arg(long)]
    parallel_games: Option<u16>,
    /// GPU sidecar socket (`scripts/nn_server.py`), for the worker on the
    /// training box only.
    #[arg(long)]
    nn_server: Option<PathBuf>,
    /// Model cache directory.
    #[arg(long, default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,
    /// Memory headroom (MB) a game thread keeps in reserve before starting
    /// its next game, on top of a running per-board-cell cost estimate
    /// (see `mem_governor`). Set 0 only for a box with no other memory
    /// pressure to worry about.
    #[arg(long, default_value_t = DEFAULT_MEM_FLOOR_MB)]
    mem_floor_mb: u32,
}

fn default_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/botbowl/models")
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "worker".to_string())
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let token = match (cli.token, cli.token_file) {
        (Some(t), _) => t,
        (None, Some(p)) => std::fs::read_to_string(&p)
            .unwrap_or_else(|e| {
                eprintln!("cannot read {}: {e}", p.display());
                std::process::exit(2)
            })
            .trim()
            .to_string(),
        (None, None) => std::env::var("BOTBOWL_HUB_TOKEN").unwrap_or_else(|_| {
            eprintln!("need --token, --token-file or BOTBOWL_HUB_TOKEN");
            std::process::exit(2)
        }),
    };
    let cfg = WorkerConfig {
        hub_url: cli.hub,
        token,
        name: cli.name.unwrap_or_else(hostname),
        parallel_games: cli.parallel_games,
        nn_server: cli.nn_server,
        cache_dir: cli.cache_dir,
        mem_floor_mb: cli.mem_floor_mb,
    };
    if let Err(e) = run(cfg).await {
        eprintln!("[worker] {e}");
        std::process::exit(1);
    }
}

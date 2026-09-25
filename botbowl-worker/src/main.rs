use std::path::PathBuf;

use clap::Parser;

use botbowl_worker::{run, WorkerConfig, DEFAULT_MEM_FLOOR_MB, DEFAULT_RECONNECT_MAX_SECS};

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
    /// Longest gap between connection attempts, in seconds. The worker starts at 5 s, doubles up
    /// to this, and resets after any connection that worked — so a hub restarted for a new commit
    /// gets its fleet back within this long, without anyone touching the helper boxes.
    #[arg(long, default_value_t = DEFAULT_RECONNECT_MAX_SECS)]
    reconnect_max_secs: u64,
}

/// Since the hub pins a complete `MctsConfig` into every task (`SearchConfig::pinned_to_env`),
/// these no longer do anything here — and someone who set one is expecting otherwise. Say so
/// once, loudly, rather than let them believe a knob is live.
fn warn_about_stale_env() {
    let stale: Vec<String> = std::env::vars()
        .map(|(k, _)| k)
        .filter(|k| k.starts_with("BLOOD_MCTS_"))
        .collect();
    if !stale.is_empty() {
        eprintln!(
            "[worker] WARN: {} set in this environment but ignored — the hub decides every search knob for the games it hands out",
            stale.join(", ")
        );
    }
    // The board is not a search knob: it is checked at the handshake instead, and a mismatch is
    // a rejection rather than a warning.
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
    warn_about_stale_env();
    let cfg = WorkerConfig {
        hub_url: cli.hub,
        token,
        name: cli.name.unwrap_or_else(hostname),
        parallel_games: cli.parallel_games,
        nn_server: cli.nn_server,
        cache_dir: cli.cache_dir,
        mem_floor_mb: cli.mem_floor_mb,
        reconnect_max: std::time::Duration::from_secs(cli.reconnect_max_secs.max(1)),
    };
    if let Err(e) = run(cfg).await {
        eprintln!("[worker] {e}");
        std::process::exit(1);
    }
}

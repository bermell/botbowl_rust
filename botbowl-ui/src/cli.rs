use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "botbowl-ui", about = "Blood Bowl terminal UI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run a live agent-vs-agent game in the terminal.
    Live(LiveArgs),
    /// Replay a previously saved recording.
    Replay(ReplayArgs),
    /// Render one frame to stdout as plain text (deterministic with --seed).
    Snapshot(SnapshotArgs),
}

#[derive(Args, Debug)]
pub struct LiveArgs {
    /// RNG seed for both the game state and bot action selection. When omitted, entropy is used.
    #[arg(long)]
    pub seed: Option<u64>,
    /// If set, write the game's state-by-state recording to this path on exit.
    #[arg(long)]
    pub save: Option<String>,
}

#[derive(Args, Debug)]
pub struct ReplayArgs {
    /// Path to a recording produced by `live --save PATH`.
    pub path: String,
}

#[derive(Args, Debug)]
pub struct SnapshotArgs {
    /// Replay a saved recording at the given step instead of running a fresh seeded game.
    #[arg(long, conflicts_with = "seed")]
    pub replay: Option<String>,
    /// Seed for a fresh agent-vs-agent game (deterministic).
    #[arg(long, conflicts_with = "replay")]
    pub seed: Option<u64>,
    /// How many runner steps (i.e. micro-steps) to advance before rendering. Each call to
    /// `runner.step()` is one unit. Many micro-steps are internal procedure transitions, so this
    /// is fine-grained.
    #[arg(long, default_value_t = 0)]
    pub step: usize,
    /// Terminal size to render at, formatted "WxH". Defaults to 120x40.
    #[arg(long, default_value = "120x40", value_parser = parse_size)]
    pub size: (u16, u16),
}

fn parse_size(s: &str) -> Result<(u16, u16), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WxH, got '{s}'"))?;
    let w: u16 = w.parse().map_err(|e| format!("bad width: {e}"))?;
    let h: u16 = h.parse().map_err(|e| format!("bad height: {e}"))?;
    Ok((w, h))
}

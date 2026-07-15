use botbowl_curriculum::RandomStartConfig;
use clap::{Args, Parser, Subcommand, ValueEnum};

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
    /// Watch a bot play one trial of a curriculum lecture; the UI stops on
    /// pass/fail and leaves the final state on screen.
    Curriculum(CurriculumArgs),
    /// Generate MCTS training data (states + search distributions + values)
    /// and write it as JSONL trajectories. Headless.
    Dataset(DatasetArgs),
    /// Interactively tune random-start placement biases: space generates a
    /// new state, 1-9 select a bias variable, up/down adjust it, q quits.
    Placement(PlacementArgs),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum DatasetMode {
    /// MctsBot vs MctsBot, full games. Samples both teams' decisions.
    #[default]
    SelfPlay,
    /// MctsBot plays a curriculum lecture against a RandomBot opponent.
    Curriculum,
    /// Like self-play, but each game starts from a randomized mid-game state
    /// (biased player placement, random half/turn/score) instead of kickoff.
    RandomStart,
}

/// Placement bias variables for random-start state generation (plan 019).
/// Defaults mirror `RandomStartConfig::default()` — keep them in sync.
#[derive(Args, Debug, Clone, Copy)]
pub struct BiasArgs {
    /// Per-square decay toward the ball (pocket players; line players' y). 1.0 = off.
    #[arg(long, default_value_t = 1.25)]
    pub ball_distance: f32,
    /// Per-square decay toward the team's front column for line players. 1.0 = off.
    #[arg(long, default_value_t = 1.8)]
    pub front_line: f32,
    /// Multiplier for squares adjacent to an already-placed teammate. 1.0 = neutral.
    #[arg(long, default_value_t = 1.5)]
    pub mark_teammate: f32,
    /// Multiplier for squares adjacent to an already-placed opponent. 1.0 = neutral.
    #[arg(long, default_value_t = 1.5)]
    pub mark_opponent: f32,
    /// Multiplier for squares between own endzone and closest opponent. 1.0 = neutral.
    #[arg(long, default_value_t = 1.5)]
    pub own_side: f32,
    /// Sharpens (<1) or flattens (>1) the square distribution.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Probability that the ball starts carried by a player.
    #[arg(long, default_value_t = 0.75)]
    pub carried_prob: f32,
    /// Fraction of each team assigned to the line (front brawl) role.
    #[arg(long, default_value_t = 0.45)]
    pub line_fraction: f32,
    /// Fraction of each team assigned to the pocket (near-ball) role; rest are wide.
    #[arg(long, default_value_t = 0.25)]
    pub pocket_fraction: f32,
}

impl BiasArgs {
    pub fn to_config(&self) -> RandomStartConfig {
        RandomStartConfig {
            ball_distance: self.ball_distance,
            front_line: self.front_line,
            mark_teammate: self.mark_teammate,
            mark_opponent: self.mark_opponent,
            own_side: self.own_side,
            temperature: self.temperature,
            carried_prob: self.carried_prob,
            line_fraction: self.line_fraction,
            pocket_fraction: self.pocket_fraction,
            board_dims: None,
        }
    }
}

#[derive(Args, Debug)]
pub struct PlacementArgs {
    /// Base RNG seed; regeneration `i` uses `seed + i`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[command(flatten)]
    pub bias: BiasArgs,
}

#[derive(Args, Debug)]
pub struct DatasetArgs {
    /// What to generate.
    #[arg(long, value_enum, default_value_t = DatasetMode::SelfPlay)]
    pub mode: DatasetMode,
    /// Output JSONL file; one trajectory per line, appended by default.
    #[arg(long, default_value = "dataset.jsonl")]
    pub out: String,
    /// Truncate the output file before writing instead of appending.
    #[arg(long, default_value_t = false)]
    pub truncate: bool,
    /// Number of games (self-play) or lecture trials (curriculum) to run.
    #[arg(long, default_value_t = 1)]
    pub games: u32,
    /// Base RNG seed; game/trial `i` uses `seed + i`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// MCTS budget: search iterations per move (ignored if --mcts-time-ms set).
    #[arg(long, default_value_t = 1000)]
    pub mcts_iters: usize,
    /// MCTS budget in milliseconds per move; overrides --mcts-iters when set.
    #[arg(long)]
    pub mcts_time_ms: Option<u64>,
    /// Worker threads for the MCTS bot.
    #[arg(long, default_value_t = 1)]
    pub mcts_workers: usize,
    /// Safety cap on micro-steps per game/trial.
    #[arg(long, default_value_t = 100_000)]
    pub max_steps: u32,
    /// (curriculum mode) Lecture name, e.g. "Score TD" (case-insensitive).
    #[arg(long)]
    pub lecture: Option<String>,
    /// (curriculum mode) Lecture difficulty.
    #[arg(long, value_enum, default_value_t = CliDifficulty::Easy)]
    pub difficulty: CliDifficulty,
    /// (random-start mode) Placement bias variables.
    #[command(flatten)]
    pub bias: BiasArgs,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliDifficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(Args, Debug)]
pub struct CurriculumArgs {
    /// Lecture name, e.g. "Score TD" or "Get the ball" (case-insensitive).
    pub name: String,
    /// Lecture difficulty.
    #[arg(long, value_enum)]
    pub difficulty: CliDifficulty,
    /// Bot under test. The opponent (if the lecture has one) is always RandomBot.
    #[arg(long, value_enum, default_value_t = BotKind::Scripted)]
    pub bot: BotKind,
    /// RNG seed for setup, opponent, and bot.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Maximum micro_steps before the trial is declared a timeout.
    #[arg(long, default_value_t = 2000)]
    pub max_steps: u32,
    /// Search iterations per move if --bot mcts.
    #[arg(long, default_value_t = 1000)]
    pub mcts_iters: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum BotKind {
    #[default]
    Random,
    Scripted,
    Mcts,
}

#[derive(Args, Debug)]
pub struct LiveArgs {
    /// RNG seed for both the game state and bot action selection. When omitted, entropy is used.
    #[arg(long)]
    pub seed: Option<u64>,
    /// If set, write the game's state-by-state recording to this path on exit.
    #[arg(long)]
    pub save: Option<String>,
    /// Which bot controls the Home team.
    #[arg(long, value_enum, default_value_t = BotKind::Random)]
    pub home_bot: BotKind,
    /// Which bot controls the Away team.
    #[arg(long, value_enum, default_value_t = BotKind::Random)]
    pub away_bot: BotKind,
    /// Search iterations per move for any MCTS bot in play.
    #[arg(long, default_value_t = 1000)]
    pub mcts_iters: usize,
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
    /// Which bot controls the Home team for the seeded game.
    #[arg(long, value_enum, default_value_t = BotKind::Random)]
    pub home_bot: BotKind,
    /// Which bot controls the Away team for the seeded game.
    #[arg(long, value_enum, default_value_t = BotKind::Random)]
    pub away_bot: BotKind,
    /// Search iterations per move for any MCTS bot in play.
    #[arg(long, default_value_t = 1000)]
    pub mcts_iters: usize,
}

fn parse_size(s: &str) -> Result<(u16, u16), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WxH, got '{s}'"))?;
    let w: u16 = w.parse().map_err(|e| format!("bad width: {e}"))?;
    let h: u16 = h.parse().map_err(|e| format!("bad height: {e}"))?;
    Ok((w, h))
}

use clap::{Args, Parser, Subcommand, ValueEnum};

// The `dataset` / `eval` / `mc-label` flags live in `botbowl-play` (feature `cli`) so `botbowl-hub
// job generate` / `job eval` / `job label` flatten the very same structs (plan 059 #8, plan 062).
pub use botbowl_play::cli_args::{
    BiasArgs, CliCandidateBot, CliDifficulty, CliEvaluator, DatasetArgs, DatasetMode, EvalArgs, EvalSizeArgs,
    McLabelArgs, SizeArgs,
};

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
    /// Evaluate a bot: lecture battery + fixed-opponent ladder (plan 020).
    Eval(EvalArgs),
    /// Interactively tune random-start placement biases: space generates a
    /// new state, 1-9 select a bias variable, up/down adjust it, q quits.
    Placement(PlacementArgs),
    /// Measure how the search output converges with iteration budget, to
    /// justify `--mcts-iters` (plan 025). Headless, read-only.
    Convergence(ConvergenceArgs),
    /// Write a frozen position set for drive rungs (plan 051): `seeds` only, before screening.
    /// Screen it with a reference self-play drive rung and `scripts/positions_screen.py`.
    Positions(PositionsArgs),
    /// Plan 055 phase 2: replay corpus decisions where the search overrules its policy, and
    /// play each move's drive out with policy-only on both sides (paired dice) for a Monte Carlo
    /// value that does not depend on the value head. Headless; summarise with
    /// `scripts/override_audit_summary.py`.
    OverrideAudit(OverrideAuditArgs),
    /// Plan 056: score a net's value head on a frozen Monte Carlo value benchmark
    /// (`scripts/value_bench_freeze.py`): replay each state, one forward, no search. Summarise with
    /// `scripts/value_bench_summary.py`.
    ValueBench(ValueBenchArgs),
    /// Plan 056 arm F: rewrite a corpus with Monte Carlo value labels, each sample's
    /// `outcome_value` the mean of `--playouts` policy-only drive playouts from its state.
    McLabel(McLabelArgs),
    /// Plan 061 (d): replay random-start trajectories under their own dice and count every roll
    /// the engine resolved, by kind (`botbowl_mcts::chance_stats::RollKind`), per drive and per
    /// decision, plus where real ball bounces could go. Headless, read-only.
    RollCensus(RollCensusArgs),
}

/// `roll-census` (plan 061). See `roll_census.rs`.
#[derive(clap::Args, Debug, Clone)]
pub struct RollCensusArgs {
    /// Trajectory shards (`dataset` output). Random-start trajectories replay from their seed; a
    /// `--next-drive` follow-on record (`drive` 2) continues from the record before it, so its
    /// setup and kickoff rolls count too. Anything else is counted as skipped.
    #[arg(long, required = true, num_args = 1..)]
    pub corpus: Vec<String>,
    /// Stop after this many replayed trajectories (0 = all).
    #[arg(long, default_value_t = 0)]
    pub max_trajectories: usize,
    /// Optional JSON summary.
    #[arg(long)]
    pub out: Option<String>,
}

/// `override-audit` (plan 055 §3 phase 2). See `override_audit.rs` for what each row holds.
#[derive(clap::Args, Debug, Clone)]
pub struct OverrideAuditArgs {
    /// Random-start trajectory shards to sample decisions from.
    #[arg(long, required = true, num_args = 1..)]
    pub corpus: Vec<String>,
    /// The net: the search's evaluator and the policy both sides play the drives out with.
    #[arg(long)]
    pub model: String,
    /// Inference sidecar socket (`scripts/nn_server.py`); env fallback `BLOOD_NN_SERVER`.
    #[arg(long)]
    pub nn_server: Option<String>,
    /// The search whose picks are audited: a deterministic eval preset.
    #[arg(long, default_value = "cfgs/gumbel16_f1000.toml")]
    pub search_config: std::path::PathBuf,
    #[arg(long, default_value_t = 1000)]
    pub search_iters: usize,
    /// Stop after this many override rows (every row under `--all`).
    #[arg(long, default_value_t = 1000)]
    pub decisions: u32,
    /// Drive playouts per audited move.
    #[arg(long, default_value_t = 64)]
    pub playouts: u32,
    /// Share of non-override decisions kept as the control.
    #[arg(long, default_value_t = 0.2)]
    pub control_frac: f64,
    /// Keep every decision, override or not.
    #[arg(long)]
    pub all: bool,
    /// Sampling order, control draws and playout dice all derive from it.
    #[arg(long, default_value_t = 55_000)]
    pub seed: u64,
    /// Decisions audited at once, one thread each. With `--nn-server` they share the GPU's batches.
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    /// Only these playable boards, `14x7` (any team size) or `14x7/4`; comma-separated or repeated.
    #[arg(long = "board", value_delimiter = ',')]
    pub boards: Vec<String>,
    /// Only decisions with at least this many legal moves (1 cannot be overruled).
    #[arg(long, default_value_t = 2)]
    pub min_fan: usize,
    /// Safety cap on engine steps per playout.
    #[arg(long, default_value_t = 100_000)]
    pub max_steps: u32,
    /// Output JSONL, one row per kept decision.
    #[arg(long)]
    pub out: String,
}

/// `value-bench` (plan 056 §2). See `value_bench.rs`.
#[derive(clap::Args, Debug, Clone)]
pub struct ValueBenchArgs {
    /// The frozen benchmark: JSONL, one state per line (`corpus`, 1-based `line`, `sample`, `mc`, ...).
    #[arg(long)]
    pub bench: String,
    /// The net whose value head is scored.
    #[arg(long)]
    pub model: String,
    /// Inference sidecar socket (`scripts/nn_server.py`); env fallback `BLOOD_NN_SERVER`.
    #[arg(long)]
    pub nn_server: Option<String>,
    /// States scored at once, one thread each.
    #[arg(long, default_value_t = 4)]
    pub parallel: usize,
    /// Output JSONL: every benchmark line with `v` (the net's V(s), mover's frame) and `model` added.
    #[arg(long)]
    pub out: String,
}

/// `positions`: the candidate positions of a drive-rung set on one board.
#[derive(clap::Args, Debug, Clone)]
pub struct PositionsArgs {
    /// Playable board, `14x7` or `14x7/4`.
    #[arg(long)]
    pub board: String,
    #[arg(long, default_value_t = botbowl_play::board_sizes::DEFAULT_CELLS_PER_PLAYER)]
    pub cells_per_player: f64,
    /// Positions to keep.
    #[arg(long, default_value_t = 500)]
    pub count: u32,
    /// First seed tried. Keep far from corpus seeds (the loop uses 10_000_000 + ...).
    #[arg(long, default_value_t = 70_000_000)]
    pub seed_base: u64,
    /// Skip positions where the side to move has fewer turns than this left in the half: the
    /// clock, not the bots, would end the drive.
    #[arg(long, default_value_t = botbowl_play::drives::MIN_TURNS_LEFT)]
    pub min_turns_left: u8,
    /// Set name for rung labels; defaults to the output file's stem.
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub out: String,
}

/// Re-search the same random-start states at a ladder of iteration budgets and
/// dump the raw per-child search stats for offline analysis (plan 025).
#[derive(clap::Args, Debug, Clone)]
pub struct ConvergenceArgs {
    /// Number of distinct random-start states to probe.
    #[arg(long, default_value_t = 50)]
    pub states: u32,
    /// Independent repeats per (state, budget) cell. >= 2 is required for the
    /// run-to-run noise floor that makes the result interpretable.
    #[arg(long, default_value_t = 3)]
    pub repeats: u32,
    /// Strictly increasing iteration budgets; the largest is the reference.
    #[arg(long, default_value = "100,200,500,1000,2000,4000,8000,16000")]
    pub budgets: String,
    /// Base seed for state generation. Keep far from corpus seeds
    /// (the loop uses 10_000_000 + gen*1e6 + shard*1e5).
    #[arg(long, default_value_t = 90_000_000)]
    pub seed: u64,
    /// Worker threads per search. Keep at 1 to match generation.
    #[arg(long, default_value_t = 1)]
    pub mcts_workers: usize,
    /// Leaf-value source; use the same one generation uses.
    #[arg(long, value_enum, default_value_t = CliEvaluator::Heuristic)]
    pub evaluator: CliEvaluator,
    /// ONNX model for --evaluator nn/nn-value.
    #[arg(long)]
    pub model: Option<String>,
    /// Output JSONL path.
    #[arg(long, default_value = "convergence.jsonl")]
    pub out: String,
    /// PUCT selection rule: `raw` (shipped) or `normalised` (plan 026).
    #[arg(long, default_value = "raw")]
    pub puct_mode: String,
    /// PUCT exploration constant. Defaults per mode: 10 for `raw`, 1 for
    /// `normalised` (the scales are not comparable — see PuctMode).
    #[arg(long)]
    pub puct_c: Option<f32>,
    /// Range floor for `--puct-mode normalised`.
    #[arg(long)]
    pub puct_range_floor: Option<f32>,
    /// Play this many decisions with a production bot (1000 iterations,
    /// same evaluator) from the random start before probing, so the probed
    /// root is a mid-turn state — e.g. 1 turns an activation root (fan
    /// ≈ 4 × players + 1) into a move fan of 30-100 squares (plan 032 #7).
    /// 0 (default) probes the random start itself.
    #[arg(long, default_value_t = 0)]
    pub advance: u32,
    /// Skip probed roots with fewer legal actions than this, so a run can
    /// target the wide-fan regime specifically. Skipped states still consume
    /// their seed slot; raise --states to compensate.
    #[arg(long, default_value_t = 0)]
    pub min_legal: usize,
    /// Playable board for the random starts (`14x7/4`); defaults to the env board.
    #[arg(long)]
    pub board: Option<String>,
    /// States probed at once, each on its own thread. With `--nn-server` they share the GPU's
    /// batches; one stream on a batching server is slower than tract.
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    /// Inference sidecar socket (`scripts/nn_server.py`); env fallback `BLOOD_NN_SERVER`.
    #[arg(long)]
    pub nn_server: Option<String>,
    /// A bot preset (`cfgs/*.toml`) instead of the `--puct-*` flags, e.g. plan 053's
    /// `cfgs/gumbel16_iters.toml`. Its name joins the selection rule in each row's `puct` field.
    #[arg(long, conflicts_with_all = ["puct_c", "puct_range_floor"])]
    pub bot_config: Option<std::path::PathBuf>,
    /// Random-start placement biases (defaults match generation).
    #[command(flatten)]
    pub bias: BiasArgs,
}

#[derive(Args, Debug)]
pub struct PlacementArgs {
    /// Base RNG seed; regeneration `i` uses `seed + i`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[command(flatten)]
    pub bias: BiasArgs,
}

/// Resolve the inference-sidecar socket: the `--nn-server` flag, else the
/// `BLOOD_NN_SERVER` env var (the repo's `BLOOD_*` convention), else
/// `None` — which keeps tract-on-CPU the default everywhere.
pub fn nn_server_path(flag: Option<&str>) -> Option<std::path::PathBuf> {
    flag.map(str::to_string)
        .or_else(|| std::env::var("BLOOD_NN_SERVER").ok())
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
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

// ---- CLI enum -> `botbowl-play` config mappings (plan 041 phase 0) ----
//
// The CLI keeps its own `ValueEnum` types so `botbowl-play` stays clap-free;
// these are the only place the two vocabularies meet.

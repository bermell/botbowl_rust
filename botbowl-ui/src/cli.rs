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
    /// Evaluate a bot: lecture battery + fixed-opponent ladder (plan 020).
    Eval(EvalArgs),
    /// Interactively tune random-start placement biases: space generates a
    /// new state, 1-9 select a bias variable, up/down adjust it, q quits.
    Placement(PlacementArgs),
    /// Measure how the search output converges with iteration budget, to
    /// justify `--mcts-iters` (plan 025). Headless, read-only.
    Convergence(ConvergenceArgs),
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
    /// Random-start placement biases (defaults match generation).
    #[command(flatten)]
    pub bias: BiasArgs,
}

/// Leaf-value source for the MCTS bot during data generation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum CliEvaluator {
    /// Shaped scripted heuristic (`leaf_score`: score + ball control +
    /// carrier distance).
    #[default]
    Heuristic,
    /// Pure touchdown reward (-1/0/+1 on the drive's score change, no
    /// shaping). For small boards where a TD fits inside the search horizon.
    PureTd,
    /// Frozen ONNX network for leaf values and priors (requires --model).
    Nn,
    /// Hybrid diagnostic: NN leaf values, scripted priors (requires --model).
    NnValue,
}

/// Which bot plays the *candidate* seat in `eval`. Defaults to the MCTS bot
/// the report card was built for; `scripted` / `random` exist to take the
/// search out of the picture entirely (plan 023's side-bias ladder).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum CliCandidateBot {
    #[default]
    Mcts,
    Scripted,
    Random,
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
    #[arg(long, default_value_t = 1.30)]
    pub ball_distance: f32,
    /// Per-square decay toward the team's front column for line players. 1.0 = off.
    #[arg(long, default_value_t = 2.20)]
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
    #[arg(long, default_value_t = 0.60)]
    pub temperature: f32,
    /// Second temperature: every other game uses this instead of
    /// --temperature, so the corpus mixes sharp and flat placements.
    /// Set equal to --temperature to disable the alternation.
    #[arg(long, default_value_t = 1.5)]
    pub temperature2: f32,
    /// Probability that the ball starts carried by a player.
    #[arg(long, default_value_t = 0.75)]
    pub carried_prob: f32,
    /// Fraction of each team assigned to the line (front brawl) role.
    #[arg(long, default_value_t = 0.80)]
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
    /// Games to play concurrently in this process (plan 024 Stage 4).
    ///
    /// Games are independent — own state, own bots, own seed — so this
    /// changes nothing about the search; it only raises how many
    /// inference requests are in flight at once, which is what a batched
    /// `--nn-server` needs to fill a batch. Prefer more shard processes
    /// when RAM allows; use this when it does not, or in a single-process
    /// phase. Output line order stops matching game order above 1.
    #[arg(long, default_value_t = 1)]
    pub parallel_games: u32,
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
    /// Leaf-value source for the MCTS bot.
    #[arg(long, value_enum, default_value_t = CliEvaluator::Heuristic)]
    pub evaluator: CliEvaluator,
    /// Path to a frozen ONNX model (required with --evaluator nn).
    #[arg(long)]
    pub model: Option<String>,
    /// Unix socket of a batched inference sidecar (`scripts/nn_server.py`,
    /// plan 024). Unset (the default) means tract on the CPU, exactly as
    /// before; falls back to tract if the server is unreachable. Env
    /// fallback: BLOOD_NN_SERVER (the repo's BLOOD_* convention).
    #[arg(long)]
    pub nn_server: Option<String>,
}

/// Report-card evaluation of one candidate bot (plan 020).
#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Leaf-value source for the candidate MCTS bot.
    #[arg(long, value_enum, default_value_t = CliEvaluator::Heuristic)]
    pub evaluator: CliEvaluator,
    /// Path to a frozen ONNX model (required with --evaluator nn/nn-value).
    #[arg(long)]
    pub model: Option<String>,
    /// Unix socket of a batched inference sidecar (`scripts/nn_server.py`,
    /// plan 024). Unset (the default) means tract on the CPU, exactly as
    /// before; falls back to tract if the server is unreachable. Env
    /// fallback: BLOOD_NN_SERVER (the repo's BLOOD_* convention).
    #[arg(long)]
    pub nn_server: Option<String>,
    /// Candidate search iterations per move.
    #[arg(long, default_value_t = 1000)]
    pub mcts_iters: usize,
    /// Worker threads for MCTS bots (candidate and ladder opponent).
    #[arg(long, default_value_t = 1)]
    pub mcts_workers: usize,
    /// Ladder games to play concurrently within a rung (plan 024 Stage 4b).
    ///
    /// Rung games are independent — own state, own bots, own seed derived
    /// from the game index — so this changes nothing about a result. It is
    /// what makes the eval phase, previously the loop's one wholly serial
    /// phase, use the machine; and it is a precondition for pointing
    /// `--nn-server` at eval, since a single stream is *slower* on a
    /// batching server than on tract. Note a rung holds two bots per
    /// worker (candidate + opponent), so memory grows about twice as fast
    /// per unit as `dataset --parallel-games`.
    #[arg(long, default_value_t = 1)]
    pub parallel_games: u32,
    /// Trials per lecture × difficulty cell.
    #[arg(long, default_value_t = 100)]
    pub trials: u32,
    /// Games per ladder opponent (half as Home, half as Away).
    #[arg(long, default_value_t = 50)]
    pub games: u32,
    /// Games for the `--vs-evaluator` rung only; defaults to `--games`.
    ///
    /// That rung is the only one a promotion gate reads, and it is the one
    /// that needs the most games: draws are commonest against a near-equal
    /// opponent (gen01 measured 23% vs the champion against 0–13% vs the
    /// fixed rungs), which is exactly where the score is noisiest. The
    /// fixed rungs are diagnostic and already decisive at 30 games
    /// (p < 0.01), so raising them too would multiply the cost of the
    /// cheapest information in the report card.
    #[arg(long)]
    pub vs_games: Option<u32>,
    /// Base seed: lecture trials and game pairs are derived from it, so two
    /// candidates run with the same seed face identical situations.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Safety cap on micro-steps per ladder game.
    #[arg(long, default_value_t = 100_000)]
    pub max_steps: u32,
    /// MCTS budget for the mcts-heuristic ladder rung (defaults to --mcts-iters).
    #[arg(long)]
    pub opponent_iters: Option<usize>,
    /// Skip the lecture battery.
    #[arg(long, default_value_t = false)]
    pub skip_lectures: bool,
    /// Skip the opponent ladder.
    #[arg(long, default_value_t = false)]
    pub skip_ladder: bool,
    /// Extra ladder rung: an arbitrary MCTS opponent with this leaf-value
    /// source (e.g. the previous generation's net, for promotion gates).
    /// Runs at --opponent-iters (defaults to --mcts-iters).
    #[arg(long, value_enum)]
    pub vs_evaluator: Option<CliEvaluator>,
    /// ONNX model for the --vs-evaluator opponent (required for nn/nn-value).
    #[arg(long)]
    pub vs_model: Option<String>,
    /// Candidate PUCT selection rule: `raw` or `normalised` (plan 026).
    #[arg(long, default_value = "raw")]
    pub puct_mode: String,
    /// Candidate PUCT exploration constant (default: 10 raw / 1 normalised).
    #[arg(long)]
    pub puct_c: Option<f32>,
    /// Opponent PUCT rule; defaults to the candidate's. Set this to run a
    /// selection-rule head-to-head in one process.
    #[arg(long)]
    pub vs_puct_mode: Option<String>,
    /// Opponent PUCT constant; defaults to the candidate's.
    #[arg(long)]
    pub vs_puct_c: Option<f32>,
    /// Candidate search horizon, in own-turns of lookahead. 1 (default) is
    /// the historical horizon: the search stops once the bot's next turn
    /// begins, i.e. one own-turn plus the opponent's reply. 2 sees a
    /// further turn-pair. A score always stays terminal at any depth.
    #[arg(long, default_value_t = 1)]
    pub horizon_turns: u8,
    /// Opponent search horizon; defaults to the candidate's. Set this to
    /// run a horizon head-to-head in one process.
    #[arg(long)]
    pub vs_horizon_turns: Option<u8>,
    /// Skip the fixed rungs (random/scripted/mcts-heuristic), keeping only
    /// the --vs-evaluator rung. E.g. mirror matches and promotion gates.
    #[arg(long, default_value_t = false)]
    pub skip_fixed_rungs: bool,
    /// Which of the fixed rungs to run, comma-separated
    /// (`random`, `scripted`, `mcts-heuristic`). Lets a mirror match run
    /// exactly one rung instead of the whole ladder.
    #[arg(long, default_value = "random,scripted,mcts-heuristic")]
    pub rungs: String,
    /// Which bot fills the candidate seat. `scripted`/`random` remove the
    /// search from the picture (plan 023).
    #[arg(long, value_enum, default_value_t = CliCandidateBot::Mcts)]
    pub candidate_bot: CliCandidateBot,
    /// Write the report as JSON here.
    #[arg(long)]
    pub out: Option<String>,
    /// Append one JSON line per ladder game here: seed, candidate side,
    /// side-relative scores and who kicked off in half 1 (plan 023).
    #[arg(long)]
    pub per_game_out: Option<String>,
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

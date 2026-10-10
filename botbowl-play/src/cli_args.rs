//! The `dataset` / `eval` command-line flags, shared by `botbowl-ui` and `botbowl-hub` (plan 059
//! #8). Behind the `cli` feature so the library itself stays clap-free for the worker and the web
//! server.
//!
//! `botbowl-ui dataset` / `eval` flatten [`DatasetArgs`] / [`EvalArgs`] whole; `botbowl-hub job
//! generate` / `job eval` flatten the same structs and add their own hub flags (`--hub`,
//! `--out-dir`, `--shards`, `--batch`, `--wait`, ...). The hub refuses the process-local flags it
//! cannot honour (`--parallel-games`, `--nn-server`, `--trials`, ...) instead of ignoring them.
//! `botbowl-ui mc-label` and `botbowl-hub job label` share [`McLabelArgs`] the same way (plan 062).
//! One definition is the point: the two copies had drifted into a panic (the ui's `vs:` label on
//! a one-sided preset) and three conventions for the random-start defaults.

use std::path::PathBuf;

use botbowl_curriculum::RandomStartConfig;
use clap::{Args, ValueEnum};

use crate::bots::{resolve_puct, NamedConfig, SearchConfig};
use crate::generate::RandomStartBias;

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
/// Defaults are [`RandomStartBias::default()`], which reads `RandomStartConfig::default()`: one
/// source for the ui, the hub and the generator.
#[derive(Args, Debug, Clone, Copy)]
pub struct BiasArgs {
    /// Per-square decay toward the ball (pocket players; line players' y). 1.0 = off.
    #[arg(long, default_value_t = RandomStartBias::default().ball_distance)]
    pub ball_distance: f32,
    /// Per-square decay toward the team's front column for line players. 1.0 = off.
    #[arg(long, default_value_t = RandomStartBias::default().front_line)]
    pub front_line: f32,
    /// Multiplier for squares adjacent to an already-placed teammate. 1.0 = neutral.
    #[arg(long, default_value_t = RandomStartBias::default().mark_teammate)]
    pub mark_teammate: f32,
    /// Multiplier for squares adjacent to an already-placed opponent. 1.0 = neutral.
    #[arg(long, default_value_t = RandomStartBias::default().mark_opponent)]
    pub mark_opponent: f32,
    /// Multiplier for squares between own endzone and closest opponent. 1.0 = neutral.
    #[arg(long, default_value_t = RandomStartBias::default().own_side)]
    pub own_side: f32,
    /// Sharpens (<1) or flattens (>1) the square distribution.
    #[arg(long, default_value_t = RandomStartBias::default().temperature)]
    pub temperature: f32,
    /// Second temperature: every other game uses this instead of
    /// --temperature, so the corpus mixes sharp and flat placements.
    /// Set equal to --temperature to disable the alternation.
    #[arg(long, default_value_t = RandomStartBias::default().temperature2)]
    pub temperature2: f32,
    /// Probability that the ball starts carried by a player.
    #[arg(long, default_value_t = RandomStartBias::default().carried_prob)]
    pub carried_prob: f32,
    /// Fraction of each team assigned to the line (front brawl) role.
    #[arg(long, default_value_t = RandomStartBias::default().line_fraction)]
    pub line_fraction: f32,
    /// Fraction of each team assigned to the pocket (near-ball) role; rest are wide.
    #[arg(long, default_value_t = RandomStartBias::default().pocket_fraction)]
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

/// Plan 042: which boards a run plays on. Unset = the process's env board
/// (`BOARD_SIZE_*`), exactly as before. Playable dims throughout.
#[derive(Args, Debug, Clone)]
pub struct SizeArgs {
    /// Comma-separated playable boards to draw each game from, each `WxH`,
    /// `WxH/T` (explicit team size) or with a `:weight` suffix, e.g.
    /// `12x5,14x7:3,16x9/6`. Team size defaults to the density rule
    /// (`--cells-per-player`). Every entry must fit the compiled capacity.
    #[arg(long)]
    pub board_sizes: Option<String>,
    /// Playable cells per fielded player when a size names no team
    /// (`round(w*h / this)`, clamped to `[2, capacity]`). Plan 017's tiers
    /// sit at 25-35.
    #[arg(long, default_value_t = crate::board_sizes::DEFAULT_CELLS_PER_PLAYER)]
    pub cells_per_player: f64,
    /// Centred distribution instead of a list: the playable area to centre
    /// on. Weights every legal board in the aspect band by a log-normal in
    /// area around this, then mixes in `--size-floor` of uniform.
    #[arg(long, conflicts_with = "board_sizes")]
    pub size_centre: Option<f64>,
    /// Std-dev of `ln(area / centre)`. 0 = the nearest legal area only;
    /// large = uniform over the band.
    #[arg(long, default_value_t = 0.3)]
    pub size_temperature: f64,
    /// Share of games drawn uniformly over every legal board regardless of
    /// the centre, so no size ever leaves the corpus.
    #[arg(long, default_value_t = 0.2)]
    pub size_floor: f64,
    /// Aspect band `min-max` (playable width / height) the centred grid keeps.
    #[arg(long, default_value = "1.5-2.8")]
    pub size_aspect: String,
    /// Largest playable area the centred grid enumerates; default = capacity.
    #[arg(long)]
    pub size_max_area: Option<f64>,
    /// Smallest playable area the centred grid enumerates; default = no bound.
    /// Plan 042 E0 measured every board below 70 as a 2v2 in which the
    /// champion scored *less* than a scripted mirror, so `--size-min-area 70`
    /// is how a run keeps only boards where skill expresses.
    #[arg(long)]
    pub size_min_area: Option<f64>,
}

impl SizeArgs {
    /// Resolve to a distribution, or `None` for the env board. Errors name
    /// the flag so a bad size fails before the first game.
    pub fn to_dist(&self) -> Result<Option<crate::board_sizes::SizeDist>, String> {
        use crate::board_sizes::{CentredSpec, SizeDist};
        if let Some(list) = &self.board_sizes {
            return SizeDist::parse_list(list, self.cells_per_player)
                .map(Some)
                .map_err(|e| format!("--board-sizes: {e}"));
        }
        let Some(centre) = self.size_centre else {
            return Ok(None);
        };
        let (lo, hi) = self
            .size_aspect
            .split_once('-')
            .and_then(|(a, b)| Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?)))
            .ok_or_else(|| format!("--size-aspect: expected `min-max`, got {:?}", self.size_aspect))?;
        SizeDist::centred(&CentredSpec {
            centre_area: centre,
            temperature: self.size_temperature,
            floor: self.size_floor,
            aspect_min: lo,
            aspect_max: hi,
            cells_per_player: self.cells_per_player,
            min_area: self.size_min_area,
            max_area: self.size_max_area,
        })
        .map(Some)
        .map_err(|e| format!("--size-centre: {e}"))
    }
}

/// The eval ladder's board set: a fixed list, each board its own rung.
#[derive(Args, Debug, Clone)]
pub struct EvalSizeArgs {
    /// Comma-separated playable boards to run every rung on, e.g.
    /// `12x5,14x7,16x9` (`WxH` or `WxH/T`). Each rung is then named
    /// `<opponent>@<board>` and reported per board. Unset = the env board
    /// and the historical rung names.
    #[arg(long)]
    pub board_sizes: Option<String>,
    /// Team size for a board that names none: `round(w*h / this)`.
    #[arg(long, default_value_t = crate::board_sizes::DEFAULT_CELLS_PER_PLAYER)]
    pub cells_per_player: f64,
}

impl EvalSizeArgs {
    /// `None` = one env-board rung set; `Some(boards)` = one per board.
    pub fn boards(&self) -> Result<Vec<Option<botbowl_engine::core::model::BoardDims>>, String> {
        match &self.board_sizes {
            None => Ok(vec![None]),
            Some(list) => Ok(crate::board_sizes::SizeDist::parse_list(list, self.cells_per_player)
                .map_err(|e| format!("--board-sizes: {e}"))?
                .boards()
                .map(Some)
                .collect()),
        }
    }
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
    // `--seed-base` is the hub's spelling (shard K adds `K * --shard-seed-stride`); hidden so
    // the ui's help is unchanged.
    #[arg(long, default_value_t = 0, alias = "seed-base")]
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
    /// Bot preset: a TOML `MctsConfig` (plan 043). Unset keeps the historical behaviour —
    /// `dataset` leaves every search knob at `MctsBot`'s env-driven default. Setting it replaces
    /// that configuration wholesale and stamps the preset's name into each trajectory's
    /// provenance. See `cfgs/README.md`.
    #[arg(long)]
    pub bot_config: Option<PathBuf>,
    /// Plan 048: root Dirichlet noise weight ε in self-play (0.25 is the AlphaZero value). Unset
    /// keeps the greedy generator. Generation only; eval has no such flag.
    #[arg(long)]
    pub explore_noise: Option<f32>,
    /// Plan 048: total Dirichlet concentration α; each root action gets α / n_legal.
    #[arg(long, default_value_t = 10.0)]
    pub explore_alpha: f32,
    /// Plan 048: each side plays its first K moves of a trajectory ∝ visits^(1/T), not best-Q.
    #[arg(long, default_value_t = 0)]
    pub explore_sample_moves: u32,
    /// Plan 048: the sampling temperature T for `--explore-sample-moves`.
    #[arg(long, default_value_t = 1.0)]
    pub explore_temperature: f32,
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
    /// (random-start mode) When the drive scores, play the next drive too — both kickoff
    /// setups, the kick and its turns — and write it as a second record (plan 047). This is
    /// how per-player setup decisions reach the corpus.
    #[arg(long, default_value_t = false)]
    pub next_drive: bool,
    /// (self-play / random-start) Board-size distribution (plan 042).
    #[command(flatten)]
    pub sizes: SizeArgs,
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

/// `mc-label` (plan 056 arm F): Monte Carlo value labels, see [`crate::mc_label`]. `botbowl-hub
/// job label` flattens it too and refuses `--parallel` and `--nn-server` (workers size themselves
/// and own their sidecar).
#[derive(Args, Debug, Clone)]
pub struct McLabelArgs {
    /// Random-start trajectory shards; each is written to `--out-dir` under its own name.
    #[arg(long, alias = "in", required = true, num_args = 1..)]
    pub corpus: Vec<String>,
    /// The net whose policy plays both sides of every playout.
    #[arg(long)]
    pub model: String,
    /// Inference sidecar socket (`scripts/nn_server.py`); env fallback `BLOOD_NN_SERVER`.
    #[arg(long)]
    pub nn_server: Option<String>,
    /// Playouts averaged per sample.
    #[arg(long, default_value_t = 8)]
    pub playouts: u32,
    /// Playout dice derive from it, the trajectory's seed and the sample index.
    #[arg(long, default_value_t = 56_000)]
    pub seed: u64,
    /// Trajectories labelled at once, one thread each.
    #[arg(long, default_value_t = 8)]
    pub parallel: usize,
    /// Safety cap on engine steps per playout.
    #[arg(long, default_value_t = 100_000)]
    pub max_steps: u32,
    #[arg(long)]
    pub out_dir: String,
}

impl McLabelArgs {
    /// What every label is a function of, besides the net and the state.
    pub fn config(&self) -> crate::mc_label::LabelConfig {
        crate::mc_label::LabelConfig {
            playouts: self.playouts,
            seed: self.seed,
            max_steps: self.max_steps,
        }
    }
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
    /// Plan 051: stop a rung once a sequential test on its mirrored pairs decides,
    /// `S0:S1[:ALPHA:BETA]` (alpha and beta default to 0.05), e.g. `0.5:0.55`. Every ladder rung,
    /// each board's included, runs its own test, and `--games` / `--vs-games` become caps. Unset:
    /// a fixed game count, exactly as before.
    #[arg(long, value_parser = parse_sprt)]
    pub sprt: Option<crate::stats::Sprt>,
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
    /// Candidate bot preset: a TOML `MctsConfig` (plan 043). The file stem names the
    /// configuration and is stamped into the rung label and `report.json`, so a result can be
    /// traced back to what produced it. A preset replaces the bot's whole configuration —
    /// including anything `BLOOD_MCTS_*` would have said — so it is exclusive with the per-knob
    /// flags below. See `cfgs/README.md`.
    #[arg(
        long,
        conflicts_with_all = ["puct_mode", "puct_c", "horizon_turns", "fpu_reduction"]
    )]
    pub bot_config: Option<PathBuf>,
    /// Per-decision tree-reuse trace, as JSONL (plan 043). Off by default: `report.json` always
    /// carries the reuse rates broken down by procedure, and this is for the next question —
    /// which concrete actions a procedure with an odd miss rate was facing. Appends.
    #[arg(long)]
    pub trace_reuse: Option<PathBuf>,
    /// Opponent bot preset; defaults to the candidate's. Set this alone to run a
    /// configuration head-to-head: the same net under two configurations.
    #[arg(
        long,
        conflicts_with_all = ["vs_puct_mode", "vs_puct_c", "vs_horizon_turns", "vs_fpu_reduction"]
    )]
    pub vs_config: Option<PathBuf>,
    /// Candidate PUCT selection rule: `raw` or `normalised` (plan 026). Unset (and no
    /// `--puct-c`): the bot's own rule, i.e. `BLOOD_MCTS_PUCT_*` else raw — as in `dataset`.
    #[arg(long)]
    pub puct_mode: Option<String>,
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
    /// Candidate FPU reduction `k` in Q points (plan 032 #3): unexplored
    /// children are estimated at `parent_Q − k·√(visited prior share)`.
    /// 0 (default) is the shipped plain-FPU behaviour.
    #[arg(long, default_value_t = 0.0)]
    pub fpu_reduction: f32,
    /// Opponent FPU reduction; defaults to the candidate's.
    #[arg(long)]
    pub vs_fpu_reduction: Option<f32>,
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
    /// Boards to run the ladder on (plan 042).
    #[command(flatten)]
    pub sizes: EvalSizeArgs,
    /// Plan 051: play every rung as paired drives from these position sets (comma-separated
    /// files from `positions` + `scripts/positions_screen.py`), one rung per set on the set's own
    /// board, instead of full games. `--board-sizes` is ignored. `--games` counts drives.
    #[arg(long)]
    pub positions: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliDifficulty {
    Easy,
    Medium,
    Hard,
}

fn parse_sprt(s: &str) -> Result<crate::stats::Sprt, String> {
    crate::stats::Sprt::parse(s)
}

impl From<CliEvaluator> for crate::bots::Evaluator {
    fn from(e: CliEvaluator) -> Self {
        use crate::bots::Evaluator as E;
        match e {
            CliEvaluator::Heuristic => E::Heuristic,
            CliEvaluator::PureTd => E::PureTd,
            CliEvaluator::Nn => E::Nn,
            CliEvaluator::NnValue => E::NnValue,
        }
    }
}

impl From<CliCandidateBot> for crate::bots::CandidateBot {
    fn from(c: CliCandidateBot) -> Self {
        use crate::bots::CandidateBot as C;
        match c {
            CliCandidateBot::Mcts => C::Mcts,
            CliCandidateBot::Scripted => C::Scripted,
            CliCandidateBot::Random => C::Random,
        }
    }
}

impl From<DatasetMode> for crate::generate::GenMode {
    fn from(m: DatasetMode) -> Self {
        use crate::generate::GenMode as G;
        match m {
            DatasetMode::SelfPlay => G::SelfPlay,
            DatasetMode::Curriculum => G::Curriculum,
            DatasetMode::RandomStart => G::RandomStart,
        }
    }
}

impl From<CliDifficulty> for botbowl_curriculum::Difficulty {
    fn from(d: CliDifficulty) -> Self {
        use botbowl_curriculum::Difficulty as D;
        match d {
            CliDifficulty::Easy => D::Easy,
            CliDifficulty::Medium => D::Medium,
            CliDifficulty::Hard => D::Hard,
        }
    }
}

impl BiasArgs {
    pub fn to_bias(&self) -> crate::generate::RandomStartBias {
        crate::generate::RandomStartBias {
            ball_distance: self.ball_distance,
            front_line: self.front_line,
            mark_teammate: self.mark_teammate,
            mark_opponent: self.mark_opponent,
            own_side: self.own_side,
            temperature: self.temperature,
            temperature2: self.temperature2,
            carried_prob: self.carried_prob,
            line_fraction: self.line_fraction,
            pocket_fraction: self.pocket_fraction,
        }
    }
}

impl EvalArgs {
    /// The candidate's search knobs. `horizon_turns` and `fpu_reduction` are always `Some` (the
    /// CLI defaults stand in for the bot's); `puct` is `None` unless `--puct-mode`/`--puct-c` is
    /// given, so `BLOOD_MCTS_PUCT_*` applies exactly as in `dataset`.
    ///
    /// Plan 043: `--bot-config` replaces all of them with a named preset. The per-knob flags are
    /// `conflicts_with` it in clap, so the two can never be mixed — a run is described entirely
    /// by a preset or entirely by flags. `Err` names the flag, so a multi-hour head-to-head
    /// refuses to start rather than run the wrong arm.
    pub fn candidate_search(&self, preset: Option<&NamedConfig>) -> Result<SearchConfig, String> {
        Ok(SearchConfig {
            budget: botbowl_mcts::SearchBudget::Iterations(self.mcts_iters),
            workers: self.mcts_workers,
            puct: match preset {
                Some(_) => None,
                None => {
                    resolve_puct(self.puct_mode.as_deref(), self.puct_c).map_err(|e| format!("--puct-mode: {e}"))?
                }
            },
            horizon_turns: preset.is_none().then_some(self.horizon_turns),
            fpu_reduction: preset.is_none().then_some(self.fpu_reduction),
            config: preset.map(|p| p.config),
        })
    }

    /// The opponent's search knobs. Unset `--vs-*` means "match the candidate", so setting one
    /// flag alone makes it a head-to-head on that knob.
    ///
    /// `--vs-config` follows the same rule: unset, the opponent inherits the candidate's preset
    /// (the caller passes it), so `--bot-config` alone configures both sides and `--vs-config`
    /// alone is a configuration head-to-head — the same net under two configurations.
    pub fn opponent_search(&self, preset: Option<&NamedConfig>) -> Result<SearchConfig, String> {
        Ok(SearchConfig {
            budget: botbowl_mcts::SearchBudget::Iterations(self.opponent_iters.unwrap_or(self.mcts_iters)),
            workers: self.mcts_workers,
            puct: match preset {
                Some(_) => None,
                None => resolve_puct(
                    self.vs_puct_mode.as_deref().or(self.puct_mode.as_deref()),
                    self.vs_puct_c.or(self.puct_c),
                )
                .map_err(|e| format!("--vs-puct-mode: {e}"))?,
            },
            horizon_turns: preset
                .is_none()
                .then(|| self.vs_horizon_turns.unwrap_or(self.horizon_turns)),
            fpu_reduction: preset
                .is_none()
                .then(|| self.vs_fpu_reduction.unwrap_or(self.fpu_reduction)),
            config: preset.map(|p| p.config),
        })
    }
}

/// The `--vs-evaluator` rung's label: how the opponent differs from the candidate. Under a preset
/// the per-knob fields are deliberately `None` — the configuration *name* is the difference, and
/// it is the thing you can look up in `cfgs/`. Four cases, none of them a panic (a one-sided
/// preset used to unwrap a `None` in the ui copy). `base` is `evaluator_label(vs, vs_model)`.
/// Downstream scripts match on this text, so the two-preset and no-preset forms are pinned.
pub fn vs_rung_label(
    base: &str,
    cand_preset: Option<&NamedConfig>,
    opp_preset: Option<&NamedConfig>,
    cand: &SearchConfig,
    opp: &SearchConfig,
) -> String {
    match (opp_preset, cand_preset) {
        (Some(o), Some(c)) if o.name == c.name => format!("vs:{base} [{}]", o.name),
        (Some(o), Some(c)) => format!("vs:{base} [{} v {}]", o.name, c.name),
        (Some(o), None) => format!("vs:{base} [{o_name} v flags]", o_name = o.name),
        (None, Some(c)) => format!("vs:{base} [flags v {c_name}]", c_name = c.name),
        (None, None) => {
            let horizon = match (opp.horizon_turns, cand.horizon_turns) {
                (Some(o), Some(c)) if o != c => format!(" horizon={o}v{c}"),
                _ => String::new(),
            };
            let fpu = match (opp.fpu_reduction, cand.fpu_reduction) {
                (Some(o), Some(c)) if o != c => format!(" fpu_k={o}v{c}"),
                _ => String::new(),
            };
            format!("vs:{base} [{}{horizon}{fpu}]", opp.effective_puct().label())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct EvalCli {
        #[command(flatten)]
        args: EvalArgs,
    }

    fn preset(name: &str) -> NamedConfig {
        NamedConfig {
            name: name.into(),
            config: botbowl_mcts::MctsConfig::new(),
        }
    }

    #[test]
    fn a_one_sided_preset_labels_without_panicking() {
        let a = EvalCli::parse_from(["eval", "--vs-config", "x.toml", "--vs-evaluator", "heuristic"]).args;
        let opp_p = preset("policy_only");
        let cand = a.candidate_search(None).unwrap();
        let opp = a.opponent_search(Some(&opp_p)).unwrap();
        assert_eq!(
            vs_rung_label("heuristic", None, Some(&opp_p), &cand, &opp),
            "vs:heuristic [policy_only v flags]"
        );
        let cand_p = preset("gumbel16_f1000");
        let cand = a.candidate_search(Some(&cand_p)).unwrap();
        let opp = a.opponent_search(None).unwrap();
        assert_eq!(
            vs_rung_label("heuristic", Some(&cand_p), None, &cand, &opp),
            "vs:heuristic [flags v gumbel16_f1000]"
        );
    }

    #[test]
    fn the_flag_label_is_unchanged() {
        let a = EvalCli::parse_from(["eval", "--puct-mode", "raw", "--vs-horizon-turns", "2"]).args;
        let (cand, opp) = (a.candidate_search(None).unwrap(), a.opponent_search(None).unwrap());
        assert_eq!(
            vs_rung_label("nn:x.onnx", None, None, &cand, &opp),
            "vs:nn:x.onnx [puct=raw(c=10) horizon=2v1]"
        );
    }

    #[test]
    fn bias_defaults_are_the_generators() {
        #[derive(Parser)]
        struct BiasCli {
            #[command(flatten)]
            bias: BiasArgs,
        }
        assert_eq!(BiasCli::parse_from(["x"]).bias.to_bias(), RandomStartBias::default());
    }
}

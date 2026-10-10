//! JSON types of the hub's local control API (`botbowl-hub job ...` talks
//! to `botbowl-hub serve` with these). Distinct from the worker wire
//! protocol: here models are *paths* the hub resolves and hashes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use botbowl_hub_proto::{BoardDims, Capacity, Evaluator, GenerateConfig, LabelConfig, SearchConfig};
use botbowl_play::eval::Report;

/// `POST /api/jobs` body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobRequest {
    Eval(EvalJobRequest),
    Generate(GenerateJobRequest),
    /// Plan 062: Monte Carlo value labels (`botbowl-ui mc-label` on the workers).
    Label(LabelJobRequest),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BotReq {
    Random,
    Scripted,
    Mcts {
        search: SearchConfig,
        evaluator: Evaluator,
        /// ONNX path, readable by the hub. Required iff the evaluator needs a net.
        model: Option<PathBuf>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RungReq {
    /// Already board-suffixed on a multi-size ladder (`scripted@14x7/4`,
    /// via `botbowl_play::eval::rung_name`).
    pub name: String,
    pub games: u32,
    pub opponent: BotReq,
    /// Plan 042: the board this rung plays on; `None` = the env board.
    #[serde(default)]
    pub board: Option<BoardDims>,
    /// Plan 051: a drive rung's position set; `None` plays full games.
    #[serde(default)]
    pub drives: Option<botbowl_play::drives::DriveRung>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalJobRequest {
    pub candidate: BotReq,
    /// Self-describing label for `report.candidate` (built by the CLI with
    /// `botbowl_play::bots::candidate_label`, same as `botbowl-ui eval`).
    pub candidate_label: String,
    /// Plan 043: the named preset each side plays under, for `report.json`. The knobs themselves
    /// ride in each `BotReq`'s `SearchConfig`; this is the name they are known by.
    #[serde(default)]
    pub candidate_config: Option<String>,
    #[serde(default)]
    pub opponent_config: Option<String>,
    pub mcts_iters: usize,
    pub rungs: Vec<RungReq>,
    pub seed: u64,
    pub max_steps: u32,
    /// Appended to, one `EvalGameLine` per game, as `botbowl-ui eval --per-game-out`.
    pub per_game_out: PathBuf,
    /// `report.json`, written when the job completes.
    pub report_out: PathBuf,
    /// Games per task handed to a worker.
    pub batch: u16,
    /// Plan 051: every rung runs this sequential test and stops once it is decided. The hub
    /// alone uses it, so the worker protocol does not change.
    #[serde(default)]
    pub sprt: Option<botbowl_play::stats::Sprt>,
    /// What the job is called when we talk about it (`gen03 drives vs gen21`), for the status
    /// page. `None` falls back to the job kind.
    #[serde(default)]
    pub label: Option<String>,
}

/// One corpus shard of a generate job: `games` trajectories with seeds
/// `seed + g`, appended to `out`, all with the same configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardReq {
    /// Progress label, e.g. `shard3`.
    pub name: String,
    pub out: PathBuf,
    pub seed: u64,
    pub games: u32,
    /// `cfg.model` is the path *string* as the user typed it: it is stamped
    /// into the corpus provenance and must match what `botbowl-ui dataset`
    /// would have written. `model_path` is where the hub reads the bytes.
    pub cfg: GenerateConfig,
    pub model_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateJobRequest {
    pub shards: Vec<ShardReq>,
    /// Truncate each shard file at submit (`botbowl-ui dataset --truncate`);
    /// otherwise append.
    pub truncate: bool,
    /// Games per task handed to a worker.
    pub batch: u16,
    /// As [`EvalJobRequest::label`] (`gen03 generate`).
    #[serde(default)]
    pub label: Option<String>,
}

/// One input shard of a label job, labelled into `out` exactly as `botbowl-ui mc-label` would
/// (`out.partial` while it is written, renamed when complete). A shard whose `out` already exists
/// at submit is skipped, as the local tool skips it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelShardReq {
    /// Progress label, e.g. `shard3`.
    pub name: String,
    /// The corpus shard, readable by the hub. Its path as given is what the summary line prints.
    pub input: PathBuf,
    pub out: PathBuf,
}

/// Plan 062: `botbowl-ui mc-label`, distributed. The hub reads and compresses the shards, splits
/// each trajectory into work items of at most `chunk_samples` samples, workers return the labels,
/// and the hub writes each shard in the local tool's format and order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelJobRequest {
    pub shards: Vec<LabelShardReq>,
    /// The net whose policy plays every playout, readable by the hub.
    pub model_path: PathBuf,
    /// The model as the user typed it: stamped into `meta.extra.value_label`, as locally.
    pub model: String,
    pub cfg: LabelConfig,
    /// Most samples per work item; a longer trajectory is split into even ranges.
    pub chunk_samples: u32,
    /// Work items per task handed to a worker.
    pub batch: u16,
    /// As [`EvalJobRequest::label`] (`gen03 mc-label`).
    #[serde(default)]
    pub label: Option<String>,
}

pub type JobId = u64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    Running,
    Done,
    Failed { error: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobKind {
    Eval,
    Generate,
    Label,
}

/// Progress of one rung (eval) or one shard (generate, label: `done`/`total` count work items).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitProgress {
    pub name: String,
    pub done: u32,
    pub total: u32,
    /// Samples written so far (generate only; 0 for eval).
    pub samples: u64,
    #[serde(default)]
    pub stats: Option<UnitStats>,
}

/// What a unit's finished games say, beyond how many there are.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnitStats {
    Eval(EvalStats),
    Generate(GenStats),
    Label(LabelStats),
}

/// A label job's shard so far: what `botbowl-ui mc-label`'s summary line says of it, so `job label
/// --wait` can print the same line ([`botbowl_play::mc_label::summary_line`]).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LabelStats {
    pub input: String,
    pub out: String,
    pub trajectories: u32,
    /// Trajectories found not to replay so far: written back unlabelled.
    pub unlabelled: u32,
    /// Samples labelled so far (all of the labellable ones once written).
    pub samples: u64,
    /// `out` existed at submit, so the shard was left alone.
    pub skipped: bool,
    pub written: bool,
    /// From the job's start to the shard written.
    pub secs: f64,
}

/// A rung so far, from the candidate's side.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvalStats {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub tds_for: u32,
    pub tds_against: u32,
    /// The candidate's own decisions (MCTS searches), summed; 0 when it does not search.
    pub decisions: u64,
    /// Plan 051: the paired score's standard error, the pairs it rests on, and the rung's
    /// sequential test, if it runs one.
    #[serde(default)]
    pub points_se: f64,
    #[serde(default)]
    pub pairs: u32,
    #[serde(default)]
    pub sprt: Option<botbowl_play::stats::SprtStatus>,
}

impl EvalStats {
    pub fn games(&self) -> u32 {
        self.wins + self.draws + self.losses
    }

    /// `(W + D/2) / N`, the number every summary script reads.
    pub fn points(&self) -> f64 {
        (self.wins as f64 + self.draws as f64 / 2.0) / self.games().max(1) as f64
    }
}

/// Drives written so far. A random-start trajectory is one drive, so `scored / drives` is the
/// corpus TD rate `td_rate.py` reports and `size_curriculum.py` steers the board centre by.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenStats {
    pub drives: u32,
    pub scored: u32,
    pub tds: u32,
    /// Recorded decisions (samples), both sides.
    pub steps: u64,
    /// The same counts per playable board (`14x7/4`).
    pub by_board: BTreeMap<String, DriveStats>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DriveStats {
    pub drives: u32,
    pub scored: u32,
    pub steps: u64,
}

impl GenStats {
    pub fn add(&mut self, board: String, tds: u32, steps: u64) {
        let scored = u32::from(tds > 0);
        self.drives += 1;
        self.scored += scored;
        self.tds += tds;
        self.steps += steps;
        let b = self.by_board.entry(board).or_default();
        b.drives += 1;
        b.scored += scored;
        b.steps += steps;
    }

    pub fn merge(&mut self, o: &GenStats) {
        self.drives += o.drives;
        self.scored += o.scored;
        self.tds += o.tds;
        self.steps += o.steps;
        for (k, v) in &o.by_board {
            let b = self.by_board.entry(k.clone()).or_default();
            b.drives += v.drives;
            b.scored += v.scored;
            b.steps += v.steps;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub id: JobId,
    pub kind: JobKind,
    #[serde(default)]
    pub label: Option<String>,
    pub state: JobState,
    pub units: Vec<UnitProgress>,
    pub elapsed_secs: u64,
    /// Workers connected to the hub right now; `job --wait` warns when a
    /// running job has none.
    pub workers_connected: usize,
    pub report: Option<Report>,
    /// What each worker (by name) contributed to this job: the share the status page shows.
    #[serde(default)]
    pub by_worker: BTreeMap<String, Counts>,
    /// This job's results over the last few minutes (the status page's rate and ETA).
    #[serde(default)]
    pub recent: Option<Rate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub name: String,
    pub parallel_games: u16,
    pub tasks_in_flight: usize,
    pub games_done: u64,
    pub triple: String,
    pub cores: u16,
    pub ram_mb: u32,
    pub last_seen_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubStatus {
    pub commit: String,
    pub dirty: bool,
    /// The board capacity this hub (and every worker it accepts) is
    /// compiled for — echoed on the status page so a new worker knows
    /// what `BOARD_SIZE_W`/`BOARD_SIZE_H`/`BOARD_PLAYERS` to build with.
    pub capacity: Capacity,
    pub workers: Vec<WorkerStatus>,
    pub jobs: Vec<JobStatus>,
    /// Games and decisions per minute, per worker and in total ([`crate::rates`]). Defaulted so
    /// a `status` client and a hub one commit apart still read each other.
    #[serde(default)]
    pub throughput: Throughput,
}

/// Results counted by the hub from what workers already send (no protocol change): a generate
/// result is one game, the records it wrote (two when `--next-drive` followed a score) and its
/// samples; an eval result is one game and the candidate's searches. Additive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Generate games (one `TrajectoryDone` each).
    pub games: u64,
    /// Corpus records (lines) those games wrote.
    pub records: u64,
    /// Samples written: recorded decisions, both sides.
    pub samples: u64,
    /// Eval games (full games or drives).
    #[serde(default)]
    pub eval_games: u64,
    /// The eval candidate's decisions (MCTS searches); the opponent's are not reported.
    #[serde(default)]
    pub eval_decisions: u64,
    /// Label work items (plan 062), and the samples they labelled.
    #[serde(default)]
    pub label_items: u64,
    #[serde(default)]
    pub label_samples: u64,
}

impl Counts {
    pub fn add(&mut self, o: &Counts) {
        self.games += o.games;
        self.records += o.records;
        self.samples += o.samples;
        self.eval_games += o.eval_games;
        self.eval_decisions += o.eval_decisions;
        self.label_items += o.label_items;
        self.label_samples += o.label_samples;
    }

    pub fn is_empty(&self) -> bool {
        *self == Counts::default()
    }

    /// Units of work of any kind: generate and eval games, label items.
    pub fn all_games(&self) -> u64 {
        self.games + self.eval_games + self.label_items
    }
}

/// [`Counts`] over `secs` seconds of wall time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub counts: Counts,
    pub secs: f64,
}

impl Rate {
    /// `n` events per minute over this span; 0 for an empty span.
    pub fn per_min(&self, n: u64) -> f64 {
        if self.secs <= 0.0 {
            0.0
        } else {
            n as f64 * 60.0 / self.secs
        }
    }
}

/// One worker *name* (a reconnect is a new connection under the same name), or the total.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkerRates {
    pub name: String,
    /// Connections under this name right now; 0 = gone (listed while it has results in the
    /// history).
    pub connected: u32,
    pub streams: u32,
    /// Streams holding a task.
    pub busy: u32,
    /// One per [`Throughput::windows`]: the last N seconds, or since it joined if that is later
    /// (so a worker that joined two minutes ago is not reported at 2/5 of its speed).
    pub windows: Vec<Rate>,
    /// Since the worker first joined this hub (the total: since the first worker did).
    pub since_start: Rate,
    /// Complete intervals of [`Throughput::bucket_secs`], oldest first, the last one ending at
    /// [`Throughput::history_end`].
    pub history: Vec<Counts>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Throughput {
    pub uptime_secs: u64,
    /// The window lengths of [`WorkerRates::windows`], in seconds (5 and 30 minutes).
    pub windows: Vec<u64>,
    /// The history's interval, in seconds; intervals are aligned to the wall clock.
    pub bucket_secs: u64,
    /// Unix seconds at which the newest complete interval ends.
    pub history_end: u64,
    /// How much of each interval the hub was up for (the first one after a start is partial).
    pub history_secs: Vec<u64>,
    pub workers: Vec<WorkerRates>,
    pub total: WorkerRates,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Submitted {
    pub id: JobId,
}

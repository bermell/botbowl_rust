//! JSON types of the hub's local control API (`botbowl-hub job ...` talks
//! to `botbowl-hub serve` with these). Distinct from the worker wire
//! protocol: here models are *paths* the hub resolves and hashes.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use botbowl_hub_proto::{BoardDims, Evaluator, GenerateConfig, SearchConfig};
use botbowl_play::eval::Report;

/// `POST /api/jobs` body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobRequest {
    Eval(EvalJobRequest),
    Generate(GenerateJobRequest),
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
}

/// Progress of one rung (eval) or one shard (generate).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitProgress {
    pub name: String,
    pub done: u32,
    pub total: u32,
    /// Samples written so far (generate only; 0 for eval).
    pub samples: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub id: JobId,
    pub kind: JobKind,
    pub state: JobState,
    pub units: Vec<UnitProgress>,
    pub elapsed_secs: u64,
    /// Workers connected to the hub right now; `job --wait` warns when a
    /// running job has none.
    pub workers_connected: usize,
    pub report: Option<Report>,
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
    pub workers: Vec<WorkerStatus>,
    pub jobs: Vec<JobStatus>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Submitted {
    pub id: JobId,
}

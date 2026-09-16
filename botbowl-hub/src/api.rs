//! JSON types of the hub's local control API (`botbowl-hub job ...` talks
//! to `botbowl-hub serve` with these). Distinct from the worker wire
//! protocol: here models are *paths* the hub resolves and hashes.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use botbowl_hub_proto::{Evaluator, SearchConfig};
use botbowl_play::eval::Report;

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
    pub name: String,
    pub games: u32,
    pub opponent: BotReq,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalJobRequest {
    pub candidate: BotReq,
    /// Self-describing label for `report.candidate` (built by the CLI with
    /// `botbowl_play::bots::candidate_label`, same as `botbowl-ui eval`).
    pub candidate_label: String,
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

pub type JobId = u64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    Running,
    Done,
    Failed { error: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RungProgress {
    pub name: String,
    pub done: u32,
    pub total: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub id: JobId,
    pub state: JobState,
    pub rungs: Vec<RungProgress>,
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

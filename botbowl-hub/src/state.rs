//! The hub's whole mutable state behind one mutex: connected workers,
//! model bytes, jobs and their queues. Every transition is a synchronous
//! method here; the websocket and HTTP layers only translate.
//!
//! Scheduling is deliberately simple (plan 041 decision 7): free streams go
//! round-robin over the running jobs (plan 047: a generation's eval shares the
//! fleet with the next generation), a task is a small batch of games from one *unit* (a
//! ladder rung of an eval job, a corpus shard of a generate job), a
//! worker holds at most `parallel_games` game tasks plus `label_streams` label tasks (label tasks
//! may also fill idle game streams, plan 069 item 4b), background jobs get only streams no
//! foreground job is waiting for (plan 071 R5), and anything a departed
//! worker had in flight goes back on the queue. Results are deduplicated
//! on `(job, unit, game)` so a slow worker that reappears can never double
//! count.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use botbowl_hub_proto::{
    BoardDims, BotSpec, BuildInfo, EvalGameLine, GenerateConfig, LabelItem, LabelResult, ModelId, Task, TaskId,
    ToWorker,
};
use botbowl_play::board_sizes::board_label;
use botbowl_play::eval::{LadderRow, Report};

use crate::api::{
    BotReq, Counts, EvalJobRequest, EvalStats, GenStats, GenerateJobRequest, HubStatus, JobId, JobKind, JobRequest,
    JobState, JobStatus, LabelJobRequest, LabelStats, Priority, UnitProgress, UnitStats, WorkerStatus,
};
use crate::label::{LabelInput, ShardWrite};
use crate::rates::{Ledger, Live};
use crate::resume::{EvalScan, ShardScan};

pub type WorkerId = u64;

/// How many times one game may fail (on any worker) before the job does.
const MAX_GAME_FAILURES: u32 = 3;

/// While generate games are waiting, eval tasks may hold at most `1 / EVAL_SHARE_DIVISOR` of a
/// worker's streams (at least one). See [`Inner::dispatch`].
const EVAL_SHARE_DIVISOR: usize = 3;

pub struct WorkerConn {
    pub name: String,
    pub build: BuildInfo,
    pub triple: String,
    pub cores: u16,
    pub ram_mb: u32,
    pub parallel_games: u16,
    /// Plan 069 item 4b: streams that take label tasks only (`botbowl-worker --label-streams`).
    pub label_streams: u16,
    pub tx: mpsc::UnboundedSender<ToWorker>,
    /// Models this worker is known to hold (reported cached, or sent by us).
    pub known_models: HashSet<ModelId>,
    /// Models whose `ModelName` this connection has been sent.
    pub named_models: HashSet<ModelId>,
    pub tasks: HashSet<TaskId>,
    pub games_done: u64,
    pub last_seen: Instant,
}

struct Rung {
    name: String,
    total: u32,
    opponent: BotSpec,
    board: Option<BoardDims>,
    drives: Option<botbowl_play::drives::DriveRung>,
    row: LadderRow,
    done: HashSet<u32>,
}

struct Shard {
    name: String,
    out: PathBuf,
    seed: u64,
    total: u32,
    cfg: GenerateConfig,
    model: Option<ModelId>,
    done: HashSet<u32>,
    written: u32,
    samples: u64,
    stats: GenStats,
    writer: io::BufWriter<std::fs::File>,
    /// Plan 069 item 4b: this shard's unit in the job's streamed-label job, if it is labelled.
    label_unit: Option<usize>,
}

/// Plan 062: one shard of a label job. Its "games" are work items ([`crate::label::Item`]).
struct LabelShard {
    input: LabelInput,
    /// Per trajectory, one label per sample, filled in as items arrive.
    labels: Vec<Vec<f32>>,
    /// Per trajectory: some item found it does not replay.
    unlabellable: Vec<bool>,
    done: HashSet<u32>,
    labelled: u64,
    /// Handed to a [`ShardWrite`]; `written` once it is on disk.
    writing: bool,
    written: bool,
    secs: f64,
    /// Plan 069 item 4b: items superseded by a streaming shard's seal (their trajectory's first
    /// drive changed); a result for one is dropped. Never pending, never counted.
    stale: HashSet<u32>,
    /// Items that count (all but the stale ones), and how many of those are in.
    live: usize,
    live_done: usize,
    /// Samples counted into `labelled`, per trajectory (so a relabel can take them back).
    counted: Vec<u64>,
}

impl LabelShard {
    fn new(input: LabelInput) -> Self {
        LabelShard {
            labels: input.samples.iter().map(|&n| vec![0.0; n]).collect(),
            // A top-up's untagged trajectories have no items: unlabelled from the start.
            unlabellable: input.untagged.clone(),
            live: input.items.len(),
            counted: vec![0; input.samples.len()],
            input,
            done: HashSet::new(),
            labelled: 0,
            writing: false,
            written: false,
            secs: 0.0,
            stale: HashSet::new(),
            live_done: 0,
        }
    }

    /// Plan 069 item 4b: one record of a streaming shard, as the generate shard wrote it. Returns
    /// the items to queue.
    fn push(&mut self, line: crate::label::Prepared) -> io::Result<std::ops::Range<u32>> {
        let new = self.input.push(line)?;
        let n = *self.input.samples.last().expect("just pushed");
        self.labels.push(vec![0.0; n]);
        self.unlabellable.push(false);
        self.counted.push(0);
        self.live += new.len();
        Ok(new)
    }

    /// Plan 069 item 4b: the generate shard has all its games. Returns the items to queue (the
    /// deferred and relabelled trajectories') and the ones gone stale (to drop from the queue).
    fn seal(&mut self) -> (Vec<u32>, HashSet<u32>) {
        let (redo, chunk) = self.input.seal();
        let mut fresh = Vec::new();
        let mut staled = HashSet::new();
        for t in redo {
            for (i, it) in self.input.items.iter().enumerate() {
                let i = i as u32;
                if it.0 == t && self.stale.insert(i) {
                    staled.insert(i);
                    self.live -= 1;
                    if self.done.contains(&i) {
                        self.live_done -= 1;
                    }
                }
            }
            self.labels[t].iter_mut().for_each(|l| *l = 0.0);
            self.unlabellable[t] = false;
            self.labelled -= std::mem::take(&mut self.counted[t]);
            let r = self.input.recut(t, chunk);
            self.live += r.len();
            fresh.extend(r);
        }
        (fresh, staled)
    }

    /// Nothing left to do: written, or skipped at submit.
    fn finished(&self) -> bool {
        self.written || self.input.skipped
    }

    /// The write, once every item is in (at most once).
    fn take_write(&mut self, job: JobId, unit: usize) -> Option<ShardWrite> {
        if self.writing || self.input.skipped || self.input.is_open() || self.live_done < self.live {
            return None;
        }
        self.writing = true;
        let labels = std::mem::take(&mut self.labels)
            .into_iter()
            .zip(&self.unlabellable)
            .map(|(l, &bad)| (!bad).then_some(l))
            .collect();
        Some(ShardWrite {
            job,
            unit,
            out: self.input.out.clone(),
            lines: Arc::clone(&self.input.lines),
            labels,
            tag: self.input.tag.clone(),
            prior: self.input.prior.clone(),
        })
    }

    /// Drop the shard's compressed lines once nothing will ship or write them again. The hub keeps
    /// finished jobs for its whole life, and a 16x9 generation's lines are ~100 MB zstd'd.
    fn release(&mut self) {
        self.input.lines = Arc::new(Vec::new());
    }

    fn held_bytes(&self) -> usize {
        self.input.lines.iter().map(Vec::len).sum()
    }

    fn unlabelled(&self) -> usize {
        self.unlabellable.iter().filter(|b| **b).count()
    }

    fn stats(&self) -> LabelStats {
        LabelStats {
            input: self.input.input.display().to_string(),
            out: self.input.out.display().to_string(),
            trajectories: self.input.samples.len() as u32,
            unlabelled: self.unlabelled() as u32,
            samples: self.labelled,
            skipped: self.input.skipped,
            written: self.written,
            secs: self.secs,
        }
    }
}

enum Kind {
    Eval {
        req: EvalJobRequest,
        candidate: BotSpec,
        rungs: Vec<Rung>,
        per_game: io::BufWriter<std::fs::File>,
        report: Option<Report>,
    },
    Generate {
        shards: Vec<Shard>,
        /// Plan 069 item 4b: the label job its shards stream into.
        labels: Option<JobId>,
    },
    /// Each shard carries its own `LabelConfig` and tag (`LabelInput.cfg` / `.tag`): a top-up
    /// continues every shard's own labelling (plan 067).
    Label { shards: Vec<LabelShard>, model: ModelId },
}

struct InFlight {
    job: JobId,
    unit: usize,
    remaining: HashSet<u32>,
    worker: WorkerId,
}

pub struct Job {
    id: JobId,
    label: Option<String>,
    kind: Kind,
    batch: u16,
    /// `(unit index, game)` not yet handed out.
    pending: VecDeque<(usize, u32)>,
    failures: HashMap<(usize, u32), u32>,
    state: JobState,
    started: Instant,
    /// When it left `Running`, so a finished job's elapsed time stops.
    ended: Option<Instant>,
    /// Accepted results per worker name: each worker's share of the job.
    by_worker: BTreeMap<String, Counts>,
    /// The request as submitted (JSON): the same request submitted while this job runs attaches
    /// to it ([`Inner::attach`]) instead of starting a second writer on its files.
    key: Option<String>,
    /// `POST /api/jobs/{id}/drain`: hand out no more of its games (a deadline is near); results
    /// for games already out are still taken. Cleared when the same request is submitted again.
    draining: bool,
    /// Plan 071 R5: a background job takes only streams no foreground job is waiting for.
    priority: Priority,
}

impl Job {
    fn units(&self) -> Vec<UnitProgress> {
        match &self.kind {
            Kind::Eval { rungs, .. } => rungs
                .iter()
                .map(|r| UnitProgress {
                    name: r.name.clone(),
                    done: r.done.len() as u32,
                    total: r.total,
                    samples: 0,
                    stats: Some(UnitStats::Eval(EvalStats {
                        wins: r.row.wins,
                        draws: r.row.draws,
                        losses: r.row.losses,
                        tds_for: r.row.tds_for,
                        tds_against: r.row.tds_against,
                        decisions: r.row.telemetry.as_ref().map_or(0, |t| t.searches),
                        points_se: r.row.pairs.se(),
                        pairs: r.row.pairs.pairs(),
                        sprt: r.row.sprt,
                    })),
                })
                .collect(),
            Kind::Generate { shards, .. } => shards
                .iter()
                .map(|s| UnitProgress {
                    name: s.name.clone(),
                    done: s.done.len() as u32,
                    total: s.total,
                    samples: s.samples,
                    stats: Some(UnitStats::Generate(s.stats.clone())),
                })
                .collect(),
            Kind::Label { shards, .. } => shards
                .iter()
                .map(|s| UnitProgress {
                    name: s.input.name.clone(),
                    done: s.live_done as u32,
                    total: s.live as u32,
                    samples: s.labelled,
                    stats: Some(UnitStats::Label(s.stats())),
                })
                .collect(),
        }
    }

    /// Every file this job writes.
    fn outputs(&self) -> Vec<&Path> {
        match &self.kind {
            Kind::Eval { req, .. } => vec![req.per_game_out.as_path(), req.report_out.as_path()],
            Kind::Generate { shards, .. } => shards.iter().map(|s| s.out.as_path()).collect(),
            Kind::Label { shards, .. } => shards.iter().map(|s| s.input.out.as_path()).collect(),
        }
    }

    fn unit_name(&self, unit: usize) -> &str {
        match &self.kind {
            Kind::Eval { rungs, .. } => &rungs[unit].name,
            Kind::Generate { shards, .. } => &shards[unit].name,
            Kind::Label { shards, .. } => &shards[unit].input.name,
        }
    }

    /// Every unit has all its games in, or (plan 051) its SPRT is decided. A decided rung's games
    /// still in flight are not waited for.
    fn all_done(&self) -> bool {
        // Plan 062: a label shard is done when it is on disk, not when its last item arrives.
        if let Kind::Label { shards, .. } = &self.kind {
            return shards.iter().all(LabelShard::finished);
        }
        self.units()
            .iter()
            .enumerate()
            .all(|(i, u)| u.done >= u.total || self.unit_decided(i))
    }

    /// Whether `(unit, game)` is still wanted: not a stale label item (plan 069 item 4b).
    fn wants(&self, unit: usize, game: u32) -> bool {
        match &self.kind {
            Kind::Label { shards, .. } => !shards[unit].stale.contains(&game),
            _ => true,
        }
    }

    /// Plan 051: an eval rung whose SPRT has a verdict takes no more games.
    fn unit_decided(&self, unit: usize) -> bool {
        match &self.kind {
            Kind::Eval { rungs, .. } => rungs[unit].row.decided(),
            Kind::Generate { .. } | Kind::Label { .. } => false,
        }
    }

    fn status(&self, workers_connected: usize) -> JobStatus {
        JobStatus {
            id: self.id,
            kind: match self.kind {
                Kind::Eval { .. } => JobKind::Eval,
                Kind::Generate { .. } => JobKind::Generate,
                Kind::Label { .. } => JobKind::Label,
            },
            label: self.label.clone(),
            workers_connected,
            state: self.state.clone(),
            units: self.units(),
            elapsed_secs: self
                .ended
                .unwrap_or_else(Instant::now)
                .duration_since(self.started)
                .as_secs(),
            report: match &self.kind {
                Kind::Eval { report, .. } => report.clone(),
                Kind::Generate { .. } | Kind::Label { .. } => None,
            },
            by_worker: self.by_worker.clone(),
            recent: None,
            labels_job: match &self.kind {
                Kind::Generate { labels, .. } => *labels,
                _ => None,
            },
            hub_instance: 0,
            draining: self.draining,
            in_flight: 0,
            priority: self.priority,
        }
    }

    /// The task for `games` of `unit`, built from this job's configuration.
    fn make_task(&self, id: TaskId, unit: usize, games: Vec<u32>) -> Task {
        match &self.kind {
            Kind::Eval {
                req, candidate, rungs, ..
            } => Task::Eval {
                id,
                rung: rungs[unit].name.clone(),
                games,
                seed: req.seed,
                max_steps: req.max_steps,
                candidate: candidate.clone(),
                opponent: rungs[unit].opponent.clone(),
                board: rungs[unit].board,
                drives: rungs[unit].drives.clone(),
            },
            Kind::Generate { shards, .. } => {
                let s = &shards[unit];
                Task::Generate {
                    id,
                    shard: s.name.clone(),
                    games,
                    seed_base: s.seed,
                    cfg: s.cfg.clone(),
                    model: s.model,
                }
            }
            Kind::Label { shards, model } => {
                let s = &shards[unit].input;
                Task::Label {
                    id,
                    shard: s.name.clone(),
                    items: games
                        .into_iter()
                        .map(|g| {
                            let (t, start, end) = s.items[g as usize];
                            LabelItem {
                                item: g,
                                start,
                                end,
                                zstd_json: s.lines[t].clone(),
                                first_zstd_json: s.firsts[t].map(|j| s.lines[j].clone()),
                            }
                        })
                        .collect(),
                    cfg: s.cfg,
                    model: *model,
                }
            }
        }
    }

    fn fail(&mut self, error: String) {
        eprintln!("[hub] job {} failed: {error}", self.id);
        self.state = JobState::Failed { error };
        self.ended.get_or_insert_with(Instant::now);
        self.pending.clear();
        // Nothing is shipped or written for a failed job (a write in progress holds its own Arc).
        if let Kind::Label { shards, .. } = &mut self.kind {
            shards.iter_mut().for_each(LabelShard::release);
        }
    }
}

/// A `TrajectoryDone` payload, decompressed, checked and peeked at by the websocket task before it
/// takes the hub's one lock (plan 069 item 6). Under the lock these cost ~2 ms a game at 20x11 (a
/// ~370 KB line inflated, then scanned whole by `drive_summary`), which at 4,000 fake streams and
/// 200 games/s held the lock most of the time: dispatch latency p99 1.2 s. Now the lock only
/// writes the lines.
pub struct TrajectoryPayload {
    /// The JSON lines, newline-separated (no trailing newline), or why they cannot be written.
    lines: Result<Vec<u8>, String>,
    /// `drive_summary` of each line, in order.
    summaries: Vec<Option<(String, u32)>>,
    /// Plan 069 item 4b: each line compressed and read for a streamed label shard, when the hub
    /// streams labels (done here, off the lock; the lock prepares any line that lacks it).
    prepared: Option<Vec<crate::label::Prepared>>,
    /// `meta.seed` of each line, in order (`None` when unreadable or absent).
    seeds: Vec<Option<u64>>,
}

impl TrajectoryPayload {
    pub fn decode(zstd_json: &[u8]) -> Self {
        Self::decode_for(zstd_json, false)
    }

    /// [`Self::decode`], also compressing each line on its own when `per_line` (a hub with a
    /// streamed-label job: the label shard keeps and ships lines one by one).
    pub fn decode_for(zstd_json: &[u8], per_line: bool) -> Self {
        let mut p = Self::decode_lines(zstd_json);
        if per_line {
            if let Ok(json) = &p.lines {
                if !json.is_empty() {
                    p.prepared = json
                        .split(|b| *b == b'\n')
                        .map(crate::label::Prepared::of)
                        .collect::<io::Result<Vec<_>>>()
                        .ok();
                }
            }
        }
        p
    }

    fn decode_lines(zstd_json: &[u8]) -> Self {
        if zstd_json.is_empty() {
            return TrajectoryPayload {
                lines: Ok(Vec::new()),
                summaries: Vec::new(),
                prepared: None,
                seeds: Vec::new(),
            };
        }
        let json = match zstd::decode_all(zstd_json) {
            Ok(json) => json,
            Err(e) => {
                return TrajectoryPayload {
                    lines: Err(format!("zstd: {e}")),
                    summaries: Vec::new(),
                    prepared: None,
                    seeds: Vec::new(),
                }
            }
        };
        if json.split(|b| *b == b'\n').any(|l| l.is_empty()) {
            return TrajectoryPayload {
                lines: Err("trajectory payload has an empty line".into()),
                summaries: Vec::new(),
                prepared: None,
                seeds: Vec::new(),
            };
        }
        let (summaries, seeds) = json.split(|b| *b == b'\n').map(peek_line).unzip();
        TrajectoryPayload {
            lines: Ok(json),
            summaries,
            prepared: None,
            seeds,
        }
    }

    /// The game legitimately produced nothing (an empty `zstd_json`): counted, not written.
    fn nothing(&self) -> bool {
        matches!(&self.lines, Ok(v) if v.is_empty())
    }
}

/// The board and the drive's own touchdowns, read from a trajectory line without keeping its
/// samples. `outcome` is the absolute scoreline; a random-start drive begins at
/// `meta.extra.start_score`, so its touchdowns are the difference (`td_rate.py`'s rule).
#[cfg(test)]
fn drive_summary(json: &[u8]) -> Option<(String, u32)> {
    peek_line(json).0
}

/// [`drive_summary`], and the line's `meta.seed` (which game it is), in one parse.
fn peek_line(json: &[u8]) -> (Option<(String, u32)>, Option<u64>) {
    #[derive(serde::Deserialize)]
    struct Meta {
        board_dims: BoardDims,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        extra: BTreeMap<String, String>,
    }
    #[derive(serde::Deserialize)]
    struct Outcome {
        home_score: u8,
        away_score: u8,
    }
    #[derive(serde::Deserialize)]
    struct Peek {
        meta: Meta,
        outcome: Outcome,
    }
    let Ok(p) = serde_json::from_slice::<Peek>(json) else {
        return (None, None);
    };
    let start = p
        .meta
        .extra
        .get("start_score")
        .and_then(|s| s.split_once('-'))
        .and_then(|(h, a)| Some(h.parse::<u32>().ok()? + a.parse::<u32>().ok()?))
        .unwrap_or(0);
    let end = p.outcome.home_score as u32 + p.outcome.away_score as u32;
    (
        Some((board_label(p.meta.board_dims), end.saturating_sub(start))),
        p.meta.seed,
    )
}

fn open_out(path: &Path, truncate: bool) -> io::Result<io::BufWriter<std::fs::File>> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let f = if truncate {
        std::fs::File::create(path)?
    } else {
        std::fs::File::options().create(true).append(true).open(path)?
    };
    Ok(io::BufWriter::new(f))
}

/// Tell one worker what model `m` is called here.
fn send_name(w: &mut WorkerConn, m: ModelId, path: &Path) {
    let _ = w.tx.send(ToWorker::ModelName {
        id: m,
        name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        source: path.to_string_lossy().into_owned(),
        hub_commit: botbowl_data::git_commit().to_string(),
    });
    w.named_models.insert(m);
}

/// Every `.onnx` under `dirs` (a few levels deep), by content hash. The first path wins.
pub fn index_models(dirs: &[PathBuf]) -> HashMap<ModelId, PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut HashMap<ModelId, PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let path = e.path();
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_dir() {
                if depth > 0 {
                    walk(&path, depth - 1, out);
                }
            } else if path.extension().is_some_and(|x| x == "onnx") {
                if let Ok(bytes) = std::fs::read(&path) {
                    out.entry(ModelId::of(&bytes)).or_insert(path);
                }
            }
        }
    }
    let mut out = HashMap::new();
    for d in dirs {
        walk(d, 6, &mut out);
    }
    out
}

#[derive(Default)]
pub struct Inner {
    pub workers: HashMap<WorkerId, WorkerConn>,
    pub models: HashMap<ModelId, Arc<Vec<u8>>>,
    /// Each model's TorchScript export (plan 066), when one sits beside its ONNX.
    pub torchscripts: HashMap<ModelId, Arc<Vec<u8>>>,
    /// The path each model was first loaded from, for `ToWorker::ModelName`.
    pub model_paths: HashMap<ModelId, PathBuf>,
    /// Every net found on this box at startup (`Hub::index_models`), so the models a worker
    /// already holds can be named on connect, not only the ones a job happens to use.
    pub model_index: HashMap<ModelId, PathBuf>,
    jobs: BTreeMap<JobId, Job>,
    in_flight: HashMap<TaskId, InFlight>,
    next_worker: WorkerId,
    next_job: JobId,
    next_task: TaskId,
    /// Where `dispatch` resumes its round-robin over running jobs.
    next_share: usize,
    /// Every accepted result, for the status page's rates and the hub.log rate line.
    pub ledger: Ledger,
    /// Random, fixed for the process's life ([`Inner::new_instance`]): what tells a client that
    /// the hub it polls is not the one it submitted to.
    pub instance: u64,
}

impl Inner {
    /// Give this hub process its own identity: a random `instance`, and task ids starting at a
    /// random point. A worker that reconnects after a hub restart still sends the results of
    /// games it started for the old hub, under the old hub's task ids; numbered from 0 again,
    /// those ids would name *this* hub's tasks, and a trajectory of one seed would be written
    /// into another shard as another game. From a random 63-bit start they name nothing here
    /// and are dropped like any unknown task (and `trajectory_done` checks the seed anyway).
    pub fn new_instance(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        self.instance = rng.gen_range(1..u64::MAX >> 1);
        self.next_task = rng.gen_range(0..u64::MAX >> 1);
    }

    // -- models ------------------------------------------------------------

    /// Read an ONNX file, hash it, keep the bytes for shipping.
    pub fn load_model(&mut self, path: &Path) -> io::Result<ModelId> {
        let bytes =
            std::fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("model {}: {e}", path.display())))?;
        let id = ModelId::of(&bytes);
        self.models.entry(id).or_insert_with(|| Arc::new(bytes));
        self.model_paths.entry(id).or_insert_with(|| path.to_path_buf());
        // Plan 066: ship the TorchScript export with it when there is one (beside the path, or
        // beside the file it links to), read again on every load so a later export is picked up.
        let ts = [
            Some(path.with_extension("ts")),
            path.canonicalize().ok().map(|p| p.with_extension("ts")),
        ]
        .into_iter()
        .flatten()
        .find(|p| p.is_file());
        if let Some(ts) = ts {
            let b = std::fs::read(&ts).map_err(|e| io::Error::new(e.kind(), format!("model {}: {e}", ts.display())))?;
            self.torchscripts.insert(id, Arc::new(b));
        }
        Ok(id)
    }

    /// Where model `m` came from on this box: a job's path, else the startup index.
    fn model_path(&self, m: &ModelId) -> Option<&PathBuf> {
        self.model_paths.get(m).or_else(|| self.model_index.get(m))
    }

    /// Send `ModelName` for every model a connected worker holds and has not been told the name
    /// of, wherever this box knows it from. Called when a worker connects and when the startup
    /// index finishes, so a cache filled before names existed gets named on its next connect.
    pub fn name_cached_models(&mut self) {
        let mut sends = Vec::new();
        for (wid, w) in &self.workers {
            for m in w.known_models.difference(&w.named_models) {
                if let Some(path) = self.model_path(m) {
                    sends.push((*wid, *m, path.clone()));
                }
            }
        }
        for (wid, m, path) in sends {
            if let Some(w) = self.workers.get_mut(&wid) {
                send_name(w, m, &path);
            }
        }
    }

    fn resolve_bot(&mut self, req: &BotReq) -> io::Result<BotSpec> {
        Ok(match req {
            BotReq::Random => BotSpec::Random,
            BotReq::Scripted => BotSpec::Scripted,
            BotReq::Mcts {
                search,
                evaluator,
                model,
            } => {
                let model = match (evaluator.needs_model(), model) {
                    (true, Some(p)) => Some(self.load_model(p)?),
                    (true, None) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "nn evaluator requires a model path",
                        ))
                    }
                    (false, _) => None,
                };
                BotSpec::Mcts {
                    search: *search,
                    evaluator: *evaluator,
                    model,
                }
            }
        })
    }

    // -- jobs --------------------------------------------------------------

    /// The synchronous path (a caller holding the lock): every file is read inline. `Hub::submit`
    /// reads label inputs and resume scans before taking the lock and writes labels on threads.
    pub fn submit(&mut self, req: JobRequest) -> io::Result<JobId> {
        let key = serde_json::to_string(&req).map_err(io::Error::other)?;
        if let Some(id) = self.attach(&key, &req.outputs())? {
            return Ok(id);
        }
        let id = match req {
            JobRequest::Eval(r) => {
                let scan = r
                    .resume
                    .then(|| crate::resume::scan_eval(&r.per_game_out))
                    .transpose()?;
                self.submit_eval_scanned(r, scan)?
            }
            JobRequest::Generate(r) => {
                let scans = r.resume.then(|| crate::resume::scan_generate(&r)).transpose()?;
                let inputs = crate::label::stream_inputs(&r, scans.as_deref())?;
                let (id, writes) = self.submit_generate_scanned(r, scans, inputs)?;
                self.run_writes(writes);
                id
            }
            JobRequest::Label(r) => {
                let inputs = crate::label::load_inputs(&r)?;
                let (id, writes) = self.submit_label(r, inputs)?;
                self.run_writes(writes);
                id
            }
        };
        self.set_key(id, key);
        Ok(id)
    }

    /// Before a job is created: the same request (`key`, its JSON) already running is that job —
    /// its id is returned, and a drain on it is lifted (the submitter is back for it). Otherwise a
    /// running job that writes any of `outputs` refuses this one: two writers would interleave in
    /// one file, and a resume would cut the other's line in progress.
    ///
    /// This is what makes resubmission safe. After a hub restart `job --wait` submits its request
    /// again, and so may the loop that launched it, an orphaned client of the run before, or the
    /// next Slurm job's loop against a persistent hub: all of them end up on one job.
    pub fn attach(&mut self, key: &str, outputs: &[PathBuf]) -> io::Result<Option<JobId>> {
        // Plan 069 item 4b: a generate job whose games are done but whose streamed labels still
        // run is still that request's job (the client follows it into the labels).
        let same = self
            .jobs
            .values()
            .find(|j| (j.state == JobState::Running || self.labels_running(j)) && j.key.as_deref() == Some(key))
            .map(|j| j.id);
        if let Some(id) = same {
            let job = self.jobs.get_mut(&id).expect("found");
            if std::mem::take(&mut job.draining) {
                eprintln!("[hub] job {id} submitted again: no longer draining");
                self.dispatch();
            } else {
                eprintln!("[hub] job {id} submitted again: attached");
            }
            return Ok(Some(id));
        }
        let mut superseded = Vec::new();
        for j in self.jobs.values().filter(|j| j.state == JobState::Running) {
            if let Some(o) = j.outputs().into_iter().find(|o| outputs.iter().any(|p| p == o)) {
                // A drained job with nothing left on workers was abandoned by its submitter (a
                // deadline, then a resubmission that differs — another environment, another
                // label): it writes nothing more, so it gives way.
                if j.draining && self.games_in_flight(j.id) == 0 {
                    superseded.push(j.id);
                    continue;
                }
                // Plan 069 item 4b: the streamed labels of a generate job that gives way (drained,
                // or already failed) give way with it; their in-flight items are simply dropped.
                if let Some(p) = self.parent_of(j.id) {
                    if self.jobs.get(&p).is_some_and(|pj| {
                        pj.state != JobState::Running && pj.state != JobState::Done
                            || pj.draining && self.games_in_flight(p) == 0
                    }) {
                        superseded.push(j.id);
                        continue;
                    }
                }
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("job {} (another request) is already writing {}", j.id, o.display()),
                ));
            }
        }
        for id in superseded {
            let job = self.jobs.get_mut(&id).expect("found");
            job.fail("drained and superseded by a new request for the same files".into());
            self.propagate_failure(id);
        }
        Ok(None)
    }

    /// Plan 069 item 4b: the generate job whose streamed labels job `lid` is.
    fn parent_of(&self, lid: JobId) -> Option<JobId> {
        self.jobs
            .values()
            .find(|j| matches!(j.kind, Kind::Generate { labels: Some(l), .. } if l == lid))
            .map(|j| j.id)
    }

    /// Plan 069 item 4b: `j` is a generate job whose streamed-label job is still running.
    fn labels_running(&self, j: &Job) -> bool {
        match j.kind {
            Kind::Generate { labels: Some(l), .. } => self.jobs.get(&l).is_some_and(|l| l.state == JobState::Running),
            _ => false,
        }
    }

    /// Games (label items) of job `id` on workers right now.
    fn games_in_flight(&self, id: JobId) -> u32 {
        self.in_flight
            .values()
            .filter(|f| f.job == id)
            .map(|f| f.remaining.len() as u32)
            .sum()
    }

    /// Record the request a new job was made from (see [`Inner::attach`]).
    pub fn set_key(&mut self, id: JobId, key: String) {
        if let Some(j) = self.jobs.get_mut(&id) {
            j.key = Some(key);
        }
    }

    /// Stop handing out a running job's games (`POST /api/jobs/{id}/drain`): `job --wait
    /// --deadline` calls it shortly before a Slurm job's end, so the games already on workers can
    /// finish and be written while nothing new starts that the walltime would cut. `false` for a
    /// job that is not running.
    pub fn drain(&mut self, id: JobId) -> bool {
        let Some(job) = self.jobs.get_mut(&id) else {
            return false;
        };
        if job.state != JobState::Running {
            return false;
        }
        if !job.draining {
            job.draining = true;
            eprintln!(
                "[hub] job {id} draining: {} game(s) not handed out are left for a resume",
                job.pending.len()
            );
        }
        true
    }

    /// Flush every open output, for a clean exit (`serve` on SIGTERM/SIGINT). Every write is
    /// already flushed per result, under this lock; holding it while the process exits is what
    /// guarantees no line is cut halfway.
    pub fn flush_all(&mut self) {
        for job in self.jobs.values_mut() {
            match &mut job.kind {
                Kind::Generate { shards, .. } => shards.iter_mut().for_each(|s| {
                    let _ = s.writer.flush();
                }),
                Kind::Eval { per_game, .. } => {
                    let _ = per_game.flush();
                }
                Kind::Label { .. } => {}
            }
        }
    }

    /// The synchronous path's shard writes: inline, under the caller's lock.
    pub fn run_writes(&mut self, writes: Vec<ShardWrite>) {
        for w in writes {
            let result = w.run();
            self.label_written(w.job, w.unit, result);
        }
    }

    /// Plan 062: a label job over shards already read ([`crate::label::load_inputs`]). Returns
    /// the writes due at once (a shard with no trajectories has nothing to wait for); the caller
    /// runs them and reports each to [`Inner::label_written`].
    pub fn submit_label(
        &mut self,
        req: LabelJobRequest,
        inputs: Vec<LabelInput>,
    ) -> io::Result<(JobId, Vec<ShardWrite>)> {
        let model = self.load_model(&req.model_path)?;
        let mut pending = VecDeque::new();
        for (i, s) in inputs.iter().enumerate() {
            if s.skipped {
                eprintln!("[hub] mc-label: {} exists, skipping", s.out.display());
            } else {
                pending.extend((0..s.items.len() as u32).map(|g| (i, g)));
            }
        }
        let shards = inputs.into_iter().map(LabelShard::new).collect();
        let kind = Kind::Label { shards, model };
        let id = self.insert_job(kind, req.label, req.batch, req.priority, pending);
        let job = self.jobs.get_mut(&id).expect("just inserted");
        let mut writes = Vec::new();
        if let Kind::Label { shards, .. } = &mut job.kind {
            for (unit, s) in shards.iter_mut().enumerate() {
                writes.extend(s.take_write(id, unit));
            }
        }
        if job.all_done() {
            Self::finish(job);
        }
        Ok((id, writes))
    }

    pub fn submit_eval(&mut self, req: EvalJobRequest) -> io::Result<JobId> {
        self.submit(JobRequest::Eval(req))
    }

    /// An eval job, resumed from `scan` (its per-game file as read, [`crate::resume::scan_eval`])
    /// when there is one: the lines are cut to whole ones and folded into the rows in file order,
    /// so each row — its SPRT included — is what it was when the file was last written.
    pub fn submit_eval_scanned(&mut self, req: EvalJobRequest, scan: Option<EvalScan>) -> io::Result<JobId> {
        let candidate = self.resolve_bot(&req.candidate)?;
        let mut rungs = Vec::new();
        let mut pending = VecDeque::new();
        for r in req.rungs.iter() {
            let opponent = self.resolve_bot(&r.opponent)?;
            let gate = r.drives.as_ref().and(req.drive_gate);
            let mut row = LadderRow::new(&r.name).with_sprt(req.sprt).with_gate(gate);
            row.board = r.board.map(board_label);
            rungs.push(Rung {
                name: r.name.clone(),
                total: r.games,
                opponent,
                board: r.board,
                drives: r.drives.clone(),
                row,
                done: HashSet::new(),
            });
        }
        if let Some(scan) = &scan {
            crate::resume::cut(&req.per_game_out, scan.file_len, scan.keep_len)?;
            let (mut kept, mut other) = (0u32, 0u32);
            for line in &scan.lines {
                match rungs.iter_mut().find(|r| r.name == line.rung && line.game < r.total) {
                    Some(r) if !r.done.contains(&line.game) => {
                        r.done.insert(line.game);
                        r.row.record(line);
                        kept += 1;
                    }
                    _ => other += 1,
                }
            }
            eprintln!(
                "[hub] eval resume from {}: {kept} game(s) on disk{}{}",
                req.per_game_out.display(),
                if other > 0 {
                    format!(", {other} line(s) of no rung of this job left as they are")
                } else {
                    String::new()
                },
                if scan.cut.is_empty() {
                    String::new()
                } else {
                    format!(", cut off: {}", scan.cut.join("; "))
                }
            );
        }
        for (i, r) in rungs.iter().enumerate() {
            if !r.row.decided() {
                pending.extend((0..r.total).filter(|g| !r.done.contains(g)).map(|g| (i, g)));
            }
        }
        if let Some(dir) = req.report_out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let per_game = open_out(&req.per_game_out, false)?;
        let batch = req.batch;
        let priority = req.priority;
        let label = req.label.clone();
        let kind = Kind::Eval {
            req,
            candidate,
            rungs,
            per_game,
            report: None,
        };
        Ok(self.insert_finishing(kind, label, batch, priority, pending))
    }

    pub fn submit_generate(&mut self, req: GenerateJobRequest) -> io::Result<JobId> {
        self.submit(JobRequest::Generate(req))
    }

    /// A generate job, resumed from `scans` (one per shard, [`crate::resume::scan_generate`]) when
    /// there are any: each shard is cut to its whole games and only the missing games are queued.
    /// With `req.mc_label` (plan 069 item 4b) also its streamed-label job over `labels`
    /// ([`crate::label::stream_inputs`], read before the lock: a resumed shard's kept lines).
    /// Returns the label writes due at once (a labelled shard with no games left to play); the
    /// caller runs them.
    pub fn submit_generate_scanned(
        &mut self,
        req: GenerateJobRequest,
        scans: Option<Vec<ShardScan>>,
        labels: Vec<LabelInput>,
    ) -> io::Result<(JobId, Vec<ShardWrite>)> {
        if req.resume && req.truncate {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--resume and --truncate exclude each other",
            ));
        }
        if req.shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "generate job with no shards",
            ));
        }
        let label_model = match &req.mc_label {
            Some(m) => Some(self.load_model(&m.model_path)?),
            None => None,
        };
        let mut shards = Vec::new();
        let mut pending = VecDeque::new();
        for (i, s) in req.shards.iter().enumerate() {
            let model = match (s.cfg.evaluator.needs_model(), &s.model_path) {
                (true, Some(p)) => Some(self.load_model(p)?),
                (true, None) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{}: nn evaluator requires a model path", s.name),
                    ))
                }
                (false, _) => None,
            };
            if s.cfg.mode == botbowl_play::generate::GenMode::Curriculum && s.cfg.lecture.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{}: curriculum mode requires a lecture", s.name),
                ));
            }
            let mut done = HashSet::new();
            if let Some(scan) = scans.as_ref().and_then(|v| v.get(i)) {
                crate::resume::cut(&s.out, scan.file_len, scan.keep_len)?;
                done.extend(scan.done.iter().copied());
                eprintln!(
                    "[hub] {}: resuming, {}/{} games on disk ({} records){}{}",
                    s.name,
                    done.len(),
                    s.games,
                    scan.lines,
                    if scan.beyond > 0 {
                        format!(", {} game(s) beyond the target kept", scan.beyond)
                    } else {
                        String::new()
                    },
                    if scan.cut.is_empty() {
                        String::new()
                    } else {
                        format!(", cut off: {}", scan.cut.join("; "))
                    }
                );
            }
            pending.extend((0..s.games).filter(|g| !done.contains(g)).map(|g| (i, g)));
            shards.push(Shard {
                name: s.name.clone(),
                out: s.out.clone(),
                seed: s.seed,
                total: s.games,
                cfg: s.cfg.clone(),
                model,
                done,
                written: 0,
                samples: 0,
                stats: GenStats::default(),
                writer: open_out(&s.out, req.truncate)?,
                label_unit: labels.iter().position(|l| l.name == s.name),
            });
        }
        // Plan 069 item 4b: label outputs made stale by this job's games go now, under the lock
        // (an identical request already running was attached to before this point).
        for l in &labels {
            if l.stale_out {
                eprintln!(
                    "[hub] mc-label: {} is stale (its shard gets more games): removing it, labelling afresh",
                    l.out.display()
                );
                std::fs::remove_file(&l.out).or_else(|e| match e.kind() {
                    io::ErrorKind::NotFound => Ok(()),
                    _ => Err(e),
                })?;
            }
        }
        let labels_job = match (label_model, req.mc_label) {
            (Some(model), Some(m)) => {
                let mut lpending = VecDeque::new();
                for (u, l) in labels.iter().enumerate() {
                    if l.skipped {
                        eprintln!(
                            "[hub] mc-label: {} exists and its shard gets no games, skipping",
                            l.out.display()
                        );
                    } else {
                        lpending.extend((0..l.items.len() as u32).map(|g| (u, g)));
                    }
                }
                let lshards = labels.into_iter().map(LabelShard::new).collect();
                let label = m
                    .label
                    .or_else(|| req.label.as_ref().map(|l| format!("{l} · mc-label (streamed)")));
                Some(self.insert_finishing(
                    Kind::Label { shards: lshards, model },
                    label,
                    m.batch,
                    req.priority,
                    lpending,
                ))
            }
            _ => None,
        };
        let id = self.insert_finishing(
            Kind::Generate {
                shards,
                labels: labels_job,
            },
            req.label,
            req.batch,
            req.priority,
            pending,
        );
        // A labelled shard with every game on disk already (none to play, or a resume that found
        // them all) is sealed now.
        let complete: Vec<usize> = match &self.jobs[&id].kind {
            Kind::Generate { shards, .. } => shards
                .iter()
                .filter(|s| s.done.len() as u32 >= s.total)
                .filter_map(|s| s.label_unit)
                .collect(),
            _ => Vec::new(),
        };
        let mut writes = Vec::new();
        if let Some(lid) = labels_job {
            for unit in complete {
                writes.extend(self.seal_label_shard(lid, unit));
            }
        }
        Ok((id, writes))
    }

    /// [`Inner::insert_job`], finished at once when a resume found nothing left to play.
    fn insert_finishing(
        &mut self,
        kind: Kind,
        label: Option<String>,
        batch: u16,
        priority: Priority,
        pending: VecDeque<(usize, u32)>,
    ) -> JobId {
        let id = self.insert_job(kind, label, batch, priority, pending);
        let job = self.jobs.get_mut(&id).expect("just inserted");
        if job.all_done() {
            Self::finish(job);
        }
        id
    }

    /// Plan 069 item 4b: a streamed label shard gets no more lines. Queues its deferred and
    /// relabelled items, drops stale ones from the queue, and returns its write if it is due.
    fn seal_label_shard(&mut self, lid: JobId, unit: usize) -> Option<ShardWrite> {
        let job = self.jobs.get_mut(&lid)?;
        if job.state != JobState::Running {
            return None;
        }
        let Kind::Label { shards, .. } = &mut job.kind else {
            return None;
        };
        let s = &mut shards[unit];
        if s.input.skipped {
            return None;
        }
        let (fresh, staled) = s.seal();
        if !staled.is_empty() {
            eprintln!(
                "[hub] {}: {} item(s) relabelled at the seal (a first drive changed)",
                s.input.name,
                fresh.len()
            );
        }
        let write = s.take_write(lid, unit);
        job.pending.retain(|&(u, g)| u != unit || !staled.contains(&g));
        job.pending.extend(fresh.into_iter().map(|g| (unit, g)));
        if job.all_done() {
            Self::finish(job);
        }
        self.dispatch();
        write
    }

    /// Plan 069 item 4b: a generate job that failed takes its streamed labels with it (its shards
    /// would never seal).
    fn propagate_failure(&mut self, id: JobId) {
        let Some(job) = self.jobs.get(&id) else { return };
        let (JobState::Failed { error }, Kind::Generate { labels: Some(lid), .. }) = (&job.state, &job.kind) else {
            return;
        };
        let (error, lid) = (error.clone(), *lid);
        if let Some(l) = self.jobs.get_mut(&lid) {
            if l.state == JobState::Running {
                l.fail(format!("its generate job {id} failed: {error}"));
            }
        }
    }

    fn insert_job(
        &mut self,
        kind: Kind,
        label: Option<String>,
        batch: u16,
        priority: Priority,
        pending: VecDeque<(usize, u32)>,
    ) -> JobId {
        let id = self.next_job;
        self.next_job += 1;
        let job = Job {
            id,
            label,
            kind,
            batch,
            pending,
            failures: HashMap::new(),
            state: JobState::Running,
            started: Instant::now(),
            ended: None,
            by_worker: BTreeMap::new(),
            key: None,
            draining: false,
            priority,
        };
        eprintln!(
            "[hub] job {id} submitted{}: {}",
            if priority.is_foreground() { "" } else { " (background)" },
            job.units()
                .iter()
                .map(|u| format!("{} x{}", u.name, u.total))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.jobs.insert(id, job);
        self.dispatch();
        id
    }

    pub fn job_status(&self, id: JobId) -> Option<JobStatus> {
        self.jobs.get(&id).map(|j| self.job_status_of(j, Instant::now()))
    }

    /// A job's status with its recent rate: its own results over the last 5 minutes (or since
    /// it started), for the page's rate and ETA.
    fn job_status_of(&self, j: &Job, now: Instant) -> JobStatus {
        let mut s = j.status(self.workers.len());
        s.hub_instance = self.instance;
        s.in_flight = self.games_in_flight(j.id);
        if j.state == JobState::Running {
            s.recent = Some(
                self.ledger
                    .window(now, crate::rates::WINDOWS[0], j.started, |_, job| job == j.id),
            );
        }
        s
    }

    /// The fleet's throughput, per worker name and in total.
    pub fn throughput(&self, now: Instant) -> crate::api::Throughput {
        let live: Vec<Live> = self
            .workers
            .values()
            .map(|w| Live {
                name: &w.name,
                streams: w.parallel_games as u32 + w.label_streams as u32,
                busy: w.tasks.len() as u32,
            })
            .collect();
        self.ledger.throughput(now, &live)
    }

    /// The `[hub] rate ...` line for the interval that just completed, if anything finished in it.
    pub fn rate_log_line(&self, now: Instant) -> Option<String> {
        crate::rates::log_line(&self.throughput(now))
    }

    fn worker_name(&self, id: WorkerId) -> String {
        self.workers
            .get(&id)
            .map_or_else(|| format!("worker {id}"), |w| w.name.clone())
    }

    pub fn status(&self) -> HubStatus {
        let now = Instant::now();
        HubStatus {
            throughput: self.throughput(now),
            commit: botbowl_data::git_commit().to_string(),
            dirty: botbowl_data::git_dirty(),
            capacity: botbowl_hub_proto::Capacity::compiled(),
            workers: self
                .workers
                .values()
                .map(|w| WorkerStatus {
                    name: w.name.clone(),
                    parallel_games: w.parallel_games,
                    label_streams: w.label_streams,
                    tasks_in_flight: w.tasks.len(),
                    games_done: w.games_done,
                    triple: w.triple.clone(),
                    cores: w.cores,
                    ram_mb: w.ram_mb,
                    last_seen_secs: w.last_seen.elapsed().as_secs(),
                })
                .collect(),
            jobs: self.jobs.values().map(|j| self.job_status_of(j, now)).collect(),
        }
    }

    // -- workers -----------------------------------------------------------

    pub fn add_worker(&mut self, conn: WorkerConn) -> WorkerId {
        let id = self.next_worker;
        self.next_worker += 1;
        eprintln!(
            "[hub] worker {id} {:?} joined ({} streams{}, {} cores, {} MB, {})",
            conn.name,
            conn.parallel_games,
            if conn.label_streams > 0 {
                format!(" + {} label streams", conn.label_streams)
            } else {
                String::new()
            },
            conn.cores,
            conn.ram_mb,
            conn.triple
        );
        self.ledger.joined(&conn.name, Instant::now());
        self.workers.insert(id, conn);
        self.dispatch();
        id
    }

    pub fn remove_worker(&mut self, id: WorkerId) {
        let Some(w) = self.workers.remove(&id) else { return };
        eprintln!(
            "[hub] worker {id} {:?} left with {} task(s) in flight",
            w.name,
            w.tasks.len()
        );
        for t in w.tasks {
            self.requeue(t);
        }
        self.dispatch();
    }

    pub fn seen(&mut self, id: WorkerId) {
        if let Some(w) = self.workers.get_mut(&id) {
            w.last_seen = Instant::now();
        }
    }

    /// Drop every worker silent for longer than `timeout` and requeue what it
    /// held. Returns the ids dropped.
    ///
    /// Worker ids are never reused, so the websocket task's own
    /// `remove_worker` when it finally notices the dead socket is a no-op,
    /// and a reaped worker that comes back to life reconnects as a new id —
    /// results it still sends under the old one are deduped on
    /// `(job, unit, game)` like any other late result, so reaping can
    /// duplicate work but never double-count it.
    pub fn reap_stale(&mut self, timeout: Duration) -> Vec<WorkerId> {
        let stale: Vec<WorkerId> = self
            .workers
            .iter()
            .filter(|(_, w)| w.last_seen.elapsed() > timeout)
            .map(|(id, _)| *id)
            .collect();
        for id in &stale {
            if let Some(w) = self.workers.get(id) {
                eprintln!(
                    "[hub] worker {id} {:?} timed out ({} s without a heartbeat)",
                    w.name,
                    w.last_seen.elapsed().as_secs()
                );
            }
            self.remove_worker(*id);
        }
        stale
    }

    // -- scheduling --------------------------------------------------------

    /// Put a task's unfinished games back at the front of its job's queue.
    fn requeue(&mut self, task: TaskId) {
        let Some(f) = self.in_flight.remove(&task) else { return };
        if let Some(w) = self.workers.get_mut(&f.worker) {
            w.tasks.remove(&task);
        }
        if let Some(job) = self.jobs.get_mut(&f.job) {
            // Plan 051: a decided rung's games are not wanted any more.
            if job.state != JobState::Running || job.unit_decided(f.unit) {
                return;
            }
            for g in f.remaining {
                if job.wants(f.unit, g) {
                    job.pending.push_front((f.unit, g));
                }
            }
        }
    }

    /// Hand out work to every worker with a free stream.
    ///
    /// Free streams go round-robin over every running job with games left, one task at a time, so
    /// concurrent jobs share the fleet: `train_loop.sh` runs a generation's eval alongside the next
    /// generation's games (plan 047 item 0), and handing everything to the oldest job would
    /// serialise them again.
    ///
    /// **Round-robin counts tasks, not stream time**, and an eval task (a full game, often at a
    /// higher budget) holds its stream for minutes where a generate task (one drive) frees it in
    /// seconds. So left alone the eval ends up on nearly every stream: on 2026-09-27 it finished
    /// 213 of its 300 games while the generation beside it managed 132 of 4800. While any generate
    /// game is waiting, eval tasks are therefore capped at [`EVAL_SHARE_DIVISOR`]⁻¹ of a
    /// worker's streams; generation is the critical path, the benchmark is not.
    ///
    /// **Two pools per worker (plan 069 item 4b).** A worker runs `parallel_games` game streams
    /// and `label_streams` label-only streams. Generate and eval tasks never exceed
    /// `parallel_games`. Label tasks fill the label streams and, **while no generate game is
    /// waiting**, any idle game stream too — so a label job alone (the mc-label phase) uses every
    /// stream as before, and labels streamed beside a generation use exactly the L streams sized
    /// for them, never a stream the generation could use. A worker with no label streams (0, the
    /// default) therefore takes a generation's streamed labels only once its games run out.
    ///
    /// **Two tiers (plan 071 R5, [`Priority`]).** The rules above pick among foreground jobs first,
    /// computed from foreground jobs alone. A background job gets a stream only when no foreground
    /// job can take it, by the same rules computed over every job — the rule streamed labels follow
    /// beside a generation, one level up. Background eval and generate tasks need a game stream like
    /// any game task (a worker's label threads run only label tasks); a background label job may
    /// take a label stream the foreground leaves idle. Nothing is preempted: a foreground task that
    /// arrives while background tasks hold every stream takes the next one that frees, after at most
    /// one background task (a game batch).
    pub fn dispatch(&mut self) {
        let job_ids: Vec<JobId> = self
            .jobs
            .values()
            .filter(|j| j.state == JobState::Running && !j.draining && !j.pending.is_empty())
            .map(|j| j.id)
            .collect();
        if job_ids.is_empty() {
            return;
        }
        let worker_ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        for wid in worker_ids {
            loop {
                let w = &self.workers[&wid];
                let streams = w.parallel_games as usize + w.label_streams as usize;
                if w.tasks.len() >= streams {
                    break;
                }
                let kind_of = |t: &TaskId| {
                    self.in_flight
                        .get(t)
                        .and_then(|f| self.jobs.get(&f.job))
                        .map(|j| match j.kind {
                            Kind::Eval { .. } => 0,
                            Kind::Generate { .. } => 1,
                            Kind::Label { .. } => 2,
                        })
                };
                let labels_here = w.tasks.iter().filter(|t| kind_of(t) == Some(2)).count();
                let evals_here = w.tasks.iter().filter(|t| kind_of(t) == Some(0)).count();
                let games_here = w.tasks.len() - labels_here;
                let game_ok = games_here < w.parallel_games as usize;
                let eval_cap = (w.parallel_games as usize / EVAL_SHARE_DIVISOR).max(1);
                // Plan 071 R5: the foreground tier first, with its rules computed from foreground
                // jobs alone (a background generation never caps a foreground eval nor keeps
                // foreground labels off an idle game stream). Only when no foreground job can use
                // this stream does the background tier get it, under the same rules computed from
                // every job, so a background task never takes a stream a foreground one could.
                let pick = |background: bool| {
                    let in_tier = |j: &Job| background || j.priority.is_foreground();
                    let waiting = |j: &Job| j.state == JobState::Running && !j.draining && !j.pending.is_empty();
                    // Plan 069 item 4b: the generation itself, not its streamed labels.
                    let gen_pending = self
                        .jobs
                        .values()
                        .any(|j| in_tier(j) && waiting(j) && matches!(j.kind, Kind::Generate { .. }));
                    // Labelling (plan 062) is on the loop's critical path too.
                    let generate_waiting = gen_pending
                        || self
                            .jobs
                            .values()
                            .any(|j| in_tier(j) && waiting(j) && matches!(j.kind, Kind::Label { .. }));
                    let eval_capped = generate_waiting && evals_here >= eval_cap;
                    let label_ok = labels_here < w.label_streams as usize || !gen_pending;
                    (0..job_ids.len())
                        .map(|k| job_ids[(self.next_share + k) % job_ids.len()])
                        .find(|id| {
                            self.jobs.get(id).is_some_and(|j| {
                                j.priority.is_foreground() != background
                                    && !j.draining
                                    && !j.pending.is_empty()
                                    && match j.kind {
                                        Kind::Eval { .. } => game_ok && !eval_capped,
                                        Kind::Generate { .. } => game_ok,
                                        Kind::Label { .. } => label_ok,
                                    }
                            })
                        })
                };
                let Some(job_id) = pick(false).or_else(|| pick(true)) else {
                    break;
                };
                self.next_share = (job_ids.iter().position(|id| *id == job_id).unwrap() + 1) % job_ids.len();
                let job = self.jobs.get_mut(&job_id).expect("job exists");
                let Some(&(unit, _)) = job.pending.front() else { break };
                // One unit per task: take up to `batch` consecutive games
                // from the same rung/shard.
                let mut games = Vec::new();
                while games.len() < job.batch.max(1) as usize {
                    match job.pending.front() {
                        Some(&(u, g)) if u == unit => {
                            games.push(g);
                            job.pending.pop_front();
                        }
                        _ => break,
                    }
                }
                let task_id = self.next_task;
                self.next_task += 1;
                let task = job.make_task(task_id, unit, games.clone());
                let needed = task.models();
                self.in_flight.insert(
                    task_id,
                    InFlight {
                        job: job_id,
                        unit,
                        remaining: games.into_iter().collect(),
                        worker: wid,
                    },
                );
                let paths: Vec<(ModelId, PathBuf)> = needed
                    .iter()
                    .filter_map(|m| self.model_path(m).map(|p| (*m, p.clone())))
                    .collect();
                let w = self.workers.get_mut(&wid).expect("worker exists");
                for (m, path) in &paths {
                    if !w.named_models.contains(m) {
                        send_name(w, *m, path);
                    }
                }
                for m in needed {
                    if !w.known_models.contains(&m) {
                        if let Some(bytes) = self.models.get(&m) {
                            let _ = w.tx.send(ToWorker::Model {
                                id: m,
                                onnx: bytes.as_ref().clone(),
                                torchscript: self.torchscripts.get(&m).map(|t| t.as_ref().clone()),
                            });
                            w.known_models.insert(m);
                        }
                    }
                }
                w.tasks.insert(task_id);
                if w.tx.send(ToWorker::Task(task)).is_err() {
                    // Socket already gone; its reader will call remove_worker.
                    break;
                }
            }
        }
    }

    // -- results -----------------------------------------------------------

    /// Book-keeping shared by every result frame: which job/unit the task
    /// belongs to, the game struck off the task, the task retired when
    /// empty. `None` for a task we no longer track (requeued and finished
    /// elsewhere).
    fn game_arrived(&mut self, worker: WorkerId, task: TaskId, game: u32) -> Option<(JobId, usize)> {
        self.seen(worker);
        let f = self.in_flight.get_mut(&task)?;
        let (job_id, unit) = (f.job, f.unit);
        f.remaining.remove(&game);
        if f.remaining.is_empty() {
            self.in_flight.remove(&task);
            if let Some(w) = self.workers.get_mut(&worker) {
                w.tasks.remove(&task);
            }
        }
        if let Some(w) = self.workers.get_mut(&worker) {
            w.games_done += 1;
        }
        Some((job_id, unit))
    }

    pub fn eval_game_done(&mut self, worker: WorkerId, task: TaskId, line: EvalGameLine) {
        let name = self.worker_name(worker);
        let Some((job_id, unit)) = self.game_arrived(worker, task, line.game) else {
            return;
        };
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        // Plan 051: a job that finished on its SPRT verdicts still has games in flight. They retire
        // above (freeing the worker's stream) but change nothing, so the report is written once.
        if job.state != JobState::Running {
            self.dispatch();
            return;
        }
        let Kind::Eval {
            rungs, per_game, req, ..
        } = &mut job.kind
        else {
            return;
        };
        let r = &mut rungs[unit];
        if r.name == line.rung && r.done.insert(line.game) {
            let c = Counts {
                eval_games: 1,
                eval_decisions: line.telemetry.as_ref().map_or(0, |t| t.searches),
                ..Default::default()
            };
            job.by_worker.entry(name.clone()).or_default().add(&c);
            self.ledger.record(Instant::now(), &name, job_id, c);
            let was_decided = r.row.decided();
            r.row.record(&line);
            if !was_decided && r.row.decided() {
                let s = r.row.sprt.expect("decided implies a test");
                eprintln!(
                    "[hub] job {job_id} rung {}: SPRT {:?} after {} pairs (LLR {:.2}), dropping its queued games",
                    r.name, s.verdict, s.pairs, s.llr
                );
                job.pending.retain(|&(u, _)| u != unit);
            }
            if let Err(e) = serde_json::to_writer(&mut *per_game, &line)
                .map_err(io::Error::other)
                .and_then(|_| per_game.write_all(b"\n"))
                .and_then(|_| per_game.flush())
            {
                let msg = format!("writing {}: {e}", req.per_game_out.display());
                job.fail(msg);
                return;
            }
        }
        if job.all_done() {
            Self::finish(job);
        }
        self.dispatch();
    }

    /// One game's records. Returns the streamed-label shard writes that became due (plan 069 item
    /// 4b: a labelled shard whose last game this was, with nothing left to label); the caller runs
    /// them off the lock and reports back to [`Inner::label_written`].
    pub fn trajectory_done(
        &mut self,
        worker: WorkerId,
        task: TaskId,
        game: u32,
        samples: u32,
        payload: TrajectoryPayload,
    ) -> Vec<ShardWrite> {
        let name = self.worker_name(worker);
        let Some((job_id, unit)) = self.game_arrived(worker, task, game) else {
            return Vec::new();
        };
        let Some(job) = self.jobs.get_mut(&job_id) else {
            return Vec::new();
        };
        let Kind::Generate { shards, labels } = &mut job.kind else {
            return Vec::new();
        };
        let labels = *labels;
        let s = &mut shards[unit];
        // Every record of game `g` is seed `seed + g`. Anything else is a result that reached the
        // wrong task (it cannot, with task ids unique per hub process — this is the check that
        // says so): not written, and the game goes back on the queue if it still needs playing.
        let expected = s.seed + game as u64;
        if let Some(wrong) = payload.seeds.iter().flatten().find(|&&x| x != expected) {
            eprintln!(
                "[hub] job {job_id} {}: a result for game {game} (seed {expected}) holds seed {wrong}; dropped",
                s.name
            );
            if !s.done.contains(&game) && job.state == JobState::Running {
                job.pending.push_front((unit, game));
            }
            self.dispatch();
            return Vec::new();
        }
        let fresh = s.done.insert(game);
        let mut records = 0u64;
        // Plan 069 item 4b: the records as written, for the shard's streamed labels.
        let mut stream: Vec<io::Result<crate::label::Prepared>> = Vec::new();
        if fresh && !payload.nothing() {
            // One JSON line per record; a game that played the drive after its
            // score sends two (plan 047). Each is written as its own line.
            let TrajectoryPayload {
                lines,
                summaries,
                prepared,
                ..
            } = payload;
            let written = lines
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|json| {
                    for line in json.split(|b| *b == b'\n') {
                        s.writer.write_all(line)?;
                        s.writer.write_all(b"\n")?;
                    }
                    s.writer.flush()?;
                    Ok(json)
                });
            match written {
                Ok(json) => {
                    s.samples += samples as u64;
                    // Page statistics only: a line the peek cannot read is still a valid
                    // corpus line, so it is written and just not counted here. The
                    // samples are attributed to the first record; the count is a page
                    // figure, not a corpus one.
                    for (i, summary) in summaries.into_iter().enumerate() {
                        s.written += 1;
                        records += 1;
                        if let Some((board, tds)) = summary {
                            s.stats.add(board, tds, if i == 0 { samples as u64 } else { 0 });
                        }
                    }
                    if s.label_unit.is_some() {
                        // Prepared before the lock when the hub knew it streams labels; else here.
                        stream = match prepared {
                            Some(p) => p.into_iter().map(Ok).collect(),
                            None => json.split(|b| *b == b'\n').map(crate::label::Prepared::of).collect(),
                        };
                    }
                }
                Err(e) => {
                    let msg = format!("writing {}: {e}", s.out.display());
                    job.fail(msg);
                    self.propagate_failure(job_id);
                    return Vec::new();
                }
            }
        }
        let label_unit = s.label_unit;
        let shard_complete = fresh && s.done.len() as u32 >= s.total;
        if fresh {
            let c = Counts {
                games: 1,
                records,
                samples: samples as u64,
                ..Default::default()
            };
            job.by_worker.entry(name.clone()).or_default().add(&c);
            self.ledger.record(Instant::now(), &name, job_id, c);
        }
        if job.all_done() {
            Self::finish(job);
        }
        let mut writes = Vec::new();
        if let (Some(lid), Some(lu)) = (labels, label_unit) {
            if !stream.is_empty() {
                self.stream_lines(lid, lu, stream);
            }
            if shard_complete {
                writes.extend(self.seal_label_shard(lid, lu));
            }
        }
        self.dispatch();
        writes
    }

    /// Plan 069 item 4b: records a generate shard just wrote, into its streamed-label shard (in
    /// the file's order: called under the lock that wrote them). A line the label reader refuses
    /// fails the label job, as `job label` would refuse the shard; the generation goes on.
    fn stream_lines(&mut self, lid: JobId, unit: usize, lines: Vec<io::Result<crate::label::Prepared>>) {
        let Some(job) = self.jobs.get_mut(&lid) else { return };
        if job.state != JobState::Running {
            return;
        }
        let Kind::Label { shards, .. } = &mut job.kind else {
            return;
        };
        let s = &mut shards[unit];
        if s.input.skipped || !s.input.is_open() {
            return;
        }
        for p in lines {
            match p.and_then(|p| s.push(p)) {
                Ok(new) => job.pending.extend(new.map(|g| (unit, g))),
                Err(e) => {
                    let msg = format!("{}: {e}", s.input.name);
                    job.fail(msg);
                    return;
                }
            }
        }
    }

    /// Plan 062: one label item's result. Returns the shard's write when this was its last item;
    /// the caller runs it off the lock and reports back to [`Inner::label_written`].
    pub fn label_done(&mut self, worker: WorkerId, task: TaskId, item: u32, result: LabelResult) -> Option<ShardWrite> {
        let name = self.worker_name(worker);
        let (job_id, unit) = self.game_arrived(worker, task, item)?;
        let write = self.record_label(&name, job_id, unit, item, result);
        self.dispatch();
        write
    }

    fn record_label(
        &mut self,
        name: &str,
        job_id: JobId,
        unit: usize,
        item: u32,
        result: LabelResult,
    ) -> Option<ShardWrite> {
        let job = self.jobs.get_mut(&job_id)?;
        if job.state != JobState::Running {
            return None;
        }
        let Kind::Label { shards, .. } = &mut job.kind else {
            return None;
        };
        let s = &mut shards[unit];
        let Some(&(t, start, end)) = s.input.items.get(item as usize) else {
            let msg = format!("{}: a result for item {item}, which does not exist", s.input.name);
            job.fail(msg);
            return None;
        };
        if s.stale.contains(&item) || !s.done.insert(item) {
            return None;
        }
        s.live_done += 1;
        let mut c = Counts {
            label_items: 1,
            ..Default::default()
        };
        match result {
            LabelResult::Labels(v) => {
                if v.len() != (end - start) as usize {
                    let msg = format!(
                        "{} item {item}: {} labels for samples {start}..{end}",
                        s.input.name,
                        v.len()
                    );
                    job.fail(msg);
                    return None;
                }
                s.labels[t][start as usize..end as usize].copy_from_slice(&v);
                s.labelled += v.len() as u64;
                s.counted[t] += v.len() as u64;
                c.label_samples = v.len() as u64;
            }
            LabelResult::Unlabellable(why) => {
                if !s.unlabellable[t] {
                    eprintln!(
                        "[hub] {}: left unlabelled (seed {:?}): {why}",
                        s.input.name, s.input.seeds[t]
                    );
                }
                s.unlabellable[t] = true;
            }
        }
        let write = s.take_write(job_id, unit);
        job.by_worker.entry(name.to_string()).or_default().add(&c);
        self.ledger.record(Instant::now(), name, job_id, c);
        write
    }

    /// A label shard's write finished (plan 062): the shard is done, and with it maybe the job.
    pub fn label_written(&mut self, job_id: JobId, unit: usize, result: io::Result<()>) {
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        if job.state != JobState::Running {
            return;
        }
        let elapsed = job.started.elapsed().as_secs_f64();
        let Kind::Label { shards, .. } = &mut job.kind else {
            return;
        };
        let s = &mut shards[unit];
        if let Err(e) = result {
            let msg = format!("writing {}: {e}", s.input.out.display());
            job.fail(msg);
            return;
        }
        s.written = true;
        s.release();
        s.secs = elapsed;
        let st = s.stats();
        eprintln!(
            "[hub] {}",
            botbowl_play::mc_label::summary_line(
                &st.input,
                &st.out,
                st.trajectories as usize,
                st.unlabelled as usize,
                st.samples as usize,
                s.input.cfg.playouts,
                st.secs
            )
        );
        if job.all_done() {
            Self::finish(job);
        }
    }

    pub fn task_failed(&mut self, worker: WorkerId, task: TaskId, error: String) {
        self.seen(worker);
        let Some(f) = self.in_flight.get(&task) else { return };
        let (job_id, unit) = (f.job, f.unit);
        let games: Vec<u32> = f.remaining.iter().copied().collect();
        eprintln!("[hub] task {task} failed on worker {worker}: {error}");
        self.requeue(task);
        if let Some(job) = self.jobs.get_mut(&job_id) {
            for g in games {
                let n = job.failures.entry((unit, g)).or_insert(0);
                *n += 1;
                if *n >= MAX_GAME_FAILURES {
                    let n = *n;
                    let msg = format!("{} game {g} failed {n} times; last: {error}", job.unit_name(unit));
                    job.fail(msg);
                    self.propagate_failure(job_id);
                    break;
                }
            }
        }
        self.dispatch();
    }

    fn finish(job: &mut Job) {
        job.ended = Some(Instant::now());
        let elapsed = job.started.elapsed().as_secs();
        match &mut job.kind {
            Kind::Eval { req, rungs, report, .. } => {
                // Plan 042: a multi-size ladder reports its board list, the
                // same string `botbowl-ui eval --board-sizes` writes.
                let mut boards: Vec<String> = rungs.iter().filter_map(|r| r.board.map(board_label)).collect();
                boards.dedup();
                let ladder: Vec<LadderRow> = rungs.iter().map(|r| r.row.clone().finish()).collect();
                let r = Report {
                    candidate: req.candidate_label.clone(),
                    mcts_iters: req.mcts_iters,
                    seed: req.seed,
                    board_env: if boards.is_empty() {
                        format!("{:?}", BoardDims::from_env())
                    } else {
                        boards.join(",")
                    },
                    git_commit: botbowl_data::git_commit().to_string(),
                    git_dirty: botbowl_data::git_dirty(),
                    lectures: Vec::new(),
                    // Plan 043: the same fold as the single-process driver, over lines the
                    // workers sent — one implementation, two callers.
                    telemetry: Report::telemetry_of(&ladder),
                    ladder,
                    candidate_config: req.candidate_config.clone(),
                    opponent_config: req.opponent_config.clone(),
                };
                match serde_json::to_string_pretty(&r)
                    .map_err(io::Error::other)
                    .and_then(|s| std::fs::write(&req.report_out, s))
                {
                    Ok(()) => {
                        eprintln!(
                            "[hub] job {} done in {elapsed} s -> {}",
                            job.id,
                            req.report_out.display()
                        );
                        job.state = JobState::Done;
                    }
                    Err(e) => {
                        job.state = JobState::Failed {
                            error: format!("writing {}: {e}", req.report_out.display()),
                        }
                    }
                }
                *report = Some(r);
            }
            Kind::Generate { shards, .. } => {
                let mut err = None;
                for s in shards.iter_mut() {
                    if let Err(e) = s.writer.flush() {
                        err = Some(format!("writing {}: {e}", s.out.display()));
                    }
                }
                match err {
                    Some(error) => job.state = JobState::Failed { error },
                    None => {
                        eprintln!(
                            "[hub] job {} done in {elapsed} s: {} trajectories / {} samples over {} shard(s)",
                            job.id,
                            shards.iter().map(|s| s.written as u64).sum::<u64>(),
                            shards.iter().map(|s| s.samples).sum::<u64>(),
                            shards.len()
                        );
                        job.state = JobState::Done;
                    }
                }
            }
            Kind::Label { shards, .. } => {
                eprintln!(
                    "[hub] job {} done in {elapsed} s: {} samples labelled over {} shard(s), {} trajectories left unlabelled",
                    job.id,
                    shards.iter().map(|s| s.labelled).sum::<u64>(),
                    shards.len(),
                    shards.iter().map(LabelShard::unlabelled).sum::<usize>()
                );
                job.state = JobState::Done;
            }
        }
    }

    /// Plan 062: the compressed corpus lines label jobs still hold, in bytes. Only shards with
    /// items yet to label or a write yet to finish should hold any.
    pub fn label_bytes_held(&self) -> usize {
        self.jobs
            .values()
            .filter_map(|j| match &j.kind {
                Kind::Label { shards, .. } => Some(shards.iter().map(LabelShard::held_bytes).sum::<usize>()),
                _ => None,
            })
            .sum()
    }

    /// Any job still running?
    pub fn busy(&self) -> bool {
        self.jobs.values().any(|j| j.state == JobState::Running)
    }
}

#[cfg(test)]
mod tests {
    use super::drive_summary;

    fn line(start: &str, home: u8, away: u8) -> Vec<u8> {
        format!(
            r#"{{"meta":{{"board_dims":{{"width":18,"height":11,"team_size":6}},"extra":{{"start_score":"{start}"}}}},"samples":[{{"state":{{}}}}],"outcome":{{"home_score":{home},"away_score":{away},"winner":null,"game_over":false,"z_home":0.0,"lecture_status":null}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn a_drive_counts_only_its_own_touchdowns() {
        assert_eq!(drive_summary(&line("1-1", 1, 2)), Some(("16x9/6".to_string(), 1)));
        assert_eq!(drive_summary(&line("2-0", 2, 0)), Some(("16x9/6".to_string(), 0)));
    }

    #[test]
    fn a_line_without_a_start_score_starts_from_nil_nil() {
        let json = br#"{"meta":{"board_dims":{"width":16,"height":9,"team_size":4}},"outcome":{"home_score":2,"away_score":1}}"#;
        assert_eq!(drive_summary(json), Some(("14x7/4".to_string(), 3)));
    }
}

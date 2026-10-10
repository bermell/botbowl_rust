//! The hub's whole mutable state behind one mutex: connected workers,
//! model bytes, jobs and their queues. Every transition is a synchronous
//! method here; the websocket and HTTP layers only translate.
//!
//! Scheduling is deliberately simple (plan 041 decision 7): free streams go
//! round-robin over the running jobs (plan 047: a generation's eval shares the
//! fleet with the next generation), a task is a small batch of games from one *unit* (a
//! ladder rung of an eval job, a corpus shard of a generate job), a
//! worker holds at most `parallel_games` tasks, and anything a departed
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
    BoardDims, BotSpec, BuildInfo, EvalGameLine, GenerateConfig, LabelConfig, LabelItem, LabelResult, ModelId, Task,
    TaskId, ToWorker,
};
use botbowl_play::board_sizes::board_label;
use botbowl_play::eval::{LadderRow, Report};

use crate::api::{
    BotReq, Counts, EvalJobRequest, EvalStats, GenStats, GenerateJobRequest, HubStatus, JobId, JobKind, JobRequest,
    JobState, JobStatus, LabelJobRequest, LabelStats, UnitProgress, UnitStats, WorkerStatus,
};
use crate::label::{LabelInput, ShardWrite};
use crate::rates::{Ledger, Live};

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
}

impl LabelShard {
    fn new(input: LabelInput) -> Self {
        LabelShard {
            labels: input.samples.iter().map(|&n| vec![0.0; n]).collect(),
            unlabellable: vec![false; input.samples.len()],
            input,
            done: HashSet::new(),
            labelled: 0,
            writing: false,
            written: false,
            secs: 0.0,
        }
    }

    /// Nothing left to do: written, or skipped at submit.
    fn finished(&self) -> bool {
        self.written || self.input.skipped
    }

    /// The write, once every item is in (at most once).
    fn take_write(&mut self, job: JobId, unit: usize, tag: &str) -> Option<ShardWrite> {
        if self.writing || self.input.skipped || self.done.len() < self.input.items.len() {
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
            tag: tag.to_string(),
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
    },
    Label {
        shards: Vec<LabelShard>,
        cfg: LabelConfig,
        model: ModelId,
        /// `meta.extra.value_label` of every labelled trajectory.
        tag: String,
    },
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
            Kind::Generate { shards } => shards
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
                    done: s.done.len() as u32,
                    total: s.input.items.len() as u32,
                    samples: s.labelled,
                    stats: Some(UnitStats::Label(s.stats())),
                })
                .collect(),
        }
    }

    fn unit_name(&self, unit: usize) -> &str {
        match &self.kind {
            Kind::Eval { rungs, .. } => &rungs[unit].name,
            Kind::Generate { shards } => &shards[unit].name,
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
            Kind::Generate { shards } => {
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
            Kind::Label { shards, cfg, model, .. } => {
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
                    cfg: *cfg,
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

/// The board and the drive's own touchdowns, read from a trajectory line without keeping its
/// samples. `outcome` is the absolute scoreline; a random-start drive begins at
/// `meta.extra.start_score`, so its touchdowns are the difference (`td_rate.py`'s rule).
fn drive_summary(json: &[u8]) -> Option<(String, u32)> {
    #[derive(serde::Deserialize)]
    struct Meta {
        board_dims: BoardDims,
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
    let p: Peek = serde_json::from_slice(json).ok()?;
    let start = p
        .meta
        .extra
        .get("start_score")
        .and_then(|s| s.split_once('-'))
        .and_then(|(h, a)| Some(h.parse::<u32>().ok()? + a.parse::<u32>().ok()?))
        .unwrap_or(0);
    let end = p.outcome.home_score as u32 + p.outcome.away_score as u32;
    Some((board_label(p.meta.board_dims), end.saturating_sub(start)))
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
}

impl Inner {
    // -- models ------------------------------------------------------------

    /// Read an ONNX file, hash it, keep the bytes for shipping.
    pub fn load_model(&mut self, path: &Path) -> io::Result<ModelId> {
        let bytes =
            std::fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("model {}: {e}", path.display())))?;
        let id = ModelId::of(&bytes);
        self.models.entry(id).or_insert_with(|| Arc::new(bytes));
        self.model_paths.entry(id).or_insert_with(|| path.to_path_buf());
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

    pub fn submit(&mut self, req: JobRequest) -> io::Result<JobId> {
        match req {
            JobRequest::Eval(r) => self.submit_eval(r),
            JobRequest::Generate(r) => self.submit_generate(r),
            // The synchronous path (a caller holding the lock): read and write inline.
            // `Hub::submit` reads the shards before taking the lock and writes on threads.
            JobRequest::Label(r) => {
                let inputs = crate::label::load_inputs(&r)?;
                let (id, writes) = self.submit_label(r, inputs)?;
                for w in writes {
                    let result = w.run();
                    self.label_written(w.job, w.unit, result);
                }
                Ok(id)
            }
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
        let tag = botbowl_play::mc_label::label_tag(&req.cfg, &req.model);
        let mut pending = VecDeque::new();
        for (i, s) in inputs.iter().enumerate() {
            if s.skipped {
                eprintln!("[hub] mc-label: {} exists, skipping", s.out.display());
            } else {
                pending.extend((0..s.items.len() as u32).map(|g| (i, g)));
            }
        }
        let shards = inputs.into_iter().map(LabelShard::new).collect();
        let kind = Kind::Label {
            shards,
            cfg: req.cfg,
            model,
            tag: tag.clone(),
        };
        let id = self.insert_job(kind, req.label, req.batch, pending);
        let job = self.jobs.get_mut(&id).expect("just inserted");
        let mut writes = Vec::new();
        if let Kind::Label { shards, .. } = &mut job.kind {
            for (unit, s) in shards.iter_mut().enumerate() {
                writes.extend(s.take_write(id, unit, &tag));
            }
        }
        if job.all_done() {
            Self::finish(job);
        }
        Ok((id, writes))
    }

    pub fn submit_eval(&mut self, req: EvalJobRequest) -> io::Result<JobId> {
        let candidate = self.resolve_bot(&req.candidate)?;
        let mut rungs = Vec::new();
        let mut pending = VecDeque::new();
        for (i, r) in req.rungs.iter().enumerate() {
            let opponent = self.resolve_bot(&r.opponent)?;
            let mut row = LadderRow::new(&r.name).with_sprt(req.sprt);
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
            pending.extend((0..r.games).map(|g| (i, g)));
        }
        if let Some(dir) = req.report_out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let per_game = open_out(&req.per_game_out, false)?;
        let batch = req.batch;
        let label = req.label.clone();
        let kind = Kind::Eval {
            req,
            candidate,
            rungs,
            per_game,
            report: None,
        };
        Ok(self.insert_job(kind, label, batch, pending))
    }

    pub fn submit_generate(&mut self, req: GenerateJobRequest) -> io::Result<JobId> {
        if req.shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "generate job with no shards",
            ));
        }
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
            shards.push(Shard {
                name: s.name.clone(),
                out: s.out.clone(),
                seed: s.seed,
                total: s.games,
                cfg: s.cfg.clone(),
                model,
                done: HashSet::new(),
                written: 0,
                samples: 0,
                stats: GenStats::default(),
                writer: open_out(&s.out, req.truncate)?,
            });
            pending.extend((0..s.games).map(|g| (i, g)));
        }
        Ok(self.insert_job(Kind::Generate { shards }, req.label, req.batch, pending))
    }

    fn insert_job(&mut self, kind: Kind, label: Option<String>, batch: u16, pending: VecDeque<(usize, u32)>) -> JobId {
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
        };
        eprintln!(
            "[hub] job {id} submitted: {}",
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
                streams: w.parallel_games as u32,
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
            "[hub] worker {id} {:?} joined ({} streams, {} cores, {} MB, {})",
            conn.name, conn.parallel_games, conn.cores, conn.ram_mb, conn.triple
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
                job.pending.push_front((f.unit, g));
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
    pub fn dispatch(&mut self) {
        let job_ids: Vec<JobId> = self
            .jobs
            .values()
            .filter(|j| j.state == JobState::Running && !j.pending.is_empty())
            .map(|j| j.id)
            .collect();
        if job_ids.is_empty() {
            return;
        }
        let worker_ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        for wid in worker_ids {
            loop {
                let w = &self.workers[&wid];
                if w.tasks.len() >= w.parallel_games as usize {
                    break;
                }
                // Labelling (plan 062) is on the loop's critical path too.
                let generate_waiting = self.jobs.values().any(|j| {
                    j.state == JobState::Running
                        && matches!(j.kind, Kind::Generate { .. } | Kind::Label { .. })
                        && !j.pending.is_empty()
                });
                let evals_here = w
                    .tasks
                    .iter()
                    .filter(|t| {
                        self.in_flight
                            .get(t)
                            .and_then(|f| self.jobs.get(&f.job))
                            .is_some_and(|j| matches!(j.kind, Kind::Eval { .. }))
                    })
                    .count();
                let eval_capped =
                    generate_waiting && evals_here >= (w.parallel_games as usize / EVAL_SHARE_DIVISOR).max(1);
                let Some(job_id) = (0..job_ids.len())
                    .map(|k| job_ids[(self.next_share + k) % job_ids.len()])
                    .find(|id| {
                        self.jobs.get(id).is_some_and(|j| {
                            !j.pending.is_empty() && !(eval_capped && matches!(j.kind, Kind::Eval { .. }))
                        })
                    })
                else {
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

    pub fn trajectory_done(&mut self, worker: WorkerId, task: TaskId, game: u32, samples: u32, zstd_json: Vec<u8>) {
        let name = self.worker_name(worker);
        let Some((job_id, unit)) = self.game_arrived(worker, task, game) else {
            return;
        };
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        let Kind::Generate { shards } = &mut job.kind else {
            return;
        };
        let s = &mut shards[unit];
        let fresh = s.done.insert(game);
        let mut records = 0u64;
        if fresh && !zstd_json.is_empty() {
            // One JSON line per record; a game that played the drive after its
            // score sends two (plan 047). Each is written as its own line.
            let written = zstd::decode_all(&zstd_json[..])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("zstd: {e}")))
                .and_then(|json| {
                    let lines: Vec<&[u8]> = json.split(|b| *b == b'\n').collect();
                    if lines.iter().any(|l| l.is_empty()) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "trajectory payload has an empty line",
                        ));
                    }
                    for line in &lines {
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
                    for (i, line) in json.split(|b| *b == b'\n').enumerate() {
                        s.written += 1;
                        records += 1;
                        if let Some((board, tds)) = drive_summary(line) {
                            s.stats.add(board, tds, if i == 0 { samples as u64 } else { 0 });
                        }
                    }
                }
                Err(e) => {
                    let msg = format!("writing {}: {e}", s.out.display());
                    job.fail(msg);
                    return;
                }
            }
        }
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
        self.dispatch();
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
        let Kind::Label { shards, tag, .. } = &mut job.kind else {
            return None;
        };
        let s = &mut shards[unit];
        let Some(&(t, start, end)) = s.input.items.get(item as usize) else {
            let msg = format!("{}: a result for item {item}, which does not exist", s.input.name);
            job.fail(msg);
            return None;
        };
        if !s.done.insert(item) {
            return None;
        }
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
        let write = s.take_write(job_id, unit, tag);
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
        let Kind::Label { shards, cfg, .. } = &mut job.kind else {
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
                cfg.playouts,
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
            Kind::Generate { shards } => {
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

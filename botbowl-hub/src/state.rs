//! The hub's whole mutable state behind one mutex: connected workers,
//! model bytes, jobs and their queues. Every transition is a synchronous
//! method here; the websocket and HTTP layers only translate.
//!
//! Scheduling is deliberately simple (plan 041 decision 7): jobs run in
//! submission order, a task is a small batch of games from one *unit* (a
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

use botbowl_hub_proto::{BotSpec, BuildInfo, EvalGameLine, GenerateConfig, ModelId, Task, TaskId, ToWorker};
use botbowl_play::eval::{LadderRow, Report};

use crate::api::{
    BotReq, EvalJobRequest, GenerateJobRequest, HubStatus, JobId, JobKind, JobRequest, JobState, JobStatus,
    UnitProgress, WorkerStatus,
};

pub type WorkerId = u64;

/// How many times one game may fail (on any worker) before the job does.
const MAX_GAME_FAILURES: u32 = 3;

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
    pub tasks: HashSet<TaskId>,
    pub games_done: u64,
    pub last_seen: Instant,
}

struct Rung {
    name: String,
    total: u32,
    opponent: BotSpec,
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
    writer: io::BufWriter<std::fs::File>,
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
}

struct InFlight {
    job: JobId,
    unit: usize,
    remaining: HashSet<u32>,
    worker: WorkerId,
}

pub struct Job {
    id: JobId,
    kind: Kind,
    batch: u16,
    /// `(unit index, game)` not yet handed out.
    pending: VecDeque<(usize, u32)>,
    failures: HashMap<(usize, u32), u32>,
    state: JobState,
    started: Instant,
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
                })
                .collect(),
            Kind::Generate { shards } => shards
                .iter()
                .map(|s| UnitProgress {
                    name: s.name.clone(),
                    done: s.done.len() as u32,
                    total: s.total,
                    samples: s.samples,
                })
                .collect(),
        }
    }

    fn unit_name(&self, unit: usize) -> &str {
        match &self.kind {
            Kind::Eval { rungs, .. } => &rungs[unit].name,
            Kind::Generate { shards } => &shards[unit].name,
        }
    }

    fn all_done(&self) -> bool {
        self.units().iter().all(|u| u.done >= u.total)
    }

    fn status(&self, workers_connected: usize) -> JobStatus {
        JobStatus {
            id: self.id,
            kind: match self.kind {
                Kind::Eval { .. } => JobKind::Eval,
                Kind::Generate { .. } => JobKind::Generate,
            },
            workers_connected,
            state: self.state.clone(),
            units: self.units(),
            elapsed_secs: self.started.elapsed().as_secs(),
            report: match &self.kind {
                Kind::Eval { report, .. } => report.clone(),
                Kind::Generate { .. } => None,
            },
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
        }
    }

    fn fail(&mut self, error: String) {
        eprintln!("[hub] job {} failed: {error}", self.id);
        self.state = JobState::Failed { error };
        self.pending.clear();
    }
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

#[derive(Default)]
pub struct Inner {
    pub workers: HashMap<WorkerId, WorkerConn>,
    pub models: HashMap<ModelId, Arc<Vec<u8>>>,
    jobs: BTreeMap<JobId, Job>,
    in_flight: HashMap<TaskId, InFlight>,
    next_worker: WorkerId,
    next_job: JobId,
    next_task: TaskId,
}

impl Inner {
    // -- models ------------------------------------------------------------

    /// Read an ONNX file, hash it, keep the bytes for shipping.
    pub fn load_model(&mut self, path: &Path) -> io::Result<ModelId> {
        let bytes =
            std::fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("model {}: {e}", path.display())))?;
        let id = ModelId::of(&bytes);
        self.models.entry(id).or_insert_with(|| Arc::new(bytes));
        Ok(id)
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
        }
    }

    pub fn submit_eval(&mut self, req: EvalJobRequest) -> io::Result<JobId> {
        let candidate = self.resolve_bot(&req.candidate)?;
        let mut rungs = Vec::new();
        let mut pending = VecDeque::new();
        for (i, r) in req.rungs.iter().enumerate() {
            let opponent = self.resolve_bot(&r.opponent)?;
            rungs.push(Rung {
                name: r.name.clone(),
                total: r.games,
                opponent,
                row: LadderRow::new(&r.name),
                done: HashSet::new(),
            });
            pending.extend((0..r.games).map(|g| (i, g)));
        }
        if let Some(dir) = req.report_out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let per_game = open_out(&req.per_game_out, false)?;
        let batch = req.batch;
        let kind = Kind::Eval {
            req,
            candidate,
            rungs,
            per_game,
            report: None,
        };
        Ok(self.insert_job(kind, batch, pending))
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
                writer: open_out(&s.out, req.truncate)?,
            });
            pending.extend((0..s.games).map(|g| (i, g)));
        }
        Ok(self.insert_job(Kind::Generate { shards }, req.batch, pending))
    }

    fn insert_job(&mut self, kind: Kind, batch: u16, pending: VecDeque<(usize, u32)>) -> JobId {
        let id = self.next_job;
        self.next_job += 1;
        let job = Job {
            id,
            kind,
            batch,
            pending,
            failures: HashMap::new(),
            state: JobState::Running,
            started: Instant::now(),
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
        self.jobs.get(&id).map(|j| j.status(self.workers.len()))
    }

    pub fn status(&self) -> HubStatus {
        HubStatus {
            commit: botbowl_data::git_commit().to_string(),
            dirty: botbowl_data::git_dirty(),
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
            jobs: self.jobs.values().map(|j| j.status(self.workers.len())).collect(),
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
            for g in f.remaining {
                job.pending.push_front((f.unit, g));
            }
        }
    }

    /// Hand out work to every worker with a free stream.
    pub fn dispatch(&mut self) {
        let Some(job_id) = self
            .jobs
            .values()
            .find(|j| j.state == JobState::Running && !j.pending.is_empty())
            .map(|j| j.id)
        else {
            return;
        };
        let worker_ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        for wid in worker_ids {
            loop {
                let w = &self.workers[&wid];
                if w.tasks.len() >= w.parallel_games as usize {
                    break;
                }
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
                let w = self.workers.get_mut(&wid).expect("worker exists");
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
        let Some((job_id, unit)) = self.game_arrived(worker, task, line.game) else {
            return;
        };
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        let Kind::Eval {
            rungs, per_game, req, ..
        } = &mut job.kind
        else {
            return;
        };
        let r = &mut rungs[unit];
        if r.name == line.rung && r.done.insert(line.game) {
            r.row.record(&line);
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
        let Some((job_id, unit)) = self.game_arrived(worker, task, game) else {
            return;
        };
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        let Kind::Generate { shards } = &mut job.kind else {
            return;
        };
        let s = &mut shards[unit];
        if s.done.insert(game) && !zstd_json.is_empty() {
            let written = zstd::decode_all(&zstd_json[..])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("zstd: {e}")))
                .and_then(|json| {
                    if json.is_empty() || json.contains(&b'\n') {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "trajectory is not a single JSON line",
                        ));
                    }
                    s.writer.write_all(&json)?;
                    s.writer.write_all(b"\n")?;
                    s.writer.flush()
                });
            match written {
                Ok(()) => {
                    s.written += 1;
                    s.samples += samples as u64;
                }
                Err(e) => {
                    let msg = format!("writing {}: {e}", s.out.display());
                    job.fail(msg);
                    return;
                }
            }
        }
        if job.all_done() {
            Self::finish(job);
        }
        self.dispatch();
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
        let elapsed = job.started.elapsed().as_secs();
        match &mut job.kind {
            Kind::Eval { req, rungs, report, .. } => {
                let r = Report {
                    candidate: req.candidate_label.clone(),
                    mcts_iters: req.mcts_iters,
                    seed: req.seed,
                    board_env: format!("{:?}", botbowl_engine::core::model::BoardDims::from_env()),
                    git_commit: botbowl_data::git_commit().to_string(),
                    git_dirty: botbowl_data::git_dirty(),
                    lectures: Vec::new(),
                    ladder: rungs.iter().map(|r| r.row.clone().finish()).collect(),
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
        }
    }

    /// Any job still running?
    pub fn busy(&self) -> bool {
        self.jobs.values().any(|j| j.state == JobState::Running)
    }
}

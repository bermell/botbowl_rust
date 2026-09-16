//! The hub's whole mutable state behind one mutex: connected workers,
//! model bytes, jobs and their queues. Every transition is a synchronous
//! method here; the websocket and HTTP layers only translate.
//!
//! Scheduling is deliberately simple (plan 040 decision 7): jobs run in
//! submission order, a task is a small batch of games from one rung, a
//! worker holds at most `parallel_games` tasks, and anything a departed
//! worker had in flight goes back on the queue. Results are deduplicated
//! on `(job, rung, game)` so a slow worker that reappears can never double
//! count.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use botbowl_hub_proto::{BotSpec, BuildInfo, EvalGameLine, ModelId, Task, TaskId, ToWorker};
use botbowl_play::eval::{LadderRow, Report};

use crate::api::{BotReq, EvalJobRequest, HubStatus, JobId, JobState, JobStatus, RungProgress, WorkerStatus};

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

struct InFlight {
    job: JobId,
    rung: usize,
    remaining: HashSet<u32>,
    worker: WorkerId,
}

pub struct Job {
    id: JobId,
    req: EvalJobRequest,
    candidate: BotSpec,
    rungs: Vec<Rung>,
    /// `(rung index, game)` not yet handed out.
    pending: VecDeque<(usize, u32)>,
    failures: HashMap<(usize, u32), u32>,
    per_game: io::BufWriter<std::fs::File>,
    state: JobState,
    started: Instant,
    report: Option<Report>,
}

impl Job {
    fn all_done(&self) -> bool {
        self.rungs.iter().all(|r| r.done.len() as u32 >= r.total)
    }

    fn status(&self) -> JobStatus {
        JobStatus {
            id: self.id,
            state: self.state.clone(),
            rungs: self
                .rungs
                .iter()
                .map(|r| RungProgress {
                    name: r.name.clone(),
                    done: r.done.len() as u32,
                    total: r.total,
                })
                .collect(),
            elapsed_secs: self.started.elapsed().as_secs(),
            report: self.report.clone(),
        }
    }
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
    pub fn load_model(&mut self, path: &PathBuf) -> io::Result<ModelId> {
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
        if let Some(dir) = req.per_game_out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if let Some(dir) = req.report_out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let per_game = io::BufWriter::new(
            std::fs::File::options()
                .create(true)
                .append(true)
                .open(&req.per_game_out)?,
        );
        let id = self.next_job;
        self.next_job += 1;
        self.jobs.insert(
            id,
            Job {
                id,
                req,
                candidate,
                rungs,
                pending,
                failures: HashMap::new(),
                per_game,
                state: JobState::Running,
                started: Instant::now(),
                report: None,
            },
        );
        self.dispatch();
        Ok(id)
    }

    pub fn job_status(&self, id: JobId) -> Option<JobStatus> {
        self.jobs.get(&id).map(Job::status)
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
            jobs: self.jobs.values().map(Job::status).collect(),
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

    // -- scheduling --------------------------------------------------------

    /// Put a task's unfinished games back at the front of its job's queue.
    fn requeue(&mut self, task: TaskId) {
        let Some(f) = self.in_flight.remove(&task) else { return };
        if let Some(w) = self.workers.get_mut(&f.worker) {
            w.tasks.remove(&task);
        }
        if let Some(job) = self.jobs.get_mut(&f.job) {
            for g in f.remaining {
                job.pending.push_front((f.rung, g));
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
                let Some(&(rung, _)) = job.pending.front() else { break };
                // One rung per task: take up to `batch` consecutive games
                // from the same rung.
                let mut games = Vec::new();
                while games.len() < job.req.batch.max(1) as usize {
                    match job.pending.front() {
                        Some(&(r, g)) if r == rung => {
                            games.push(g);
                            job.pending.pop_front();
                        }
                        _ => break,
                    }
                }
                let task_id = self.next_task;
                self.next_task += 1;
                let task = Task::Eval {
                    id: task_id,
                    rung: job.rungs[rung].name.clone(),
                    games: games.clone(),
                    seed: job.req.seed,
                    max_steps: job.req.max_steps,
                    candidate: job.candidate.clone(),
                    opponent: job.rungs[rung].opponent.clone(),
                };
                let needed = task.models();
                self.in_flight.insert(
                    task_id,
                    InFlight {
                        job: job_id,
                        rung,
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

    pub fn eval_game_done(&mut self, worker: WorkerId, task: TaskId, line: EvalGameLine) {
        self.seen(worker);
        let Some(f) = self.in_flight.get_mut(&task) else {
            // Late result for a task we already requeued and finished
            // elsewhere; still worth recording if the game is not done.
            return;
        };
        let job_id = f.job;
        let rung = f.rung;
        f.remaining.remove(&line.game);
        let task_finished = f.remaining.is_empty();
        if task_finished {
            self.in_flight.remove(&task);
            if let Some(w) = self.workers.get_mut(&worker) {
                w.tasks.remove(&task);
            }
        }
        if let Some(w) = self.workers.get_mut(&worker) {
            w.games_done += 1;
        }
        let Some(job) = self.jobs.get_mut(&job_id) else { return };
        let r = &mut job.rungs[rung];
        if r.name == line.rung && r.done.insert(line.game) {
            r.row.record(&line);
            if let Err(e) = serde_json::to_writer(&mut job.per_game, &line)
                .map_err(io::Error::other)
                .and_then(|_| job.per_game.write_all(b"\n"))
                .and_then(|_| job.per_game.flush())
            {
                job.state = JobState::Failed {
                    error: format!("writing {}: {e}", job.req.per_game_out.display()),
                };
                return;
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
        let (job_id, rung) = (f.job, f.rung);
        let games: Vec<u32> = f.remaining.iter().copied().collect();
        eprintln!("[hub] task {task} failed on worker {worker}: {error}");
        self.requeue(task);
        if let Some(job) = self.jobs.get_mut(&job_id) {
            for g in games {
                let n = job.failures.entry((rung, g)).or_insert(0);
                *n += 1;
                if *n >= MAX_GAME_FAILURES {
                    job.state = JobState::Failed {
                        error: format!(
                            "rung {:?} game {g} failed {n} times; last: {error}",
                            job.rungs[rung].name
                        ),
                    };
                    job.pending.clear();
                }
            }
        }
        self.dispatch();
    }

    fn finish(job: &mut Job) {
        let report = Report {
            candidate: job.req.candidate_label.clone(),
            mcts_iters: job.req.mcts_iters,
            seed: job.req.seed,
            board_env: format!("{:?}", botbowl_engine::core::model::BoardDims::from_env()),
            git_commit: botbowl_data::git_commit().to_string(),
            git_dirty: botbowl_data::git_dirty(),
            lectures: Vec::new(),
            ladder: job.rungs.iter().map(|r| r.row.clone().finish()).collect(),
        };
        match serde_json::to_string_pretty(&report)
            .map_err(io::Error::other)
            .and_then(|s| std::fs::write(&job.req.report_out, s))
        {
            Ok(()) => {
                eprintln!(
                    "[hub] job {} done in {} s -> {}",
                    job.id,
                    job.started.elapsed().as_secs(),
                    job.req.report_out.display()
                );
                job.state = JobState::Done;
            }
            Err(e) => {
                job.state = JobState::Failed {
                    error: format!("writing {}: {e}", job.req.report_out.display()),
                }
            }
        }
        job.report = Some(report);
    }

    /// Any job still running?
    pub fn busy(&self) -> bool {
        self.jobs.values().any(|j| j.state == JobState::Running)
    }
}

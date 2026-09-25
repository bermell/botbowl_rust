//! `botbowl-worker`: dial a hub, play what it hands out, stream results back.
//!
//! Layout (plan 041):
//!
//! - one tokio task reads hub frames: `Model` goes to the on-disk
//!   [`ModelStore`], `Task` goes onto a queue, `Drain` closes the queue;
//! - `parallel_games` **std** threads pull tasks off that queue and play
//!   them with `botbowl_play` — games are sync and `MctsBot` spawns its own
//!   scoped threads, so they must not run on the async runtime;
//! - one tokio task drains a result channel to the socket.
//!
//! The result channel outlives a connection: games that finish while the
//! hub is unreachable are sent on reconnect, and the hub dedupes on
//! `(job, rung, game)`. Un-started tasks are dropped on disconnect because
//! the hub requeues them elsewhere.

mod mem_governor;

use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::model::BoardDims;
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_hub_proto::{
    decode, encode, BotSpec, BuildInfo, ModelId, RejectReason, Task, ToHub, ToWorker, PROTOCOL_VERSION,
};
use botbowl_mcts::SearchBudget;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::bots::make_mcts;
use botbowl_play::eval::{ladder_assignment, play_ladder_game};
use botbowl_play::generate::play_trajectory;
use botbowl_play::GAME_STACK_SIZE;
use mem_governor::{GameSlot, MemGovernor};

/// zstd level for trajectory frames: ~575 KB of JSON -> ~30 KB, fast
/// enough to be invisible next to the game that produced it.
const TRAJECTORY_ZSTD_LEVEL: i32 = 3;

/// Default floor left unpredicted-for (MB) below which a game thread backs
/// off rather than starting its next game. Leaves headroom for the
/// nn_server sidecar, the hub, and (on the local worker) a desktop session
/// sharing the box — see `mem_governor` for why this exists at all.
pub const DEFAULT_MEM_FLOOR_MB: u32 = 1024;

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// `ws://host:port/ws`.
    pub hub_url: String,
    pub token: String,
    pub name: String,
    /// `None` lets the hub size it from cores and RAM.
    pub parallel_games: Option<u16>,
    /// GPU sidecar socket for the local worker (`botbowl-nn` remote mode).
    pub nn_server: Option<PathBuf>,
    /// Where `<model-id>.onnx` files live between runs.
    pub cache_dir: PathBuf,
    /// Memory headroom (MB) a game thread keeps in reserve, on top of its
    /// predicted tree cost, before starting its next game.
    pub mem_floor_mb: u32,
    /// Longest wait between connection attempts. The hub is restarted whenever the code changes,
    /// so this is the worst case for a helper box noticing it is back.
    pub reconnect_max: Duration,
}

/// Where the reconnect backoff starts, and what it resets to after a connection that worked.
const RECONNECT_MIN: Duration = Duration::from_secs(5);

/// Default for [`WorkerConfig::reconnect_max`].
pub const DEFAULT_RECONNECT_MAX_SECS: u64 = 30;

#[derive(Debug)]
pub enum Fatal {
    Rejected(RejectReason),
    Io(io::Error),
}

impl std::fmt::Display for Fatal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fatal::Rejected(r) => write!(f, "hub rejected us: {r}"),
            Fatal::Io(e) => write!(f, "{e}"),
        }
    }
}

impl From<io::Error> for Fatal {
    fn from(e: io::Error) -> Self {
        Fatal::Io(e)
    }
}

/// Why one connection ended. Only `Rejected` stops the reconnect loop.
#[derive(Debug)]
pub enum Ended {
    /// The hub said `Drain` and we finished our in-flight games.
    Drained,
    /// Socket closed or errored; reconnect.
    Lost(String),
}

// ---------------------------------------------------------------------------
// Model cache

pub struct ModelStore {
    dir: PathBuf,
    server: Option<PathBuf>,
    loaded: Mutex<HashMap<ModelId, Arc<NnEvaluator>>>,
}

impl ModelStore {
    pub fn open(dir: &Path, server: Option<PathBuf>) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(ModelStore {
            dir: dir.to_path_buf(),
            server,
            loaded: Mutex::new(HashMap::new()),
        })
    }

    fn path(&self, id: &ModelId) -> PathBuf {
        self.dir.join(format!("{}.onnx", id.to_hex()))
    }

    /// Ids on disk. Verified by content hash so a truncated download can
    /// never masquerade as a model.
    pub fn cached_ids(&self) -> Vec<ModelId> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return out;
        };
        for e in rd.flatten() {
            let name = e.file_name();
            let Some(stem) = name.to_str().and_then(|s| s.strip_suffix(".onnx")) else {
                continue;
            };
            let Some(id) = ModelId::from_hex(stem) else { continue };
            match std::fs::read(e.path()) {
                Ok(bytes) if ModelId::of(&bytes) == id => out.push(id),
                _ => {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        out
    }

    pub fn put(&self, id: ModelId, onnx: &[u8]) -> io::Result<()> {
        if ModelId::of(onnx) != id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model bytes do not hash to their id",
            ));
        }
        let tmp = self.path(&id).with_extension("onnx.part");
        std::fs::write(&tmp, onnx)?;
        std::fs::rename(&tmp, self.path(&id))
    }

    pub fn has(&self, id: &ModelId) -> bool {
        self.loaded.lock().unwrap().contains_key(id) || self.path(id).exists()
    }

    /// Load (once) and share. The net is frozen, so every bot in every
    /// game shares the `Arc`.
    pub fn get(&self, id: &ModelId) -> io::Result<Arc<NnEvaluator>> {
        if let Some(e) = self.loaded.lock().unwrap().get(id) {
            return Ok(Arc::clone(e));
        }
        let path = self.path(id);
        let eval = NnEvaluator::from_path_with_server(&path, self.server.as_deref()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to load {}: {e}", path.display()),
            )
        })?;
        // Probe one forward on a fresh state *outside* any search. A net
        // whose encoder schema does not match this binary makes tract panic
        // on first use; inside `MctsBot` that poisons the tree's locks and
        // the unwinding drop aborts the whole worker (seen 2026-09-16 with a
        // 53-channel net on a 75-channel binary). Here it is a plain error
        // the hub gets as `TaskFailed`.
        let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let state = botbowl_engine::core::gamestate::GameStateBuilder::new().build();
            eval.value_home_i64(&state)
        }));
        if probe.is_err() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model {} is incompatible with this binary's encoder (probe forward panicked)",
                    id.to_hex()
                ),
            ));
        }
        let eval = Arc::new(eval);
        self.loaded.lock().unwrap().insert(*id, Arc::clone(&eval));
        Ok(eval)
    }
}

pub fn make_bot(spec: &BotSpec, store: &ModelStore) -> io::Result<Box<dyn Bot>> {
    Ok(match spec {
        BotSpec::Random => Box::new(RandomBot::new()),
        BotSpec::Scripted => Box::new(ScriptedBot::new()),
        BotSpec::Mcts {
            search,
            evaluator,
            model,
        } => {
            let nn = match (evaluator.needs_model(), model) {
                (true, Some(id)) => Some(store.get(id)?),
                (true, None) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "nn evaluator without a model id",
                    ))
                }
                (false, _) => None,
            };
            Box::new(make_mcts(search, *evaluator, nn.as_ref()))
        }
    })
}

// ---------------------------------------------------------------------------
// Task queue shared between the reader task and the game threads

#[derive(Default)]
struct QueueState {
    tasks: VecDeque<Task>,
    /// Set by `Drain` or disconnect: threads exit once the queue is empty.
    closed: bool,
}

struct Queue {
    state: Mutex<QueueState>,
    cv: Condvar,
}

impl Queue {
    fn new() -> Self {
        Queue {
            state: Mutex::new(QueueState::default()),
            cv: Condvar::new(),
        }
    }
    fn push(&self, t: Task) {
        self.state.lock().unwrap().tasks.push_back(t);
        self.cv.notify_one();
    }
    fn close(&self) {
        let mut s = self.state.lock().unwrap();
        s.closed = true;
        s.tasks.clear();
        self.cv.notify_all();
    }
    /// `None` once closed and empty.
    fn pop(&self) -> Option<Task> {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(t) = s.tasks.pop_front() {
                return Some(t);
            }
            if s.closed {
                return None;
            }
            s = self.cv.wait(s).unwrap();
        }
    }
}

/// Playable cell count of a board — the unit `MemGovernor` predicts tree
/// memory from. Matches `botbowl_play::board_sizes`'s private `playable_area`
/// (engine dims include a 1-cell border on each side).
fn cell_area(d: BoardDims) -> u32 {
    (d.width as i64 - 2).max(0) as u32 * (d.height as i64 - 2).max(0) as u32
}

/// Iteration budget of a bot, for the memory prediction. Search-free bots hold no tree at all,
/// hence `Some(0)`; a time budget is unknowable up front, hence `None`.
fn iters_of(spec: &BotSpec) -> Option<u64> {
    match spec {
        BotSpec::Random | BotSpec::Scripted => Some(0),
        BotSpec::Mcts { search, .. } => budget_iters(&search.budget),
    }
}

fn budget_iters(budget: &SearchBudget) -> Option<u64> {
    match budget {
        SearchBudget::Iterations(n) => Some(*n as u64),
        SearchBudget::Time(_) => None,
    }
}

/// Blocks (with backoff) until the governor judges a game of `area` cost units
/// safe to start, then accounts for it; logs once per stall so a stuck
/// worker is visible in its log rather than silently idle.
fn admit_game<'a>(governor: &'a MemGovernor, area: u32, ctx: &str) -> GameSlot<'a> {
    let mut warned = false;
    loop {
        let avail_kb = match available_memory_mb() {
            Some(mb) => mb as u64 * 1024,
            // Can't read memory on this platform/box — never block on a
            // signal we don't have.
            None => return governor.admit_unconditionally(area),
        };
        // Calibrate from what's already reserved and (mostly) manifested,
        // before this call's own reservation attempt changes that count.
        governor.observe(avail_kb, governor.active_area());
        if let Some(slot) = governor.try_admit(area, avail_kb) {
            return slot;
        }
        if !warned {
            eprintln!(
                "[worker] {ctx}: holding back a game costing {area} cell-kiloiterations, {} MB \
                 available, {} MB/cost-unit estimate against a {} MB baseline — waiting for headroom",
                avail_kb / 1024,
                governor.predicted_cost_kb(1) / 1024,
                governor.baseline_available_kb() / 1024
            );
            warned = true;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn run_task(task: &Task, store: &ModelStore, out: &mpsc::UnboundedSender<ToHub>, governor: &MemGovernor) {
    match task {
        Task::Eval {
            id,
            rung,
            games,
            seed,
            max_steps,
            candidate,
            opponent,
            board,
        } => {
            let (mut cand, mut opp) = match (make_bot(candidate, store), make_bot(opponent, store)) {
                (Ok(c), Ok(o)) => (c, o),
                (Err(e), _) | (_, Err(e)) => {
                    let _ = out.send(ToHub::TaskFailed {
                        task: *id,
                        error: format!("bot construction: {e}"),
                    });
                    return;
                }
            };
            // Two trees per game (candidate + opponent) against `Generate`'s one, each scaled by
            // its own budget — an asymmetric arm (plan 045 runs 4000 against 1000) costs the sum,
            // not twice the larger or twice the smaller.
            let cells = cell_area(board.unwrap_or_else(BoardDims::from_env));
            let area = mem_governor::cost_units(cells, iters_of(candidate))
                + mem_governor::cost_units(cells, iters_of(opponent));
            for &g in games {
                let (team, game_seed) = ladder_assignment(*seed, g);
                let _slot = admit_game(governor, area, rung);
                // No `--trace-reuse` on the distributed path: a per-decision trace is a local
                // diagnostic, and the telemetry the hub's report needs already rides in the line.
                let line = play_ladder_game(
                    &mut *cand, &mut *opp, rung, g, team, game_seed, *max_steps, *board, None,
                );
                // Send failures mean the hub is gone; the channel is
                // unbounded and outlives the socket, so this only fails
                // when the whole worker is shutting down.
                let _ = out.send(ToHub::EvalGameDone { task: *id, line });
            }
        }
        Task::Generate {
            id,
            shard,
            games,
            seed_base,
            cfg,
            model,
        } => {
            let nn = match (cfg.evaluator.needs_model(), model) {
                (true, Some(m)) => match store.get(m) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        let _ = out.send(ToHub::TaskFailed {
                            task: *id,
                            error: format!("model: {e}"),
                        });
                        return;
                    }
                },
                (true, None) => {
                    let _ = out.send(ToHub::TaskFailed {
                        task: *id,
                        error: "nn evaluator without a model id".into(),
                    });
                    return;
                }
                (false, _) => None,
            };
            for &g in games {
                // Same numbering as `botbowl-ui dataset --seed seed_base`.
                let seed = seed_base.wrapping_add(g as u64);
                let area = mem_governor::cost_units(
                    cell_area(cfg.board_for(seed).unwrap_or_else(BoardDims::from_env)),
                    budget_iters(&cfg.search.budget),
                );
                let _slot = admit_game(governor, area, shard);
                match play_trajectory(cfg, nn.as_ref(), seed) {
                    Err(e) => {
                        // A configuration error (unknown lecture) recurs on
                        // every seed; fail the whole task.
                        let _ = out.send(ToHub::TaskFailed { task: *id, error: e });
                        return;
                    }
                    Ok(None) => {
                        let _ = out.send(ToHub::TrajectoryDone {
                            task: *id,
                            game: g,
                            samples: 0,
                            zstd_json: Vec::new(),
                        });
                    }
                    Ok(Some(traj)) => {
                        let json = serde_json::to_vec(&traj).expect("trajectory serializes");
                        let zstd_json = zstd::encode_all(&json[..], TRAJECTORY_ZSTD_LEVEL).expect("zstd encode");
                        eprintln!(
                            "[worker] {shard} seed={seed} samples={} z_home={:+} score={}-{} ({} KB -> {} KB)",
                            traj.samples.len(),
                            traj.outcome.z_home,
                            traj.outcome.home_score,
                            traj.outcome.away_score,
                            json.len() / 1024,
                            zstd_json.len() / 1024
                        );
                        let _ = out.send(ToHub::TrajectoryDone {
                            task: *id,
                            game: g,
                            samples: traj.samples.len() as u32,
                            zstd_json,
                        });
                    }
                }
            }
        }
    }
}

fn spawn_game_threads(
    n: u16,
    queue: Arc<Queue>,
    store: Arc<ModelStore>,
    out: mpsc::UnboundedSender<ToHub>,
    in_flight: Arc<AtomicU16>,
    governor: Arc<MemGovernor>,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..n)
        .map(|i| {
            let queue = Arc::clone(&queue);
            let store = Arc::clone(&store);
            let out = out.clone();
            let in_flight = Arc::clone(&in_flight);
            let governor = Arc::clone(&governor);
            std::thread::Builder::new()
                .name(format!("game-{i}"))
                .stack_size(GAME_STACK_SIZE)
                .spawn(move || {
                    while let Some(task) = queue.pop() {
                        in_flight.fetch_add(1, Ordering::Relaxed);
                        // A panic inside a game (an engine bug, a net whose
                        // schema does not match this binary) must reach the
                        // hub as a failure, not leave the task in flight
                        // forever on a worker that is still heartbeating.
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_task(&task, &store, &out, &governor)
                        }));
                        if let Err(p) = r {
                            let msg = p
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "panic".to_string());
                            let _ = out.send(ToHub::TaskFailed {
                                task: task.id(),
                                error: format!("game panicked: {msg}"),
                            });
                        }
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                    }
                })
                .expect("spawn game thread")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One connection

fn hello(cfg: &WorkerConfig, store: &ModelStore) -> ToHub {
    ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: cfg.token.clone(),
        build: BuildInfo::current(),
        triple: target_triple().to_string(),
        name: cfg.name.clone(),
        cores: std::thread::available_parallelism()
            .map(|n| n.get() as u16)
            .unwrap_or(1),
        ram_mb: total_ram_mb(),
        cached_models: store.cached_ids(),
        parallel_games: cfg.parallel_games,
    }
}

/// Connect, handshake, serve until the hub drains us or the socket drops.
/// `results` carries games finished under any earlier connection too.
pub async fn run_once(
    cfg: &WorkerConfig,
    store: Arc<ModelStore>,
    results: &mut mpsc::UnboundedReceiver<ToHub>,
    results_tx: &mpsc::UnboundedSender<ToHub>,
) -> Result<Ended, Fatal> {
    let (ws, _) = tokio_tungstenite::connect_async(&cfg.hub_url)
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, format!("{}: {e}", cfg.hub_url)))?;
    let (mut sink, mut stream) = ws.split();

    sink.send(Message::Binary(encode(&hello(cfg, &store)).into()))
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?;

    let parallel = match stream.next().await {
        Some(Ok(Message::Binary(b))) => match decode::<ToWorker>(&b) {
            Ok(ToWorker::Welcome { parallel_games }) => parallel_games.max(1),
            Ok(ToWorker::Reject { reason }) => return Err(Fatal::Rejected(reason)),
            Ok(other) => return Ok(Ended::Lost(format!("expected Welcome, got {other:?}"))),
            Err(e) => return Ok(Ended::Lost(format!("undecodable first frame: {e}"))),
        },
        other => return Ok(Ended::Lost(format!("no Welcome: {other:?}"))),
    };
    eprintln!(
        "[worker] connected to {} as {:?}, {parallel} streams",
        cfg.hub_url, cfg.name
    );

    let queue = Arc::new(Queue::new());
    let in_flight = Arc::new(AtomicU16::new(0));
    // Sampled fresh each connection, before this connection's game threads
    // exist — see `mem_governor` for why a static thread-pool size can't
    // track a variable board-size curriculum on its own.
    let governor = Arc::new(MemGovernor::new(cfg.mem_floor_mb, available_memory_mb()));
    let threads = spawn_game_threads(
        parallel,
        Arc::clone(&queue),
        Arc::clone(&store),
        results_tx.clone(),
        Arc::clone(&in_flight),
        governor,
    );

    // Writer: results + heartbeats out. Owns the sink.
    let (ctl_tx, mut ctl_rx) = mpsc::unbounded_channel::<Message>();
    let hb_in_flight = Arc::clone(&in_flight);
    let hb_ctl = ctl_tx.clone();
    let heartbeat = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        tick.tick().await;
        loop {
            tick.tick().await;
            let msg = ToHub::Heartbeat {
                games_in_flight: hb_in_flight.load(Ordering::Relaxed),
            };
            if hb_ctl.send(Message::Binary(encode(&msg).into())).is_err() {
                return;
            }
        }
    });

    let mut ended = Ended::Lost("socket closed".into());
    let mut draining = false;
    loop {
        tokio::select! {
            frame = stream.next() => {
                match frame {
                    Some(Ok(Message::Binary(b))) => match decode::<ToWorker>(&b) {
                        Ok(ToWorker::Model { id, onnx }) => {
                            if let Err(e) = store.put(id, &onnx) {
                                eprintln!("[worker] rejecting model {id}: {e}");
                            }
                        }
                        Ok(ToWorker::Task(task)) => {
                            let missing: Vec<ModelId> = task.models().into_iter().filter(|m| !store.has(m)).collect();
                            if missing.is_empty() {
                                queue.push(task);
                            } else {
                                let msg = ToHub::TaskFailed {
                                    task: task.id(),
                                    error: format!("missing models {missing:?}"),
                                };
                                let _ = ctl_tx.send(Message::Binary(encode(&msg).into()));
                            }
                        }
                        Ok(ToWorker::Drain) => {
                            draining = true;
                            queue.close();
                        }
                        Ok(ToWorker::Welcome { .. }) | Ok(ToWorker::Reject { .. }) => {}
                        Err(e) => {
                            ended = Ended::Lost(format!("undecodable frame: {e}"));
                            break;
                        }
                    },
                    Some(Ok(Message::Ping(p))) => { let _ = ctl_tx.send(Message::Pong(p)); }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => { ended = Ended::Lost(e.to_string()); break; }
                }
            }
            Some(msg) = results.recv() => {
                if sink.send(Message::Binary(encode(&msg).into())).await.is_err() {
                    // Put it back for the next connection.
                    let _ = results_tx.send(msg);
                    ended = Ended::Lost("send failed".into());
                    break;
                }
            }
            Some(msg) = ctl_rx.recv() => {
                if sink.send(msg).await.is_err() {
                    ended = Ended::Lost("send failed".into());
                    break;
                }
            }
            // While draining nothing else may arrive; poll for "all quiet".
            _ = tokio::time::sleep(Duration::from_millis(250)), if draining => {}
        }
        if draining && in_flight.load(Ordering::Relaxed) == 0 && queue.state.lock().unwrap().tasks.is_empty() {
            // Flush whatever the threads produced, then leave.
            while let Ok(msg) = results.try_recv() {
                let _ = sink.send(Message::Binary(encode(&msg).into())).await;
            }
            let _ = sink.send(Message::Close(None)).await;
            ended = Ended::Drained;
            break;
        }
    }

    heartbeat.abort();
    queue.close();
    // Game threads finish their current task (results land in the channel
    // for the next connection) and exit.
    tokio::task::spawn_blocking(move || {
        for t in threads {
            let _ = t.join();
        }
    })
    .await
    .ok();
    Ok(ended)
}

/// Reconnect loop. Returns only on a bad token or after a `Drain`.
///
/// A helper box is expected to outlive many hub restarts: the hub goes down whenever the training
/// box takes a new commit, and it comes back within seconds. So the backoff is short (5 s,
/// doubling to [`WorkerConfig::reconnect_max`], 30 s by default) and — the part that used to be
/// missing — **resets after every connection that worked**, instead of creeping to five minutes
/// and staying there for the rest of the week.
///
/// A rejection is also retried rather than fatal, with one exception. The hub restarted on a new
/// commit is exactly the case the operator resolves *on the hub* — by updating
/// `hub-allowed-commits.toml` — and a worker that has exited cannot benefit from that. So it keeps
/// dialling and rejoins by itself, typically before anyone looks. `BadToken` is the exception: no
/// action on the hub makes a wrong secret right, so that one stops.
pub async fn run(cfg: WorkerConfig) -> Result<(), Fatal> {
    let store = Arc::new(ModelStore::open(&cfg.cache_dir, cfg.nn_server.clone())?);
    let (tx, mut rx) = mpsc::unbounded_channel::<ToHub>();
    let mut backoff = RECONNECT_MIN;
    // Say a repeated reason once, not every 30 s forever: a worker waiting out a commit mismatch
    // overnight should leave a readable log, not 1200 identical lines.
    let mut last_complaint: Option<String> = None;
    loop {
        let outcome = run_once(&cfg, Arc::clone(&store), &mut rx, &tx).await;
        let why = match outcome {
            Ok(Ended::Drained) => return Ok(()),
            Ok(Ended::Lost(why)) => {
                // We got as far as a working connection, so the next outage starts over.
                backoff = RECONNECT_MIN;
                last_complaint = None;
                format!("connection lost ({why})")
            }
            Err(Fatal::Rejected(RejectReason::BadToken)) => {
                return Err(Fatal::Rejected(RejectReason::BadToken));
            }
            Err(Fatal::Rejected(r)) => format!("hub rejected us: {r}"),
            Err(Fatal::Io(e)) => e.to_string(),
        };
        if last_complaint.as_deref() != Some(why.as_str()) {
            eprintln!("[worker] {why}; retrying every {}s until it changes", backoff.as_secs());
            last_complaint = Some(why);
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(cfg.reconnect_max.max(RECONNECT_MIN));
    }
}

fn target_triple() -> &'static str {
    // Enough to pick a download in phase 4; refined there.
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("x86_64", "linux") => "x86_64-unknown-linux-musl",
        ("aarch64", "linux") => "aarch64-unknown-linux-musl",
        ("aarch64", "macos") => "aarch64-apple-darwin",
        ("x86_64", "macos") => "x86_64-apple-darwin",
        ("x86_64", "windows") => "x86_64-pc-windows-gnu",
        _ => "unknown",
    }
}

fn total_ram_mb() -> u32 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                    return (kb / 1024) as u32;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(b) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                return (b / (1024 * 1024)) as u32;
            }
        }
    }
    0
}

/// Memory the kernel would hand out right now without swapping — the
/// signal `MemGovernor` gates game admission on. `None` when it can't be
/// read, which `admit_game` treats as "never block" rather than guessing.
fn available_memory_mb() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                    return Some((kb / 1024) as u32);
                }
            }
        }
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        // `vm_stat` reports page counts, not MB. Free + inactive + speculative
        // pages approximate what the kernel would hand out before it would
        // need to compress or swap — inactive pages are reclaimable, and
        // speculative pages are read-ahead nothing has actually touched.
        let out = std::process::Command::new("vm_stat").output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let page_size = text
            .lines()
            .next()
            .and_then(|l| l.split("page size of").nth(1))
            .and_then(|s| s.trim().split(' ').next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(4096);
        let page_count = |key: &str| -> u64 {
            text.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|n| n.trim().trim_end_matches('.').parse::<u64>().ok())
                .unwrap_or(0)
        };
        let pages = page_count("Pages free") + page_count("Pages inactive") + page_count("Pages speculative");
        return Some(((pages * page_size) / (1024 * 1024)) as u32);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    None
}

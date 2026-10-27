//! Plan 069 item 6: a load-test worker for the hub. It speaks the real protocol — the same
//! `Hello` (this binary's commit, capacity and env board, so the hub admits it exactly as it
//! admits `botbowl-worker`), heartbeats, `Model` frames taken and dropped — but plays nothing:
//! a generate game is a sleep followed by a canned trajectory (real corpus lines, zstd'd once at
//! startup, sent as `TrajectoryDone` the way the worker sends them), a label item a sleep followed
//! by `LabelDone` with one label per sample. One process opens many connections (`--connections`,
//! each `--streams` wide), so a single box can put thousands of streams on one hub.
//!
//! What it measures, from the worker's side of the socket:
//! - **dispatch latency**: from a task's last result leaving this process to the next task
//!   arriving on that connection — the hub's whole per-result path (decode, lock, book-keeping,
//!   the shard write, dispatch, encode) plus the network, which is what an idle stream waits;
//! - **connect latency**: dial to `Welcome`, for a fleet that dials at once;
//! - results and bytes per second each way.
//!
//! It prints one JSON line per `--report-every` seconds and a final one (`"final": true`) on
//! `--duration-secs`, SIGINT or SIGTERM; `scripts/hub_scale.py` drives it. Eval tasks are refused
//! with `TaskFailed` (an `EvalGameLine` is not worth faking): submit generate and label jobs only.
//!
//!   botbowl-fake-worker --hub ws://127.0.0.1:14000/ws --connections 16 --streams 64 \
//!       --canned shard.jsonl --gen-secs 60 --label-secs 20

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use botbowl_hub_proto::{decode, encode, BuildInfo, LabelResult, Task, ToHub, ToWorker, PROTOCOL_VERSION};

/// zstd level the real worker uses for trajectories (`TRAJECTORY_ZSTD_LEVEL`).
const ZSTD_LEVEL: i32 = 3;

#[derive(Parser, Debug)]
#[command(name = "botbowl-fake-worker")]
struct Cli {
    /// Hub websocket URL, e.g. ws://127.0.0.1:14000/ws
    #[arg(long)]
    hub: String,
    /// Token file; default the machine token (`~/.config/botbowl/hub.token`).
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// Connections (simulated worker processes).
    #[arg(long, default_value_t = 1)]
    connections: u32,
    /// Streams per connection (`Hello.parallel_games`).
    #[arg(long, default_value_t = 64)]
    streams: u16,
    /// Seconds a generate game takes (before jitter).
    #[arg(long, default_value_t = 60.0)]
    gen_secs: f64,
    /// Seconds a label item takes (before jitter).
    #[arg(long, default_value_t = 20.0)]
    label_secs: f64,
    /// Each sleep is uniform in `[1 - jitter, 1 + jitter]` times its base.
    #[arg(long, default_value_t = 0.3)]
    jitter: f64,
    /// Corpus lines to send as generate results, round robin; a next-drive record
    /// (`meta.extra.drive` 2) travels with the line before it, as one game, like the worker sends.
    #[arg(long)]
    canned: Option<PathBuf>,
    /// Without `--canned`: a synthetic line of this many KB (compresses worse than a real one).
    #[arg(long, default_value_t = 130)]
    synthetic_kb: usize,
    /// Milliseconds between opening successive connections (0 = all at once).
    #[arg(long, default_value_t = 0)]
    ramp_ms: u64,
    /// Seconds between report lines.
    #[arg(long, default_value_t = 10)]
    report_every: u64,
    /// Exit (with the final report) after this long; 0 = run until SIGINT/SIGTERM.
    #[arg(long, default_value_t = 0)]
    duration_secs: u64,
    /// Connection names are `<prefix>-<i>`.
    #[arg(long, default_value = "fake")]
    name_prefix: String,
}

/// One canned game: what a `TrajectoryDone` carries.
struct Game {
    zstd: Vec<u8>,
    samples: u32,
    raw_len: usize,
}

fn load_canned(path: &PathBuf) -> Vec<Game> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("[fake] cannot read {}: {e}", path.display());
        std::process::exit(2)
    });
    // Group a next-drive record with its first drive: one game, newline-separated, as the worker
    // sends them (plan 047).
    let mut groups: Vec<(Vec<u8>, u32)> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
            eprintln!("[fake] {}: not a JSON line: {e}", path.display());
            std::process::exit(2)
        });
        let samples = v["samples"].as_array().map_or(0, |a| a.len()) as u32;
        let follow_on = v["meta"]["extra"]["drive"].as_str() == Some("2");
        match groups.last_mut() {
            Some((json, n)) if follow_on => {
                json.push(b'\n');
                json.extend_from_slice(line.as_bytes());
                *n += samples;
            }
            _ => groups.push((line.as_bytes().to_vec(), samples)),
        }
    }
    if groups.is_empty() {
        eprintln!("[fake] {}: no lines", path.display());
        std::process::exit(2)
    }
    groups
        .into_iter()
        .map(|(json, samples)| Game {
            zstd: zstd::encode_all(&json[..], ZSTD_LEVEL).expect("zstd encode"),
            samples,
            raw_len: json.len(),
        })
        .collect()
}

/// A JSON-shaped line of `kb` KB with varied small numbers, for runs without a corpus at hand.
fn synthetic(kb: usize) -> Vec<Game> {
    let mut s = String::from("{\"meta\":{\"seed\":0},\"samples\":[");
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut n = 0u32;
    while s.len() < kb * 1024 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"a\":{},\"p\":[{},{},{}],\"v\":{}}}",
            x % 97,
            x % 7,
            (x >> 8) % 13,
            (x >> 16) % 5,
            (x >> 24) % 1000
        ));
        n += 1;
    }
    s.push_str("]}");
    vec![Game {
        zstd: zstd::encode_all(s.as_bytes(), ZSTD_LEVEL).expect("zstd encode"),
        samples: n,
        raw_len: s.len(),
    }]
}

/// Latency samples in microseconds: the current report interval's, and the whole run's.
#[derive(Default)]
struct Hist {
    interval: Vec<u32>,
    all: Vec<u32>,
}

impl Hist {
    fn record(&mut self, d: Duration) {
        let us = d.as_micros().min(u32::MAX as u128) as u32;
        self.interval.push(us);
        self.all.push(us);
    }
}

fn summary(v: &mut [u32]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::json!({ "n": 0 });
    }
    v.sort_unstable();
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64 / 1000.0;
    serde_json::json!({
        "n": v.len(),
        "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": q(1.0),
        "mean": v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 / 1000.0,
    })
}

#[derive(Default)]
struct Stats {
    connected: AtomicU64,
    tasks: AtomicU64,
    results: AtomicU64,
    gen_results: AtomicU64,
    label_results: AtomicU64,
    refused: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_recv: AtomicU64,
    model_frames: AtomicU64,
    reconnects: AtomicU64,
    dispatch: Mutex<Hist>,
    connect: Mutex<Hist>,
}

struct Shared {
    cli: Cli,
    canned: Vec<Game>,
    stats: Stats,
    next_game: AtomicU64,
}

/// Uniform in `[1 - j, 1 + j]` times `base`, from a hash of `(task, k)` — no rand dependency, and
/// two runs of one job sleep alike.
fn jittered(base: f64, j: f64, task: u64, k: u64) -> Duration {
    let mut x = task.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ k.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 29;
    let u = (x >> 11) as f64 / (1u64 << 53) as f64;
    Duration::from_secs_f64((base * (1.0 + j * (2.0 * u - 1.0))).max(0.0))
}

/// One task, played as sleeps. Its completion time goes on the connection's queue: the next task
/// that arrives pops it, and the difference is the dispatch latency.
async fn play(task: Task, sh: Arc<Shared>, tx: mpsc::UnboundedSender<ToHub>, done: Arc<Mutex<VecDeque<Instant>>>) {
    let cli = &sh.cli;
    match task {
        Task::Generate { id, games, .. } => {
            for (k, g) in games.into_iter().enumerate() {
                tokio::time::sleep(jittered(cli.gen_secs, cli.jitter, id, k as u64)).await;
                let i = sh.next_game.fetch_add(1, Ordering::Relaxed) as usize % sh.canned.len();
                let game = &sh.canned[i];
                let _ = tx.send(ToHub::TrajectoryDone {
                    task: id,
                    game: g,
                    samples: game.samples,
                    zstd_json: game.zstd.clone(),
                });
                sh.stats.gen_results.fetch_add(1, Ordering::Relaxed);
            }
        }
        Task::Label { id, items, .. } => {
            for (k, it) in items.into_iter().enumerate() {
                tokio::time::sleep(jittered(cli.label_secs, cli.jitter, id, k as u64)).await;
                let _ = tx.send(ToHub::LabelDone {
                    task: id,
                    item: it.item,
                    labels: LabelResult::Labels(vec![0.0; it.end.saturating_sub(it.start) as usize]),
                });
                sh.stats.label_results.fetch_add(1, Ordering::Relaxed);
            }
        }
        Task::Eval { id, .. } => {
            sh.stats.refused.fetch_add(1, Ordering::Relaxed);
            let _ = tx.send(ToHub::TaskFailed {
                task: id,
                error: "botbowl-fake-worker plays no eval games".into(),
            });
            return;
        }
    }
    done.lock().unwrap().push_back(Instant::now());
}

/// What ended a connection.
enum End {
    Lost(String),
    Fatal(String),
}

async fn connect_once(i: u32, sh: &Arc<Shared>, token: &str) -> End {
    let cli = &sh.cli;
    let ws_cfg = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_frame_size(Some(64 << 20))
        .max_message_size(Some(64 << 20));
    let t0 = Instant::now();
    let ws = match tokio_tungstenite::connect_async_with_config(&cli.hub, Some(ws_cfg), false).await {
        Ok((ws, _)) => ws,
        Err(e) => return End::Lost(format!("dial: {e}")),
    };
    let (mut sink, mut stream) = ws.split();
    let hello = ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: token.to_string(),
        build: BuildInfo::current(),
        triple: "fake".into(),
        name: format!("{}-{i:04}", cli.name_prefix),
        cores: cli.streams,
        ram_mb: 0,
        cached_models: Vec::new(),
        parallel_games: Some(cli.streams),
        label_streams: 0,
    };
    if let Err(e) = sink.send(Message::Binary(encode(&hello).into())).await {
        return End::Lost(format!("hello: {e}"));
    }
    match stream.next().await {
        Some(Ok(Message::Binary(b))) => match decode::<ToWorker>(&b) {
            Ok(ToWorker::Welcome { .. }) => {}
            Ok(ToWorker::Reject { reason }) => return End::Fatal(format!("rejected: {reason}")),
            other => return End::Lost(format!("expected Welcome, got {other:?}")),
        },
        other => return End::Lost(format!("no Welcome: {other:?}")),
    }
    sh.stats.connect.lock().unwrap().record(t0.elapsed());
    sh.stats.connected.fetch_add(1, Ordering::Relaxed);

    let (tx, mut rx) = mpsc::unbounded_channel::<ToHub>();
    let done: Arc<Mutex<VecDeque<Instant>>> = Arc::default();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(30));
    heartbeat.tick().await;
    let end = loop {
        tokio::select! {
            frame = stream.next() => match frame {
                Some(Ok(Message::Binary(b))) => {
                    sh.stats.bytes_recv.fetch_add(b.len() as u64, Ordering::Relaxed);
                    match decode::<ToWorker>(&b) {
                        Ok(ToWorker::Task(task)) => {
                            if let Some(t) = done.lock().unwrap().pop_front() {
                                sh.stats.dispatch.lock().unwrap().record(t.elapsed());
                            }
                            sh.stats.tasks.fetch_add(1, Ordering::Relaxed);
                            tokio::spawn(play(task, Arc::clone(sh), tx.clone(), Arc::clone(&done)));
                        }
                        Ok(ToWorker::Model { .. }) => {
                            sh.stats.model_frames.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(ToWorker::Drain) => break End::Lost("drained".into()),
                        Ok(_) => {}
                        Err(e) => break End::Lost(format!("undecodable frame: {e}")),
                    }
                }
                Some(Ok(Message::Close(_))) | None => break End::Lost("socket closed".into()),
                Some(Ok(_)) => {}
                Some(Err(e)) => break End::Lost(e.to_string()),
            },
            Some(msg) = rx.recv() => {
                let bytes = encode(&msg);
                sh.stats.bytes_sent.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                sh.stats.results.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = sink.send(Message::Binary(bytes.into())).await {
                    break End::Lost(format!("send: {e}"));
                }
            }
            _ = heartbeat.tick() => {
                let hb = encode(&ToHub::Heartbeat { games_in_flight: 0 });
                if let Err(e) = sink.send(Message::Binary(hb.into())).await {
                    break End::Lost(format!("heartbeat: {e}"));
                }
            }
        }
    };
    sh.stats.connected.fetch_sub(1, Ordering::Relaxed);
    end
}

async fn connection(i: u32, sh: Arc<Shared>, token: String) {
    loop {
        match connect_once(i, &sh, &token).await {
            End::Fatal(why) => {
                eprintln!("[fake] connection {i}: {why}");
                std::process::exit(1);
            }
            End::Lost(why) => {
                // Games in flight under the lost connection are dropped: the hub requeues them,
                // as it does for a real worker.
                if sh.stats.reconnects.fetch_add(1, Ordering::Relaxed) < 20 {
                    eprintln!("[fake] connection {i}: {why}; redialling in 5 s");
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

fn report(sh: &Shared, t0: Instant, last: &mut (Instant, u64, u64, u64, u64), fin: bool) -> String {
    let s = &sh.stats;
    let now = Instant::now();
    let dt = now.duration_since(last.0).as_secs_f64().max(1e-9);
    let (results, tasks, sent, recv) = (
        s.results.load(Ordering::Relaxed),
        s.tasks.load(Ordering::Relaxed),
        s.bytes_sent.load(Ordering::Relaxed),
        s.bytes_recv.load(Ordering::Relaxed),
    );
    let (dispatch, connect) = {
        let mut d = s.dispatch.lock().unwrap();
        let mut c = s.connect.lock().unwrap();
        let out = if fin {
            (summary(&mut d.all), summary(&mut c.all))
        } else {
            (summary(&mut d.interval), summary(&mut c.interval))
        };
        d.interval.clear();
        c.interval.clear();
        out
    };
    let line = serde_json::json!({
        "final": fin,
        "t": now.duration_since(t0).as_secs_f64(),
        "connected": s.connected.load(Ordering::Relaxed),
        "streams": s.connected.load(Ordering::Relaxed) * sh.cli.streams as u64,
        "tasks": tasks,
        "results": results,
        "gen_results": s.gen_results.load(Ordering::Relaxed),
        "label_results": s.label_results.load(Ordering::Relaxed),
        "refused": s.refused.load(Ordering::Relaxed),
        "model_frames": s.model_frames.load(Ordering::Relaxed),
        "reconnects": s.reconnects.load(Ordering::Relaxed),
        "results_per_s": if fin { results as f64 / now.duration_since(t0).as_secs_f64().max(1e-9) } else { (results - last.1) as f64 / dt },
        "tasks_per_s": (tasks - last.2) as f64 / dt,
        "mb_sent": sent as f64 / 1e6,
        "mb_recv": recv as f64 / 1e6,
        "mb_sent_per_s": (sent - last.3) as f64 / 1e6 / dt,
        "mb_recv_per_s": (recv - last.4) as f64 / 1e6 / dt,
        "dispatch_ms": dispatch,
        "connect_ms": connect,
    });
    *last = (now, results, tasks, sent, recv);
    line.to_string()
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let token_path = cli.token_file.clone().unwrap_or_else(botbowl_hub_proto::default_token_path);
    let token = match std::fs::read_to_string(&token_path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => {
            eprintln!("[fake] cannot read token {}: {e}", token_path.display());
            std::process::exit(2)
        }
    };
    let canned = match &cli.canned {
        Some(p) => load_canned(p),
        None => synthetic(cli.synthetic_kb),
    };
    let raw: usize = canned.iter().map(|g| g.raw_len).sum();
    let z: usize = canned.iter().map(|g| g.zstd.len()).sum();
    eprintln!(
        "[fake] {} canned game(s), mean {:.0} KB raw -> {:.1} KB zstd, {:.0} samples; {} connection(s) x {} streams to {}",
        canned.len(),
        raw as f64 / canned.len() as f64 / 1024.0,
        z as f64 / canned.len() as f64 / 1024.0,
        canned.iter().map(|g| g.samples as f64).sum::<f64>() / canned.len() as f64,
        cli.connections,
        cli.streams,
        cli.hub
    );
    let (connections, ramp, every, duration) = (
        cli.connections,
        Duration::from_millis(cli.ramp_ms),
        Duration::from_secs(cli.report_every.max(1)),
        cli.duration_secs,
    );
    let sh = Arc::new(Shared {
        cli,
        canned,
        stats: Stats::default(),
        next_game: AtomicU64::new(0),
    });
    let t0 = Instant::now();
    {
        let sh = Arc::clone(&sh);
        tokio::spawn(async move {
            for i in 0..connections {
                tokio::spawn(connection(i, Arc::clone(&sh), token.clone()));
                if !ramp.is_zero() {
                    tokio::time::sleep(ramp).await;
                }
            }
        });
    }
    let mut last = (t0, 0, 0, 0, 0);
    let mut tick = tokio::time::interval(every);
    tick.tick().await;
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
    let deadline = async {
        if duration > 0 {
            tokio::time::sleep(Duration::from_secs(duration)).await
        } else {
            std::future::pending::<()>().await
        }
    };
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = tick.tick() => println!("{}", report(&sh, t0, &mut last, false)),
            _ = &mut deadline => break,
            _ = tokio::signal::ctrl_c() => break,
            _ = term.recv() => break,
        }
    }
    println!("{}", report(&sh, t0, &mut last, true));
    std::process::exit(0);
}

//! Hub + workers in one process: a generate job must write, per shard,
//! exactly the seed set `botbowl-ui dataset --seed S --games N` would
//! have (`S..S+N`), once each, with the same provenance labels, and the
//! bytes must be the `DatasetWriter` format `prepare` reads.
//!
//! MCTS output is not reproducible across threads, so the game contents
//! are not compared — only metadata and the file format (the same rule the
//! phase-0 extraction was verified under).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use botbowl_hub::api::{GenerateJobRequest, JobKind, JobState, ShardReq};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{decode, encode, BuildInfo, Evaluator, SearchConfig, ToHub, ToWorker, PROTOCOL_VERSION};
use botbowl_play::generate::{budget_label, GenMode, GenerateConfig, RandomStartBias};
use botbowl_worker::{run_once, Ended, Fatal, ModelStore, WorkerConfig};

const TOKEN: &str = "test-token";
const GAMES: u32 = 3;
const SEED0: u64 = 10_000_000;
const STRIDE: u64 = 100_000;
/// Short drives: the test pins bookkeeping, not play quality.
const MAX_STEPS: u32 = 400;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "botbowl-hub-gen-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

async fn start_hub() -> (Hub, String) {
    // Long enough that a real worker in these tests is never reaped for
    // being busy; the reaper's own test sets its own.
    start_hub_with(Duration::from_secs(300)).await
}

async fn start_hub_with(worker_timeout: Duration) -> (Hub, String) {
    let (hub, addr, _task) = Hub::start(HubConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        token: TOKEN.into(),
        allow_commit_mismatch: false,
        worker_timeout,
    })
    .await
    .unwrap();
    (hub, format!("ws://{addr}/ws"))
}

fn worker_cfg(url: &str, name: &str, parallel: u16) -> WorkerConfig {
    WorkerConfig {
        hub_url: url.to_string(),
        token: TOKEN.into(),
        name: name.into(),
        parallel_games: Some(parallel),
        nn_server: None,
        cache_dir: tmp(&format!("cache-{name}")),
        mem_floor_mb: 0,
    }
}

fn spawn_worker(cfg: WorkerConfig) -> tokio::task::JoinHandle<Result<Ended, Fatal>> {
    tokio::spawn(async move {
        let store = Arc::new(ModelStore::open(&cfg.cache_dir, None).unwrap());
        let (tx, mut rx) = mpsc::unbounded_channel();
        run_once(&cfg, store, &mut rx, &tx).await
    })
}

fn cfg() -> GenerateConfig {
    GenerateConfig {
        config_name: None,
        mode: GenMode::RandomStart,
        search: SearchConfig::iterations(2),
        evaluator: Evaluator::Heuristic,
        model: None,
        max_steps: MAX_STEPS,
        lecture: None,
        difficulty: botbowl_curriculum::lecture::Difficulty::Easy,
        bias: RandomStartBias::default(),
        board_sizes: None,
    }
}

/// Two shards, laid out as `train_loop.sh` lays them out.
fn job(dir: &PathBuf, truncate: bool) -> GenerateJobRequest {
    GenerateJobRequest {
        shards: (0..2u32)
            .map(|k| ShardReq {
                name: format!("shard{k}"),
                out: dir.join(format!("shard{k}.jsonl")),
                seed: SEED0 + k as u64 * STRIDE,
                games: GAMES,
                cfg: cfg(),
                model_path: None,
            })
            .collect(),
        truncate,
        batch: 2,
    }
}

fn seeds_in(path: &PathBuf) -> Vec<u64> {
    botbowl_data::read_trajectories(path)
        .unwrap()
        .iter()
        .map(|t| t.meta.seed.expect("seed stamped"))
        .collect()
}

fn assert_shard(dir: &PathBuf, k: u32) {
    let path = dir.join(format!("shard{k}.jsonl"));
    let trajs = botbowl_data::read_trajectories(&path).unwrap();
    let base = SEED0 + k as u64 * STRIDE;
    let want: BTreeSet<u64> = (0..GAMES as u64).map(|g| base + g).collect();
    let got: Vec<u64> = trajs.iter().map(|t| t.meta.seed.unwrap()).collect();
    assert_eq!(got.len(), GAMES as usize, "{}: one line per game", path.display());
    assert_eq!(
        got.iter().copied().collect::<BTreeSet<_>>(),
        want,
        "{}: seed set",
        path.display()
    );
    let label = budget_label(&cfg());
    for t in &trajs {
        assert_eq!(t.meta.home_bot, label);
        assert_eq!(t.meta.away_bot, label);
        assert_eq!(t.meta.source, "random-start");
        assert_eq!(t.meta.extra.get("mode").map(String::as_str), Some("random-start"));
        assert_eq!(t.meta.format_version, botbowl_data::FORMAT_VERSION);
        assert!(
            !t.samples.is_empty(),
            "seed {} produced no samples",
            t.meta.seed.unwrap()
        );
    }
    // File shape: one JSON object per line, newline-terminated, no blanks
    // (`DatasetReader` skips blanks, `prepare` does not). The bytes are the
    // worker's `serde_json::to_vec` written verbatim, i.e. `DatasetWriter`'s.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.ends_with('\n'));
    assert_eq!(text.lines().count(), GAMES as usize);
    for line in text.lines() {
        assert!(line.starts_with('{') && line.ends_with('}'), "not a JSON object line");
        let _: botbowl_data::Trajectory = serde_json::from_str(line).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_workers_write_each_shard_exactly_once() {
    let (hub, url) = start_hub().await;
    let _w1 = spawn_worker(worker_cfg(&url, "w1", 2));
    let _w2 = spawn_worker(worker_cfg(&url, "w2", 1));

    let dir = tmp("job");
    // Stale content from an earlier attempt must go with `truncate`.
    std::fs::write(dir.join("shard0.jsonl"), "not json\n").unwrap();
    let id = hub.submit_generate(job(&dir, true)).unwrap();
    let status = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .expect("job finished in time")
        .expect("job exists");
    assert_eq!(status.state, JobState::Done, "{status:?}");
    assert_eq!(status.kind, JobKind::Generate);
    assert_eq!(status.units.len(), 2);
    for u in &status.units {
        assert_eq!(u.done, GAMES);
        assert!(u.samples > 0, "{u:?}");
    }

    assert_shard(&dir, 0);
    assert_shard(&dir, 1);

    // The samples the hub counted are the samples on disk.
    let on_disk: usize = (0..2)
        .map(|k| {
            botbowl_data::read_trajectories(dir.join(format!("shard{k}.jsonl")))
                .unwrap()
                .iter()
                .map(|t| t.samples.len())
                .sum::<usize>()
        })
        .sum();
    assert_eq!(status.units.iter().map(|u| u.samples).sum::<u64>(), on_disk as u64);

    // Both workers got work.
    let st = hub.inner.lock().unwrap().status();
    let done: BTreeSet<(String, u64)> = st.workers.iter().map(|w| (w.name.clone(), w.games_done)).collect();
    assert_eq!(done.iter().map(|(_, n)| n).sum::<u64>(), 2 * GAMES as u64, "{done:?}");
    assert!(done.iter().all(|(_, n)| *n > 0), "a worker sat idle: {done:?}");
}

/// Without `truncate` the job appends, as `botbowl-ui dataset` does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_mode_keeps_existing_lines() {
    let (hub, url) = start_hub().await;
    let _w = spawn_worker(worker_cfg(&url, "w", 2));
    let dir = tmp("append");
    let first = job(&dir, true);
    let id = hub.submit_generate(first).unwrap();
    let s = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(s.state, JobState::Done, "{s:?}");
    let before = seeds_in(&dir.join("shard0.jsonl"));

    let mut again = job(&dir, false);
    for sh in &mut again.shards {
        sh.seed += 50; // disjoint seeds so the union is checkable
    }
    let id = hub.submit_generate(again).unwrap();
    let s = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(s.state, JobState::Done, "{s:?}");
    let after = seeds_in(&dir.join("shard0.jsonl"));
    assert_eq!(after.len(), 2 * GAMES as usize);
    assert_eq!(
        &after[..GAMES as usize],
        &before[..],
        "earlier lines preserved in order"
    );
    let late: BTreeSet<u64> = after[GAMES as usize..].iter().copied().collect();
    assert_eq!(late, (0..GAMES as u64).map(|g| SEED0 + 50 + g).collect());
}

/// A worker that takes tasks and vanishes must not lose seeds or double
/// them: they are requeued and each appears exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vanishing_worker_gets_its_seeds_requeued() {
    let (hub, url) = start_hub().await;
    let dir = tmp("requeue");
    let id = hub.submit_generate(job(&dir, true)).unwrap();

    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    let hello = ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: TOKEN.into(),
        build: BuildInfo::current(),
        triple: "test".into(),
        name: "ghost".into(),
        cores: 1,
        ram_mb: 0,
        cached_models: vec![],
        parallel_games: Some(2),
    };
    sink.send(Message::Binary(encode(&hello).into())).await.unwrap();
    let mut tasks = 0;
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(2), stream.next()).await {
        match decode::<ToWorker>(&b).unwrap() {
            ToWorker::Welcome { .. } => {}
            ToWorker::Task(_) => tasks += 1,
            other => panic!("{other:?}"),
        }
        if tasks == 2 {
            break;
        }
    }
    assert_eq!(tasks, 2, "ghost should have been handed 2 tasks");
    drop(sink);
    drop(stream);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let _w = spawn_worker(worker_cfg(&url, "real", 2));
    let status = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .expect("job finished in time")
        .unwrap();
    assert_eq!(status.state, JobState::Done, "{status:?}");
    assert_shard(&dir, 0);
    assert_shard(&dir, 1);
}

/// A worker whose machine goes away without closing the socket — a laptop
/// that sleeps — must not strand the games it holds.
///
/// This is the case `a_vanishing_worker_gets_its_seeds_requeued` does *not*
/// cover: there the socket closes and the websocket task requeues at once.
/// Here the connection stays ESTABLISHED and silent, which is what actually
/// happened on 2026-09-18 (gen10 generate sat at 4791/4800 for three hours
/// with every live worker idle). Only the heartbeat timeout can tell the two
/// apart.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silent_worker_is_reaped_and_its_seeds_requeued() {
    let (hub, url) = start_hub_with(Duration::from_secs(2)).await;
    let dir = tmp("reap");
    let id = hub.submit_generate(job(&dir, true)).unwrap();

    // Connect, take work, then hold the socket open and say nothing.
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    let hello = ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: TOKEN.into(),
        build: BuildInfo::current(),
        triple: "test".into(),
        name: "sleeper".into(),
        cores: 1,
        ram_mb: 0,
        cached_models: vec![],
        parallel_games: Some(2),
    };
    sink.send(Message::Binary(encode(&hello).into())).await.unwrap();
    let mut tasks = 0;
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(2), stream.next()).await {
        match decode::<ToWorker>(&b).unwrap() {
            ToWorker::Welcome { .. } => {}
            ToWorker::Task(_) => tasks += 1,
            other => panic!("{other:?}"),
        }
        if tasks == 2 {
            break;
        }
    }
    assert_eq!(tasks, 2, "sleeper should have been handed 2 tasks");

    let live = spawn_worker(worker_cfg(&url, "real", 2));
    let status = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .expect("job finished in time — the sleeper's games must be requeued")
        .unwrap();
    assert_eq!(status.state, JobState::Done, "{status:?}");
    assert_shard(&dir, 0);
    assert_shard(&dir, 1);
    // The socket was never closed by either side: the reaper, not a
    // disconnect, is what freed the work.
    drop(sink);
    drop(stream);
    drop(live);
}

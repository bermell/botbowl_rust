//! Plan 062: MC value labels through the hub (`botbowl-hub job label`, workers in-process) are
//! byte-identical to `botbowl-ui mc-label` on the same backend (tract, the fixture net), on a corpus
//! that holds every case the labeller distinguishes:
//!
//! - a `--next-drive` pair whose follow-on comes *before* its first drive in the shard (the
//!   follow-on replays through the first drive, which travels with each of its items);
//! - a follow-on record whose first drive is in another shard (left unlabelled, as locally);
//! - a record whose seed no longer reproduces its states (replay diverges: left unlabelled);
//! - trajectories longer than `--chunk-samples`, so samples of one trajectory are labelled in
//!   different tasks, on different workers, and stitched back together.
//!
//! Policy playouts are deterministic given the net, the state and the dice, so unlike search
//! output the comparison is of whole files.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use botbowl_data::Trajectory;
use botbowl_engine::core::model::BoardDims;
use botbowl_hub::api::{JobKind, JobState, JobStatus, LabelJobRequest, LabelShardReq, UnitStats};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{decode, encode, BuildInfo, LabelConfig, ToHub, ToWorker, PROTOCOL_VERSION};
use botbowl_play::bots::{Evaluator, SearchConfig};
use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};
use botbowl_play::mc_label::drive_of;
use botbowl_ui::cli::McLabelArgs;
use botbowl_worker::{run_once, Ended, Fatal, ModelStore, WorkerConfig};

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");
const TOKEN: &str = "label-test-token";
const CFG: LabelConfig = LabelConfig {
    playouts: 2,
    seed: 62,
    // Short playouts: the test pins the plumbing and the bytes, not the labels' quality.
    max_steps: 300,
};
const SHARDS: [&str; 2] = ["shard0", "shard1"];

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "mc-label-hub-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn gen_config(next_drive: bool, max_steps: u32) -> GenerateConfig {
    let board = BoardDims::try_new(16, 9, 4).unwrap_or_else(|_| BoardDims::from_env());
    GenerateConfig {
        mode: GenMode::RandomStart,
        search: SearchConfig::iterations(4),
        evaluator: Evaluator::Heuristic,
        model: None,
        max_steps,
        lecture: None,
        difficulty: botbowl_curriculum::Difficulty::Easy,
        bias: RandomStartBias::default(),
        board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
        config_name: None,
        exploration: None,
        next_drive,
    }
}

/// The corpus, and what `botbowl-ui mc-label` makes of it: built once, shared by every test.
struct Fixture {
    corpus: PathBuf,
    local: PathBuf,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let corpus = tmp("corpus");
        let pair = (6000..6200u64)
            .map(|seed| play_trajectory(&gen_config(true, 2000), None, seed).unwrap())
            .find(|v| v.len() == 2)
            .expect("no seed in 200 scored and produced a next-drive record");
        assert_eq!((drive_of(&pair[0].meta), drive_of(&pair[1].meta)), (1, 2));
        let plain = |seed: u64| play_trajectory(&gen_config(false, 40), None, seed).unwrap().remove(0);
        let mut divergent = plain(5154);
        divergent.meta.seed = divergent.meta.seed.map(|s| s + 7_000_000);
        let (a, b, c) = (plain(5151), plain(5152), plain(5153));
        let shards: [Vec<&Trajectory>; 2] = [
            // The follow-on first: the first-drive lookup must not depend on line order.
            vec![&pair[1], &a, &pair[0], &b],
            // An orphan follow-on (its first drive is in shard0) and a divergent record.
            vec![&c, &pair[1], &divergent],
        ];
        for (name, trajs) in SHARDS.iter().zip(&shards) {
            let text: String = trajs.iter().map(|t| serde_json::to_string(t).unwrap() + "\n").collect();
            std::fs::write(corpus.join(format!("{name}.jsonl")), text).unwrap();
        }
        assert!(
            shards[0].iter().any(|t| t.samples.len() > 3),
            "the corpus needs a trajectory longer than a chunk"
        );
        let local = corpus.join("local");
        botbowl_ui::mc_label::run(McLabelArgs {
            corpus: inputs(&corpus),
            model: TINY.into(),
            nn_server: None,
            playouts: CFG.playouts,
            seed: CFG.seed,
            parallel: 2,
            max_steps: CFG.max_steps,
            out_dir: local.to_str().unwrap().into(),
        })
        .unwrap();
        Fixture { corpus, local }
    })
}

fn inputs(corpus: &Path) -> Vec<String> {
    SHARDS
        .iter()
        .map(|s| corpus.join(format!("{s}.jsonl")).to_str().unwrap().to_string())
        .collect()
}

fn request(f: &Fixture, out_dir: &Path, chunk_samples: u32, batch: u16) -> LabelJobRequest {
    LabelJobRequest {
        shards: SHARDS
            .iter()
            .map(|s| LabelShardReq {
                name: s.to_string(),
                input: f.corpus.join(format!("{s}.jsonl")),
                out: out_dir.join(format!("{s}.jsonl")),
            })
            .collect(),
        model_path: PathBuf::from(TINY),
        // The same string the local run was given, so the provenance tag matches.
        model: TINY.into(),
        cfg: CFG,
        chunk_samples,
        batch,
        label: Some("test mc-label".into()),
    }
}

async fn start_hub() -> (Hub, String) {
    let (hub, addr, _task) = Hub::start(HubConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        token: TOKEN.into(),
        allow_commit_mismatch: false,
        allowed_commits: std::env::temp_dir().join("botbowl-hub-test-no-such-allowlist.toml"),
        worker_timeout: Duration::from_secs(300),
        run_dir: None,
        allow_from: Vec::new(),
        rate_interval: Duration::from_secs(300),
        registry_dir: None,
    })
    .await
    .unwrap();
    (hub, format!("ws://{addr}/ws"))
}

fn spawn_worker(url: &str, name: &str, parallel: u16) -> tokio::task::JoinHandle<Result<Ended, Fatal>> {
    let cfg = WorkerConfig {
        hub_url: url.to_string(),
        token: TOKEN.into(),
        name: name.into(),
        parallel_games: Some(parallel),
        nn_server: None,
        cache_dir: tmp(&format!("cache-{name}")),
        mem_floor_mb: 0,
        reconnect_max: Duration::from_secs(1),
    };
    tokio::spawn(async move {
        let store = Arc::new(ModelStore::open(&cfg.cache_dir, None).unwrap());
        let (tx, mut rx) = mpsc::unbounded_channel();
        run_once(&cfg, store, &mut rx, &tx).await
    })
}

async fn wait(hub: &Hub, id: u64) -> JobStatus {
    tokio::time::timeout(Duration::from_secs(900), hub.wait(id))
        .await
        .expect("label job finished in time")
        .expect("job exists")
}

/// Every output shard equals the local tool's, byte for byte, and the job's counts are the local
/// summary lines' counts.
fn assert_same_as_local(f: &Fixture, out_dir: &Path, status: &JobStatus) {
    assert_eq!(status.state, JobState::Done, "{status:?}");
    assert_eq!(status.kind, JobKind::Label);
    for (k, s) in SHARDS.iter().enumerate() {
        let file = format!("{s}.jsonl");
        let want = std::fs::read(f.local.join(&file)).unwrap();
        let got = std::fs::read(out_dir.join(&file)).unwrap();
        assert!(
            got == want,
            "{file}: the hub's output differs from mc-label's ({} vs {} bytes)",
            got.len(),
            want.len()
        );
        assert!(!out_dir.join(format!("{file}.partial")).exists());
        let Some(UnitStats::Label(l)) = &status.units[k].stats else {
            panic!("no label stats: {:?}", status.units[k])
        };
        let trajs = botbowl_data::read_trajectories(out_dir.join(&file)).unwrap();
        let labelled: Vec<&Trajectory> = trajs
            .iter()
            .filter(|t| t.meta.extra.contains_key("value_label"))
            .collect();
        assert_eq!(l.trajectories as usize, trajs.len(), "{file}");
        assert_eq!(l.unlabelled as usize, trajs.len() - labelled.len(), "{file}");
        assert_eq!(
            l.samples as usize,
            labelled.iter().map(|t| t.samples.len()).sum::<usize>(),
            "{file}"
        );
        assert!(l.written && !l.skipped);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_labels_equal_local_mc_label_byte_for_byte() {
    let f = fixture();
    // What the corpus is for: both drives of the pair labelled in shard0; the orphan and the
    // divergent record left unlabelled in shard1, everything else labelled.
    let local: Vec<Vec<Trajectory>> = SHARDS
        .iter()
        .map(|s| botbowl_data::read_trajectories(f.local.join(format!("{s}.jsonl"))).unwrap())
        .collect();
    let stamped = |t: &Trajectory| t.meta.extra.contains_key("value_label");
    assert!(local[0].iter().all(stamped), "shard0 is all labellable");
    assert_eq!(local[1].iter().map(stamped).collect::<Vec<_>>(), [true, false, false]);
    assert_eq!(
        drive_of(&local[0][0].meta),
        2,
        "the follow-on was labelled through its first drive"
    );

    let (hub, url) = start_hub().await;
    let _w1 = spawn_worker(&url, "w1", 1);
    let _w2 = spawn_worker(&url, "w2", 1);
    let out = tmp("hub-out");
    // Chunks of 3 samples, two items per task: one trajectory's samples cross tasks and workers.
    let id = hub.submit_label(request(f, &out, 3, 2)).unwrap();
    let status = wait(&hub, id).await;
    assert_same_as_local(f, &out, &status);
    let st = hub.inner.lock().unwrap().status();
    let job = st.jobs.iter().find(|j| j.id == id).unwrap();
    let items: u32 = status.units.iter().map(|u| u.total).sum();
    assert!(
        items as usize > local.iter().map(Vec::len).sum::<usize>(),
        "no trajectory was split into several items"
    );
    assert_eq!(job.by_worker.values().map(|c| c.label_items).sum::<u64>(), items as u64);
    assert!(
        job.by_worker.len() == 2 && job.by_worker.values().all(|c| c.label_items > 0),
        "a worker sat idle: {:?}",
        job.by_worker
    );

    // A rerun skips the shards already written, as `mc-label` does, and touches nothing.
    let before: Vec<Vec<u8>> = SHARDS
        .iter()
        .map(|s| std::fs::read(out.join(format!("{s}.jsonl"))).unwrap())
        .collect();
    let again = hub.submit_label(request(f, &out, 3, 2)).unwrap();
    let status = wait(&hub, again).await;
    assert_eq!(status.state, JobState::Done, "{status:?}");
    for (k, u) in status.units.iter().enumerate() {
        assert!(matches!(&u.stats, Some(UnitStats::Label(l)) if l.skipped), "{u:?}");
        assert_eq!(
            std::fs::read(out.join(format!("{}.jsonl", SHARDS[k]))).unwrap(),
            before[k]
        );
    }
}

/// A worker that takes label tasks and vanishes mid-chunk loses nothing: its items are requeued,
/// each is labelled once, and the shards are still the local tool's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vanishing_worker_gets_its_label_items_requeued() {
    let f = fixture();
    let (hub, url) = start_hub().await;
    let out = tmp("hub-requeue");
    let id = hub.submit_label(request(f, &out, 4, 2)).unwrap();

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
    let mut tasks = Vec::new();
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
        match decode::<ToWorker>(&b).unwrap() {
            ToWorker::Welcome { .. } | ToWorker::Model { .. } | ToWorker::ModelName { .. } => {}
            ToWorker::Task(t) => tasks.push(t),
            other => panic!("{other:?}"),
        }
        if tasks.len() == 2 {
            break;
        }
    }
    assert_eq!(tasks.len(), 2, "the ghost should have been handed 2 label tasks");
    assert!(tasks.iter().all(|t| matches!(t, botbowl_hub_proto::Task::Label { .. })));
    // It labels the first item of its first task (correctly: the output must still match), then
    // its connection drops mid-chunk, holding that task's second item and the whole second task.
    let botbowl_hub_proto::Task::Label {
        id: task, items, cfg, ..
    } = &tasks[0]
    else {
        unreachable!()
    };
    assert_eq!(items.len(), 2, "two items per task");
    let read = |z: &[u8]| -> Trajectory { serde_json::from_slice(&zstd::decode_all(z).unwrap()).unwrap() };
    let traj = read(&items[0].zstd_json);
    let first = items[0].first_zstd_json.as_deref().map(read);
    let nn = Arc::new(botbowl_nn::eval::NnEvaluator::from_path(TINY).unwrap());
    let range = items[0].start as usize..items[0].end as usize;
    let labels = botbowl_play::mc_label::label_range(&nn, &traj, first.as_ref(), range, cfg).map_or_else(
        botbowl_hub_proto::LabelResult::Unlabellable,
        botbowl_hub_proto::LabelResult::Labels,
    );
    let done = ToHub::LabelDone {
        task: *task,
        item: items[0].item,
        labels,
    };
    sink.send(Message::Binary(encode(&done).into())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    drop((sink, stream));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        hub.inner.lock().unwrap().status().workers.len(),
        0,
        "the ghost should be gone"
    );

    let _w = spawn_worker(&url, "real", 2);
    let status = wait(&hub, id).await;
    assert_same_as_local(f, &out, &status);
    let st = hub.inner.lock().unwrap().status();
    let job = st.jobs.iter().find(|j| j.id == id).unwrap();
    // The ghost's one item counted once; the real worker did every other item, its requeued ones
    // included, and nothing was labelled twice.
    let items: u64 = status.units.iter().map(|u| u.total as u64).sum();
    assert_eq!(job.by_worker["ghost"].label_items, 1, "{:?}", job.by_worker);
    assert_eq!(job.by_worker["real"].label_items, items - 1, "{:?}", job.by_worker);
}

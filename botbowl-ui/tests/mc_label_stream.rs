//! Plan 069 item 4b: MC labels streamed by a generate job (`job generate --mc-label-playouts N`)
//! are byte-identical to labelling the finished shards afterwards — `botbowl-ui mc-label` and
//! `botbowl-hub job label` — on the same backend (tract, the fixture net):
//!
//! - a fresh generation (`--truncate`) of `--next-drive` games over two shards, items of 3 samples
//!   on two workers with label streams, one shard left unlabelled;
//! - a generation that appends to shards already on disk (a resumed one): the lines there are
//!   labelled too, a stale label output is replaced, and a seed generated again moves its follow-on
//!   record's first drive to the later line — the record is labelled again at the seal, as the
//!   reading rule says.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::mpsc;

use botbowl_engine::core::model::BoardDims;
use botbowl_hub::api::{GenerateJobRequest, JobKind, JobState, JobStatus, LabelJobRequest, LabelShardReq, ShardReq};
use botbowl_hub::api::{StreamLabelRequest, UnitStats};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::LabelConfig;
use botbowl_play::bots::{Evaluator, SearchConfig};
use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};
use botbowl_ui::cli::McLabelArgs;
use botbowl_worker::{run_once, Ended, Fatal, ModelStore, WorkerConfig};

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");
const TOKEN: &str = "stream-label-test-token";
const CFG: LabelConfig = LabelConfig {
    playouts: 2,
    seed: 69,
    // Playouts to the drive's end, so labels are not all 0 and a misplaced range would show.
    max_steps: 100_000,
    skip: 0,
};
const GAMES: u32 = 4;
const STRIDE: u64 = 100;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "mc-label-stream-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn gen_config() -> GenerateConfig {
    let board = BoardDims::try_new(16, 9, 4).unwrap_or_else(|_| BoardDims::from_env());
    GenerateConfig {
        mode: GenMode::RandomStart,
        // One search thread: the games (and so the corpora) are reproducible.
        search: SearchConfig::iterations(4),
        evaluator: Evaluator::Heuristic,
        model: None,
        max_steps: 2000,
        lecture: None,
        difficulty: botbowl_curriculum::Difficulty::Easy,
        bias: RandomStartBias::default(),
        board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
        config_name: None,
        exploration: None,
        next_drive: true,
    }
}

/// A seed whose drive scores (so it writes a next-drive pair), shared by the tests.
fn pair_seed() -> u64 {
    static S: OnceLock<u64> = OnceLock::new();
    *S.get_or_init(|| {
        (6000..6200u64)
            .find(|&s| play_trajectory(&gen_config(), None, s).unwrap().len() == 2)
            .expect("no seed in 200 scored")
    })
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

fn spawn_worker(
    url: &str,
    name: &str,
    parallel: u16,
    label_streams: u16,
) -> tokio::task::JoinHandle<Result<Ended, Fatal>> {
    let cfg = WorkerConfig {
        hub_url: url.to_string(),
        token: TOKEN.into(),
        name: name.into(),
        parallel_games: Some(parallel),
        label_streams,
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
    tokio::time::timeout(Duration::from_secs(1800), hub.wait(id))
        .await
        .expect("job finished in time")
        .expect("job exists")
}

/// Shards `names` of a generate job into `dir`, shard `k` seeded at `base + k * STRIDE`; the
/// named ones streamed into `dir/mc`.
fn generate(dir: &Path, base: u64, shards: &[&str], labelled: &[&str], truncate: bool) -> GenerateJobRequest {
    GenerateJobRequest {
        shards: shards
            .iter()
            .enumerate()
            .map(|(k, name)| ShardReq {
                name: name.to_string(),
                out: dir.join(format!("{name}.jsonl")),
                seed: base + k as u64 * STRIDE,
                games: GAMES,
                cfg: gen_config(),
                model_path: None,
            })
            .collect(),
        truncate,
        batch: 1,
        label: Some("test generate".into()),
        mc_label: Some(StreamLabelRequest {
            shards: labelled
                .iter()
                .map(|n| (n.to_string(), dir.join("mc").join(format!("{n}.jsonl"))))
                .collect(),
            model_path: PathBuf::from(TINY),
            model: TINY.into(),
            cfg: CFG,
            chunk_samples: 3,
            batch: 1,
            label: None,
        }),
    }
}

/// `botbowl-ui mc-label` over `shards` of `dir`, into `out`.
fn local_labels(dir: &Path, shards: &[&str], out: &Path) {
    botbowl_ui::mc_label::run(McLabelArgs {
        corpus: shards
            .iter()
            .map(|s| dir.join(format!("{s}.jsonl")).to_str().unwrap().to_string())
            .collect(),
        model: TINY.into(),
        nn_server: None,
        backend: Default::default(),
        playouts: CFG.playouts,
        seed: CFG.seed,
        parallel: 2,
        max_steps: CFG.max_steps,
        top_up: None,
        out_dir: out.to_str().unwrap().into(),
    })
    .unwrap();
}

fn label_job(dir: &Path, shards: &[&str], out: &Path) -> LabelJobRequest {
    LabelJobRequest {
        shards: shards
            .iter()
            .map(|s| LabelShardReq {
                name: s.to_string(),
                input: dir.join(format!("{s}.jsonl")),
                out: out.join(format!("{s}.jsonl")),
            })
            .collect(),
        model_path: PathBuf::from(TINY),
        model: TINY.into(),
        cfg: CFG,
        chunk_samples: 5,
        batch: 2,
        label: None,
        top_up: false,
    }
}

fn assert_same_files(a: &Path, b: &Path, shards: &[&str], what: &str) {
    for s in shards {
        let file = format!("{s}.jsonl");
        let x = std::fs::read(a.join(&file)).unwrap();
        let y = std::fs::read(b.join(&file)).unwrap();
        assert!(
            x == y,
            "{file}: streamed labels differ from {what} ({} vs {} bytes)",
            x.len(),
            y.len()
        );
    }
}

/// Generate with streamed labels, wait for both jobs; the label job's status.
async fn run_streamed(hub: &Hub, req: GenerateJobRequest) -> JobStatus {
    let id = hub.submit_generate(req).unwrap();
    let g = wait(hub, id).await;
    assert_eq!(g.state, JobState::Done, "{g:?}");
    let lid = g.labels_job.expect("a generate job with mc_label has a labels job");
    let l = wait(hub, lid).await;
    assert_eq!(l.state, JobState::Done, "{l:?}");
    assert_eq!(l.kind, JobKind::Label);
    l
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streamed_labels_equal_labelling_the_finished_shards() {
    let base = pair_seed() - 1;
    let (hub, url) = start_hub().await;
    // Game streams and label streams on both: labels run beside the games, not after them.
    let _w1 = spawn_worker(&url, "w1", 1, 1);
    let _w2 = spawn_worker(&url, "w2", 1, 1);
    let dir = tmp("fresh");
    let shards = ["shard0", "shard1"];
    let l = run_streamed(&hub, generate(&dir, base, &shards, &["shard0", "shard1"], true)).await;
    // The corpus holds what it is for: a next-drive pair, and trajectories longer than a chunk.
    let raw = botbowl_data::read_trajectories(dir.join("shard0.jsonl")).unwrap();
    assert!(raw.len() > GAMES as usize, "no next-drive record in shard0");
    assert!(raw.iter().any(|t| t.samples.len() > 3));
    for u in &l.units {
        let Some(UnitStats::Label(s)) = &u.stats else {
            panic!("{u:?}")
        };
        assert!(s.written && !s.skipped, "{s:?}");
    }

    let mc = botbowl_data::read_trajectories(dir.join("mc/shard0.jsonl")).unwrap();
    let values: Vec<f32> = mc
        .iter()
        .flat_map(|t| t.samples.iter().filter_map(|s| s.outcome_value))
        .collect();
    assert!(
        values.iter().any(|v| *v != 0.0),
        "every label is 0: the comparison would check little"
    );
    let local = tmp("fresh-local");
    local_labels(&dir, &shards, &local);
    assert_same_files(&dir.join("mc"), &local, &shards, "mc-label");
    let job = tmp("fresh-job");
    let id = hub.submit_label(label_job(&dir, &shards, &job)).unwrap();
    assert_eq!(wait(&hub, id).await.state, JobState::Done);
    assert_same_files(&dir.join("mc"), &job, &shards, "job label");
    assert_eq!(hub.inner.lock().unwrap().label_bytes_held(), 0);

    // A shard not named is not labelled.
    let dir2 = tmp("partial");
    run_streamed(&hub, generate(&dir2, base + 50, &shards, &["shard1"], true)).await;
    assert!(dir2.join("mc/shard1.jsonl").exists());
    assert!(!dir2.join("mc/shard0.jsonl").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_resumed_generation_labels_the_lines_already_on_disk_and_relabels_a_moved_first_drive() {
    let pair = pair_seed();
    let (hub, url) = start_hub().await;
    let _w1 = spawn_worker(&url, "w1", 2, 1);
    let dir = tmp("resume");
    // On disk (written by earlier attempts): the pair's follow-on before any first drive (deferred
    // to the seal), its first drive twice, the follow-on again (cut against the later of the two),
    // and a stale label output.
    let recs = play_trajectory(&gen_config(), None, pair).unwrap();
    let line = |t| serde_json::to_string(t).unwrap() + "\n";
    let text = [&recs[1], &recs[0], &recs[0], &recs[1]].map(line).concat();
    std::fs::write(dir.join("shard0.jsonl"), text).unwrap();
    std::fs::create_dir_all(dir.join("mc")).unwrap();
    std::fs::write(dir.join("mc/shard0.jsonl"), "stale\n").unwrap();
    // The job plays `pair` again (seeds pair-1 .. pair+2): its drive 1 comes later in the file, so
    // both follow-ons' first drive moves to it at the seal and they are labelled again.
    let l = run_streamed(&hub, generate(&dir, pair - 1, &["shard0"], &["shard0"], false)).await;
    let raw = botbowl_data::read_trajectories(dir.join("shard0.jsonl")).unwrap();
    assert!(raw.iter().filter(|t| t.meta.seed == Some(pair)).count() >= 5);
    let Some(UnitStats::Label(s)) = &l.units[0].stats else {
        panic!()
    };
    assert_eq!(s.trajectories as usize, raw.len());

    let local = tmp("resume-local");
    local_labels(&dir, &["shard0"], &local);
    assert_same_files(&dir.join("mc"), &local, &["shard0"], "mc-label");

    // Appending nothing to a shard keeps its labels: skipped, as `job label` skips it.
    let before = std::fs::read(dir.join("mc/shard0.jsonl")).unwrap();
    let mut again = generate(&dir, pair - 1, &["shard0"], &["shard0"], false);
    again.shards[0].games = 0;
    let l = run_streamed(&hub, again).await;
    assert!(matches!(&l.units[0].stats, Some(UnitStats::Label(s)) if s.skipped));
    assert_eq!(std::fs::read(dir.join("mc/shard0.jsonl")).unwrap(), before);
}

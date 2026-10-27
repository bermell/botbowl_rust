//! Plan 069 item 5 (plan 041 phase 3): a job resumes from its files after the hub that ran it
//! died. Generate: a shard cut anywhere — a torn line, a game missing its follow-on record, a
//! missing file — ends with exactly its seed set, each seed's records once, nothing torn. Eval:
//! the resumed report equals the uninterrupted one. Plus the rules that make resubmission safe:
//! the same request attaches to its running job, another request on the same files is refused, a
//! drained job hands out nothing, and a result for another hub's task is never written.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use botbowl_hub::api::{BotReq, EvalJobRequest, GenerateJobRequest, JobState, RungReq, ShardReq};
use botbowl_hub::state::{Inner, TrajectoryPayload, WorkerConn};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{BuildInfo, Evaluator, SearchConfig, ToWorker};
use botbowl_play::generate::{GenMode, GenerateConfig, RandomStartBias};
use botbowl_worker::{run_once, ModelStore, WorkerConfig};

const TOKEN: &str = "test-token";
const SEED0: u64 = 20_000_000;
const STRIDE: u64 = 100_000;
const GAMES: u32 = 6;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "botbowl-hub-resume-{tag}-{}-{}",
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

fn spawn_worker(url: &str, name: &str, parallel: u16) {
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
    });
}

fn cfg(next_drive: bool) -> GenerateConfig {
    GenerateConfig {
        config_name: None,
        exploration: None,
        mode: GenMode::RandomStart,
        search: SearchConfig::iterations(2),
        evaluator: Evaluator::Heuristic,
        model: None,
        max_steps: 400,
        lecture: None,
        difficulty: botbowl_curriculum::lecture::Difficulty::Easy,
        bias: RandomStartBias::default(),
        board_sizes: None,
        next_drive,
    }
}

fn gen_job(dir: &Path, next_drive: bool, truncate: bool, resume: bool) -> GenerateJobRequest {
    GenerateJobRequest {
        shards: (0..2u32)
            .map(|k| ShardReq {
                name: format!("shard{k}"),
                out: dir.join(format!("shard{k}.jsonl")),
                seed: SEED0 + k as u64 * STRIDE,
                games: GAMES,
                cfg: cfg(next_drive),
                model_path: None,
            })
            .collect(),
        truncate,
        batch: 2,
        label: None,
        resume,
    }
}

async fn wait_done(hub: &Hub, id: u64) {
    let s = tokio::time::timeout(Duration::from_secs(300), hub.wait(id))
        .await
        .expect("job finished in time")
        .expect("job exists");
    assert_eq!(s.state, JobState::Done, "{s:?}");
}

/// Per seed, the records on disk as `(drive, line)`, checking every line parses whole.
fn records(path: &Path) -> BTreeMap<u64, Vec<String>> {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(
        text.is_empty() || text.ends_with('\n'),
        "{} ends mid-line",
        path.display()
    );
    let mut out: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for line in text.lines() {
        let t: botbowl_data::Trajectory = serde_json::from_str(line).expect("a whole trajectory line");
        out.entry(t.meta.seed.unwrap())
            .or_default()
            .push(t.meta.extra.get("drive").cloned().unwrap_or_default());
    }
    out
}

fn assert_complete(dir: &Path, reference: &[BTreeMap<u64, Vec<String>>]) {
    for (k, want) in reference.iter().enumerate() {
        let got = records(&dir.join(format!("shard{k}.jsonl")));
        let base = SEED0 + k as u64 * STRIDE;
        assert_eq!(
            got.keys().copied().collect::<BTreeSet<_>>(),
            (base..base + GAMES as u64).collect::<BTreeSet<_>>(),
            "shard{k}: the seed set"
        );
        for (seed, drives) in &got {
            assert!(
                drives == &["1"] || drives == &["1", "2"],
                "shard{k} seed {seed}: records {drives:?} (each game once, drive 2 after drive 1)"
            );
        }
        // Heuristic search is deterministic per seed, so the resumed corpus is the uninterrupted
        // one, record for record (the order of lines aside).
        assert_eq!(&got, want, "shard{k}: same records per seed as the uninterrupted run");
    }
}

/// Cut a shard file the ways a dying hub can leave it.
fn damage(path: &Path, keep_lines: usize, torn: usize) {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let mut out: String = lines[..keep_lines.min(lines.len())]
        .iter()
        .map(|l| format!("{l}\n"))
        .collect();
    if let Some(next) = lines.get(keep_lines) {
        out.push_str(&next[..torn.min(next.len() - 1)]);
    }
    std::fs::write(path, out).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cut_generate_job_resumes_to_the_uninterrupted_corpus() {
    let (hub, url) = start_hub().await;
    spawn_worker(&url, "w1", 2);
    spawn_worker(&url, "w2", 1);

    // The uninterrupted run, with --next-drive so some games have two records.
    let full = tmp("full");
    let id = hub.submit_generate(gen_job(&full, true, true, false)).unwrap();
    wait_done(&hub, id).await;
    let reference: Vec<_> = (0..2).map(|k| records(&full.join(format!("shard{k}.jsonl")))).collect();

    // Shard 0 torn mid-line after three records; the last whole line, if it is a scoring first
    // drive, may lose its follow-on too. Shard 1 lost entirely.
    let cut = tmp("cut");
    std::fs::copy(full.join("shard0.jsonl"), cut.join("shard0.jsonl")).unwrap();
    damage(&cut.join("shard0.jsonl"), 3, 500);
    let id = hub.submit_generate(gen_job(&cut, true, false, true)).unwrap();
    wait_done(&hub, id).await;
    assert_complete(&cut, &reference);

    // Resuming a finished job changes nothing. (It may replay a shard's last game, when that is
    // a lone scoring drive — under --next-drive the line cannot show it had no follow-on.)
    let id = hub.submit_generate(gen_job(&cut, true, false, true)).unwrap();
    wait_done(&hub, id).await;
    assert_complete(&cut, &reference);
    // Without --next-drive every whole line is a whole game: nothing is played.
    let plain = tmp("plain");
    let id = hub.submit_generate(gen_job(&plain, false, true, false)).unwrap();
    wait_done(&hub, id).await;
    let before = std::fs::read(plain.join("shard0.jsonl")).unwrap();
    let id = hub.submit_generate(gen_job(&plain, false, false, true)).unwrap();
    let s = hub.job_status(id).unwrap();
    assert_eq!(s.state, JobState::Done, "nothing left to play: {s:?}");
    assert_eq!(std::fs::read(plain.join("shard0.jsonl")).unwrap(), before);

    // A file cut right after some game's first record: when that game had a follow-on (or could
    // have: a scoring drive under --next-drive), it is played again whole. Every game is tried, so
    // whatever the drives did, both the cut-follow-on and the kept-last-game paths run.
    for (k, seed) in reference
        .iter()
        .enumerate()
        .flat_map(|(k, m)| m.keys().map(move |s| (k, *s)))
    {
        let nd = tmp("nd");
        let text = std::fs::read_to_string(full.join(format!("shard{k}.jsonl"))).unwrap();
        let mut kept = String::new();
        for line in text.lines() {
            let t: botbowl_data::Trajectory = serde_json::from_str(line).unwrap();
            kept.push_str(line);
            kept.push('\n');
            if t.meta.seed == Some(seed) {
                break; // its first record only
            }
        }
        std::fs::write(nd.join(format!("shard{k}.jsonl")), kept).unwrap();
        let id = hub.submit_generate(gen_job(&nd, true, false, true)).unwrap();
        wait_done(&hub, id).await;
        assert_complete(&nd, &reference);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shard_from_another_generator_is_refused_not_extended() {
    let (hub, _url) = start_hub().await;
    let dir = tmp("other");
    let line = r#"{"meta":{"seed":20000000,"home_bot":"mcts(iters=9,workers=1,eval=nn:other.onnx)"},"samples":[],"outcome":{"home_score":0,"away_score":0}}"#;
    std::fs::write(dir.join("shard0.jsonl"), format!("{line}\n")).unwrap();
    let e = hub.submit_generate(gen_job(&dir, false, false, true)).unwrap_err();
    assert!(e.to_string().contains("generated by"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resubmitting_attaches_and_other_requests_on_the_same_files_are_refused() {
    let (hub, url) = start_hub().await;
    let dir = tmp("attach");
    let id = hub.submit_generate(gen_job(&dir, false, false, true)).unwrap();
    assert_eq!(hub.submit_generate(gen_job(&dir, false, false, true)).unwrap(), id);
    let e = hub.submit_generate(gen_job(&dir, true, false, true)).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists, "{e}");

    // Drained: a worker that joins gets nothing of it. Submitted again: it runs to the end.
    assert!(hub.drain(id));
    spawn_worker(&url, "w1", 2);
    let t0 = Instant::now();
    while hub.inner.lock().unwrap().status().workers.is_empty() && t0.elapsed() < Duration::from_secs(30) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let s = hub.job_status(id).unwrap();
    assert!(s.draining && s.in_flight == 0, "{s:?}");
    assert_eq!(s.units.iter().map(|u| u.done).sum::<u32>(), 0);
    assert_eq!(hub.submit_generate(gen_job(&dir, false, false, true)).unwrap(), id);
    assert!(!hub.job_status(id).unwrap().draining);
    wait_done(&hub, id).await;
}

fn eval_job(dir: &Path, resume: bool) -> EvalJobRequest {
    EvalJobRequest {
        candidate: BotReq::Scripted,
        candidate_label: "scripted".into(),
        candidate_config: None,
        opponent_config: None,
        mcts_iters: 0,
        rungs: ["random", "scripted"]
            .iter()
            .map(|r| RungReq {
                name: r.to_string(),
                games: GAMES,
                opponent: if *r == "random" {
                    BotReq::Random
                } else {
                    BotReq::Scripted
                },
                board: None,
                drives: None,
            })
            .collect(),
        seed: 5,
        max_steps: 100_000,
        per_game_out: dir.join("eval.games.jsonl"),
        report_out: dir.join("report.json"),
        batch: 2,
        sprt: None,
        drive_gate: None,
        label: None,
        resume,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cut_eval_job_resumes_to_the_same_report() {
    let (hub, url) = start_hub().await;
    spawn_worker(&url, "w1", 2);
    let full = tmp("eval-full");
    let id = hub.submit_eval(eval_job(&full, false)).unwrap();
    wait_done(&hub, id).await;
    let want = hub.job_status(id).unwrap().report.unwrap().ladder;

    let cut = tmp("eval-cut");
    std::fs::copy(full.join("eval.games.jsonl"), cut.join("eval.games.jsonl")).unwrap();
    damage(&cut.join("eval.games.jsonl"), 5, 30);
    let id = hub.submit_eval(eval_job(&cut, true)).unwrap();
    wait_done(&hub, id).await;
    assert_eq!(hub.job_status(id).unwrap().report.unwrap().ladder, want);
    let text = std::fs::read_to_string(cut.join("eval.games.jsonl")).unwrap();
    let keys: Vec<(String, u32)> = text
        .lines()
        .map(|l| {
            let l: botbowl_hub_proto::EvalGameLine = serde_json::from_str(l).unwrap();
            (l.rung, l.game)
        })
        .collect();
    assert_eq!(keys.len(), 2 * GAMES as usize, "each game once: {keys:?}");
    assert_eq!(keys.iter().collect::<BTreeSet<_>>().len(), keys.len());
}

/// A worker that reconnects after a hub restart still sends results for the old hub's tasks. With
/// task ids counted from 0 in both hubs, old task 0 (shard0 game 0) would be new task 0 (here,
/// shard 1 game 0): the hub must not write a trajectory of the wrong seed into the wrong shard.
#[test]
fn a_result_for_another_hubs_task_is_never_written() {
    let dir = tmp("foreign");
    let mut inner = Inner::default(); // instance 0, task ids from 0: the worst case
    let (tx, mut rx) = mpsc::unbounded_channel();
    let wid = inner.add_worker(WorkerConn {
        name: "w".into(),
        build: BuildInfo::current(),
        triple: String::new(),
        cores: 1,
        ram_mb: 1024,
        parallel_games: 1,
        tx,
        known_models: Default::default(),
        named_models: Default::default(),
        tasks: Default::default(),
        games_done: 0,
        last_seen: Instant::now(),
    });
    let mut req = gen_job(&dir, false, true, false);
    req.shards.remove(0); // only shard 1: task 0 is shard1's games 0 and 1
    req.batch = 1;
    let id = inner.submit_generate(req).unwrap();
    let task = loop {
        match rx.try_recv().expect("a task was sent") {
            ToWorker::Task(t) => break t.id(),
            _ => continue,
        }
    };
    // The old hub's shard 0, game 0: seed SEED0, not shard 1's SEED0 + STRIDE.
    let line = format!(
        r#"{{"meta":{{"seed":{SEED0},"board_dims":{{"width":16,"height":9,"team_size":6}}}},"samples":[],"outcome":{{"home_score":0,"away_score":0}}}}"#
    );
    let payload = TrajectoryPayload::decode(&zstd::encode_all(line.as_bytes(), 3).unwrap());
    inner.trajectory_done(wid, task, 0, 0, payload);
    assert_eq!(
        std::fs::read(dir.join("shard1.jsonl")).unwrap().len(),
        0,
        "nothing written"
    );
    let s = inner.job_status(id).unwrap();
    assert_eq!(s.units[0].done, 0, "{s:?}");
    assert_eq!(s.state, JobState::Running);

    // And a hub that numbers its tasks from a random start does not hand out task 0 at all.
    let mut fresh = Inner::default();
    fresh.new_instance();
    assert_ne!(fresh.instance, 0);
}

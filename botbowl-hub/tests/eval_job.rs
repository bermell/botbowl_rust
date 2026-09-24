//! Hub + workers in one process: an eval job's per-game lines and report
//! must equal what `botbowl-play` produces for the same games directly.
//!
//! Search-free bots (scripted candidate, random/scripted rungs) so the
//! comparison is exact on any board tier: `play_ladder_game` is a pure
//! function of `(bots, seed)` when no MCTS is involved.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use botbowl_engine::bots::RandomBot;
use botbowl_engine::core::model::{BoardDims, HEIGHT, TEAM_SIZE, WIDTH};
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_hub::api::{BotReq, EvalJobRequest, JobState, RungReq};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{decode, encode, BuildInfo, EvalGameLine, RejectReason, ToHub, ToWorker, PROTOCOL_VERSION};
use botbowl_play::board_sizes::board_label;
use botbowl_play::eval::{ladder_assignment, play_ladder_game, rung_name, LadderRow};
use botbowl_worker::{run_once, Ended, Fatal, ModelStore, WorkerConfig};

const TOKEN: &str = "test-token";
const SEED: u64 = 5;
const GAMES: u32 = 6;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "botbowl-hub-test-{tag}-{}-{}",
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

/// A real worker, on its own reconnect-free connection.
fn spawn_worker(cfg: WorkerConfig) -> tokio::task::JoinHandle<Result<Ended, Fatal>> {
    tokio::spawn(async move {
        let store = Arc::new(ModelStore::open(&cfg.cache_dir, None).unwrap());
        let (tx, mut rx) = mpsc::unbounded_channel();
        run_once(&cfg, store, &mut rx, &tx).await
    })
}

fn job(dir: &PathBuf) -> EvalJobRequest {
    EvalJobRequest {
        candidate: BotReq::Scripted,
        candidate_label: "scripted".into(),
        candidate_config: None,
        opponent_config: None,
        mcts_iters: 0,
        rungs: vec![
            RungReq {
                name: "random".into(),
                games: GAMES,
                opponent: BotReq::Random,
                board: None,
            },
            RungReq {
                name: "scripted".into(),
                games: GAMES,
                opponent: BotReq::Scripted,
                board: None,
            },
        ],
        seed: SEED,
        max_steps: 100_000,
        per_game_out: dir.join("eval.games.jsonl"),
        report_out: dir.join("report.json"),
        batch: 2,
    }
}

/// What `botbowl-ui eval` would have produced, computed in-process.
fn expected() -> (Vec<EvalGameLine>, Vec<LadderRow>) {
    let mut lines = Vec::new();
    let mut rows = Vec::new();
    for rung in ["random", "scripted"] {
        let mut row = LadderRow::new(rung);
        let mut cand = ScriptedBot::new();
        for g in 0..GAMES {
            let (team, seed) = ladder_assignment(SEED, g);
            let line = match rung {
                "random" => play_ladder_game(
                    &mut cand,
                    &mut RandomBot::new(),
                    rung,
                    g,
                    team,
                    seed,
                    100_000,
                    None,
                    None,
                ),
                _ => play_ladder_game(
                    &mut cand,
                    &mut ScriptedBot::new(),
                    rung,
                    g,
                    team,
                    seed,
                    100_000,
                    None,
                    None,
                ),
            };
            row.record(&line);
            lines.push(line);
        }
        rows.push(row.finish());
    }
    (lines, rows)
}

fn read_lines(p: &PathBuf) -> Vec<EvalGameLine> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn key(l: &EvalGameLine) -> (String, u32) {
    (l.rung.clone(), l.game)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_workers_reproduce_the_single_process_eval() {
    let (hub, url) = start_hub().await;
    let _w1 = spawn_worker(worker_cfg(&url, "w1", 2));
    let _w2 = spawn_worker(worker_cfg(&url, "w2", 1));

    let dir = tmp("job");
    let id = hub.submit_eval(job(&dir)).unwrap();
    let status = tokio::time::timeout(Duration::from_secs(120), hub.wait(id))
        .await
        .expect("job finished in time")
        .expect("job exists");
    assert_eq!(status.state, JobState::Done, "{status:?}");

    let (want_lines, want_rows) = expected();
    let mut got = read_lines(&dir.join("eval.games.jsonl"));
    assert_eq!(got.len(), (2 * GAMES) as usize, "one line per game, no duplicates");
    got.sort_by_key(key);
    let mut want = want_lines;
    want.sort_by_key(key);
    assert_eq!(got, want, "per-game lines differ from the direct computation");

    let report = status.report.expect("report attached");
    assert_eq!(report.ladder, want_rows);
    assert_eq!(report.candidate, "scripted");
    let on_disk: botbowl_play::eval::Report =
        serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(on_disk.ladder, want_rows);

    // Both workers got work.
    let st = hub.inner.lock().unwrap().status();
    let done: BTreeSet<(String, u64)> = st.workers.iter().map(|w| (w.name.clone(), w.games_done)).collect();
    assert_eq!(done.iter().map(|(_, n)| n).sum::<u64>(), 2 * GAMES as u64, "{done:?}");
    assert!(done.iter().all(|(_, n)| *n > 0), "a worker sat idle: {done:?}");
}

/// Plan 042: a rung that names a board plays on it, on every worker, and
/// the lines and rows say which board — reproduced line for line by the
/// direct computation with the same `board`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rungs_on_explicit_boards_carry_the_board_through() {
    let boards = [BoardDims::try_new(12, 7, 2), BoardDims::try_new(16, 9, 4)];
    if boards.iter().any(|b| b.is_err()) {
        eprintln!("skipped: capacity {WIDTH}x{HEIGHT}/{TEAM_SIZE} too small for the test boards");
        return;
    }
    let boards: Vec<BoardDims> = boards.into_iter().map(Result::unwrap).collect();
    let (hub, url) = start_hub().await;
    let _w1 = spawn_worker(worker_cfg(&url, "w1", 2));
    let _w2 = spawn_worker(worker_cfg(&url, "w2", 1));

    let dir = tmp("boards");
    let games = 4u32;
    let req = EvalJobRequest {
        candidate: BotReq::Scripted,
        candidate_label: "scripted".into(),
        candidate_config: None,
        opponent_config: None,
        mcts_iters: 0,
        rungs: boards
            .iter()
            .map(|&b| RungReq {
                name: rung_name("scripted", Some(b)),
                games,
                opponent: BotReq::Scripted,
                board: Some(b),
            })
            .collect(),
        seed: SEED,
        max_steps: 100_000,
        per_game_out: dir.join("eval.games.jsonl"),
        report_out: dir.join("report.json"),
        batch: 2,
    };
    let id = hub.submit_eval(req).unwrap();
    let status = tokio::time::timeout(Duration::from_secs(120), hub.wait(id))
        .await
        .expect("job finished in time")
        .expect("job exists");
    assert_eq!(status.state, JobState::Done, "{status:?}");

    let mut want_lines = Vec::new();
    let mut want_rows = Vec::new();
    for &b in &boards {
        let name = rung_name("scripted", Some(b));
        let mut row = LadderRow::on_board("scripted", Some(b));
        let mut cand = ScriptedBot::new();
        for g in 0..games {
            let (team, seed) = ladder_assignment(SEED, g);
            let line = play_ladder_game(
                &mut cand,
                &mut ScriptedBot::new(),
                &name,
                g,
                team,
                seed,
                100_000,
                Some(b),
                None,
            );
            assert_eq!(line.board.as_deref(), Some(board_label(b).as_str()));
            row.record(&line);
            want_lines.push(line);
        }
        want_rows.push(row.finish());
    }
    let mut got = read_lines(&dir.join("eval.games.jsonl"));
    got.sort_by_key(key);
    want_lines.sort_by_key(key);
    assert_eq!(got, want_lines, "per-game lines differ from the direct computation");
    // The file carries the board tag on every line.
    let text = std::fs::read_to_string(dir.join("eval.games.jsonl")).unwrap();
    assert!(text.lines().all(|l| l.contains("\"board\":\"")), "{text}");
    let report = status.report.expect("report attached");
    assert_eq!(report.ladder, want_rows);
    assert_eq!(report.board_env, "10x5/2,14x7/4");
    assert_eq!(report.ladder[0].opponent, "scripted@10x5/2");
    assert_eq!(report.ladder[0].board.as_deref(), Some("10x5/2"));
}

/// A worker that takes tasks and vanishes must not lose games: they are
/// requeued and the job still completes exactly once per game.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vanishing_worker_gets_its_games_requeued() {
    let (hub, url) = start_hub().await;
    let dir = tmp("requeue");
    let id = hub.submit_eval(job(&dir)).unwrap();

    // Raw client: Hello, swallow the tasks it is given, then drop the socket.
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
        parallel_games: Some(3),
    };
    sink.send(Message::Binary(encode(&hello).into())).await.unwrap();
    let mut tasks = 0;
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(2), stream.next()).await {
        match decode::<ToWorker>(&b).unwrap() {
            ToWorker::Welcome { parallel_games } => assert_eq!(parallel_games, 3),
            ToWorker::Task(_) => tasks += 1,
            other => panic!("{other:?}"),
        }
        if tasks == 3 {
            break;
        }
    }
    assert_eq!(tasks, 3, "ghost should have been handed 3 tasks");
    drop(sink);
    drop(stream);

    // Give the hub a moment to notice, then a real worker finishes the job.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let _w = spawn_worker(worker_cfg(&url, "real", 2));
    let status = tokio::time::timeout(Duration::from_secs(120), hub.wait(id))
        .await
        .expect("job finished in time")
        .unwrap();
    assert_eq!(status.state, JobState::Done, "{status:?}");
    let got = read_lines(&dir.join("eval.games.jsonl"));
    let keys: BTreeSet<(String, u32)> = got.iter().map(key).collect();
    assert_eq!(got.len(), keys.len(), "duplicate lines");
    assert_eq!(keys.len(), (2 * GAMES) as usize);
}

async fn reject_reason(url: &str, hello: ToHub) -> RejectReason {
    let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    sink.send(Message::Binary(encode(&hello).into())).await.unwrap();
    match stream.next().await {
        Some(Ok(Message::Binary(b))) => match decode::<ToWorker>(&b).unwrap() {
            ToWorker::Reject { reason } => reason,
            other => panic!("expected Reject, got {other:?}"),
        },
        other => panic!("expected a frame, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incompatible_workers_are_rejected_with_a_reason() {
    let (_hub, url) = start_hub().await;
    let base = |token: &str, build: BuildInfo, protocol: u32| ToHub::Hello {
        protocol,
        token: token.into(),
        build,
        triple: "test".into(),
        name: "bad".into(),
        cores: 1,
        ram_mb: 0,
        cached_models: vec![],
        parallel_games: None,
    };
    let me = BuildInfo::current();

    assert_eq!(
        reject_reason(&url, base("wrong", me.clone(), PROTOCOL_VERSION)).await,
        RejectReason::BadToken
    );
    assert_eq!(
        reject_reason(&url, base(TOKEN, me.clone(), PROTOCOL_VERSION + 1)).await,
        RejectReason::Protocol { hub: PROTOCOL_VERSION }
    );
    let other_commit = BuildInfo {
        commit: "0000000000000000000000000000000000000000".into(),
        ..me.clone()
    };
    assert_eq!(
        reject_reason(&url, base(TOKEN, other_commit, PROTOCOL_VERSION)).await,
        RejectReason::Commit { hub: me.commit.clone() }
    );
    let mut small = me.clone();
    small.capacity.width += 2;
    assert_eq!(
        reject_reason(&url, base(TOKEN, small, PROTOCOL_VERSION)).await,
        RejectReason::Capacity { hub: me.capacity }
    );
    // A dirty worker is refused only when the hub itself is clean.
    if !me.dirty {
        let dirty = BuildInfo {
            dirty: true,
            ..me.clone()
        };
        assert_eq!(
            reject_reason(&url, base(TOKEN, dirty, PROTOCOL_VERSION)).await,
            RejectReason::Dirty
        );
    }
}

//! Protocol v14: a worker learns what the hub calls each model it is asked to use — whether it
//! already holds the bytes or not — and records it beside the cache entry, so the cache reads as
//! `bbnet_14x7_gen23.onnx` instead of a hash.

use std::path::PathBuf;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use botbowl_hub::api::{BotReq, EvalJobRequest, RungReq};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{
    decode, encode, BuildInfo, Evaluator, ModelId, ModelMeta, SearchConfig, ToHub, ToWorker, PROTOCOL_VERSION,
};
use botbowl_worker::ModelStore;

const TOKEN: &str = "test-token";

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "botbowl-hub-names-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The committed test net, copied under a name carrying a board tag.
fn named_net(dir: &std::path::Path) -> (PathBuf, ModelId) {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../botbowl-nn/tests/fixtures/tiny.onnx"
    ))
    .unwrap();
    let path = dir.join("bbnet_14x7_gen23.onnx");
    std::fs::write(&path, &bytes).unwrap();
    (path, ModelId::of(&bytes))
}

fn job(dir: &std::path::Path, model: PathBuf) -> EvalJobRequest {
    EvalJobRequest {
        candidate: BotReq::Mcts {
            search: SearchConfig::iterations(2),
            evaluator: Evaluator::Nn,
            model: Some(model),
        },
        candidate_label: "mcts".into(),
        candidate_config: None,
        opponent_config: None,
        mcts_iters: 2,
        rungs: vec![RungReq {
            name: "random".into(),
            games: 2,
            opponent: BotReq::Random,
            board: None,
            drives: None,
        }],
        seed: 1,
        max_steps: 100_000,
        per_game_out: dir.join("eval.games.jsonl"),
        report_out: dir.join("report.json"),
        batch: 2,
        sprt: None,
        label: None,
    }
}

/// Connect as a worker holding `cached`, and return the frames up to the first task.
async fn frames_until_task(url: &str, cached: Vec<ModelId>) -> Vec<ToWorker> {
    let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    let (mut sink, mut stream) = ws.split();
    let hello = ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: TOKEN.into(),
        build: BuildInfo::current(),
        triple: "test".into(),
        name: "raw".into(),
        cores: 1,
        ram_mb: 0,
        cached_models: cached,
        parallel_games: Some(1),
    };
    sink.send(Message::Binary(encode(&hello).into())).await.unwrap();
    let mut out = Vec::new();
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
        let frame = decode::<ToWorker>(&b).unwrap();
        let done = matches!(frame, ToWorker::Task(_));
        out.push(frame);
        if done {
            break;
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_arrives_with_its_name_cached_or_not() {
    let dir = tmp("job");
    let (path, id) = named_net(&dir);
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
    let url = format!("ws://{addr}/ws");
    hub.submit_eval(job(&dir, path.clone())).unwrap();

    let name_of = |frames: &[ToWorker]| {
        frames.iter().find_map(|f| match f {
            ToWorker::ModelName {
                id: m, name, source, ..
            } if *m == id => Some((name.clone(), source.clone())),
            _ => None,
        })
    };
    let has_bytes = |frames: &[ToWorker]| {
        frames
            .iter()
            .any(|f| matches!(f, ToWorker::Model { id: m, .. } if *m == id))
    };

    // Already holds the bytes: named, not re-sent. (This connection's task is requeued when it
    // drops, so the next one is handed a task too.)
    let cached = frames_until_task(&url, vec![id]).await;
    let (name, source) = name_of(&cached).expect("a cached model is still named");
    assert_eq!(name, "bbnet_14x7_gen23.onnx");
    assert_eq!(source, path.to_string_lossy());
    assert!(!has_bytes(&cached), "a cached model's bytes are not re-sent");

    tokio::time::sleep(Duration::from_millis(200)).await;
    let fresh = frames_until_task(&url, vec![]).await;
    assert!(name_of(&fresh).is_some() && has_bytes(&fresh), "{fresh:?}");
}

#[test]
fn the_store_writes_a_sidecar_and_keeps_its_first_seen_time() {
    let dir = tmp("store");
    let (path, id) = named_net(&dir);
    let store = ModelStore::open(&dir.join("cache"), None).unwrap();
    store.put(id, &std::fs::read(&path).unwrap()).unwrap();
    store
        .name(
            &id,
            "bbnet_14x7_gen23.onnx",
            "runs/a/models/bbnet_14x7_gen23.onnx",
            "abc",
        )
        .unwrap();
    let sidecar = dir.join("cache").join(format!("{}.json", id.to_hex()));
    let first: ModelMeta = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    assert_eq!(first.name, "bbnet_14x7_gen23.onnx");
    store
        .name(&id, "renamed_14x7.onnx", "runs/b/renamed_14x7.onnx", "def")
        .unwrap();
    let second: ModelMeta = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    assert_eq!(second.name, "renamed_14x7.onnx");
    assert_eq!(second.first_seen_unix, first.first_seen_unix);
    // The sidecar does not disturb the cache's own verification.
    assert_eq!(store.cached_ids(), vec![id]);
}

/// Read frames until `want` matches one, or give up after a quiet second.
async fn wait_for<S>(stream: &mut S, want: impl Fn(&ToWorker) -> bool) -> Option<ToWorker>
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Ok(Some(Ok(Message::Binary(b)))) = tokio::time::timeout(Duration::from_secs(1), stream.next()).await {
        let frame = decode::<ToWorker>(&b).unwrap();
        if want(&frame) {
            return Some(frame);
        }
    }
    None
}

/// No job needed: a worker holding nets that sit in the hub's indexed directories is told their
/// names — on connect when the index is ready, or when it finishes if the worker came first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_startup_index_names_a_cache_without_any_job() {
    let dir = tmp("index");
    let nested = dir.join("runs/loop14/models");
    std::fs::create_dir_all(&nested).unwrap();
    let (_, id) = named_net(&nested);
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
    let url = format!("ws://{addr}/ws");
    let hello = || ToHub::Hello {
        protocol: PROTOCOL_VERSION,
        token: TOKEN.into(),
        build: BuildInfo::current(),
        triple: "test".into(),
        name: "raw".into(),
        cores: 1,
        ram_mb: 0,
        cached_models: vec![id],
        parallel_games: Some(1),
    };
    let named = |f: &ToWorker| matches!(f, ToWorker::ModelName { id: m, name, .. } if *m == id && name == "bbnet_14x7_gen23.onnx");

    // Connected before the index exists: named once it is built.
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut sink, mut early) = ws.split();
    sink.send(Message::Binary(encode(&hello()).into())).await.unwrap();
    assert!(wait_for(&mut early, &named).await.is_none(), "nothing is known yet");
    assert_eq!(hub.index_models(vec![dir.join("runs")]).join().unwrap(), 1);
    assert!(
        wait_for(&mut early, &named).await.is_some(),
        "named when the index finished"
    );

    // Connected after: named on arrival.
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut sink2, mut late) = ws.split();
    sink2.send(Message::Binary(encode(&hello()).into())).await.unwrap();
    assert!(wait_for(&mut late, &named).await.is_some(), "named on connect");
    drop((sink, sink2));
}

//! Plan 069 item 4b: a worker's two stream pools. Generate (and eval) tasks never exceed its
//! `parallel_games`; label tasks fill its `label_streams`, and any idle game stream only while no
//! generate game is waiting. A worker without label streams takes no label task while a generation
//! has games to hand out, and a label job alone still uses every stream (the mc-label phase).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use botbowl_hub::api::{GenerateJobRequest, LabelJobRequest, ShardReq};
use botbowl_hub::label::LabelInput;
use botbowl_hub::state::{Inner, TrajectoryPayload, WorkerConn, WorkerId};
use botbowl_hub_proto::{BuildInfo, Evaluator, LabelConfig, SearchConfig, Task, ToWorker};
use botbowl_play::generate::{GenMode, GenerateConfig, RandomStartBias};

const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");
const CFG: LabelConfig = LabelConfig {
    playouts: 2,
    seed: 1,
    max_steps: 100,
    skip: 0,
};

fn worker(inner: &mut Inner, name: &str, games: u16, labels: u16) -> (WorkerId, mpsc::UnboundedReceiver<ToWorker>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let id = inner.add_worker(WorkerConn {
        name: name.into(),
        build: BuildInfo::current(),
        triple: "test".into(),
        cores: 8,
        ram_mb: 16000,
        parallel_games: games,
        label_streams: labels,
        tx,
        known_models: Default::default(),
        named_models: Default::default(),
        tasks: Default::default(),
        games_done: 0,
        last_seen: Instant::now(),
    });
    (id, rx)
}

/// The tasks a worker has been sent so far: `(task id, is a label task)`.
fn tasks(rx: &mut mpsc::UnboundedReceiver<ToWorker>, into: &mut VecDeque<(u64, bool, u32)>) {
    while let Ok(m) = rx.try_recv() {
        if let ToWorker::Task(t) = m {
            let first = match &t {
                Task::Generate { games, .. } => games[0],
                Task::Label { items, .. } => items[0].item,
                Task::Eval { games, .. } => games[0],
            };
            into.push_back((t.id(), matches!(t, Task::Label { .. }), first));
        }
    }
}

fn generate(games: u32) -> GenerateJobRequest {
    let dir = std::env::temp_dir().join(format!("dispatch-pools-{}-{games}", std::process::id()));
    GenerateJobRequest {
        shards: vec![ShardReq {
            name: "shard0".into(),
            out: dir.join("shard0.jsonl"),
            seed: 1,
            games,
            cfg: GenerateConfig {
                config_name: None,
                exploration: None,
                mode: GenMode::RandomStart,
                search: SearchConfig::iterations(2),
                evaluator: Evaluator::Heuristic,
                model: None,
                max_steps: 10,
                lecture: None,
                difficulty: botbowl_curriculum::lecture::Difficulty::Easy,
                bias: RandomStartBias::default(),
                board_sizes: None,
                next_drive: false,
            },
            model_path: None,
        }],
        truncate: true,
        batch: 1,
        label: None,
        mc_label: None,
        resume: false,
    }
}

/// A label job of `items` one-sample items (never decoded here: the workers are channels).
fn label(inner: &mut Inner, items: u32) -> u64 {
    let spec = botbowl_hub::label::LabelSpec {
        cfg: CFG,
        model: TINY.into(),
        top_up: false,
        chunk_samples: 1,
    };
    let dir = std::env::temp_dir().join(format!("dispatch-pools-label-{}", std::process::id()));
    let mut input = LabelInput::streaming("shard0", &dir.join("in.jsonl"), &dir.join("out.jsonl"), &spec);
    input.stream = None;
    input.lines = Arc::new(vec![b"x".to_vec()]);
    input.seeds = vec![Some(1)];
    input.samples = vec![items as usize];
    input.firsts = vec![None];
    input.untagged = vec![false];
    input.items = (0..items).map(|i| (0, i, i + 1)).collect();
    let req = LabelJobRequest {
        shards: Vec::new(),
        model_path: PathBuf::from(TINY),
        model: TINY.into(),
        cfg: CFG,
        chunk_samples: 1,
        batch: 1,
        label: None,
        top_up: false,
    };
    inner.submit_label(req, vec![input]).unwrap().0
}

#[test]
fn games_stay_in_their_streams_and_labels_fill_the_rest() {
    let mut inner = Inner::default();
    let (w, mut rx) = worker(&mut inner, "gpu", 2, 1);
    let (bare, mut bare_rx) = worker(&mut inner, "laptop", 2, 0);
    let mut sent = VecDeque::new();
    let mut bare_sent = VecDeque::new();
    // Labels queued first, then a generation with plenty of games.
    label(&mut inner, 20);
    tasks(&mut rx, &mut sent);
    tasks(&mut bare_rx, &mut bare_sent);
    // Alone, a label job takes every stream (and a worker without label streams all of its own).
    assert_eq!(sent.iter().filter(|t| t.1).count(), 3, "{sent:?}");
    assert_eq!(bare_sent.iter().filter(|t| t.1).count(), 2, "{bare_sent:?}");
    inner.submit_generate(generate(10)).unwrap();
    // Labels hold the streams until they finish: a label result frees one game stream, and the
    // generation takes it, since its games are waiting.
    for (rx, sent, wid) in [(&mut rx, &mut sent, w), (&mut bare_rx, &mut bare_sent, bare)] {
        while let Some((task, true, item)) = sent.front().copied() {
            sent.pop_front();
            inner.label_done(wid, task, item, botbowl_hub_proto::LabelResult::Labels(vec![0.0]));
            tasks(rx, sent);
        }
    }
    let labels_in = |s: &VecDeque<(u64, bool, u32)>| s.iter().filter(|t| t.1).count();
    let games_in = |s: &VecDeque<(u64, bool, u32)>| s.iter().filter(|t| !t.1).count();
    // With a generation waiting: two game streams and the one label stream; none on the laptop.
    assert_eq!((games_in(&sent), labels_in(&sent)), (2, 1), "{sent:?}");
    assert_eq!((games_in(&bare_sent), labels_in(&bare_sent)), (2, 0), "{bare_sent:?}");
    // Games finish one by one; while any are left to hand out the split holds.
    let mut generated = 0;
    loop {
        let Some(pos) = sent.iter().position(|t| !t.1) else {
            break;
        };
        let (task, _, game) = sent.remove(pos).unwrap();
        inner.trajectory_done(w, task, game, 0, TrajectoryPayload::decode(&[]));
        generated += 1;
        tasks(&mut rx, &mut sent);
        tasks(&mut bare_rx, &mut bare_sent);
        assert!(games_in(&sent) <= 2 && sent.len() <= 3, "{sent:?}");
        if games_in(&sent) + games_in(&bare_sent) == 0 {
            break;
        }
        // Keep the laptop's games moving too.
        if let Some(pos) = bare_sent.iter().position(|t| !t.1) {
            let (task, _, game) = bare_sent.remove(pos).unwrap();
            inner.trajectory_done(bare, task, game, 0, TrajectoryPayload::decode(&[]));
            tasks(&mut bare_rx, &mut bare_sent);
            tasks(&mut rx, &mut sent);
        }
    }
    assert!(generated > 0);
    // Once no game is waiting, labels take the idle game streams of both workers.
    assert_eq!(labels_in(&sent), 3, "{sent:?}");
    assert_eq!(labels_in(&bare_sent), 2, "{bare_sent:?}");
}

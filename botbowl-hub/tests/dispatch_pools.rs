//! Plan 069 item 4b: a worker's two stream pools. Generate (and eval) tasks never exceed its
//! `parallel_games`; label tasks fill its `label_streams`, and any idle game stream only while no
//! generate game is waiting. A worker without label streams takes no label task while a generation
//! has games to hand out, and a label job alone still uses every stream (the mc-label phase).
//!
//! Plan 071 R5: a background job takes only the streams no foreground job is waiting for, under
//! the same rules, and a foreground task that arrives later takes the next stream that frees.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use botbowl_hub::api::{
    BotReq, EvalJobRequest, GenerateJobRequest, JobId, JobState, LabelJobRequest, Priority, RungReq, ShardReq,
};
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
    generate_as("shard0", games, Priority::Foreground)
}

/// A one-shard generate job whose shard (and so its tasks) is called `name`, in its own directory.
fn generate_as(name: &str, games: u32, priority: Priority) -> GenerateJobRequest {
    let dir = std::env::temp_dir().join(format!("dispatch-pools-{}-{name}-{games}", std::process::id()));
    GenerateJobRequest {
        shards: vec![ShardReq {
            name: name.into(),
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
        priority,
    }
}

/// A label job of `items` one-sample items (never decoded here: the workers are channels).
fn label(inner: &mut Inner, items: u32) -> u64 {
    label_as(inner, "shard0", items, Priority::Foreground)
}

fn label_as(inner: &mut Inner, name: &str, items: u32, priority: Priority) -> u64 {
    let spec = botbowl_hub::label::LabelSpec {
        cfg: CFG,
        model: TINY.into(),
        top_up: false,
        chunk_samples: 1,
    };
    let dir = std::env::temp_dir().join(format!("dispatch-pools-label-{}-{name}", std::process::id()));
    let mut input = LabelInput::streaming(name, &dir.join("in.jsonl"), &dir.join("out.jsonl"), &spec);
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
        priority,
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

// -- plan 071 R5: job priority -------------------------------------------------

/// Every task sent so far: `(task id, unit name, first game or item)`; a unit is a generate shard,
/// an eval rung or (`label`) a label shard.
fn named(rx: &mut mpsc::UnboundedReceiver<ToWorker>, into: &mut VecDeque<(u64, String, u32)>) {
    while let Ok(m) = rx.try_recv() {
        if let ToWorker::Task(t) = m {
            let (name, first) = match &t {
                Task::Generate { shard, games, .. } => (shard.clone(), games[0]),
                Task::Eval { rung, games, .. } => (rung.clone(), games[0]),
                Task::Label { items, .. } => ("label".to_string(), items[0].item),
            };
            into.push_back((t.id(), name, first));
        }
    }
}

/// An eval job of one random-vs-random rung called `rung`, in its own directory.
fn eval_as(rung: &str, games: u32, priority: Priority) -> EvalJobRequest {
    let dir = std::env::temp_dir().join(format!("dispatch-pools-eval-{}-{rung}", std::process::id()));
    EvalJobRequest {
        candidate: BotReq::Random,
        candidate_label: "random".into(),
        candidate_config: None,
        opponent_config: None,
        mcts_iters: 0,
        rungs: vec![RungReq {
            name: rung.into(),
            games,
            opponent: BotReq::Random,
            board: None,
            drives: None,
        }],
        seed: 1,
        max_steps: 100,
        per_game_out: dir.join("eval.games.jsonl"),
        report_out: dir.join("report.json"),
        batch: 1,
        sprt: None,
        drive_gate: None,
        label: None,
        resume: false,
        priority,
    }
}

/// Games (items) a job has not handed out yet.
fn waiting(inner: &Inner, id: JobId) -> u32 {
    let s = inner.job_status(id).unwrap();
    s.units.iter().map(|u| u.total - u.done.min(u.total)).sum::<u32>() - s.in_flight
}

fn in_flight(inner: &Inner, id: JobId) -> u32 {
    inner.job_status(id).unwrap().in_flight
}

#[test]
fn background_takes_only_streams_the_foreground_leaves_idle() {
    let mut inner = Inner::default();
    let (w, mut rx) = worker(&mut inner, "gpu", 3, 0);
    let mut sent = VecDeque::new();
    // Alone, a background job drains on every stream.
    let bg = inner
        .submit_generate(generate_as("bg", 30, Priority::Background))
        .unwrap();
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert!(sent.iter().all(|t| t.1 == "bg"), "{sent:?}");
    assert_eq!(inner.job_status(bg).unwrap().priority, Priority::Background);
    // A foreground job arrives with every stream busy: nothing running is preempted...
    let fg = inner
        .submit_generate(generate_as("fg", 5, Priority::Foreground))
        .unwrap();
    assert_eq!(inner.job_status(fg).unwrap().priority, Priority::Foreground);
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 3, "no stream was free: {sent:?}");
    // ...but every stream that frees goes to it until its games are all out; only then does the
    // background job get one again.
    let mut order = Vec::new();
    while let Some((task, _, game)) = sent.pop_front() {
        inner.trajectory_done(w, task, game, 0, TrajectoryPayload::decode(&[]));
        let before = sent.len();
        named(&mut rx, &mut sent);
        order.extend(sent.iter().skip(before).map(|t| t.1.clone()));
    }
    assert_eq!(&order[..5], ["fg"; 5], "{order:?}");
    assert!(order[5..].iter().all(|n| n == "bg"), "{order:?}");
    // Both finish, every game once.
    for (id, games) in [(fg, 5), (bg, 30)] {
        let s = inner.job_status(id).unwrap();
        assert_eq!(s.state, JobState::Done, "{s:?}");
        assert_eq!(s.units.iter().map(|u| u.done).sum::<u32>(), games);
    }
}

#[test]
fn a_background_generation_never_caps_a_foreground_eval() {
    // Plan 047's cap keeps a foreground eval to a third of a worker's game streams while a
    // foreground generation waits. A background generation is not the critical path: the eval
    // takes every stream, and the generation none.
    let mut inner = Inner::default();
    let bg = inner
        .submit_generate(generate_as("bg", 30, Priority::Background))
        .unwrap();
    inner.submit_eval(eval_as("fg-eval", 10, Priority::Foreground)).unwrap();
    let (_w, mut rx) = worker(&mut inner, "gpu", 3, 0);
    let mut sent = VecDeque::new();
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert!(sent.iter().all(|t| t.1 == "fg-eval"), "{sent:?}");
    assert_eq!(in_flight(&inner, bg), 0);
    // The same pair with a foreground generation: the eval is capped at one stream, as before.
    let mut inner = Inner::default();
    inner
        .submit_generate(generate_as("fg", 30, Priority::Foreground))
        .unwrap();
    inner
        .submit_eval(eval_as("fg-eval2", 10, Priority::Foreground))
        .unwrap();
    let (_w, mut rx) = worker(&mut inner, "gpu", 3, 0);
    let mut sent = VecDeque::new();
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert_eq!(sent.iter().filter(|t| t.1 == "fg-eval2").count(), 1, "{sent:?}");
}

#[test]
fn background_evals_and_labels_wait_for_foreground_work() {
    let mut inner = Inner::default();
    let (w, mut rx) = worker(&mut inner, "gpu", 2, 1);
    let mut sent = VecDeque::new();
    // A foreground generation holds both game streams; the label stream sits idle.
    let gen = inner
        .submit_generate(generate_as("fg", 12, Priority::Foreground))
        .unwrap();
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 2, "{sent:?}");
    // A background eval cannot use the label stream: a worker's label threads run label tasks only.
    let ev = inner.submit_eval(eval_as("bg-eval", 4, Priority::Background)).unwrap();
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 2, "{sent:?}");
    // A foreground label job takes the idle label stream; a background one queued behind it waits.
    let fl = label_as(&mut inner, "lfg", 3, Priority::Foreground);
    let bl = label_as(&mut inner, "lbg", 4, Priority::Background);
    named(&mut rx, &mut sent);
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert_eq!((in_flight(&inner, fl), in_flight(&inner, bl)), (1, 0));
    // Run everything but the background eval's games (no worker here plays them) to the end,
    // checking after every result that no background task went out while foreground work that
    // could use the same kind of stream was waiting.
    let mut evals = Vec::new();
    while let Some((task, name, game)) = sent.pop_front() {
        match name.as_str() {
            "label" => {
                inner.label_done(w, task, game, botbowl_hub_proto::LabelResult::Labels(vec![0.0]));
            }
            "bg-eval" => {
                evals.push(task);
                continue;
            }
            _ => {
                inner.trajectory_done(w, task, game, 0, TrajectoryPayload::decode(&[]));
            }
        }
        named(&mut rx, &mut sent);
        if waiting(&inner, gen) > 0 {
            assert_eq!(
                in_flight(&inner, ev),
                0,
                "a background eval took a game stream: {sent:?}"
            );
        }
        if waiting(&inner, fl) > 0 {
            assert_eq!(
                in_flight(&inner, bl),
                0,
                "a background label took a label stream: {sent:?}"
            );
        }
    }
    // The foreground jobs are done; the background label job finished on the streams they left
    // (every item in: a label job is `Done` once its shard is written, which no test thread runs
    // here), and the background eval took both game streams once the generation's games were out.
    assert_eq!(inner.job_status(gen).unwrap().state, JobState::Done);
    for (id, items) in [(fl, 3), (bl, 4)] {
        let s = inner.job_status(id).unwrap();
        assert_eq!((s.units[0].done, s.in_flight), (items, 0), "{s:?}");
    }
    assert_eq!(evals.len(), 2, "both game streams went to the background eval");
}

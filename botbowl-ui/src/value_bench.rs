//! The value benchmark (plan 056 §2): how far is a net's V(s) from Monte Carlo truth?
//!
//! The benchmark is a frozen set of corpus decisions with MC(s), the policy-only drive outcome
//! from `s` averaged over many playouts (`policy.mc_*` of `override-audit` rows, frozen by
//! `scripts/value_bench_freeze.py`). Scoring a net is one forward per state: each state is
//! rebuilt by [`replay_to`] (a deserialised state has lost its path buffer, and the encoder sees
//! it), and `v` is the net's value in the mover's frame, as `override-audit`'s `v_state`. Every
//! benchmark line is written back with `v` and `model` added; `scripts/value_bench_summary.py`
//! turns one or more outputs into RMS (MC sampling noise removed) and bias, by board and phase,
//! and pairs two nets state by state.
//!
//! ```text
//! BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
//! cargo run --release -p botbowl-ui -- value-bench --bench runs/value_bench/g05_gen06.jsonl \
//!     --model models/az_v7/bbnet_mix16x9g_gen05.onnx --out runs/value_bench/g05_gen06.g05.jsonl
//! scripts/value_bench_summary.py runs/value_bench/g05_gen06.g05.jsonl
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{Map, Value};

use botbowl_data::{BoardCapacity, Team};
use botbowl_play::bots::{load_nn, Evaluator};
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::ValueBenchArgs;
use crate::override_audit::{load_trajectory, replay_to, value_for};

/// One benchmark state: where it lives in the corpus, plus the line itself, written back.
struct Entry {
    corpus: String,
    line: usize,
    sample: usize,
    raw: Map<String, Value>,
}

fn parse_entry(text: &str, at: usize) -> io::Result<Entry> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, format!("benchmark line {at}: {m}"));
    let raw: Map<String, Value> = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
    let corpus = raw
        .get("corpus")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("no corpus".into()))?;
    let num = |k: &str| raw.get(k).and_then(Value::as_u64).ok_or_else(|| bad(format!("no {k}")));
    Ok(Entry {
        corpus: corpus.to_string(),
        line: num("line")? as usize,
        sample: num("sample")? as usize,
        raw: raw.clone(),
    })
}

/// Byte offsets of the wanted 1-based lines of `path`, in one pass.
fn line_offsets(path: &str, wanted: &HashSet<usize>) -> io::Result<HashMap<usize, u64>> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?);
    let (mut out, mut offset, mut li, mut buf) = (HashMap::new(), 0u64, 0usize, Vec::new());
    while out.len() < wanted.len() {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        li += 1;
        if wanted.contains(&li) {
            out.insert(li, offset);
        }
        offset += n as u64;
    }
    Ok(out)
}

/// `Ok(None)` when the state cannot be rebuilt or is not the benchmark's (named on stderr).
fn score(
    nn: &botbowl_nn::eval::NnEvaluator,
    e: &Entry,
    offset: Option<u64>,
    model: &str,
) -> io::Result<Option<String>> {
    let id = format!("{}:{}:{}", e.corpus, e.line, e.sample);
    let Some(offset) = offset else {
        eprintln!("skipped {id}: the corpus has no line {}", e.line);
        return Ok(None);
    };
    let traj = load_trajectory(&e.corpus, offset)?;
    if traj.meta.board_capacity != BoardCapacity::current() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: corpus built at capacity {:?}, this binary is {:?}; rebuild with the matching BOARD_SIZE_*",
                e.corpus,
                traj.meta.board_capacity,
                BoardCapacity::current()
            ),
        ));
    }
    let state = match replay_to(&traj, e.sample) {
        Ok(s) => s,
        Err(why) => {
            eprintln!("skipped {id}: {why}");
            return Ok(None);
        }
    };
    let Some(mover) = state.available_actions.team else {
        eprintln!("skipped {id}: no team to move");
        return Ok(None);
    };
    if let Some(want) = e.raw.get("mover") {
        if *want != serde_json::to_value(Team::from(mover))? {
            eprintln!("skipped {id}: the benchmark's mover is {want}, the replayed state's is {mover:?}");
            return Ok(None);
        }
    }
    let mut row = e.raw.clone();
    row.insert("v".into(), Value::from(value_for(nn, &state, mover) as f64));
    row.insert("model".into(), Value::from(model));
    Ok(Some(serde_json::to_string(&row)?))
}

pub fn run(args: ValueBenchArgs) -> io::Result<()> {
    let started = Instant::now();
    let mut entries = Vec::new();
    for (i, line) in BufReader::new(File::open(&args.bench)?).lines().enumerate() {
        let line = line?;
        if !line.trim().is_empty() {
            entries.push(parse_entry(&line, i + 1)?);
        }
    }
    let mut wanted: HashMap<&str, HashSet<usize>> = HashMap::new();
    for e in &entries {
        wanted.entry(&e.corpus).or_default().insert(e.line);
    }
    let mut offsets: HashMap<(String, usize), u64> = HashMap::new();
    for (path, lines) in &wanted {
        for (li, off) in line_offsets(path, lines)? {
            offsets.insert((path.to_string(), li), off);
        }
    }

    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        Evaluator::Nn,
        Some(&args.model),
        "--model is required",
        server.as_deref(),
    )?
    .expect("Evaluator::Nn always loads a net");

    let out = Mutex::new(File::create(&args.out)?);
    let next = AtomicUsize::new(0);
    let scored = AtomicUsize::new(0);
    let worker = || -> io::Result<()> {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(e) = entries.get(i) else { return Ok(()) };
            let offset = offsets.get(&(e.corpus.clone(), e.line)).copied();
            if let Some(line) = score(&nn, e, offset, &args.model)? {
                let mut f = out.lock().expect("output mutex");
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
                scored.fetch_add(1, Ordering::Relaxed);
            }
        }
    };
    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = (0..args.parallel.max(1))
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("vbench-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(scope, worker)
                    .expect("spawn value-bench thread")
            })
            .collect();
        for h in handles {
            h.join().expect("value-bench thread panicked")?;
        }
        Ok(())
    })?;
    out.lock().expect("output mutex").flush()?;
    eprintln!(
        "value-bench: scored {} of {} states from {} on {} in {:.0}s",
        scored.load(Ordering::Relaxed),
        entries.len(),
        args.bench,
        args.model,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_data::Trajectory;
    use botbowl_engine::core::model::BoardDims;
    use botbowl_nn::eval::NnEvaluator;
    use botbowl_play::bots::SearchConfig;
    use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};

    const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");

    fn trajectory(seed: u64) -> Trajectory {
        let board = BoardDims::try_new(16, 9, 4).unwrap_or_else(|_| BoardDims::from_env());
        let cfg = GenerateConfig {
            mode: GenMode::RandomStart,
            search: SearchConfig::iterations(4),
            evaluator: Evaluator::Heuristic,
            model: None,
            max_steps: 40,
            lecture: None,
            difficulty: botbowl_curriculum::Difficulty::Easy,
            bias: RandomStartBias::default(),
            board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
            config_name: None,
            exploration: None,
        };
        play_trajectory(&cfg, None, seed).unwrap().unwrap()
    }

    /// Every benchmark line comes back with the net's V of the *replayed* state in the mover's
    /// frame, its own fields untouched; a line the corpus does not have, or whose mover is wrong,
    /// is skipped rather than scored on the wrong state.
    #[test]
    fn scores_the_replayed_state_in_the_movers_frame_and_skips_what_it_cannot_rebuild() {
        let dir = std::env::temp_dir().join(format!("value-bench-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("shard0.jsonl");
        let trajs = [trajectory(4242), trajectory(4243)];
        let mut f = File::create(&corpus).unwrap();
        for t in &trajs {
            writeln!(f, "{}", serde_json::to_string(t).unwrap()).unwrap();
        }
        drop(f);
        let corpus = corpus.to_str().unwrap().to_string();

        let nn = NnEvaluator::from_path(TINY).unwrap();
        let mut bench = String::new();
        let mut want = HashMap::new();
        for (li, t) in trajs.iter().enumerate() {
            for k in (0..t.samples.len()).step_by(3) {
                let state = replay_to(t, k).unwrap();
                let Some(mover) = state.available_actions.team else {
                    continue;
                };
                let id = format!("{corpus}:{}:{k}", li + 1);
                want.insert(id.clone(), value_for(&nn, &state, mover));
                let mover = serde_json::to_value(Team::from(mover)).unwrap();
                bench += &format!(
                    "{}\n",
                    serde_json::json!({"id": id, "corpus": corpus, "line": li + 1, "sample": k,
                                       "mover": mover, "mc": 0.25, "phase": "mid_turn"})
                );
            }
        }
        assert!(want.len() >= 4, "too few benchmark states: {}", want.len());
        // A line the corpus does not have, and a state with the other mover.
        bench += &format!(
            "{}\n",
            serde_json::json!({"id": "missing", "corpus": corpus, "line": 9, "sample": 0})
        );
        let t0 = replay_to(&trajs[0], 0).unwrap();
        let other = match t0.available_actions.team.unwrap() {
            botbowl_engine::core::model::TeamType::Home => "Away",
            botbowl_engine::core::model::TeamType::Away => "Home",
        };
        bench += &format!(
            "{}\n",
            serde_json::json!({"id": "wrong-mover", "corpus": corpus, "line": 1, "sample": 0, "mover": other})
        );
        let bench_path = dir.join("bench.jsonl");
        std::fs::write(&bench_path, bench).unwrap();
        let out = dir.join("out.jsonl");
        run(ValueBenchArgs {
            bench: bench_path.to_str().unwrap().into(),
            model: TINY.into(),
            nn_server: None,
            parallel: 3,
            out: out.to_str().unwrap().into(),
        })
        .unwrap();

        let rows: Vec<Map<String, Value>> = std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            rows.len(),
            want.len(),
            "every good line scored, the two bad ones skipped"
        );
        for r in &rows {
            let id = r["id"].as_str().unwrap();
            let v = r["v"].as_f64().unwrap() as f32;
            assert_eq!(v, want[id], "{id}");
            assert_eq!(r["mc"], 0.25);
            assert_eq!(r["phase"], "mid_turn");
            assert_eq!(r["model"], TINY);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

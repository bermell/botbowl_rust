//! Per-forward latency of `NnEvaluator`, off the search.
//!
//! Times `forward_raw` on the committed 9x16 canary input, single sample
//! at a time, the way the search issues it. With `--server SOCK` the same
//! calls go through `RemoteClient` to a running `scripts/nn_server.py`,
//! so the number is the full Rust→socket→Python→GPU→socket→Rust round
//! trip that one MCTS leaf pays; without it the number is tract in-process.
//!
//! ```sh
//! BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4 \
//!   cargo run --release -p botbowl-nn --example nn_bench -- MODEL.onnx [--server SOCK] [--threads N] [--iters N] [--policy]
//! ```
//!
//! `--threads N` runs N independent client threads, each with its own
//! connection, and reports per-thread latency plus aggregate throughput —
//! that is what `--parallel-games N` offers the server.

use std::path::PathBuf;
use std::time::Instant;

use botbowl_nn::eval::NnEvaluator;
use botbowl_nn::remote::{canary_input, CANARY_H, CANARY_W};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model: Option<PathBuf> = None;
    let mut server: Option<PathBuf> = None;
    let mut threads = 1usize;
    let mut iters = 2000usize;
    let mut want_policy = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--server" => {
                server = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--threads" => {
                threads = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--iters" => {
                iters = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--policy" => want_policy = true,
            other => model = Some(PathBuf::from(other)),
        }
        i += 1;
    }
    let model = model.expect("usage: nn_bench MODEL.onnx [--server SOCK] [--threads N] [--iters N] [--policy]");
    let eval = NnEvaluator::from_path_with_server(&model, server.as_deref()).expect("load model");
    let (spatial, global) = canary_input();

    // Warm: builds the tract plan / opens the connection.
    for _ in 0..20 {
        eval.forward_raw(&spatial, &global, CANARY_H, CANARY_W);
    }

    let cpu0 = cpu_time();
    let t0 = Instant::now();
    let per_thread: Vec<Vec<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut lat = Vec::with_capacity(iters);
                    // Perturb the input per call so the memo in eval.rs cannot hit.
                    let mut sp = spatial.clone();
                    let len = sp.len();
                    for k in 0..iters {
                        sp[k % len] += 1e-3;
                        let t = Instant::now();
                        if want_policy {
                            eval.forward_raw(&sp, &global, CANARY_H, CANARY_W);
                        } else {
                            eval.value_only_raw(&sp, &global, CANARY_H, CANARY_W);
                        }
                        lat.push(t.elapsed().as_secs_f64() * 1e6);
                    }
                    lat
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let cpu = cpu_time() - cpu0;

    let mut all: Vec<f64> = per_thread.iter().flatten().copied().collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = all.len();
    let pct = |p: f64| all[((n as f64 * p) as usize).min(n - 1)];
    let backend = if server.is_some() { "remote" } else { "tract" };
    println!(
        "{backend} threads={threads} policy={want_policy} n={n}: median {:.0} us  p10 {:.0}  p90 {:.0}  p99 {:.0}  | {:.0} forwards/s aggregate | {:.0} us CPU/forward (this process)",
        pct(0.5),
        pct(0.1),
        pct(0.9),
        pct(0.99),
        n as f64 / wall,
        cpu / n as f64 * 1e6
    );
    if let Some((served, fell_back)) = eval.remote_stats() {
        println!("remote served={served} fell_back={fell_back}");
    }
}

/// Process CPU time (user + sys) in seconds, from /proc.
fn cpu_time() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the ")" : state is field 3; utime is 14, stime is 15 (1-based).
    let after = stat.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    let ticks: f64 = f.get(11).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
        + f.get(12).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    ticks / 100.0
}

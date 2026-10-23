//! Net throughput with no search (plan 069 item 4): the first, cheapest row of the inference tuning
//! suite (`hpc/tune/`). Two modes, both on the in-process backend's own code:
//!
//! ```sh
//! # direct: per bucket, `lanes` threads each replay that bucket's graph on their own lane as fast
//! # as it returns, restaging every row first (what the batcher does per batch, minus the queue).
//! cargo run --release -p botbowl-nn --features torch --example inproc_bench -- \
//!     --ts NET.fp16.ts --mode direct --batches 1,16,64,256,1024 --lanes 2 --phase-secs 8
//! # queue: N client threads call `BatchQueue::forward` back to back (zero search CPU), so the
//! # queue, the batchers and the buckets are measured at a given stream count.
//! ... --mode queue --clients 64,256,1024 --buckets 1,2,4,8,16,32,64,128,256,512,1024 --lanes 2
//! ```
//!
//! Numerics come from the file (`--ts`: the plain export or a `bbnn.export_ts --dtype` variant) and
//! the process flags (`BLOOD_NN_PRECISION` for fp32/tf32, `BLOOD_NN_DETERMINISTIC`); the canvas from
//! `BLOOD_NN_CANVAS` (default: the compiled capacity). Inputs are fixed pseudo-random planes on the
//! boards of `--boards` (tensor `HxW`, border included; default the full canvas) — throughput does
//! not depend on the values.
//!
//! Protocol with the suite's driver (`hpc/tune/tune.py`), on stdout, one line each:
//! `BENCH_READY …` after load; per phase `PHASE_START id=K secs=S` right before a measured phase of
//! S seconds (the driver counts the process's CPU, instructions and the GPU over it) and a result
//! line `MICRO key=value …` after it. Each phase is preceded by `--warm-secs` of the same work.

#[cfg(not(feature = "torch"))]
fn main() {
    eprintln!("inproc_bench needs the in-process backend: rebuild with --features torch");
    std::process::exit(2);
}

#[cfg(feature = "torch")]
fn main() {
    bench::main()
}

#[cfg(feature = "torch")]
mod bench {
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    use botbowl_nn::actions::POLICY_CHANNELS;
    use botbowl_nn::batch_queue::{stage_canvas, BatchQueue, Executor, Geometry};
    use botbowl_nn::encode::{GLOBAL_FEATURES, SPATIAL_CHANNELS};
    use botbowl_nn::inproc::{
        buckets_from_env, crop_policy, parse_buckets, set_cudnn, set_precision, sidecar_validation_calls, CudnnMode,
        GraphRunner, InProcConfig, Precision, TorchExecutor, TorchModel,
    };

    fn arg(args: &[String], name: &str) -> Option<String> {
        args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
    }

    fn list(v: &str) -> Vec<usize> {
        v.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().expect("a number")).collect()
    }

    /// On-CPU time of the calling thread, ns (`/proc/thread-self/schedstat`, no privileges needed).
    fn thread_cpu_ns() -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .ok()
            .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
            .unwrap_or(0)
    }

    fn say(line: String) {
        let mut o = std::io::stdout().lock();
        let _ = writeln!(o, "{line}");
        let _ = o.flush();
    }

    /// One board's unstaged input: `C × h × w` planes in [0, 1) and `F` globals.
    fn synthetic(h: usize, w: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut x = seed.wrapping_mul(0x9e37_79b9) | 1;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as f32 / u32::MAX as f32
        };
        let s = (0..SPATIAL_CHANNELS * h * w).map(|_| rnd()).collect();
        let g = (0..GLOBAL_FEATURES).map(|_| rnd()).collect();
        (s, g)
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let ts = PathBuf::from(arg(&args, "--ts").expect("--ts NET.ts"));
        let device = arg(&args, "--device").unwrap_or_else(|| "cuda".into());
        let mode = arg(&args, "--mode").unwrap_or_else(|| "direct".into());
        let lanes: usize = arg(&args, "--lanes").map_or(1, |v| v.parse().unwrap());
        let phase = Duration::from_secs_f64(arg(&args, "--phase-secs").map_or(8.0, |v| v.parse().unwrap()));
        let warm = Duration::from_secs_f64(arg(&args, "--warm-secs").map_or(1.0, |v| v.parse().unwrap()));
        let want_policy = arg(&args, "--policy").is_none_or(|v| v != "0");
        let cfg = InProcConfig::from_env(&device).unwrap();
        let geo = Geometry {
            c: SPATIAL_CHANNELS,
            f: GLOBAL_FEATURES,
            a: POLICY_CHANNELS,
            canvas_h: cfg.canvas.0,
            canvas_w: cfg.canvas.1,
        };
        let boards: Vec<(usize, usize)> = arg(&args, "--boards")
            .map(|v| {
                v.split(',')
                    .map(|b| {
                        let (h, w) = b.split_once('x').expect("--boards HxW,...");
                        (h.parse().unwrap(), w.parse().unwrap())
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(geo.canvas_h, geo.canvas_w)]);
        assert!(boards.iter().all(|&(h, w)| h <= geo.canvas_h && w <= geo.canvas_w), "a board exceeds the canvas");

        let precision = Precision::from_env().unwrap();
        set_cudnn(CudnnMode::from_env()).unwrap();
        set_precision(precision).unwrap();
        let t0 = Instant::now();
        let model = TorchModel::load(&ts, &device, cfg.optimize).unwrap();
        let variant = model.variant();
        sidecar_validation_calls(&model, &geo).unwrap();
        say(format!(
            "BENCH_READY load_s={:.1} file={} variant={} precision={precision:?} canvas={}x{} mode={mode} lanes={lanes} policy={}",
            t0.elapsed().as_secs_f64(),
            ts.display(),
            if variant.is_empty() { "fp32" } else { &variant },
            geo.canvas_h,
            geo.canvas_w,
            want_policy as u8
        ));
        // A pool of staged rows to copy from, cycled: the per-batch copy is the batcher's real work.
        let pool: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> = (0..64)
            .map(|i| {
                let (h, w) = boards[i % boards.len()];
                let (s, g) = synthetic(h, w, i as u32 + 1);
                let (cs, m) = stage_canvas(&geo, &s, h, w);
                (cs, g, m, h, w)
            })
            .collect();
        let mut phase_id = 0usize;
        match mode.as_str() {
            "direct" => {
                let batches = list(&arg(&args, "--batches").unwrap_or_else(|| "1,8,64".into()));
                for b in batches {
                    direct(&model, geo, &pool, b, lanes, want_policy, warm, phase, phase_id);
                    phase_id += 1;
                }
            }
            "queue" => {
                let buckets = match arg(&args, "--buckets") {
                    Some(v) => parse_buckets(&v).unwrap(),
                    None => buckets_from_env().unwrap(),
                };
                let clients = list(&arg(&args, "--clients").unwrap_or_else(|| "64".into()));
                let slots = cfg.slots.max(clients.iter().copied().max().unwrap_or(1) + 64);
                let t1 = Instant::now();
                let exec = Arc::new(TorchExecutor::capture_buckets(model, geo, lanes, &buckets).unwrap());
                say(format!(
                    "BENCH_CAPTURED graphs={} capture_s={:.1}",
                    lanes * buckets.len(),
                    t1.elapsed().as_secs_f64()
                ));
                let queue = BatchQueue::new(exec.clone() as Arc<dyn Executor>, geo, slots);
                let inputs: Vec<(Vec<f32>, Vec<f32>, usize, usize)> = (0..64)
                    .map(|i| {
                        let (h, w) = boards[i % boards.len()];
                        let (s, g) = synthetic(h, w, i as u32 + 1);
                        (s, g, h, w)
                    })
                    .collect();
                for c in clients {
                    queued(&queue, &inputs, c, want_policy, warm, phase, phase_id, buckets[buckets.len() - 1]);
                    phase_id += 1;
                }
            }
            other => panic!("--mode {other}: want direct or queue"),
        }
        say("BENCH_DONE".into());
    }

    /// One bucket: capture it on every lane, warm, then measure `lanes` threads replaying it.
    #[allow(clippy::too_many_arguments)]
    fn direct(
        model: &TorchModel,
        geo: Geometry,
        pool: &[(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)],
        b: usize,
        lanes: usize,
        want_policy: bool,
        warm: Duration,
        phase: Duration,
        id: usize,
    ) {
        let t1 = Instant::now();
        let runners: Vec<GraphRunner> = (0..lanes)
            .map(|lane| GraphRunner::new(model, geo, b, lane).unwrap_or_else(|e| panic!("bucket {b}: {e}")))
            .collect();
        let capture_s = t1.elapsed().as_secs_f64();
        let start = Arc::new(Barrier::new(lanes + 1));
        let measure = Arc::new(Barrier::new(lanes + 1));
        let results: Vec<(GraphRunner, u64, Duration, u64)> = std::thread::scope(|s| {
            let handles: Vec<_> = runners
                .into_iter()
                .map(|mut r| {
                    let (start, measure) = (start.clone(), measure.clone());
                    s.spawn(move || {
                        let mut out = Vec::new();
                        let mut k = 0usize;
                        let mut one = |r: &mut GraphRunner| -> Duration {
                            let base = k;
                            k += b;
                            for i in 0..b {
                                let (sp, g, m, _, _) = &pool[(base + i) % pool.len()];
                                r.stage_row(i, sp, g, m);
                            }
                            let d = r.run(b, want_policy).unwrap();
                            if want_policy {
                                for i in 0..b {
                                    let (_, _, _, h, w) = pool[(base + i) % pool.len()];
                                    crop_policy(r.policy_row(i), &geo, h, w, &mut out);
                                }
                            }
                            std::hint::black_box(r.values()[0]);
                            d
                        };
                        start.wait();
                        let until = Instant::now() + warm;
                        while Instant::now() < until {
                            one(&mut r);
                        }
                        measure.wait();
                        let (mut n, mut dev) = (0u64, Duration::ZERO);
                        let cpu0 = thread_cpu_ns();
                        let until = Instant::now() + phase;
                        while Instant::now() < until {
                            dev += one(&mut r);
                            n += 1;
                        }
                        (r, n, dev, thread_cpu_ns() - cpu0)
                    })
                })
                .collect();
            start.wait();
            measure.wait();
            say(format!("PHASE_START id={id} secs={:.1}", phase.as_secs_f64()));
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let secs = phase.as_secs_f64();
        let n: u64 = results.iter().map(|r| r.1).sum();
        let dev: f64 = results.iter().map(|r| r.2.as_secs_f64()).sum();
        let cpu: u64 = results.iter().map(|r| r.3).sum();
        say(format!(
            "MICRO id={id} mode=direct bucket={b} lanes={lanes} policy={} batches={n} secs={secs:.2} capture_s={capture_s:.2} \
             positions_per_s={:.0} batches_per_s={:.1} device_us_per_batch={:.1} wall_us_per_batch_per_lane={:.1} \
             lane_cpu_us_per_batch={:.1} lane_cpu_us_per_position={:.3} device_busy={:.3}",
            want_policy as u8,
            (n as f64 * b as f64) / secs,
            n as f64 / secs,
            1e6 * dev / n.max(1) as f64,
            1e6 * secs * lanes as f64 / n.max(1) as f64,
            cpu as f64 / 1e3 / n.max(1) as f64,
            cpu as f64 / 1e3 / (n.max(1) * b as u64) as f64,
            dev / secs
        ));
        for (r, ..) in results {
            r.release();
        }
    }

    /// `clients` threads calling the queue back to back.
    #[allow(clippy::too_many_arguments)]
    fn queued(
        queue: &BatchQueue,
        inputs: &[(Vec<f32>, Vec<f32>, usize, usize)],
        clients: usize,
        want_policy: bool,
        warm: Duration,
        phase: Duration,
        id: usize,
        max_bucket: usize,
    ) {
        let start = Arc::new(Barrier::new(clients + 1));
        let measure = Arc::new(Barrier::new(clients + 1));
        let mut before = None;
        let counts: Vec<u64> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..clients)
                .map(|t| {
                    let (start, measure) = (start.clone(), measure.clone());
                    s.spawn(move || {
                        let mut k = t;
                        let mut one = || {
                            let (sp, g, h, w) = &inputs[k % inputs.len()];
                            k += 1;
                            std::hint::black_box(queue.forward(sp, g, *h, *w, want_policy).unwrap());
                        };
                        start.wait();
                        let until = Instant::now() + warm;
                        while Instant::now() < until {
                            one();
                        }
                        measure.wait();
                        let mut n = 0u64;
                        let until = Instant::now() + phase;
                        while Instant::now() < until {
                            one();
                            n += 1;
                        }
                        n
                    })
                })
                .collect();
            start.wait();
            measure.wait();
            before = Some(queue.stats());
            say(format!("PHASE_START id={id} secs={:.1}", phase.as_secs_f64()));
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (a, z) = (before.unwrap(), queue.stats());
        let secs = phase.as_secs_f64();
        let n: u64 = counts.iter().sum();
        let batches = (z.batches - a.batches).max(1) as f64;
        let samples = (z.samples - a.samples).max(1) as f64;
        say(format!(
            "MICRO id={id} mode=queue clients={clients} max_bucket={max_bucket} policy={} forwards={n} secs={secs:.2} \
             positions_per_s={:.0} mean_batch={:.2} pad_per_sample={:.3} device_us_per_batch={:.1} queue_us_per_sample={:.1} \
             device_busy={:.3}",
            want_policy as u8,
            n as f64 / secs,
            samples / batches,
            (z.padded - a.padded) as f64 / samples,
            (z.device - a.device).as_secs_f64() * 1e6 / batches,
            (z.queue - a.queue).as_secs_f64() * 1e6 / samples,
            (z.device - a.device).as_secs_f64() / (z.elapsed - a.elapsed).as_secs_f64().max(1e-9)
        ));
    }
}

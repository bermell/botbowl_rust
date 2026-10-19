//! The in-process backend's contract (plan 066), on the committed fixture nets.
//!
//! Compiled only with `--features torch`, and each test skips (with a line saying so) when libtorch
//! sees no CUDA device, so `cargo test --workspace` stays green on any machine. The bit-level parity
//! with the sidecar is not here — it needs the real net, the corpus and the Python sidecar, and lives
//! in `scripts/parity066.sh` (plan 066 §4) — these pin the plumbing around it: the canary interlock,
//! numerics against tract (the reference), batch invariance under concurrency, every head knob, and
//! that a missing export or an unusable device is a clear error rather than a silent tract run.
#![cfg(feature = "torch")]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use botbowl_nn::eval::{configure_backend, BackendChoice, NnEvaluator};
use botbowl_nn::inproc::{cuda_device_count, InProcBackend, InProcConfig};
use botbowl_nn::remote::{canary_input, CANARY_H, CANARY_W};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn have_gpu() -> bool {
    let n = cuda_device_count();
    if n == 0 {
        eprintln!("skipped: libtorch sees no CUDA device");
    }
    n > 0
}

fn perturb(spatial: &[f32], k: u32) -> Vec<f32> {
    let mut s = spatial.to_vec();
    let i = (k as usize * 7919) % s.len();
    s[i] += 0.25 * (k as f32 + 1.0);
    s
}

fn npy(path: PathBuf) -> Vec<f32> {
    botbowl_nn::npy::parse(&std::fs::read(&path).unwrap()).unwrap().as_f32()
}

/// Through `NnEvaluator`, the way every tool gets it: the process-wide choice, the `.ts` beside
/// the ONNX, the canary at load. Then tract agreement on fresh inputs, and batch invariance:
/// 24 threads with different inputs, each compared with its own batch-1 reference.
#[test]
fn evaluator_matches_tract_and_is_batch_invariant() {
    if !have_gpu() {
        return;
    }
    let choice = configure_backend(Some("inproc"), Some("cuda")).unwrap();
    assert!(matches!(choice, BackendChoice::InProcess { .. }));
    let onnx = fixtures().join("tiny.onnx");
    let inproc = Arc::new(NnEvaluator::from_path_with_server(&onnx, None).expect("in-process load + canary"));
    let tract = NnEvaluator::from_path(&onnx).unwrap();
    let (cs, cg) = canary_input();

    for k in 0..6u32 {
        let s = perturb(&cs, k);
        let (tp, tv) = tract.forward_raw(&s, &cg, CANARY_H, CANARY_W);
        let (ip, iv) = inproc.forward_raw(&s, &cg, CANARY_H, CANARY_W);
        assert!((tv - iv).abs() < 1e-3, "value {iv} vs tract {tv}");
        assert!(max_abs_diff(&ip, &tp) < 1e-3, "policy vs tract");
        assert_eq!(inproc.value_only_raw(&s, &cg, CANARY_H, CANARY_W), iv, "value-only path == policy path");
    }

    let refs: Arc<Vec<f32>> = Arc::new(
        (0..24)
            .map(|k| inproc.value_only_raw(&perturb(&cs, k), &cg, CANARY_H, CANARY_W))
            .collect(),
    );
    let handles: Vec<_> = (0..24u32)
        .map(|k| {
            let (inproc, refs, cs, cg) = (Arc::clone(&inproc), Arc::clone(&refs), cs.clone(), cg.clone());
            thread::spawn(move || {
                let s = perturb(&cs, k);
                let mut worst = 0.0f32;
                for i in 0..40 {
                    let v = if i % 3 == 0 {
                        inproc.forward_raw(&s, &cg, CANARY_H, CANARY_W).1
                    } else {
                        inproc.value_only_raw(&s, &cg, CANARY_H, CANARY_W)
                    };
                    worst = worst.max((v - refs[k as usize]).abs());
                }
                worst
            })
        })
        .collect();
    for (k, h) in handles.into_iter().enumerate() {
        let d = h.join().unwrap();
        assert!(d < 1e-5, "sample {k}: max |Δ| {d} vs its batch-1 reference — batch composition leaked");
    }
    assert_eq!(inproc.remote_stats().unwrap().1, 0, "no fallbacks");
    let line = inproc.inproc_stats_line().unwrap();
    assert!(line.contains("mean_batch="), "{line}");

    // A second evaluator on the same net shares the loaded backend (one queue, one set of graphs).
    let again = NnEvaluator::from_path_with_server(&onnx, None).expect("second evaluator");
    assert_eq!(again.value_only_raw(&cs, &cg, CANARY_H, CANARY_W), inproc.value_only_raw(&cs, &cg, CANARY_H, CANARY_W));
}

/// Every non-default head knob (3x3 policy conv, mean+max pooling, wider value head), at both of
/// the heads fixture's boards, staged on the canvas: the in-process backend against tract and
/// against the PyTorch reference outputs committed with the fixture.
#[test]
fn heads_net_matches_tract_on_the_canvas() {
    if !have_gpu() {
        return;
    }
    let dir = fixtures().join("heads");
    let cfg = InProcConfig::from_env("cuda").unwrap();
    let backend = InProcBackend::start(&dir.join("tiny_heads.ts"), &cfg).expect("load heads net");
    let tract = NnEvaluator::from_path(dir.join("tiny_heads.onnx")).unwrap();
    for (h, w) in [(9usize, 16usize), (7, 12)] {
        let s = npy(dir.join(format!("parity_{h}x{w}_spatial.npy")));
        let g = npy(dir.join(format!("parity_{h}x{w}_global.npy")));
        let want_p = npy(dir.join(format!("parity_{h}x{w}_policy.npy")));
        let want_v = npy(dir.join(format!("parity_{h}x{w}_value.npy")))[0];
        let (p, v) = backend.forward(&s, &g, h, w, true).unwrap();
        let (tp, tv) = tract.forward_raw(&s, &g, h, w);
        assert!((v - want_v).abs() < 1e-3 && (v - tv).abs() < 1e-3, "{h}x{w}: value {v} vs torch {want_v} / tract {tv}");
        assert!(max_abs_diff(&p, &want_p) < 1e-3, "{h}x{w}: policy vs torch");
        assert!(max_abs_diff(&p, &tp) < 1e-3, "{h}x{w}: policy vs tract");
    }
}

/// A net with no TorchScript export beside it is an error that names the fix, not a tract run.
#[test]
fn missing_export_is_an_error() {
    if !have_gpu() {
        return;
    }
    configure_backend(Some("inproc"), None).unwrap();
    let dir = std::env::temp_dir().join(format!("bbnn_inproc_missing_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let onnx = dir.join("lonely.onnx");
    std::fs::copy(fixtures().join("tiny.onnx"), &onnx).unwrap();
    let err = NnEvaluator::from_path_with_server(&onnx, None).unwrap_err().to_string();
    assert!(err.contains("no TorchScript export") && err.contains("export_ts"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A device that does not exist fails the load with libtorch's reason.
#[test]
fn bad_device_is_an_error() {
    if !have_gpu() {
        return;
    }
    let cfg = InProcConfig::from_env("cuda:57").unwrap();
    let err = InProcBackend::start(&fixtures().join("tiny.ts"), &cfg).unwrap_err();
    assert!(err.contains("out of range") || err.contains("device"), "{err}");
}

/// Nets loaded while another serves must not disturb it. libtorch's stream pool hands out 32
/// streams round-robin and a load draws 27 of them, so with pool streams the fifth net's capture
/// stream was the first net's lane-1 replay stream: the first net's batches, enqueued from its
/// batcher thread during that capture, were recorded into the new graph instead of run, and its
/// queue died (`operation not permitted when stream is capturing`). A long-lived worker loads a
/// net per generation and eval opponent, so it reaches five. The shim's lane and capture streams
/// are its own now.
#[test]
fn loading_more_nets_never_disturbs_one_that_serves() {
    if !have_gpu() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("bbnn_inproc_many_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ts: Vec<PathBuf> = (0..6)
        .map(|k| {
            let p = dir.join(format!("net{k}.ts"));
            std::fs::copy(fixtures().join("tiny.ts"), &p).unwrap();
            p
        })
        .collect();
    let cfg = InProcConfig::from_env("cuda").unwrap();
    let first = InProcBackend::start(&ts[0], &cfg).expect("load the serving net");
    let (cs, cg) = canary_input();
    let reference = first.forward(&cs, &cg, CANARY_H, CANARY_W, false).unwrap().1;
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut later = Vec::new();
    thread::scope(|s| {
        let serving: Vec<_> = (0..16)
            .map(|_| {
                let (first, cs, cg, stop) = (&first, &cs, &cg, &stop);
                s.spawn(move || {
                    let mut n = 0u64;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let v = first
                            .forward(cs, cg, CANARY_H, CANARY_W, false)
                            .expect("serving net failed during a load")
                            .1;
                        assert!((v - reference).abs() < 1e-5, "value {v} vs {reference}");
                        n += 1;
                    }
                    n
                })
            })
            .collect();
        for p in &ts[1..] {
            later.push(InProcBackend::start(p, &cfg).expect("load a later net"));
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in serving {
            assert!(h.join().unwrap() > 0);
        }
    });
    assert!(first.queue.dead().is_none(), "{:?}", first.queue.dead());
    for b in &later {
        assert_eq!(b.forward(&cs, &cg, CANARY_H, CANARY_W, false).unwrap().1, reference);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

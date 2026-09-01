//! The remote-backend contract (plan 024 Stage 1).
//!
//! Two kinds of test live here:
//!
//! * **Always-on**, against a tiny in-process fake server that speaks the
//!   wire protocol and answers with tract. These pin the framing, the
//!   canary interlock and the fallback path on every machine — including
//!   a Mac with no GPU and no Python sidecar — which is the whole point
//!   of debugging the protocol before CUDA enters the picture.
//! * **Env-gated**, against a real `scripts/nn_server.py` named by
//!   `BLOOD_NN_SERVER` (+ `BLOOD_NN_MODEL`). Without those variables they
//!   skip, so `cargo test --workspace` stays green anywhere.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use botbowl_nn::actions::POLICY_CHANNELS;
use botbowl_nn::encode::{GLOBAL_FEATURES, SPATIAL_CHANNELS};
use botbowl_nn::eval::NnEvaluator;
use botbowl_nn::remote::{canary_input, CANARY_H, CANARY_W, FLAG_WANT_POLICY, MAGIC, PROTOCOL_VERSION};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn tmp_socket(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "bbnn_test_{}_{}_{}.sock",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed),
        name
    ))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// ---------------------------------------------------------------------
// a fake server: the protocol, backed by tract
// ---------------------------------------------------------------------

/// Serves the protocol from a tract model. `canary_bias` is added to the
/// canary value only, to simulate a server holding different weights.
fn spawn_fake_server(socket: &Path, canary_bias: f32) -> thread::JoinHandle<()> {
    let listener = UnixListener::bind(socket).expect("bind fake server");
    let onnx = fixtures().join("tiny.onnx");
    thread::spawn(move || {
        let eval = Arc::new(NnEvaluator::from_path(&onnx).expect("fake server model"));
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { break };
            let eval = Arc::clone(&eval);
            thread::spawn(move || {
                // handshake
                let mut head = [0u8; 12];
                if s.read_exact(&mut head).is_err() {
                    return;
                }
                let path_len = u16::from_le_bytes([head[10], head[11]]) as usize;
                let mut path = vec![0u8; path_len];
                s.read_exact(&mut path).unwrap();
                assert_eq!(&head[0..4], &MAGIC);
                assert_eq!(u16::from_le_bytes([head[4], head[5]]), PROTOCOL_VERSION);

                let (cs, cg) = canary_input();
                let (policy, value) = eval.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
                let mut payload = Vec::new();
                payload.extend_from_slice(&(value + canary_bias).to_le_bytes());
                for p in &policy {
                    payload.extend_from_slice(&p.to_le_bytes());
                }
                let mut frame = Vec::new();
                frame.extend_from_slice(&0u16.to_le_bytes()); // status ok
                frame.extend_from_slice(&7u16.to_le_bytes()); // arbitrary model_id
                frame.extend_from_slice(&(CANARY_H as u16).to_le_bytes());
                frame.extend_from_slice(&(CANARY_W as u16).to_le_bytes());
                frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                frame.extend_from_slice(&payload);
                if s.write_all(&frame).is_err() {
                    return;
                }

                // requests
                loop {
                    let mut len_buf = [0u8; 4];
                    if s.read_exact(&mut len_buf).is_err() {
                        return;
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;
                    let mut body = vec![0u8; len];
                    s.read_exact(&mut body).unwrap();
                    let model_id = u16::from_le_bytes([body[0], body[1]]);
                    assert_eq!(model_id, 7, "client must echo the handshake model_id");
                    let flags = u16::from_le_bytes([body[2], body[3]]);
                    let h = u16::from_le_bytes([body[4], body[5]]) as usize;
                    let w = u16::from_le_bytes([body[6], body[7]]) as usize;
                    let n_spatial = SPATIAL_CHANNELS * h * w;
                    let floats: Vec<f32> = body[8..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    assert_eq!(floats.len(), n_spatial + GLOBAL_FEATURES);
                    let (policy, value) = eval.forward_raw(&floats[..n_spatial], &floats[n_spatial..], h, w);
                    let mut payload = value.to_le_bytes().to_vec();
                    if flags & FLAG_WANT_POLICY != 0 {
                        for p in &policy {
                            payload.extend_from_slice(&p.to_le_bytes());
                        }
                    }
                    let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
                    frame.extend_from_slice(&payload);
                    if s.write_all(&frame).is_err() {
                        return;
                    }
                }
            });
        }
    })
}

#[test]
fn remote_matches_tract_on_fixture() {
    let socket = tmp_socket("match");
    let _server = spawn_fake_server(&socket, 0.0);

    let onnx = fixtures().join("tiny.onnx");
    let tract = NnEvaluator::from_path(&onnx).unwrap();
    let remote = NnEvaluator::from_path_with_server(&onnx, Some(&socket)).expect("handshake + canary");

    let (cs, cg) = canary_input();
    let (tp, tv) = tract.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    let (rp, rv) = remote.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    assert!(max_abs_diff(&tp, &rp) < 1e-3, "policy differs");
    assert!((tv - rv).abs() < 1e-3, "value differs: {tv} vs {rv}");
    assert_eq!(rp.len(), POLICY_CHANNELS * CANARY_H * CANARY_W);

    let (served, fell_back) = remote.remote_stats().expect("remote backend");
    assert_eq!(fell_back, 0, "no fallbacks expected against a live server");
    assert!(served >= 1, "the forward should have been served remotely");
    std::fs::remove_file(&socket).ok();
}

#[test]
fn remote_falls_back_to_tract_when_server_absent() {
    let socket = tmp_socket("absent"); // never bound
    let onnx = fixtures().join("tiny.onnx");
    let tract = NnEvaluator::from_path(&onnx).unwrap();
    let remote = NnEvaluator::from_path_with_server(&onnx, Some(&socket)).expect("absent server is not fatal");

    let (cs, cg) = canary_input();
    let (tp, tv) = tract.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    let (rp, rv) = remote.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    // Bit-identical: the fallback *is* tract.
    assert_eq!(tp, rp);
    assert_eq!(tv, rv);
    let (served, fell_back) = remote.remote_stats().expect("remote backend");
    assert_eq!(served, 0);
    assert_eq!(fell_back, 1, "exactly one fallback for one forward");
}

#[test]
fn canary_mismatch_refuses_to_start() {
    let socket = tmp_socket("canary");
    // 0.01 is far above the 1e-3 tolerance but well inside the value
    // range — a plausible "wrong generation of the same net" error.
    let _server = spawn_fake_server(&socket, 0.01);
    let onnx = fixtures().join("tiny.onnx");
    let err = NnEvaluator::from_path_with_server(&onnx, Some(&socket))
        .expect_err("a wrong-net server must be fatal, not a fallback");
    let msg = err.to_string();
    assert!(msg.contains("canary"), "expected a canary error, got: {msg}");
    std::fs::remove_file(&socket).ok();
}

// ---------------------------------------------------------------------
// against a real scripts/nn_server.py
// ---------------------------------------------------------------------

/// `(socket, model)` from the environment, or `None` (test skips).
fn live_server() -> Option<(PathBuf, PathBuf)> {
    let sock = std::env::var("BLOOD_NN_SERVER").ok()?;
    let model = std::env::var("BLOOD_NN_MODEL").ok()?;
    Some((PathBuf::from(sock), PathBuf::from(model)))
}

#[test]
fn live_server_matches_tract() {
    let Some((socket, model)) = live_server() else {
        eprintln!("skipped: set BLOOD_NN_SERVER and BLOOD_NN_MODEL to run against a live sidecar");
        return;
    };
    let tract = NnEvaluator::from_path(&model).unwrap();
    let remote = NnEvaluator::from_path_with_server(&model, Some(&socket)).expect("handshake + canary");
    let (cs, cg) = canary_input();
    let (tp, tv) = tract.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    let (rp, rv) = remote.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
    let dp = max_abs_diff(&tp, &rp);
    assert!(dp < 1e-3, "policy max-abs-diff {dp}");
    assert!((tv - rv).abs() < 1e-3, "value {tv} vs {rv}");
    assert_eq!(remote.remote_stats().unwrap().1, 0, "no fallbacks");
}

/// Batch invariance (plan 024 Stage 2, and the guard on Stage 3's graph
/// padding): a sample's result must not depend on who it was batched with,
/// nor on how far its batch was padded up to a bucket. 24 threads hammer
/// the server with *different* inputs; each compares every response
/// against its own single-threaded reference taken before the storm.
///
/// This is the test that makes the recombination-purity argument true in
/// practice rather than only on paper, and the one that would catch a
/// BatchNorm left in train mode — the single change that would make a
/// padding row leak into a real one.
#[test]
fn live_server_is_batch_invariant() {
    let Some((socket, model)) = live_server() else {
        eprintln!("skipped: set BLOOD_NN_SERVER and BLOOD_NN_MODEL to run against a live sidecar");
        return;
    };
    let remote = Arc::new(NnEvaluator::from_path_with_server(&model, Some(&socket)).expect("handshake"));
    let (cs, cg) = canary_input();

    // Reference: one thread, no concurrency, so every batch is size 1.
    let refs: Arc<Vec<f32>> = Arc::new(
        (0..24)
            .map(|k| {
                let s = perturb(&cs, k);
                remote.forward_raw(&s, &cg, CANARY_H, CANARY_W).1
            })
            .collect(),
    );

    let mut handles = Vec::new();
    for k in 0..24u32 {
        let remote = Arc::clone(&remote);
        let refs = Arc::clone(&refs);
        let cs = cs.clone();
        let cg = cg.clone();
        handles.push(thread::spawn(move || {
            let s = perturb(&cs, k);
            let want = refs[k as usize];
            // Track the largest *deviation from the reference*, not the
            // largest value: a net whose values happen to be negative
            // would make `max(0.0, v)` return 0 no matter what the server
            // said, and the assertion would be vacuous.
            let mut worst = 0.0f32;
            for _ in 0..40 {
                let (_p, v) = remote.forward_raw(&s, &cg, CANARY_H, CANARY_W);
                worst = worst.max((v - want).abs());
            }
            worst
        }));
    }
    for (k, h) in handles.into_iter().enumerate() {
        let d = h.join().unwrap();
        assert!(
            d < 1e-5,
            "sample {k}: max |Δ| {d} vs its batch-1 reference {} — batch composition leaked into the result",
            refs[k]
        );
    }
    assert_eq!(remote.remote_stats().unwrap().1, 0, "no fallbacks");
}

fn perturb(spatial: &[f32], k: u32) -> Vec<f32> {
    let mut s = spatial.to_vec();
    let i = k as usize % s.len();
    s[i] += 0.25 * (k as f32 + 1.0);
    s
}

/// A request whose `model_id` disagrees with the handshake must cost the
/// connection — the guard against a server-side routing bug once the
/// registry grows past one model (Stage 4b).
#[test]
fn live_server_drops_connection_on_model_id_mismatch() {
    let Some((socket, model)) = live_server() else {
        eprintln!("skipped: set BLOOD_NN_SERVER and BLOOD_NN_MODEL to run against a live sidecar");
        return;
    };
    let mut s = UnixStream::connect(&socket).expect("connect");
    let path = model.to_string_lossy();
    let mut frame = Vec::new();
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    frame.extend_from_slice(&(SPATIAL_CHANNELS as u16).to_le_bytes());
    frame.extend_from_slice(&(GLOBAL_FEATURES as u16).to_le_bytes());
    frame.extend_from_slice(&(path.len() as u16).to_le_bytes());
    frame.extend_from_slice(path.as_bytes());
    s.write_all(&frame).unwrap();

    let mut head = [0u8; 12];
    s.read_exact(&mut head).unwrap();
    assert_eq!(u16::from_le_bytes([head[0], head[1]]), 0, "handshake rejected");
    let model_id = u16::from_le_bytes([head[2], head[3]]);
    let len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut payload = vec![0u8; len];
    s.read_exact(&mut payload).unwrap();

    // A request naming a different model than the handshake bound.
    let (cs, cg) = canary_input();
    let body_len = 8 + 4 * cs.len() + 4 * cg.len();
    let mut req = Vec::new();
    req.extend_from_slice(&(body_len as u32).to_le_bytes());
    req.extend_from_slice(&(model_id.wrapping_add(1)).to_le_bytes());
    req.extend_from_slice(&0u16.to_le_bytes());
    req.extend_from_slice(&(CANARY_H as u16).to_le_bytes());
    req.extend_from_slice(&(CANARY_W as u16).to_le_bytes());
    for x in cs.iter().chain(cg.iter()) {
        req.extend_from_slice(&x.to_le_bytes());
    }
    s.write_all(&req).unwrap();

    s.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
    let mut buf = [0u8; 4];
    match s.read(&mut buf) {
        Ok(0) => {}
        Ok(n) => panic!("server answered {n} bytes instead of dropping the connection"),
        Err(e) => panic!("expected clean EOF, got {e}"),
    }
}

/// Model identity is the *weights file*, not the string the client sent.
///
/// A shard launched from the repo root names `models/x.onnx`; one launched
/// anywhere else names the absolute path. They are the same net and must
/// get the same `model_id` and the same batch queue. Keying the registry
/// on the raw string instead fails silently in whichever direction the
/// capacity happens to allow: at `--max-models 1` the second spelling is
/// rejected and that shard runs its whole generation on tract, and above 1
/// the weights load twice and each copy sees half the batch. Both show up
/// only as lost speed, which is why this is a test and not a comment.
#[test]
fn live_server_identifies_a_model_by_its_weights_not_its_path_string() {
    let Some((socket, model)) = live_server() else {
        eprintln!("skipped: set BLOOD_NN_SERVER and BLOOD_NN_MODEL to run against a live sidecar");
        return;
    };
    let canonical = std::fs::canonicalize(&model).expect("model path exists");
    // Same file, three spellings the loop could plausibly produce.
    let detour = canonical.parent().unwrap().join("..").join("models").join(canonical.file_name().unwrap());
    let spellings = [canonical.clone(), detour, PathBuf::from("models").join(canonical.file_name().unwrap())];

    let mut ids = Vec::new();
    for p in &spellings {
        let ev = NnEvaluator::from_path_with_server(&canonical, Some(&socket)).expect("handshake + canary");
        // Force a forward so a rejected handshake shows up as a fallback.
        let (cs, cg) = canary_input();
        ev.forward_raw(&cs, &cg, CANARY_H, CANARY_W);
        let (served, fell_back) = ev.remote_stats().unwrap();
        assert_eq!(fell_back, 0, "spelling {p:?} was not served remotely — registry keyed on the string?");
        assert!(served > 0, "spelling {p:?} served nothing");
        ids.push(handshake_model_id(&socket, p));
    }
    assert!(
        ids.windows(2).all(|w| w[0] == w[1]),
        "same weights got different model_ids {ids:?} — the registry split one net across batch queues"
    );
}

/// Handshake by hand and return the `model_id` the server assigned.
fn handshake_model_id(socket: &Path, model: &Path) -> u16 {
    let mut s = UnixStream::connect(socket).expect("connect");
    let path = model.to_string_lossy();
    let mut frame = Vec::new();
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    frame.extend_from_slice(&(SPATIAL_CHANNELS as u16).to_le_bytes());
    frame.extend_from_slice(&(GLOBAL_FEATURES as u16).to_le_bytes());
    frame.extend_from_slice(&(path.len() as u16).to_le_bytes());
    frame.extend_from_slice(path.as_bytes());
    s.write_all(&frame).unwrap();
    let mut head = [0u8; 12];
    s.read_exact(&mut head).unwrap();
    let status = u16::from_le_bytes([head[0], head[1]]);
    let len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut payload = vec![0u8; len];
    s.read_exact(&mut payload).unwrap();
    assert_eq!(status, 0, "handshake rejected for {model:?}: {}", String::from_utf8_lossy(&payload));
    u16::from_le_bytes([head[2], head[3]])
}

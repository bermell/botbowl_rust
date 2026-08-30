//! The Stage-0 forward counter (plan 024): exactly one tick per
//! `forward_raw`, and the accumulated time is the time actually spent
//! inside it. Its own test binary — the counters are process-wide, so a
//! test sharing a process with other NN tests could not assert an exact
//! delta.

use std::path::PathBuf;

use botbowl_nn::eval::{profile_counters, NnEvaluator};
use botbowl_nn::npy;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn forward_counter_ticks_once_per_forward() {
    let dir = fixtures();
    let spatial = npy::read(dir.join("parity_9x16_spatial.npy")).unwrap().as_f32();
    let global = npy::read(dir.join("parity_9x16_global.npy")).unwrap().as_f32();
    let eval = NnEvaluator::from_path(dir.join("tiny.onnx")).expect("load tiny.onnx");

    let (fw0, ns0) = profile_counters();
    for _ in 0..3 {
        eval.forward_raw(&spatial, &global, 9, 16);
    }
    let (fw1, ns1) = profile_counters();

    assert_eq!(fw1 - fw0, 3, "expected 3 forwards, got {}", fw1 - fw0);
    assert!(ns1 > ns0, "forward time must accumulate");
}

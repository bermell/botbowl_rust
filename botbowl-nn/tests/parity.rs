//! tract-vs-PyTorch parity at two board sizes.
//!
//! Loads the committed `tiny.onnx` fixture and the reference tensors
//! dumped by `train/src/bbnn/fixture.py`, runs tract, and asserts the
//! outputs match PyTorch to `< 1e-4`. Passing at TWO sizes proves the
//! dynamic-axes concretization (Expand/Shape broadcast, ReduceMean, BN
//! fold) works — the main tract op-coverage risk in plan 017.
//!
//! Not `#[ignore]`d: the fixture is committed, so this runs in CI.

use std::path::PathBuf;

use botbowl_nn::eval::NnEvaluator;
use botbowl_nn::npy;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn check_size(eval: &NnEvaluator, h: usize, w: usize) {
    let dir = fixtures();
    let tag = format!("{h}x{w}");
    let spatial = npy::read(dir.join(format!("parity_{tag}_spatial.npy"))).unwrap();
    let global = npy::read(dir.join(format!("parity_{tag}_global.npy"))).unwrap();
    let ref_policy = npy::read(dir.join(format!("parity_{tag}_policy.npy"))).unwrap();
    let ref_value = npy::read(dir.join(format!("parity_{tag}_value.npy"))).unwrap();

    // Sanity on shapes read from the reference tensors.
    assert_eq!(spatial.shape, vec![1, botbowl_nn::encode::SPATIAL_CHANNELS, h, w]);
    assert_eq!(global.shape, vec![1, botbowl_nn::encode::GLOBAL_FEATURES]);

    let (policy, value) = eval.forward_raw(&spatial.as_f32(), &global.as_f32(), h, w);

    let dp = max_abs_diff(&policy, &ref_policy.as_f32());
    assert!(dp < 1e-4, "policy parity failed at {tag}: max-abs-diff {dp}");

    let rv = ref_value.as_f32();
    let dv = (value - rv[0]).abs();
    assert!(dv < 1e-4, "value parity failed at {tag}: |{value} - {}| = {dv}", rv[0]);
}

#[test]
fn tract_matches_pytorch_at_two_board_sizes() {
    let onnx = fixtures().join("tiny.onnx");
    assert!(
        onnx.exists(),
        "missing fixture {} — run `uv run python -m bbnn.fixture`",
        onnx.display()
    );
    let eval = NnEvaluator::from_path(&onnx).expect("load tiny.onnx");
    check_size(&eval, 17, 28);
    check_size(&eval, 9, 16);
}

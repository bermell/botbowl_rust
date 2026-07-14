//! `NnEvaluator` — frozen ONNX inference (tract, pure-Rust CPU) for MCTS
//! priors and leaf value.
//!
//! The model has dynamic `H`/`W`; tract needs them concrete to optimise,
//! so we build one runnable plan per `(H, W)` on first use and cache it
//! (board dims are constant per game → one entry in practice).
//!
//! **Priors** ([`NnEvaluator::priors`]): one forward per expanded node.
//! Legal-action logits are gathered from the spatial policy map exactly
//! as the Python trainer does (positional → one cell; simple → channel
//! spatial max), softmaxed in Rust, then rescaled `× legal.len()` so the
//! mean prior ≈ 1.0 — matching the un-normalised `BASE = 1.0` scale the
//! scripted priors and `PUCT_C` are calibrated against.
//!
//! **Value** ([`NnEvaluator::value_home_i64`]): the mover-centric scalar
//! `v ∈ [-1, 1]`, sign-flipped to Home-centric and rescaled `× 1000` to
//! match `leaf_score`'s TD = ±1000 scale.
//!
//! A frozen, deterministic CPU network is a **pure function of state**, so
//! it satisfies the recombination-purity invariant.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType};
use tract_onnx::prelude::*;

use crate::actions::action_cell;
use crate::encode::{encode, Encoded, GLOBAL_FEATURES, SPATIAL_CHANNELS};
use crate::perspective::mover_for;

type Runnable = TypedRunnableModel<TypedModel>;

/// Frozen ONNX value/policy network. `Send + Sync` — safe to share across
/// MCTS worker threads via `Arc`.
pub struct NnEvaluator {
    /// Parsed model with symbolic `H`/`W`; concretised per board size.
    proto: InferenceModel,
    /// Runnable plans keyed by `(h, w)`.
    cache: Mutex<HashMap<(usize, usize), Arc<Runnable>>>,
}

impl std::fmt::Debug for NnEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NnEvaluator{{..}}")
    }
}

impl NnEvaluator {
    /// Load an ONNX model from `path`. The graph must have inputs
    /// `spatial (N,C,H,W)` + `global (N,F)` and outputs `policy (N,A,H,W)`
    /// + `value (N,1)` (see `train/src/bbnn/export.py`).
    pub fn from_path(path: impl AsRef<Path>) -> TractResult<Self> {
        let proto = tract_onnx::onnx().model_for_path(path)?;
        Ok(NnEvaluator {
            proto,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Get (or build + cache) the runnable plan for a concrete board size.
    fn runnable_for(&self, h: usize, w: usize) -> TractResult<Arc<Runnable>> {
        if let Some(r) = self.cache.lock().unwrap().get(&(h, w)) {
            return Ok(r.clone());
        }
        let plan = self
            .proto
            .clone()
            .with_input_fact(0, f32::fact([1, SPATIAL_CHANNELS, h, w]).into())?
            .with_input_fact(1, f32::fact([1, GLOBAL_FEATURES]).into())?
            .into_optimized()?
            .into_runnable()?;
        let arc = Arc::new(plan);
        self.cache.lock().unwrap().insert((h, w), arc.clone());
        Ok(arc)
    }

    /// Run one forward pass on already-encoded raw tensors. Returns the
    /// flat policy map (`A*H*W`, C-major/row-major) and the raw value
    /// output (`value[0]`). Exposed for the parity test; production paths
    /// go through [`priors`](Self::priors) / [`value_home_i64`](Self::value_home_i64).
    pub fn forward_raw(&self, spatial: &[f32], global: &[f32], h: usize, w: usize) -> (Vec<f32>, f32) {
        let runnable = self.runnable_for(h, w).expect("build runnable");
        let spatial_t = Tensor::from_shape(&[1, SPATIAL_CHANNELS, h, w], spatial).expect("spatial tensor shape");
        let global_t = Tensor::from_shape(&[1, GLOBAL_FEATURES], global).expect("global tensor shape");
        let out = runnable
            .run(tvec!(spatial_t.into(), global_t.into()))
            .expect("nn forward");
        let policy: Vec<f32> = out[0]
            .to_array_view::<f32>()
            .expect("policy f32")
            .iter()
            .copied()
            .collect();
        let value = out[1].as_slice::<f32>().expect("value f32")[0];
        (policy, value)
    }

    fn forward(&self, enc: &Encoded) -> (Vec<f32>, f32) {
        self.forward_raw(&enc.spatial, &enc.global, enc.h, enc.w)
    }

    /// Priors for a node's already-filtered legal actions. One forward
    /// pass; softmax over the gathered per-action logits, rescaled by the
    /// action count so the mean is ≈ 1.0.
    pub fn priors(&self, state: &GameState, actions: &[EngineAction]) -> Vec<f32> {
        if actions.is_empty() {
            return Vec::new();
        }
        let enc = encode(state);
        let (policy, _v) = self.forward(&enc);
        let plane = enc.h * enc.w;
        let mover = enc.mover;
        let dims = state.board_dims;

        let logits: Vec<f32> = actions
            .iter()
            .map(|a| {
                let cell = action_cell(*a, mover, dims);
                if cell.is_simple {
                    // Channel spatial max — mirrors the Python gather.
                    let base = cell.channel * plane;
                    policy[base..base + plane]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                } else {
                    policy[cell.channel * plane + cell.y * enc.w + cell.x]
                }
            })
            .collect();

        softmax_rescaled(&logits)
    }

    /// Home-centric leaf value in `leaf_score`'s scale (`±1000`). The
    /// network emits a mover-centric `v ∈ [-1, 1]`; we clamp, flip sign
    /// for an Away mover, and rescale.
    pub fn value_home_i64(&self, state: &GameState) -> i64 {
        let enc = encode(state);
        let (_policy, v) = self.forward(&enc);
        let v = v.clamp(-1.0, 1.0);
        let home_centric = match mover_for(state) {
            TeamType::Home => v,
            TeamType::Away => -v,
        };
        (home_centric * 1000.0) as i64
    }
}

/// Softmax over `logits`, then rescale by the count so the mean weight is
/// ≈ 1.0 (un-normalised prior scale expected by PUCT).
fn softmax_rescaled(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let n = logits.len() as f32;
    exps.iter().map(|e| e / sum * n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rescaled_mean_is_one() {
        let p = softmax_rescaled(&[1.0, 2.0, 3.0, 0.5]);
        let mean: f32 = p.iter().sum::<f32>() / p.len() as f32;
        assert!((mean - 1.0).abs() < 1e-6, "mean prior should be ~1, got {mean}");
        assert!(p.iter().all(|&x| x > 0.0));
    }
}

//! `NnEvaluator` — frozen ONNX inference (tract, pure-Rust CPU) for MCTS
//! priors and leaf value, with an optional batched remote backend.
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
//! A frozen, deterministic network is a **pure function of state**, so it
//! satisfies the recombination-purity invariant — that holds for the
//! remote backend too (the server is a pure function; see plan 024
//! §Invariants for why batch composition does not leak into the result).
//!
//! **Backends** (plan 024). [`Backend::Tract`] is the default and the
//! reference. [`Backend::Remote`] forwards to `scripts/nn_server.py` over
//! a Unix socket so inference can be batched across the generation
//! loop's shard processes — and always carries a tract fallback, so a
//! dead or wedged server makes a shard slow rather than broken. Only
//! [`NnEvaluator::forward_raw`] and its value-only sibling dispatch;
//! everything above them is backend-agnostic, so no call site in
//! `botbowl-mcts` changes and nothing becomes async.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType};
use tract_onnx::prelude::*;

use crate::actions::action_cell;
use crate::encode::{encode, Encoded, GLOBAL_FEATURES, SPATIAL_CHANNELS};
use crate::perspective::mover_for;
use crate::remote::RemoteClient;

type Runnable = TypedRunnableModel<TypedModel>;

/// Process-wide forward-pass counters (plan 024 Stage 0). Every forward
/// bumps both, whatever the backend — the counters are what turn
/// "generation is inference-bound" from a derivation into a measurement.
/// Cost is one `Instant::now()` pair and two relaxed atomic adds against
/// a ~2.6 ms forward: unmeasurable, so they are always on and
/// `BLOOD_NN_PROFILE` only controls *printing*.
static FORWARDS: AtomicU64 = AtomicU64::new(0);
static FORWARD_NANOS: AtomicU64 = AtomicU64::new(0);

/// `(forwards, total nanoseconds)` spent in NN forwards so far.
pub fn profile_counters() -> (u64, u64) {
    (FORWARDS.load(Ordering::Relaxed), FORWARD_NANOS.load(Ordering::Relaxed))
}

/// Whether `BLOOD_NN_PROFILE` asks for the profile line to be printed.
pub fn profile_enabled() -> bool {
    std::env::var("BLOOD_NN_PROFILE").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// tract-onnx CPU inference: the default backend, the numerics reference
/// (`tests/parity.rs`), and the fallback for every remote failure.
pub struct TractBackend {
    /// Parsed model with symbolic `H`/`W`; concretised per board size.
    proto: InferenceModel,
    /// Runnable plans keyed by `(h, w)`.
    cache: Mutex<HashMap<(usize, usize), Arc<Runnable>>>,
}

impl TractBackend {
    fn from_path(path: impl AsRef<Path>) -> TractResult<Self> {
        Ok(TractBackend {
            proto: tract_onnx::onnx().model_for_path(path)?,
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

    fn forward(&self, spatial: &[f32], global: &[f32], h: usize, w: usize) -> (Vec<f32>, f32) {
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
}

/// Where a forward is actually executed. `Remote` always owns a `Tract`
/// fallback — that is what keeps "a dead server is a slow shard, not a
/// dead shard" true by construction.
pub enum Backend {
    Tract(TractBackend),
    Remote { client: RemoteClient, fallback: TractBackend },
}

/// Frozen value/policy network. `Send + Sync` — safe to share across MCTS
/// worker threads via `Arc`.
pub struct NnEvaluator {
    backend: Backend,
}

thread_local! {
    /// Last forward on this thread, so the two heads of one state cost one
    /// pass instead of two.
    ///
    /// `BBNet` emits `(policy, value)` from a shared trunk, but the search
    /// asks for them separately: `available_actions` calls [`NnEvaluator::priors`]
    /// at expansion (`dynamics.rs:653`) and `score_leaf` calls
    /// [`NnEvaluator::value_home_i64`] when scoring the same node
    /// (`dynamics.rs:1165`). Each used to re-encode the state and run a full
    /// forward, discarding exactly the head the other one needed — so
    /// `--evaluator nn` paid two passes per expanded node where `nn-value`
    /// pays one. Measured cost of that waste: gen05's generate phase ran 425
    /// min against gen04's 292 (+46%) and its eval 396 against 276 (+43%).
    ///
    /// The two calls land back to back on the same state, so a single slot
    /// hits essentially always. The key is the **encoded input compared
    /// exactly**, not a hash: a `GameState` hash collision is precisely the
    /// bug that got `recon_mcts`'s `HashOnly` banned from this repo, and a
    /// collision here would silently return another state's value. ~7 KB per
    /// memcmp against a full network pass is not a close call.
    ///
    /// Thread-local because MCTS workers run concurrently and each descends
    /// its own path; sharing one slot across threads would thrash it.
    static LAST_FORWARD: std::cell::RefCell<Option<CachedForward>> = const { std::cell::RefCell::new(None) };
}

struct CachedForward {
    spatial: Vec<f32>,
    global: Vec<f32>,
    /// `None` when the cached pass was value-only (the remote backend can
    /// skip returning the policy tensor). A later `priors` call on the same
    /// state must then re-forward rather than invent one.
    policy: Option<Vec<f32>>,
    value: f32,
}

impl std::fmt::Debug for NnEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backend {
            Backend::Tract(_) => write!(f, "NnEvaluator{{tract}}"),
            Backend::Remote { client, .. } => write!(f, "NnEvaluator{{remote: {client:?}}}"),
        }
    }
}

impl NnEvaluator {
    /// Load an ONNX model from `path` and run it on the CPU with tract.
    /// The graph must have inputs `spatial (N,C,H,W)` + `global (N,F)`
    /// and outputs `policy (N,A,H,W)` + `value (N,1)` (see
    /// `train/src/bbnn/export.py`).
    pub fn from_path(path: impl AsRef<Path>) -> TractResult<Self> {
        Ok(NnEvaluator {
            backend: Backend::Tract(TractBackend::from_path(path)?),
        })
    }

    /// Like [`from_path`](Self::from_path), but when `server` is `Some`,
    /// route forwards to the batching sidecar listening on that socket,
    /// keeping tract as the fallback.
    ///
    /// `path` does double duty: it is the tract fallback model *and* the
    /// identity sent at handshake, so the server can resolve its own
    /// `.pt` sibling and return a canary this client can check against
    /// its own tract result. Two evaluators in one process (the eval
    /// phase's candidate and champion) each name their own model over the
    /// same socket.
    ///
    /// Errors only if the ONNX fails to load or the canary disagrees. An
    /// unreachable server is *not* an error: it warns and uses tract.
    pub fn from_path_with_server(path: impl AsRef<Path>, server: Option<&Path>) -> TractResult<Self> {
        let path = path.as_ref();
        let tract = TractBackend::from_path(path)?;
        let Some(socket) = server else {
            return Ok(NnEvaluator {
                backend: Backend::Tract(tract),
            });
        };
        // Our own answer on the canary input, via the very ONNX the
        // caller named. Uncounted: this is not a search forward.
        let (cs, cg) = crate::remote::canary_input();
        let (policy, value) = tract.forward(&cs, &cg, crate::remote::CANARY_H, crate::remote::CANARY_W);
        let client = RemoteClient::new(socket, &path.to_string_lossy(), (value, policy))
            .map_err(|e| TractError::msg(e.to_string()))?;
        Ok(NnEvaluator {
            backend: Backend::Remote { client, fallback: tract },
        })
    }

    /// `(served remotely, fell back to tract)`, or `None` on the pure
    /// tract backend.
    pub fn remote_stats(&self) -> Option<(u64, u64)> {
        match &self.backend {
            Backend::Tract(_) => None,
            Backend::Remote { client, .. } => Some((client.served_count(), client.fallback_count())),
        }
    }

    /// Run one forward pass on already-encoded raw tensors. Returns the
    /// flat policy map (`A*H*W`, C-major/row-major) and the raw value
    /// output (`value[0]`). Exposed for the parity test; production paths
    /// go through [`priors`](Self::priors) / [`value_home_i64`](Self::value_home_i64).
    pub fn forward_raw(&self, spatial: &[f32], global: &[f32], h: usize, w: usize) -> (Vec<f32>, f32) {
        self.forward_counted(spatial, global, h, w, true)
    }

    /// The counted dispatch point. `want_policy = false` lets the remote
    /// backend send back 4 bytes instead of 17 KB — which is the whole
    /// response under `nn-value`, the generator's evaluator.
    fn forward_counted(
        &self,
        spatial: &[f32],
        global: &[f32],
        h: usize,
        w: usize,
        want_policy: bool,
    ) -> (Vec<f32>, f32) {
        let t0 = Instant::now();
        let out = match &self.backend {
            Backend::Tract(t) => t.forward(spatial, global, h, w),
            Backend::Remote { client, fallback } => match client.forward(spatial, global, h, w, want_policy) {
                Ok(out) => out,
                Err(_) => fallback.forward(spatial, global, h, w),
            },
        };
        FORWARDS.fetch_add(1, Ordering::Relaxed);
        FORWARD_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    /// One forward per state, shared by both heads. Returns
    /// `(policy, value)`; `policy` is empty when the caller did not ask for
    /// it and none was cached.
    ///
    /// Behaviour-preserving by construction: a hit is returned only when the
    /// encoded input matches the cached one *element for element*, and the
    /// network is deterministic, so callers see exactly the values a fresh
    /// forward would have produced. That also keeps the recombination purity
    /// invariant intact — priors and leaf values stay pure functions of the
    /// state.
    fn forward_memo(&self, enc: &Encoded, want_policy: bool) -> (Vec<f32>, f32) {
        let hit = LAST_FORWARD.with(|slot| {
            let slot = slot.borrow();
            let c = slot.as_ref()?;
            if c.spatial != enc.spatial || c.global != enc.global {
                return None;
            }
            match (want_policy, &c.policy) {
                // Asked for priors but the cached pass was value-only.
                (true, None) => None,
                (true, Some(p)) => Some((p.clone(), c.value)),
                (false, _) => Some((Vec::new(), c.value)),
            }
        });
        if let Some(hit) = hit {
            return hit;
        }

        let (policy, value) = self.forward_counted(&enc.spatial, &enc.global, enc.h, enc.w, want_policy);
        LAST_FORWARD.with(|slot| {
            *slot.borrow_mut() = Some(CachedForward {
                spatial: enc.spatial.clone(),
                global: enc.global.clone(),
                policy: if policy.is_empty() { None } else { Some(policy.clone()) },
                value,
            });
        });
        (policy, value)
    }

    /// Priors for a node's already-filtered legal actions. One forward
    /// pass; softmax over the gathered per-action logits, rescaled by the
    /// action count so the mean is ≈ 1.0.
    pub fn priors(&self, state: &GameState, actions: &[EngineAction]) -> Vec<f32> {
        if actions.is_empty() {
            return Vec::new();
        }
        let enc = encode(state);
        let (policy, _v) = self.forward_memo(&enc, true);
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
        let enc: Encoded = encode(state);
        let (_policy, v) = self.forward_memo(&enc, false);
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

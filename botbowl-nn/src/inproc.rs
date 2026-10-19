//! The in-process libtorch backend (plan 066), behind the `torch` cargo feature.
//!
//! The net runs inside the search process through libtorch — the same library, kernels and CUDA
//! graph capture the Python sidecar uses — fed by the [`BatchQueue`]. What the sidecar did over a
//! socket per request (frame, parse, stage, answer) is a slot write and a futex wake here; what it
//! did per batch (stage into pinned memory, one H2D copy, graph replay, D2H) is the same calls,
//! made from Rust through `csrc/inproc_shim.cpp`.
//!
//! **The module is the sidecar's.** `python -m bbnn.export_ts NET.pt` saves the sidecar's trace of
//! `BBNet.forward_masked` (unoptimised, device-free); at load it is moved to the device and put
//! through `torch::jit::optimize_for_inference`, as the sidecar does in Python. The load sequence
//! then mirrors the sidecar's own (its trace-validation calls, then three warm-up calls and a
//! capture per bucket), so the TorchScript profiling executor and the cuDNN algorithm cache see the
//! same calls in the same order. Plan 066 §4 gates bit parity with the sidecar on that.
//!
//! **Model identity is the resolved `.ts` file and the device** (the sidecar's `Registry` keys on
//! the resolved `.pt`): two evaluators naming one net share one queue, one set of graphs and one
//! set of batchers. See [`resolve_torchscript`] for how an ONNX path finds its `.ts`, and why a
//! missing one is an error rather than a silent tract run.
//!
//! **The canary stays.** At load the committed `parity_9x16` fixture goes through the queue (on
//! the canvas, i.e. the masked path every real request takes) and must agree with tract on the
//! caller's ONNX to `< 1e-3` ([`crate::remote::CANARY_TOL`]), or the load fails.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use crate::actions::POLICY_CHANNELS;
use crate::batch_queue::{BatchQueue, Executor, Geometry, QueueStats, SlotData};
use crate::encode::{GLOBAL_FEATURES, SPATIAL_CHANNELS};
use crate::remote::{canary_input, CANARY_H, CANARY_TOL, CANARY_W};

/// The sidecar's CUDA-graph buckets (`nn_server.py` `BUCKETS`): a batch of 9 runs as a padded 12.
pub const BUCKETS: [usize; 12] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

mod ffi {
    use super::*;
    extern "C" {
        pub fn bbt_last_error() -> *const c_char;
        pub fn bbt_cuda_device_count() -> c_int;
        pub fn bbt_set_cudnn(benchmark: c_int, deterministic: c_int) -> c_int;
        pub fn bbt_set_cudnn_tf32(allow: c_int) -> c_int;
        pub fn bbt_model_load(path: *const c_char, device: *const c_char, optimize: c_int, c: i64, f: i64, a: i64) -> *mut c_void;
        pub fn bbt_model_graph(model: *mut c_void, buf: *mut c_char, cap: i64) -> i64;
        pub fn bbt_model_forward(
            model: *mut c_void,
            input: *const f32,
            n: i64,
            h: i64,
            w: i64,
            out_policy: *mut f32,
            out_value: *mut f32,
        ) -> c_int;
        pub fn bbt_runner_new(model: *mut c_void, bucket: i64, h: i64, w: i64, lane: c_int) -> *mut c_void;
        pub fn bbt_runner_host_in(runner: *mut c_void) -> *mut f32;
        pub fn bbt_runner_host_value(runner: *mut c_void) -> *mut f32;
        pub fn bbt_runner_host_policy(runner: *mut c_void) -> *mut f32;
        pub fn bbt_runner_launch(runner: *mut c_void, n: i64, want_policy: c_int) -> c_int;
        pub fn bbt_runner_wait(runner: *mut c_void, device_ms: *mut f32) -> c_int;
    }
}

fn last_error() -> String {
    // SAFETY: the shim returns a pointer into a thread-local std::string, NUL-terminated.
    unsafe { CStr::from_ptr(ffi::bbt_last_error()) }.to_string_lossy().into_owned()
}

/// cuDNN algorithm selection, process-wide (as `torch.backends.cudnn` is in Python).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudnnMode {
    /// What the sidecar runs: `benchmark = True` (algorithms chosen by timing, per process).
    Production,
    /// `deterministic = True, benchmark = False`: the same algorithms in every process.
    Deterministic,
}

impl CudnnMode {
    /// `BLOOD_NN_DETERMINISTIC=1` selects [`CudnnMode::Deterministic`].
    pub fn from_env() -> Self {
        match std::env::var("BLOOD_NN_DETERMINISTIC") {
            Ok(v) if !v.is_empty() && v != "0" => CudnnMode::Deterministic,
            _ => CudnnMode::Production,
        }
    }
}

/// Set the process-wide cuDNN flags. Call before loading a model.
pub fn set_cudnn(mode: CudnnMode) -> Result<(), String> {
    let (b, d) = match mode {
        CudnnMode::Production => (1, 0),
        CudnnMode::Deterministic => (0, 1),
    };
    // SAFETY: plain FFI call with integer arguments.
    if unsafe { ffi::bbt_set_cudnn(b, d) } != 0 {
        return Err(last_error());
    }
    // `BLOOD_NN_TF32=0|1` sets cuDNN's TF32 permission; unset keeps torch's default (on), which is
    // what the sidecar runs. Irrelevant before Ampere; on an H100 it decides whether every
    // convolution is fp32 or TF32 (plan 066 §8).
    if let Ok(v) = std::env::var("BLOOD_NN_TF32") {
        let allow = !(v.is_empty() || v == "0");
        // SAFETY: plain FFI call.
        if unsafe { ffi::bbt_set_cudnn_tf32(allow as c_int) } != 0 {
            return Err(last_error());
        }
    }
    Ok(())
}

/// Number of CUDA devices libtorch sees.
pub fn cuda_device_count() -> usize {
    // SAFETY: plain FFI call.
    unsafe { ffi::bbt_cuda_device_count() }.max(0) as usize
}

/// A loaded TorchScript module. Freed never: models live for the process (the registry holds
/// them), and tearing CUDA objects down during process exit is a known way to crash at exit.
pub struct TorchModel {
    ptr: *mut c_void,
}
// SAFETY: the shim serialises every model-level call behind the model's own mutex.
unsafe impl Send for TorchModel {}
unsafe impl Sync for TorchModel {}

impl TorchModel {
    /// Load a `bbnn.export_ts` file onto `device` and (if `optimize`) run
    /// `torch::jit::optimize_for_inference` on it, as the sidecar does.
    pub fn load(ts: &Path, device: &str, optimize: bool) -> Result<Self, String> {
        let p = CString::new(ts.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let d = CString::new(device).map_err(|e| e.to_string())?;
        // SAFETY: valid NUL-terminated strings; the shim returns NULL on error.
        let ptr = unsafe {
            ffi::bbt_model_load(
                p.as_ptr(),
                d.as_ptr(),
                optimize as c_int,
                SPATIAL_CHANNELS as i64,
                GLOBAL_FEATURES as i64,
                POLICY_CHANNELS as i64,
            )
        };
        if ptr.is_null() {
            return Err(format!("loading {}: {}", ts.display(), last_error()));
        }
        Ok(TorchModel { ptr })
    }

    /// The (optimised) forward graph as TorchScript IR text.
    pub fn graph(&self) -> String {
        // SAFETY: the first call measures, the second fills a buffer of that size.
        let n = unsafe { ffi::bbt_model_graph(self.ptr, std::ptr::null_mut(), 0) };
        if n < 0 {
            return format!("<error: {}>", last_error());
        }
        let mut buf = vec![0u8; n as usize + 1];
        unsafe { ffi::bbt_model_graph(self.ptr, buf.as_mut_ptr() as *mut c_char, buf.len() as i64) };
        buf.truncate(n as usize);
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// One uncaptured forward of `n` rows already staged on an `h × w` canvas (staging layout:
    /// spatial block, global block, mask block). Returns `(policy n·A·h·w, values n)`.
    pub fn forward_staged(&self, input: &[f32], n: usize, h: usize, w: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let expect = n * (SPATIAL_CHANNELS * h * w + GLOBAL_FEATURES + h * w);
        if input.len() != expect {
            return Err(format!("staged input is {} floats, expected {expect}", input.len()));
        }
        let mut policy = vec![0.0f32; n * POLICY_CHANNELS * h * w];
        let mut value = vec![0.0f32; n];
        // SAFETY: buffers sized as the shim expects.
        let rc = unsafe {
            ffi::bbt_model_forward(
                self.ptr,
                input.as_ptr(),
                n as i64,
                h as i64,
                w as i64,
                policy.as_mut_ptr(),
                value.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(last_error());
        }
        Ok((policy, value))
    }
}

/// One captured CUDA graph for `(bucket, canvas)` on one lane, with its pinned buffers.
pub struct GraphRunner {
    ptr: *mut c_void,
    pub bucket: usize,
    geo: Geometry,
}
// SAFETY: a runner is used by one thread at a time (its lane's batcher, behind a mutex).
unsafe impl Send for GraphRunner {}

impl GraphRunner {
    /// Warm up three times on a side stream and capture, as `GraphRunner.__init__` does.
    pub fn new(model: &TorchModel, geo: Geometry, bucket: usize, lane: usize) -> Result<Self, String> {
        // SAFETY: model pointer is live for the process; the shim returns NULL on error.
        let ptr = unsafe {
            ffi::bbt_runner_new(
                model.ptr,
                bucket as i64,
                geo.canvas_h as i64,
                geo.canvas_w as i64,
                lane as c_int,
            )
        };
        if ptr.is_null() {
            return Err(format!("capturing bucket {bucket}: {}", last_error()));
        }
        Ok(GraphRunner { ptr, bucket, geo })
    }

    fn n_s(&self) -> usize {
        self.bucket * self.geo.spatial_len()
    }

    fn n_g(&self) -> usize {
        self.bucket * self.geo.f
    }

    fn n_m(&self) -> usize {
        self.bucket * self.geo.plane()
    }

    /// The pinned staging buffer (spatial block | global block | mask block).
    pub fn host_in(&mut self) -> &mut [f32] {
        let len = self.n_s() + self.n_g() + self.n_m();
        // SAFETY: the shim allocated exactly this many floats of pinned memory for the runner's
        // lifetime, and `&mut self` makes this the only view.
        unsafe { std::slice::from_raw_parts_mut(ffi::bbt_runner_host_in(self.ptr), len) }
    }

    /// Stage row `i` (the queue's canvas layout) into the pinned buffer.
    pub fn stage_row(&mut self, i: usize, spatial: &[f32], global: &[f32], mask: &[f32]) {
        let (sl, f, pl) = (self.geo.spatial_len(), self.geo.f, self.geo.plane());
        let (n_s, n_g) = (self.n_s(), self.n_g());
        let host = self.host_in();
        host[i * sl..(i + 1) * sl].copy_from_slice(spatial);
        host[n_s + i * f..n_s + (i + 1) * f].copy_from_slice(global);
        host[n_s + n_g + i * pl..n_s + n_g + (i + 1) * pl].copy_from_slice(mask);
    }

    /// Launch the first `n` rows and wait for them. Returns the device time.
    pub fn run(&mut self, n: usize, want_policy: bool) -> Result<Duration, String> {
        // SAFETY: runner is live; launch then wait on the same thread.
        if unsafe { ffi::bbt_runner_launch(self.ptr, n as i64, want_policy as c_int) } != 0 {
            return Err(last_error());
        }
        let mut ms = 0.0f32;
        if unsafe { ffi::bbt_runner_wait(self.ptr, &mut ms) } != 0 {
            return Err(last_error());
        }
        Ok(Duration::from_secs_f64(ms as f64 / 1e3))
    }

    /// The value output (after [`run`](Self::run)).
    pub fn values(&self) -> &[f32] {
        // SAFETY: `bucket` pinned floats owned by the runner.
        unsafe { std::slice::from_raw_parts(ffi::bbt_runner_host_value(self.ptr), self.bucket) }
    }

    /// Canvas policy row `i` (`A × canvas_h × canvas_w`), valid after a `run` that asked for it.
    pub fn policy_row(&self, i: usize) -> &[f32] {
        let row = POLICY_CHANNELS * self.geo.plane();
        // SAFETY: `bucket × row` pinned floats owned by the runner.
        let all = unsafe { std::slice::from_raw_parts(ffi::bbt_runner_host_policy(self.ptr), self.bucket * row) };
        &all[i * row..(i + 1) * row]
    }
}

/// Crop a canvas policy row (`A × ch × cw`) to the request's own `A × h × w`, C-order — the
/// sidecar's `policies[i].reshape(A, *canvas)[:, :h, :w]`.
pub fn crop_policy(row: &[f32], geo: &Geometry, h: usize, w: usize, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(POLICY_CHANNELS * h * w);
    for a in 0..POLICY_CHANNELS {
        let plane = &row[a * geo.plane()..(a + 1) * geo.plane()];
        for y in 0..h {
            out.extend_from_slice(&plane[y * geo.canvas_w..y * geo.canvas_w + w]);
        }
    }
}

/// The smallest bucket that holds `n` (the largest one if none does).
pub fn bucket_for(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(BUCKETS[BUCKETS.len() - 1])
}

/// The [`Executor`] over captured graphs: `runners[lane][bucket index]`.
pub struct TorchExecutor {
    pub model: TorchModel,
    pub geo: Geometry,
    runners: Vec<Vec<Mutex<GraphRunner>>>,
}

impl TorchExecutor {
    /// Capture every bucket on every lane: lane 0's twelve graphs in bucket order (the sidecar's
    /// `prewarm` order), then lane 1's, …
    pub fn capture(model: TorchModel, geo: Geometry, lanes: usize) -> Result<Self, String> {
        let mut runners = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let mut row = Vec::with_capacity(BUCKETS.len());
            for &b in &BUCKETS {
                row.push(Mutex::new(GraphRunner::new(&model, geo, b, lane)?));
            }
            runners.push(row);
        }
        Ok(TorchExecutor { model, geo, runners })
    }

    /// Run a batch directly on `lane` (bypassing the queue): the parity harness's entry point.
    /// `rows` are `(canvas spatial, global, mask, h, w)`; returns `(values, cropped policies)`.
    pub fn run_rows(
        &self,
        lane: usize,
        rows: &[(&[f32], &[f32], &[f32], usize, usize)],
        want_policy: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        let bi = BUCKETS.iter().position(|&b| b >= rows.len()).ok_or("batch larger than the largest bucket")?;
        let mut r = self.runners[lane][bi].lock().unwrap();
        for (i, (s, g, m, _, _)) in rows.iter().enumerate() {
            r.stage_row(i, s, g, m);
        }
        r.run(rows.len(), want_policy)?;
        let values = r.values()[..rows.len()].to_vec();
        let mut policies = Vec::new();
        if want_policy {
            for (i, (_, _, _, h, w)) in rows.iter().enumerate() {
                let mut p = Vec::new();
                crop_policy(r.policy_row(i), &self.geo, *h, *w, &mut p);
                policies.push(p);
            }
        }
        Ok((values, policies))
    }
}

impl Executor for TorchExecutor {
    fn lanes(&self) -> usize {
        self.runners.len()
    }

    fn max_batch(&self) -> usize {
        BUCKETS[BUCKETS.len() - 1]
    }

    fn bucket_for(&self, n: usize) -> usize {
        bucket_for(n)
    }

    fn run(&self, lane: usize, rows: &mut [MutexGuard<'_, SlotData>], want_policy: bool) -> Result<Duration, String> {
        let bi = BUCKETS
            .iter()
            .position(|&b| b >= rows.len())
            .ok_or_else(|| format!("batch of {} exceeds the largest bucket", rows.len()))?;
        let mut r = self.runners[lane][bi].lock().unwrap();
        for (i, row) in rows.iter().enumerate() {
            r.stage_row(i, &row.spatial, &row.global, &row.mask);
        }
        let device = r.run(rows.len(), want_policy)?;
        let values = r.values();
        for (i, row) in rows.iter_mut().enumerate() {
            row.value = values[i];
            if want_policy && row.want_policy {
                let (h, w) = (row.h, row.w);
                let mut out = std::mem::take(&mut row.policy);
                crop_policy(r.policy_row(i), &self.geo, h, w, &mut out);
                row.policy = out;
            }
        }
        Ok(device)
    }
}

/// How the in-process backend is set up. Every field has an env default (see [`InProcConfig::from_env`]).
#[derive(Clone, Debug)]
pub struct InProcConfig {
    /// `cuda`, `cuda:N` (libtorch device string).
    pub device: String,
    /// Canvas `(h, w)`, border included. Default: the compiled board capacity.
    pub canvas: (usize, usize),
    /// Batcher threads (CUDA streams); 2 = the sidecar's depth-2 pipeline.
    pub lanes: usize,
    /// Slot ring size.
    pub slots: usize,
    pub cudnn: CudnnMode,
}

impl InProcConfig {
    /// `device` from the caller; `BLOOD_NN_CANVAS=HxW` (default: the compiled capacity),
    /// `BLOOD_NN_LANES` (2), `BLOOD_NN_SLOTS` (256), `BLOOD_NN_DETERMINISTIC` (off).
    pub fn from_env(device: &str) -> Result<Self, String> {
        use botbowl_engine::core::model::{HEIGHT, WIDTH};
        let canvas = match std::env::var("BLOOD_NN_CANVAS") {
            Ok(v) if !v.is_empty() => {
                let (h, w) = v.split_once('x').ok_or_else(|| format!("BLOOD_NN_CANVAS={v}: want HxW"))?;
                (
                    h.parse().map_err(|_| format!("BLOOD_NN_CANVAS={v}: bad height"))?,
                    w.parse().map_err(|_| format!("BLOOD_NN_CANVAS={v}: bad width"))?,
                )
            }
            _ => (HEIGHT, WIDTH),
        };
        let num = |name: &str, default: usize| -> Result<usize, String> {
            match std::env::var(name) {
                Ok(v) if !v.is_empty() => v.parse().map_err(|_| format!("{name}={v} is not a number")),
                _ => Ok(default),
            }
        };
        Ok(InProcConfig {
            device: device.to_string(),
            canvas,
            lanes: num("BLOOD_NN_LANES", 2)?.max(1),
            slots: num("BLOOD_NN_SLOTS", 256)?.max(1),
            cudnn: CudnnMode::from_env(),
        })
    }

    fn geometry(&self) -> Geometry {
        Geometry {
            c: SPATIAL_CHANNELS,
            f: GLOBAL_FEATURES,
            a: POLICY_CHANNELS,
            canvas_h: self.canvas.0,
            canvas_w: self.canvas.1,
        }
    }
}

/// The sidecar's trace-validation calls (`maybe_trace_masked`: a 9x16 board at batch 1, then the
/// full canvas at batch 5), on the optimised module, so the TorchScript profiling executor has
/// seen the same calls the sidecar's had before its first capture. Inputs are fixed pseudo-random
/// values; the outputs are discarded.
pub fn sidecar_validation_calls(model: &TorchModel, geo: &Geometry) -> Result<(), String> {
    for (b, h, w) in [(1usize, CANARY_H, CANARY_W), (5, geo.canvas_h, geo.canvas_w)] {
        if h > geo.canvas_h || w > geo.canvas_w {
            continue;
        }
        let (ch, cw) = (geo.canvas_h, geo.canvas_w);
        let n_s = b * geo.spatial_len();
        let mut input = vec![0.0f32; n_s + b * geo.f + b * geo.plane()];
        let mut x: u32 = 0x9e37_79b9;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for i in 0..b {
            for c in 0..geo.c {
                for y in 0..h {
                    for xx in 0..w {
                        input[i * geo.spatial_len() + c * ch * cw + y * cw + xx] = rnd();
                    }
                }
            }
            for k in 0..geo.f {
                input[n_s + i * geo.f + k] = rnd();
            }
            for y in 0..h {
                for xx in 0..w {
                    input[n_s + b * geo.f + i * geo.plane() + y * cw + xx] = 1.0;
                }
            }
        }
        model.forward_staged(&input, b, ch, cw)?;
    }
    Ok(())
}

/// One loaded net with its queue: what an `NnEvaluator` holds for the in-process backend.
pub struct InProcBackend {
    pub ts: PathBuf,
    pub device: String,
    pub queue: BatchQueue,
    /// Kept for the parity harness and tests (direct batches, the graph text).
    pub exec: Arc<TorchExecutor>,
}

impl std::fmt::Debug for InProcBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InProc{{{} on {}}}", self.ts.display(), self.device)
    }
}

impl InProcBackend {
    /// Load, validate, capture and start the batchers — no registry, no canary.
    pub fn start(ts: &Path, cfg: &InProcConfig) -> Result<Self, String> {
        // One load/capture at a time per process: two captures interleaving on the device would
        // each see the other's work.
        static LOADING: Mutex<()> = Mutex::new(());
        let _one_at_a_time = LOADING.lock().unwrap_or_else(|e| e.into_inner());
        set_cudnn(cfg.cudnn)?;
        let geo = cfg.geometry();
        let model = TorchModel::load(ts, &cfg.device, true)?;
        sidecar_validation_calls(&model, &geo)?;
        let exec = Arc::new(TorchExecutor::capture(model, geo, cfg.lanes)?);
        let queue = BatchQueue::new(exec.clone() as Arc<dyn Executor>, geo, cfg.slots);
        Ok(InProcBackend {
            ts: ts.to_path_buf(),
            device: cfg.device.clone(),
            queue,
            exec,
        })
    }

    pub fn forward(&self, spatial: &[f32], global: &[f32], h: usize, w: usize, want_policy: bool) -> Result<(Vec<f32>, f32), String> {
        self.queue.forward(spatial, global, h, w, want_policy)
    }

    pub fn stats(&self) -> QueueStats {
        self.queue.stats()
    }

    /// The canary interlock: the committed 9x16 fixture through the queue vs `expected` (tract
    /// on the caller's ONNX), `< CANARY_TOL` on the value and every policy logit.
    pub fn check_canary(&self, expected: &(f32, Vec<f32>), what: &str) -> Result<(), String> {
        let (s, g) = canary_input();
        let (policy, value) = self.forward(&s, &g, CANARY_H, CANARY_W, true)?;
        let (ev, ep) = expected;
        let dv = (value - ev).abs();
        if dv >= CANARY_TOL || !dv.is_finite() {
            return Err(format!("canary value {value} vs tract {ev} (|Δ| = {dv}) for {what}"));
        }
        if policy.len() != ep.len() {
            return Err(format!("canary policy length {} vs tract {} for {what}", policy.len(), ep.len()));
        }
        let dp = policy.iter().zip(ep).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if dp >= CANARY_TOL || !dp.is_finite() {
            return Err(format!("canary policy max-abs-diff {dp} for {what}"));
        }
        Ok(())
    }
}

/// Process-wide: one backend per `(device, resolved .ts)` (the sidecar's `Registry`).
fn registry() -> &'static Mutex<HashMap<(String, PathBuf), Arc<InProcBackend>>> {
    static R: OnceLock<Mutex<HashMap<(String, PathBuf), Arc<InProcBackend>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The shared backend for `onnx`'s TorchScript on `device`, loading it (and checking its canary
/// against `expected_canary`, tract's answer on that ONNX) on first use.
pub fn get_or_load(onnx: &Path, device: &str, expected_canary: &(f32, Vec<f32>)) -> Result<Arc<InProcBackend>, String> {
    let ts = resolve_torchscript(onnx)?;
    let key = (device.to_string(), ts.clone());
    // Held across the load: two evaluators racing for one net must not capture it twice.
    let mut reg = registry().lock().unwrap();
    if let Some(b) = reg.get(&key) {
        // The same weights under another ONNX spelling still gets its own canary check.
        b.check_canary(expected_canary, &onnx.display().to_string())?;
        return Ok(Arc::clone(b));
    }
    let cfg = InProcConfig::from_env(device)?;
    let t0 = std::time::Instant::now();
    let backend = InProcBackend::start(&ts, &cfg)?;
    backend.check_canary(expected_canary, &onnx.display().to_string())?;
    eprintln!(
        "NN_INPROC loaded {} on {} (canvas {}x{}, {} lanes x {} graphs, cudnn {:?}) in {:.1}s (canary ok)",
        ts.display(),
        device,
        cfg.canvas.0,
        cfg.canvas.1,
        cfg.lanes,
        BUCKETS.len(),
        cfg.cudnn,
        t0.elapsed().as_secs_f64()
    );
    let backend = Arc::new(backend);
    reg.insert(key, Arc::clone(&backend));
    if let Some(secs) = std::env::var("BLOOD_NN_INPROC_STATS").ok().and_then(|v| v.parse::<u64>().ok()) {
        if secs > 0 {
            let b = Arc::clone(&backend);
            std::thread::Builder::new()
                .name("nn-inproc-stats".into())
                .spawn(move || loop {
                    std::thread::sleep(Duration::from_secs(secs));
                    eprintln!("NN_INPROC_STATS {}", b.stats().line());
                })
                .ok();
        }
    }
    Ok(backend)
}

/// Where an ONNX path's TorchScript export lives:
///
/// 1. beside it, `<stem>.ts` (the loop exports the three side by side; a worker cache holds
///    `<model-id>.ts` beside `<model-id>.onnx` when the hub shipped it);
/// 2. beside the file it links to, if the ONNX is a symlink;
/// 3. in a directory of `BLOOD_NN_TS_DIRS` (`:`-separated), as the `.ts` of an `.onnx` there with
///    the same bytes — the sidecar's content-addressed resolution, for a cache copy whose `.ts`
///    did not come with it.
///
/// Anything else is an error that names the fix. Never a silent tract run: a worker that could
/// not find the sidecar's `.pt` once ran a whole phase on the CPU with the right weights loaded
/// on an idle GPU, and the only trace was one warning line.
pub fn resolve_torchscript(onnx: &Path) -> Result<PathBuf, String> {
    let sibling = onnx.with_extension("ts");
    if sibling.is_file() {
        return sibling.canonicalize().map_err(|e| e.to_string());
    }
    if let Ok(real) = onnx.canonicalize() {
        let s = real.with_extension("ts");
        if s.is_file() {
            return Ok(s);
        }
    }
    if let Ok(dirs) = std::env::var("BLOOD_NN_TS_DIRS") {
        if let Ok(want) = std::fs::read(onnx) {
            for d in dirs.split(':').filter(|d| !d.is_empty()) {
                let Ok(rd) = std::fs::read_dir(d) else { continue };
                let mut cands: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "onnx"))
                    .collect();
                cands.sort();
                for c in cands {
                    let ts = c.with_extension("ts");
                    if !ts.is_file() || c.metadata().map(|m| m.len()).unwrap_or(0) != want.len() as u64 {
                        continue;
                    }
                    if std::fs::read(&c).is_ok_and(|b| b == want) {
                        return ts.canonicalize().map_err(|e| e.to_string());
                    }
                }
            }
        }
    }
    Err(format!(
        "no TorchScript export for {} (looked for {}{}) — export it with \
         `train/.venv/bin/python -m bbnn.export_ts <the net's .pt> --out {}`",
        onnx.display(),
        sibling.display(),
        if std::env::var_os("BLOOD_NN_TS_DIRS").is_some() {
            " and in BLOOD_NN_TS_DIRS"
        } else {
            ""
        },
        sibling.display()
    ))
}

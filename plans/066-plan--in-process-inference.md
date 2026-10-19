# Plan 066 — in-process batched inference (retire the Python sidecar)

**Status:** phases 1–4 done 2026-10-19, phase 6 written (§8); **the parity gate passes: 0 differing
bits** in tests A and B (120,000 corpus states, every board, every bucket, policy and value-only,
deterministic *and* production cuDNN) and test C (identical corpora modulo two hash-collision
counters). Throughput on this box: +12% GPU samples/s in generation, +14% in mc-label, ~1.9 GB RSS
back (§7.4). **Phase 5 done 2026-10-19 (the user approved): the loop runs its nets in-process by
default (`NN_BACKEND=inproc`), smoke-tested end to end (§9);** relaunch pending. Results in §7.
The loop (`runs/loopmix16x9v9`) paused after gen45's generate for this work (the user: "more
important than generating data"). Registry: exp077.

**The idea (the user, 2026-10-19):** the search computes its input tensor in Rust, writes it into a
preallocated slot, and a thread that owns the GPU notices, runs the net on every slot that is ready,
and puts the result back next to the input. On a Grace Hopper node the CPU and GPU share coherent
memory, so the slot needs no copy at all; on a PCIe card it is pinned memory and one async copy.
**Batching stays**: one board is a tiny fraction of any GPU and a forward is dozens of kernel
launches, so what this removes is the socket, the serialisation, the Python process and its GIL, not
the batch. **Commit to the switch only after bit parity with today's sidecar (§4).**

## 1. Where the time goes today (measured 2026-10-19, gen45 generate, 96x8 + heads, canvas 13x22)

Two sidecars (`scripts/nn_server.py`), 24 search streams each, on the GTX 1060 6GB / i7-7700 (4c/8t):

| per sidecar | value |
|---|---|
| mean batch | 11.8 (histogram peaks at the 8-bucket and above) |
| per batch: stage / GPU forward / wait / post | 650 µs / 7,358 µs / 297 µs / 828 µs |
| GPU time per sample | 625 µs |
| throughput | 1,450 samples/s (2,900 for both) |
| queue time per sample | 4.3 ms |
| CPU | ~50% of a core each; 860 MB RSS each |

GPU utilisation reads 100%; the load average is 4.3 of 8 threads. **On this box generation is bound
by GPU arithmetic** (7.4 ms of kernels for a batch of 12), not by Python: the sidecar's host work
(stage + post, ~1.5 ms per batch) is hidden behind the forward by its two-deep pipeline. So the
in-process backend's gains *here* are modest and specific:

- ~1 core back to the searches (a quarter of the physical cores) and ~1.7 GB of sidecar RSS back
  on a 16 GB box that swapped during gen44's labelling;
- one batch queue for all 48 streams instead of two of 24 (bigger batches per launch);
- no socket round trip in the latency of a mostly idle phase (end of a generation, drives, net
  check, value bench, tools run by hand).

Expected generation throughput on this box: **+0-10%**. That is not the reason to do it.

**On an HPC GPU the picture flips.** An H100 runs this net one to two orders of magnitude faster
than the 1060, so the GPU forward for a batch of 12 drops from 7.4 ms to tens of µs, while the
sidecar's ~1.5 ms of single-threaded Python per batch does not shrink. One sidecar then caps at
roughly 700 batches/s whatever the GPU, and feeding one H100 would take a dozen sidecar processes
fighting over one CUDA device. In-process, the host cost per batch is a memcpy into pinned memory
and one CUDA-graph launch, and the searches (72 Grace cores per H100) fill the batches. The design
below is the same on this box, on Grace Hopper (aarch64 + CUDA) and on AMD (ROCm builds of libtorch
expose the same `at::cuda` API through HIP), so this box is where it is built and proven.

## 2. Design

`botbowl-nn` gets a third backend next to tract (default, numerics reference) and `Remote` (the
sidecar, kept for helper workers and Macs): **`Backend::InProcess`**, behind a cargo feature
(`torch`, off by default) so default builds keep no new dependency and stay buildable on a Mac.

- **Engine: libtorch, the same binaries the sidecar runs.** Link against the `torch` wheel already in
  `train/.venv` (its `include/` and `lib/` ship the C++ API), through a thin C++ shim (`cc`/`cxx`
  crate, a handful of `extern "C"` functions) or `tch` if a release matches torch 2.13. Same ATen,
  same cuDNN, same graph capture as the sidecar: the only route to *bit* parity. ONNX Runtime and
  TensorRT pick their own kernels and cannot match bits; AOTInductor generates Triton kernels
  (different bits, and no Triton on Pascal). They stay possible later as a separately gated speed
  step (§6).
- **The net: TorchScript of `BBNet.forward_masked`**, exported next to the `.onnx`/`.pt` by the
  trainer (`python -m bbnn.export_ts`, and `train_loop.sh` alongside the ONNX export), JIT
  optimisation and fusion off so it runs exactly the eager op sequence. If TorchScript cannot
  reproduce the eager bits, or torch 2.13 has dropped what is needed, the fallback is the forward
  written in ATen calls in the shim, mirroring `model.py` op for op, pinned by the same parity test.
  Either way the hub's content-addressed model cache must carry the new file (the `.pt` lesson: a
  worker that cannot find it falls back silently).
- **The batch queue.** A process-wide registry keyed by (device, resolved weights file), like the
  sidecar's `Registry`. A fixed ring of preallocated slots, each holding one request's canvas-staged
  spatial planes, globals, mask and its output row. A search thread claims a slot, encodes straight
  into it (no intermediate `Vec`), marks it ready and parks (spin briefly, then a futex/condvar).
  **Batcher threads** (one per CUDA stream, two by default = today's two sidecars and its two-deep
  pipeline) take every ready slot up to the largest bucket, copy them into the bucket's pinned input
  buffer, replay the bucket's captured CUDA graph, copy back, write each output row into its slot and
  wake its owner. No max-wait timer: as in the sidecar, the batch is whatever arrived while the
  previous one was on the GPU.
- **Same staging as the sidecar**, so the bits can match: the `--canvas HxW` top-left staging and
  mask, the `BUCKETS` padding (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64), BatchNorm in eval mode,
  `cudnn.benchmark` as the sidecar sets it, the value-only fast path.
- **Callers do not change.** `forward_raw` stays the single dispatch point (`remote.rs`); `priors`,
  `value_home_i64`, the `LAST_FORWARD` memo and the `dynamics.rs` call sites are untouched. Turned on
  by `--nn-backend inproc --nn-device cuda[:N]` (and `BLOOD_NN_BACKEND`), in `dataset`, `eval`,
  `mc-label`, the hub worker and the tools (`value-bench`, `override-audit`, `probe`, net check).
- **The canary stays.** At load, the in-process backend runs the committed `parity_9x16_*` fixture and
  compares it with tract at `< 1e-3`, aborting on mismatch — the same interlock the sidecar
  handshake gives today.
- **Workers.** One worker process per GPU (`CUDA_VISIBLE_DEVICES`), its search threads sharing the
  one in-process queue. On this box: one process, 48 streams, two batcher threads, replacing the two
  sidecars. On a 4-GPU Grace Hopper node: four processes, each pinned to its superchip's cores and
  memory (NUMA).

## 3. Phases

1. **Spike: can libtorch-from-Rust reproduce the sidecar's bits?** Build the shim against the venv's
   torch, load the TorchScript (or ATen) forward, and run fixed batches of encoded corpus states
   (gen44 val shards, every board in the mix, padded to buckets on the 13x22 canvas) through it and
   through the sidecar's own `GraphRunner` path in Python. Report the number of differing bits per
   output, under (a) `cudnn.deterministic = True, benchmark = False` and (b) the production settings.
   Also measure the floor: **two sidecar processes on the same batches** under (b). `cudnn.benchmark`
   picks algorithms by timing in each process, so today's sidecars may not match *each other* bit for
   bit; that floor decides what "parity" can mean in production (§4). Go/no-go for the rest.
2. **The batch queue and batcher threads** (§2), graph capture per bucket, the canary, the feature
   flag, the CLI/env switch, unit tests (slot reuse, wake-ups, batch invariance as the sidecar's
   `live_server_is_batch_invariant` pins it, a dead GPU → clear error, not a hang).
3. **The parity gate (§4).**
4. **Throughput on this box**, loop paused: `scripts/perf_gen_steady.sh`-style steady-state generation
   (games/min, decisions/min, GPU samples/s, mean batch, CPU, RSS) and an MC-label rate, sidecars vs
   in-process, same commit, same net (gen44), same seeds and streams. Then the stream count
   in-process (48 vs 64 vs 96: the RSS it frees may buy more streams).
5. **Switch** (the user's call, once the gate passes): `train_loop.sh` uses the in-process backend for
   local generation, labelling and the per-gen tools; the sidecar stays for helper workers that do
   not build the `torch` feature. Allowlist the commit.
6. **HPC readiness (written, not run here):** aarch64 build against an aarch64 CUDA torch wheel, ROCm
   build notes, one process per GPU, NUMA pinning, Slurm job-array workers dialing the hub, a scaling
   test plan for a pilot allocation.

## 4. Bit parity — what the gate is

Parity is defined on **the same batch** (same rows, same bucket, same canvas): in-process and sidecar
must produce identical output bits. Three tests, all must pass before the switch:

- **A. Kernel parity, deterministic settings.** `cudnn.deterministic = True, benchmark = False` in both.
  ≥100k encoded corpus states over every board in the gen44/gen45 mix, in batches of every bucket
  size, policy and value-only paths: **0 differing bits.**
- **B. Production settings.** The settings the loop runs (`benchmark = True`). If phase 1 shows two
  sidecar processes already agree bit for bit, the bar is 0 differing bits here too. If they do not
  (algorithm choice by timing), the bar is: in-process vs sidecar differs no more often and by no
  more than sidecar vs sidecar, and the in-process backend is offered with a deterministic-algorithm
  mode that makes it reproducible run to run. Either way the user sees the numbers before the switch.
- **C. End to end.** One search stream (batches of 1, no scheduling nondeterminism), fixed seeds,
  deterministic settings: `dataset` drives and an `mc-label` run produce **byte-identical corpora**
  (modulo timing/telemetry fields) through the sidecar and in-process. Also the net check's
  numbers on a fixed sample.

## 5. Risks

- Linking libtorch: C++ ABI flags, `LD_LIBRARY_PATH`/rpath to the venv, binary size, link time.
  Feature-gated, so only the loop's own build pays.
- torch 2.13 may have deprecated TorchScript pieces; the ATen fallback covers it at the cost of a
  second implementation of the forward, pinned by test A.
- CUDA graph capture from C++ (`at::cuda::CUDAGraph`) must mirror the sidecar's memory pool and
  warm-up, or graph replay picks different algorithms than the eager warm-up did.
- Memory: the slot ring is small (≈100 KB per slot); the GPU workspace replaces two sidecars'.
- Builds on this box: ≥60 GB disk free, `-j 4`, the loop's `target/20x11` untouched.

## 6. Out of scope, noted

- **TF32 / bf16 / TensorRT on Hopper:** large speedups, different bits. A numerics change with its
  own gate (value bench, net check, drives), not part of parity.
- **Per-size canvases:** every board is staged on the 13x22 canvas, so a 16x9 board's batch row
  spends ~30% of its GPU time on padding, smaller boards more. Two or three canvas buckets in the
  one in-process queue would win GPU time on this box, where the GPU is the floor. A follow-up.
- **Zero-copy on Grace Hopper:** the graph's input buffer can point at the slots directly once the
  copy shows up in a profile; the first version keeps the explicit pinned copy everywhere.

## 7. Results (2026-10-19, commit 696d1d6 on the plan-066 branch, net gen44 96x8 + heads)

### 7.1 What was built

- **`botbowl-nn` feature `torch`** (off by default; a default build is unchanged and needs no
  libtorch): `csrc/inproc_shim.cpp` (≈300 lines of `extern "C"` over the libtorch C++ API, compiled
  by `build.rs` with `cc`), `src/inproc.rs` (FFI, the graph runners, the executor, the registry,
  the canary, `.ts` resolution), `src/batch_queue.rs` (the slot ring and batcher threads; always
  compiled, unit-tested on the CPU with a fake executor: slot reuse with re-staging, wake-ups under
  more callers than slots, batch-composition invariance, a failing device fails fast and kills the
  queue, a wedged device times out). `tests/inproc.rs` runs on the GPU: canary, tract agreement,
  batch invariance under 24 threads, every head knob on the canvas, missing export and bad device
  are errors. `botbowl-nn/CLAUDE.md` §inproc documents it.
- **Linking against the trainer's wheel works**: `train/.venv`'s torch 2.13.0+cu126 (`include/`,
  `lib/`, new C++ ABI), no `tch`. The wheel lacks two header sets libtorch's CUDA headers include
  (`crt/host_config.h` from nvcc, `nv/target` from cccl): `scripts/fetch_cuda_headers.sh` unpacks
  them from the matching PyPI wheels into `~/.cache/botbowl/cuda_nvcc/12.6` (16 MB, headers only).
  Binaries carry an rpath to the wheel's `torch/lib` and `nvidia/cuda_runtime/lib`, so nothing
  needs `LD_LIBRARY_PATH`. The shim compiles in ~20 s; a `--features torch` `botbowl-ui` adds
  nothing measurable to link time.
- **The net file: `python -m bbnn.export_ts NET.pt` → `NET.ts`** (the sidecar's trace of
  `MaskedNet`, unoptimised, device-free, ≈6 MB for gen44). The shim moves it to the device and runs
  `torch::jit::optimize_for_inference`. The hub ships `<stem>.ts` with the ONNX
  (`ToWorker::Model.torchscript`, **hub protocol v20**), a worker stores `<id>.ts`, and an
  in-process worker counts a cached net only with its `.ts`. Checked end to end on a test hub
  (port 13399): a 4-game `job generate` through `botbowl-worker --nn-backend inproc` loaded the
  shipped `.ts` (canary ok); a net with no `.ts` on the hub failed its task three times with "no
  TorchScript export … export it with bbnn.export_ts" in `hub.log` — never a tract run.
- **Switches**: `--nn-backend inproc [--nn-device cuda:N]` on `dataset`, `eval`, `mc-label`,
  `value-bench`, `override-audit`, `convergence` and `botbowl-worker` (env `BLOOD_NN_BACKEND`,
  `BLOOD_NN_DEVICE`); `hub job …` refuses them (the worker decides). Knobs: `BLOOD_NN_CANVAS`,
  `BLOOD_NN_LANES` (2), `BLOOD_NN_SLOTS` (256), `BLOOD_NN_DETERMINISTIC`, `BLOOD_NN_TF32`,
  `BLOOD_NN_INPROC_STATS`. `nn_server.py --deterministic` is new (default unchanged) for test C.
- Two deviations from §2, both forced: graph capture uses CUDA's *thread-local* capture mode (the
  sidecar's global mode turned another net's concurrent replays into capture errors — the eval
  phase loads its second net while the first serves; the mode only scopes error checking), and a
  process-wide lock serialises loads/captures.
  Review fix (2026-10-19): the lane and capture streams are dedicated CUDA streams, not
  libtorch's 32-stream pool — with pool streams a process's fifth net captured on its first net's
  lane-1 stream and that net's queue died mid-load (`loading_more_nets_never_disturbs_one_that_serves`).
  A reduced test A/B rerun after the fix (12,000 gen44 states, 19 boards, det and prod): 0 differing
  bits in all ten comparisons.

### 7.2 Phase 1 spike: which route reproduces the sidecar

**The sidecar does not run eager `forward_masked`.** With its default `--jit auto` it runs
`torch.jit.trace` + `torch.jit.optimize_for_inference` of `MaskedNet`: frozen, BatchNorm folded
into the convolutions on the GPU, and conv+ReLU fused into `aten::cudnn_convolution_relu` (11 of
them in gen44's graph). So "TorchScript with JIT optimisation off" (§2) is the *wrong* target:

| route (gen44, 120,000 states, cuDNN deterministic) | values differing | policy logits differing | max abs diff |
|---|---|---|---|
| TorchScript trace, **no** `optimize_for_inference` (`--no-optimize`) | 96,042 / 120,000 | 411,565,263 / 455,108,416 | value 3.1e-6, policy 1.8e-4 |
| TorchScript trace + `torch::jit::optimize_for_inference` in C++ (**built**) | **0** | **0** | 0 |

The optimised graphs are the same op for op (146 ops, Python `model.masked.graph` vs the shim's
`graph()`). The ATen fallback was not needed.

### 7.3 The gate (§4)

Harness: `scripts/parity066.sh` (A, B: `inproc_parity` example + `parity066.py`, which drives the
sidecar's own `Registry`/`warm`/`RunnerPool`/`GraphRunner` exactly as `nn_server.main` does) and
`scripts/gatec066.sh` (C). States: 120,000 from `runs/loopmix16x9v9/gen44/shard0-7`, all 22 boards
of the mix (14x8 … 22x13), staged on the 13x22 canvas; plan: 6,959 batches cycling through all 12
buckets, full and partial (padded) batches, policy and value-only alternating (≥290 rows per bucket
and path; 59,984 rows with policy = 455,108,416 logits). Every comparison is over all 455,228,416
output floats, counting differing **bits** (popcount of XOR).

| test | comparison | differing bits | values / logits differing |
|---|---|---|---|
| A (deterministic) | in-process vs sidecar | **0** | 0 / 0 |
| A | sidecar vs sidecar (two processes) | 0 | 0 / 0 |
| A | in-process vs in-process (two processes) | 0 | 0 / 0 |
| A | in-process lane 1 vs lane 0 | 0 | 0 / 0 |
| A | in-process run 2 vs sidecar run 2 | 0 | 0 / 0 |
| B (production, `benchmark=True`) | in-process vs sidecar | **0** | 0 / 0 |
| B | **sidecar vs sidecar: the floor** | **0** | 0 / 0 |
| B | in-process vs in-process | 0 | 0 / 0 |
| B | in-process lane 1 vs lane 0 | 0 | 0 / 0 |
| B | in-process run 2 vs sidecar run 2 | 0 | 0 / 0 |

So on this box the floor is 0 too: cuDNN's timing-based algorithm choice lands on the same
algorithms in every process for this net and these shapes, and B's bar is the strict one, 0 bits —
met. (B ran while the loop's gen45 net check shared the GPU; contention did not move a choice.
Nothing guarantees that on another GPU, which is why `BLOOD_NN_DETERMINISTIC` exists.)

**C (end to end, cuDNN deterministic, one stream, batches of 1).** `botbowl-ui` built with
`torch,deterministic_hash`, run under `fixed_getrandom.so` with the new `FIXRAND_PER_CALL=1` (every
`getrandom` call gets the same bytes, so a process that also loads libtorch hands std the same
`RandomState` keys). `dataset --mode random-start --next-drive --games 3 --seed 77000 --bot-config
cfgs/chance_all_partial_gen.toml --mcts-iters 1000 --evaluator nn` (centred 20x11 size mix; 4
records, 161 samples, 71,257 forwards each) and `mc-label` (2 trajectories, 4 playouts):

| corpus | sidecar vs sidecar (floor) | in-process vs sidecar |
|---|---|---|
| `dataset` | identical (4 / 4 lines) | every sample, visit count, value, tree block, outcome identical; `meta.extra.eq_checks` / `eq_rejects` differ |
| `mc-label` | identical | **identical** |

`eq_checks`/`eq_rejects` count `GameState` equality checks on hash-bucket collisions in the search's
registry — a function of hash-table layout, i.e. of std's per-thread `RandomState`, which a process
with extra threads (the batchers, libtorch's) seeds differently even under the shim. Every
recombination *hit* and probe count matches, as does everything the search decided. I count that
as "modulo telemetry"; the user may want to see it.

### 7.4 Throughput on this box

`scripts/perf066.sh` (→ `perf_gen_steady.sh` with `BACKEND`, `perf_label.sh`), same commit, net
gen44, C1 generation preset, 1000 iterations, the 20x11 loop's centred size mix, canvas 13x22,
production cuDNN, GPU otherwise idle; warm-up 240 s, window 600 s. Sidecar arm = the loop's shape
(48 streams over 2 sidecars, 2 `dataset` processes of 24).

| generation | decisions/min | GPU samples/s | mean batch | CPU (generator + sidecars) | peak RSS (generator + sidecars) | min MemAvailable |
|---|---|---|---|---|---|---|
| sidecars, 48 streams | 137.4 | 2,810 | 11.8 / 11.8 | 294% + 90% | 6,498 + 2,330 MB | 4,715 MB |
| **in-process, 48 streams** | **201.8** | **3,150 (+12%)** | 23.3 | 410% + 0 | **6,952 MB** | 5,612 MB |
| in-process, 64 streams | 160.5 | 3,038 | 30.8 | 412% | 8,762 MB | 3,657 MB |
| in-process, 96 streams | 138.9 | 3,199 | 46.6 | 409% | 11,486 MB | **1,303 MB** |

| mc-label (48 trajectories, 2,138 samples x 8 playouts, `--parallel 48`) | samples/min | CPU | peak RSS | labels |
|---|---|---|---|---|
| sidecar | 532 | 181% + 68% | 252 + 1,184 MB | — |
| in-process | **608 (+14%)** | 250% | 1,257 MB | byte-identical to the sidecar's (48 / 48 lines) |

Reading it:
- **GPU samples/s is the number to trust: +12% generation, +14% labelling.** The GPU is the floor
  (98-99% busy in every arm). One queue for all 48 streams doubles the mean batch (11.8 → 23.3), so
  the bigger, better-occupied buckets do more per kernel launch; the in-process stats line reports
  317 µs of device time per sample against the sidecar's 643 µs, but with two lanes on two CUDA
  streams the batches overlap on the device, so per-batch device times are not additive — use
  samples/s.
- decisions/min (+47% at 48 streams) overstates it: the window is 10 minutes of a few dozen games,
  and the two arms play *different* games (the sidecar arm's two processes take seeds
  `SEED+100000·i` for i = 1, 2; the in-process arm only i = 1), so the decision mix differs. Treat
  it as "at least not slower", and the samples/s as the gain.
- **~1.9 GB of RSS back** (8.8 GB → 7.0 GB for 48 streams) and no Python process; the CPU the
  sidecars used (90%) goes to the searches (the generator's own CPU rises 294% → 410% because it
  does more work per minute).
- **More streams do not help here**: 64 and 96 streams keep the GPU as busy (3,038 / 3,199
  samples/s) while each search waits longer, so decisions/min falls, and 96 streams take the box
  to 1.3 GB MemAvailable. **48 stays the right stream count on this box.** On an HPC GPU, where
  the device is ~50x faster, the stream count is the knob that matters (§8).

### 7.5 Open items

- **Phase 5 (the switch)**: done, see §9.
- The canary still runs the committed *9x16* fixture, now through the masked canvas path (the path
  every request takes). It does not cover a bigger board's padding; test A does.
- Batch composition: the in-process queue, like the sidecar, gives a state bit-identical results
  only within the same bucket; across buckets the difference is ≤ 1e-5 (`tests/inproc.rs`), as
  before.

## 8. Phase 6 — HPC readiness (written, not run)

**What carries over unchanged.** The `.ts` file is device-independent (traced on the CPU,
unoptimised); `optimize_for_inference` and the graph capture run at load on whatever device the
worker has, so one export serves this box, a Grace Hopper node and an x86 A100/H100 node. The hub
already ships it with the ONNX. The queue, the canary and the fallback semantics are the same
everywhere.

**aarch64 + CUDA (Grace Hopper, GH200).**
- PyTorch publishes `manylinux_2_28_aarch64` CUDA wheels (cu126/cu128) with the same `include/` +
  `lib/` layout. Create the venv on the node (`cd train && uv sync`, or any venv with that torch),
  then build with `BOTBOWL_TORCH_DIR=<venv>/lib/python3.X/site-packages/torch`.
- Headers: `scripts/fetch_cuda_headers.sh` picks the aarch64 nvcc/cccl wheels by `uname -m`; or set
  `BOTBOWL_CUDA_INCLUDE` to the site's CUDA module (`$CUDA_HOME/include`), which has everything.
- ABI: check `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` and set
  `BOTBOWL_TORCH_CXX11_ABI` to match (default 1). The wheel's libstdc++ requirement may exceed the
  system compiler's on old RHEL images: load a newer GCC module.
- Hopper is sm_90, in every cu12x wheel. Build the botbowl binaries with the same
  `BOARD_SIZE_*` env as the hub, as for any worker.

**Numerics on Ampere/Hopper — decide before the first HPC corpus.** torch's default
`cudnn.allow_tf32 = True` turns every convolution into TF32 on sm_80+, in the sidecar and the
in-process backend alike. That is a numerics change (≈1e-3 relative on conv outputs) that this box
has never seen; the canary's `< 1e-3` against tract may or may not hold for the logits. Run the
pilot with `BLOOD_NN_TF32=0` (fp32 convolutions, same arithmetic as here up to kernel choice), and
treat TF32 (and bf16) as the separate speed step of §6 with its own gate (value bench, net check,
drives). Bit parity with *this* box is not expected on another GPU (different kernels); parity there
means in-process vs that node's own sidecar, which `scripts/parity066.sh` measures as is.

**ROCm (MI250/MI300).** ROCm wheels of torch (`torch+rocm6.x`) keep the `c10/cuda`, `ATen/cuda`
header names (hipified underneath), but the shim's direct CUDA runtime use (`cudaEventDefault`,
`cudaEventBlockingSync`, `cudaStreamCaptureModeThreadLocal`, `cuda_runtime_api.h`) must become HIP
(`hipify-perl csrc/inproc_shim.cpp`, or a `#ifdef USE_ROCM` block), `build.rs` must link
`torch_hip`/`c10_hip`/`amdhip64` instead of `torch_cuda`/`c10_cuda`/`cudart`, and the device string
stays `cuda` (PyTorch maps it to HIP). Graph capture works on ROCm ≥ 5.3 builds of torch 2.x. Not
attempted; budget a day, and run test A against a ROCm sidecar first.

**One process per GPU, NUMA-pinned.** One `botbowl-worker --nn-backend inproc` per GPU, all its
game threads sharing that GPU's queue:
```sh
# per GH200 superchip k (72 Grace cores, 1 H100); BOARD_SIZE_* as the hub
CUDA_VISIBLE_DEVICES=$k BLOOD_NN_LANES=4 BLOOD_NN_TF32=0 numactl --cpunodebind=$k --membind=$k \
  botbowl-worker --hub ws://HUB:PORT/ws --token-file ~/.config/botbowl/hub.token \
  --nn-backend inproc --nn-device cuda --parallel-games 128 --cache-dir $TMPDIR/models-$k
```
`--parallel-games` sets the offered batch: with the GPU ~50x faster than the 1060 the search, not
the GPU, becomes the floor, so start at ~2 streams per core and raise until the batcher's
`mean_batch` sits near the top bucket (64) or the cores saturate. `BLOOD_NN_LANES` (CUDA streams /
batcher threads) 2 → 4 if the stats line shows the device idle between batches. The largest bucket
is fixed at 64 (the sidecar's `BUCKETS`); an H100 may want 128/256 — a one-line change to
`inproc::BUCKETS`, but a different bucket is a different kernel shape, so re-run test A.

**Slurm.** A job array, one task per GPU, each task one worker dialing the hub on the training box
(or a hub on a login node: it is CPU-only and stateless across restarts):
```sh
#SBATCH --array=0-15 --gpus-per-task=1 --cpus-per-task=72 --mem=0 --time=12:00:00
BLOOD_NN_TF32=0 srun --cpu-bind=none numactl --cpunodebind=$SLURM_LOCALID --membind=$SLURM_LOCALID \
     botbowl-worker --hub ws://$HUB/ws --nn-backend inproc --parallel-games 128 --name $SLURMD_NODENAME-$SLURM_LOCALID
```
The worker reconnects on its own, so preemption and requeue need nothing; the hub's
`--worker-timeout` requeues a dead node's games. Outbound websocket from compute nodes to the hub
must be allowed (or tunnel through the login node). Model files arrive over the socket (≈12 MB per
net with its `.ts`), cached per task in `$TMPDIR`. The commit allowlist applies as for any worker.

**Scaling test for a pilot allocation (≈ one node-day).**
1. One GPU, `dataset --nn-backend inproc` via `scripts/perf_gen_steady.sh` (`BACKEND=inproc`,
   `BOARD=20x11/7`, `CANVAS=13x22`): streams 64 / 128 / 256 × lanes 2 / 4 → decisions/min,
   samples/s, mean batch, GPU util, CPU. Find the knee; record the host cost per batch from the
   stats line (`fwd` vs wall).
2. Same with `BLOOD_NN_TF32=1` — the speed of the numerics step, for the §6 decision.
3. Four workers on one node (one per GPU): linear in GPUs? (NUMA pinning on vs off.)
4. 2–4 nodes against one hub: hub CPU, websocket throughput (trajectories are zstd'd, ~20x), and
   whether the hub's writer keeps up; `job label` items the same way.
5. Before any HPC corpus feeds training: test A/B in-process vs a sidecar *on that node*, the value
   bench and the net check of the generator net on that node's numerics.

## 9. Phase 5 — the loop on the in-process backend (2026-10-19)

**What changed** (`NN_BACKEND`, default `inproc`; `sidecar` = everything exactly as before):

| script | under `NN_BACKEND=inproc` |
|---|---|
| `train_loop.sh` | builds `botbowl-ui`/`botbowl-nn`/`botbowl-worker` with `--features botbowl-ui/torch,botbowl-worker/torch` into the loop's target dir (`target/20x11`), the hub separately without it (it runs no net and so does not link libtorch); fetches the CUDA headers (`scripts/fetch_cuda_headers.sh`) if `~/.cache/botbowl/cuda_nvcc` has none. `NN_SERVER` is forced off, so no sidecar starts, and `worker_start` gives the one local worker every stream of the phase plus `--nn-backend inproc` (`GEN_SIDECARS`/`LABEL_SIDECARS` ignored: 48 generation / 48 label streams in one queue, 2 lanes). `ensure_ts` exports `NET.ts` (`bbnn.export_ts --canvas $NN_CANVAS`, from `NET.pt`; skipped when the `.ts` is newer than the `.pt`; a missing `.pt` is fatal) at launch for the init champion, the champion, the drive and origin references and the anchor (when used), right after each training for the new net, and before every phase or job that loads a net (generate, mc-label, net check, value benches, eval jobs incl. P1, shadow). The hub reads the `.ts` when a job names the net and ships it to workers (protocol v20). `mc-label` (local path) gets `--nn-backend inproc`. `backend_check` puts a `WARN` on status.md for any `NN_SERVER_FALLBACK` in a phase's log and for an in-process phase whose log never shows `NN_INPROC loaded` (generate, mc-label local and hub, eval worker). `PY` is overridable and `PYTHONPATH` points at the checkout's own `train/src`. |
| `launch_plan065.sh` | `NN_BACKEND=inproc` (overridable); comments. `GEN_SIDECARS=2` stays for the rollback. |
| `net_check.sh` | the whole budget ladder in **one** `override-audit --ladder 64,250,1000,4000 --out b{iters}.jsonl` process (new flag: the rungs run concurrently on threads and share one loaded net, i.e. one CUDA context and one GPU queue — four processes would hold four contexts); exits 3 on a tract fallback or no in-process load. |
| `value_bench.sh` | one `value-bench --nn-backend inproc` process per net, in turn; exits 3 on a fallback / no load. |

**GPU memory policy: per-phase processes, no eviction code.** The registry never evicts, but every
process that loads nets is short-lived: the loop already restarts its local worker every phase (it
then holds that phase's nets: the generator, plus the drive reference while the previous
generation's overlapped eval runs), the net check is one process per generation, and each value
bench net is its own process. Measured in the smoke test: worker with one net 1,022 MiB, with two
(gen01 + drive reference) 1,564 MiB; net-check process 1,022 MiB; whole card peak 1,811 MiB (the
label worker and the background net check side by side). The heaviest real overlap is the trainer
plus the background net check (≈1 GB, as the net check's sidecar was) — the same as before.

**Smoke test** (`scripts/loop_smoke066.sh runs/exp077/loop_smoke`, commit e634dfd, one generation of
the 20x11 loop's settings shrunk: 8x2 games at 200 iterations, 2 MC playouts, 1 epoch, drives capped
at 8 per set, net check ladder 64/250 at 4 overrides, both value benches, own hub on port 13398,
own run and models dirs with the live nets symlinked): `train_loop.sh` exit 0 in 6 min; every
phase ran in-process:

| phase | process | evidence |
|---|---|---|
| launch | — | `exported bbnet_mix16x9v9_gen44.ts`, `bbnet_mix16x9d1k_gen04_v9.ts`; "in-process inference … no sidecars" |
| generate (29 records) | local worker x48 | `NN_INPROC loaded …/worker-cache/<id>.ts … (1 net)`, 1,022 MiB GPU |
| net check (background) | one `botbowl-ui override-audit --ladder` | `NN_INPROC loaded …gen44.ts`; MONOTONE line on status.md |
| mc-label via the hub | local worker x48 | `NN_INPROC loaded` (1 net), 1,022 MiB |
| train, absorption | python | `exported bbnet_smoke_gen01.ts` right after training |
| value bench, value bench 2 | `value-bench` per net | `NN_INPROC loaded` per net; gen44 RMS 0.1784 / 0.1863 — exactly the live loop's sidecar numbers for gen44 |
| eval (drives vs d1k gen04, in `eval_finish`) | local worker | two loads (gen01, then the reference: "2 net(s) loaded"), 1,564 MiB |

No `NN_SERVER_FALLBACK` anywhere, no backend `WARN`, and 0 sampler ticks with an `nn_server.py`
process. Peak GPU 1,811 MiB of 6,144; lowest MemAvailable 10.3 GB; largest process RSS 2.4 GB (the
worker). (The value benches first failed in the smoke because the benchmark rows name their corpus
by a path relative to the main checkout and the smoke ran from the worktree; rerun with a `runs`
link there they passed. The real loop runs from the main checkout.)

**Parity re-run from the loop's own build** (`target/20x11` built by `train_loop.sh`, the `.ts` the
loop exported; `MAX_STATES=12000`, 19 boards, 696 batches): 0 differing bits in all ten comparisons
(det and prod: in-process vs sidecar, sidecar vs sidecar, in-process vs in-process, lane 1 vs lane
0, run 2 vs run 2).

**Relaunch** (from the main checkout, after the merge and with the STOP file removed):
`nohup scripts/launch_plan065.sh > /dev/null 2>&1 &`. The first launch builds `target/20x11` with
the torch feature (~5 min) and exports `.ts` files beside gen44, the d1k gen04 reference and
g056 gen04 in `models/az_v7`. Helper workers stay on their sidecars or tract; protocol v20 means
they need this commit (as every bump).

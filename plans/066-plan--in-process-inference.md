# Plan 066 — in-process batched inference (retire the Python sidecar)

**Status:** planned 2026-10-19. The loop (`runs/loopmix16x9v9`) pauses after gen45's generate so the
GPU and memory are free for this work (the user: "more important than generating data").
Registry: exp077 (to be added when the parity gate runs).

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

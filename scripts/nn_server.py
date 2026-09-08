#!/usr/bin/env python3
"""Batched inference sidecar for self-play generation (plan 024).

Generation is inference-bound and single-sample: every shard process
issues one `tract` forward at a time, and the GPU sits idle. MCTS leaf
evaluation cannot be batched *within* a search (the next leaf is only
known once the current one is backed up), so the batch has to come from
somewhere else — here, from the loop's independent shard processes.

This server owns one CUDA context and one copy of the weights, accepts a
Unix-domain socket connection per client thread, and batches whatever
requests have arrived. Clients stay synchronous and blocking; nothing in
the Rust search changes. The Rust side is `botbowl-nn/src/remote.rs`,
which also documents the wire protocol.

    scripts/nn_server.py --socket /tmp/bbnn.sock --device cuda \\
        --model models/bbnet_14x7_gen01.onnx

Design notes worth keeping in mind before editing:

* **One thread, one event loop, a two-deep GPU pipeline.** The first
  version ran a Python thread per connection feeding a `queue.Queue`, and a
  batcher thread that did stage → replay → `.cpu()` → send strictly in
  series. Measured on the 14x7 tier (6 parallel games, 64x6 net): the GPU
  needs ~300 us per batch whatever its size up to 4, but the batch cost
  777 us wall and each request queued another 840 us behind the previous
  batch — the GPU idled while Python worked and Python spun (`.cpu()` is a
  busy-wait) while the GPU worked, and the batcher lost the GIL to a dozen
  connection threads every time it woke. This version reads every socket
  from one `selectors` loop (no threads, no queue, no GIL handoffs), and
  keeps one batch *in flight* on the GPU while it stages the next one and
  answers the previous one. Completion is a `blocking=True` CUDA event, so
  the process sleeps instead of spinning: a whole core comes back to the
  games. See `Server.run` for the loop and `GraphRunner` for the
  double-buffered pinned I/O that makes the overlap safe.
* **No max-wait timer.** The obvious "wait 1 ms to collect a batch" makes
  the server *slower than tract* whenever few shards are active, which is
  exactly the end of a generation and the whole eval phase. The pipeline
  batches naturally instead: the next batch is whatever arrived while the
  current one was on the GPU, so batch size grows precisely as fast as
  offered load. `--max-wait-us` exists as a knob and defaults to 0.
* **The canary is a safety interlock, not a smoke test.** Each model's
  handshake returns its result on the committed parity fixture; the
  client compares against its own tract result and refuses to run on a
  mismatch. That is what stops a corpus — or a promotion-gate verdict —
  from being labelled with the wrong network.
* **Model identity is the resolved weights file, not the client's path
  string.** Two spellings of one net must share a `model_id` and a batch
  queue; see `Registry`. `--max-models` (4) bounds the registry and
  nothing is evicted.
* **CUDA graphs are the point, not an optimisation (Stage 3).** This
  model is tiny (0.48 M params, 0.13 GFLOP) and the GPU is never the
  constraint: at batch 1 a traced module costs ~870 us end to end, of
  which ~300 us is the kernels themselves (24 cuDNN winograd launches at
  hopeless occupancy — the fixed cost of a 3x3 conv on a 16x9 board) and
  the rest is ATen dispatch. A captured graph replays the whole tower as
  one launch. Graphs need static shapes, so batches are padded up to a
  bucket. fp16 and channels_last were measured and are *slower* on Pascal.
"""

from __future__ import annotations

import argparse
import errno
import gc
import itertools
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bbnn.model import GLOBAL_FEATURES, POLICY_CHANNELS, SPATIAL_CHANNELS, BBNet  # noqa: E402

MAGIC = b"BBNN"
PROTOCOL_VERSION = 1
FLAG_WANT_POLICY = 1
FIXTURES = REPO / "botbowl-nn" / "tests" / "fixtures"
CANARY_H, CANARY_W = 9, 16
# Batch buckets for CUDA graph capture. A graph is a fixed shape, so a
# batch of 9 runs as a padded 12 and the padding rows are discarded.
# Fine-grained at the bottom because that is where the offered batch
# actually lands (measured mean_batch 1.6–3.4) *and* where a graph saves
# the most; coarse at the top, where the GPU is doing real work and one
# more launch is noise.
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
# How many batches may be on the device at once. Each GraphRunner holds
# this many pinned host slots; see `GraphRunner` for why the two must agree.
DEPTH = 2
# `Server.serve` timing. SPIN_S: how long before a batch's expected
# completion to stop sleeping and start polling the event (CPU against
# latency). JIT_S: how long before the device goes idle the next batch must
# launch (batch size against latency; must cover the ~60 us host cost of a
# launch plus the ~65 us a sleep wakes late). OVERDUE_S: how far past the
# estimate to give up polling and block on the event.
SPIN_S = 120e-6
JIT_S = 250e-6
OVERDUE_S = 2e-3
# Per-connection receive buffer. A request is 8 + 4·(C·h·w + F) bytes —
# 21 KB on the 14x7 tier — and a client has at most one in flight, so one
# `recv_into` normally lands the whole frame.
RECV_BUF = 1 << 17

HANDSHAKES = itertools.count()

STATUS_OK = 0
STATUS_BAD_MAGIC = 1
STATUS_BAD_VERSION = 2
STATUS_BAD_SHAPES = 3
STATUS_LOAD_FAILED = 4
STATUS_NO_CAPACITY = 5


def log(msg: str) -> None:
    print(f"[nn_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------


@dataclass
class Model:
    model_id: int
    path: str
    module: torch.nn.Module
    canary: bytes  # f32 value + f32[A*h*w] policy, little-endian


class Registry:
    """Resolves a client's `--model` string to loaded weights.

    **Identity is the resolved `.pt` file, not the string the client sent.**
    Two clients naming the same weights differently — `models/x.onnx` from
    a shard launched at the repo root and `/abs/path/models/x.onnx` from
    one launched elsewhere — must share one entry, one `model_id` and one
    batch queue. Keying on the raw string instead is silently expensive in
    both directions: at capacity 1 the second spelling is *rejected* and
    that shard falls back to tract for its whole run (one warning line,
    12.5x slower); once the cap is raised it loads the same net twice and
    splits the batch in half, which is invisible except as lost speed.

    Capacity is `--max-models` (default 4), which covers generation (one
    champion) and the eval phase's candidate-vs-champion pair with room
    for a small strength ladder. Samples cannot be batched across
    different weights, so a second model means a second batch key, not a
    bigger batch.

    **Nothing is evicted.** A `model_id` is also the key of that model's
    captured CUDA graphs; exceeding the cap is a loud refusal instead, and
    the fix is to raise the flag.
    """

    def __init__(self, device: str, capacity: int, jit: str):
        self.device = device
        self.capacity = capacity
        self.jit = jit
        self.by_weights: dict[str, Model] = {}

    def get(self, path: str) -> Model:
        pt = resolve_weights(path)
        key = str(pt)
        if key in self.by_weights:
            return self.by_weights[key]
        if len(self.by_weights) >= self.capacity:
            raise RuntimeError(
                f"registry full ({self.capacity}): already serving "
                f"{[m.path for m in self.by_weights.values()]} — raise --max-models"
            )
        model = self._load(pt, path, model_id=len(self.by_weights))
        self.by_weights[key] = model
        return model

    def _load(self, pt: Path, path: str, model_id: int) -> Model:
        t0 = time.perf_counter()
        state = torch.load(pt, map_location="cpu", weights_only=True)
        module = BBNet.from_state_dict(state)  # width/blocks come from the weights
        # `.eval()` is load-bearing, not hygiene: it is what freezes
        # BatchNorm onto its running statistics, and therefore what makes a
        # sample's result independent of the rest of its batch. Batching
        # and graph padding are both unsound without it.
        module.eval().to(self.device)
        module = maybe_trace(module, self.device, self.jit)
        canary = compute_canary(module, self.device)
        log(f"loaded model_id={model_id} {path} → {pt} in {time.perf_counter() - t0:.1f}s")
        return Model(model_id=model_id, path=pt.name, module=module, canary=canary)


def resolve_weights(path: str) -> Path:
    """`models/bbnet_14x7_gen01.onnx` → the absolute `…/bbnet_14x7_gen01.pt`.

    The loop's train phase exports both side by side; the client names the
    ONNX (which is also its tract fallback) and the server consumes the
    trainer's own `.pt`, so there is no third implementation of BBNet and
    no new numerics surface.

    Always returns a fully resolved absolute path, because the result is
    the registry's identity key (see `Registry`) — a relative path, a
    `..`, and a symlink to the same weights must all collapse to one.
    """
    p = Path(path)
    if p.suffix != ".pt":
        p = p.with_suffix(".pt")
    if not p.is_absolute() and not p.exists():
        # A relative path is the client's cwd first, the repo root second.
        p = REPO / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"no weights at {p} (from client model path {path!r})")
    return p


def maybe_trace(module: torch.nn.Module, device: str, jit: str) -> torch.nn.Module:
    """TorchScript-trace the tower, but only if the trace is shape-general.

    `BBNet.forward` reads `spatial.shape[0/2/3]` and feeds them to
    `view`/`expand`. A tracer can bake those into constants, which would
    silently pin the module to one batch size or board size — the exact
    failure this server must not have. So: trace, then check the traced
    module against eager at a *different* batch and a *different* board
    size, and fall back to eager if they disagree.

    Worth it even under CUDA graphs: freezing folds each BatchNorm into its
    conv, and the captured graph of the frozen module replays in ~295 us
    against ~380 us for the eager module's graph (batch 1, GTX 1060).
    """
    if jit == "off":
        return module
    with torch.no_grad():
        ex_s = torch.zeros(2, SPATIAL_CHANNELS, CANARY_H, CANARY_W, device=device)
        ex_g = torch.zeros(2, GLOBAL_FEATURES, device=device)
        try:
            traced = torch.jit.trace(module, (ex_s, ex_g), check_trace=False)
            traced = torch.jit.optimize_for_inference(traced)
        except Exception as e:  # pragma: no cover - depends on torch version
            log(f"jit trace failed ({e}) — using eager")
            return module
        worst = 0.0
        for (b, h, w) in ((1, CANARY_H, CANARY_W), (5, 11, 20)):
            s = torch.randn(b, SPATIAL_CHANNELS, h, w, device=device)
            g = torch.randn(b, GLOBAL_FEATURES, device=device)
            try:
                tp, tv = traced(s, g)
                ep, ev = module(s, g)
            except Exception as e:
                log(f"jit trace not shape-general at {b}x{h}x{w} ({e}) — using eager")
                return module
            dp = (tp - ep).abs().max().item()
            dv = (tv - ev).abs().max().item()
            # The bar is shape-generality, not bit-equality: freezing folds
            # conv+BN and reassociates, which moves the last fp32 digit.
            # A *baked* shape does not look like 1e-5 — it raises, or is
            # wrong by everything.
            if dp > 1e-3 or dv > 1e-3:
                log(f"jit trace disagrees with eager at {b}x{h}x{w} (dp={dp:.2e} dv={dv:.2e}) — using eager")
                return module
            worst = max(worst, dp, dv)
    log(f"jit trace validated shape-general at two batch sizes and two board sizes (max |Δ| vs eager {worst:.2e})")
    return traced


def canary_input() -> tuple[np.ndarray, np.ndarray]:
    spatial = np.load(FIXTURES / f"parity_{CANARY_H}x{CANARY_W}_spatial.npy").astype(np.float32)
    global_ = np.load(FIXTURES / f"parity_{CANARY_H}x{CANARY_W}_global.npy").astype(np.float32)
    return spatial, global_


def compute_canary(module: torch.nn.Module, device: str) -> bytes:
    spatial, global_ = canary_input()
    with torch.no_grad():
        policy, value = module(
            torch.from_numpy(spatial).to(device),
            torch.from_numpy(global_).to(device),
        )
    v = np.float32(value.detach().cpu().numpy().reshape(-1)[0])
    p = policy.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return v.tobytes() + p.tobytes()


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


@dataclass
class Request:
    conn: "Connection | None"
    model: Model
    h: int
    w: int
    want_policy: bool
    spatial: np.ndarray  # (C, h, w)
    global_: np.ndarray  # (F,)
    t_enqueued: float = field(default_factory=time.perf_counter)

    @property
    def key(self):
        return (self.model.model_id, self.h, self.w)


@dataclass
class Launched:
    """A batch that has been handed to the device and not yet answered."""

    batch: list[Request]
    runner: "EagerRunner | GraphRunner"
    slot: "Slot | None"
    want_policy: bool
    t_launch: float
    stage_ns: int
    queue_ns: int
    # Eager results are already on the host when `launch` returns.
    values: np.ndarray | None = None
    policies: np.ndarray | None = None
    # perf_counter at which the server expects the device to be done; set
    # by `Server.launch` from the runner's running estimate.
    eta: float = 0.0


def bucket_list(max_batch: int) -> list[int]:
    """The buckets a server with this `--max-batch` can ever be asked for."""
    bs = [b for b in BUCKETS if b < max_batch]
    bs.append(max_batch)
    return bs


class EagerRunner:
    """Stage-2 behaviour: stack, H2D, call the module, D2H. No padding.

    Still the path for `--device cpu` and for `--graphs off`, and the
    fallback whenever a graph cannot be captured. Synchronous: `launch`
    has the answer in hand when it returns and `wait` is a no-op, so the
    pipeline degrades to the old serial loop rather than to a special case.
    """

    def __init__(self, module, device: str):
        self.module = module
        self.device = device
        self.bucket = 0  # "no padding" — reported in the stats histogram
        self.eta_s = 0.0  # results are in hand when `launch` returns

    def ready(self, launched: Launched) -> bool:
        return True

    def launch(self, batch: list[Request], want_policy: bool, queue_ns: int) -> Launched:
        t0 = time.perf_counter()
        spatial = np.stack([r.spatial for r in batch])
        global_ = np.stack([r.global_ for r in batch])
        with torch.inference_mode():
            s = torch.from_numpy(spatial).to(self.device, non_blocking=True)
            g = torch.from_numpy(global_).to(self.device, non_blocking=True)
            t1 = time.perf_counter()
            policy, value = self.module(s, g)
            # `.cpu()` is the sync point, so `fwd` below is real end-to-end
            # GPU time and not just the launch. Only pay the 17 KB/sample
            # readback when somebody asked for the policy — `nn-value` (the
            # generator) never does.
            values = value.reshape(-1).cpu().numpy().astype(np.float32)
            policies = policy.reshape(len(batch), -1).cpu().numpy().astype(np.float32) if want_policy else None
        return Launched(batch, self, None, want_policy, t1, int((t1 - t0) * 1e9), queue_ns, values, policies)

    def wait(self, launched: Launched) -> tuple[np.ndarray, np.ndarray | None, int]:
        return launched.values, launched.policies, int((time.perf_counter() - launched.t_launch) * 1e9)

    def run(self, batch: list[Request], want_policy: bool):
        """Synchronous convenience for `bench`."""
        launched = self.launch(batch, want_policy, 0)
        values, policies, fwd = self.wait(launched)
        return values, policies, launched.stage_ns, fwd


@dataclass
class Slot:
    """One set of pinned host buffers for a graph bucket — the in-flight
    batch's staging and results. `GraphRunner` keeps two and alternates.

    Inputs live in *one* pinned buffer (`host`), of which `np_s`/`np_g` are
    views: the batch goes up in a single H2D copy. Every separate copy is
    its own ~25 us device-side launch plus ~15 us of ATen dispatch, and at
    batch 1 the whole tower is 300 us, so two input copies plus a sliced
    output copy were a fifth of the device time and half the host time.
    """

    host: torch.Tensor
    host_v: torch.Tensor
    host_p: torch.Tensor
    np_s: np.ndarray
    np_g: np.ndarray
    np_v: np.ndarray
    np_p: np.ndarray
    start: torch.cuda.Event
    done: torch.cuda.Event


class GraphRunner:
    """One CUDA graph captured for a fixed `(module, bucket, h, w)`.

    Padding is safe because the tower is *sample-independent*: convolutions
    and linears act per row, and BatchNorm is in **eval** mode, so it uses
    the frozen running statistics rather than the batch's. That is the
    whole reason a result cannot depend on batch composition, and it is
    what `live_server_is_batch_invariant` pins. If BatchNorm were ever left
    in train mode, padding rows would leak into real ones and that test is
    what would catch it.

    **Pipelining.** `launch` enqueues H2D → replay → D2H on the current
    stream and records an event; nothing here blocks. `wait` sleeps on that
    event (`blocking=True`: a real futex wait, not the CUDA default spin —
    the old `.cpu()` sync burned ~40% of a core doing nothing). The device
    buffers are shared between consecutive launches, which is sound because
    a single stream executes in order: launch k+1's H2D cannot overwrite
    `dev_s` before replay k has read it, and replay k+1 cannot overwrite
    `value` before D2H k has copied it out. The *host* side is not ordered
    by the stream, so each runner owns `DEPTH` `Slot`s of pinned buffers and
    rotates: a slot is reused only `DEPTH` launches later, and `Server.serve`
    never launches with `DEPTH` batches already in flight, so by then the
    server has waited on the slot's previous batch and copied its results
    out. Deepen the pipeline and the slots come with it.
    """

    def __init__(self, module, bucket: int, h: int, w: int, device: str):
        self.bucket, self.h, self.w = bucket, h, w
        self.n_s = bucket * SPATIAL_CHANNELS * h * w
        self.n_g = bucket * GLOBAL_FEATURES
        # One device buffer, two views — so the batch is one H2D copy.
        self.dev = torch.zeros(self.n_s + self.n_g, device=device)
        dev_s = self.dev[: self.n_s].view(bucket, SPATIAL_CHANNELS, h, w)
        dev_g = self.dev[self.n_s :].view(bucket, GLOBAL_FEATURES)
        self.slots = [self._slot() for _ in range(DEPTH)]
        self.next_slot = 0
        # Running estimate of device time per batch (H2D → D2H), which the
        # server uses to sleep until just before the result is due. Seeded
        # high; the first batches correct it.
        self.eta_s = 600e-6
        # Capture must not record cuDNN autotune or lazy allocator work, so
        # warm on a side stream first — this is required, not defensive.
        # (`cudnn.benchmark` must be on before this runs: the heuristic
        # picks a 4 ms algorithm for batch 1 of this net on Pascal, and the
        # graph would bake it in. `main` sets it; `bench` inherits it.)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                module(dev_s, dev_g)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.policy, self.value = module(dev_s, dev_g)
        self.value_flat = self.value.reshape(-1)
        self.policy_flat = self.policy.reshape(bucket, -1)

    def _slot(self) -> Slot:
        b, h, w = self.bucket, self.h, self.w
        host = torch.zeros(self.n_s + self.n_g).pin_memory()
        host_v = torch.zeros(b).pin_memory()
        host_p = torch.zeros(b, POLICY_CHANNELS * h * w).pin_memory()
        return Slot(
            host, host_v, host_p,
            host[: self.n_s].view(b, SPATIAL_CHANNELS, h, w).numpy(),
            host[self.n_s :].view(b, GLOBAL_FEATURES).numpy(),
            host_v.numpy(), host_p.numpy(),
            torch.cuda.Event(enable_timing=True),
            # `blocking=True`: if the server does have to sleep on this event
            # it sleeps in the kernel instead of spinning on the driver.
            torch.cuda.Event(enable_timing=True, blocking=True),
        )

    def launch(self, batch: list[Request], want_policy: bool, queue_ns: int) -> Launched:
        t0 = time.perf_counter()
        slot = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % DEPTH
        n = len(batch)
        for i, r in enumerate(batch):
            slot.np_s[i] = r.spatial
            slot.np_g[i] = r.global_
        slot.start.record()
        self.dev.copy_(slot.host, non_blocking=True)
        t1 = time.perf_counter()
        self.graph.replay()
        # The padding rows are computed (they are in the graph) but never
        # sent. The value vector is `bucket` floats, so copying all of it
        # is cheaper than slicing it; the policy is 17 KB a row, so slice.
        slot.host_v.copy_(self.value_flat, non_blocking=True)
        if want_policy:
            slot.host_p[:n].copy_(self.policy_flat[:n], non_blocking=True)
        slot.done.record()
        return Launched(batch, self, slot, want_policy, t1, int((t1 - t0) * 1e9), queue_ns)

    def ready(self, launched: Launched) -> bool:
        return launched.slot.done.query()

    def wait(self, launched: Launched) -> tuple[np.ndarray, np.ndarray | None, int]:
        slot = launched.slot
        slot.done.synchronize()
        n = len(launched.batch)
        fwd_ns = int(slot.start.elapsed_time(slot.done) * 1e6)  # device-side ms → ns
        self.eta_s += 0.1 * (fwd_ns / 1e9 - self.eta_s)
        return slot.np_v[:n], (slot.np_p[:n] if launched.want_policy else None), fwd_ns

    def run(self, batch: list[Request], want_policy: bool):
        """Synchronous convenience for `bench`."""
        launched = self.launch(batch, want_policy, 0)
        values, policies, fwd = self.wait(launched)
        return values, policies, launched.stage_ns, fwd


class RunnerPool:
    """Chooses how to run a batch: a captured graph if one fits, else eager.

    Keyed `(model_id, h, w, bucket)`. Capture is lazy and happens on the
    serving thread (there is only one), so a graph is always replayed by
    the thread that recorded it; the first batch of an unseen bucket pays
    ~50 ms once. A bucket whose capture fails is remembered as failed and
    falls through to eager forever, so a torch/driver that cannot capture
    degrades to Stage-2 speed rather than to a dead server.
    """

    def __init__(self, device: str, max_batch: int, enabled: bool):
        self.device = device
        self.max_batch = max_batch
        self.enabled = enabled and device == "cuda"
        self.buckets = bucket_list(max_batch)
        self.graphs: dict = {}
        self.eager: dict = {}
        self.failed: set = set()

    def _bucket_for(self, n: int) -> int:
        for b in self.buckets:
            if b >= n:
                return b
        return self.buckets[-1]

    def get(self, model: Model, h: int, w: int, n: int):
        if not self.enabled:
            return self.eager.setdefault(model.model_id, EagerRunner(model.module, self.device))
        bucket = self._bucket_for(n)
        key = (model.model_id, h, w, bucket)
        runner = self.graphs.get(key)
        if runner is not None:
            return runner
        if key in self.failed:
            return self.eager.setdefault(model.model_id, EagerRunner(model.module, self.device))
        try:
            t0 = time.perf_counter()
            runner = GraphRunner(model.module, bucket, h, w, self.device)
            log(
                f"captured graph model_id={model.model_id} {h}x{w} bucket={bucket} "
                f"in {time.perf_counter() - t0:.2f}s "
                f"(vram reserved {torch.cuda.memory_reserved() / 1e6:.0f} MB)"
            )
        except Exception as e:  # pragma: no cover - driver/torch dependent
            log(f"graph capture failed for {h}x{w} bucket={bucket} ({e!r}) — eager for this bucket")
            self.failed.add(key)
            return self.eager.setdefault(model.model_id, EagerRunner(model.module, self.device))
        self.graphs[key] = runner
        return runner

    def prewarm(self, model: Model, sizes: list[tuple[int, int]]) -> None:
        """Capture every `(h, w) × bucket` up front, so no client ever pays
        a capture and the first measured batches are not outliers."""
        if not self.enabled:
            return
        for (h, w) in sizes:
            for b in self.buckets:
                self.get(model, h, w, b)


@dataclass
class Stats:
    """Where a batch's time actually goes.

    `stage` is host work per batch (row copies into pinned memory + the H2D
    enqueue), `fwd` is **device** time from the H2D to the end of the D2H
    (CUDA events, so it is not inflated by a busy host), `wait` is how long
    the loop slept for a result it needed, and `post` is the response
    writes. With the pipeline, wall time per batch is *not* their sum: the
    GPU runs `fwd` of batch k while the host does `stage`/`post` of k±1.
    `queue` is the per-sample time between a frame arriving and its batch
    launching — the number a client actually feels on top of `fwd`.
    """

    batches: int = 0
    samples: int = 0
    padded: int = 0
    forward_ns: int = 0
    stage_ns: int = 0
    wait_ns: int = 0
    post_ns: int = 0
    queue_ns: int = 0
    hist: dict = field(default_factory=dict)
    t_start: float = field(default_factory=time.perf_counter)

    def record(self, n: int, bucket: int, forward_ns: int, stage_ns: int, wait_ns: int, post_ns: int, queue_ns: int) -> None:
        self.batches += 1
        self.samples += n
        self.padded += max(bucket, n) - n
        self.forward_ns += forward_ns
        self.stage_ns += stage_ns
        self.wait_ns += wait_ns
        self.post_ns += post_ns
        self.queue_ns += queue_ns
        self.hist[n] = self.hist.get(n, 0) + 1

    def line(self) -> str:
        if not self.batches:
            return "no batches yet"
        mean_batch = self.samples / self.batches
        us = lambda x: x / self.batches / 1e3  # noqa: E731
        per_sample = self.forward_ns / max(self.samples, 1) / 1e3
        q_us = self.queue_ns / max(self.samples, 1) / 1e3
        top = sorted(self.hist.items())[:8]
        # `pad` is the share of rows the GPU computed and threw away — the
        # price of a fixed-shape graph. It should stay well under 1, or the
        # bucket ladder is too coarse for the offered batch distribution.
        pad = self.padded / max(self.samples, 1)
        elapsed = max(time.perf_counter() - self.t_start, 1e-9)
        return (
            f"batches={self.batches} samples={self.samples} mean_batch={mean_batch:.2f} pad={pad:.2f} "
            f"stage {us(self.stage_ns):.0f}us fwd {us(self.forward_ns):.0f}us wait {us(self.wait_ns):.0f}us "
            f"post {us(self.post_ns):.0f}us per batch; {per_sample:.0f}us GPU/sample queue={q_us:.0f}us/sample "
            f"{self.samples / elapsed:.0f} samples/s hist={top}"
        )


# --------------------------------------------------------------------------
# connections
# --------------------------------------------------------------------------


class Connection:
    """One client socket, parsed incrementally by the event loop.

    Non-blocking. `on_readable` pulls whatever the kernel has into a
    per-connection buffer and peels off complete frames: first the
    handshake, then requests, each of which becomes a `Request` on the
    server's pending list. A client has at most one request outstanding,
    so in practice one `recv_into` delivers one whole frame and the buffer
    is empty again afterwards; the partial-frame path exists for
    correctness, not speed.
    """

    def __init__(self, sock: socket.socket, server: "Server"):
        self.sock = sock
        self.server = server
        self.model: Model | None = None
        self.alive = True
        self.buf = bytearray(RECV_BUF)
        self.view = memoryview(self.buf)
        self.have = 0

    # -- io ---------------------------------------------------------------
    def on_readable(self) -> None:
        try:
            k = self.sock.recv_into(self.view[self.have:], RECV_BUF - self.have)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as e:
            log(f"connection error: {e}")
            self.close()
            return
        if k == 0:
            self.close()
            return
        self.have += k
        consumed = 0
        while self.alive:
            used = self._parse(self.view[consumed:self.have])
            if used == 0:
                break
            consumed += used
        if consumed:
            rest = self.have - consumed
            if rest:
                self.buf[:rest] = self.buf[consumed:self.have]
            self.have = rest
        elif self.have == RECV_BUF:
            log("frame larger than the receive buffer — dropping connection")
            self.close()

    def send(self, data: bytes) -> None:
        """Write a whole response. The client is always blocked reading, and
        the socket buffer is empty (it holds one response at a time), so
        this almost never has to wait; if it does, wait briefly rather than
        drop the frame."""
        if not self.alive:
            return
        mv = memoryview(data)
        try:
            while mv:
                try:
                    n = self.sock.send(mv)
                except BlockingIOError:
                    if not self.server.wait_writable(self.sock):
                        raise OSError(errno.ETIMEDOUT, "send would block")
                    continue
                mv = mv[n:]
        except OSError as e:
            log(f"send failed: {e}")
            self.close()

    def close(self) -> None:
        if self.alive:
            self.alive = False
            self.server.forget(self)
            try:
                self.sock.close()
            except OSError:
                pass

    # -- protocol ---------------------------------------------------------
    def _parse(self, mv: memoryview) -> int:
        """Consume one complete frame from `mv` and return its length, or 0
        if the frame is not all here yet."""
        if self.model is None:
            return self._parse_handshake(mv)
        if len(mv) < 4:
            return 0
        (length,) = struct.unpack_from("<I", mv, 0)
        if len(mv) < 4 + length:
            return 0
        body = mv[4:4 + length]
        if length < 8:
            log(f"request body is {length} B — dropping connection")
            self.close()
            return 0
        model_id, flags, h, w = struct.unpack_from("<HHHH", body, 0)
        if model_id != self.model.model_id:
            # Guards against a server-side routing bug once the registry
            # grows: a request must name the model its connection was
            # bound to, or the connection dies.
            log(f"model_id {model_id} != handshake {self.model.model_id} — dropping connection")
            self.close()
            return 0
        n_spatial = SPATIAL_CHANNELS * h * w
        want = 8 + 4 * n_spatial + 4 * GLOBAL_FEATURES
        if length != want:
            log(f"request is {length} B, expected {want} B for {h}x{w} — dropping connection")
            self.close()
            return 0
        # One copy out of the receive buffer (the buffer is reused for the
        # next frame); the runner copies once more into pinned memory.
        flat = np.frombuffer(bytes(body[8:]), dtype="<f4", count=n_spatial + GLOBAL_FEATURES)
        self.server.enqueue(
            Request(
                conn=self,
                model=self.model,
                h=h,
                w=w,
                want_policy=bool(flags & FLAG_WANT_POLICY),
                spatial=flat[:n_spatial].reshape(SPATIAL_CHANNELS, h, w),
                global_=flat[n_spatial:],
            )
        )
        return 4 + length

    def _parse_handshake(self, mv: memoryview) -> int:
        if len(mv) < 12:
            return 0
        magic, version, c, f, path_len = struct.unpack_from("<4sHHHH", mv, 0)
        if magic != MAGIC:
            self._reject(STATUS_BAD_MAGIC, f"bad magic {bytes(magic)!r}")
            return 0
        if version != PROTOCOL_VERSION:
            self._reject(STATUS_BAD_VERSION, f"version {version} != {PROTOCOL_VERSION}")
            return 0
        if (c, f) != (SPATIAL_CHANNELS, GLOBAL_FEATURES):
            self._reject(
                STATUS_BAD_SHAPES, f"client encodes C={c} F={f}, server model wants {SPATIAL_CHANNELS}/{GLOBAL_FEATURES}"
            )
            return 0
        if len(mv) < 12 + path_len:
            return 0
        path = bytes(mv[12:12 + path_len]).decode("utf-8", "replace")
        try:
            model = self.server.registry.get(path)
        except Exception as e:
            status = STATUS_NO_CAPACITY if "registry full" in str(e) else STATUS_LOAD_FAILED
            self._reject(status, str(e))
            return 0
        self.model = model
        self.send(
            struct.pack("<HHHHI", STATUS_OK, model.model_id, CANARY_H, CANARY_W, len(model.canary)) + model.canary
        )
        # `MctsBot` spawns its worker threads per decision (`thread::scope`
        # in `dynamics.rs`), and connections are thread-local, so a shard
        # opens roughly one connection per decision — a few per second, not
        # one per process. Log the first few and then only every 1000th, or
        # the log is nothing but handshakes.
        n = next(HANDSHAKES)
        if n < 4 or n % 1000 == 0:
            log(f"connection #{n} bound to model_id={model.model_id} ({path})")
        return 12 + path_len

    def _reject(self, status: int, msg: str) -> None:
        log(f"rejecting connection: {msg}")
        payload = msg.encode()
        self.send(struct.pack("<HHHHI", status, 0, 0, 0, len(payload)) + payload)
        self.close()


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


class Server:
    """The event loop: sockets, batching and the two-deep GPU pipeline, all
    on one thread.

    Per iteration: poll every socket (without blocking if there is work in
    hand), launch the pending requests as one batch if the device is idle
    or about to be, else answer the oldest in-flight batch if its result
    has landed, else sleep until one of those becomes true. The GPU
    therefore has the next batch queued just before the current one ends,
    and the host's parsing/staging/responding runs while the device
    computes. A request arriving at a random moment waits for at most one
    batch to clear the device instead of a whole serial
    stage→forward→respond cycle, and requests that arrive while a batch is
    running leave together as the next one.
    """

    def __init__(self, registry: Registry, pool: RunnerPool, stats: Stats, max_batch: int, max_wait_us: int):
        self.registry = registry
        self.pool = pool
        self.stats = stats
        self.max_batch = max_batch
        self.max_wait_us = max_wait_us
        # `select`, not the default `epoll`: `epoll_wait` takes its timeout
        # in whole milliseconds, so the sub-millisecond sleeps in `serve`
        # (sleep until ~120 us before the batch is due) would all round up
        # to 1 ms — measured as a flat 1.36 ms round trip for a lone client.
        # `select` honours microseconds and wakes ~65 us late on this box.
        # A shard holds one connection per search thread, so the fd count
        # stays far below select's 1024.
        self.sel = selectors.SelectSelector()
        self.listener: socket.socket | None = None
        self.conns: set[Connection] = set()
        # FIFO of pending requests; batches are formed per key from it.
        self.pending: list[Request] = []
        self.gpu_free_at = 0.0
        self.stopping = threading.Event()

    # -- connection bookkeeping -------------------------------------------
    def enqueue(self, r: Request) -> None:
        self.pending.append(r)

    def forget(self, conn: Connection) -> None:
        self.conns.discard(conn)
        try:
            self.sel.unregister(conn.sock)
        except (KeyError, ValueError, OSError):
            pass

    def wait_writable(self, sock: socket.socket, timeout: float = 1.0) -> bool:
        with selectors.SelectSelector() as s:
            s.register(sock, selectors.EVENT_WRITE)
            return bool(s.select(timeout))

    def _accept(self) -> None:
        try:
            sock, _ = self.listener.accept()
        except OSError:
            return
        sock.setblocking(False)
        conn = Connection(sock, self)
        self.conns.add(conn)
        self.sel.register(sock, selectors.EVENT_READ, conn)

    def poll(self, timeout: float | None) -> None:
        """Service every readable socket once. `timeout=0` never blocks."""
        try:
            events = self.sel.select(timeout)
        except OSError:
            return
        for key, _mask in events:
            if key.data is None:
                self._accept()
            else:
                key.data.on_readable()

    # -- batching ----------------------------------------------------------
    def take(self) -> list[Request]:
        """Pull the oldest pending request's key out of `pending`, up to
        `max_batch`, leaving other keys in place for the next iteration."""
        if not self.pending:
            return []
        key = self.pending[0].key
        batch: list[Request] = []
        rest: list[Request] = []
        for r in self.pending:
            if r.key == key and len(batch) < self.max_batch:
                batch.append(r)
            else:
                rest.append(r)
        self.pending = rest
        return batch

    def launch(self, batch: list[Request]) -> Launched:
        t0 = time.perf_counter()
        queue_ns = sum(int((t0 - r.t_enqueued) * 1e9) for r in batch)
        first = batch[0]
        want_policy = any(r.want_policy for r in batch)
        runner = self.pool.get(first.model, first.h, first.w, len(batch))
        launched = runner.launch(batch, want_policy, queue_ns)
        # The device runs batches in order, so this one starts when the
        # previous one is expected to finish, or now if the GPU is idle.
        launched.eta = max(launched.t_launch, self.gpu_free_at) + runner.eta_s
        self.gpu_free_at = launched.eta
        return launched

    def finish(self, launched: Launched) -> None:
        t0 = time.perf_counter()
        values, policies, forward_ns = launched.runner.wait(launched)
        t1 = time.perf_counter()
        for i, r in enumerate(launched.batch):
            body = values[i:i + 1].tobytes()
            if r.want_policy:
                body += policies[i].tobytes()
            r.conn.send(struct.pack("<I", len(body)) + body)
        self.stats.record(
            len(launched.batch),
            bucket=launched.runner.bucket,
            forward_ns=forward_ns,
            stage_ns=launched.stage_ns,
            wait_ns=int((t1 - t0) * 1e9),
            post_ns=int((time.perf_counter() - t1) * 1e9),
            queue_ns=launched.queue_ns,
        )

    # -- main loop ---------------------------------------------------------
    def serve(self, path: str) -> None:
        if os.path.exists(path):
            os.unlink(path)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        self.listener.listen(128)
        self.listener.setblocking(False)
        self.sel.register(self.listener, selectors.EVENT_READ, None)

        # Batches on the device, oldest first. At most DEPTH: a GraphRunner
        # has that many pinned slots, and a request only ever waits behind
        # that many batches.
        inflight: list[Launched] = []
        while not self.stopping.is_set():
            # Block only when there is nothing to do at all; otherwise just
            # sweep up what has arrived.
            self.poll(0.0 if (inflight or self.pending) else 0.25)
            if self.pending and self.max_wait_us and len(self.pending) < self.max_batch:
                deadline = time.perf_counter() + self.max_wait_us / 1e6
                while len(self.pending) < self.max_batch and time.perf_counter() < deadline:
                    self.poll(0.0)
            now = time.perf_counter()
            # Launch when the device is idle, or just in time to keep it
            # busy: with a batch in flight, hold the pending requests until
            # it is within JIT_S of finishing, so they leave as one batch
            # rather than a trickle of singles queued behind it. A batch
            # costs the device about the same at 1 as at 4, so that is
            # throughput for free when the GPU is the bottleneck, and at
            # most JIT_S of latency for a lone request when it is not.
            can_launch = len(inflight) < DEPTH and (
                not inflight or len(self.pending) >= self.max_batch or inflight[-1].eta - now < JIT_S
            )
            if self.pending and can_launch:
                batch = self.take()
                try:
                    inflight.append(self.launch(batch))
                except Exception as e:  # pragma: no cover
                    log(f"batch failed: {e!r}")
                    for r in batch:
                        r.conn.close()
                continue
            if not inflight:
                continue
            head = inflight[0]
            if head.runner.ready(head):
                inflight.pop(0)
                self.finish(head)
                continue
            # Nothing to do until either the pending batch may launch or the
            # oldest batch is due. Sleep in `select` until then — a request
            # arriving meanwhile wakes us — and spin on `query()` for the
            # last SPIN_S before a result: the spin buys back the ~65 us it
            # takes to be woken from a real sleep, and bounds what waiting
            # costs in CPU per batch.
            if self.pending and len(inflight) < DEPTH:
                wake_at = inflight[-1].eta - JIT_S
            else:
                wake_at = head.eta - SPIN_S
            if wake_at > now:
                self.poll(wake_at - now)
            elif head.eta - now < -OVERDUE_S:
                # The estimate was badly wrong (or the device stalled): stop
                # spinning and block on the event properly.
                inflight.pop(0)
                self.finish(head)

        for head in inflight:
            self.finish(head)
        try:
            self.listener.close()
        except OSError:
            pass
        for c in list(self.conns):
            c.close()
        if os.path.exists(path):
            os.unlink(path)


def warm(registry: Registry, model: Model, sizes: list[tuple[int, int]], max_batch: int) -> None:
    """Run every `(h, w) × bucket` once before accepting connections, so
    cuDNN autotune and the first allocations are not charged to a client.
    """
    t0 = time.perf_counter()
    buckets = bucket_list(max_batch)
    with torch.no_grad():
        for (h, w) in sizes:
            for b in buckets:
                s = torch.zeros(b, SPATIAL_CHANNELS, h, w, device=registry.device)
                g = torch.zeros(b, GLOBAL_FEATURES, device=registry.device)
                model.module(s, g)
    if registry.device == "cuda":
        torch.cuda.synchronize()
    log(f"warmed {sizes} × {buckets} in {time.perf_counter() - t0:.1f}s")


def bench(registry: Registry, model: Model, sizes: list[tuple[int, int]], max_batch: int, iters: int = 200) -> None:
    """Sweep batch sizes and fit `t(b) = F + g·b` (plan 024 §Throughput model).

    `F`, the fixed per-batch cost, is the number the whole plan turns on:
    at `F = 4.2 ms` a server is a regression at every stream count we can
    afford; at `F = 0.3 ms` it is ~5x.

    **Each iteration ends in a device→host read**, exactly as a served
    batch does. That matters: without it the loop only measures how fast
    launches can be *queued*, which pipelines across iterations and
    understates a batch's real latency by roughly a third. The Stage-2
    fit was taken that way and read `g` about 30% low.

    Reports the eager/traced path and, on CUDA, the captured-graph path
    side by side, since the gap between them is Stage 3's whole claim.
    """
    device = registry.device
    buckets = bucket_list(max_batch)
    for (h, w) in sizes:
        rows = []
        eager = EagerRunner(model.module, device)
        graphs = RunnerPool(device, max_batch, enabled=True)
        for b in buckets:
            reqs = fake_batch(model, b, h, w)
            te = time_runner(eager, reqs, device, iters)
            row = [b, te, float("nan")]
            if device == "cuda":
                tg = time_runner(graphs.get(model, h, w, b), reqs, device, iters)
                row[2] = tg
            rows.append(tuple(row))
            log(
                f"bench {h}x{w} batch {b:3d}: eager {row[1]:7.0f} us/batch "
                f"({row[1] / b:6.1f} us/sample)   graph {row[2]:7.0f} us/batch ({row[2] / b:6.1f} us/sample)"
            )
        bs = np.array([r[0] for r in rows], dtype=np.float64)
        for label, col in (("eager", 1), ("graph", 2)):
            ts = np.array([r[col] for r in rows], dtype=np.float64)
            if np.isnan(ts).any():
                continue
            gg, ff = np.polyfit(bs, ts, 1)
            log(f"fit {h}x{w} {label}: F = {ff:.0f} us/batch, g = {gg:.1f} us/sample  (device={device})")


def fake_batch(model: Model, n: int, h: int, w: int) -> list["Request"]:
    """`n` synthetic requests, so a runner can be timed off the wire."""
    rng = np.random.default_rng(0)
    return [
        Request(
            conn=None,
            model=model,
            h=h,
            w=w,
            want_policy=False,
            spatial=rng.standard_normal((SPATIAL_CHANNELS, h, w), dtype=np.float32),
            global_=rng.standard_normal(GLOBAL_FEATURES, dtype=np.float32),
        )
        for _ in range(n)
    ]


def time_runner(runner, reqs: list["Request"], device: str, iters: int) -> float:
    for _ in range(30):
        runner.run(reqs, False)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run(reqs, False)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def loadgen(sock_path: str, model_path: str, streams: int, seconds: float) -> None:
    """Hammer a running server from `streams` zero-think-time clients.

    This is the server's *ceiling* at a given stream count, with the MCTS
    search taken out of the picture: it answers "if Stage 4 raised the
    number of independent streams to N, what could the server deliver?"
    without needing N shards' worth of CPU to ask the question. Real
    shards spend most of their time searching, so they never offer this
    much load — the point is the shape of the curve, not the absolute.
    """
    spatial, global_ = canary_input()
    body = struct.pack("<HHHH", 0, 0, CANARY_H, CANARY_W) + spatial.tobytes() + global_.tobytes()
    counts = [0] * streams
    stop = threading.Event()

    def client(i: int) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        path = model_path.encode()
        s.sendall(struct.pack("<4sHHHH", MAGIC, PROTOCOL_VERSION, SPATIAL_CHANNELS, GLOBAL_FEATURES, len(path)) + path)
        head = recv_all(s, 12)
        status, model_id, _h, _w, n = struct.unpack("<HHHHI", head)
        payload = recv_all(s, n)
        if status != 0:
            raise RuntimeError(payload.decode())
        req = struct.pack("<I", len(body)) + struct.pack("<H", model_id) + body[2:]
        while not stop.is_set():
            s.sendall(req)
            (rlen,) = struct.unpack("<I", recv_all(s, 4))
            recv_all(s, rlen)
            counts[i] += 1
        s.close()

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(streams)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    dt = time.perf_counter() - t0
    total = sum(counts)
    log(f"loadgen streams={streams}: {total / dt:.0f} forwards/s ({total} in {dt:.1f}s, {dt / total * 1e6:.0f}us/forward/stream)")


def recv_all(s: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("server closed the connection")
        got += k
    return bytes(buf)


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        h, w = tok.split("x")
        out.append((int(h), int(w)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default="/tmp/bbnn.sock", help="Unix socket path to listen on")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--model", default=None, help="preload + warm this model (client paths still resolved on demand)")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument(
        "--max-wait-us",
        type=int,
        default=0,
        help="poll this long for more requests before launching a batch. DEFAULT 0 — a timer makes the "
        "server slower than tract whenever few shards are active; the pipeline batches naturally.",
    )
    ap.add_argument(
        "--max-models",
        type=int,
        default=4,
        help="registry capacity. 1 net for generation, 2 for the eval phase's candidate-vs-champion; "
        "nothing is evicted, so exceeding this is a refusal, not a stall.",
    )
    ap.add_argument("--jit", default="auto", choices=("auto", "off"))
    ap.add_argument(
        "--graphs",
        default="auto",
        choices=("auto", "off"),
        help="CUDA graph capture per batch bucket (Stage 3). DEFAULT auto — it roughly halves batch-1 "
        "latency and cuts F from ~560us to ~210us. `off` reverts to the Stage-2 eager path.",
    )
    ap.add_argument("--warm-sizes", default=f"{CANARY_H}x{CANARY_W}", help="comma-separated HxW to warm")
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument(
        "--switch-interval",
        type=float,
        default=None,
        help="accepted for compatibility; the server is single-threaded and ignores it",
    )
    ap.add_argument("--stats-every", type=float, default=60.0, help="seconds between stats lines (0 = off)")
    ap.add_argument(
        "--bench",
        action="store_true",
        help="sweep batch sizes, fit F and g, and exit instead of serving",
    )
    ap.add_argument(
        "--loadgen",
        type=int,
        default=0,
        help="act as a client instead: N zero-think-time streams against --socket, printing forwards/s",
    )
    ap.add_argument("--loadgen-seconds", type=float, default=10.0)
    args = ap.parse_args()

    if args.loadgen:
        if not args.model:
            log("FATAL --loadgen needs --model (the path the server will resolve)")
            return 1
        loadgen(args.socket, args.model, args.loadgen, args.loadgen_seconds)
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        log("FATAL --device cuda but torch.cuda.is_available() is False")
        return 1
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.benchmark = True

    registry = Registry(args.device, args.max_models, args.jit)
    if args.bench:
        if not args.model:
            log("FATAL --bench needs --model")
            return 1
        if args.device == "cuda":
            log(f"cuda device: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        bench(registry, registry.get(args.model), parse_sizes(args.warm_sizes), args.max_batch)
        return 0

    sizes = parse_sizes(args.warm_sizes)
    pool = RunnerPool(args.device, args.max_batch, enabled=args.graphs == "auto")
    if args.model:
        model = registry.get(args.model)
        warm(registry, model, sizes, args.max_batch)
        # Capture before the socket exists, so the first client never eats a
        # capture storm — and the socket appearing is the readiness signal
        # `train_loop.sh` waits for.
        t0 = time.perf_counter()
        pool.prewarm(model, sizes)
        log(f"prewarmed {len(pool.graphs)} graphs in {time.perf_counter() - t0:.1f}s")
    gc.collect()
    gc.freeze()

    stats = Stats()
    server = Server(registry, pool, stats, args.max_batch, args.max_wait_us)

    def shutdown(signum, _frame):
        log(f"signal {signum} — shutting down. {stats.line()}")
        server.stopping.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.stats_every:
        def ticker():
            while not server.stopping.wait(args.stats_every):
                log(stats.line())

        threading.Thread(target=ticker, daemon=True, name="stats").start()

    dev = args.device
    if dev == "cuda":
        log(f"cuda device: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    log(
        f"listening on {args.socket} device={dev} max_batch={args.max_batch} "
        f"max_wait_us={args.max_wait_us} graphs={len(pool.graphs)}"
    )
    server.serve(args.socket)
    log(f"stopped. {stats.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

* **No max-wait timer.** The obvious "wait 1 ms to collect a batch" makes
  the server *slower than tract* whenever few shards are active, which is
  exactly the end of a generation and the whole eval phase. Greedy
  draining self-regulates instead: batch size grows precisely as fast as
  offered load, because requests accumulate during the previous forward.
  `--max-wait-us` exists as a knob and defaults to 0.
* **The canary is a safety interlock, not a smoke test.** Each model's
  handshake returns its result on the committed parity fixture; the
  client compares against its own tract result and refuses to run on a
  mismatch. That is what stops a corpus — or a promotion-gate verdict —
  from being labelled with the wrong network.
* **Capacity-1 registry (plan 024 Stage 1).** Generation shards use one
  net each. `model_path`/`model_id` are on the wire from day one so the
  multi-model registry the eval phase needs (Stage 4b) is a server-side
  change only, with no protocol bump.
* **CUDA graphs are the point, not an optimisation (Stage 3).** This
  model is tiny (0.48 M params, 0.13 GFLOP) and the GPU is never the
  constraint: at batch 1 a traced module costs ~870 us end to end, of
  which ~70 us is arithmetic. The rest is ATen dispatch and ~40 kernel
  launches — the `F` term the plan's throughput model turns on. A
  captured graph replays the whole tower as one launch, which is why
  `F` falls from ~560 us to ~210 us and batch-1 latency roughly halves.
  Graphs need static shapes, so batches are padded up to a bucket.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import queue
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
# actually lands (measured mean_batch 3.4 at 8 shards) *and* where a
# graph saves the most; coarse at the top, where the GPU is doing real
# work and one more launch is noise.
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

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

    Capacity is 1 for now (Stage 1): every generation shard names the same
    champion, so one entry serves them all, and a second *different* model
    is a configuration error worth refusing loudly rather than a case to
    silently support. Stage 4b raises the cap and adds LRU eviction for
    the eval phase's candidate-vs-champion pair.
    """

    def __init__(self, device: str, capacity: int, jit: str):
        self.device = device
        self.capacity = capacity
        self.jit = jit
        self.by_weights: dict[str, Model] = {}
        self.lock = threading.Lock()

    def get(self, path: str) -> Model:
        # Resolve outside the lock: it touches the filesystem, and a bad
        # path should raise without blocking other handshakes.
        pt = resolve_weights(path)
        key = str(pt)
        with self.lock:
            if key in self.by_weights:
                return self.by_weights[key]
            if len(self.by_weights) >= self.capacity:
                raise RuntimeError(
                    f"registry full ({self.capacity}): already serving "
                    f"{[m.path for m in self.by_weights.values()]} — Stage 4b raises --max-models"
                )
            model = self._load(pt, path, model_id=len(self.by_weights))
            self.by_weights[key] = model
            return model

    def _load(self, pt: Path, path: str, model_id: int) -> Model:
        t0 = time.perf_counter()
        module = BBNet()
        state = torch.load(pt, map_location="cpu", weights_only=True)
        module.load_state_dict(state)
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
    conn: "Connection"
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


def bucket_list(max_batch: int) -> list[int]:
    """The buckets a server with this `--max-batch` can ever be asked for."""
    bs = [b for b in BUCKETS if b < max_batch]
    bs.append(max_batch)
    return bs


class EagerRunner:
    """Stage-2 behaviour: stack, H2D, call the module, D2H. No padding.

    Still the path for `--device cpu` and for `--graphs off`, and the
    fallback whenever a graph cannot be captured.
    """

    def __init__(self, module, device: str):
        self.module = module
        self.device = device
        self.bucket = 0  # "no padding" — reported in the stats histogram

    def run(self, batch: list["Request"], want_policy: bool):
        t0 = time.perf_counter()
        spatial = np.stack([r.spatial for r in batch])
        global_ = np.stack([r.global_ for r in batch])
        with torch.no_grad():
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
        return values, policies, int((t1 - t0) * 1e9), int((time.perf_counter() - t1) * 1e9)


class GraphRunner:
    """One CUDA graph captured for a fixed `(module, bucket, h, w)`.

    Padding is safe because the tower is *sample-independent*: convolutions
    and linears act per row, and BatchNorm is in **eval** mode, so it uses
    the frozen running statistics rather than the batch's. That is the
    whole reason a result cannot depend on batch composition, and it is
    what `live_server_is_batch_invariant` pins. If BatchNorm were ever left
    in train mode, padding rows would leak into real ones and that test is
    what would catch it.
    """

    def __init__(self, module, bucket: int, h: int, w: int, device: str):
        self.bucket, self.h, self.w = bucket, h, w
        self.dev_s = torch.zeros(bucket, SPATIAL_CHANNELS, h, w, device=device)
        self.dev_g = torch.zeros(bucket, GLOBAL_FEATURES, device=device)
        # Persistent pinned staging. Under Stage 2 the H2D term was 294 us
        # of a 1676 us batch and pinning was correctly judged not worth it;
        # once graphs take the batch down to ~400 us it is a third of the
        # cost, so it is worth it now. `.numpy()` aliases the pinned
        # storage, so filling a row is a plain memcpy with no allocation.
        self.host_s = torch.zeros(bucket, SPATIAL_CHANNELS, h, w).pin_memory()
        self.host_g = torch.zeros(bucket, GLOBAL_FEATURES).pin_memory()
        self.np_s = self.host_s.numpy()
        self.np_g = self.host_g.numpy()
        # Capture must not record cuDNN autotune or lazy allocator work, so
        # warm on a side stream first — this is required, not defensive.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                module(self.dev_s, self.dev_g)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.policy, self.value = module(self.dev_s, self.dev_g)

    def run(self, batch: list["Request"], want_policy: bool):
        t0 = time.perf_counter()
        n = len(batch)
        for i, r in enumerate(batch):
            self.np_s[i] = r.spatial
            self.np_g[i] = r.global_
        self.dev_s.copy_(self.host_s, non_blocking=True)
        self.dev_g.copy_(self.host_g, non_blocking=True)
        t1 = time.perf_counter()
        self.graph.replay()
        # Slice before `.cpu()`: the padding rows are computed (they are in
        # the graph) but never copied back and never sent.
        values = self.value.reshape(-1)[:n].cpu().numpy().astype(np.float32)
        policies = self.policy.reshape(self.bucket, -1)[:n].cpu().numpy().astype(np.float32) if want_policy else None
        return values, policies, int((t1 - t0) * 1e9), int((time.perf_counter() - t1) * 1e9)


class RunnerPool:
    """Chooses how to run a batch: a captured graph if one fits, else eager.

    Keyed `(model_id, h, w, bucket)`. Capture is lazy and happens on the
    batcher thread, so a graph is always replayed by the thread that
    recorded it; the first batch of an unseen bucket pays ~50 ms once. A
    bucket whose capture fails is remembered as failed and falls through to
    eager forever, so a torch/driver that cannot capture degrades to
    Stage-2 speed rather than to a dead server.
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
    """Where a batch's wall time actually goes.

    The split matters more than the total: `F`, the fixed per-batch cost,
    is what the plan's throughput model turns on, and it is the sum of
    `stage` (H2D + numpy stack), the launch/dispatch part of `fwd`, and
    `post` (D2H + the socket writes). Knowing which of those dominates is
    what decides whether the next move is CUDA graphs or a shared-memory
    ring.
    """

    batches: int = 0
    samples: int = 0
    padded: int = 0
    forward_ns: int = 0
    stage_ns: int = 0
    post_ns: int = 0
    queue_ns: int = 0
    hist: dict = field(default_factory=dict)

    def record(self, n: int, bucket: int, forward_ns: int, stage_ns: int, post_ns: int, queue_ns: int) -> None:
        self.batches += 1
        self.samples += n
        self.padded += max(bucket, n) - n
        self.forward_ns += forward_ns
        self.stage_ns += stage_ns
        self.post_ns += post_ns
        self.queue_ns += queue_ns
        self.hist[n] = self.hist.get(n, 0) + 1

    def line(self) -> str:
        if not self.batches:
            return "no batches yet"
        mean_batch = self.samples / self.batches
        total_ns = self.forward_ns + self.stage_ns + self.post_ns
        us = lambda x: x / self.batches / 1e3  # noqa: E731
        per_sample = total_ns / max(self.samples, 1) / 1e3
        q_us = self.queue_ns / max(self.samples, 1) / 1e3
        top = sorted(self.hist.items())[:8]
        # `pad` is the share of rows the GPU computed and threw away — the
        # price of a fixed-shape graph. It should stay well under 1, or the
        # bucket ladder is too coarse for the offered batch distribution.
        pad = self.padded / max(self.samples, 1)
        return (
            f"batches={self.batches} samples={self.samples} mean_batch={mean_batch:.2f} pad={pad:.2f} "
            f"batch={us(total_ns):.0f}us (stage {us(self.stage_ns):.0f} + fwd {us(self.forward_ns):.0f} "
            f"+ post {us(self.post_ns):.0f}) {per_sample:.0f}us/sample queue={q_us:.0f}us/sample "
            f"hist={top}"
        )


class Batcher(threading.Thread):
    """Greedy drain: block for one request, sweep up everything that has
    piled up during the previous forward, run it, write the responses.
    """

    def __init__(
        self,
        q: queue.Queue,
        device: str,
        max_batch: int,
        max_wait_us: int,
        stats: Stats,
        pool: RunnerPool,
        prewarm: tuple[Model, list[tuple[int, int]]] | None = None,
    ):
        super().__init__(daemon=True, name="batcher")
        self.q = q
        self.device = device
        self.max_batch = max_batch
        self.max_wait_us = max_wait_us
        self.stats = stats
        self.pool = pool
        self.prewarm = prewarm
        self.stop = threading.Event()
        # Main thread waits on this before it starts accepting, so no
        # client ever races a graph capture.
        self.ready = threading.Event()

    def run(self) -> None:
        # Capture on *this* thread: a graph is then always replayed by the
        # thread that recorded it, which sidesteps every cross-thread
        # stream question.
        if self.prewarm is not None:
            model, sizes = self.prewarm
            t0 = time.perf_counter()
            self.pool.prewarm(model, sizes)
            log(f"prewarmed {len(self.pool.graphs)} graphs in {time.perf_counter() - t0:.1f}s")
        self.ready.set()
        leftover: list[Request] = []
        while not self.stop.is_set():
            if leftover:
                batch, leftover = self._take(leftover)
            else:
                try:
                    first = self.q.get(timeout=0.25)
                except queue.Empty:
                    continue
                batch, leftover = self._take([first])
            if batch:
                try:
                    self._run_batch(batch)
                except Exception as e:  # pragma: no cover
                    log(f"batch failed: {e!r}")
                    for r in batch:
                        r.conn.close()

    def _take(self, seed: list[Request]) -> tuple[list[Request], list[Request]]:
        """Pull the seed's key out of the queue, deferring other keys."""
        key = seed[0].key
        batch = [r for r in seed if r.key == key]
        deferred = [r for r in seed if r.key != key]
        if self.max_wait_us:
            deadline = time.perf_counter() + self.max_wait_us / 1e6
        while len(batch) < self.max_batch:
            try:
                r = self.q.get_nowait()
            except queue.Empty:
                if self.max_wait_us and time.perf_counter() < deadline:
                    time.sleep(0)
                    continue
                break
            (batch if r.key == key else deferred).append(r)
        return batch, deferred

    def _run_batch(self, batch: list[Request]) -> None:
        t0 = time.perf_counter()
        queue_ns = sum(int((t0 - r.t_enqueued) * 1e9) for r in batch)
        first = batch[0]
        want_policy = any(r.want_policy for r in batch)
        runner = self.pool.get(first.model, first.h, first.w, len(batch))
        values, policies, stage_ns, forward_ns = runner.run(batch, want_policy)
        t2 = time.perf_counter()
        for i, r in enumerate(batch):
            body = values[i].tobytes()
            if r.want_policy:
                body += policies[i].tobytes()
            r.conn.send(struct.pack("<I", len(body)) + body)
        self.stats.record(
            len(batch),
            bucket=runner.bucket,
            forward_ns=forward_ns,
            stage_ns=stage_ns,
            post_ns=int((time.perf_counter() - t2) * 1e9),
            queue_ns=queue_ns,
        )


# --------------------------------------------------------------------------
# connections
# --------------------------------------------------------------------------


class Connection:
    def __init__(self, sock: socket.socket, registry: Registry, q: queue.Queue):
        self.sock = sock
        self.registry = registry
        self.q = q
        self.model: Model | None = None
        self.alive = True

    # -- io ---------------------------------------------------------------
    def recv_exact(self, n: int) -> bytes | None:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                return None
            got += k
        return bytes(buf)

    def send(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except OSError:
            self.close()

    def close(self) -> None:
        if self.alive:
            self.alive = False
            try:
                self.sock.close()
            except OSError:
                pass

    # -- protocol ---------------------------------------------------------
    def handshake(self) -> bool:
        head = self.recv_exact(12)
        if head is None:
            return False
        magic, version, c, f, path_len = struct.unpack("<4sHHHH", head)
        if magic != MAGIC:
            return self._reject(STATUS_BAD_MAGIC, f"bad magic {magic!r}")
        if version != PROTOCOL_VERSION:
            return self._reject(STATUS_BAD_VERSION, f"version {version} != {PROTOCOL_VERSION}")
        if (c, f) != (SPATIAL_CHANNELS, GLOBAL_FEATURES):
            return self._reject(
                STATUS_BAD_SHAPES, f"client encodes C={c} F={f}, server model wants {SPATIAL_CHANNELS}/{GLOBAL_FEATURES}"
            )
        raw = self.recv_exact(path_len)
        if raw is None:
            return False
        path = raw.decode("utf-8", "replace")
        try:
            self.model = self.registry.get(path)
        except Exception as e:
            status = STATUS_NO_CAPACITY if "registry full" in str(e) else STATUS_LOAD_FAILED
            return self._reject(status, str(e))
        self.send(
            struct.pack("<HHHHI", STATUS_OK, self.model.model_id, CANARY_H, CANARY_W, len(self.model.canary))
            + self.model.canary
        )
        # `MctsBot` spawns its worker threads per decision (`thread::scope`
        # in `dynamics.rs`), and connections are thread-local, so a shard
        # opens roughly one connection per decision — a few per second, not
        # one per process. Log the first few and then only every 1000th, or
        # the log is nothing but handshakes.
        n = next(HANDSHAKES)
        if n < 4 or n % 1000 == 0:
            log(f"connection #{n} bound to model_id={self.model.model_id} ({path})")
        return True

    def _reject(self, status: int, msg: str) -> bool:
        log(f"rejecting connection: {msg}")
        payload = msg.encode()
        self.send(struct.pack("<HHHHI", status, 0, 0, 0, len(payload)) + payload)
        self.close()
        return False

    def serve(self) -> None:
        try:
            if not self.handshake():
                return
            while self.alive:
                head = self.recv_exact(4)
                if head is None:
                    break
                (length,) = struct.unpack("<I", head)
                body = self.recv_exact(length)
                if body is None:
                    break
                model_id, flags, h, w = struct.unpack_from("<HHHH", body, 0)
                if model_id != self.model.model_id:
                    # Guards against a server-side routing bug once the
                    # registry grows: a request must name the model its
                    # connection was bound to, or the connection dies.
                    log(f"model_id {model_id} != handshake {self.model.model_id} — dropping connection")
                    break
                n_spatial = SPATIAL_CHANNELS * h * w
                want = 8 + 4 * n_spatial + 4 * GLOBAL_FEATURES
                if length != want:
                    log(f"request is {length} B, expected {want} B for {h}x{w} — dropping connection")
                    break
                # Zero-copy view straight onto the frame we just read; the
                # only copy is the `np.stack` the batcher does.
                flat = np.frombuffer(body, dtype="<f4", count=n_spatial + GLOBAL_FEATURES, offset=8)
                self.q.put(
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
        except OSError as e:
            log(f"connection error: {e}")
        finally:
            self.close()


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


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
        help="spin this long for more requests before running a batch. DEFAULT 0 — a timer makes the "
        "server slower than tract whenever few shards are active; greedy draining self-regulates.",
    )
    ap.add_argument("--max-models", type=int, default=1, help="registry capacity (Stage 1: 1)")
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
    prewarm = None
    if args.model:
        model = registry.get(args.model)
        warm(registry, model, sizes, args.max_batch)
        prewarm = (model, sizes)

    q: queue.Queue = queue.Queue()
    stats = Stats()
    batcher = Batcher(q, args.device, args.max_batch, args.max_wait_us, stats, pool, prewarm)
    batcher.start()
    # Capture happens on the batcher thread; don't take a connection until
    # it is done, or the first client eats a multi-second capture storm.
    batcher.ready.wait()
    gc.collect()
    gc.freeze()

    path = args.socket
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(128)

    stopping = threading.Event()

    def shutdown(signum, _frame):
        log(f"signal {signum} — shutting down. {stats.line()}")
        stopping.set()
        try:
            listener.close()
        except OSError:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.stats_every:
        def ticker():
            while not stopping.wait(args.stats_every):
                log(stats.line())

        threading.Thread(target=ticker, daemon=True, name="stats").start()

    dev = args.device
    if dev == "cuda":
        log(f"cuda device: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    log(
        f"listening on {path} device={dev} max_batch={args.max_batch} "
        f"max_wait_us={args.max_wait_us} graphs={len(pool.graphs)}"
    )

    conns = 0
    while not stopping.is_set():
        try:
            sock, _ = listener.accept()
        except OSError:
            break
        conns += 1
        c = Connection(sock, registry, q)
        threading.Thread(target=c.serve, daemon=True, name=f"conn{conns}").start()

    batcher.stop.set()
    log(f"stopped. {stats.line()}")
    if os.path.exists(path):
        os.unlink(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

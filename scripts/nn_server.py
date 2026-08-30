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
BUCKETS = (1, 2, 4, 8, 16, 32, 64)

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
        self.by_path: dict[str, Model] = {}
        self.lock = threading.Lock()

    def get(self, path: str) -> Model:
        with self.lock:
            if path in self.by_path:
                return self.by_path[path]
            if len(self.by_path) >= self.capacity:
                raise RuntimeError(
                    f"registry full ({self.capacity}): already serving "
                    f"{list(self.by_path)} — Stage 4b raises --max-models"
                )
            model = self._load(path, model_id=len(self.by_path))
            self.by_path[path] = model
            return model

    def _load(self, path: str, model_id: int) -> Model:
        pt = resolve_weights(path)
        t0 = time.perf_counter()
        module = BBNet()
        state = torch.load(pt, map_location="cpu", weights_only=True)
        module.load_state_dict(state)
        module.eval().to(self.device)
        module = maybe_trace(module, self.device, self.jit)
        canary = compute_canary(module, self.device)
        log(f"loaded model_id={model_id} {path} → {pt.name} in {time.perf_counter() - t0:.1f}s")
        return Model(model_id=model_id, path=path, module=module, canary=canary)


def resolve_weights(path: str) -> Path:
    """`models/bbnet_14x7_gen01.onnx` → `models/bbnet_14x7_gen01.pt`.

    The loop's train phase exports both side by side; the client names the
    ONNX (which is also its tract fallback) and the server consumes the
    trainer's own `.pt`, so there is no third implementation of BBNet and
    no new numerics surface.
    """
    p = Path(path)
    if p.suffix != ".pt":
        p = p.with_suffix(".pt")
    if not p.is_absolute():
        p = (REPO / p).resolve() if not p.exists() else p.resolve()
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
    forward_ns: int = 0
    stage_ns: int = 0
    post_ns: int = 0
    queue_ns: int = 0
    hist: dict = field(default_factory=dict)

    def record(self, n: int, forward_ns: int, stage_ns: int, post_ns: int, queue_ns: int) -> None:
        self.batches += 1
        self.samples += n
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
        return (
            f"batches={self.batches} samples={self.samples} mean_batch={mean_batch:.2f} "
            f"batch={us(total_ns):.0f}us (stage {us(self.stage_ns):.0f} + fwd {us(self.forward_ns):.0f} "
            f"+ post {us(self.post_ns):.0f}) {per_sample:.0f}us/sample queue={q_us:.0f}us/sample "
            f"hist={top}"
        )


class Batcher(threading.Thread):
    """Greedy drain: block for one request, sweep up everything that has
    piled up during the previous forward, run it, write the responses.
    """

    def __init__(self, q: queue.Queue, device: str, max_batch: int, max_wait_us: int, stats: Stats):
        super().__init__(daemon=True, name="batcher")
        self.q = q
        self.device = device
        self.max_batch = max_batch
        self.max_wait_us = max_wait_us
        self.stats = stats
        self.stop = threading.Event()

    def run(self) -> None:
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
        spatial = np.stack([r.spatial for r in batch])
        global_ = np.stack([r.global_ for r in batch])
        want_policy = any(r.want_policy for r in batch)
        with torch.no_grad():
            s = torch.from_numpy(spatial).to(self.device, non_blocking=True)
            g = torch.from_numpy(global_).to(self.device, non_blocking=True)
            t1 = time.perf_counter()
            policy, value = batch[0].model.module(s, g)
            # Only pay the 17 KB/sample device→host copy when somebody
            # asked for the policy — `nn-value` (the generator) never does.
            # `.cpu()` is also the sync point, so `fwd` below is the real
            # end-to-end GPU time, not just the launch.
            values = value.reshape(-1).cpu().numpy().astype(np.float32)
            policies = policy.reshape(len(batch), -1).cpu().numpy().astype(np.float32) if want_policy else None
            t2 = time.perf_counter()
        for i, r in enumerate(batch):
            body = values[i].tobytes()
            if r.want_policy:
                body += policies[i].tobytes()
            r.conn.send(struct.pack("<I", len(body)) + body)
        t3 = time.perf_counter()
        self.stats.record(
            len(batch),
            forward_ns=int((t2 - t1) * 1e9),
            stage_ns=int((t1 - t0) * 1e9),
            post_ns=int((t3 - t2) * 1e9),
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
    buckets = [b for b in BUCKETS if b <= max_batch] or [1]
    with torch.no_grad():
        for (h, w) in sizes:
            for b in buckets:
                s = torch.zeros(b, SPATIAL_CHANNELS, h, w, device=registry.device)
                g = torch.zeros(b, GLOBAL_FEATURES, device=registry.device)
                model.module(s, g)
    if registry.device == "cuda":
        torch.cuda.synchronize()
    log(f"warmed {sizes} × {buckets} in {time.perf_counter() - t0:.1f}s")


def bench(registry: Registry, model: Model, sizes: list[tuple[int, int]], iters: int = 200) -> None:
    """Sweep batch sizes and fit `t(b) = F + g·b` (plan 024 §Throughput model).

    `F`, the fixed per-batch cost, is the number the whole plan turns on:
    at `F = 4.2 ms` a server is a regression at every stream count we can
    afford; at `F = 0.3 ms` it is ~5x. Measured warmed, so cuDNN autotune
    is not charged to the fit, and with one `synchronize()` per *batch*
    (not per op), which is what the server actually pays.
    """
    device = registry.device
    for (h, w) in sizes:
        rows = []
        with torch.no_grad():
            for b in BUCKETS:
                s = torch.zeros(b, SPATIAL_CHANNELS, h, w, device=device)
                g = torch.zeros(b, GLOBAL_FEATURES, device=device)
                for _ in range(50):
                    model.module(s, g)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    model.module(s, g)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters * 1e6
                rows.append((b, dt))
                log(f"bench {h}x{w} batch {b:3d}: {dt:8.0f} us/batch  {dt / b:7.1f} us/sample")
        bs = np.array([r[0] for r in rows], dtype=np.float64)
        ts = np.array([r[1] for r in rows], dtype=np.float64)
        gg, ff = np.polyfit(bs, ts, 1)
        log(f"fit {h}x{w}: F = {ff:.0f} us/batch, g = {gg:.1f} us/sample  (device={device})")


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
        bench(registry, registry.get(args.model), parse_sizes(args.warm_sizes))
        return 0
    if args.model:
        model = registry.get(args.model)
        warm(registry, model, parse_sizes(args.warm_sizes), args.max_batch)
    gc.collect()
    gc.freeze()

    path = args.socket
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(128)

    q: queue.Queue = queue.Queue()
    stats = Stats()
    batcher = Batcher(q, args.device, args.max_batch, args.max_wait_us, stats)
    batcher.start()

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
    log(f"listening on {path} device={dev} max_batch={args.max_batch} max_wait_us={args.max_wait_us}")

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

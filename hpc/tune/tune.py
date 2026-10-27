#!/usr/bin/env python3
"""The inference tuning suite's driver (plan 069 item 4): sweep the in-process backend's knobs on one
GPU and write positions/s and CPU per forward per setting.

    python3 hpc/tune/tune.py SPEC.toml OUT_DIR [--only micro,gen] [--dry-run] [--redo]

Normally run through `hpc/tune/run_suite.sh` (builds the binaries, picks the python). Python >= 3.11,
standard library only, so it runs inside the Apptainer image (`/opt/venv/bin/python`) as well as on
the training box (`train/.venv/bin/python`).

**Workloads** (one `[[kind]]` table per sweep in the spec; every list-valued key is swept as a
cartesian product, except the lists a single process consumes: `micro.batches`, `queue.clients`):

* `micro` — `examples/inproc_bench --mode direct`: per bucket, `lanes` threads replaying that bucket's
  CUDA graph back to back, restaging every row (the batcher's per-batch work without the queue). The
  pure net throughput: positions/s vs batch, precision, layout, lanes.
* `queue` — `inproc_bench --mode queue`: N client threads calling the batch queue with no search, so
  the queue, the batchers and the bucket list are measured at a given stream count.
* `gen` — the loop's generation: `botbowl-ui dataset --mode random-start --next-drive`, the C1 preset
  (`cfgs/chance_all_partial_gen.toml`), 1000 descents, the 20x11 size mix, S streams over P processes.
* `label` — `botbowl-ui mc-label`, bare-policy playouts (8), S streams over P processes.
* `mix` — plan 069 item 4b: G generation streams and L label streams on the same GPU at once.
  `mode = "procs"` (the default): one `gen` process and one `label` process (optionally under MPS).
  `mode = "worker"`: the real thing — a private `botbowl-hub serve` on a local port and ONE
  `botbowl-worker --parallel-games G --label-streams L` (one batch queue for both), fed a `job
  generate` (the gen workload) and a `job label` over `label_corpus` (always pending); the queue's
  `label_samples=` count and the threads' names (`game-*` / `label-*`) split positions/s and CPU per
  forward by kind.

**What a row measures**, over a window after a warm-up (steady state, nothing runs out):

* positions/s: forwards that reached the GPU, from the queue's own `NN_INPROC_STATS` lines (the memo's
  hits never reach it); mean batch, padding rows per position, device time per second (`device_busy`,
  > 1 when lanes overlap on the device), the queue wait;
* CPU per forward, from the kernel's per-thread accounting (`/proc/PID/task/*/schedstat`: on-CPU ns and
  run-queue wait ns; `status`: voluntary / involuntary context switches), split by thread name into the
  batcher lanes (`nn-batcher-*`), CUDA's own threads, and the rest (search, encoder, inference client);
* instructions and cycles per forward, user and (if `perf_event_paranoid` allows) kernel mode, from
  per-thread hardware counters opened here with `perf_event_open` (no `perf` binary needed: the image
  has none). `eff_ghz` = cycles / on-CPU ns: exp080's "scheduled but idle" anomaly (threads on a CPU at
  an effective 0.9-1.3 GHz) shows as an `eff_ghz` far under the clock;
* GPU utilisation, memory, power and SM clock (`nvidia-smi`, sampled every 0.5 s over the window);
* decisions/min (gen), peak RSS, threads that came or went inside the window (their counts are partial).

**Output** in OUT_DIR: `results.csv` (every column, one row per setting — per bucket / client count for
`micro` / `queue`), `results.md` (the tables to paste into plan 069), `meta.json` (commit, host, GPU,
driver, CPU, NUMA, page size, perf permissions, the spec), and `rows/<id>/` with each row's logs. A
rerun skips rows already in `results.csv` (`--redo` reruns them), so a job that hit its walltime
resumes. Every row is seeded and records its settings and the commit.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import itertools
import json
import os
import platform
import queue
import re
import resource
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Bucket presets: the sidecar's twelve, then dense enough above 64 that padding stays under ~1/3.
DEFAULT_BUCKETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
BUCKET_PRESETS = {
    "default": DEFAULT_BUCKETS,
    "to128": DEFAULT_BUCKETS + [96, 128],
    "to256": DEFAULT_BUCKETS + [96, 128, 192, 256],
    "to512": DEFAULT_BUCKETS + [96, 128, 192, 256, 384, 512],
    "to1024": DEFAULT_BUCKETS + [96, 128, 192, 256, 384, 512, 768, 1024],
    "pow2_1024": [1 << k for k in range(11)],
}

# The loop's 20x11 size mix (plan 065; scripts/perf_search_bench.sh SIZES=mix).
MIX_SIZES = ["--size-centre", "220", "--size-temperature", "0.3", "--size-floor", "0.2", "--size-aspect", "1.5-2.8",
             "--size-max-area", "220", "--size-min-area", "70", "--cells-per-player", "26"]

KINDS = ("micro", "queue", "gen", "label", "mix")
PROCESS_LISTS = {"micro": {"batches"}, "queue": {"clients"}}
DEFAULTS = {
    "precision": "default", "channels_last": False, "lanes": 2, "buckets": "default", "pin": "none",
    "processes": 1, "mps": False, "slots": 0, "deterministic": False,
}
KIND_DEFAULTS = {
    "micro": {"batches": [1, 8, 64], "policy": True, "phase_secs": 8, "warm_secs": 2},
    "queue": {"clients": [64], "policy": True, "phase_secs": 10, "warm_secs": 3},
    "gen": {"streams": 48},
    "label": {"streams": 48, "playouts": 8},
    "mix": {"gen_streams": 48, "label_streams": 48, "playouts": 8},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ---------------------------------------------------------------------------------------------------
# Spec


def bucket_list(spec) -> list[int]:
    if isinstance(spec, list):
        return [int(b) for b in spec]
    if spec in BUCKET_PRESETS:
        return BUCKET_PRESETS[spec]
    return [int(b) for b in str(spec).split(",") if b.strip()]


def expand(spec: dict) -> list[dict]:
    """Every row the spec asks for, in spec order, each a flat dict of settings with its `kind`."""
    suite = spec.get("suite", {})
    rows = []
    for kind in KINDS:
        for block in spec.get(kind, []):
            base = {**DEFAULTS, **KIND_DEFAULTS[kind], "warm": suite.get("warm", 60), "window": suite.get("window", 120)}
            fixed = {k: v for k, v in block.items() if not isinstance(v, list) or k in PROCESS_LISTS.get(kind, ())}
            swept = {k: v for k, v in block.items() if k not in fixed}
            for combo in itertools.product(*swept.values()):
                row = {**base, **fixed, **dict(zip(swept, combo)), "kind": kind}
                row["id"] = row_id(row, suite)
                rows.append(row)
    return rows


def row_id(row: dict, suite: dict) -> str:
    """Stable over reruns: the row's settings and everything in [suite] that changes its workload."""
    keys = {k: v for k, v in row.items() if k != "id"}
    for k in ("net", "config", "iters", "sizes", "seed", "canvas", "label_corpus"):
        keys[f"suite.{k}"] = suite.get(k)
    h = hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{row['kind']}-{h}"


# ---------------------------------------------------------------------------------------------------
# Hardware counters: perf_event_open on every thread of a process, user (and kernel) mode.

PERF_TYPE_HARDWARE = 0
HW_CYCLES, HW_INSTRUCTIONS = 0, 1
NR_PERF_EVENT_OPEN = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
EXCLUDE_USER, EXCLUDE_KERNEL, EXCLUDE_HV = 1 << 4, 1 << 5, 1 << 6
READ_FORMAT = 1 | 2  # TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING, to scale if the PMU multiplexed
EVENTS = [  # name, config, flags
    ("instr_u", HW_INSTRUCTIONS, EXCLUDE_KERNEL | EXCLUDE_HV),
    ("cycles_u", HW_CYCLES, EXCLUDE_KERNEL | EXCLUDE_HV),
    ("instr_k", HW_INSTRUCTIONS, EXCLUDE_USER | EXCLUDE_HV),
    ("cycles_k", HW_CYCLES, EXCLUDE_USER | EXCLUDE_HV),
]


class Pmu:
    """Counts EVENTS on a fixed set of threads from `open` to `read`. Threads started after `open`
    are not counted (the caller reports how many); events the kernel refuses are dropped once,
    with the reason, so a box with `perf_event_paranoid` 2 still gets the user-mode counts."""

    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.events = list(EVENTS) if NR_PERF_EVENT_OPEN else []
        self.refused: dict[str, str] = {}
        self.fds: list[tuple[str, int]] = []
        try:  # one fd per thread per event: 1,000 streams want thousands
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ValueError, OSError):
            pass

    def _open(self, tid: int, config: int, flags: int) -> int:
        attr = struct.pack("IIQQQQQIIQ", PERF_TYPE_HARDWARE, 64, config, 0, 0, READ_FORMAT, flags, 0, 0, 0)
        buf = ctypes.create_string_buffer(attr, 64)
        fd = self.libc.syscall(NR_PERF_EVENT_OPEN, buf, tid, -1, -1, 8)  # PERF_FLAG_FD_CLOEXEC
        if fd < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return fd

    def open(self, tids: list[int]) -> None:
        self.fds = []
        for name, config, flags in list(self.events):
            for tid in tids:
                try:
                    self.fds.append((name, tid, self._open(tid, config, flags)))
                except OSError as e:
                    if e.errno in (13, 1, 2, 19, 95, 22) and not any(n == name for n, _, _ in self.fds):
                        # EACCES/EPERM (paranoid), ENOENT/ENODEV/EOPNOTSUPP (no such PMU event), EINVAL.
                        self.refused[name] = e.strerror
                        self.events = [ev for ev in self.events if ev[0] != name]
                        break
                    if e.errno == 24:  # EMFILE: out of fds; count what is open
                        self.refused[name] = "out of file descriptors"
                        break
                    # ESRCH: the thread exited between listing and opening; skip it.

    def read(self) -> dict[int, dict[str, float]]:
        """tid -> {event: count} (a thread that exited keeps the count it had)."""
        out: dict[int, dict[str, float]] = {}
        for name, tid, fd in self.fds:
            try:
                v, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
                out.setdefault(tid, {})[name] = v * enabled / running if 0 < running < enabled else v
            except OSError:
                pass
            finally:
                os.close(fd)
        self.fds = []
        return out


# ---------------------------------------------------------------------------------------------------
# Per-thread kernel accounting


def thread_class(comm: str) -> str:
    if comm.startswith("nn-batcher"):
        return "batcher"
    if comm.startswith(("cuda", "nvidia", "CUDA")) or comm in ("cuda-EvtHandlr", "cuda00001400006"):
        return "cuda"
    return "rest"


def threads_of(pid: int) -> dict[int, dict]:
    """tid -> {class, run_ns, wait_ns, vol, invol} for every thread of `pid` right now."""
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for t in tids:
        base = f"/proc/{pid}/task/{t}"
        try:
            comm = open(f"{base}/comm").read().strip()
            # Plan 069 item 4b: a worker's stream threads (search threads inherit their creator's name).
            sub = "gen" if comm.startswith("game-") else "label" if comm.startswith("label-") else None
            run_ns, wait_ns, _ = (int(x) for x in open(f"{base}/schedstat").read().split()[:3])
            st = open(f"{base}/status").read()
            vol = int(re.search(r"voluntary_ctxt_switches:\s+(\d+)", st).group(1))
            invol = int(re.search(r"nonvoluntary_ctxt_switches:\s+(\d+)", st).group(1))
        except (OSError, AttributeError, ValueError):
            continue
        out[int(t)] = {"class": thread_class(comm), "sub": sub, "run_ns": run_ns, "wait_ns": wait_ns, "vol": vol,
                       "invol": invol}
    return out


def rss_mb(pid: int) -> tuple[float, float]:
    try:
        st = open(f"/proc/{pid}/status").read()
        rss = int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) / 1024
        hwm = int(re.search(r"VmHWM:\s+(\d+)", st).group(1)) / 1024
        return rss, hwm
    except (OSError, AttributeError):
        return 0.0, 0.0


class GpuSampler:
    """`nvidia-smi` every 0.5 s over a window (absent nvidia-smi: no GPU columns)."""

    FIELDS = "utilization.gpu,memory.used,power.draw,clocks.sm"

    def __init__(self, gpu: str):
        self.gpu, self.proc, self.samples = gpu, None, []

    def start(self):
        if not shutil.which("nvidia-smi"):
            return
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "500", "-i", self.gpu],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append([float(x) for x in line.split(",")])
            except ValueError:
                pass

    def stop(self) -> dict:
        if not self.proc:
            return {}
        self.proc.terminate()
        self.proc.wait()
        s = self.samples
        if not s:
            return {}
        mean = lambda i: sum(r[i] for r in s) / len(s)  # noqa: E731
        return {"gpu_util": mean(0), "gpu_mem_mb": max(r[1] for r in s), "gpu_power_w": mean(2), "gpu_sm_mhz": mean(3)}


def gpu_apps_mem() -> dict[int, float]:
    out = {}
    for line in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        try:
            pid, mem = line.split(",")
            out[int(pid)] = float(mem)
        except ValueError:
            pass
    return out


class Window:
    """CPU, counters and GPU over [start, stop] for a set of processes."""

    def __init__(self, pids: list[int], gpu: str, use_pmu: bool):
        self.pids, self.gpu = pids, GpuSampler(gpu)
        self.pmu = Pmu() if use_pmu else None
        self.t0 = self.before = None
        self.freqs: list[float] = []
        self.done = threading.Event()

    def _sample_freq(self):
        """The clock (`scaling_cur_freq`) of the CPU under each running thread, every 0.5 s. On
        the training box this is what exp080's anomaly turned out to be: cores at ~1 GHz under the
        GPU workload (HWP, `powersave` + `balance_performance`), so eff_ghz ≈ this column."""
        while not self.done.wait(0.5):
            for pid in self.pids:
                for t in Path(f"/proc/{pid}/task").glob("*/stat"):
                    try:
                        f = t.read_text().rsplit(")", 1)[1].split()
                        if f[0] == "R":
                            self.freqs.append(int(Path(
                                f"/sys/devices/system/cpu/cpu{f[36]}/cpufreq/scaling_cur_freq").read_text()) / 1e6)
                    except (OSError, IndexError, ValueError):
                        pass

    def start(self):
        self.before = {pid: threads_of(pid) for pid in self.pids}
        if self.pmu:
            self.pmu.open([tid for th in self.before.values() for tid in th])
        self.gpu.start()
        threading.Thread(target=self._sample_freq, daemon=True).start()
        self.t0 = time.monotonic()

    def stop(self) -> dict:
        """Threads and counters are read first and back to back (stopping nvidia-smi can take a
        second, long enough for a bench phase's threads to exit). Every sum — schedstat and
        counters alike — runs over the threads alive at both ends, so a class's cycles and its
        on-CPU time cover the same threads; `threads_new` / `threads_gone` say what was left out."""
        secs = time.monotonic() - self.t0
        after = {pid: threads_of(pid) for pid in self.pids}
        counters = self.pmu.read() if self.pmu else {}
        self.done.set()
        gpu = self.gpu.stop()
        if self.freqs:
            gpu["cpu_freq_ghz"] = sum(self.freqs) / len(self.freqs)
        agg = {f"{c}_{k}": 0 for c in ("batcher", "cuda", "rest") for k in ("run_ns", "wait_ns")}
        agg.update(vol=0, invol=0, threads_new=0, threads_gone=0, threads=0)
        for pid in self.pids:
            b, a = self.before.get(pid, {}), after.get(pid, {})
            agg["threads"] += len(a)
            agg["threads_gone"] += len(set(b) - set(a))
            agg["threads_new"] += len(set(a) - set(b))
            for tid, t in a.items():
                p = b.get(tid)
                if p is None:
                    continue
                cls = t["class"]
                agg[f"{cls}_run_ns"] += t["run_ns"] - p["run_ns"]
                agg[f"{cls}_wait_ns"] += t["wait_ns"] - p["wait_ns"]
                agg["vol"] += t["vol"] - p["vol"]
                agg["invol"] += t["invol"] - p["invol"]
                agg[f"{cls}_vol"] = agg.get(f"{cls}_vol", 0) + t["vol"] - p["vol"]
                agg[f"{cls}_threads"] = agg.get(f"{cls}_threads", 0) + 1
                if t.get("sub"):  # inside `rest`: a worker's gen / label stream threads
                    agg[f"{t['sub']}_run_ns"] = agg.get(f"{t['sub']}_run_ns", 0) + t["run_ns"] - p["run_ns"]
                for name, v in counters.get(tid, {}).items():
                    agg[name] = agg.get(name, 0.0) + v
                    agg[f"{cls}.{name}"] = agg.get(f"{cls}.{name}", 0.0) + v
        mem = gpu_apps_mem()
        agg["gpu_proc_mem_mb"] = sum(mem.get(pid, 0.0) for pid in self.pids)
        agg["rss_mb"] = sum(rss_mb(pid)[0] for pid in self.pids)
        return {"secs": secs, **agg, **gpu, "pmu_refused": dict(self.pmu.refused) if self.pmu else {}}


# ---------------------------------------------------------------------------------------------------
# Processes


class Proc:
    """A child with its stdout/stderr timestamped into memory and into log files."""

    def __init__(self, name: str, cmd: list[str], env: dict, logdir: Path):
        self.name, self.cmd = name, cmd
        self.lines: queue.Queue = queue.Queue()
        self.stats: list[tuple[float, dict]] = []
        self.loaded = threading.Event()
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / f"{name}.cmd").write_text(" ".join(cmd) + "\n" + json.dumps(
            {k: v for k, v in env.items() if k.startswith(("BLOOD_", "BOARD_", "CUDA_"))}, indent=1) + "\n")
        self.p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  errors="replace", start_new_session=True)
        for stream, path in ((self.p.stdout, logdir / f"{name}.out"), (self.p.stderr, logdir / f"{name}.log")):
            threading.Thread(target=self._pump, args=(stream, path), daemon=True).start()

    @property
    def pid(self) -> int:
        return self.p.pid

    def _pump(self, stream, path: Path):
        with open(path, "w") as f:
            for line in stream:
                t = time.monotonic()
                f.write(line)
                f.flush()
                if line.startswith("NN_INPROC_STATS"):
                    st = parse_stats(line)
                    if st:
                        self.stats.append((t, st))
                elif line.startswith("NN_INPROC loaded") or line.startswith("BENCH_READY"):
                    self.loaded.set()
                self.lines.put((t, line.rstrip("\n")))

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.p.wait()

    def wait_line(self, prefix: str, timeout: float):
        """The next stdout/stderr line starting with `prefix`, or None (timeout or exit)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                t, line = self.lines.get(timeout=0.2)
            except queue.Empty:
                if not self.alive() and self.lines.empty():
                    return None
                continue
            if line.startswith(prefix):
                return t, line
        return None


STATS_RE = {
    "batches": r"batches=(\d+)", "samples": r"samples=(\d+)", "pad": r"pad=([\d.]+)",
    "fwd_us": r"fwd ([\d.]+)us per batch", "queue_us": r"queue=([\d.]+)us/sample",
}


def parse_stats(line: str) -> dict | None:
    out = {}
    for k, rx in STATS_RE.items():
        m = re.search(rx, line)
        if not m:
            return None
        out[k] = float(m.group(1))
    # Plan 069 item 4b: forwards from a worker's label streams (absent = 0).
    m = re.search(r"label_samples=(\d+)", line)
    out["label_samples"] = float(m.group(1)) if m else 0.0
    return out


def stats_rates(stats: list[tuple[float, dict]], t0: float, t1: float) -> dict:
    """Queue rates between the last stats line at or before t0 (+ one interval of slack) and the
    last one at or before t1. The lines are cumulative since load."""
    inside = [(t, s) for t, s in stats if t <= t1 + 0.5]
    first = [(t, s) for t, s in inside if t <= t0 + 0.5]
    if not first or len(inside) < 2 or inside[-1][0] <= first[-1][0]:
        return {}
    (ta, a), (tb, b) = first[-1], inside[-1]
    dt, ds, dbatch = tb - ta, b["samples"] - a["samples"], b["batches"] - a["batches"]
    if ds <= 0 or dbatch <= 0:
        return {"positions_per_s": 0.0}
    dev_s = (b["fwd_us"] * b["batches"] - a["fwd_us"] * a["batches"]) / 1e6
    return {
        "positions_per_s": ds / dt,
        "mean_batch": ds / dbatch,
        "pad_per_position": (b["pad"] * b["samples"] - a["pad"] * a["samples"]) / ds,
        "device_us_per_batch": dev_s * 1e6 / dbatch,
        "device_busy": dev_s / dt,
        "queue_us": (b["queue_us"] * b["samples"] - a["queue_us"] * a["samples"]) / ds,
        "stats_secs": dt,
        "label_positions_per_s": (b.get("label_samples", 0.0) - a.get("label_samples", 0.0)) / dt,
    }


def count_in(path: Path, needle: bytes) -> int:
    """Occurrences of `needle` in a (large, growing) file, streamed. The carried tail is one byte
    shorter than the needle, so no occurrence is counted twice."""
    n = 0
    try:
        with open(path, "rb") as f:
            tail = b""
            while chunk := f.read(1 << 24):
                buf = tail + chunk
                n += buf.count(needle)
                tail = buf[-(len(needle) - 1):]
    except OSError:
        pass
    return n


# ---------------------------------------------------------------------------------------------------
# The machine


class Machine:
    def __init__(self, suite: dict):
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # Under Slurm the job's GPU is already in CUDA_VISIBLE_DEVICES; the children see it as cuda:0.
        self.gpu = (cvd.split(",")[0] if cvd else str(suite.get("gpu", "0"))).strip()
        self.allowed = sorted(os.sched_getaffinity(0))
        self.numa = suite.get("numa_node")
        if self.numa is None:
            self.numa = self._gpu_numa()
        self.node_cpus = self._node_cpus(self.numa) if self.numa is not None else []
        self.numactl = shutil.which("numactl")
        self.taskset = shutil.which("taskset")

    def _gpu_numa(self):
        bus = sh(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", self.gpu])
        m = re.match(r"([0-9A-Fa-f]+):([0-9A-Fa-f]+):([0-9A-Fa-f]+)\.(\d)", bus)
        if not m:
            return None
        dev = f"{int(m.group(1), 16):04x}:{m.group(2).lower()}:{m.group(3).lower()}.{m.group(4)}"
        try:
            n = int(Path(f"/sys/bus/pci/devices/{dev}/numa_node").read_text())
            return n if n >= 0 else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _node_cpus(node: int) -> list[int]:
        try:
            spec = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
        except OSError:
            return []
        cpus = []
        for part in spec.split(","):
            a, _, b = part.partition("-")
            cpus += range(int(a), int(b or a) + 1)
        return cpus

    def cpu_set(self, pin: str, j: int, of: int, weights: list[float] | None = None) -> list[int]:
        """The CPUs process `j` of `of` gets under `pin`."""
        base = [c for c in self.node_cpus if c in self.allowed] or self.allowed
        if pin in ("none", "node"):
            return base
        if pin == "split":
            w = weights or [1.0] * of
            edges = [round(len(base) * sum(w[:k]) / sum(w)) for k in range(of + 1)]
            return base[edges[j]:edges[j + 1]] or base[:1]
        if pin.startswith("cores:"):
            return [int(c) for c in pin[6:].split(",")]
        raise ValueError(f"pin {pin!r}: want none, node, split or cores:LIST")

    def prefix(self, pin: str, j: int, of: int, weights=None) -> list[str]:
        if pin == "none":
            return []
        cpus = self.cpu_set(pin, j, of, weights)
        cl = ",".join(map(str, cpus))
        if self.numactl and self.numa is not None:
            return [self.numactl, f"--physcpubind={cl}", f"--membind={self.numa}"]
        if self.taskset:
            return [self.taskset, "-c", cl]
        raise RuntimeError("pinning asked for, but neither numactl nor taskset is installed")

    def describe(self) -> dict:
        d = {
            "host": platform.node(), "arch": platform.machine(), "kernel": platform.release(),
            "cpu_model": next((l.split(":", 1)[1].strip() for l in sh(["lscpu"]).splitlines()
                               if l.startswith("Model name")), ""),
            "allowed_cpus": len(self.allowed), "gpu": self.gpu, "gpu_numa_node": self.numa,
            "gpu_node_cpus": len(self.node_cpus), "numactl": bool(self.numactl), "taskset": bool(self.taskset),
            "page_size": os.sysconf("SC_PAGESIZE"),
            # The CPU clock policy decides the search's speed (exp080's anomaly was HWP clocking
            # the box to ~1 GHz under the GPU workload): record it.
            "cpufreq": {k: (Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{k}").read_text().strip()
                            if Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{k}").exists() else None)
                        for k in ("scaling_driver", "scaling_governor", "energy_performance_preference",
                                  "cpuinfo_max_freq")},
            "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
            if Path("/proc/sys/kernel/perf_event_paranoid").exists() else "?",
            "numa": [l for l in sh(["lscpu"]).splitlines() if "NUMA" in l],
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit",
                              "--format=csv,noheader", "-i", self.gpu]),
            "nvidia_topo": sh(["nvidia-smi", "topo", "-m"]),
            "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
                "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_GPUS", "SLURM_JOB_NODELIST")},
            "image_commit": Path("/opt/botbowl/COMMIT").read_text().strip() if Path("/opt/botbowl/COMMIT").exists() else None,
        }
        return d


def commit() -> str:
    head = sh(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"]) or "unknown"
    dirty = sh(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    return head + ("-dirty" if dirty else "")


# ---------------------------------------------------------------------------------------------------
# The suite


class Suite:
    def __init__(self, spec: dict, out: Path, args):
        self.spec, self.s, self.out, self.args = spec, spec.get("suite", {}), out, args
        self.ui = Path(os.environ.get("BIN_UI", "")) if os.environ.get("BIN_UI") else None
        self.bench = Path(os.environ.get("BIN_BENCH", "")) if os.environ.get("BIN_BENCH") else None
        # Plan 069 item 4b: `mix` rows with `mode = "worker"`.
        self.hub = Path(os.environ["BIN_HUB"]) if os.environ.get("BIN_HUB") else None
        self.worker = Path(os.environ["BIN_WORKER"]) if os.environ.get("BIN_WORKER") else None
        self.py = os.environ.get("PY", sys.executable)
        self.machine = Machine(self.s)
        self.net = self._path(self.s.get("net", ""))
        self.commit = commit()
        self.csv = out / "results.csv"
        self.done = self._done_ids()
        self.use_pmu = bool(self.s.get("pmu", True)) and NR_PERF_EVENT_OPEN is not None

    def _path(self, p) -> Path:
        """Env-expanded; relative to this checkout, else to the main checkout (a worktree has no
        models/ or runs/ of its own)."""
        p = Path(os.path.expandvars(str(p)))
        if p.is_absolute():
            return p
        if (REPO / p).exists():
            return REPO / p
        main = sh(["git", "-C", str(REPO), "worktree", "list"]).split("\n")[0].split(" ")[0]
        return Path(main) / p if main and (Path(main) / p).exists() else REPO / p

    def _done_ids(self) -> set:
        if not self.csv.exists():
            return set()
        with open(self.csv) as f:
            return {r["id"] for r in csv.DictReader(f) if r.get("status") == "ok"}

    # ---- the net's files

    def ts_for(self, precision: str, channels_last: bool) -> Path:
        """The TorchScript file a row loads, exporting the variant first if it is missing."""
        base = self.net.with_suffix(".ts")
        dtype = {"fp16": "fp16", "bf16": "bf16"}.get(precision, "fp32")
        tag = "" if dtype == "fp32" and not channels_last else dtype + ("-cl" if channels_last else "")
        path = base if not tag else base.with_name(f"{base.stem}.{tag}.ts")
        if path.exists():
            return path
        pt = self.net.with_suffix(".pt")
        if not pt.exists():
            raise FileNotFoundError(f"{path} is missing and there is no {pt} to export it from")
        cmd = [self.py, "-m", "bbnn.export_ts", str(pt), "--dtype", dtype, "--out", str(path)]
        if channels_last:
            cmd.append("--channels-last")
        log(f"exporting {path.name}")
        env = {**os.environ, "PYTHONPATH": f"{REPO / 'train/src'}:{os.environ.get('PYTHONPATH', '')}"}
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (self.out / "exports.log").open("a").write(" ".join(cmd) + "\n" + r.stdout + r.stderr + "\n")
        if r.returncode != 0:
            raise RuntimeError(f"export of {path.name} failed: {r.stderr.strip().splitlines()[-1:]}")
        return path

    def env_for(self, row: dict, streams: int) -> dict:
        b = bucket_list(row["buckets"])
        env = {**os.environ}
        env.update({
            "BLOOD_NN_CANVAS": str(self.s.get("canvas", "13x22")),
            "BLOOD_NN_LANES": str(row["lanes"]),
            # Every search thread holds at most one slot: never let slots be what the row measures.
            "BLOOD_NN_SLOTS": str(row["slots"] or max(256, streams + 2 * b[-1] + 64)),
            "BLOOD_NN_BUCKETS": ",".join(map(str, b)),
            "BLOOD_NN_PRECISION": row["precision"],
            "BLOOD_NN_CHANNELS_LAST": "1" if row["channels_last"] else "0",
            "BLOOD_NN_INPROC_STATS": str(self.s.get("stats_every", 5)),
            "BLOOD_NN_PROFILE": "1",
            "CUDA_VISIBLE_DEVICES": self.machine.gpu,
        })
        if row["deterministic"]:
            env["BLOOD_NN_DETERMINISTIC"] = "1"
        else:
            env.pop("BLOOD_NN_DETERMINISTIC", None)
        if row["precision"] == "default":
            env.pop("BLOOD_NN_PRECISION")  # exactly the loop's process
        # The capacity the binaries were built for; a worker must also run with it (plan 065).
        w, rest = str(self.s.get("capacity", "20x11/7")).split("x")
        h, p = rest.split("/")
        env.update({"BOARD_SIZE_W": w, "BOARD_SIZE_H": h, "BOARD_PLAYERS": p})
        return env

    # ---- commands

    def size_args(self) -> list[str]:
        sizes = self.s.get("sizes", "mix")
        return MIX_SIZES if sizes == "mix" else ["--board-sizes", str(sizes)]

    def gen_cmd(self, streams: int, seed: int, out: Path) -> list[str]:
        return [str(self.ui), "dataset", "--mode", "random-start", "--next-drive", "--games", str(max(64, streams * 50)),
                "--seed", str(seed), "--bot-config", str(self._path(self.s.get("config", "cfgs/chance_all_partial_gen.toml"))),
                "--mcts-iters", str(self.s.get("iters", 1000)), "--evaluator", "nn", "--model", str(self.net),
                "--nn-backend", "inproc", "--nn-device", "cuda:0", "--parallel-games", str(streams),
                *self.size_args(), "--out", str(out), "--truncate"]

    def scratch(self, row: dict) -> Path:
        """Where a row's corpora and labels go: `[suite] scratch` (default OUT/scratch), e.g. a node's
        local NVMe (`$TMPDIR` on a Slurm node), since 1,000 streams write gigabytes in a window and
        the parallel file system is no place for throwaway output. Deleted after the row unless
        --keep."""
        base = self.out / "scratch"
        if self.s.get("scratch"):
            s = os.path.expandvars(str(self.s["scratch"]))
            if "$" not in s:  # an unset variable (no $TMPDIR off Slurm) falls back to OUT/scratch
                base = Path(s)
        d = base / row["id"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def label_cmd(self, streams: int, playouts: int, row: dict, tag: str) -> list[str]:
        corpus = self.s.get("label_corpus", [])
        corpus = [self._path(c) for c in ([corpus] if isinstance(corpus, str) else corpus)]
        if not corpus:
            raise ValueError("[suite] label_corpus: no shards to label")
        row_dir = self.scratch(row)
        # mc-label writes one output per input name and skips a name it has done: repeat the shards
        # under new names so the run outlasts warm + window (`label_repeat`, default 4).
        links = row_dir / f"corpus_{tag}"
        links.mkdir(parents=True, exist_ok=True)
        names = []
        for k in range(int(self.s.get("label_repeat", 4))):
            for c in corpus:
                ln = links / f"r{k}_{c.name}"
                if not ln.exists():
                    ln.symlink_to(c)
                names.append(str(ln))
        return [str(self.ui), "mc-label", "--corpus", *names, "--model", str(self.net), "--nn-backend", "inproc",
                "--nn-device", "cuda:0", "--playouts", str(playouts), "--parallel", str(streams),
                "--seed", str(self.s.get("seed", 69400)), "--out-dir", str(row_dir / f"labels_{tag}")]

    # ---- rows

    def run(self, rows: list[dict]):
        only = set(self.args.only.split(",")) if self.args.only else None
        todo = [r for r in rows if (not only or r["kind"] in only)]
        if self.args.shard:  # K/N: every N-th row from K, so N tasks (one per superchip) split a spec
            k, n = (int(x) for x in self.args.shard.split("/"))
            todo = todo[k::n]
        todo = [r for r in todo if self.args.redo or r["id"] not in self.done]
        log(f"{len(rows)} rows in the spec, {len(todo)} to run (commit {self.commit}, gpu {self.machine.gpu}, "
            f"NUMA node {self.machine.numa}, {len(self.machine.allowed)} CPUs allowed)")
        canvas_md = self.out / "canvas.md"
        if not self.args.dry_run and not canvas_md.exists() and self.s.get("label_corpus"):
            # The padding report rides along: the corpus is the one the label rows read anyway.
            sys.path.insert(0, str(Path(__file__).parent))
            import canvas_waste
            corpus = self.s["label_corpus"]
            paths = [self._path(c) for c in ([corpus] if isinstance(corpus, str) else corpus)]
            h, w = (int(x) for x in str(self.s.get("canvas", "13x22")).split("x"))
            boards = canvas_waste.board_counts([p for p in paths if p.exists()])
            if boards:
                canvas_md.write_text(canvas_waste.report(boards, (h, w), 4))
        for i, row in enumerate(todo):
            log(f"row {i + 1}/{len(todo)} {row['id']}: " + json.dumps({k: v for k, v in row.items() if k != "id"}))
            if self.args.dry_run:
                continue
            row_dir = self.out / "rows" / row["id"]
            t0 = time.time()
            try:
                results = getattr(self, f"run_{row['kind']}")(row, row_dir)
                status = "ok"
            except Exception as e:  # a row that fails (OOM, bf16 on Pascal, …) is a result too
                results, status = [{"note": f"{type(e).__name__}: {e}"}], "error"
                log(f"  ERROR {e}")
            for r in results:
                self.write({**row, **r, "status": status, "commit": self.commit,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                            "elapsed_s": round(time.time() - t0, 1)})
            if not self.args.keep and row["kind"] in ("gen", "label", "mix"):
                shutil.rmtree(self.scratch(row), ignore_errors=True)
        if not self.args.dry_run:
            write_markdown(self.out, self.csv)

    def write(self, rec: dict):
        rec = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in rec.items()}
        old = []
        if self.csv.exists():
            with open(self.csv) as f:
                old = list(csv.DictReader(f))
        # A rerun (--redo) replaces its own line: one per row id, and per bucket / client count for
        # the bench rows.
        key = lambda r: (str(r.get("id")), str(r.get("bucket", "")), str(r.get("clients", "")))  # noqa: E731
        old = [r for r in old if key(r) != key(rec)]
        cols = list(dict.fromkeys([*(old[0].keys() if old else []), *rec.keys()]))
        with open(self.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in old:
                w.writerow(r)
            w.writerow(rec)

    def _bench_cmd(self, row: dict, ts: Path, mode: str) -> list[str]:
        cmd = [str(self.bench), "--ts", str(ts), "--device", "cuda:0", "--mode", mode, "--lanes", str(row["lanes"]),
               "--phase-secs", str(row["phase_secs"]), "--warm-secs", str(row["warm_secs"]),
               "--policy", "1" if row["policy"] else "0"]
        if row.get("boards"):
            cmd += ["--boards", str(row["boards"])]
        if mode == "direct":
            cmd += ["--batches", ",".join(map(str, row["batches"]))]
        else:
            cmd += ["--clients", ",".join(map(str, row["clients"])), "--buckets", ",".join(map(str, bucket_list(row["buckets"])))]
        return cmd

    def _run_bench(self, row: dict, row_dir: Path, mode: str) -> list[dict]:
        ts = self.ts_for(row["precision"], row["channels_last"])
        n = max(row["clients"]) if mode == "queue" else 1
        env = self.env_for(row, n)
        cmd = self.machine.prefix(row["pin"], 0, 1) + self._bench_cmd(row, ts, mode)
        p = Proc("bench", cmd, env, row_dir)
        out = []
        try:
            if not p.wait_line("BENCH_READY", 900):
                raise RuntimeError(f"bench did not load: {tail(row_dir / 'bench.log')}")
            phases = len(row["batches"]) if mode == "direct" else len(row["clients"])
            for _ in range(phases):
                got = p.wait_line("PHASE_START", 1800)
                if not got:
                    raise RuntimeError(f"bench stopped: {tail(row_dir / 'bench.log')}")
                secs = float(re.search(r"secs=([\d.]+)", got[1]).group(1))
                win = Window([p.pid], self.machine.gpu, self.use_pmu)
                win.start()
                time.sleep(max(0.5, secs - 0.5))
                w = win.stop()
                res = p.wait_line("MICRO", 600)
                if not res:
                    raise RuntimeError(f"no result line: {tail(row_dir / 'bench.log')}")
                m = dict(kv.split("=", 1) for kv in res[1].split()[1:])
                rec = {k: try_num(v) for k, v in m.items() if k not in ("id",)}
                out.append({**rec, **per_forward(w, rec.get("positions_per_s", 0.0))})
        finally:
            p.stop()
        return out

    def run_micro(self, row, row_dir):
        return self._run_bench(row, row_dir, "direct")

    def run_queue(self, row, row_dir):
        return self._run_bench(row, row_dir, "queue")

    def _steady(self, row: dict, row_dir: Path, procs: list[tuple[str, Proc]], outputs: dict) -> list[dict]:
        """Wait for every process's net to load, warm, measure one window, stop them all."""
        try:
            for _, p in procs:
                if not p.loaded.wait(900) or not p.alive():
                    raise RuntimeError(f"{p.name} did not load the net: {tail(row_dir / (p.name + '.log'))}")
            time.sleep(row["warm"])
            dead = [p.name for _, p in procs if not p.alive()]
            if dead:
                raise RuntimeError(f"{dead} exited during warm-up: {tail(row_dir / (dead[0] + '.log'))}")
            d0 = {name: count_in(path, b'"chosen_action"') for name, path in outputs.items()}
            win = Window([p.pid for _, p in procs], self.machine.gpu, self.use_pmu)
            t0 = time.monotonic()
            win.start()
            end = t0 + row["window"]
            while time.monotonic() < end and all(p.alive() for _, p in procs):
                time.sleep(1)
            t1 = time.monotonic()
            w = win.stop()
            short = [p.name for _, p in procs if not p.alive()]
            d1 = {name: count_in(path, b'"chosen_action"') for name, path in outputs.items()}
            time.sleep(float(self.s.get("stats_every", 5)) + 0.5)  # one more stats line past t1
        finally:
            for _, p in procs:
                p.stop()
        rec = {"window_s": round(t1 - t0, 1)}
        total = 0.0
        for kind in sorted({k for k, _ in procs}):
            rates = [stats_rates(p.stats, t0, t1) for k, p in procs if k == kind]
            pos = sum(r.get("positions_per_s", 0.0) for r in rates)
            total += pos
            rec[f"{kind}_positions_per_s"] = round(pos)
            for key in ("mean_batch", "pad_per_position", "device_us_per_batch", "device_busy", "queue_us"):
                vals = [r[key] for r in rates if key in r]
                if vals:
                    rec[f"{kind}_{key}"] = round(sum(vals) / len(vals), 3)
        rec["positions_per_s"] = round(total)
        if any(k == "worker" for k, _ in procs):
            # Plan 069 item 4b: one worker, both kinds; the queue counts the label streams' forwards.
            lab = sum(stats_rates(p.stats, t0, t1).get("label_positions_per_s", 0.0) for k, p in procs if k == "worker")
            rec["label_positions_per_s"] = round(lab)
            rec["gen_positions_per_s"] = round(total - lab)
            w["gen_forwards_per_s"], w["label_forwards_per_s"] = total - lab, lab
        if outputs:
            # Records land when a drive ends, so a short window counts in whole drives: coarse.
            rec["gen_decisions_per_min"] = round(sum(d1[n] - d0[n] for n in outputs) * 60 / max(t1 - t0, 1e-9), 1)
        rec.update(per_forward(w, total))
        if short:
            rec["note"] = f"{short} exited inside the window (window cut short; workload ran out?)"
        return [rec]

    def run_gen(self, row, row_dir):
        self.ts_for(row["precision"], row["channels_last"])
        P = int(row["processes"])
        with self.mps(row, row_dir):  # the daemon must be up before the first CUDA context
            procs, outputs = [], {}
            for j in range(P):
                streams = row["streams"] // P
                out = self.scratch(row) / f"gen{j}.jsonl"
                cmd = self.machine.prefix(row["pin"], j, P) + self.gen_cmd(
                    streams, int(self.s.get("seed", 69400)) + 1000 * j, out)
                procs.append(("gen", Proc(f"gen{j}", cmd, self.env_for(row, streams), row_dir)))
                outputs[f"gen{j}"] = out
            return self._steady(row, row_dir, procs, outputs)

    def run_label(self, row, row_dir):
        self.ts_for(row["precision"], row["channels_last"])
        P = int(row["processes"])
        with self.mps(row, row_dir):
            procs = []
            for j in range(P):
                streams = row["streams"] // P
                cmd = self.machine.prefix(row["pin"], j, P) + self.label_cmd(streams, int(row["playouts"]), row, f"p{j}")
                procs.append(("label", Proc(f"label{j}", cmd, self.env_for(row, streams), row_dir)))
            return self._steady(row, row_dir, procs, {})

    def run_mix(self, row, row_dir):
        if row.get("mode", "procs") == "worker":
            return self.run_mix_worker(row, row_dir)
        self.ts_for(row["precision"], row["channels_last"])
        G, L = int(row["gen_streams"]), int(row["label_streams"])
        # `split` gives each process cores in proportion to its expected CPU: a generated forward
        # costs ~2x a labelled one (exp080), so weights 2G : L.
        weights = [2.0 * G, 1.0 * L]
        out = self.scratch(row) / "gen0.jsonl"
        gcmd = self.machine.prefix(row["pin"], 0, 2, weights) + self.gen_cmd(G, int(self.s.get("seed", 69400)), out)
        lcmd = self.machine.prefix(row["pin"], 1, 2, weights) + self.label_cmd(L, int(row["playouts"]), row, "mix")
        with self.mps(row, row_dir):
            procs = [("gen", Proc("gen0", gcmd, self.env_for(row, G), row_dir)),
                     ("label", Proc("label0", lcmd, self.env_for(row, L), row_dir))]
            return self._steady(row, row_dir, procs, {"gen0": out})

    def run_mix_worker(self, row, row_dir):
        """Plan 069 item 4b as it runs: a private hub and one worker with G game + L label streams."""
        if not (self.hub and self.hub.exists() and self.worker and self.worker.exists()):
            raise RuntimeError("mode = worker needs BIN_HUB and BIN_WORKER (hpc/tune/run_suite.sh builds them)")
        self.ts_for(row["precision"], row["channels_last"])
        G, L = int(row["gen_streams"]), int(row["label_streams"])
        d = self.scratch(row)
        port = int(self.s.get("hub_port", 13500))
        token, url = d / "hub.token", f"http://127.0.0.1:{port}"
        (d / "noindex").mkdir(exist_ok=True)
        env = self.env_for(row, G + L)
        hub = Proc("hub", [str(self.hub), "serve", "--bind", f"127.0.0.1:{port}", "--token-file", str(token),
                           "--no-play", "--model-index-dir", str(d / "noindex"), "--registry-dir", str(d / "noindex"),
                           "--allowed-commits", str(d / "no-allowlist.toml")], env, row_dir)
        procs = [("hub", hub)]
        try:
            end = time.monotonic() + 60
            while not token.exists() or not hub.wait_line("[hub] listening", 1):
                if time.monotonic() > end or not hub.alive():
                    raise RuntimeError(f"hub did not start: {tail(row_dir / 'hub.log')}")
            worker = Proc("worker", self.machine.prefix(row["pin"], 0, 1) + [
                str(self.worker), "--hub", f"ws://127.0.0.1:{port}/ws", "--token-file", str(token), "--name", "tune",
                "--parallel-games", str(G), "--label-streams", str(L), "--nn-backend", "inproc", "--nn-device", "cuda:0",
                "--cache-dir", str(d / "cache"), "--mem-floor-mb", "512"], env, row_dir)
            procs.append(("worker", worker))
            client = [str(self.hub), "job"]
            auth = ["--hub", url, "--token-file", str(token)]
            out = d / "gen" / "shard0.jsonl"
            gen = client + ["generate", *auth, "--mode", "random-start", "--next-drive", "--games", str(max(64, G * 50)),
                            "--seed-base", str(self.s.get("seed", 69400)), "--bot-config",
                            str(self._path(self.s.get("config", "cfgs/chance_all_partial_gen.toml"))),
                            "--mcts-iters", str(self.s.get("iters", 1000)), "--evaluator", "nn", "--model", str(self.net),
                            *self.size_args(), "--shards", "0", "--out-dir", str(d / "gen"), "--truncate", "--batch", "1"]
            labels = self.label_cmd(L, int(row["playouts"]), row, "mix")  # its corpus links and settings
            corpus = labels[labels.index("--corpus") + 1:labels.index("--model")]
            lab = client + ["label", *auth, "--corpus", *corpus, "--model", str(self.net), "--playouts", str(row["playouts"]),
                            "--seed", str(self.s.get("seed", 69400)), "--out-dir", str(d / "labels")]
            for cmd in (gen, lab):
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
                (row_dir / "jobs.log").open("a").write(" ".join(cmd) + "\n" + r.stdout + r.stderr + "\n")
                if r.returncode != 0:
                    raise RuntimeError(f"job submit failed: {r.stderr.strip()[-300:]}")
            # `_steady` waits for the worker's net to load; the hub never loads one.
            hub.loaded.set()
            return self._steady(row, row_dir, procs, {"gen0": out})
        finally:
            for _, p in procs:
                p.stop()

    def mps(self, row: dict, row_dir: Path):
        return Mps(row_dir, self.machine.gpu) if row.get("mps") else _Null()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Mps:
    """A private MPS daemon for the row (its pipe and log dirs under the row), so several processes
    share the GPU concurrently instead of time-slicing contexts. The children find it through the
    CUDA_MPS_* variables, which are set in this process's environment while it runs."""

    def __init__(self, row_dir: Path, gpu: str):
        self.env = {"CUDA_MPS_PIPE_DIRECTORY": str(row_dir / "mps/pipe"), "CUDA_MPS_LOG_DIRECTORY": str(row_dir / "mps/log")}
        self.gpu = gpu

    def __enter__(self):
        ctl = shutil.which("nvidia-cuda-mps-control")
        if not ctl:
            raise RuntimeError("mps = true, but nvidia-cuda-mps-control is not on PATH (apptainer --nv binds it if the host has it)")
        for d in self.env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        os.environ.update(self.env)
        subprocess.run([ctl, "-d"], env={**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu}, check=True)
        return self

    def __exit__(self, *a):
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ})
        for k in self.env:
            os.environ.pop(k, None)
        return False


def per_forward(w: dict, positions_per_s: float) -> dict:
    """Window totals as rates and as per-forward costs (rates divided by positions/s)."""
    secs = max(w["secs"], 1e-9)
    pf = positions_per_s * secs  # forwards inside the window, at the measured rate
    run_ns = w["batcher_run_ns"] + w["cuda_run_ns"] + w["rest_run_ns"]
    out = {
        "cpu_cores_busy": round(run_ns / 1e9 / secs, 2),
        "runq_wait_cores": round((w["batcher_wait_ns"] + w["cuda_wait_ns"] + w["rest_wait_ns"]) / 1e9 / secs, 2),
        "threads": w["threads"], "threads_new": w["threads_new"], "threads_gone": w["threads_gone"],
        "rss_mb": round(w["rss_mb"]), "gpu_proc_mem_mb": round(w["gpu_proc_mem_mb"]),
    }
    for k in ("gpu_util", "gpu_mem_mb", "gpu_power_w", "gpu_sm_mhz", "cpu_freq_ghz"):
        if k in w:
            out[k] = round(w[k], 1)
    if w.get("pmu_refused"):
        out["pmu_refused"] = w["pmu_refused"]
    if pf <= 0:
        return out
    out.update({
        "cpu_us_per_forward": round(run_ns / 1e3 / pf, 1),
        "search_cpu_us_per_forward": round(w["rest_run_ns"] / 1e3 / pf, 1),
        "batcher_cpu_us_per_forward": round(w["batcher_run_ns"] / 1e3 / pf, 2),
        "cuda_cpu_us_per_forward": round(w["cuda_run_ns"] / 1e3 / pf, 2),
        "runq_wait_us_per_forward": round((w["batcher_wait_ns"] + w["cuda_wait_ns"] + w["rest_wait_ns"]) / 1e3 / pf, 1),
        "ctx_vol_per_forward": round(w["vol"] / pf, 2),
        "ctx_invol_per_forward": round(w["invol"] / pf, 3),
    })
    for k in ("instr_u", "cycles_u", "instr_k", "cycles_k"):
        if k in w:
            out[f"{k}_per_forward"] = round(w[k] / pf)
    # Plan 069 item 4b (a `mix` row in worker mode): each kind's stream threads per its own forward.
    for kind in ("gen", "label"):
        fwd = w.get(f"{kind}_forwards_per_s", 0.0) * secs
        if f"{kind}_run_ns" in w and fwd > 0:
            out[f"{kind}_cpu_us_per_forward"] = round(w[f"{kind}_run_ns"] / 1e3 / fwd, 1)
            out[f"{kind}_cores_busy"] = round(w[f"{kind}_run_ns"] / 1e9 / secs, 2)
    if "instr_u" in w and w.get("cycles_u"):
        out["ipc_u"] = round(w["instr_u"] / w["cycles_u"], 2)
    cycles = w.get("cycles_u", 0) + w.get("cycles_k", 0)
    if cycles and run_ns:
        out["eff_ghz"] = round(cycles / run_ns, 2)
    # Per thread class (exp080's anomaly: which threads are on a CPU without retiring cycles?).
    # `eff_ghz` = (user + kernel cycles) / on-CPU time of that class's threads.
    for cls in ("rest", "batcher", "cuda"):
        cyc = w.get(f"{cls}.cycles_u", 0) + w.get(f"{cls}.cycles_k", 0)
        rn = w.get(f"{cls}_run_ns", 0)
        out[f"{cls}_threads"] = w.get(f"{cls}_threads", 0)
        if rn:
            out[f"{cls}_cores_busy"] = round(rn / 1e9 / secs, 2)
            out[f"{cls}_vol_per_forward"] = round(w.get(f"{cls}_vol", 0) / pf, 2)
        if cyc and rn:
            out[f"{cls}_eff_ghz"] = round(cyc / rn, 2)
            out[f"{cls}_kernel_cycle_share"] = round(w.get(f"{cls}.cycles_k", 0) / cyc, 2)
        if w.get(f"{cls}.cycles_u"):
            out[f"{cls}_ipc_u"] = round(w.get(f"{cls}.instr_u", 0) / w[f"{cls}.cycles_u"], 2)
            out[f"{cls}_instr_u_per_forward"] = round(w.get(f"{cls}.instr_u", 0) / pf)
    return out


def try_num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def tail(path: Path, n: int = 3) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").strip().splitlines()[-n:])
    except OSError:
        return "(no log)"


# ---------------------------------------------------------------------------------------------------
# Tables

TABLE_COLS = {
    "micro": ["precision", "channels_last", "lanes", "bucket", "positions_per_s", "device_us_per_batch",
              "lane_cpu_us_per_position", "device_busy", "cpu_cores_busy", "gpu_util", "gpu_power_w", "status"],
    "queue": ["precision", "channels_last", "lanes", "buckets", "clients", "positions_per_s", "mean_batch",
              "pad_per_sample", "device_busy", "cpu_us_per_forward", "batcher_cpu_us_per_forward", "gpu_util", "status"],
    "gen": ["precision", "channels_last", "lanes", "buckets", "streams", "processes", "pin", "positions_per_s",
            "gen_decisions_per_min", "gen_mean_batch", "gen_device_busy", "cpu_cores_busy", "cpu_us_per_forward",
            "search_cpu_us_per_forward", "batcher_cpu_us_per_forward", "runq_wait_us_per_forward", "instr_u_per_forward",
            "eff_ghz", "rest_eff_ghz", "batcher_eff_ghz", "cuda_eff_ghz", "cpu_freq_ghz", "gpu_util", "rss_mb", "status"],
    "label": ["precision", "channels_last", "lanes", "buckets", "streams", "processes", "pin", "positions_per_s",
              "label_mean_batch", "label_device_busy", "cpu_cores_busy", "cpu_us_per_forward", "batcher_cpu_us_per_forward",
              "instr_u_per_forward", "eff_ghz", "gpu_util", "status"],
    "mix": ["mode", "precision", "lanes", "buckets", "gen_streams", "label_streams", "pin", "mps", "gen_positions_per_s",
            "label_positions_per_s", "positions_per_s", "gen_decisions_per_min", "cpu_cores_busy", "cpu_us_per_forward",
            "gen_cpu_us_per_forward", "label_cpu_us_per_forward", "gen_cores_busy", "label_cores_busy",
            "batcher_cpu_us_per_forward", "cpu_freq_ghz", "gpu_util", "status"],
}


def write_markdown(out: Path, csv_path: Path):
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    m = meta.get("machine", {})
    lines = [f"# Inference tuning suite: {out.name}", "",
             f"commit `{meta.get('commit')}`, {m.get('host')} ({m.get('arch')}, {m.get('cpu_model')}, "
             f"{m.get('allowed_cpus')} CPUs allowed, GPU NUMA node {m.get('gpu_numa_node')}, page {m.get('page_size')}), "
             f"GPU {m.get('nvidia_smi')}; net `{meta.get('suite', {}).get('net')}`; perf_event_paranoid "
             f"{m.get('perf_event_paranoid')}. Per-forward CPU is on-CPU time of every thread of the process(es) over "
             f"the window divided by the forwards that reached the GPU; `eff_ghz` = cycles / on-CPU time.", ""]
    for kind, cols in TABLE_COLS.items():
        sub = [r for r in rows if r.get("kind") == kind]
        if not sub:
            continue
        lines += [f"## {kind}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in sub:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        errs = [r for r in sub if r.get("status") != "ok"]
        for r in errs:
            lines.append(f"\n`{r['id']}`: {r.get('note', '')}")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", help="comma-separated kinds to run (micro,queue,gen,label,mix)")
    ap.add_argument("--dry-run", action="store_true", help="list the rows, run nothing")
    ap.add_argument("--redo", action="store_true", help="rerun rows already in results.csv")
    ap.add_argument("--keep", action="store_true", help="keep each row's corpus/label outputs")
    ap.add_argument("--table", action="store_true", help="only rewrite results.md from results.csv")
    ap.add_argument("--shard", help="K/N: run every N-th row starting at K (N tasks, one GPU each, own OUT dirs)")
    args = ap.parse_args()
    spec = tomllib.loads(args.spec.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.table:
        write_markdown(args.out, args.out / "results.csv")
        return 0
    suite = Suite(spec, args.out, args)
    rows = expand(spec)
    if not args.dry_run:
        for name, b in (("BIN_UI", suite.ui), ("BIN_BENCH", suite.bench)):
            needed = (name == "BIN_UI" and any(r["kind"] in ("gen", "label", "mix") for r in rows)) or \
                     (name == "BIN_BENCH" and any(r["kind"] in ("micro", "queue") for r in rows))
            if needed and (b is None or not b.exists()):
                print(f"{name} is not set to a built binary (run through hpc/tune/run_suite.sh)", file=sys.stderr)
                return 2
        meta = {"commit": commit(), "spec": spec, "suite": suite.s, "machine": suite.machine.describe(),
                "bin_ui": str(suite.ui), "bin_bench": str(suite.bench), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python": sys.version.split()[0]}
        (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    suite.run(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

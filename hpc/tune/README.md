# Inference tuning suite (plan 069 item 4)

So the first pilot hours on a GH200 measure instead of fiddle: one command sweeps the in-process
backend's knobs on one GPU and writes **positions/s and CPU per forward** per setting, as a CSV and as
markdown tables to paste into plan 069.

| file | what |
|---|---|
| `run_suite.sh` | builds `botbowl-ui` (torch) and `examples/inproc_bench` from this checkout (own target dir, capacity 20x11/7), then runs `tune.py` |
| `tune.py` | the driver: expands a spec into rows, runs each, measures, writes `results.csv` / `results.md` / `meta.json`; resumes |
| `gh200.toml` | the pilot sweep for one superchip (~2.5 h on one GPU, ~40 min over a node's four with `--shard`) |
| `smoke.toml` | the same at toy sizes for the training box's GTX 1060 (~15 min) |
| `pilot.sbatch` | the Slurm job: build once, one task per superchip, each `--shard K/4` |
| `canvas_waste.py` | padding waste of the one-canvas batch per board of the size mix, and what 2-4 canvases would buy (report only) |
| `precision_gate.sh`, `precision_gate.py` | the numerics gate for TF32 / fp16 / bf16: deltas on held-out samples on the GPU's kernels, value benches through the real queue, paired drives |

## What a row is

| kind | runs | sweeps |
|---|---|---|
| `micro` | `inproc_bench --mode direct`: per bucket, `lanes` threads replaying that bucket's CUDA graph back to back, restaging every row (the batcher's per-batch work, no queue, no search) — the cheapest, first row | precision, channels-last, lanes, batch size |
| `queue` | `inproc_bench --mode queue`: N threads calling the batch queue with no search | clients (streams), buckets, lanes |
| `gen` | `botbowl-ui dataset` as the loop generates: random-start `--next-drive`, `cfgs/chance_all_partial_gen.toml`, 1000 descents, the 20x11 mix | streams, processes per GPU, lanes, bucket list, precision, pinning, MPS |
| `label` | `botbowl-ui mc-label --playouts 8` (bare policy) | the same |
| `mix` | one `gen` + one `label` process on one GPU (item 4b: G + L streams) | G, L, pinning, MPS |

Each `gen`/`label`/`mix` row: wait for the net to load and capture, `warm` seconds, then a `window`
(default 60 + 90 s). Measured over the window, per process and summed:

- **positions/s** from the queue's own counters (`NN_INPROC_STATS`, forwards that reached the GPU), mean
  batch, padded rows per position, device time per second (`device_busy`; > 1 when lanes overlap),
  queue wait; decisions/min for generation;
- **CPU per forward** from the kernel's per-thread accounting (`/proc/PID/task/*/schedstat`), split into
  batcher lanes / CUDA's own threads / the rest (search, encoder, client); run-queue wait; voluntary and
  involuntary context switches;
- **instructions and cycles per forward**, user and (if `perf_event_paranoid` <= 1) kernel, from
  per-thread hardware counters the driver opens itself (`perf_event_open` through ctypes — the image has
  no `perf`); `ipc_u`; **`eff_ghz`** = cycles / on-CPU time, which is where exp080's unexplained
  "scheduled but not retiring" overhead (0.9-1.3 GHz effective on this box with the GPU in the process)
  shows up if Grace has it too;
- the **CPU clock** under the running threads (`cpu_freq_ghz`, `scaling_cur_freq` every 0.5 s) and `eff_ghz` per thread class (`rest_` = search, `batcher_`, `cuda_`);
- **GPU** utilisation, memory, power and SM clock (`nvidia-smi`, 0.5 s), GPU memory per process, RSS.

`micro` and `queue` rows report the bench's own numbers per bucket / client count (device µs per batch,
lane CPU µs per batch and per position) plus the same CPU/counter/GPU window around each phase.

Every row is seeded (`[suite] seed`), records its settings and the commit (`git rev-parse`, `-dirty` if
the tree is), and gets a stable id; a rerun skips rows already in `results.csv` (`--redo` reruns), so a
walltime kill costs only the row in flight. Logs and the exact command/env per process are in
`OUT/rows/<id>/`. `OUT/canvas.md` is the padding report on the label corpus.

## The knobs (all opt-in; unset = exactly today's process, bit parity with the sidecar)

| env | effect |
|---|---|
| `BLOOD_NN_BUCKETS=1,2,…,1024` | the CUDA-graph buckets (default the sidecar's twelve up to 64); the largest is the largest batch |
| `BLOOD_NN_LANES` | batcher threads / CUDA streams (2) |
| `BLOOD_NN_SLOTS` | slot ring; must exceed the streams (the suite sets streams + 2 x largest bucket + 64) |
| `BLOOD_NN_PRECISION=fp32\|tf32\|fp16\|bf16` | `fp32`/`tf32` set cuDNN and cuBLAS TF32 flags; `fp16`/`bf16` load `NET.fp16.ts` / `NET.bf16.ts` (`python -m bbnn.export_ts NET.pt --dtype fp16`: BatchNorm folded in fp32, weights cast, fp32 in and out). Unset: torch's defaults — **on Hopper that is TF32 convolutions** |
| `BLOOD_NN_CHANNELS_LAST=1` | load the `-cl` export (`--channels-last`: NHWC weights and inputs) |
| `BLOOD_NN_CANARY_TOL` | override the load canary's tolerance (1e-3 for fp32 files, 0.1 fp16/TF32, 0.5 bf16; gen49 fp16 misses the 1e-3 fixture by 0.042) |
| `BLOOD_NN_OPTIMIZE=0` | skip `optimize_for_inference` (escape hatch) |

Processes per GPU (`processes`), pinning (`pin = none | node | split | cores:LIST`: `node` = the GPU's
NUMA node's CPUs within the job's allowed set, `numactl --membind` too when `numactl` exists, else
`taskset`; `split` = disjoint slices, weighted 2:1 gen:label in a `mix` row since a generated forward
costs ~2x a labelled one), and `mps = true` (a private MPS daemon per row, if the host's
`nvidia-cuda-mps-control` is bound into the container) are row settings, not env.

## On the first pilot node

After the image is built and the first-login checklist is done (`hpc/arrhenius/README.md`). Copy from
the training box to project storage: the net with its `.pt` and `.ts` (gen49:
`models/az_v7/bbnet_mix16x9v9_gen49.{onnx,pt,ts}` → `$P/models/`) and two shards it did not train on for
the label rows (`runs/loopmix16x9v9/gen49/shard4.jsonl`, `shard5.jsonl` → `$P/runs/corpus/gen49_shard{4,5}.jsonl`,
~150 MB each), plus a value bench and a contested set for the gate.

```sh
export P=/nobackup/proj/disk/<project>/botbowl SIF=$P/images/botbowl-<commit>.sif
cd $P/botbowl_rust                                   # the checkout the image was built from, or newer
# 0. Parity first, at the default precision (plan 066 tests A/B on the H100):
apptainer exec --nv --bind $P $SIF bash -c "CARGO_TARGET_DIR=$P/target/20x11-torch \
    scripts/parity066.sh $P/runs/parity $P/models/bbnet_mix16x9v9_gen49.onnx '$P/runs/corpus/gen49_shard4.jsonl'"
# 1. The cheap rows interactively (one GPU, ~20 min): does everything load, what does the H100 do per batch?
interactive -A <project> -p gpu --gpus 1 -c 72 -t 1:00:00
apptainer exec --nv --bind $P --bind $TMPDIR $SIF hpc/tune/run_suite.sh hpc/tune/gh200.toml $P/runs/hpc_tune/pilot1 --only micro,queue
# 2. The whole spec as a job over one node's four superchips (~40 min):
sbatch -A <project> hpc/tune/pilot.sbatch           # OUT=$P/runs/hpc_tune/pilot-<jobid>/task{0..3}
# 3. The numerics gate for the precision the tables favour (fp16 expected, exp081):
apptainer exec --nv --bind $P $SIF bash -c "$P/target/tune-20x11/release/prepare ... "   # gen49 shard 4, --policy-target cq --tau 100
apptainer exec --nv --bind $P $SIF env DRIVES_SET=$P/runs/positions/contested_20x11_gen48.json \
    hpc/tune/precision_gate.sh $P/models/bbnet_mix16x9v9_gen49.onnx $P/prepared/gen49_s4 $P/runs/precision_gate \
    $P/runs/value_bench/<the 20x11 bench>.jsonl
```

Read first: the `micro` table (positions/s per batch and precision: is fp16 the ~4-7x over fp32 that
§1 assumed, and where does it saturate?), `eff_ghz` in the `gen` rows (is the scheduled-but-idle
anomaly there on Grace?), `gen` positions/s against streams (where the CPU binds) and `mix` (does
labelling fill the GPU generation leaves?). Copy `results.md`, `results.csv`, `meta.json` and
`canvas.md` back to the box and paste into plan 069 §1; the registry gets an EXPERIMENTS row.

Watch for: a `default` row whose load fails its canary (TF32 convolutions on Hopper vs the 1e-3
tolerance — the loop would fail the same way: set `BLOOD_NN_PRECISION` explicitly there); `pmu_refused`
in the CSV (paranoid too high for counters: CPU time still comes from schedstat); `threads_new` /
`threads_gone` far from 0 (the window was not steady); a `note` saying a process exited inside the
window (the workload ran out: raise `label_repeat` or `--games`).

## Here, on the GTX 1060

```sh
mkdir -p /home/mattias/repos/botbowl_rust/runs/hpc_tune
head -n 40 /home/mattias/repos/botbowl_rust/runs/loopmix16x9v9/gen49/shard4.jsonl > /home/mattias/repos/botbowl_rust/runs/hpc_tune/label_slice.jsonl
hpc/tune/run_suite.sh hpc/tune/smoke.toml /home/mattias/repos/botbowl_rust/runs/hpc_tune/smoke
```

A smoke test only: Pascal has no tensor cores (fp16 runs, slower than fp32; bf16's fused conv+ReLU
has no cuDNN engine, so the gate runs bf16 unoptimised), and other work may share the box. First run
2026-10-23: `runs/hpc_tune/smoke/` (results.md, canvas.md), `runs/hpc_tune/parity/` (0 differing bits),
`runs/hpc_tune/precision_gate/` (numerics.md, value bench); plan 069 item 4 has the numbers.

**Read `cpu_freq_ghz` next to `eff_ghz`.** On this box `eff_ghz` tracks the clock in every row (0.1-0.5 GHz under it): exp080's
"scheduled but idle" CPU time is the clock (HWP `powersave` + `balance_performance` running the
cores at ~1-2 GHz under the in-process workload), not lost cycles, and it swings ~3x between runs. A
row whose `eff_ghz` sits far below the CPU's rated clock is measuring the clock policy, not the
code: check `meta.json`'s `cpufreq` before comparing CPU per forward across machines or runs.

The value bench's states name their corpora by paths relative to the main checkout (`runs/…`):
from a worktree, `precision_gate.sh` needs a `runs` symlink to the main checkout's `runs/`; on the
cluster, put the corpus at the same relative path under the checkout.

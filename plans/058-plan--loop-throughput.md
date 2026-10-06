# Plan 058 — loop throughput: more generations per day

**Status:** Started 2026-10-06 evening, when the user lifted the "performance is deprioritized" rule
("we need all significant speed ups we can get"). Two decisions came with it:
- per generation, trust the cheap signals (value bench, absorption, net check);
- play the drive benchmark only every few generations (option 3, `DRIVE_EVAL_EVERY`, `fa539a2`).

This plan is option 5: raise the raw throughput of generation, MC labelling and eval.

## 1. Where a generation's time went (loop g056, 2026-10-06)

| phase | time | note |
|---|---|---|
| generate 8 × 300 drives, Gumbel m=16 @ 1000 | 72-233 min | 12 local streams; the laptop (6-10 streams) when connected |
| MC labels, 8 shards × ~9.5k samples × 8 playouts | 45-59 min | sidecar shared with the background net check |
| prepare + train | ~15 min | |
| drive benchmark vs the anchor | 110-250 min, overlapped | SPRT 0.5:0.55 until H0/H1 or 800 drives per set; it held up the loop twice |
| net check (background) | 82-100 min | 4 rungs at once, own sidecar |

## 2. Measurements

**Tooling:** `scripts/perf_gen_bench.sh` (games/min vs `--parallel-games`, sidecar stats) and
`scripts/perf_two_sidecars.sh`.

**First pass, a flawed benchmark.** It used games ≈ threads (48 games at 12-36 streams). Drive
lengths vary from 2 to 116 decisions, so most threads finished their one game and sat idle while
the longest finished. That read as "throughput flat at 12-14 games/min, CPU half idle, mean batch
2.4-3.6". It is the tail, not the system, and the steady-state rerun uses 4 games per thread.

**Profile of one game (`BLOOD_NN_PROFILE=1`, P=1):** per search descent, ~0.65 ms network round
trip and ~0.6 ms CPU. ~470 forwards per decision at 1000 descents; the rest hit the memo or solved
nodes.

**Steady state, generation** (4 games per thread; Gumbel m=16 @ 1000, the loop's sizes, one sidecar):

| streams | games/min | decisions/s | dataset CPU (of 800%) | mean batch | sidecar samples/s |
|---|---|---|---|---|---|
| 12 | 14.3 | 6.6 | 364% | 2.6 | 3515 |
| 24 | 14.3 | 7.1 | 426% | 5.1 | 3992 |
| 48 | 17.0-20 | 8.7 | ~590% (box ~85-100% busy) | 7.9-17.9 | 5000-5900 |

- Two sidecars with 24 streams each: 9.3 decisions/s, +7% over one with 48. The Python front end
  is not the wall.
- At 48 streams the box is near CPU-saturated (top: generator 776%, sidecar ~67%), the GPU is
  60-68% busy, and the generator holds 6-8 GB.
- A gdb thread sample said "86% waiting on the socket", but that was gdb slowing the process
  down; top and nvidia-smi are the trustworthy readings.
- **Generation is CPU-bound from ~36-48 streams.** The next lever is the search's own CPU cost
  per descent.

**MC labelling** (2859 samples × 8 playouts):

| version | `--parallel` 16 | 64 | 128 |
|---|---|---|---|
| per-trajectory work (before) | 114 s | 106 s | — |
| per-sample work (`683fb34`) | 72 s | 78 s | **57 s** |

- Before, a long drive ran serially on one thread while the rest sat idle.
- After, at 128 the sidecar serves batches of 17 at ~8000 samples/s, with the GPU ~85% busy:
  GPU-bound, as it should be.

## 3. What the loop runs with (`scripts/launch_plan058.sh`, run `runs/loopmix16x9v9`)

- `GEN_PARALLEL_GAMES=36` (12 before): ~+40% generation, memory-capped below 48 on this 16 GB box.
- `MC_LABEL_PARALLEL=96` with per-sample work: ~2x labelling.
- `DRIVE_EVAL_EVERY=3`: the drive benchmark, which took 2-4 h of shared workers every
  generation, runs on every third generation.

## 4. Candidate speedups, not yet done

- **Search CPU per descent** (generation is CPU-bound): profile the hot paths without root
  perf, e.g. sampling with gdb on a launched child, but with care: gdb distorts timing.
  Candidates seen in samples: `descend`, state `clone`, `expand_to`, path filling, `HashMap`
  churn.
- **Bigger:**
  - client-side batching (one framed multi-sample request per process, not one socket message
    per leaf);
  - several leaves in flight per search.

## 5. Engine bug found on the way

`FrenzyBlock::step` unwrapped the active player before its own guards. When the first block ended
the activation, a generator thread panicked; that happened 3 times in 48 self-play drives on the new
master. Fixed with a test that steps the procedure with no active player (`c27c30c`).

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

(Steady-state results follow.)

## 3. Candidate speedups (to rank once §2 is complete)

- **MC labelling:** playouts are cheap on the CPU and wait on the GPU, so more in flight
  (`--parallel` 16 → 48-96) should fill batches with no code change.
- **Generation:** streams per worker, chosen by the measured scaling; a second sidecar if the
  Python front end saturates.
- **Eval:** `DRIVE_EVAL_EVERY` (done).
- **Bigger, if the above is not enough:**
  - client-side batching (one framed multi-sample request per process, not one socket message
    per leaf);
  - several leaves in flight per search.

## 4. Engine bug found on the way

`FrenzyBlock::step` unwrapped the active player before its own guards. When the first block ended
the activation, a generator thread panicked; that happened 3 times in 48 self-play drives on the new
master. Fixed with a test that steps the procedure with no active player (`c27c30c`).

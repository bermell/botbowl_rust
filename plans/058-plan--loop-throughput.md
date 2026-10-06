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

## 6. The run (`runs/loopmix16x9v9`, launched 2026-10-06 21:37, `ae721b7`)

**Expectation (the user):** the first generations may read oddly, because of the new rules and the
net migration; the loop should converge.

**Value benchmark under the new rules:** `runs/value_bench/v9_gen01_val.jsonl`, 3023 states from
gen01's held-out shards (4, 7; never trained on). MC is under the init net's policy, 48 playouts,
mean SE 0.071. Built by `scripts/value_bench_build_v9.sh` while gen01 labelled. The next-drive
records are skipped there (the audit's replay is random-start only).

**Kickoff setups and skills** (`scripts/corpus_skills_setup.py`):
- Next-drive records are 45% of all records: 87% of drives score, so most get a second record.
  That nearly doubles the samples per shard (~21k vs ~9.5k), so generation and labelling take
  longer per generation.

| gen | generate | drives / next-drive records (TD rate) | optional skills asked during, offered → used | setups (decisions each; players on the LoS) | MC label | train | value bench | net check | drives vs d1k gen04 v9 |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 149 min (36 local + laptop 10) | 2400 (0.87) / 1944 (0.80) | Push 1508 → 64%, Wrestle 189 → 50%, Dodge 32 → 84%, Block 27 → 93% | 1944 (6.8; 2.8) | (running, ~15 min/shard alongside the audits) | | | | (gen03) |

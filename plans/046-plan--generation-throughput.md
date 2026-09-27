# Plan 046 — Generation throughput: more self-play data per day on Trunker

**Status:** In progress 2026-09-27. Item 2 done (c355cfb: pooled client connections, cached
content-addressed resolve). Item 1b done (`--canvas`, masked forward, on in `train_loop.sh` from
the next relaunch). Item 0 done on the user's call, together with item 1: the hub
now shares streams round-robin between running jobs, and `train_loop.sh` submits gen G's eval in the
background, where it runs alongside gen G+1's generation and is collected before training. On its
own it would mostly have shared a launch-bound GPU; with the canvas the two jobs' requests batch
together (the eval candidate is the generator). Planned 2026-09-27.

## Why

The user's framing, which this plan set out to test rather than assume:

> CPU-bound compute, memory footprint and GPU throughput all tie together. The GPU looks like the
> bottleneck partly *because* the local worker was cut from 16 to 10 streams after OOM kills. Trees
> are smaller now (1000 → 500 iterations, and `BudgetMode::Visits` exists), so memory may allow
> more streams again, which would push more load onto the GPU and CPU.

**Verdict:** the chain is right, but the binding constraint isn't the one it names. The GPU is
about 95% busy, but it spends that time on **batches of 1.55 samples**. The batch is that small
because the sidecar batches only requests with the same `(model, h, w)`, and the size curriculum
spreads 10 streams over 12 board shapes. The batch size is fully explained by the board mix (model
below: 1.55 predicted, 1.55 observed). More streams alone buy roughly +25%. One batch key per
launch unlocks about 3× GPU capacity, and after that CPU and memory become the real limits.

A second finding sits outside the generate phase itself. **Eval now takes longer than
generation**: 168 min against 99–106 per cycle, with generation only about 36% of the loop's wall
time. For data per day that is the biggest lever of all (item 0).

## Measured baseline

"M" = measured here, with the method. "I" = inferred from measured numbers. All numbers are from
2026-09-27 unless stated.

### The loop cycle (M: `runs/loopmix16x9/status.md`)

| gen | generate | prepare | train | eval |
|---|---|---|---|---|
| gen08 | 131 min | 0 | 21 | 77 (40 anchor games/board) |
| gen09 | 106 | 0 | 21 | **168** (100 anchor games/board, commit 0561cda) |
| gen10 | 99 | 0 | 21 (running) | — |

A cycle is about 295 min, so about 4.9 generations (about 770k samples) per day. Generation is
about 36% of it.

### Who generated what (M: counted `[worker] shard` lines in `gen*/generate.worker.log`)

The local worker played **2124 / 4800** games in gen10 (2073 in gen09); the laptop (tract, 10
streams) played the rest. The GPU box and a CPU-only laptop contribute about equally.

### The sidecar during gen10 generate (M: final stats line in `runs/loopmix16x9/nn_server.log`, 5955 s)

```
batches=14401990 samples=22304775 mean_batch=1.55 pad=0.00 stage 56us fwd 395us wait 10us post 26us
per batch; 255us GPU/sample queue=1101us/sample 3745 samples/s
hist=[(1, 8781395), (2, 3811366), (3, 1393211), (4, 362277), (5, 50613), (6, 3054), (7, 74)]
```

- `fwd` is **device time**: CUDA events recorded around H2D, graph replay and D2H
  (`GraphRunner.wait`). 14.4M × 395 µs = 5689 s of 5955 s, so **the GPU is 95.5% busy (M)**,
  which agrees with `nvidia-smi`. The GPU really is saturated, but by per-launch cost, not by
  arithmetic: 61% of launches carry one sample.
- 12 distinct tensor shapes were captured during the phase (M: `captured graph` lines), 7x16 to
  11x18, one per board in the curriculum.
- **Batch-key model (I, but it matches):** a stream waits for about 75% of its cycle (below), so
  about 8.5 of 10 requests are pending at any moment. Drawn from the curriculum's board weights
  (16x9 15%, 16x8 14%, 14x9 13.7%, …), 8.5–10 requests span 5.9–6.5 distinct keys, which gives
  **1.45–1.55 per key**. Observed: 1.55. With 16 streams the same model gives 1.95.
- **One forward per node, not two (M, code):** the `botbowl-mcts/CLAUDE.md` line "two forwards
  per expanded node … accepted" is **stale**. Since e9ccabd, `score_leaf` calls
  `value_home_i64_prefetch_policy` and `LAST_FORWARD` serves the priors call from the same pass.
  Fusing the heads is already done. Live: 22.3M forwards / 69,844 decisions = **319 forwards per
  decision** at 500 iterations. The only other forward is one per decision for
  `SearchSummary::evaluator_value` (about 0.3%).

### Batch-size cost curve (M: `nn_server.py --bench --device cuda`, gen09 net, **contended**)

This ran while the gen10 trainer held about 60% of the GPU, so the absolute numbers are about 2×
high (live batch-1.55 device time is 395 µs against 787 µs here). The shape of the curve is the
usable part:

| batch (11x18 = 16x9 board) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| graph µs/batch | 787 | 950 | 1015 | 1311 | 2485 |
| µs/sample | 787 | 475 | 254 | 164 | 155 |

Batch 8 costs 1.67× batch 1 for 8× the samples. Past about 8–12 the GPU is compute-bound (about
115–155 µs/sample contended, so roughly 60–80 µs uncontended; I). Plan 024's uncontended curve at
9x16 on the old schema had the same shape (408 / 443 / 482 / 578 / 921 µs for 1 / 2 / 4 / 8 / 16).
**Uncontended GPU ceiling ≈ 12k forwards/s at batch ≥ 8, about 3.2× today's 3745 (I).**

### CPU per forward (M, two ways that agree)

- Live: the local worker ran at about 250% of 800% (user's `top`), serving 3745 forwards/s, so
  **≈ 0.67 ms CPU per forward**. A stream's cycle is 10 / 3745 = 2.67 ms, so each stream is
  **about 25% busy and 75% waiting** on the sidecar (queue 1.1 ms plus device and socket time).
- Private run (below): non-NN time per forward from `BLOOD_NN_PROFILE`,
  `(wall − inference) / forwards` = **0.65–0.74 ms** on one pinned, niced core.

### Per-connection cost (M)

The sidecar logged **69,504 connections** in gen10 generate against 69,844 local decisions, so it
is **one new connection per decision**. The cause: `MctsBot::run_search` spawns a fresh scoped
thread for every search even at `workers = 1`, and `remote.rs` keys connections on the thread.
Each handshake runs `resolve_content_addressed` inside the single-threaded event loop. That reads
the client's 2 MB ONNX, globs `models/az_v7/*.onnx` and reads the matching one. Timed offline at
**3.85 ms per resolve** (M: `scratchpad/resolve_bench.py`, niced, page-cached). 69.5k × 3.85 ms
= **268 s ≈ 4.5% of the phase with the GPU loop blocked** (I). The per-thread `LAST_FORWARD` memo
is also discarded at every decision.

### Memory and search shape at 500 iterations (M: private build, see Method)

This is one stream on the largest board (16x9/6), random-start, gen09 net on tract, 4 games per
arm (77 decisions), same seeds. The sample is small, but it covers the worst board.

| | `Iterations` (live loop) | `Visits` (branch `budget-mode`) |
|---|---|---|
| descents / decision | 500 | **394 (−21%)** |
| forwards / decision | 328 | **257 (−22%)** |
| peak RSS, whole process | 327 MB | **256 MB** |
| max live heap at search end | 313 MB (486 MB in a second 2-game run) | 242 MB (271 MB) |
| largest registry (nodes) | 1393 (1746) | 850 (788) |
| median registry | 529 | 509 |
| tree reuse | 62% | 64% |

- **mem_governor's estimate at 500 iterations:** 4.5 MB/cell × 144 × 0.5 = **324 MB per 16x9
  game**. The measured single-stream peak is 0.31–0.49 GB, so the seed is **about right for the
  peak and not conservative**. The per-game average is about half that (median live heap
  100–200 MB). The governor reserves the predicted cost for the whole game, so it admits for the
  peak, which is the correct behaviour.
- **Bytes per node is unexplained (open).** A root `GameState` clone is 6.5 KB inline plus
  **7–9 allocations and 6 KB of heap** (M). Yet max live / max registry works out to about
  **280–340 KB per registry entry**. A per-search regression is weak (r = 0.04–0.54), and my attempt
  to read live heap before a fresh tree came out negative: the old tree is not freed where I
  sampled, so something else holds it past `self.cached_tree = None`. The leading hypothesis (I)
  is lazy placeholder `Node`s: every expansion allocates one per child, and 16x9 move fans are
  60+ wide. Placeholders are not counted in the registry. This matters because it sets how many
  streams fit.

### Allocations (M: counting `#[global_allocator]` in a throwaway worktree)

| arm | allocs / descent | bytes allocated / descent |
|---|---|---|
| heuristic evaluator (engine + tree only) | **957** | **0.50 MB** |
| NN on tract | 1526 | 2.62 MB |

On tract the net therefore adds about 570 allocations and 2.1 MB per descent, which is about
**880 allocations and 3.2 MB per forward** (I). That doesn't matter on the sidecar path but does
on the laptop. The heuristic arm costs 0.36 ms per descent all-in. At a typical 20–40 ns per
malloc/free pair plus memcpy of 0.5 MB, **allocation is plausibly 5–15% of search CPU (I)**:
material, not dominant. The user's suspicion is right about the count, but the numbers don't make
it the first thing to fix.

### Box (M)

- i7-7700: **4 physical cores, 8 threads**. "8 cores" overstates CPU headroom; hyperthreads
  typically add 20–30%, not 100%.
- 15.9 GB RAM; about 12.8 GB was available with the trainer running.
- GTX 1060 6 GB (GP106, sm_61). **Pascal consumer parts run fp16 at 1/64 rate**, and `nn_server.py`'s
  docstring records fp16 and channels_last as measured slower.
- Eval phase (M: gen09's sidecar session, 13:33–16:22): mean_batch 1.29, 2997 samples/s,
  23.5M × 386 µs / 10124 s = **GPU about 90% busy**. Eval is launch-bound on tiny batches too,
  with only 5 streams and two models alternating per turn.

## Where one generated game's wall time goes (question 1)

Per forward, per stream, at 10 streams (I, from the numbers above; about 33 decisions and 10.5k
forwards per game):

| component | per forward | share |
|---|---|---|
| CPU: search + engine + encode + socket (client) | ~0.67 ms | ~25% |
| waiting in sidecar queue behind other keys' launches | ~1.1 ms | ~41% |
| device time of own batch | ~0.4 ms | ~15% |
| rest: host stage/post, wake-ups, per-decision reconnect | ~0.5 ms | ~19% |

- **Binding at 10 streams:** GPU launch rate, i.e. about 2,400 launches/s × 1.55 samples. Python
  host work is not the limit. It is about 90 µs per batch, and the event loop already overlaps it
  with device time.
- **At 16 streams with today's batching:** still the launch rate, with batches of about 1.95, so
  about +25%.
- **At 16 streams with one batch key:** CPU. About 7.5k forwards/s × 0.67 ms is 5 cores, which is
  what 4C/8T plus the sidecar's own core can give, so it is CPU-bound right at the edge.
- Memory then sets how far past 16 you can go.

## Ranked opportunities

Gains are for the generate phase on Trunker alone unless stated. The laptop adds about the same
again, unchanged.

### 0. Take eval off the generation critical path (loop-level, biggest data-per-day lever)

- **Evidence (M):** eval 168 min vs generate 99–106 min per cycle. The loop is gateless (plan 030:
  "gen09 is the generator now … benchmark follows"), so gen N+1's generation does not wait on gen
  N's eval for any decision. Only the script's phase order serialises them.
- **Options:**
  - run eval as a second hub job alongside the next generate: the hub already dispatches jobs in
    order, so this needs a concurrent-jobs policy or a second local worker;
  - give eval to the laptop only;
  - evaluate every other generation.
- **Expected gain (I):** at most 295/127 ≈ 2.3× generations/day if eval were free. Realistically
  **1.3–1.6×**, because an overlapped eval shares GPU launches and cores with generation, and
  eval's own batches are tiny (mean 1.29).
- **Cost/risk:** `train_loop.sh` restructuring, which must wait for a relaunch (never edit the
  running script; see memory note "Editing a running train_loop"). Eval results arrive one
  generation later.
- **Confirm:** on the next relaunch, time one overlapped generate+eval against the serial pair,
  with the same games.

### 1. One batch key per GPU launch (the root cause of mean_batch 1.55)

Two ways to get it; either one works.

- **1a. Hub dispatches generate games board-major.**
  - `SizeDist::sample(seed)` is a pure function, so `submit_generate` can stable-sort `pending`
    by board (it is shard-major today, `botbowl-hub/src/state.rs`).
  - The corpus is unchanged: same seeds, same boards. Output line order within a shard changes;
    it is already completion-ordered.
  - About 20 lines of Rust plus a test that the seed set is unchanged.
  - **Risk:** all streams then play the *largest* board at the same time, so peak memory becomes
    N × the 16x9 tree instead of a mix. mem_governor throttles that safely, but the stream count
    must be sized for it (item 3).
- **1b. Pad to one canvas in the sidecar and mask.**
  - `BBNet` is a 3x3 residual conv tower with zero padding plus a global mean
    (`train/src/bbnn/model.py`). A masked forward, which multiplies activations by the board mask
    after each conv and uses a masked mean in the value head, reproduces every board's unpadded
    output exactly up to float reassociation, **with the same weights and no retraining**.
  - Every board then shares key `(model, 11, 18)`.
  - Python only. It needs a test that the masked padded forward matches the unpadded forward to
    `< 1e-5` at every curriculum board, and the batch-invariance test stays as it is.
  - The GPU computes the full 11x18 for smaller boards, but the curriculum is already centred at
    144 cells, so the waste is small (I).
  - **Risk:** numerics, but the canary and a parity test cover them. This is the more robust
    option because it doesn't touch memory.
- **Expected gain (I):**
  - at 10 streams, batch ≈ 8.5 at about 650 µs device time: the GPU stops binding, the cycle
    drops to about 0.67 + ~1.0 ms, and throughput is **≈ 1.5×**;
  - with 14–16 streams (item 3) it reaches the CPU ceiling at **≈ 1.8–2.2×**.
  - Eval gains less: its keys are split by model, not board, and rungs are already board-major.
- **Confirm:** replay gen10's shape with a private hub and worker (1a), or `--loadgen` with
  mixed-shape clients (1b), and read `mean_batch` and samples/s from the sidecar's stats line.
  Success is mean_batch ≥ 5 at 10 streams and ≥ 1.4× samples/s over the 3745 baseline.

### 2. Stop reconnecting to the sidecar every decision

- **Evidence (M):** 69.5k handshakes per phase, 3.85 ms each on the GPU loop, which is about 4.5%
  of the loop's time.
- **Fixes, any of:**
  - **(a)** cache `resolve_weights(path)` by path string in `nn_server.py` (3 lines; content
    identity is already the `.pt`);
  - **(b)** in `run_search`, run the step loop on the calling thread when `n_workers == 1`
    instead of spawning a scoped thread. Both stacks are 16 MB (`GAME_STACK_SIZE`,
    `WORKER_STACK_SIZE`), so this is stack-safe. It also keeps `LAST_FORWARD` and one socket per
    game.
- **Expected gain (I):** about +4–5% while the GPU binds, less once item 1 lands. (b) also
  removes one thread spawn and 16 MB stack mmap per decision.
- **Cost/risk:** trivial. (b) must keep search output identical; `tests/lazy_mover_identity.rs`
  is the style of gate.
- **Confirm:** the `connection #` count in `nn_server.log` drops from about 1 per decision to
  about 1 per stream. Compare samples/s on the next generation.

### 3. Raise the local stream count from 10 to 14–16

- **Evidence:**
  - Peak one-stream tree at 16x9, 500 iterations: 0.31–0.49 GB (M). 16 streams all at peak is
    5–8 GB, against about 12 GB available during generation when the trainer is not resident (I;
    the next generate phase must confirm available memory).
  - The OOMs that forced 16 → 10 were at **1000** iterations, and before the governor existed.
  - `scale_parallel_games` (commit 42dde75) still assumes the 1000-iteration calibration and
    ignores `MCTS_ITERS`.
- **Expected gain (I):** about +25% on its own (batch 1.55 → 1.95). It is a prerequisite for item
  1 to reach the CPU ceiling.
- **Cost/risk:** set `GEN_PARALLEL_GAMES=16` in the launch script at the next relaunch; the
  governor is the backstop. The risk is desktop-session pressure (`systemd-oomd` killed the
  session on 2026-09-24), so keep `--mem-floor-mb` at or above 1024 and watch `holding back`
  lines.
- **Confirm:** during one generation, log `free -m` and `vmstat 60` (si/so), and grep
  `holding back`. The go criterion is no swap-in and at least 2 GB available at the worst minute.

### 4. `BudgetMode::Visits` (branch `budget-mode`, 2bac2fa, unmerged)

- **Evidence (M):** −21% descents and −22% forwards per decision, and −22% peak RSS (−15% to −44%
  across runs in max live heap). Tree reuse was unchanged at 62–64%.
- **Expected gain (I):** about +25% decisions per unit of work in every regime, plus room for
  about 1.3× the streams in the same memory.
- **Cost/risk:** **it changes the search**, because reused decisions search less. Plan 045 found
  budget slack (500 = 1000 on these boards), which is suggestive but not proof. It needs a strength
  A/B and a corpus-shape check (TD/drive, samples/drive) before the loop uses it. Protocol v6
  means the laptop must be rebuilt.
- **Confirm:** `botbowl-hub job eval --bot-config visits.toml --vs-config baseline.toml` at 500
  iterations, 100+ games per board on 14x7 / 16x9 / 12x9. Keep it if the score is within ±0.03
  of 0.5.

### 5. CPU-side search cost (becomes binding after items 1 and 3)

- **Evidence (M):** 0.67 ms CPU per forward; 957 allocations and 0.5 MB allocated per descent
  on the heuristic path. The share of time in any given function is **not measured**: `perf` is
  blocked (`kernel.perf_event_paranoid=4`) and no heap profiler (heaptrack, valgrind) is
  installed.
- **Cheap first try:** a mimalloc or jemalloc global allocator in the worker. It is one
  dependency and one line, with an expected gain of 3–10% of CPU (I).
- **The profile to take (the user needs to run it):**
  ```sh
  sudo sysctl kernel.perf_event_paranoid=1
  perf record -g -p <botbowl-worker pid> -- sleep 60     # during a generate phase
  perf report --no-children --sort symbol | head -60
  ```
  Beyond allocation, check the pathfinder in `encode` (about 50 µs per forward, plan 038), the
  `GameState` clone per `apply_action`, and `available_actions`.
- **Also useful:** `sudo apt install heaptrack`, then `heaptrack` one 16x9 game, to explain the
  280–340 KB per registry entry (see Memory). Fewer bytes per node means more streams.

### 6. Tract on the idle cores (stopgap only)

- **Evidence (M):** about 4.0 ms per tract forward single-threaded at 16x9 (contended), plus about
  880 allocations. About 5 hyperthreads sit idle today.
- **Expected gain (I):** 4 tract streams would add about +20% now. **After item 1 it is
  net-negative:** a tract forward costs about 4 ms of CPU against about 0.1 ms of client time on
  the sidecar, once CPU is what binds.
- **Verdict:** only as a stopgap before item 1. The laptop remains the right place for tract.

### Not worth it (evidence against)

- **Fusing the priors and value forwards:** already done (e9ccabd).
- **fp16/AMP:** Pascal GP106 runs fp16 at 1/64 rate, and the sidecar's docstring records it as
  measured slower.
- **torch.compile / TensorRT:** at batch 1–2 the cost is the fixed per-launch cost, which CUDA
  graphs already amortise. Once item 1 lands, batches become compute-bound near 60–80 µs/sample
  (I), where fusion could buy perhaps 10–20% of GPU time, but the GPU is no longer the limit
  then.
- **Moving the batcher out of Python:** the Python loop is at about 93% of a core, but its
  per-batch host cost (about 92 µs) overlaps device time. It isn't the constraint at any
  projected load. Re-check once item 1 lands and requests exceed about 8k/s.
- **Sidecar knobs:** `--max-wait-us` cannot merge requests that have different keys. `JIT_S`
  already holds pending requests until the device frees. Neither helps until item 1 lands.
- **`--mcts-workers > 1`:** plan 024 measured a monotone +18% CPU per forward at W=8, and it
  changes the search.

## Interactions

**memory → streams → batch size → CPU:**

- **Memory → streams:** bytes per stream set the number of streams. At 500 iterations, 16 streams
  fit (item 3). `Visits` (item 4) cuts peak memory by about a quarter, and 1a raises the peak
  because all streams hit the largest board together. **Prefer 1b over 1a** if memory is tight.
- **Streams → batch size:** the effect is weak today: batch ≈ waiting ÷ distinct keys, 1.55 → 1.95
  for 10 → 16 streams. With one key (item 1) it is strong: batch ≈ waiting.
- **Batch size → throughput:** device time per sample falls about 5× from batch 1 to 8. Past
  about 8 the GPU is compute-bound near 12k/s (I).
- **Throughput → CPU:** each forward costs about 0.67 ms of CPU, so 7.5k forwards/s needs about 5
  cores. On 4C/8T that is the ceiling. Item 5 is then what moves the ceiling, and item 6 becomes
  counter-productive.
- **Budget:** `Visits` lowers forwards per decision, which multiplies decisions per second in
  whatever regime binds.
- **Eval (item 0)** competes for the same GPU launches and cores if overlapped. Item 1 frees
  exactly the launch capacity an overlapped eval needs.

**Suggested order:**

1. Items 2 and 3: configuration-sized, next relaunch.
2. Item 1b (or 1a): measure mean_batch.
3. Item 0: loop design.
4. Item 4: after its strength A/B.
5. Item 5: after a `perf` profile.

**Rough combined estimate (I, wide error bars):** the generate phase gets 1.8–2.5× faster on
Trunker. With eval overlapped, generations per day rise about 1.5–2.5×.

## Method (so the numbers can be reproduced)

- **Sidecar numbers:** the final `batches=` line of the gen10 session in
  `runs/loopmix16x9/nn_server.log` (from line 1408732). Connection count from `connection #`
  lines; shapes from `captured graph` lines.
- **Bench:**
  `nn_server.py --bench --device cuda --model models/az_v7/bbnet_mix16x9_gen09.onnx --warm-sizes 11x18,9x16`,
  run while the trainer used about 60% of the GPU (contended; rerun it idle, e.g. during
  `prepare`).
- **Private runs:**
  - worktree of 2bac2fa (0561cda + `BudgetMode`) at
    `$SCRATCH/wt-budget`, built with `BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6` into a
    scratchpad `CARGO_TARGET_DIR`;
  - a throwaway `botbowl-mcts/src/alloc_count.rs` counting allocator, plus an `ALLOC_STATS` line
    per search (registry length, descents, allocations, bytes, live and peak heap, clone cost),
    enabled by `BLOOD_ALLOC_STATS=1`. **Not committed.**
  - Runs:
    `taskset -c {6,7} nice -n 19 botbowl-ui dataset --mode random-start --board-sizes 16x9/6 --games 4 --seed 20000000 --mcts-iters 500 --evaluator nn --model …gen09.onnx`
    with `BLOOD_MCTS_BUDGET={iterations,visits}`, `BLOOD_NN_PROFILE=1`, and `/usr/bin/time -v`
    for peak RSS.
  - Heuristic arm: same flags with `--evaluator heuristic`.
- **Resolve cost:** `scratchpad/resolve_bench.py`, the same code as `resolve_content_addressed`,
  300 runs.

## Could not measure, and why

- **Live CPU profile and share of time per function:** `perf` is blocked
  (`perf_event_paranoid=4`, no sudo). The command is in item 5.
- **Live memory headroom during generation, and worker RSS at 10 streams:** generation had ended
  (gen10 finished at 18:01) before this investigation reached the box, and no memory history is
  logged. Read `free -m` / `vmstat` during the next generate phase.
- **Uncontended batch-cost curve for the current net:** the trainer held the GPU. The bench shape
  is trustworthy; its absolute values are about 2× high.
- **Bytes per tree node:** no heap profiler, and my before/after reading was confounded by where
  the old tree is actually freed. The 280–340 KB/entry figure is an upper-bound ratio, not a
  per-node measurement.
- **Sidecar Python cost per request at high load:** needs a `--loadgen` run with a live server,
  which would have competed with the trainer.
- **Anything over many games:** the private arms are 4 + 4 + 2 + 4 games on one board. They are
  good for per-decision ratios and weak for per-game distributions. Search output also varies
  run to run (`dataset-runs-nondeterministic`).

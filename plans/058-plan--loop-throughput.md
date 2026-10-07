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

- **Search CPU per descent:** done, see §7 (−76% instructions per search, identical games).
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
| 01 | 149 min (36 local + laptop 10) | 2400 (0.87) / 1944 (0.80) | Push 1508 → 64%, Wrestle 189 → 50%, Dodge 32 → 84%, Block 27 → 93% | 1944 (6.8; 2.8) | 90 min (shared GPU with two audits), 0 unlabelled | 9 min, final step 10965; val_value 0.116 → 0.092, val_policy 0.681 → 0.674, top1 0.715 → 0.720 | 0.209 → 0.218 (paired +4.3%, ~5 SE) | init net on gen01: +0.002/+0.004/+0.008/+0.007 MONOTONE (4000 within 1 SE of 1000); slope 0.72 | (gen03) |
| 02 | 143 min | 2400 (0.88) / 1949 (0.81) | Push 1475 → 66%, Wrestle 154 → 48%, Dodge 40 → 85%, Block 27 → 93% | 1949 (6.9; 2.8) | 78 min, 0 unlabelled | 25 min, final step 21636; val_value 0.095 → 0.091, val_policy 0.679 → 0.676; absorb dP(played) +0.004, dtop1 +0.003 | 0.218 → **0.212 (paired −2.9%, ~7 SE)**, bias −0.005 | gen01 on gen02: +0.001/+0.001/+0.003/+0.005 MONOTONE; slope 0.66 | |
| 03 | 141 min | 2400 (0.87) / 1924 (0.81) | Push 1460 → 69%, Wrestle 180 → 56%, Dodge 37 → 84%, Block 17 → 88% | 1924 (6.9; 2.8) | 75 min, 0 unlabelled | 42 min (3-gen window), step 32000; absorb dP(played) +0.001, dtop1 −0.000, dvalMSE −0.006 | 0.212 → 0.214 (paired +0.9%, ~1.6 SE: flat) | gen02 on gen03: +0.003/+0.003/+0.006/**+0.010** MONOTONE; slope 0.76 | 14x7 0.449 ± 0.035 (H0, 68 pairs); 16x9 0.481 ± 0.025 (H0, 101 pairs) |
| 04 | 292 min (36 local; shared 112 min with gen03's drive benchmark; laptop 5 streams only near the end) | 2400 (0.87) / 1910 (0.80) | Push 1560 → 68%, Wrestle 181 → 46%, Dodge 38 → 89%, Block 28 → 96% | 1910 (6.9; 2.8) | 75 min, 0 unlabelled; prepare drops 20% as forced | 28 min, step 25000; combined val 0.924 (not comparable: forced samples, zero policy loss, left val too); absorb dP(played) +0.004, dtop1 +0.002, dvalMSE −0.005 | 0.214 → **0.209 (paired −2.3%, ~4 SE)**, bias −0.015 → −0.006 | gen03 on gen04: +0.002/+0.004/+0.007/+0.009 MONOTONE; slope 0.74 | |
| 05 | 196 min (28 local; laptop ~1 h of it); first corpus from the forced-move bot | 2400 (0.89) / 1955 (0.80) | Push 1447 → 69%, Wrestle 163 → 41%, Dodge 41 → 88%, Block 25 → 84% | 1955 (6.9; 2.8) | 79 min, 0 unlabelled | 28 min, step 25000; absorb dP(played) +0.001, dtop1 −0.000, dvalMSE −0.005 | 0.209 → **0.206 (paired −1.2%, ~3 SE)**, bias +0.004 | gen04 on gen05: +0.002/+0.003/+0.003/+0.007 MONOTONE; slope 0.43 (gen04 corpus: 0.74) | (gen06) |

**gen01, the first fine-tune on the new rules.** It learned steadily (every validation metric moved
at almost every checkpoint, unlike g056's flat curves), and the absorption probe is strong (dtop1
+0.009, dvalMSE −0.056 vs the raw outcome). Yet the value bench, which holds random-start drive-1
states only, reads 4.3% worse than its generator. A plausible reading: the gains are in the setup
and kickoff states (the next-drive records, in val but not in the bench). The user expected odd
early numbers; watch the trend.

**gen02 recovers the value bench:** 0.218 → 0.212 (paired −2.9%), about back to the init net's
0.209, with bias near zero. Training still moves every metric, more slowly than gen01.

**Forced moves (the user, 2026-10-07; `3dc4902`, `2afd0db`, done by a subagent and merged):**
- **The bot.** `MctsBot` plays a decision with exactly one action left after pruning
  (`pruning::search_actions`, the set a search root expands, with its empty-set fallback) without
  searching: no tree, no descent, no network forward. This covers every entry point and both roots
  (PUCT, Gumbel). Before, only setup placements and decisions with one engine-legal action skipped
  the search, so every other forced root spent a full 1000-descent search on its only move.
- **The corpus.** It still records forced decisions, as one-child scripted samples, because replay
  (mc-label, the audits) steps through every recorded move.
- **Training.** `prepare` drops every sample with fewer than two children by default (opt-out
  `--keep-forced`), so the network never trains on a decision with nothing to choose.
- **Rollout.** The loop stopped before gen04's mc-label and was relaunched on it (2026-10-07 15:12,
  `407b335`, 28 generation streams, the hub open on `0.0.0.0:13337` with `/play/` for human games), with
  `EXTRA_ALLOW=ae721b7` keeping the laptop: its build finds the same move by search instead, and
  prepare drops the record either way. From then on, gen04 trains without forced samples and gen05
  is generated by the fixed bot.
- **How much it was (gen04, the last corpus from the old bot):** 30,133 of 150,520 samples (20.0%)
  had fewer than two children after pruning, each one a full 1000-descent search. By the
  procedure on top: MoveAction 59% of them (31% of all MoveAction decisions), Turn 12%,
  BlockAction 9% (61% of its decisions), Block 7%, Kickoff 6% (all of them), FollowUp 4%, Push 3%.
  gen04's 3-generation window prepared 361k samples where gen03's held 456k (−21%). So gen05
  should generate measurably faster per drive.
- **Known, not caused by it:** `botbowl-mcts/tests/solved_early_stop.rs` fails its 3 s debug-build
  limit under load (also on the base commit), and one `botbowl-ui` unit test failed once under load
  and passed on two reruns.

**gen03 vs the anchor: below even on both boards** (14x7 0.449 ± 0.035, 16x9 0.481 ± 0.025, both
H0, both within ~1.5 SE of 0.5).
- The anchor is d1k gen04, migrated, which has never trained on the new rules. Yet it holds its
  own against a net that has had three generations of them.
- There is no baseline for the loop's start: the init net (g056 gen04) was never measured
  against the anchor under the new rules. So "the loop regressed" and "the loop has not yet
  caught up to where it started" can't be told apart.
- The cheap fix is one drive match, the init net vs the anchor, under the same settings.

## 7. Search CPU: profiled and cut by 76% (2026-10-07)

The user enabled `perf` (`kernel.perf_event_paranoid=1`) and installed heaptrack/valgrind
(plan 046 item 5).

**Live profile.** 45 s flat profile plus a 40 s LBR call-graph profile (`perf record --call-graph
lbr`; the release binary has no frame pointers) of the running `botbowl-worker` during gen05
generate (28 streams, sidecar):

| inclusive share of worker CPU | what |
|---|---|
| 41% | `BloodBowlDynamics::apply_action` inside `Tree::descend`, 33% of it `MoveAction::step` → `fill_player_paths` |
| 19% | node teardown: `Node::on_drop` → `detach` → `parents`/registry `remove` → `Node::eq` → **two `GameState` clones per comparison** |
| 11% | NN forward client side (`forward_memo`), 4% of it `encode` (3% path-finding again) |
| 13% / 9% | malloc+free / memmove, spread over the above |

**Benchmark that the load cannot distort** (`scripts/perf_search_bench.sh`, `3e49097`; run it on two commits). `instructions:u` (perf stat) of a fixed workload:
`dataset --mode random-start --board-sizes 16x9/6 --games 3 --seed 4242 --mcts-iters 1000
--bot-config cfgs/gumbel16_f1000_gen.toml --evaluator heuristic`. It needs two things to be
reproducible:
- `getrandom` pinned by an `LD_PRELOAD` shim. std's `RandomState` keys feed the children-map
  iteration order, which breaks search ties;
- `recon_mcts`'s `deterministic_hash` on in the build (test-only in the repo). Without it, any
  change that creates fewer `HashMap`s shifts every later map's keys, because std advances the
  per-thread key on each `RandomState::new`, and the games diverge.
With both, the search is exactly reproducible, and two runs agree to 0.002% in instructions. The
only remaining run-to-run difference was the JSON order of `HashSet` fields (`skills`, `simple`),
so the output check sorts those. That is probably what plan 032 saw when `deterministic_hash`
alone "did not restore reproducibility".
Every change below keeps the trajectories byte-identical (same hash), also with the real gen04
net on CPU (tract).

| commit | change | instructions | cumulative |
|---|---|---|---|
| base `1ec7517` | | 126.8G | |
| A `3d3ba5e` | `recon_mcts`: node equality compares the stored states in place and short-circuits on pointer identity; it used to clone both `GameState`s. Teardown no longer re-materialises a state the child already holds. | 83.6G | −34% |
| B `a71770c` | engine: `SkillSet` (a `u64` bitset) replaces `HashSet<Skill>` for `skills` and `used_skills`: clones without allocating, compares by integer. Hash value-for-value and serde unchanged (tests pin both). | 77.8G | −39% |
| C `9d6ecdf` | `recon_mcts::Tree::descend` reads a registered child's stored state instead of re-deriving it with `apply_action`. The descent replayed the engine along every edge of every descent, path-finding at each `MoveAction`. Placeholders (and `GetState`) still derive. | 30.5G | **−76%** |

On the NN path (tract, 1 game, 300 descents), the non-network instructions fall about as much
(≈35G → ≈8G). `eq_checks`/`eq_rejects` move by ~2%: they count hash-bucket false candidates,
which depend on the registry's `RandomState` keys. `recomb_hits`/`recomb_probes` are identical.

**What it means for the loop.** Generation was CPU-bound from ~36 streams (§2) at ~0.6 ms of
worker CPU per forward. With about a quarter of that left, the GPU sidecar should bind instead
(60-68% busy at 48 streams before), and the stream count is capped by memory, not CPU. Expected:
more games/min per stream at the same stream count, until the GPU saturates. To measure on the
next relaunch: worker CPU% and sidecar `mean_batch`/samples/s at 28 streams against gen05's.
Eval (drive benchmark) and the net check gain the same per search. mc-label does no search and
gains only from B.

**Next candidates (after the relaunch measurement):**
- path-finding is still the top engine cost wherever a state *is* derived (new leaves);
- `encode` recomputes the paths `MoveAction` already put in `path_buffer`.


## 8. `GameState` shrunk 4x: search memory −35%, CPU −23% (2026-10-07)

The question behind it: could the search pre-allocate a fixed arena of nodes with their states,
so a game's memory is known up front and a box's stream count follows from its RAM? Sizing the
pieces first said the arena is secondary: a stored state was ~22 KB (`GameState` 10.0 KB, of
which the board 7.6 KB at 16 bytes a square as `Option<usize>`; plus a boxed `AvailableActions`
of 11.5 KB, a 24-byte `SmallVec<PosAT>` a square), and the worker's governor was calibrated at
~450 KB per descent, i.e. tens of states a descent. The node itself is a few hundred bytes. So
the state was shrunk before anything is done about allocation:

- `board: FullPitch<BoardCell>`, one byte a square (id + 1, `0` empty); serde unchanged.
- `AvailableActions`: `SimpleATSet` (`u32`) for the simple set, `FullPitch<PosATSet>` (`u16` a
  square) for the positional offerings, `enum_bitset!` in `table.rs` in `SkillSet`'s mould;
  `Copy` contents, derived `Hash`; serde unchanged except that a square's list now reads in
  variant order (sets, not sequences; the benchmark hash sorts them).
- `proc_stack: SmallVec<[AnyProc; 16]>`: a random game peaks 9 deep, so a clone copies the stack
  instead of allocating it; deeper spills.
- `botbowl-engine/tests/state_size.rs` pins all three and the stack depth.

| | `GameState` | `AvailableActions` (boxed) | per stored state |
|---|---|---|---|
| before | 10,024 B | 11,488 B | ~21.5 KB |
| after | 4,416 B | 960 B | ~5.4 KB |

Of the 4.4 KB left: rosters 1.7 KB (`[Option<FieldedPlayer>; 24]` at 40 B and dugout at 32 B),
the inline stack 1.5 KB, the board 0.5 KB, the rng ~0.3 KB. The one allocation a clone still
makes is the `Box<AvailableActions>`.

Benchmark (plan 058 §7's workload on macOS: 3 games, seed 4242, 1000 descents, heuristic,
`cfgs/gumbel16_f1000_gen.toml`, 16x9/6, `deterministic_hash`; two runs each, `/usr/bin/time -l`):

| | user CPU | peak RSS | trajectories |
|---|---|---|---|
| `5076090` | 3.64 s / 3.60 s | 330 MB | `a6d6c2d1d4bf` |
| this | 2.79 s / 2.77 s | 214 MB | `a6d6c2d1d4bf` |

Same trajectories, so the search is unchanged; −23% CPU, −35% peak memory. The remaining
footprint is not the state: 214 MB over 3 games × ~28 searches × 1000 descents is still well
over 5 KB a descent, so the next measurement is nodes per descent and bytes per node in
`recon_mcts` (`Node` holds four `RwLock`s, a `HashSet` of parents and a `HashMap` of children).
A node cap in `recon_mcts` would then make the per-game bound exact and the worker's
`mem_governor` arithmetic instead of an EWMA; an index-based arena only if a profile shows the
allocator and pointer chasing as the residual.

# Plan 069 — getting ready for the Grace Hopper allocation

**Status:** started 2026-10-21. **The user (2026-10-21):** "I've sorted out thousands of GPU hours in an
HPC cluster … an Nvidia Grace Hopper system. From now on running the loop to generate data and training
even stronger networks isn't as important anymore … We should instead spend the time preparing so that we
make the most out of those GPU hours: bug chasing, optimizing, planning experiments and planning how to
tune the inference in the new system."

## 0. What we have

- **Allocation:** at least **5,000 GPU-hours**, likely **50,000** after that. One node = 4 GH200 = 4
  GPU-hours per wall hour, so 5,000 GPU-h = 1,250 node-hours ≈ 52 node-days (50k ≈ 520 node-days).
- **Machine:** 382 GPU nodes × 4 GH200 superchips (each: 72 Neoverse V2 Arm cores, one H100, coherent
  NVLink-C2C memory); an HPC CPU partition of 424 AMD Turin 128-core CPUs; a persistent partition with a
  cloud interface; a 29 PB parallel file system.
- **Target (the user):** **20x11/7 first**, until it plays above human level — the user will rally the
  community to evaluate it — then full size (26x15/11) if hours remain. The current loop already runs
  20x11/7 (plan 065).
- **Open questions for the centre** (they decide the orchestration): job walltime limit, queue/backfill
  policy, job arrays, whether the CPU partition is billed from the same hours, container runtime
  (Apptainer?), whether compute nodes can reach a service on the persistent partition (the hub), outbound
  internet, storage quotas, allocation expiry.

## 1. The cost model (CPU side measured 2026-10-22, exp080; GPU side still estimates)

On this box generation is bound by GPU arithmetic (GTX 1060: 625 µs of kernels per position, 3,150
positions/s in-process, plan 066). On a GH200 the balance is CPU against GPU. The GPU rows are still
estimates (work item 4 measures them on an H100):

| | estimate | basis |
|---|---|---|
| net cost (96x8 + heads, 13x22 canvas) | ~0.85 GFLOP / position | 16 3x3 convs at 96 ch on 286 cells + stem and heads |
| H100, fp32 | ~40k positions/s | ~67 TFLOPS fp32 at ~50% |
| H100, TF32 / bf16 tensor cores | ~150–300k positions/s | numerics change; needs its own gate (§3) |

**CPU per forward and per descent, measured** (generation at the loop's settings: 20x11/7 capacity, C1
preset `cfgs/chance_all_partial_gen.toml`, 1000 descents, random-start, `--next-drive`; gen48 in-process
on the 1060, one stream, one lane, cuDNN deterministic; user-mode instructions counted after the net is
loaded; split from an LBR profile of the same workload). "base" is `ef820ec`, "now" is this branch after
the cuts in §1b; both play **bit-identical games** (same trajectory hash, `1d15aaee67ff`). CPU-µs =
instructions / 5.2 per ns (see "how measured").

| 20x11/7, real net, per forward | base: instructions | base: µs | now: instructions | now: µs |
|---|---|---|---|---|
| (a) search: recon_mcts + dynamics + engine stepping, cloning, hashing, eq | 0.788M | 152 | 0.553M | 106 |
| … of which the pathfinder filling move offerings | 0.270M | 52 | 0.195M | 37 |
| (b) encoder | 0.263M | 51 | 0.065M | 12 |
| … of which the path-probability plane's pathfinder | 0.196M | 38 | 0.010M | 2 |
| (c1) inference client on the search thread (memo key and clone, staging, priors gather, softmax) | 0.062M | 12 | 0.059M | 11 |
| **per forward, batch-independent (a + b + c1)** | **1.11M** | **214** | **0.68M** | **130** |
| (c2) batcher thread + CUDA driver, **per batch** (libcuda 60% of it, memcpy, libtorch) | 0.19M | 36 | 0.20M | 38 |
| (d) rest (game driver, JSON, random starts) | <0.1% | | <0.1% | |
| kernel-mode instructions per forward (futex, ioctl; latency-dependent) | 0.34M | | 0.15M | |
| per descent, batch-independent (× 0.613 forwards per descent) | 0.68M | 131 | 0.42M | 80 |

| per descent, user instructions | heuristic (no net) | null net (`--nn-backend null`) | real net, in-process |
|---|---|---|---|
| 16x9/6, base → now | 0.349M → 0.240M | 0.519M → 0.281M | 0.656M → 0.441M |
| 20x11/7, base → now | 0.399M → 0.283M | 0.614M → 0.334M | 0.798M → 0.535M |
| loop's size mix, base → now | 0.375M → 0.263M | 0.533M → 0.291M | 0.762M → 0.508M |

(The real-net column includes the batch-1 batcher; null and heuristic arms subtract their 1.5G-instruction
start-up. Null is the NN code path with value 0 and uniform priors: its trees are shallower and wider than
the real net's, so it is the search-cost benchmark, not a substitute for the real-net split.)

**Forwards.** 0.613 forwards per descent, 600 per search, 471 per decision (80 of 374 decisions were
forced, no search). The search asks the evaluator 1.20 times per descent; the `LAST_FORWARD` memo serves
49% of those asks (the priors call after the value call on the same state). The rest of the descents end
without a new forward: recombination hits (6.2% of registry probes), known outcomes (score change, half
end) and chance nodes (never scored), solved subtrees.

**MC labels** (`mc-label --nn-backend inproc --playouts 8`, 8 trajectories of gen48 `shard4.jsonl`, 295
samples, the 20x11 mix, one stream; labels bit-identical base vs now, `ca63cab2c9af`): **186 forwards per
sample** (23 per playout; the memo never hits), per forward:

| mc-label, per forward | base | now |
|---|---|---|
| engine stepping the playout (almost all of it the move offerings' pathfinder) | 0.295M | 0.186M |
| encoder (path plane's pathfinder) | 0.247M (0.188M) | 0.071M (0.020M) |
| policy: legal set, priors gather, softmax, argmax | 0.060M | 0.033M |
| inference client | 0.052M | 0.042M |
| **batch-independent** | **0.65M (126 µs)** | **0.33M (64 µs)** |
| batcher, per batch | 0.19M | 0.19M |

**CPU per forward, generation / labelling = 0.68M / 0.33M ≈ 2.0** (base 1.7). Per sample: a generated
decision costs 471 forwards × 0.68M ≈ 320M instructions, a labelled sample 186 × 0.33M ≈ 62M (5× less
CPU, 2.5× fewer forwards). Before the cuts the path plane was labelling's largest single cost after the
engine's own pathfinding (22% vs 34%); the path memo (§1b) removed it exactly, so "a cheaper encode for
playouts" is done, with identical labels. What is left is the engine's pathfinding (35%) and the batcher.

**Streams.** GPU-free (null net, 4 logical CPUs, the same 96 games at 1 / 24 / 48 / 96 streams):
instructions per descent 0.3463M / 0.3464M / 0.3464M / 0.3465M — flat; kernel instructions per descent
0.003M → 0.009M (futex, page faults; 3% at 96); IPC 1.26 → 1.13 and task-clock per descent 70 → 80 µs
(SMT siblings and shared caches); peak RSS 1.24 / 1.96 / 2.75 GB at 24 / 48 / 96 streams, ~16 MB per
20x11 stream. In-process at 4 and 8 streams (2 lanes, the same 8 games): user instructions per forward
0.82M / 0.78M (mean batch 1.8 / 2.7 — the batcher's 0.2M per batch amortises as expected). **Not
explained:** task-clock per forward is 870 / 1106 µs there, while cycles:u+k per task-clock second say the
threads ran at an effective 1.3 / 0.9 GHz (4.0 GHz in every GPU-free arm). With the GPU in the process the
threads are scheduled without retiring instructions at speed — that, not instructions, is most of the
loop's "1.3 ms CPU per forward" (410% CPU at 3,150/s). Candidates: blocking paths in the NVIDIA driver,
SMT siblings busy with the loop's own 48-stream worker on the same 4 cores. Measure it first on a GH200
(work item 4: instructions, cycles and task-clock per forward per setting).

**Projection for one GH200 (72 Neoverse V2 cores).** Assumptions: one V2 core retires this branchy,
pointer-heavy code at about this box's single-core rate (5.2 instructions per ns; V2 is wider but clocks
~3.2–3.4 GHz; aarch64 instruction counts differ by ±15%) — range 0.7–1.2×; no SMT on Grace; batches of
≥64 make the batcher's per-batch cost negligible per sample; ~6 cores for batcher lanes, driver threads and
the hub client; the scheduled-but-idle overhead above does not exist there. Then generation needs ~130 µs
of CPU per forward → ~7.7k forwards/s per core → **~500k positions/s from 66 cores** (range ~300–700k;
~310k with the base binary). Against an H100 at ~40k/s fp32 and ~150–300k/s bf16, **the GPU binds, by
~10× at fp32 and ~2× at bf16**: the CPU side is not where the hours go unless the unexplained overhead is
real on Grace (it would cost up to 5×, i.e. ~100k/s, CPU-bound against bf16). Consequences: precision
(TF32/bf16 with its gate) and batch size are worth more than further search CPU cuts; spare cores can run
labelling streams, but labelling also needs the GPU (186 forwards per sample), so it fills GPU time only
where generation leaves some; to keep an H100 busy at ~300k/s with ~3 ms round trips takes ~1,000
concurrent streams (~16 GB of trees at 20x11, fine in 480 GB).

**How measured.** `scripts/perf_search_bench.sh` (plan 058 §7's reproducible benchmark, extended:
`CAP`, `SIZES` incl. `mix`, `NEXT_DRIVE`, `BACKEND=inproc|null`, `STREAMS`, `EVENTS`, `BIN`, `RECORD=1` for
an LBR profile split by `scripts/perf_buckets.py`, `AFTER_LOAD=1` to count only after libtorch's load and
graph capture, which cost 95–132G instructions and vary run to run), `scripts/perf_label_bench.sh` (its
mc-label twin, label hash), `scripts/exp080_cpu_matrix.sh` (the heuristic / null / in-process × 16x9 /
20x11 / mix matrix) and `scripts/exp080_table.py`; outputs in `runs/exp080/`. Pinned to one logical CPU,
nice 19, `deterministic_hash` + the fixed-`getrandom` shim (restarted per call in a torch build), cuDNN
deterministic, one lane: two runs agree to ~0.05% in instructions and exactly in trajectories. **IPC:**
from the heuristic and null arms (single pinned core, no GPU), user IPC 1.24–1.39 at an effective 3.8–4.0
GHz (cycles / task-clock), i.e. 5.2 ± 0.5 instructions per ns; their task-clock per descent agrees with
instructions / 5.2 within 3%. The in-process arms' own task-clock is not used for µs (see Streams).

## 1b. Where the search CPU went, and the cuts (exp080, 2026-10-22)

**Profile before** (`ef820ec`, LBR call graphs, null net, 20x11/7; inclusive shares of the instructions
after start-up): the pathfinder 48% — 29% filling `MoveAction`'s offerings for new nodes and 19% the
encoder recomputing **the same paths** for the path plane right after; of the pathfinder, the per-batch
lock-in pass over the whole capacity pitch (`fill_player_paths` self), SipHashing the risky batches'
probability keys, `get_intercepters` scanning every square of a pass's bounding box, and the
`get_player_at` calls. Then node teardown (`on_drop`/`detach`) 8%, SipHash (children maps, parents sets,
registry) 7%, `collect_non_path_actions` 5% (itertools' `cartesian_product` over the pitch), `GameState`
clones and moves along each descent edge 5%, the encoder's own conversion 6%.

**Cuts** (all on this branch; each keeps every game bit-identical — same trajectory hash on the
reproducible workloads, heuristic `37daa6507bb2` and null net `086a7a3e9c67` at 20x11/7, 20 games;
the real net through the in-process backend `1d15aaee67ff` / `ad5ddcb3dc70` / `a2ba9272e33a` on
20x11 / 16x9 / mix; MC labels `ca63cab2c9af`):

| commit | change | null net, 20x11 | heuristic, 20x11 |
|---|---|---|---|
| base `ef820ec` | | 640.9G | 493.4G |
| `3ee67bd` | pathfinder lock-in visits only the squares written since the last batch; `RiskySet` a short `Vec` keyed by the probability's bits (no SipHash); `FullPitch::iter_position` a nested `enumerate`; `collect_non_path_actions` skips empty squares; `encode` skips the division for planes of scale 1 | 556.2G (−13.2%) | 433.1G (−12.2%) |
| `9f6898d` | `PathFinder::fill_player_paths` keeps a one-entry per-thread memo keyed on a copy of every field it reads (`GameState::path_inputs`: board, info, ball, fielded players, team states, board dims): the encoder's path plane is served from the offerings `MoveAction` just computed. Debug builds recompute on every hit and assert equality; `path_memo_serves_only_exact_repeats` | 426.9G (−23.2%) | 434.6G (+0.4%) |
| `7e7ebf5` | `recon_mcts`: a node's state lives behind an `Arc`; a descent carries `Arc<State>` instead of cloning the state at every edge (`SearchTree::step` returns `Option<Arc<State>>`) | 382.8G (−10.3%) | 378.6G (−12.9%) |
| `bacde07` | the board lookups the pathfinder makes per neighbour inlined | 372.9G (−2.6%) | 368.8G (−2.6%) |
| `acca8b5` | `get_intercepters` tests the team's catchers against the pass corridor instead of scanning its bounding box (same set and order; `interception_predicate_matches_the_square_scan`) | 349.2G (−6.4%) | 350.4G (−5.0%) |
| **total** | | **−45.5%** | **−29.0%** |

Same ratio on 16x9 and the mix (null 0.54–0.55, heuristic 0.69–0.70); real net in-process −33% user
instructions per descent on all three boards, MC labels −38% per forward. `-C target-cpu=native` on top
(not committed, a build flag) is another −5.2% with the same games: build the Grace binaries with
`-C target-cpu=neoverse-v2`. Tests: engine, nn, mcts, play, ui (default capacity) and `recon_mcts` pass.

**Profile after** (null net, 20x11, inclusive): engine stepping of new nodes 38%, of it the move
offerings' pathfinder 35% (`expand_to` 10% self, the expansion fold 7%, `Arc<Node>` allocation and drop
~3%, `RiskySet` 2%); node teardown 11% (SipHash of `(action, parent)` pairs in the parents sets ~6%);
encoder 10% (the u8 → f32 conversion 5%); SipHash 9.5% in total; backprop 7%; `get_all_actions` 5%
(sorting the legal set 3.4%); tree drops 3.5%; selection 3.7%. With the real net: search 63%, encoder 7%,
client 7%, batcher thread 22% at batch 1.

**Next cuts that keep games identical** (not done; each gated on the trajectory hash):
1. *Pathfinder arena*: nodes in a per-thread reusable `Vec` with parent indices instead of `Arc` chains,
   the expansion loop specialised per action type. The pathfinder is ~35% of search instructions; ~−10–15%
   of the total if it halves. Insertion order decides ties (`is_better_than`), so it must be kept.
2. *Bulk teardown*: when a whole tree is discarded (anchor miss, `release_stale_tree`, the root move's
   dropped siblings), free it without unlinking every node from its children's parents sets and the
   registry. ~−5–8%.
3. *Encoder into the slot*: write f32 straight into the batch slot's canvas (skips the intermediate
   `Vec<f32>` and the staging copy, ~5% of client + encoder); further, ship u8 planes and divide on the
   device (an f32 division is the same IEEE operation there; needs `scripts/parity066.sh`).
4. `-C target-cpu` for the Grace build (above).

**Proposals that change games** (not done):
1. *A cheaper hasher* (FxHash/aHash) for the children maps, parents sets and registry: SipHash is ~9.5% of
   search instructions, ~−6–8%. It changes the children maps' iteration order and with it selection's tie
   breaks, so games differ (not worse in expectation). Do it with work item 2 (make selection and backup
   order-independent first; then the hasher is free).
2. *More streams per process* to raise the batch (the batcher costs ~0.2M instructions per batch, ~38 µs):
   not a search change, but batch composition changes the cuDNN algorithm per bucket and so the bits.

## 2. Work items, in order

1. **Measure the CPU side** (here, now): instructions and CPU-µs per descent and per forward at 20x11/7
   with the in-process backend and with an instant evaluator (`scripts/perf_search_bench.sh`, extended to
   20x11 and the C1 preset); split into search, encoder, batcher. A profile of where the remaining search
   CPU goes, and the next round of cuts (plan 058 §7 found −76% the first time).
2. **Search determinism** (exp079): backup order at equal depth makes a net disagree with itself at
   ~4–6% of decisions. Reproducibility across thousands of workers matters more there; fix it behind an
   identity gate (same games under the shipped preset or a measured, accepted change).
3. **Arm build and container:** aarch64 Rust (`cargo check --target aarch64-unknown-linux-gnu` here),
   libtorch for aarch64 + CUDA (the torch wheel's C++ API), the CUDA-headers script for aarch64, an
   Apptainer definition with Rust binaries, the venv and the trainer; rpaths.
4. **Inference tuning suite** (scripted, so the first pilot hours measure instead of fiddle): bucket sizes
   up to 512–1024, lanes, streams per process, processes per GPU, multiple canvases (padding waste across
   the size mix), fp32 / TF32 / bf16, `channels_last`, NUMA pinning; later zero-copy slots in coherent
   memory. Output: positions/s and CPU per forward per setting. Parity first (`scripts/parity066.sh` on
   the cluster), then a numerics gate for TF32/bf16 (value benches, net check, drives) — not bit parity.
5. **Orchestration:** the hub (and registry, web app for the community) on the persistent partition;
   workers as Slurm job arrays (one process per GPU, pinned to its superchip) dialling it; the trainer on
   its own GPU; corpora on the parallel file system; resume across walltime limits (the loop's markers
   already resume). Possibly tract workers on the CPU partition for labels if it is billed separately.
6. **Scale tests here with fake workers:** the hub with thousands of streams (websockets, zstd, label
   items of 32 samples → many small messages), `prepare` and the data loader at 10–100× today's corpus,
   multi-GPU training (DDP) if one H100 cannot keep up.
7. **Bug chasing before hours burn:** a 20x11/7 random-game fuzz at scale (panics, hash/eq invariants,
   kickoff table, 7-player setups); the probe findings (mirror asymmetry, rare-skill values, AV sign);
   full-size 26x15/11 fuzz and a tiny end-to-end loop for later.
8. **Experiment plan for the hours** (draft split): ~10% shakedown and benchmarks; ~20% one-step A/Bs run
   in parallel to choose the main configuration — net size (96x8 → 128x10 → 192x12), descents in
   generation (1000 vs 4000), label playouts (exp078 at 8/16/32), τ, value weight; ~60% the main 20x11
   run from the current champion; ~10% slack. Human-level evaluation is the user's community step.

## 3. The box meanwhile

The loop keeps running as a test bed (its strength is no longer the goal). The 20x11 contested re-screen
(gen48, 1000 descents) and the 16x9 one (250 descents) finish on the loop's hub. Box time goes to §2's measurements, builds,
fuzzing and scale tests.

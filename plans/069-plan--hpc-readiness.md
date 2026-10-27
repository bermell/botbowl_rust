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

## 0b. What the centre's docs say (Arrhenius, NAISS; the user, 2026-10-21) and what follows

| fact | consequence for us |
|---|---|
| SSH login with password + 2FA (`login.hpc.arrhenius.naiss.se`); rsync the same way | nothing on this box can drive the cluster unattended: **the loop must run entirely inside Slurm jobs**; transfers in and out are manual (a multiplexed SSH session, if allowed, makes them one 2FA per session) |
| GPU partition = GH200 (**aarch64**); CPU partition = AMD Turin (**x86_64**); a `fat` large-memory CPU partition | two architectures: Arm builds for GPU nodes, today's x86 binaries run on CPU nodes unchanged (tract workers there are an option if CPU time is cheap) |
| Lmod; GPU toolchain `GPU/buildenv-nvhpc/25.9-cu13.0` (CUDA 13.0) | our libtorch is torch 2.13 + cu126; don't depend on modules — **Apptainer container** with its own CUDA user space (driver must support it) |
| Apptainer supported; build inside `interactive -A <project>` | build the `.sif` on an Arm GPU node, not here (x86 + qemu is slow); possibly no internet there → `cargo vendor` + a wheelhouse prepared here |
| home 30 GiB / 1 M files; project storage `/nobackup/proj/disk/<project>` (**not backed up**; quota via `storagequota`) | venvs and target dirs live in the container or project storage; corpora have a retention policy; nets, registry and positions are copied back here (they are small) |
| Slurm: `-A`, `-p cpu|gpu|fat`, `-n`, `-c`, `-t`, `--gpus`; `srun`/`mpprun` | no MPI needed: **loop-in-a-job** — node 0 of a job runs the hub, the trainer and a worker; nodes 1..N run one worker per GPU (`srun`, `CUDA_VISIBLE_DEVICES`, pinned to the superchip) dialling node 0; the loop resumes from its markers in the next job (`sbatch --dependency=afterany`) |

More from the centre's docs (the user, 2026-10-23):

| fact | consequence for us |
|---|---|
| GPU node: 4x GH200, each 72 Arm cores + H100 with **96 GB HBM** + **128 GB LPDDR**; 4x Slingshot 200 Gb; **1.8 TB node-local NVMe** usable during a job; 382 nodes | one Slurm task per superchip (`-c 72`, one GPU each); worker model caches and (optionally) the trainer's prepared data on the NVMe (`LOCAL_SCRATCH`, `PREPARED_ROOT=local`) |
| CPU partition: 192 thin nodes, 2x EPYC 9755 Turin (128 cores per socket), 768 GB, 1.8 TB NVMe; 20 fat nodes 2x EPYC 9745, 3 TB | x86 tract workers are possible there (same commit, an x86 build at 20x11/7) |
| Lustre: disk tier 25 PB / 260 GiB/s; **flash tier 2 PB / 15M IOPS random 4K reads** | corpora and run dirs on the disk tier; training data (the loader mmaps `spatial.npy` and reads it at random) on the flash tier if quota allows |
| PCD (persistent) partition = **virtual machines** on 32 nodes of 2x EPYC 9555 (3 with an L40S), CEPH 2.5 PB, Ethernet; container services planned | a hub there must reach GPU nodes *and* mount project Lustre to write corpora; until both are confirmed, the hub lives in the loop job and PCD hosts at most the status/registry/web side |
| Slurm: `-A naiss<N>-gpu` / `-A naiss<N>-cpu` (separate accounts: the CPU partition is a separate allocation); `-p` default cpu; **`-t` default 2 h**, max unknown; jobs ≥ 10 min; arrays `-a`; `-C <feature>`; `sbatch --test-only`; `srun`/`mpprun`; their GPU example `-n 8 -c 72 --gpus 8 -p gpu` | the job shape is theirs (task per GPU); every loop phase must fit in one job, so the loop checks at each phase boundary against the last duration of that phase and stops the chain if one cannot fit; `submit.sh --test-only` |

Still unknown: the walltime maximum, whether a job can take 1 GPU (their example allocates in GPU units)
and how GPU-hours are billed (per GPU or per node), outbound internet from compute nodes, whether compute
nodes reach a PCD VM and whether a VM can mount Lustre, the NVMe's path in a job, SSH multiplexing
policy, project quota (incl. flash tier), allocation expiry.

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
   **Built 2026-10-22, verified as far as an x86 box can (`hpc/arrhenius/`, README there).**
   Decisions: the base is arm64 `python:3.12-bookworm` (pinned digest; gcc 12, git, glibc 2.36,
   Python 3.12), **not** an NGC or `nvidia/cuda` image — the CUDA user space is the torch wheel's
   own, as on this box: **torch 2.13.0+cu126, the exact `train/uv.lock` build** (the cu126 index
   publishes `manylinux_2_28_aarch64` wheels and the universal lock already pins them), CUDA 12.6
   runtime, cuDNN 9.10.2.21, NCCL 2.29.3; its `libtorch_cuda.so` carries sm_80 + sm_90 SASS. The
   driver is the host's (`--nv`); a 12.6 user space under the cu13.0-era driver (≥ 580) needs no
   forward-compat package. Binaries in the image: `botbowl-ui`/`botbowl-worker` (torch), `botbowl-hub`,
   `prepare` at 20x11/7, `-C target-cpu=neoverse-v2`; the source is a shallow clone, so the git
   provenance stamp works inside the container (system `safe.directory '*'`). Offline:
   `make_bundles.sh` fetches the vendored crates of both workspaces (51 MB tarball), the lock's
   aarch64 wheels with a hash-checked `requirements.txt` (2.9 GB), the nvcc/cccl header wheels, the
   Rust 1.98.0 installer and the base as a docker-archive (4.1 GB in all), so neither the image build
   nor a job needs internet; `build_image.sh` builds HEAD on an Arm node. Code: `build.rs` takes
   `BOTBOWL_CUDA_NVCC_INCLUDE` and accepts crt/ and nv/target from separate wheel dirs;
   `fetch_cuda_headers.sh` gained offline/cross-arch/dest knobs. Verified here: aarch64 check of
   both workspaces (nothing x86-specific found: no intrinsics, asm or arch cfgs; `perf stat -e
   instructions:u` works on Arm, nothing uses LBR), release cross builds of all four binaries with
   the shim linked against the aarch64 wheel (`cross_check.sh`, Debian's cross gcc 12 in docker, no
   qemu), offline vendored builds with an empty `CARGO_HOME`. **Left for the cluster:** the
   `apptainer build` and its `%test`, torch on the H100, `parity066.sh`/`gatec066.sh` there, the
   first-login checklist (README), and the 64K-page question. Arm results are not expected to be
   bit-identical to x86 (glibc libm FMA paths, tract kernels): determinism checks compare Arm with Arm.
4. **Inference tuning suite** (scripted, so the first pilot hours measure instead of fiddle): bucket sizes
   up to 512–1024, lanes, streams per process, processes per GPU, multiple canvases (padding waste across
   the size mix), fp32 / TF32 / bf16, `channels_last`, NUMA pinning; later zero-copy slots in coherent
   memory. Output: positions/s and CPU per forward per setting. Parity first (`scripts/parity066.sh` on
   the cluster), then a numerics gate for TF32/bf16 (value benches, net check, drives) — not bit parity.
   **Smoke-tested on the 1060 2026-10-23** (outputs `runs/hpc_tune/{smoke,parity,precision_gate}/`,
   registry exp082): build clean against torch 2.13; `parity066.sh` at the default precision **0 differing
   bits** in all ten det/prod comparisons (20,000 states); every smoke row ran (micro, queue, gen 1-24
   streams, 2 processes split, label, mix, an fp16 gen row). Found and fixed on the way: the first fp16
   load failed its canary (gen49 fp16 misses the 1e-3 fixture by **0.042** on its worst policy logit),
   so reduced-precision tolerances are now 0.1 (fp16/TF32) and 0.5 (bf16); bf16's fused
   `cudnn_convolution_relu` has no cuDNN engine on Pascal (the gate falls back to the unoptimised
   module there; fine on Hopper). Numerics gate on gen49's held-out shard 4 (12,849 samples, real
   1060 kernels): fp16 |Δv| 0.00023 mean / 0.0052 max, top-1 changes 0.19% (all 24 at near-ties),
   value MSE unchanged; bf16 0.0028 / 0.018, 0.78%; fp16-cl as fp16; value bench (v9_20x11_gen49_val,
   3,023 states, through the real queue) fp16 vs fp32 dMSE 0.0000 ± 0.0000, RMS 0.157 both. Drives not
   run here (the command is printed by the gate).
   **exp080's "scheduled but idle" anomaly is CPU clock frequency** (per-thread counters plus the clock
   of the CPU under each running thread): every thread class of the in-process worker — search,
   batchers, CUDA driver threads alike — runs at the same low cycles/on-CPU-time, and that equals
   `scaling_cur_freq` of the cores they are on (24 streams, ~2.8 cores busy: cores at 0.85-1.15 GHz,
   search threads at 1.0-1.2 GHz effective; 4 streams: cores ~2.0 GHz, threads ~2.0). It is not the
   GPU (a pure-CPU spinner in another process reads 4.0-4.1 GHz with the GPU at 87%, and four spinners
   at once read 4.0 GHz; no thermal throttling), and not missing cycles: the box's HWP (`intel_pstate`
   active, governor `powersave`, EPP `balance_performance`, one clock for all cores on this i7-7700)
   picks a low clock for this bursty futex-heavy load, and a spinner pegging one core lifts the whole
   package (gen threads 1.1 → 2.5 GHz within the same run). Runs vary with it (8 streams: 0.94 GHz /
   966 µs CPU per forward in one run, 3.3 GHz / 327 µs in the next). So on this box the search's CPU
   cost is ~3x inflated by clock policy: `cpupower frequency-set -g performance` (or EPP
   `performance`, root) is the fix to try; on Grace check `scaling_governor` first — the suite now
   records the cpufreq policy in `meta.json` and `cpu_freq_ghz` (the clock under the running threads)
   per row, next to per-class `eff_ghz`.
   **Written 2026-10-23 (`hpc/tune/`, README there):**
   `hpc/tune/run_suite.sh SPEC OUT` builds `botbowl-ui` (torch) and `examples/inproc_bench` and runs
   `hpc/tune/tune.py` (stdlib Python ≥ 3.11, runs in the image) over a TOML spec: `micro` (graph replay
   per bucket, no queue, no search: positions/s vs batch 1-1024, precision, channels-last, lanes),
   `queue` (N clients on the batch queue, no search), `gen` (the loop's generation: C1 preset, 1000
   descents, random-start `--next-drive`, the 20x11 mix), `label` (`mc-label`, 8 playouts) and `mix`
   (item 4b: G generation + L label streams on one GPU); swept: streams, processes per GPU, lanes,
   bucket lists, precision, channels-last, pinning (`none`/`node`/`split`, numactl or taskset on the
   GPU's NUMA node), MPS. Per row, over a window after warm-up: positions/s, mean batch, padding,
   device busy (the queue's counters); CPU per forward split batcher / CUDA threads / search from
   per-thread schedstat, run-queue wait and context switches; user/kernel instructions and cycles per
   forward from per-thread `perf_event_open` counters (no `perf` binary needed) and `eff_ghz` =
   cycles / on-CPU time — the §1 anomaly's signature; GPU util/memory/power/SM clock. Seeded,
   commit-stamped, resumable (`results.csv`, `results.md`, `meta.json`); `--shard K/4` splits a spec
   over a node's four superchips (`hpc/tune/pilot.sbatch`). `gh200.toml` is the pilot (~2.5 h on one
   GPU, ~40 min per node; up to 2,048 streams and buckets to 1,024 for 96 GB HBM), `smoke.toml` the
   1060 version. New opt-in backend knobs (unset = bit-identical to today): `BLOOD_NN_BUCKETS`,
   `BLOOD_NN_PRECISION=fp32|tf32|fp16|bf16`, `BLOOD_NN_CHANNELS_LAST`, `BLOOD_NN_OPTIMIZE`,
   `BLOOD_NN_CANARY_TOL`; fp16/bf16/-cl are export variants (`bbnn.export_ts --dtype fp16
   [--channels-last]` → `NET.fp16.ts`: BatchNorm folded in fp32, cast, fp32 in/out), so the shim only
   gained the cuBLAS TF32 flag and a variant tag read. **Note: unset precision on Hopper means TF32
   convolutions** (torch's `cudnn.allow_tf32` default) — the suite's `default` row shows whether the
   1e-3 load canary survives that. Numerics gate: `hpc/tune/precision_gate.sh` (exp081's metrics on the
   GPU's kernels on held-out samples, value benches through the real queue per precision, paired drives
   variant-bot vs fp32-bot in one process). Padding: `hpc/tune/canvas_waste.py` (per board of the mix,
   and the best 2-4 canvas sets; written to `OUT/canvas.md`). Not built: multiple canvases, zero-copy
   slots in coherent memory, u8 planes on the wire (H2D at ~300k positions/s is ~21 GB/s of fp32
   input, fine over NVLink-C2C).
4b. **Generation and labelling share each GPU** (the user, 2026-10-21: "if CPU becomes the bottleneck in
   generation we should do the bare policy labeling in parallel; and if there's spare GPU we can increase
   the number of rollouts up from 8 almost for free"). Labelling is bare-policy playouts — a forward per
   step with no tree — so its CPU per forward is a fraction of the search's; it is the natural filler for
   a GPU that generation's CPU cannot keep busy. Two pieces: (a) **stream labels within a generation** —
   label items for a trajectory are queued as soon as it arrives (same generator net, so no lag), and the
   labels finish shortly after the last game instead of a separate phase (plan 062's "pipelining" step);
   (b) per GPU, a worker runs G generation streams (enough to saturate the cores) plus L label streams
   (enough to fill the GPU), both tuned in the suite (item 4). Playouts per sample then follow the spare
   GPU: exp078 (8 vs 16, plan 067's exact top-up) becomes an 8/16/32 sweep in the A/B phase. Needs the
   labelling CPU per forward (item 1) to size G and L.
5. **Orchestration:** the hub (and registry, web app for the community) on the persistent partition;
   workers as Slurm job arrays (one process per GPU, pinned to its superchip) dialling it; the trainer on
   its own GPU; corpora on the parallel file system; resume across walltime limits (the loop's markers
   already resume). Possibly tract workers on the CPU partition for labels if it is billed separately.
   **Built 2026-10-23 (`hpc/arrhenius/slurm/`, README there), checked by dry run only.** Loop in a
   job: one srun step, a task per GH200 (`-c 72`, a GPU each, the centre's own example); task 0 runs
   `train_loop.sh` in the image (hub, trainer, its per-phase worker, as on the box), every other task a
   long-lived `botbowl-worker` dialling it; `workers.sbatch` adds worker-only jobs (they follow the
   loop job's `hub.addr`). Chain: each job queues its successor (`afterany`), or all up front; a
   `chain.state` file says continue/stop and why; `submit.sh stop` is the loop's STOP file. New
   train_loop knobs, all default-off: `BOTBOWL_BIN_DIR` (prebuilt binaries, no cargo; also
   net_check.sh/value_bench.sh), `HUB_HOST`/`HUB_EXTERNAL`/`HUB_ALLOWED_COMMITS`, `LOCAL_WORKER`,
   `LOOP_DEADLINE` + `STOP_MARGIN_SECS` + `STOP_PHASE_ESTIMATE` (exit 75 at a boundary the next
   phase cannot finish before, 76 when a phase is longer than a whole job), `EXIT_ON_SIGNAL`,
   `PREPARED_ROOT`. The persistent-hub variant is knobs only (`HUB_EXTERNAL=1` + worker arrays) and
   needs PCD → compute reachability and Lustre on the VM. **Gap:** a generate phase cut by the
   walltime restarts from scratch (the hub has no restart recovery, plan 041 phase 3), so with 2 h
   jobs `GAMES_PER_SHARD` must keep a generate phase well inside one job.
6. **Scale tests here with fake workers:** the hub with thousands of streams (websockets, zstd, label
   items of 32 samples → many small messages), `prepare` and the data loader at 10–100× today's corpus,
   multi-GPU training (DDP) if one H100 cannot keep up.
   **Tools (2026-10-23):** `botbowl-fake-worker` (a bin of botbowl-worker: the real handshake and
   frames, canned corpus lines as results, sleeps for games; many connections per process; reports
   dispatch latency = last result sent → next task, connect latency, MB/s), `scripts/hub_scale.sh` +
   `hub_scale.py` (stages of generate/label at 100/1,000/4,000 streams, realistic and compressed
   time, hub CPU/RSS sampled, `summary.md`), `scripts/prepare_scale.sh` (prepare and the loader at
   R× a corpus).
   **Results (2026-10-23, this 8-core box, shared with a GPU tuning run, so absolute CPU is approximate;
   canned = 48 heuristic 20x11 games, ~370 KB a line raw / ~15 KB zstd, ~2x a 16x9 NN line):**

   | stage (streams x s/unit) | achieved / offered | hub CPU (% of a core) | hub RSS | dispatch latency p50 / p99 |
   |---|---|---|---|---|
   | generate 100 x 20 s (5 games/s) | 1.00 | 4 | 17 MB | 12 / 40 ms |
   | generate 992 x 20 s (50 games/s) | 0.99 | 26 | 33 MB | 6 / 43 ms |
   | generate 4,030 x 20 s (200 games/s), before the fix | 0.99 | 60 | 56 MB | 7 / **1,184 ms** (max 2.1 s) |
   | same, after (`TrajectoryPayload`, 7d7f487) | 0.99 | 73 | 49 MB | 5 / 38 ms |
   | generate 4,030 x 4 s (1,000 games/s): before | **overload** (~600 games/s, latency unbounded, API timed out) | | | |
   | same, after | 1.00 | 252 | 66 MB | 6 / 330 ms |
   | generate 4,030 x 1 s (4,000 games/s), after | 0.31 (= ~1,250 games/s, the ceiling here) | 282 | 99 MB | 6.8 / 31 s |
   | generate 1,000 connections x 1 stream x 10 s | 0.98 | 54 | 172 MB | 6 / 29 ms (after) |
   | label 100 x 15 s | 0.93 (tail) | 2 | 97 MB | 0.7 / 1.5 ms |
   | label 992 x 15 s (66 items/s) | 1.00 | 12 | 544 MB | 0.2 / 0.9 ms |
   | label 4,030 x 15 s (270 items/s; 10.8 GB input) | 1.00 | 51 | 1.96 GB | 0.5 / 1.2 ms |
   | label 4,030 x 4 s (1,000 items/s) | 0.98 | 144 | 1.91 GB | 0.2 / 1.1 ms |

   Reading: (1) **The one clear bottleneck was per-result work under the hub's single lock**: zstd inflate
   plus the `drive_summary` scan of each ~370 KB line, ~2 ms a game. Fixed (7d7f487): decoded in the
   websocket task, the lock only writes; ceiling here ~600 → ~1,250 games/s. A realistic cluster load
   (60 s games: 4,000 streams = 67 games/s; 16,000 = 270/s) is now well inside it. (2) Labels are cheap
   per message (~0.5 ms p50 at 1,000 items/s); their cost is **memory and submit time**: the hub holds
   every input line zstd'd, ~0.18 GB RSS per GB of corpus, and `job label` parses the corpus at submit,
   ~1.6 s per GB (CLI submit timeout 600 s → ~350 GB per job). At 100x today's generation (~100 GB of
   JSON) that is ~18 GB on node 0 and ~3 min of submit. (3) Many thin connections (1,000 x 1 stream)
   cost nothing visible. (4) **Residual, not fixed:** shard writes (`write_all` + `flush` per game) are
   still under the lock — fine on local disk, but on Lustre a slow write stalls every result and the
   API (`job --wait` polls, the status page) behind it; if the cluster shows it, give each generate
   shard a writer thread as label shards already have (`label::ShardWrite`). The API shares the lock,
   so an overloaded hub answers its own `/api/jobs` late (`job --wait` retries, harmless).

   `prepare` and the loader (`scripts/prepare_scale.sh`, a current-schema 1,800-game heuristic 20x11-mix
   corpus, 820 MB, 76k kept samples, repeated R times):

   | R | prepare | prepare RSS | output | loader open RSS | loader samples/s, 0 / 4 workers |
   |---|---|---|---|---|---|
   | 1 | 6.3 s | 12 MB | 1.0 GB | 549 MB | 13.9k / 25.3k |
   | 3 | 18.9 s | 16 MB | 3.1 GB | 721 MB | 13.6k / 25.2k |
   | 10 | 64 s | 35 MB | 10.3 GB | 1.3 GB | **3.6k / 15.4k** |

   `prepare` is linear and streams (~12k samples/s, flat memory): 100x is ~11 min and ~100 GB of output
   — no problem. The loader is the one to watch: everything but `spatial.npy` is read into RAM
   (~85 MB per 76k samples → ~9 GB at 100x, per process; forked workers share it copy-on-write), and
   `spatial.npy` is mmapped and read at random, so throughput holds only while it fits the page cache:
   at 10x (10 GB on a 15 GB box) it fell 4x with no workers and 40% with four. On a GH200 node (480 GB
   LPDDR) 100x fits in memory once read; from Lustre the first epoch is random 13 KB reads, which is
   what the flash tier (or node-local NVMe, `PREPARED_ROOT=local`) is for. Loader workers are needed
   either way: 25k samples/s from cache with four may not keep an H100 training a 96x8 net busy —
   measure the trainer's own samples/s on the H100 before choosing `--loader-workers`.
7. **Bug chasing before hours burn:** a 20x11/7 random-game fuzz at scale (panics, hash/eq invariants,
   kickoff table, 7-player setups); the probe findings (mirror asymmetry, rare-skill values, AV sign);
   full-size 26x15/11 fuzz and a tiny end-to-end loop for later.

   **The S3 fuzz (exp085, 2026-10-27): `botbowl-ui fuzz`, `scripts/fuzz_s3.sh`.** Seeded, parallel
   random games, each a pure function of its seed: the board from the loop's size mix at the binary's
   capacity (`--boards loop`; `full` or a `--board-sizes` list), a full game from the coin toss or a
   random-start drive played on to the end of the game (so `--next-drive` continuations, half time and
   every later kick-off), a uniform / type-uniform / scripted policy (optionally a small MCTS search on
   the loop's `chance_all_partial_gen` preset, or a net's bare policy via tract), fair or flattened dice
   (pass/fail 50/50, 2D6 uniform, so every kick-off result and fumble comes up), `RegisterRolls` with
   the fuzzer rolling or `RollDice` + `GameState::step` (the production path `mc-label` replays),
   "wide" rosters (random characteristics, skills from every `Skill`, random re-rolls, bribes, fan
   factor, cheerleaders, coaches), "tag-team" rosters full of Throw / Kick Team-mate and Right Stuff
   playing hidden (human-only) actions 30% of the time, "brutal" rosters (soft armour, Mighty Blow,
   Claws, Dirty Player) that run teams out of players, and the jump toggle. Built with the new `fuzz`
   cargo profile (release + debug assertions + overflow checks). Checked after **every micro-step**
   (classes in `summary.json`): panics; step states; the mask (a decision lists something, every
   listed action is legal and has a policy cell, no hidden type listed, a deciding team that matches
   `turn_team` / `setup_team`, `search_actions` non-empty, a subset and the same on a clone; every
   4th decision every listed and hidden action applied to a clone); the encoder (shape, finite, cells
   in range and round-tripping, a JSON round trip encodes the same, the mirror encodes identically and
   the y-flip as the flipped planes at a `Turn` decision); Hash/PartialEq (clone every step; a JSON
   round trip and a double mirror / y-flip every 8th decision); roster conservation, board/player
   agreement, players and the ball inside the frame; at a `Turn` decision nobody out of bounds, at most
   `team_size` a side, a standing carrier not in the end zone they attack (a missed touchdown), the
   ball on the pitch and not under a player, re-rolls at most the half's start; once-per-turn
   declarations, Secure the Ball's conditions, nobody `used` as a team turn starts; legal setups at the
   kick; half / turn markers and scores monotone (Time-out's step back excepted), winner = score;
   the search's chance children at every 4th roll (shipped and the loop's roll models: compatible,
   probabilities summing to 1, each applying); termination (400k micro-steps); and replay of 25% of
   games from seed + inputs, every state compared by hash. `--only-seed S [--dump-from STEP]` replays
   one failure and writes its states as JSON lines for a test.

   | capacity | boards | games | micro-steps | decisions | rolls | replays | failures |
   |---|---|---|---|---|---|---|---|
   | 20x11/7 | loop mix (12x6/3 .. 20x11/7), first check set (56c3f45), 6 threads, 40 min | 57,246 | 14.73M | 11.49M | 3.24M | 14,303 | 0 |
   | 20x11/7 | loop mix, full check set (85f7ade), MCTS on 1% of decisions, 25 min | 28,624 | 7.36M | 5.73M | 1.63M | 7,024 | 0 |
   | 20x11/7 | loop mix, gen58's bare policy on 80% of decisions (tract), 20 min | 1,080 | 0.48M | 0.37M | 0.11M | 247 | 0 |
   | 16x9/6 | loop mix at 16x9 capacity, 10 min | 8,351 | 1.91M | 1.48M | 0.42M | 2,180 | 0 |
   | 26x15/11 | full board only, 15 min | 4,398 | 1.30M | 1.03M | 0.27M | 1,102 | 0 |
   | 20x11/7 | `roll-census` replay of the loop's S3 corpora gen56-59, fuzz profile | 8,590 trajectories | | 497,664 | | 8,590 | 0 diverged |

   Coverage over the two 20x11 runs, as micro-steps with the procedure on top of the stack: 68.5k
   kick-off table rolls (every result; Charge 15k, Quick
   Snap 34k, Solid Defence 18k, High Kick 8k, Pitch Invasion 10k, Dodgy Snack 10k, Sweltering Heat
   1.8k), Ejection (bribe offers) 5.5k, Stalling 13.9k, Secure the Ball 11.8k, jumps 99.7k, Throw /
   Kick Team-mate 18.6k (4.4k landings), pass-to-square 1.4k, interceptions 0.6k, MA <= 2 stand-ups
   18k, 95k touchdowns, 30k kicks with a side short of `team_size` and 191 with a side fielding
   nobody. **No engine bug found**: every failure the first runs reported was a harness assumption,
   now encoded as a rule of the checks (a player pushed into the crowd stands on the border square
   while `Push` resolves the follow-up and crowd injury; a Time-out moves both markers back in one
   micro-step; the side not to move keeps its `used` flags until its turn; a declared Throw
   Team-mate asks only hidden decisions). Notes, not bugs: `GameState::get_player(id)` returns a
   `Result` but panics on an id past the roster array; the search's "pruning emptied the action
   list" fallback (a debug-build warning in `dynamics.rs`) fires at Blitz / Pass / Hand-off
   activations whose pruning leaves nothing (P2 with an unreachable loose ball, P5 with no reachable
   square next to a standing opponent, P1 before any move) — by design, but the search then expands
   every raw move there. For the cluster's CPU partition: one `scripts/fuzz_s3.sh` task per node,
   `THREADS` = its cores, a disjoint `SEED` range per task.
8. **Experiment plan for the hours** (draft split): ~10% shakedown and benchmarks; ~20% one-step A/Bs run
   in parallel to choose the main configuration — net size (96x8 → 128x10 → 192x12), descents in
   generation (1000 vs 4000), label playouts (exp078 at 8/16/32), τ, value weight; ~60% the main 20x11
   run from the current champion; ~10% slack. Human-level evaluation is the user's community step.

## 3. The box meanwhile

The loop keeps running as a test bed (its strength is no longer the goal). The 20x11 contested re-screen
(gen48, 1000 descents) and the 16x9 one (250 descents) finish on the loop's hub. Box time goes to §2's measurements, builds,
fuzzing and scale tests.

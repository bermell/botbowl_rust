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

## 1. The cost model flips (estimates — §2 replaces them with measurements)

On this box generation is bound by GPU arithmetic (GTX 1060: 625 µs of kernels per position, 3,150
positions/s in-process, plan 066). On a GH200 the balance is CPU against GPU:

| | estimate | basis |
|---|---|---|
| net cost (96x8 + heads, 13x22 canvas) | ~0.85 GFLOP / position | 16 3x3 convs at 96 ch on 286 cells + stem and heads |
| H100, fp32 | ~40k positions/s | ~67 TFLOPS fp32 at ~50% |
| H100, TF32 / bf16 tensor cores | ~150–300k positions/s | numerics change; needs its own gate (§3) |
| CPU per forward on this box | ~1.3 ms (410% CPU at 3,150/s) | search + encoder (path planes ~50 µs) + batcher |
| 72 Grace cores | ~55–110k positions/s | a Neoverse V2 core vs an i7-7700 hyperthread: to measure |

So at fp32 one GPU and its 72 cores are roughly balanced; with tensor cores the CPU binds. Consequences:
**search CPU per descent is worth GPU-hours** (every % cut is throughput), and **a bigger net or more
descents may be nearly free** while the GPU waits on the CPU. Both are decided by two numbers we must
have before the pilot: CPU µs per descent and per forward (here, by instruction counts that a busy box
cannot distort, then on Arm), and the net's positions/s per batch size and precision on an H100.

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
8. **Experiment plan for the hours** (draft split): ~10% shakedown and benchmarks; ~20% one-step A/Bs run
   in parallel to choose the main configuration — net size (96x8 → 128x10 → 192x12), descents in
   generation (1000 vs 4000), label playouts (exp078 at 8/16/32), τ, value weight; ~60% the main 20x11
   run from the current champion; ~10% slack. Human-level evaluation is the user's community step.

## 3. The box meanwhile

The loop keeps running as a test bed (its strength is no longer the goal). The 20x11 contested re-screen
(gen48, 1000 descents) and the 16x9 one (250 descents) finish on the loop's hub. Box time goes to §2's measurements, builds,
fuzzing and scale tests.

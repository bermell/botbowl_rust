# Plan 071 — the Grace Hopper hours: a from-scratch 20x11/7 run, and the experiments around it

**Status:** draft 2026-10-27, not started. Replaces plan 069 item 8's draft split. Nothing here has
spent a GPU-hour; every throughput number is an estimate until the pilot (§1.3) replaces it.
Experiment numbers are assigned in the registry when each piece launches.

**The user's decisions (2026-10-27):**
- the cluster run starts **from random weights** ("a good time to start from scratch to see a proper
  training curve without all the hiccups we've had along the way with this network"); the box's net
  is not uploaded as a start or a teacher;
- target **20x11/7** until it plays above human level (the user rallies the community to judge), then
  full size (26x15/11) if hours remain;
- judge on **paired contested drives** (SPRT or fixed pairs), nothing else head to head;
- allocation **≥ 5,000 GPU-hours**, likely 50,000 after; one GPU = one GH200 superchip (H100 + 72
  Neoverse V2 cores), 4 per node (plan 069 §0/0b); the engine is on the S3 rules, NN schema v10.

## 0. Summary

1. **Shakedown (~200 GPU-h):** image, parity, the tuning pilot (`hpc/tune/gh200.toml`), the fp16
   gate, trainer calibration, a 1-node end-to-end loop with a deadline exit, S3 drive sets and the
   gen58 yardstick. Fuzz runs on the CPU partition (no GPU-hours).
2. **Bootstrap (~300 GPU-h):** gen 0 played by a random 96x8+heads net with formation setups, raw
   outcome labels, and generations that grow from 8k to 16k games on one node. A second arm at 128x10+heads runs
   side by side on its own node, so the net size is chosen on Elo per GPU-hour, not on a guess.
3. **A/B program (~750 GPU-h):** short runs and single generation steps branched off the bootstrap
   and the main run. The net size and the generation shape are settled before the main run; the
   rest (descents, playouts, τ and value weight, Gumbel, window) run alongside it.
4. **Main run (~3,250 GPU-h):** 4 nodes, ~1 h per generation, about 200 generations of ~50k games
   each (at the central throughput estimate). The net grows once (KataGo style, through the shadow
   lineage) when the curve flattens.
5. **Slack (~500 GPU-h).**
6. **The curve:** an Elo fit over fixed-pair drives against the previous generation and a geometric
   ladder of the run's own frozen generations, with gen58 (the box's S3 net) as a yardstick only.
   About 0.5 GPU-h per generation.

## 1. Budget and cost model

### 1.1 Unit costs (what is measured, what is estimated)

| quantity | value | source |
|---|---|---|
| forwards per 20x11-mix game, generation (C1, 1000 descents, random start + `--next-drive`) | **49.7k** | plan 065 §5 (CPU replay of the loop's mix, ±20%) |
| samples per game | **108.7** (1.8 drive records) | plan 065 §5; gen49 corpus 128k / 1,200 games (DATA.md) |
| label forwards per generated forward | **r = 0.39 × playouts/8** | exp086 (0.39 measured; exp080 186/471 = 0.395) |
| forwards per game incl. labels | 69k (8 playouts), 88k (16), 127k (32) | the two rows above |
| CPU per forward, generation / labelling | 130 µs / 64 µs per box core | exp080; ratio 1.9 in exp086 |
| CPU ceiling per GH200 (66 cores for streams) | ~500k positions/s (300–700k) | plan 069 §1 projection |
| H100 positions/s, 96x8+heads on the 13x22 canvas | **P ≈ 40k (fp32), ~100k (TF32 default), 150–300k (fp16)** | plan 069 §1 estimate; **unmeasured** |
| net cost relative to 96x8+heads | 64x6 ≈ 0.5×, 128x10+heads ≈ 2.2×, 192x12+heads ≈ 6× | exp072 (96x8 = 2.0× 64x6 per sample on the GPU); FLOPs ∝ width² × blocks |
| 4000 descents instead of 1000 | ~3.2× per game (labels unchanged per sample) | forwards per decision ∝ descents (exp080: 0.61 per descent) |
| trainer samples/s, 96x8 on one H100 | **R ≈ 15k** (planning figure) | loader 25k/s with 4 workers from cache (exp083); trainer itself **unmeasured** |
| bytes per sample: JSONL / prepared / zstd'd JSONL | ~10.8 KB / ~13 KB / ~0.45 KB | exp083 (820 MB per 76k samples, prepared 1.0 GB; ~25× zstd) |

The GPU binds generation at every precision (plan 069 §1: CPU ceiling ~500k/s vs ≤ 300k/s on the
GPU), unless exp080's low-clock effect shows up on Grace. exp082 traced that effect to the box's
clock policy, so check `scaling_governor` first.

### 1.2 Converting hours

With P the H100's positions/s, one GPU-hour plays `3600 P / (49.7k × (1 + 0.39 p/8))` games
(p = label playouts).

| per GPU-hour (96x8+heads, 1000 descents, 8 playouts) | P = 40k | **P = 100k** | P = 250k |
|---|---|---|---|
| games | 2,080 | **5,200** | 13,000 |
| samples | 227k | **566k** | 1.41M |

At P = 100k one GPU-hour is ~4.3 box-generations of games (1,200 per generation). All 5,000 hours
would be ~26M games at the central estimate. **What decides the number of generations is the wall time
per generation, not P.** A generate phase needs ≥ ~40 min so that the tail (the longest game, ~137k
forwards ≈ 8–10 min per stream) stays near 10%. Training takes another ~15–25 min on node 0's GPU.
At about 1 h per generation, a run on N GPUs does ~N GPU-h per generation, and P only sets how many
games fit into the 40 minutes.

| main-run shape | GPU-h per gen | gens in 3,250 GPU-h | wall | games per gen at P = 40k / 100k / 250k |
|---|---|---|---|---|
| 2 nodes (8 GPUs) | ~8 | ~400 | ~17 d | 10k / 26k / 65k |
| **4 nodes (16 GPUs)** | ~16 | **~200** | **~8.5 d** | **22k / 52k / 130k** |
| 8 nodes (32 GPUs) | ~32 | ~100 | ~4.2 d | 44k / 104k / 260k |

Efficiency η (busy GPU time over billed time) is assumed at **0.75**: the generation tail ~10%, job
start-up and model fetch ~3%, failures ~5%, and ~15% of the worker GPUs idle while node 0 trains.
That last item is the largest, and it is recoverable (§5.1, R5).

**Training is the throughput risk at high P.** The time to train one generation, as a fraction of its
generate time, is `ρ · N_gen · s / R`, where ρ is the sample passes per generated sample, N_gen the
generating GPUs and s the samples/s per GPU. At P = 100k, 15 generating GPUs, R = 15k and ρ = 3,
that is 0.47: 20 min of training per 40 min of generation. At P = 250k it is 1.2, and the trainer
binds. The levers, in order: bf16 autocast and a larger batch (the trainer runs fp32 at batch 32
today), more loader workers (72 cores), ρ (the window row of §4), DDP over node 0's four GPUs
(plan 069 item 6: not built).

### 1.3 What the pilot changes

| pilot measurement (shakedown, §2.1) | changes |
|---|---|
| P per precision, streams per GPU, mean batch (`gh200.toml` micro/queue/gen) | games per generation, streams per worker (`WORKER_STREAMS`), every games figure here |
| CPU per forward and `eff_ghz` on Grace (gen rows) | whether CPU or GPU binds; if CPU, extra label playouts are nearly free (plan 069 item 4b) and the playouts row moves up |
| `mix` worker rows (G game + L label streams) | `WORKER_LABEL_STREAMS`; the label tail |
| fp16 numerics gate + fp16-vs-fp32 drives (`precision_gate.sh`) | whether generation runs fp16 (~2–3× P over TF32) |
| trainer samples/s at batch 256/1024, bf16, loader workers | ρ, window, whether node 0 needs DDP |
| a 1-node end-to-end loop (2 h, deadline exit, resume) | η; the phase durations behind `STOP_PHASE_ESTIMATE` |
| walltime max, billing per GPU or per node, Lustre/NVMe quotas | job chain length, fleet shape, corpus retention (§5.3) |

### 1.4 The 5,000 hours

| phase | GPU-h | share | what it buys (P = 100k) | gate to the next phase |
|---|---|---|---|---|
| 0. shakedown and benchmarks | 200 | 4% | pilot, gates, smoke loop, S3 sets, yardstick numbers | every item in §2.1 green |
| 1. bootstrap, two net sizes in parallel | 300 | 6% | ~13 generations per arm on 1 node each, ~160k games for the 96x8h arm | the net-size read-out (§3.4) |
| 2. A/B program | 750 | 15% | ~25 short-run arms or ~50 one-step arms (§4) | adoption at generation boundaries |
| 3. main run | 3,250 | 65% | ~200 generations, ~10–13M games, ~1.1–1.4B samples (η 0.75) | stop rules (§3.6) |
| 4. slack | 500 | 10% | reruns, failures, a second capacity step, support for the community evaluation | — |

Unspent A/B hours go to the main run; an overrun in shakedown comes out of slack.

### 1.5 From 5k to 50k

The 5k plan is self-contained. If 45k more hours arrive:

| use | GPU-h | why |
|---|---|---|
| continue the 20x11 main run with the next capacity step (192x12+heads, ~6× per forward) and 4000 descents if §4 says so | 15–20k | the scaling bet: the curve flattening at 128x10 says capacity, not data |
| full size 26x15/11, warm-started from the best 20x11 net (upward transfer is nearly free: plan 042 E0, plan 065) | 15–20k | the user's second target; only after the community has judged 20x11 |
| a replicate of the from-scratch bootstrap (another seed), for the curve's run-to-run spread | 1–2k | one run's curve is one sample |
| a wider A/B program on the scaled-up configuration | 3–5k | the 5k-hour A/Bs were run on a smaller net |
| slack | ~5k | |

Full size needs its own build (26x15/11 capacity, a 17x28 canvas: 1.67× the cells of 13x22), its own
contested sets and benches, and longer games. The fuzz already covered it (exp085: 4,398 games at
26x15/11, 0 failures).

## 2. Phase 0 — shakedown and benchmarks (~200 GPU-h, week 1)

### 2.1 Items

| item | cost | done when |
|---|---|---|
| `apptainer build` on an Arm node, `%test`, first-login checklist (`hpc/arrhenius/README.md`) | ~2 GPU-h | binaries report `nets run in-process on cuda`; the provenance stamp works in the container |
| `scripts/parity066.sh` + `gatec066.sh` on an H100 | ~2 | 0 differing bits in-process vs itself across lanes; mc-label byte-identical (Arm vs Arm only: plan 069 item 3) |
| tuning pilot `hpc/tune/gh200.toml` on one node (`--shard K/4`), twice (different times of day) | ~6 | §1.3's table filled; `results.md` copied home |
| fp16 gate: `hpc/tune/precision_gate.sh` with gen58, including fp16-vs-fp32 drives (400 fixed pairs, 20x11) | ~5 | value bench dMSE within SE, drives 0.5 ± 0.025 (exp081/082 offline: fp16 8× closer to fp32 than bf16) |
| trainer calibration on one GPU: batch 256/1024, bf16 autocast, loader workers 4/16, `PREPARED_ROOT=local` vs Lustre flash | ~6 | R measured; a batch/lr pair that matches batch-32 val loss within noise |
| hub and loop at scale: 8 nodes × 30 min of generation from the random net | ~32 | hub CPU < 1 core, dispatch p99 < 1 s (exp083's ceiling ~1,250 games/s), shard writes on Lustre not stalling the lock (exp083's open residual) |
| 1-node end-to-end loop, `MAX_GENS=3`, small shards, a forced deadline exit + resume (`hpc/arrhenius/slurm/README.md`, "First jobs") | ~16 | the chain resumes generate, eval and train; `chain.state` correct |
| random-net TD probe: the random 96x8+heads net, 1000 descents, 400 games at centre 220 and 400 at centre 144 | ~1 | per-board TD/drive (decides the bootstrap's centre, §3.2) |
| S3 contested sets screened by gen58 vs itself: 300 candidates × 4 playouts on 20x11, 500 on 16x9 (plan 065 §4's pipeline) | ~3 | `contested_20x11_s3g58.json`, `contested_16x9_s3g58.json` |
| yardstick numbers: gen58 vs scripted on both sets; gen58's net check | ~2 | rows in MATCHES/NETS |
| fuzz on the CPU partition: `scripts/fuzz_s3.sh`, one task per node, disjoint seeds, 20x11/7 + 26x15/11 | 0 GPU-h (CPU account) | 0 failures at the image's commit |
| margin for failed jobs and reruns | ~120 | |

Everything here also writes registry rows: the pilot and the gates as experiments, the sets in DATA,
gen58's numbers in NETS and MATCHES.

### 2.2 Readiness work on the box before the hours (no GPU-hours)

Small code items, each default-off and testable here:

- **R1** `scripts/make_random_net.py`: the head knobs (`--value-conv --value-pool --value-hidden
  --policy-conv`), so gen 0 is a random 96x8+heads (or 128x10+heads) net. Today it builds plain
  `BBNet(width, blocks)` only. **Done (bfda8bf):** same flags as `bbnn.train`, defaults unchanged
  bit for bit; `test_make_random_net.py` round-trips them through `BBNet.shape_of` and `NET.ts`.
- **R2** `train_loop.sh`: a scratch-training recipe for the `NO_WARM_FROM` path
  (`SCRATCH_TRAIN_ARGS`, `SCRATCH_EPOCHS`). Today that path reuses `EPOCHS` and
  `TRAIN_EXTRA_ARGS`, i.e. `--freeze-bn` at 3 epochs and a flat lr, where it wants plan 063's
  recipe (cosine, warmup, BN live). Also `TRAIN_BATCH`/`TRAIN_LOADER_WORKERS` pass-throughs and the
  head knobs for a fresh net. **Done (fa1a04c):** the NO_WARM_FROM training takes `SCRATCH_EPOCHS`
  (10), `SCRATCH_TRAIN_ARGS` (`--lr-schedule cosine --warmup-steps 2000`, BN live, lr `SCRATCH_LR`)
  instead of `EPOCHS`/`TRAIN_EXTRA_ARGS`, and `SCRATCH_ARCH_ARGS` (empty = the NO_WARM_FROM net's own
  shape); `TRAIN_BATCH`, `TRAIN_LOADER_WORKERS`, `TRAIN_AMP` reach every training. Unset = as before.
- **R3** `bbnn.train`: bf16 autocast (`--amp bf16`, off by default; exp081 found bf16 fine for
  training), pinned by a seeded test. **Done (526349e):** autocast on the training forward only
  (val, restore and export fp32); `test_amp.py`; `TRAIN_AMP=bf16` in the loop.
- **R4** the curve: `DRIVE_LADDER` (a list of reference nets each generation plays at fixed pairs,
  next to `DRIVE_REF=parent`) and `scripts/elo_fit.py` (Bradley–Terry over every pair result of the
  run, with SEs). Today the loop has one reference plus `ORIGIN_REF`.
  **Done (e8212d7):** `DRIVE_LADDER` takes model files and/or `geo` (the run's gens 1, 2, 4, 8, …
  below G, the parent left out, cut to the `DRIVE_LADDER_ANCHORS`=2 nearest the parent in the
  pooled fit — `elo_fit.py --nearest`; newest first before any fit), every `DRIVE_LADDER_EVERY`
  (1) gens at `DRIVE_LADDER_DRIVES` (400 per set = 200 pairs) without SPRT, each a hub eval job in
  `genNN/drives_ladder_<net>` with its own status line, then a `genNN elo:` line (Elo per board,
  the gain over 10 gens ± SE) and `RUN_DIR/elo.{json,csv,png}`. `DRIVE_SPRT=none` plays the parent
  rung at a fixed `DRIVE_CAP`, so it too enters the fit unbiased (`elo_fit.py --fixed-only` drops
  SPRT-stopped rows otherwise). The fit: quasi-binomial Bradley–Terry on pentanomial pair scores,
  SE = inverse information × Pearson dispersion (equals plan 051's points_se for one match; z-scores
  calibrated on synthetic ladders, `train/tests/test_elo_fit.py`), one fit per board, anchor
  `ELO_ANCHOR` (default `auto` = gen00, else the init gen01 played; set `ELO_ANCHOR=gen01` for
  §3.5's "gen 1 = 0"). All default off. Tested in isolation (the loop's `ladder_refs`,
  `eval_submit`, `eval_report` against a stub hub), not in a live loop.
- **R5** hub job priority: a `background` job (A/B arms, benchmarks) gets only streams no foreground
  generate game is waiting for, using the same rule label tasks follow today (plan 069 item 4b). This
  fills the training windows. **Done (b283a1c):** `job eval|generate|label --priority background`,
  hub-only (no protocol bump, still v25); foreground jobs are dispatched first, a background job gets
  a stream only when no foreground job can take it (game streams for eval/generate, an idle label
  stream for a label job); no preemption, so a foreground job arriving waits at most one background
  task per stream; `/status` marks `· background`; `train_loop.sh` `BENCH_PRIORITY` (default
  `foreground`) for the loop's benchmark/drive evals. Details in `botbowl-hub/CLAUDE.md`.
- **R6** `cfgs/chance_all_partial_gen_formation.toml`: the C1 generation preset with plan 047's
  formation setup teacher (`setup = "formation"`, `setup_formation = "random"`), for gen 0 only.
  **Done (95bb104):** pinned to C1-gen + the teacher by `bot_config.rs`; smoke: every setup
  placement a scripted one-hot sample (50/50), the C1-gen control searched all 48.
- **R7** `hpc/arrhenius/pull_home.sh`: one rsync with a filter list (§5.2), and a corpus
  retention script (§5.3). **Done (adc41dd):** `pull_home.sh` and `retain_corpora.sh` (dry run
  by default; the window + 2 generations never touched), documented in `hpc/arrhenius/README.md`.
- **R8** `prepare` per shard in parallel inside the loop, if the calibration shows a 5M-sample
  prepare taking more than a few minutes (exp083: ~12k samples/s single-process).
  **Done (38fd716, f738b3f):** `prepare --jobs N` encodes one part per input shard on N threads and
  appends the parts in input order (file-to-file copy, header checked): byte-identical to the serial
  pass on gen61-63's real window (176 `.npy`, manifests equal but for the commit). `PREPARE_JOBS` in
  `train_loop.sh` (default 1 = unchanged; ignored under `--dedup`). exp087: 1.6-3.6x at 6 jobs on
  the 8-core box while the loop was generating; it scales with free cores. Transient disk: the
  parts double the prepared size until merged.

All of these merge before the image is built: **the run pins one commit for its whole life**
(`BIN_MODE=image`). The az14x7 run died after gen02 when the encoder changed under it (plan 032
"Reset"); a game-code change mid-run means a new run or an explicit migration with a registry note.

## 3. The from-scratch run

### 3.1 Gen 0

| choice | recommendation | evidence |
|---|---|---|
| who plays gen 0 | **the random net** (`make_random_net.py --seed 0`, the main net's shape), C1 search at 1000 descents | az14x7v6 (plan 032 "Reset"): with a random net the search alone converted 0.596 of 14x7 drives, and gen01 beat the random net 40-0-0. A heuristic-MCTS gen 0 (`train_loop.sh`'s own bootstrap) makes the scripted evaluator the teacher, which is what the user wants to avoid |
| setups | **formation teacher for gen 0 only** (R6): placements answered by a random formation and recorded as one-hot scripted samples; `setup = "auto"` (searched) from gen 1 | plan 047: per-player setup is 7 decisions per kickoff; a random net searching them wastes budget and teaches uniform placements. One generation of formation priors is the smallest intervention. **Open decision D1** |
| value labels in gen 0–2 | **raw drive outcomes** (`VALUE_BLEND=1.0`, `MC_LABEL_PLAYOUTS=0`) | an MC label is the value of the generator's *bare policy* (plan 056). With a random or near-random policy the playouts measure random play, not the search's. Outcomes are unbiased for what the search did |
| when MC labels start | when the bare policy carries the drives: label one held-out shard each generation from gen 2; switch on (with `LABEL_STREAMING=1`) once mean(MC label) − mean(outcome) on that shard is within 0.05 TD of 0, or at gen 6 at the latest | plan 056: MC labels cut value RMS by 15% once the policy is decent; the bias test costs one shard's labels |
| gen 0 size | 8k games (~0.9M samples; ~1.5 GPU-h at P = 100k) | big enough for 10 scratch epochs without memorising; it is the noisiest corpus of the run |
| gen 1 training | from random init (`NO_WARM_FROM`), plan 063's recipe: Adam, lr 1e-3, 2000 warmup steps, cosine, BN live, 10 epochs, value weight 0.36 per drive, cq τ 100, restore on val | exp072 recipe; plan 032 "Reset" |

### 3.2 The early schedule (gens 1–12)

| gens | games/gen | window | train step (warm start) | sizes | labels |
|---|---|---|---|---|---|
| 0 | 8k (random net) | — | — | per the TD probe | outcomes |
| 1 | 8k | gen 0 | scratch recipe (§3.1) | | outcomes |
| 2–3 | 8k | 2 gens | lr 2e-4, `--freeze-bn`, 2 epochs | | outcomes; MC probe on one val shard |
| 4–7 | 12k | 2 gens | lr 1e-4, 2 epochs | | MC (8 playouts, streamed) once the probe passes |
| 8–12 | 16k | 3 gens | lr 5e-5 (the loop's), 1 epoch over the window | 220 | MC |

- **Learning rate:** the warm-start lr steps down over the first generations (2e-4 → 1e-4 → 5e-5),
  as knob changes at generation boundaries. 2e-4 was the loop's warm lr until plan 054 found it
  undid the absorption of a mature net. Early nets move a lot per generation, and exp071's "lr 1e-4
  is worse" was measured on a plateaued gen26.
- **BatchNorm:** live in the scratch step, frozen from gen 2. The running statistics exist by then,
  and the per-board BN mismatch (code review 2026-10-03) argues for frozen.
- **Descents:** 1000 throughout the bootstrap (C1 generation preset). The search carries the
  conversion early (plan 032), and §4 decides whether to go to 4000.
- **Board sizes:** the full 20x11 mix (centre 220, floor 0.2, min area 70) from gen 0 if the TD
  probe's large boards (area ≥ 187) convert ≥ 0.4 of their drives under the random net. Otherwise
  centre 144 for gens 0–3, then 220. Reasons: upward transfer is nearly free (plan 042 E0); the
  random-start drives make the reward dense; the floor keeps 20% on small boards anyway; boards
  under 70 cells are 2v2 non-games (E0). `size_curriculum.py`'s automatic advance needs a baseline
  this run will not have (no heuristic gen 0) and holds at an absolute 0.73 even for the box's net
  (gen59 status), so the move is by hand, at a boundary.
- **Next drive:** on (`NEXT_DRIVE=1`), so setups and kick-offs are in every generation.

### 3.3 Net size from the start

**Start at 96x8 + bigger heads** (`--value-conv 64 --value-pool meanmax --value-hidden 128
--policy-conv 96`, exp072's arm B), with a **128x10+heads arm in parallel through the bootstrap**.

- exp072: at equal training 96x8 beats 64x6 on every number (value MSE −6%, test KL −10%, value
  bench 0.219 → 0.196), and the bigger heads add ~1.6% value.
- exp074: once it generates its own data, 96x8 plays even with 64x6 at equal compute (16x9 drives
  0.512 ± 0.015) with a clearly better value head (0.179 vs 0.192). It paid its inference cost on
  ≤16x9 boards, on a box where data was the bottleneck.
- The cluster changes both premises. 20x11 boards are 1.5× the cells, and there will be ~100× the
  data per generation. A bigger net turns compute into strength when data is plentiful.
- 64x6 is ruled out. 192x12 (~6× per forward) is a 50k-hour step (§1.5).
- exp075: many failures are not capacity (mirror asymmetry, rare skills). Capacity is one lever, not
  the fix.

### 3.4 Bootstrap read-out: the net size

Two arms, the same schedule (§3.2), one node each, gen 0 to ~gen 12, for the same wall time
(~25 h, ~100 GPU-h per arm). The games column of §3.2 is the 96x8h arm's (~160k games, ~31 GPU-h
of pure generation at P = 100k; the rest is small-generation tails and training on one node). The
128x10h arm plays ~1/2.2 of them per generation, so that both arms' generations take the same time.

- **Metric:** Elo per GPU-hour. Every 2 generations, each arm's latest net plays the other arm's
  latest and the shared ladder (§3.5), at fixed pairs, with both bots at **equal compute**: the
  128x10h side gets 1000 descents, the 96x8h side ~2,200 (`--opponent-iters`, as exp074 did).
- **Decision:** the main run continues the arm that is ahead at equal GPU-hours spent. A tie goes to
  96x8h: it is cheaper, and the capacity step in §3.6 is still available.
- The losing arm's last net is kept as a ladder point.

### 3.5 Measuring progress without a champion

**The ladder** (all drives, C1 eval preset `cfgs/chance_all_partial.toml`, 1000 descents, both sides
unless equal-compute is stated):

| opponent | role | when |
|---|---|---|
| the previous generation (`DRIVE_REF=parent`) | the step; one point per generation | every gen, 200 fixed pairs per board |
| the run's own frozen anchors: gens 1, 2, 4, 8, 16, 32, 64, 128 (geometric) | the curve's backbone; each net plays the **two anchors whose score lands nearest 0.5** | every 2 gens, 200 fixed pairs |
| scripted bot (`CandidateBot::Scripted`) | absolute floor; gen01's only informative number in az14x7 (0.383) | gens 1–10, then every 10 |
| v9 gen58 (the box's S3 net, `models/az_v7/bbnet_mix16x9v9_gen58`) | **yardstick only**: never a start, a teacher or a screen for the run's own sets | every 5 gens, 200 fixed pairs |

- **Boards:** 20x11 is primary. 16x9 is secondary: cheaper, and it catches small-board regressions.
  The sets are the S3 rescreens from shakedown (`contested_*_s3g58`, screened by gen58). The
  existing sets were screened under the old rules, so "contested" is only approximate under S3
  (DATA.md "S3 rules switch").
- **The fit:** `scripts/elo_fit.py` (R4) runs Bradley–Terry over every pair result of the run. It
  pools the pentanomial pair scores, sets gen 1 = 0 Elo and reports a SE per net. The curve is
  Elo against GPU-hours spent (and against games). Fixed pairs, not SPRT, because SPRT stops at a
  sample size that depends on the result and biases the fit. 200 pairs resolve about ±0.016 in
  score (MATCHES: 300 pairs ±0.012–0.014), i.e. ±11 Elo near 0.5.
- **Re-anchoring:** an anchor drops out of the active set once every new net beats it by > 0.85. Its
  past results stay in the fit (plan 032: "an anchor for a from-scratch run should be a net from
  that run, not a floor").
- **Contested sets drift** as the run passes gen58's strength. Rescreen with the run's own net at
  milestones (first H1 over gen58, then every ~50 gens). Keep every old set in the fit as its own
  "board"; nothing is replaced.

**Single-net checks (every generation):**
- **Value benches:** re-frozen from the run's own held-out shards (4, 7), 48 playouts under the
  then-current net, 3,000 states over the 20x11/20x10/18x11 boards and one 16x9-class bench
  (`scripts/value_bench_build_v9.sh`). Freeze the first at gen ~5 (once MC labels are on), then at
  each milestone. Every net is scored on every bench frozen before it; held-out shards are never
  trained on, so all of them are clean. The v9 benches do not replay under S3 (DATA.md), so none
  carry over.
- **net check** (`scripts/net_check.sh`, plan 055): search gain over the bare policy at
  64/250/1000/4000 descents on the next generation's corpus. The user's criterion: monotone in
  budget.
- **Absorption probe**, corpus TD/drive per board, tree statistics per board (plan 060). The 14x5
  throw-in bug was found from a depth outlier on one board (exp "14x5 edge-bounce chains"), so
  per-board depth and chance share go on the status line.
- **Probes** (plan 064, CPU): at milestones.

**Cadence and cost (P = 100k; drives at ~50k positions/s, since two nets batch worse):**

| measurement | cadence | GPU-h each |
|---|---|---|
| drives vs parent, both boards, 200 pairs | every gen | ~0.1 (a 20x11 pair ≈ 62k forwards, plan 065 §5) |
| drives vs two anchors | every 2 gens | ~0.2 |
| drives vs gen58 / scripted | every 5 gens | ~0.1 |
| net check | every gen | ~0.1 |
| value benches (scoring) | every gen | < 0.01 |
| value bench freeze, contested rescreen | milestones | ~1–3 |

That is ~0.5 GPU-h per generation, ~3% of a 16 GPU-h generation, run as background jobs (R5) in
the training window. On the box this cadence was unaffordable (drives every 3 gens, plan 058); on
the cluster it is noise.

### 3.6 Stop and branch rules

| signal | action |
|---|---|
| Elo gain over the last 10 gens < 2 SE of the fit, value benches flat (paired dRMS within ±1%), net check monotone | **plateau → capacity step:** start the shadow lineage at the next size (`SHADOW_INIT`, plan 063 §shadow: the bigger net fine-tuned on every window, never generating). Switch the generator when the shadow beats the main net on drives at **equal compute** (H1 on 20x11, not worse on 16x9) |
| a second plateau after the capacity step | stop the 5k main run; propose the community evaluation to the user; the remaining hours go to §1.5's choices |
| a generation loses to its parent by > 3 SE on both boards, or net check non-monotone two gens running | roll the generator back to the parent (training lineage continues) and investigate: the loop is gateless, the operator decides (plan 030) |
| corpus TD/drive drops > 0.1 over 2 gens on the big boards | stalling or a bug: check tree stats per board, then the latest code change |
| first SPRT H1 over gen58 on the 20x11 set | milestone: tell the user, rescreen sets, freeze a bench, add the net to the anchors |
| an A/B arm wins (§4) | adopt at the next generation boundary; note the generation in the registry |

## 4. The A/B menu

Two protocols, both judged on drives:
- **One step:** from parent gen G, the treatment changes one thing for one generation (its own
  generate if the change is in generation, else training only on the main run's data). It is
  compared with the main run's gen G+1 on the same parent: 400 fixed pairs per board, plus the
  value benches, absorption and net check. One-step effects are small: τ=50's one-step lead
  (exp069) did not replicate over three generations (plan 059 read-out). So a one-step winner is
  confirmed by a short run before adoption.
- **Short run:** the treatment forks from gen G and runs 4–6 generations at the main run's shape on
  its own node(s), while the main run continues. Each generation plays the main run's same-numbered
  generation (fixed pairs). Judge on the Elo curve against GPU-hours.

| # | question | protocol | metric | cost (GPU-h, P = 100k) | when | priority |
|---|---|---|---|---|---|---|
| A1 | **net size** 96x8h vs 128x10h (192x12h only at 50k) | parallel bootstrap arms (§3.4); later a shadow at the capacity step | Elo per GPU-h at equal compute | in the bootstrap budget | **before the main run** | 1 |
| A2 | **games per generation**: generate phases of ~20 / 40 / 80 min on the same fleet (on 1 node: ~6k / 13k / 26k games; ×4 on the main run's 4 nodes) | short runs, 1 node per arm, ~10 h wall each | Elo per GPU-h at equal hours | ~120 (3 arms × ~40) | **before** (it fixes the run's shape); runs in the bootstrap's second half | 2 |
| A3 | **descents in generation** 1000 vs 4000 (~3.2× per game) | short run, equal GPU-h (4000 arm plays ~1/3 the games) | Elo per GPU-h; net check at 4000 | ~100 | alongside, early in the main run | 3 |
| A4 | **label playouts** 8 / 16 / 32 | one step with plan 067's exact top-up (exp078's script: same games, labels topped up; both arms select on the same 32-playout val labels) | value benches (paired), net check, then drives | ~10–30 (labels only: +28% / +84% GPU per sample when GPU-bound; ~free if CPU-bound) | alongside | 4 (1 if the pilot shows spare GPU, plan 069 item 4b) |
| A5 | **window and reuse**: 2 vs 3 vs 5 gens; 1 vs 3 epochs over it | one step, training only, on the main run's data; 2 trainer seeds (fine-tune seed spread SD 0.021, plan 049) | val, absorption, value benches; drives on the winner | ~5 per arm | alongside, at gen ~20 and again at ~100 | 5 |
| A6 | **cq τ** 100 vs 50 and **value weight** 0.36 vs 0.5 | one step, training only | as A5 | ~5 per arm | alongside | 6 (exp069/071: τ=50 did not hold over three generations, vw 0.36 won; re-test only because from-scratch nets differ) |
| A7 | **Gumbel root** (m = 16, q floor 1000) vs PUCT; m = 16 vs 32 on 20x11's wider fans | eval-only first (same net, two presets, 400 pairs); short run only if eval-only wins by > 0.03 | drives | ~1 eval-only; ~100 short run | alongside | 7 (plan 053: Gumbel beat PUCT at 1000 on both boards and PUCT at 4000 on 16x9; C1 built on it, exp070) |
| A8 | **fp16 generation** | the shakedown gate (§2.1) | drives fp16 vs fp32 bot; value bench | ~5 | **before** | gate |

Order: A8 and A1 settle during shakedown and bootstrap. A2 runs in the bootstrap's second half on
the bootstrap's own net. A3–A7 run alongside the main run, one or two at a time, in the training
windows (R5) or on 1–2 extra nodes, so the main run never waits on them. Budget: A2 ~120, A3 ~100, A4–A6 ~70, A7 ~5–100, and
~300 for short-run confirmations of one-step winners: ~600–700 of the 750. A
**playout-cap randomisation** (KataGo: full search on a share of moves and only those recorded as
policy targets) is not built; it is the next A/B candidate if A3 says more descents pay.

## 5. Operations

### 5.1 Job shape

The loop runs in a Slurm job, as built (`hpc/arrhenius/slurm/`, plan 069 item 5):
- one srun step, one task per GH200 (`-c 72`, one GPU each);
- task 0 runs `train_loop.sh` (hub, trainer, its per-phase worker, net check, value benches); every
  other task runs a long-lived `botbowl-worker` dialling it;
- jobs are chained with `afterany`, and the loop resumes from its markers (generate and eval mid-phase
  via `--resume`, plan 069 item 5).

| | bootstrap arm | main run |
|---|---|---|
| nodes | 1 (4 GPUs) per arm | 4 (16 GPUs), plus 1–2 for A/B arms when needed |
| walltime | the centre's maximum (2 h default; unknown max) | same |
| streams per GPU | from the pilot (`WORKER_STREAMS`, `WORKER_LABEL_STREAMS`) | same |
| trainer | node 0, GPU 0, `PREPARED_ROOT=local` (NVMe) | same; DDP over node 0 if the trainer binds (§1.2) |
| env file | `loop.env.example` with: random `INIT_CHAMPION` + `NO_WARM_FROM` (R1, R2), `DRIVE_REF=parent` + `DRIVE_LADDER` (R4), `MC_LABEL_PLAYOUTS` per §3.1 | |

**The training window.** While node 0 trains (~15–25 min a generation), the other 15 GPUs idle
unless something is queued. The measurement jobs (§3.5) and A/B arms go there as background jobs
(R5). The alternative, a one-generation-stale "generate ahead" (workers start gen G+1 with net G
while G+1 trains, as in AlphaZero's asynchronous loop), would change the data. It is not proposed
unless R5 cannot fill the window.

### 5.2 What goes home (daily, one 2FA rsync session; R7)

- **nets:** `.pt`, `.onnx`, `.ts` per generation (~6–25 MB each), into `models/hpc_<run>/`;
- **run records:** `status.md`, `chain.log`, every `report.json` / `eval.games.jsonl` (drives),
  `absorb.txt`, `train.log`, net check and value-bench summaries, tree stats, `size_curve.tsv`,
  `td_rate.json`;
- **positions and benches:** the contested sets, the frozen value benches (small JSONL);
- **pilot outputs:** `hpc_tune/*/results.{csv,md}`, `meta.json`.

Not corpora. The registry is edited at home from the pulled files (one row per net, match, set and
experiment, as now), and the home checkout stays the source of truth.

### 5.3 Corpus retention

At P = 100k a 50k-game generation is ~5.4M samples: ~60 GB of raw JSONL, ~60 GB of `mc/` shards and
~70 GB prepared, about 190 GB live. 200 generations would be ~38 TB raw.

| data | kept | form |
|---|---|---|
| window generations (raw + `mc/` + prepared) | while in the window + 2 | as written; prepared on node-local NVMe, rebuilt when missing |
| `mc/` train shards, older gens | all | zstd (~25×): ~0.45 KB per sample, ~0.5 TB for the run. A from-scratch retrain of a bigger net on everything stays possible (plan 063 / exp072) |
| raw train shards, older gens | deleted once `mc/` is written | `mc/` carries the same lines with labels |
| val shards 4, 7 | all | zstd; the source of every future bench and rescreen |
| eval `games.jsonl`, reports | all | small |

Quotas are unknown (plan 069 §0b). Ask before the main run; until then budget ~2 TB on the disk tier.

### 5.4 Failure modes

| failure | what happens | do |
|---|---|---|
| job hits walltime mid-generate | drained at the deadline, exit 75, the next job resumes the missing seeds (`scripts/hub_resume_test.sh`, `loop_resume_test.sh`) | nothing |
| a phase longer than a whole job (exit 76) | the chain stops | shrink games per gen or add nodes; raise the walltime |
| worker node dies | the hub requeues its games after the silence timeout | nothing; check `hub.log` |
| node 0 dies (hub + trainer) | the job ends; next job resumes; a cut train retrains | nothing |
| a cut train | retrains from the same warm start | nothing (minutes) |
| silent tract fallback (worker without the net on GPU) | slow generation | `NN_SERVER_FALLBACK` WARN; workers must log `nets run in-process on cuda` |
| hub lock stalled by Lustre writes | dispatch latency up | per-shard writer thread (exp083's residual); watch p99 in the scale test |
| disk quota | writes fail | retention (§5.3); the loop prints free space per phase |
| loss of the corpus (no backup on `/nobackup`) | window lost | nets are home daily; a run continues from its last net with a fresh generation |
| a code bug found mid-run | | stop at a boundary; fix; new image; the registry notes the generation; corpora before it marked |

### 5.5 What to watch (status page + `status.md`)

GPU-hours burned against this plan (`sacct`), generations/day, games/min, positions/s per GPU and mean
batch (`NN_INPROC_STATS`), the generation tail and the label tail (generate done → labels done), train
minutes and best step, TD/drive per board, tree depth and chance share per board, absorption, value
benches, net check, drives against parent, ladder and yardstick, Elo curve per GPU-hour, and disk
usage against quota. Alarms: any `NN_SERVER_FALLBACK`, a phase taking more than 1.5× its previous
duration, a non-monotone net check, a TD/drive drop.

### 5.6 The first week

| day | |
|---|---|
| 1 | first login, quotas, `apptainer build` + `%test`, parity, fuzz job on the CPU partition (CPU account) |
| 2 | tuning pilot (one node, ~40 min) and its read-out; fp16 gate; trainer calibration; random-net TD probe |
| 3 | 1-node end-to-end smoke with a forced deadline + resume; the 8-node scale test; S3 contested sets and gen58's yardstick numbers |
| 4 | launch the two bootstrap arms (1 node each, chained) |
| 5–6 | bootstrap running; first value bench frozen at gen ~5; first anchors; A2 arms forked from the leading arm at gen ~8 |
| 7 | net-size read-out (§3.4); fix the main run's shape from A2; launch the main run on 4 nodes from the chosen arm's latest generation |

The bootstrap generations become the first generations of the main run. Nothing is thrown away, and
the curve starts at gen 0.

## 6. Open decisions for the user

- **D1 — gen 0's teacher.** Random net with formation setups for gen 0 only (recommended); or a
  pure random net with no scripted setups; or heuristic-MCTS games (cheap, CPU partition, but the
  scripted evaluator becomes the teacher).
- **D2 — fleet size, i.e. wall time against generations.** The main run on 4 nodes (~200
  generations, ~8.5 days) or 8 nodes (~100 larger generations, ~4 days). This depends on the queue
  and on A2.
- **D3 — growing the net mid-run.** One planned capacity step (96x8h → 128x10h through the shadow
  lineage) at the first plateau. It is a kink in the curve, but a documented one. The alternative
  is one net size for the whole 5k.
- **D4 — CPU allocation.** Apply for CPU-partition hours (a separate account) for fuzzing and
  possibly x86 tract label workers, or keep everything on the GPU account.
- **D5 — the community evaluation trigger.** Propose it at the first plateau after an H1 over gen58,
  or let the user call it on the Elo curve.

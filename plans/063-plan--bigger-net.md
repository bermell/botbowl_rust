# Plan 063 — a bigger net, trained from scratch on the clean v9 corpora (exp072)

**Status:** Built 2026-10-13, not run. The code (architecture knobs, from-scratch schedule, loader
workers, one-pass validation) is on master with every default equal to today's net and trainer
(byte-identical, pinned by tests). `scripts/exp072_bigger_net.sh` runs the experiment once the loop
is stopped; nothing has used the GPU yet. The user approved a 96x8 trunk and asked whether the heads
should grow too: arm B tests that.

## 1. Why now

The v9 loop (`runs/loopmix16x9v9`, plan 058) has improved generation over generation since gen10
(C1 search from gen25, value weight 0.36 from gen28, gen30 drives vs the d1k anchor 0.595 / 0.567),
but each fine-tune moves its own val loss by ~0.001 and absorbs ~+0.001 P(played) (exp071). exp071
ruled out the step size (lr 1e-4 worse on every number) and the policy:value balance beyond 0.36;
the two levers left were the target's noise and **net capacity**. The launcher already says it: when
the drive benchmark and the value bench stall, the next step is a bigger net.

Capacity was tried once, plan 032 #9 (2026-09-08): 96x8 from scratch on D7's 2.2 M samples scored
0.508 ± 0.043 against the 64x6 twin in 120 full games — "no gain at this data size". Three things
differ now: the corpus is ~3.5 M prepared samples of far better labels (MC-averaged values, cq τ=100
targets from a mean-backup, C1 search that beats its policy), the heads get capacity too (arm B), and
the verdict is paired contested drives with SPRT, which resolves ~±0.02 where 120 games resolved ±0.09.

## 2. The architecture knobs (`train/src/bbnn/model.py`)

`BBNet(width, blocks, value_conv=32, value_hidden=64, value_pool="mean", policy_conv=0, ...)` —
every default is today's net:

| knob | default (today) | meaning |
|---|---|---|
| `width`, `blocks` | 64, 6 | residual trunk (existed) |
| `value_conv` | 32 | channels of the value head's 1x1 conv (+BN+ReLU) |
| `value_pool` | `mean` | `meanmax` concatenates the global mean and the global max (fc input doubles) |
| `value_hidden` | 64 | width of the value head's hidden layer (existed) |
| `policy_conv` | 0 | 0 = the policy is one 1x1 conv of the trunk; k = a 3x3 conv + BN + ReLU of k channels (`policy_conv`, `policy_bn`) before that 1x1 (`policy_head`) |

- **Loading any shape.** `BBNet.shape_of(state_dict)` reads every knob back from the weights
  (`width`/`blocks` always, a head knob only when it is not the default, so a 64x6 still reads
  `{"width": 64, "blocks": 6}`). `from_state_dict`, the sidecar's registry (`nn_server.py`), the
  trainer's `--init`, `absorb_probe.py`, `bbnn.migrate` and `bbnn.from_onnx` (which now knows
  `policy_conv`'s BatchNorm) all rebuild the net from it — no shape flag anywhere downstream.
- **ONNX is unchanged at the interface:** inputs `spatial (N,61,H,W)`, `global (N,18)`, outputs
  `policy (N,32,H,W)`, `value (N,1)`. tract runs the new ops (ReduceMax, Concat, a BN folded into the
  policy's 3x3 conv): `botbowl-nn/tests/parity.rs` has a second case on a committed tiny big-head
  fixture (`tests/fixtures/heads/`, `python -m bbnn.fixture --heads`), and tract loads a 96x8
  big-head export (`examples/nn_bench`).
- **`forward_masked`** (the sidecar's `--canvas` path) re-zeroes the padding after the policy conv;
  the max pool needs no correction (ReLU output, padding zero). Pinned on every curriculum board.
- **Byte-identical for today's nets:** gen30's ONNX re-exports to the same md5 (6ca1ded…); the
  seeded init of `BBNet()` and `BBNet(96, 8)` and a seeded train + fine-tune match the pre-063
  code's digests (`test_arch.py`, `test_scratch_train.py`, pinned at torch 2.13).

### Head choices, and why

- **Trunk 96x8** (0.50 M → 1.41 M params, 97 M → 277 M MAC per sample at 11x18). Depth is not for
  reach — 64x6 already sees 27 cells, more than the widest board — it is capacity: 2.9× the FLOPs.
- **Value head, arm B:** 64 channels, mean+max pooling, 128 hidden. The head sees the board only
  through a global pool, and the mean over up to 198 cells dilutes exactly the local facts a drive
  turns on (a free ball carrier, one open lane, a player next to the endzone). Max pooling keeps
  them; KataGo pools mean and max for the same reason on variable board sizes. A flattened FC head
  (AlphaZero) is not an option: the net plays 12 board shapes.
- **Policy head, arm B:** a 3x3 conv + BN + ReLU of 96 channels before the 1x1 logits. Today a move's
  logit is a *linear* function of the trunk features at its cell, so every policy nonlinearity is
  shared with the value head; AlphaZero and KataGo both give the policy private layers. 3x3 rather
  than 1x1 lets a destination's logit read its neighbourhood (tackle zones, the next square). Cost:
  +6% of the 96x8 trunk's FLOPs.
- **Arm A (96x8, today's heads)** separates "bigger trunk" from "bigger heads"; **arm C (64x6 from
  scratch)** separates both from "fresh training on all the data", which no loop net has had.

## 3. Training from scratch (`bbnn.train`)

New flags, all off by default: `--value-conv --value-hidden --value-pool --policy-conv` (a fresh
net's heads; `--init` fixes the shape from its weights instead), `--lr-schedule cosine` with
`--warmup-steps` / `--min-lr` (linear warmup, then a half cosine to `--min-lr` at the last step),
`--loader-workers N` (DataLoader worker processes, persistent across epochs).

- **Validation is one pass now.** A mixed corpus was validated twice: once pooled, once per board.
  `evaluate_by_group` gets both from the pooled pass (each batch is one board; its shape names it),
  bit-identical numbers at half the cost. It draws the RNG the old per-board passes drew, so seeded
  runs keep their trajectory. The loop gets this at its next train phase after the merge (each phase
  starts a fresh `python -m bbnn.train` on the main checkout's `train/src`; no relaunch): its 64x6
  fine-tune spends most of its ~39 min in 36 validations of 119k samples, twice each. Checked
  2026-10-13 on real data: the loop's exact fine-tune flags (`--init` gen30, lr 5e-5, `--freeze-bn
  --eval-at 250,500 --eval-every 1000 --select-on combined --value-weight 0.36
  --per-drive-value-weight`, seeded, CPU) on one gen30 MC shard pair print the same log and save
  byte-identical `.pt` and `.onnx` under the old and the new trainer.
- **The loader was the bottleneck.** Measured on this box with the loop running: the single-process
  loader serves 2.4k samples/s at batch 32 and 2.9k at 256; 4 workers 9.8k/s. The loop's train phase
  (1.09 M train + 8.6 M val samples in 39 min) is consistent with being loader-bound.
- **The exp072 recipe** (same for all arms): batch 256 (the 1060's 6 GB holds a 96x8 batch of 256 on
  the 18x11 canvas with room: ~1.5 GB of activations), Adam, peak lr 1e-3, 2000 warmup steps, cosine
  to 0, 10 epochs (~104k steps), validation every 5000 steps (~21 checks), restore on
  val_policy + val_value, value weight 0.36, per-drive value weight, MC labels (`--value-blend 1.0`),
  cq τ=100 targets, 6 loader workers, seed 72, **BatchNorm live** (`--freeze-bn` is for fine-tunes
  of a net whose running statistics exist). Validation runs in eval mode on the running statistics,
  so the restore already scores the per-board BN mismatch (code review 2026-10-03); if the arms'
  small-board val lags, a short frozen-BN tail for every arm is the fix, not part of the run.

## 4. Data

`runs/loopmix16x9v9` gen10–gen31 (gen10+ has the 14x5 throw-in fix; gen25+ are C1 searches), the
MC-labelled shards `genG/mc/shardK.jsonl`, prepared once for all arms with
`--policy-target cq --tau 100 --value-blend 1.0`:

| split | shards | samples (est.) | disk (est.) |
|---|---|---|---|
| train | 0-3,5,6 of gen10–31 | ~2.66 M | ~28 GB |
| val (selection) | 4,7 of gen10–30 | ~0.84 M | ~9 GB |
| test (never selected on) | 4,7 of gen31 | ~40 k | ~0.45 GB |

Measured on gen30's 3-generation window: 362 k train + 119 k val prepared samples, 3.8 + 1.3 GB, i.e.
1.7 GB and ~160 k samples per generation; every generation's MC corpus is 1.31–1.40 GB, so the
extrapolation is flat. **~38 GB** in `runs/exp072/prepared_*` (176 GB free), removed after the absorb
phase unless `KEEP_PREPARED=1`. The trainer holds ~3.5 GB of action/target arrays in RAM (shared
copy-on-write with the workers); `spatial.npy` is memory-mapped. `MultiDimsDataset` already mixes
every `dims_*` dir (12 board shapes) with single-board batches.

The test set is gen31's held-out shards alone, so the arms are compared to the loop's gen31 on data
neither trained on. The loop's gen31 *selected* its checkpoint on a val window that includes those
two shards, a small edge to gen31 — the comparison is conservative for the arms.

## 5. Cost

**CPU-measured relative cost** (this box, loop running, 4 torch threads, interleaved, medians):

| arm | params | MAC/sample (11x18) | fwd b=12–24 | train step (b=64) | tract b=1 (9x16) |
|---|---|---|---|---|---|
| C64x6 | 0.50 M | 97 M (1.00×) | 1.00× (1.06 ms/sample) | 1.00× (4.6 ms/sample) | 3.1 ms |
| A96x8 | 1.41 M | 277 M (2.86×) | 2.7× | 2.5× | 7.8 ms (2.5×) |
| B96x8h | 1.51 M | 294 M (3.03×) | 2.8–2.9× | 2.7× | 8.0 ms (2.6×) |

**GPU estimates (GTX 1060 6GB, not measured — the GPU is the loop's).** The sidecar's 64x6 ceiling
is ~10k forwards/s at full occupancy (plan 058 §9: 8.6k/s over two sidecars); training is ~3× a
forward, so 64x6 trains at ~3.3k samples/s GPU-bound, 96x8 at ~1.0–1.5k/s (÷2.5–2.9), with the
6-worker loader (~10k/s) out of the way. Per arm, 10 epochs = 26.6 M train samples + 21 validations
of 0.84 M:

| arm | train | validation | total |
|---|---|---|---|
| C64x6 | ~2.2 h | ~0.5 h | **~2.7 h** |
| A96x8 | ~5.9 h | ~1.4 h | **~7.3 h** (6–9 h) |
| B96x8h | ~6.3 h | ~1.4 h | **~7.8 h** (6.5–9.5 h) |

~18 h of GPU training, plus ~10 min prepare (CPU), ~30 min benches/absorb/value bench, and the
drives (§6; ~2–3 h per arm on a hub of our own). `train.progress.json` (step, started, updated)
gives the real rate after the first check: if A96x8's first 5000 steps take far more than ~20 min,
cut `EPOCHS` for all arms.

**Inference in the loop.** Generation and MC labelling are GPU-bound since plan 058 (two sidecars,
GPU ~100%): a 96x8 net costs ~2.7× per forward, so a generate phase goes from ~160 to ~400–450 min
and mc-label from ~96 to ~250–270 min; with train, net check and value bench a generation cycle goes
from ~5 h to ~11–12 h. Helper workers on tract pay 2.5× per forward. The script measures the real
sidecar numbers (`nn_server.py --bench`, 11x18, batches 1–32) for every arm and gen31.

## 6. The experiment (`scripts/exp072_bigger_net.sh`)

Phases `train bench absorb vbench drives`, each skipped when its output exists (a relaunch resumes;
an unfinished arm retrains from scratch):

1. waits for gen31 to be trained (its MC labels and its net), then prepares (CPU);
2. **waits for the loop to exit and the GPU to be empty** (no `bash scripts/train_loop.sh`, no CUDA
   process). Stop the loop after gen31's training with
   `nohup scripts/stop_after.sh "gen31 train done" runs/loopmix16x9v9 > /dev/null 2>&1 &`, or pass
   `STOP_LOOP=1` (touches `runs/loopmix16x9v9/STOP` once gen31 has trained);
3. trains C64x6, A96x8, B96x8h (`runs/exp072/<arm>/net.{pt,onnx}`, `train.log`);
4. `bench`: sidecar forward cost per arm and gen31; `absorb`: `absorb_probe.py --summary` on the
   gen31 test set, every arm against gen31 (CE, KL, log P(played), top-1, value MSE vs the MC label);
   `vbench`: `value_bench.sh` on `runs/value_bench/v9_gen01_val.jsonl`, paired vs gen31;
5. relaunches the loop only with `RELAUNCH=1` (then the drives share its hub); a manual relaunch
   must `rm runs/loopmix16x9v9/STOP` first, or the loop exits at its first phase boundary. A rerun
   whose GPU phases are all done (every arm trained, benched, with a row in `absorb.txt` and the
   value bench summary) skips the GPU wait, so the drives can resume with the loop running;
6. `drives`: each arm vs the loop's gen31, C1 (`cfgs/chance_all_partial.toml`) both sides, 1000
   descents, both contested position sets (`runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json`),
   SPRT 0.5:0.55, at most 800 drives per board. On the loop's hub if one answers on :13337, else on a
   hub of its own on :13338 with one sidecar and 24 local streams. The arms are copied to
   `models/az_v7/exp072_<arm>.{pt,onnx}`, because a sidecar resolves a worker's content-addressed
   ONNX against the directory of its own `--model`. On the loop's hub, mind its sidecars'
   `--max-models 4` (generator + its own eval pair + the arm + gen31 can reach 5; the overflow falls
   back to tract, slowly); a remote worker needs this commit's `model.py` to serve arm B on its own
   sidecar (A and C load on older code, tract runs all three).

```sh
nohup scripts/stop_after.sh "gen31 train done" runs/loopmix16x9v9 > /dev/null 2>&1 &
setsid nohup scripts/exp072_bigger_net.sh > /dev/null 2>&1 < /dev/null &    # status: runs/exp072/status.md
# subsets: ARMS="A96x8" PHASES="train absorb" scripts/exp072_bigger_net.sh
```

## 7. Decision rule

1. **Strength at equal search:** an arm passes if its drives against gen31 end **H1 on both boards,
   or H1 on one with the other level** (95% interval containing 0.5). An arm whose drives end H0 on
   either board below 0.5 − 2 SE fails.
2. **No regression on the cheap signals:** test-set value MSE and the value bench not worse than
   gen31 by more than 1%, log P(played) not lower.
3. **A vs B:** if both pass, take the higher pooled drive score; within 1 SE of each other, take A
   (cheaper heads, simpler). B alone passing says the heads, not the trunk, were the limit.
4. **The control decides what was learned.** If C64x6 also passes, fresh training on all the data
   is (part of) the gain — then C64x6 is adopted first, at no throughput cost, and the 96x8 net must
   beat C64x6 head to head (one more drive match) to justify its cost. If C fails and A/B pass, the
   gain is capacity.
5. **Throughput is the user's call, stated up front:** a 96x8 net makes each loop generation ~2.3×
   longer (§5). A pass at equal descents does not prove a win per GPU-second; an equal-cost match
   (gen31 at ~2.7× the descents of the arm) answers that, and needs a per-side search budget the
   eval does not have yet (`--vs-mcts-iters` or a preset `budget_scale`; not built). Run it first if
   the equal-descent margin is small (≲0.55).

## 8. How the loop continues with the new net

`train_loop.sh` already handles any shape: `--init` builds the warm start's shape from its weights,
`nn_server.py` rebuilds from the `.pt` beside the ONNX (`--canvas` included), the export keeps the
ONNX interface, and absorb probe / value bench / net check / drives load by shape. To continue
`runs/loopmix16x9v9` on the adopted arm:

1. copy `runs/exp072/<arm>/net.{pt,onnx}` to `models/az_v7/bbnet_exp072_<arm>.{pt,onnx}`;
2. with the loop stopped, write the `.onnx` path into `runs/loopmix16x9v9/champion.txt` and the
   `.pt` path into `latest_net.txt` (`INIT_CHAMPION` only applies when `champion.txt` is absent; plan
   032 #7c installed Q7 the same way);
3. relaunch `scripts/launch_plan058.sh`. gen32 is generated by the new net and its fine-tune
   warm-starts from it (WARM_LR 5e-5, `--freeze-bn`: the scratch run's running statistics exist).
   Add `--loader-workers 4` to `TRAIN_EXTRA_ARGS` (the fine-tune of a 2.7× net would otherwise wait
   on the loader and the GPU in series), and re-measure the generation streams with
   `scripts/perf_gen_steady.sh` on the new net (`CONFIG=cfgs/chance_all_partial_gen.toml`): a slower
   forward may want more streams per sidecar to keep batches full;
4. the anchor stays (`DRIVE_REF` d1k gen04); the first drive check after adoption is also run against
   gen31.

## 9. Results

Not run yet. Registry: [EXPERIMENTS.md#exp072](../registry/EXPERIMENTS.md#exp072).


## Shadow lineage (the user, 2026-10-15)

exp073's B96x8h (96x8 + bigger heads, fine-tuned on gen31-33) is fine-tuned every generation alongside
the loop's 64x6 on the same window and recipe (`train_loop.sh` §3b, `SHADOW_INIT`, models
`models/az_v7/shadow96h_genNN`), never generating. Each row: held-out shards of that generation
(cq-100 target, MC labels), loop net vs shadow, and the value bench paired. Judge again head to head
once the shadow's columns stop improving for ~3 generations.

| gen | loop KL / P(played) / valMSE | shadow KL / P(played) / valMSE | value bench loop → shadow (paired) | shadow train |
|---|---|---|---|---|
| 34 | 0.0535 / 0.6798 / 0.0832 | 0.0866 / 0.6711 / 0.0774 | 0.194 → **0.182 (−6.2%)** | 44 min, step 29000 |
| 35 | 0.0525 / 0.6821 / 0.0791 | 0.0829 / 0.6741 / 0.0746 | 0.194 → **0.183 (−5.7%)** | 41 min, step 11000 |
| 36 | 0.0524 / 0.6841 / 0.0808 | 0.0840 / 0.6756 / 0.0765 | 0.196 → **0.179 (−8.7%)**; shadow's own bench 0.1834 → 0.1792 | 42 min, step 24000 |

Bare policies, shadow96h_gen36 vs gen36 (policy-only drives, tract, 600 pairs per board): 14x7 0.507 ± 0.010, 16x9 0.512 ± 0.010 — the shadow's policy plays slightly better despite its 0.03 higher KL to the 64x6-anchored target, i.e. most of the KL gap was the metric's bias toward the generator. The fair test (its own data) is exp074.

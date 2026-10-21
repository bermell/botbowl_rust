# Plan 067 — exact MC-label top-up, and the 8-vs-16-playout A/B (exp078)

**Status:** top-up built 2026-10-19 (`mc-label --top-up K`, `job label --top-up K`, hub protocol
v21); 8-playout label noise measured directly on a gen44 sample: **0.041 ± 0.004 per sample, 0.036 ± 0.004 with val_value's per-drive weighting = 47% of gen44's val_value** (§3); exp078 scripted and smoke-tested, **not run** — it
runs at the loop's next plateau (the user approved the A/B, 2026-10-19).

## 1. Question

The loop labels every sample's value with the mean of 8 policy-only playouts (plan 056, `mc-label`
under the generator's net). A label on the 1/8 grid carries the playouts' sampling noise. The estimate
that started this: label-noise variance ≈ 0.049 of gen44's val value MSE 0.080, so most of the value
loss would be noise the net cannot fit. Two questions:

1. How big is the noise, measured directly rather than estimated? (§3)
2. Does a net trained on 16-playout labels beat one trained on 8, one generation step from the same
   parent? (§4, exp078)

## 2. The top-up

Plan 062 §5 noted a top-up from n to m playouts is exact. Built here.

**Seed scheme.** Sample `i` of a labelling draws its playouts' dice from its own ChaCha8 stream,
`sample_rng(seed, trajectory seed, drive, i)`; playout `j` seeds its dice from the stream's `j`-th
`u64` and draws nothing else from it. `LabelConfig.skip` (new, default 0) makes a labelling start at
draw `skip`. A top-up of `k` on an `n`-playout labelling runs with the labelling's **own seed** (read
back from its tag) and `skip = n`, so it plays draws `n..n+k`:

- **disjoint** from the prior's draws `0..n` by position in the same stream (a test checks no top-up
  playout seed equals any prior playout seed over 4,800 samples of four trajectories, two drives);
- **exactly** the last `k` playouts of a fresh `(n+k)`-playout labelling. Every playout scores −1, 0
  or +1, so `round(n · label)` is the prior's integer sum, and `(sum_n + sum_k) / (n+k)`, divided in
  f64 and rounded to f32 as `sample_label` does, is the fresh `(n+k)` label **bit for bit** (on one
  backend; across backends the usual argmax flips).

Why not a fresh seed for the extra playouts: a new base seed would give a labelling with the right
distribution, but not *the* `(n+k)` labelling, and `sample_rng` mixes seeds by XOR, so a new base can
collide with another sample's key; continuing the stream is disjoint by construction, reproducible,
and can be topped up again (16 → 24 continues at draw 16).

**Command.** `botbowl-ui mc-label --top-up K --corpus <mc/shard*.jsonl> --model NET --out-dir D` and
`botbowl-hub job label --top-up K ...` (same flags; `--seed` and `--playouts` are refused alongside
`--top-up`: the seed and the prior count come from the labels). Input is a labelled shard (the `mc/`
copy); its trajectories must all carry one `value_label` tag (`shard_prior`); the net must be the
labels' net (file stems must match: `gen43.onnx` and a `gen43.ts` copy are one net); every prior label
must sit on the 1/n grid. A raw shard, two labellings in one shard, another net or an off-grid label is
refused before any playout is played. `--max-steps` must be the original's (the loop uses the default
100000; it is not in the tag).

**Output.**

- `D/<shard>`: the shard with every topped-up sample's `outcome_value` the `(n+k)` mean,
  `value_label = mc_policy(playouts=n+k,model=NET,seed=S)` — the string a fresh `(n+k)` labelling
  stamps — and `value_label_history` = the tags it went through, `|`-separated, oldest first.
  Trajectories without a label (left unlabelled by the labelling) and those that no longer replay are
  written back unchanged and counted as "left unlabelled" in the summary line (the latter keep their
  n-playout label and tag). `prepare` reads only `outcome_value`: nothing downstream changes.
- `D/topup/<shard>` (out of the way of `mc/shard*.jsonl` globs): one line per topped-up trajectory,
  `{seed, drive, board, prior_playouts, added_playouts, prior: [..], added: [..]}` — each sample's
  prior n-playout mean and the mean of the k added playouts. Written and renamed before the shard
  (the shard is the done-marker, as for labelling).

**Hub.** Each shard of a label job now carries its own `LabelConfig` and tag (a top-up job over
several gens continues each gen's own seed). Workers are unchanged apart from `skip` riding in
`Task::Label.cfg` (**protocol v21**: helper workers must rebuild at this commit). The worker returns
the added means; the hub's `ShardWrite` combines them with the prior labels it reads back from the
lines, as the local tool does (`mc_label::apply_topup`).

**Tests** (all pass):

| test | what it pins |
|---|---|
| `botbowl-play mc_label::topping_up_n_playouts_to_m_gives_the_m_playout_label` | 3 → 7 playouts through `skip` + `combine` = the 7-playout label bitwise, on four real positions with scoring playouts |
| `…::combining_two_means_is_the_mean_of_all_playouts_bit_for_bit` | every n, k ≤ 32 and every pair of sums; off-grid and out-of-range priors refused |
| `…::a_topups_playout_seeds_are_disjoint_from_the_prior_and_continue_its_stream` | prior seeds ∪ top-up seeds = the fresh 16's seeds, in order; no top-up seed is any prior seed (4,800 samples) |
| `…::a_topup_records_the_total_and_its_history` | tag = the fresh total's, history chains over two top-ups (8 → 16 → 20, draws continue at 16), refusals |
| `…::a_shard_has_one_prior_labelling_and_the_topup_plays_its_net` | `shard_prior`, `check_topup_model`, tag parsing, record path |
| `…::a_label_config_without_skip_reads_as_a_fresh_labelling` | JSON without `skip` → 0; postcard round trip |
| `botbowl-ui --test mc_label_hub a_topup_through_the_hub_equals_the_local_topup_and_a_fresh_labelling` | the local top-up of a 2-playout labelling by 3 = a fresh 5-playout labelling, trajectory for trajectory (only `value_label_history` differs; orphan and divergent records untouched); records combine to the labels; the hub top-up (items of 3, two workers) writes the local bytes, shards and records; a raw shard is refused locally and at submit |
| `botbowl-hub main::the_loops_label_command_parses` | `--top-up` parses, `--seed`/`--playouts` beside it and `--top-up 0` are refused |
| `botbowl-hub-proto label_frames_roundtrip` | `skip` rides in the frame |

The plain labelling path is unchanged (`skip = 0` draws as before; the existing byte-identity tests
pass untouched).

## 3. The label noise, measured

With the prior mean `a` (n playouts) and the added mean `b` (k playouts) independent estimates of the
state's MC value `μ(x)` with per-playout variance `s²(x)`, `E[(a−b)²] = E[s²](1/n + 1/k)`; with
n = k = 8 the noise variance of an 8-playout label is `mean((a−b)²)/2`. `scripts/label_noise.py`
prints it per sample and with the trainer's per-drive weighting (val_value weights samples
1/len(drive)), per board, with SEs clustered by trajectory, and the mean of `a−b` (must be ~0).

**Measured 2026-10-19** on a quarter of `runs/loopmix16x9v9/gen44/mc/shard4.jsonl` (a held-out
shard): every 4th seed with its next-drive record, 63 trajectories, 3,891 samples, 20x11-mix boards.
Its labels were made under gen43 (sidecar, GPU); the top-up added 8 playouts under the same
`bbnet_mix16x9v9_gen43.onnx` on tract (CPU, 3 threads at nice 19 beside the live loop: 4,985 s, ~47
samples/min), `mc-label --top-up 8`; 0 trajectories left un-topped-up.

| | 8-playout label noise | SE (clustered) |
|---|---|---|
| per sample | **0.0408** | 0.0041 |
| per drive (val_value's weighting) | **0.0364** | 0.0044 |
| prior − added (must be 0) | −0.0023 | 0.0040 |

Per board (per sample): 20x11/7 0.052 ± 0.013, 18x11/7 0.047 ± 0.012, 20x10/7 0.044 ± 0.008, 18x9/6
0.038 ± 0.010, 20x9/7 0.028 ± 0.013. Against gen44's warm-start val_value 0.0774 (its train.log), the
per-drive noise is **47%** of the loss; what is left, ~0.041, is the net's error against the MC value.
The 0.049 estimate was a little high (per sample 0.041 ± 0.004), but the conclusion holds: about half
of val_value is label noise, and a 16-playout label halves it (0.018-0.020).

Cross-check against the frozen 20x11 value bench (48 playouts per state under gen43, from the same
shard; 324 of its states fall in the sample): `(a−b)²/2` = 0.0525 ± 0.0053 on those states, and the
bench's own `mc_se² · 48/8` = 0.0534 ± 0.0022 — the two independent noise estimates agree (both above
the population figure: the bench over-weights contested decisions). `(a − mc₄₈)²` = 0.069 ± 0.007 and
0.071 ± 0.007 for the two halves (expected 0.0525 · 7/6 = 0.061), `((a+b)/2 − mc₄₈)²` = 0.044 ±
0.004 (expected 0.035): the 16-playout mean sits closer to the 48-playout truth, as it should; the
residual excess is within ~2 SE (tract vs sidecar argmax flips and the bench's own playouts' rules).

## 4. exp078: the A/B

`scripts/exp078_label_topup.sh` (env `P` = the plateau generation), one generation step:

- **parent**: the loop's gen P net (gen P+1's generator and warm start);
- **window**: exactly `train_loop.sh`'s `window_shards` for gen P+1 — gens P−1..P+1, train shards
  0-3,5,6, val shards 4,7, each gen's `mc/` copy;
- **arm A**: the loop's 8-playout labels; **arm B**: the same shards topped up by 8 (all train and val
  shards of all three gens; each gen under the net its labels name), via a hub (own on port 13378 with
  a 48-stream in-process local worker, or `EXTERNAL_HUB`) or locally (`TOPUP_VIA=local`);
- **recipe** (both arms, the loop's): `prepare --policy-target cq --tau 100 --value-blend 1.0`;
  `bbnn.train --init genP.pt --lr 5e-5 --epochs 3 --eval-every 1000 --eval-at 250,500 --select-on
  combined --value-weight 0.36 --per-drive-value-weight --freeze-bn`, plus one shared `--seed 78`
  (the loop runs unseeded; a shared seed makes the arms see the same batches and flips, so the labels
  are the only difference); `.ts` exported for each (plan 066);
- **judged by**: absorption vs the parent on gen P+1's raw held-out shards (the loop's probe) and on
  the window's val set under 8- and under 16-playout labels; both value benches (`v9_gen01_val`,
  `v9_20x11_val`), paired vs the parent; the net check (C1, ladder 64/250/1000/4000, boards
  16x9,20x11) on gen P+1's held-out shards; with `DRIVES=1`, paired drives B vs A (C1 both sides, both
  contested sets, SPRT 0.5:0.55, cap 800, as the loop's benchmark). If the loop later trains its own
  gen P+1, it is added to the probes and benches as an unseeded replicate of arm A.

**The fair-selection rule.** Each fine-tune restores its best checkpoint on `val_policy +
val_value`. If each arm selected on its own val labels, B's `val_value` would be measured against
labels with half A's noise: its numbers would sit ~noise/2 lower (not comparable), and — the real
confound — its restore statistic would be less noisy, so B would also get better *model selection*,
not only better training data. So **both arms restore on the same prepared val set**: the window's
val shards with **16-playout** labels (`SEL=16`, default; `SEL=8` = the loop's own val labels). Label
noise in the val target adds the same constant to every checkpoint's expected val loss, so either
choice ranks checkpoints without bias; 16 halves the noise in the ranking for both arms, and it is
independent of both arms' training data (held-out games). The arms are then compared on numbers
selection never saw: the value benches (48-playout MC truth on frozen states outside every window),
absorption on both label sets, the net check and drives.

**Reading it.** B wins if its value bench RMS is lower (paired vs the parent, both benches) without a
policy cost in absorption, its net check stays monotone, and (with drives) it is not worse than A.
Adopting 16 playouts in the loop doubles the mc-label phase (§6); a top-up of only the newest
generation each step (8 → 16 when it enters the window) is the cheaper variant to consider if B wins.

## 5. How to run

At a plateau P (the loop has labelled gen P+1): touch `runs/loopmix16x9v9/STOP` while gen P+1's
mc-label runs, so the loop exits "before genP+1 prepare"; then, from the main checkout at a commit with
this plan merged:

```sh
P=46 nohup scripts/exp078_label_topup.sh > /dev/null 2>&1 &        # + DRIVES=1 for drives B vs A
tail -f runs/exp078/status.md
```

It waits for gen P+1's `.mc_labelled` and for the loop to exit (`WAIT_LOOP=1`), builds into
`target/exp078` (never the loop's `target/20x11`), and resumes from whatever it finished. It never
writes into the loop's run or models dirs (a net with no `.ts` is copied to `runs/exp078/nets/` and
exported there) and never relaunches the loop. With the loop stopped, `PORT=13337 HUB_BIND=0.0.0.0`
lets the helper workers (which dial 13337) rejoin and share the top-up — they need this commit
(protocol v21); stop the script's hub before relaunching the loop (`launch_plan065.sh` refuses a busy
13337).

**Smoke test** (2026-10-19, worktree, commit 1a3849f, out of tree in `/tmp/exp078_smoke`, the live
loop running alongside): P=43 (window gen42-44), `SMOKE_LINES=1` (one trajectory per shard),
`TOPUP=1`, own hub on 13379 with 4 in-process streams, `TRAIN_DEVICE=cpu PROBE_DEVICE=cpu
TRAIN_EXTRA="--max-steps 20 --eval-every 10 --eval-at 5"`, `VB_ROWS=40`, net check ladder 64 at 2
overrides, `DRIVES=1 DRIVE_CAP=4`. Every step ran and wrote its status line: the three gens topped up
through the hub under their own nets (gen41/42/43; gen43 had no `.ts` and was exported into
`nets/`), 0 trajectories left un-topped-up (gen42/43 are 16x9-capacity corpora, read by the 20x11
build), the label-noise table, both arms trained and exported to `.ts`, absorption on raw / val8 /
val16, both value benches (paired vs the parent, in-process: `NN_INPROC loaded`), the net check per
arm (one in-process `override-audit --ladder`), drives B vs A on both sets (2 pairs each), `done`; a
rerun skips every finished step. 24 min including the build.

## 6. Cost (this box: GTX 1060, the loop stopped)

| step | estimate |
|---|---|
| top-up, 3 gens × ~128k samples × 8 playouts | at the mc-label rate of 510-650 samples/min (8 playouts, full GPU): **~10-12.5 h** (gen44's label phase: 127.6k samples in 319 min) |
| prepare | minutes per arm; ~10 GB per arm's prepared train set, deleted after training |
| train, 2 arms | 40-70 min each (gen42-44's train phases) |
| absorption ×3, value benches ×2 | minutes |
| net check, 2 arms | ~100 min each (the loop's gen45 net check: 99 min) |
| drives B vs A (optional) | 2-4 h of hub-worker time (the loop's drive benchmark) |

About 15-17 h without drives, 18-21 h with; the loop is stopped for all of it (or shares the GPU and
both run slower). Disk: ~3.6 GB of topped-up shards plus ~12 GB transient.

## 7. Risks

- **Same net, same `max_steps`, or the top-up is not the (n+k) labelling.** The net is checked by
  file stem; `max_steps` is not in the tag (the loop always used the default).
- **Backend.** A top-up on another backend than the original labels (the loop's gen ≤ 44 labels came
  from the Python sidecar; exp078 runs in-process; a helper on tract) is still the policy's MC value
  with independent dice, but not bitwise the fresh labelling: rare near-tie argmax flips. The `offset`
  in `label_noise.py` (mean of prior − added) would show a systematic difference.
- **Engine drift.** A trajectory that no longer replays at the current commit is not topped up and
  keeps its 8-playout label (counted on status.md per gen). At the time of writing the one engine
  commit since gen42's corpus is plan 065's (kickoff time-out fix, reached only on 7-player boards),
  and the smoke's gen42-44 trajectories all replayed; check the per-gen count before reading arm B.
- **Protocol v21** locks out helper workers until they rebuild.
- **One step, one seed.** A single generation step at one trainer seed resolves only large effects;
  exp069 measured re-run noise of these numbers with a tau-100 replicate. The value benches are paired
  (dMSE ± SE), which is where a label-noise effect should show first.

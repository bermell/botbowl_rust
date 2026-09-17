# Plan 036 — Stop the value head overfitting the window

**Status:** W1-W5 implemented, default-off, 2026-09-16 (`7b4b428`, branch
`plan-036-value-overfit`). W6 deferred on purpose. Experiments not yet run — waiting on a window
from the from-scratch run `runs/az14x7v6`. Ranked into plan 032's queue as a training-side item;
it costs no generation games until the last step.

### What exists

| Item | Flag | Default | Where |
|---|---|---|---|
| W1 value loss weight | `train.py --value-weight` | 1.0 | `train.py` backward pass only; `--select-on` stays unweighted |
| W2 weight decay | `train.py --weight-decay` | 0.0 | AdamW, BN + bias exempt (18 decayed / 46 exempt on the 64x6 tower); stays on plain Adam at 0.0 |
| W3 blended value target | `prepare --value-blend L` | 1.0 | `targets::value_target_blended`; manifest `value_target` records the blend |
| W4 per-drive value weight | `train.py --per-drive-value-weight` | off | `prepare` always writes `weight.npy`; no re-prepare needed to try the arm |
| W5 exact dedup | `prepare --dedup` | off | 128-bit digest of `(spatial, global)` |
| Selection rule | — | always on | every training log now prints the val_policy-only optimum next to the restore step |

Two properties make the A/B mean anything, and both are checked: `train.py` with no new flag
reproduces master digit for digit on the same seed, and `prepare` with no new flag writes a
`value.npy` bit-identical to `--value-blend 1.0`. `weight.npy` sums to exactly 1.0 per drive, before
and after dedup drops rows.

`nn_schema_version` deliberately does **not** move for W3/W4, against what the W3 section below
says: that version gates the *tensor layout* a checkpoint is compatible with, and a v6 net loads a
blended corpus perfectly well. Bumping it would have falsely invalidated the corpus being generated.
The manifest's `value_target` / `value_blend` / `dedup` / `value_weight` fields are what identify
which corpus a net was fitted to.

### Runner

`scripts/exp036_value_overfit.sh` (E1-E3) and `scripts/exp036_report.py` (the table). Every arm is
resumable and niced, since it shares the box with a generation run. The report prints
`(restore_step, val_policy@restore, val_value@restore, val_value@end)` plus `drift`
(val_value@end - @restore, E2's signal) and `pgap` (policy optimum - restore, the selection-rule
question).

### Which window — the regime caveat

The symptom this plan is about is a property of **warm-started fine-tunes**: "every warm-started
fine-tune since gen04 restores at epoch 0-2 of 10". `gen01` of a from-scratch run trains from random
init at 1e-3, where mechanism 1 (two thirds of the window already fitted) is absent by construction
and only mechanism 2 (value-label multiplicity) is in play. So gen01 is a real but *partial* read —
it measures W1-W5 against label noise alone. The full read needs gen02+ with `--init` at 2e-4.
Run both; the runner stamps which regime it used into its log.

## Problem

Every warm-started fine-tune since gen04 restores its best checkpoint at epoch 0-2 of 10
(plan 030:185, plan 031:427-430, plan 032:1234). The remaining epochs are burned compute, and
because `SELECT_ON=combined` restores at the *value* minimum, the policy gains from epochs 2-9
(val_policy still falling in every generation) are discarded with them.

The premise "4800 highly correlated trajectories → the net memorises which run is which" is half
right. The facts, from `scripts/train_loop.sh`, `botbowl-ui/src/dataset.rs`,
`botbowl-nn/src/bin/prepare.rs`, `train/src/bbnn/train.py`:

| Quantity | Value | Where |
|---|---|---|
| Drives per generation | 8 × 600 = 4800 | `train_loop.sh:71,232` |
| Decisions per drive | ~30, every one written | `dataset.rs:314-349`, plan 031:297 |
| Samples per generation | ~118k | plan 032:990 |
| Training window | 3 generations, ~350k samples, 6/8 shards train | `train_loop.sh:142,236-237` |
| Steps per epoch | ~8k at batch 32 | `train.py:334` |
| Fraction of window new to the warm-started net | 1/3 | window G-2..G; net trained on G-3..G-1 |
| Exact duplicate rows | 12.8% | plan 031 D3 |
| Value target | drive score delta, one scalar per drive, clamped [-1,1] | `targets.rs:215-220`, `botbowl-data/src/lib.rs:300-318` |
| Loss | `policy_ce + value_mse`, unweighted | `train.py:87-98, :267` |
| Regularisation | BatchNorm only. No weight decay, dropout, clipping, LR schedule | `train.py`, `model.py` |
| Optimiser | Adam, fixed lr 2e-4 warm / 1e-3 scratch | `train_loop.sh:175-176` |
| Model | 6 blocks × width 64, ~0.52M params | `model.py:34-97` |

Two separate mechanisms:

1. **Low marginal information.** Two thirds of every window was in the previous fine-tune's
   training set. The warm-started net is already at its optimum on that part; only ~80k train
   samples (the newest generation's 6 shards) are new. One epoch over the window is ~3 epochs
   over the novel data. Convergence inside epoch 0 is expected, not pathological.
2. **Value-head noise fitting.** The value label is one high-variance scalar (dice, 1000-iter
   self-play) copied to all ~30 positions of the drive. The value head effectively sees 4800
   labels per generation, each ~30×, and learns drive identity → label. The policy target varies
   per position and does not show this; val_policy keeps falling to epoch 9. This is the
   correlation the question is about, and it is confined to the value head.

The `chosen`/top-1 and the y-flip augmentation are irrelevant here (plan 031:605-630 showed the
flip is regularisation for the policy, not a correction).

## Comparison to AlphaZero / successors

| Knob | AlphaZero (Silver 2017/18) | KataGo / Leela | Here (now) | Here (proposed) |
|---|---|---|---|---|
| Value loss weight | 1.0 (AGZ loss `(z-v)² − πᵀlog p + c‖θ‖²`) | Leela Zero divides the MSE by 4 (= 0.25) after seeing value over-fit; lc0 configs run 0.25-1.0 | 1.0 | **0.25** (tune 0.1-1.0) |
| L2 / weight decay | c = 1e-4 (SGD L2) | lc0 1e-4; KataGo ~3e-5 | none | **1e-4 AdamW-style** (tune 3e-5-3e-4) |
| Value target | game outcome z only | lc0 z only; KataGo z + score; MuZero/Gumbel n-step bootstrap with search value | drive outcome | **λ·outcome + (1-λ)·root_search_value**, λ=0.5 |
| Positions per game used | AlphaGo (2016) fan-in: **one position per game** for the value net, explicitly "to avoid overfitting to correlated positions" (Nature 2016 §Methods). AGZ/AZ: all positions, but ~25M-game windows | all positions, 500k-game window | all ~30, 14.4k-drive window | all positions; **value loss weighted 1/len(drive)** |
| Window | last 500k games (AGZ), ≈ 1M (AZ) | 250k-1M games | 3 gens = 14.4k drives | unchanged (larger window does not help mechanism 2, see below) |
| Steps per generation | continuous; 700k × 2048 samples over ~1B generated positions, i.e. **each position seen ≈ 1-2 times ever** | lc0 ≈ 1 epoch/window per net | 10 epochs, restore at 0-2 | **2 epochs** (or `--max-steps 16000`) |
| LR | SGD 0.2→0.02→0.002→0.0002 | cosine / step | Adam fixed | keep Adam; cosine over the shortened budget optional |
| Optimiser state | carried | carried | dropped on warm start | unchanged (not the cause) |
| Dedup | none | none | none | drop exact (spatial, global) dupes at prepare (12.8%) |

The one-line summary of the comparison: AlphaZero trains each position once or twice, and it has 1e-4 L2. We train every position 10× and have no L2. Everything else about our
setup is closer to AZ than the epoch count suggests.

## Work items

Each item is one flag, default off, so every step is an A/B against the current loop. Numbers in
**bold** are my best guess; ranges are what to sweep if the first guess does not move the value
minimum later.

### W1 — Value loss weight (`train.py`)

`(pl + value_weight * vl).backward()`. New `--value-weight`, default 1.0 (current behaviour);
loop passes `VALUE_WEIGHT="${VALUE_WEIGHT:-0.25}"`.

- Guess **0.25** (Leela Zero's MSE/4 after the same symptom). Range 0.1-1.0.
- Note the restore criterion `vp + vv` should stay *unweighted* — the criterion measures the
  heads, the weight is only about gradient balance.
- Risk: the value head under-trains and `nn-value` leaf scores degrade. Check with
  val_value at restore, not just the position of the minimum.

### W2 — Weight decay (`train.py`)

Switch `torch.optim.Adam` → `torch.optim.AdamW(weight_decay=wd)`. New `--weight-decay`, default
0.0; loop passes **1e-4**. Range 3e-5-3e-4. Exclude BatchNorm and bias params from decay
(standard; the model has BN everywhere).

- AZ's 1e-4 is an SGD L2 coefficient at lr 0.2; AdamW decoupled decay at lr 2e-4 is a different
  unit, so 1e-4 is a starting point, not a translation. If 1e-4 does nothing visible, jump to
  1e-3 before giving up.

### W3 — Blended value target (`targets.rs`, `prepare.rs`)

`value_target = λ·outcome + (1-λ)·clamp(root_value / TD_POINTS, -1, 1)`, both mover-signed.
`Sample::root_value` is already stored (Home-centric Q points, 1000 = one TD, `lib.rs:130-132`).
Samples with `root_value == None` fall back to λ=1.

- New `prepare --value-blend <λ>`; default 1.0 (current). Loop passes **0.5**.
  Range 0.3-0.7. λ=0 is MuZero-style pure bootstrap and is *not* recommended: the search value is
  biased by the same net we are training (self-confirmation), and at 1000 iters plan 031 D2
  showed the search is far from converged.
- Bumps `NN_SCHEMA_VERSION`; manifest `value_target` becomes
  `"mover_signed_blend(outcome=λ,root=1-λ)"`.
- Expected effect: the biggest of the four. The outcome label has variance ≈ 0.25 per drive
  (Bernoulli-ish TD/no-TD at 0.6-0.8 conversion); root Q averages ~1000 leaf scores. Halving the
  label variance roughly halves what the head can over-fit.
- Caveat: the value head's *calibration* changes — its output is no longer "expected score
  delta" but a blend. `nn-value` leaf scoring and the eval-side `td_rate.py` are unaffected
  (they consume the sign/ordering), but note it in the manifest.

### W4 — Per-drive value weighting (`prepare.rs`, `data.py`, `train.py`)

Write a new `weight.npy (N,) f32` = `1 / len(drive)` for the value term (policy term keeps
weight 1). `compute_losses` computes `(w * (v - target)²).sum() / w.sum()`, so the loss scale
stays comparable to today's plain MSE and W1's weight means the same thing with W4 on or off. Drive length is known at prepare time since `traj.samples` is one drive.

- This is AlphaGo-2016's "one position per game" done without discarding data: in expectation
  each drive contributes one unit of value gradient per epoch.
- Guess: **on**, exponent 1 (`1/len`). Alternative `1/sqrt(len)` if val_value at restore gets
  worse (the head sees less signal from long drives, which are the interesting ones).
- Interaction with W3: W3 reduces per-label noise, W4 reduces label multiplicity. They are
  independent and should both be on. If W3 alone moves the value minimum past epoch 3, W4 is
  optional.

### W5 — Exact dedup at prepare (`prepare.rs`)

Hash `(spatial, global)` bytes, keep the first row. Plan 031 D3: 12.8% of rows, all
intra-drive block/push/reroll chains with 100% disjoint legal sets and zero value variance
inside a group, so keeping one loses nothing. Mostly a compute saving; small regularisation
effect. Default on, `--no-dedup` to revert.

### W6 — Shrink the training budget (`train_loop.sh`)

`EPOCHS=10` → **2**, keep `EVAL_EVERY=2500`. Or `--max-steps 16000`, which decouples the
budget from window size (the window shrinks when a generation is skipped). Under AZ's
"each position seen 1-2× ever" and our 3-gen window, 1 epoch/generation is the direct analogue;
2 is a hedge because W1-W4 should move the value minimum later and the policy was still
improving at epoch 9.

- Do this **after** W1-W4 are measured, not before: shrinking first hides whether the fix
  worked. Then set the budget to ~1.5× the observed restore step.
- Optional: cosine LR over the shortened budget. Not needed to fix the problem; skip unless
  the restore step lands right at the end of the budget (= LR too high at the end).

### W7 — More independent drives (generation-side, expensive)

Only after W1-W6. Twice the drives per generation halves mechanism 2's per-label multiplicity
*ratio* only if positions per drive stay fixed, and it doubles generation time, which is the
loop's cost driver. Longer trajectories (full halves instead of one drive) would make things
worse. If more data is bought, buy it as drives, keep `--mode random-start`.

## Selection rule

Keep `SELECT_ON=combined` but add a second printed criterion: the checkpoint that minimises
`val_policy` alone. If after W1-W4 the two restore points still differ by >5k steps, split the
restore: keep the trunk+policy from the policy optimum and the value head from the value optimum
(the heads are separate modules after the shared tower, `model.py:57-63`). Not planned until the
data says it is needed.

## Experiments

All on the *current* gen's window, offline (`train.py --max-steps` on a fixed prepare output),
no games needed until E4. Report `(restore_step, val_policy@restore, val_value@restore,
val_value@end)`. Baseline = current loop settings.

| # | Arm | Success signal |
|---|---|---|
| E1 | W1 alone at 0.25; W2 alone at 1e-4; W1+W2 | value minimum moves ≥ 2× later in steps; val_policy@restore improves |
| E2 | W3 at λ ∈ {0.3, 0.5, 0.7} on top of E1's winner | val_value@end − val_value@restore shrinks (less drift) |
| E3 | + W4, + W5 | same, plus wall-clock saving from W5 |
| E4 | Winner vs baseline, 4800-drive generation each, 600 games vs `scripted` (plan 032 sizing: SE 0.02) | points ≥ +0.03; anything less is a compute win only, still worth keeping |

The `val_*` numbers select the arm for E4 but do not decide it (plan 032 ground rule).

## Results (2026-09-17) — E1-E3 run, adopted

Two windows of `runs/az14x7v6`, nine arms each, offline: `gen01` (random init, 1e-3 — mechanism 2
only) and `gen02` (warm start from gen01 at 2e-4 — the regime the symptom was observed in).
Raw tables in `runs/exp036/{gen01,gen02}/`.

**The premise is confirmed.** The loop's own fine-tunes, independent of these arms: gen01 (from
scratch) restored at epoch 2, gen02 and gen03 (both warm-started) restored inside **epoch 0** of 10.

**`val_value` is not comparable across W3 arms** — the blend changes the label, and its variance
falls from 0.495 to 0.180 on the gen02 val set. Everything below is normalised: `R²` = 1 −
val_value/var(label), `drift` = (val_value@end − @restore)/var(label).

| arm | gen02 restore | R² | drift | vs base | gen01 vs base |
|---|---|---|---|---|---|
| baseline | 10000 (ep 1) | 0.331 | 0.157 | 1.00× | 1.00× |
| W1 value-weight 0.25 | 17500 | 0.341 | 0.107 | 1.75× | 1.67× |
| W2 weight-decay 1e-4 | 10000 | 0.331 | 0.147 | **1.00×** | **1.00×** |
| W1+W2 | 17500 | 0.342 | 0.122 | 1.75× | 2.33× |
| W3 λ=0.5 | 57500 (ep 7) | 0.573 | 0.005 | 5.75× | 3.67× |
| W3 λ=0.3 | 70000 (ep 9) | 0.682 | 0.000 | 7.00× | 4.33× |
| +W4 per-drive | 57500 | 0.591 | 0.003 | 5.75× | 4.00× |
| +W5 dedup | 47500 | 0.591 | 0.005 | 4.75× | 3.67× |

**Adopted from gen04 (`train_loop.sh`): W1 at 0.25, W3 at λ=0.5, W4 on.**

- λ=0.5 over λ=0.3 despite 0.3's better R²: that label is 70% the net's own search output, so part
  of the fit is the self-confirmation this plan's W3 section warned about. The R² gain is not
  evidence of better positional judgement.
- **W2 rejected.** Inert on both windows — identical restore step and R² to baseline — and added
  nothing on top of W1. AlphaZero's 1e-4 is an SGD L2 coefficient; it does not transfer to AdamW at
  2e-4. Retry at 1e-3 or drop.
- **W5 rejected.** 2.7% duplicates here, not plan 031 D3's 12.8% — that figure was measured on the
  old identical-lineman corpus, and varied players make exact state collisions rare. It cost policy
  quality on both windows (val_policy 1.6445 vs 1.6124; 1.8059 vs 1.7771). Delete D3's 12.8% from
  your priors for this corpus.

### W6 is rejected, for the opposite of the expected reason

The plan assumed epochs 2-9 were burned compute carrying discarded policy gains. Both halves are
wrong on this data:

1. After W1+W3+W4 the restore moves to **epoch 7 of 10**, and λ=0.3 was still improving when the
   budget ran out at epoch 9. The plan's own rule — budget ≈ 1.5× the observed restore — gives ~86k
   steps against 72.5k for ten epochs of the gen02 window, i.e. the budget should grow, not shrink.
   The fix did not save compute; it converted wasted compute into useful compute.
2. There were no policy gains to discard. gen02 baseline `val_policy` reaches 1.6290 by step 10000
   and then sits at 1.626 ± 0.001 for 60,000 more steps. The large `pgap` values in the raw tables
   are noise picking a winner off a plateau, not a head still learning — so the "split the restore
   between the heads" idea in **Selection rule** above is answered: not needed, there is nothing to
   split.

### Follow-on: the budget became the binding constraint (2026-09-17)

gen04, the first generation trained with the adopted recipe, restored at **step 90000, epoch 9 of
10** — against gen02's and gen03's epoch 0 — with the policy optimum a further 5000 steps out. The
overfit is gone and the budget is now what stops training. `EPOCHS` raised 10 → 15 (plan 036's own
1.5× rule gives ~135k steps against ~95k for ten epochs of that window). This is the exact opposite
of W6 and follows from the same rule.

### E4 — run 2026-09-17, recipe confirmed

**new 0.605 ± 0.029 paired (110 pairs, 220 games), z = 3.62, p = 0.0003, 95% CI [0.548, 0.662].**

Head to head rather than the plan's "600 games vs `scripted`" each: the two nets share a window, a
warm start, a seed and a batch order, so the difference is estimated directly for half the games.
Both arms trained at 10 epochs on purpose, so `EPOCHS=15` could not confound the value-target
comparison. Candidate `new` = blend 0.5 + value-weight 0.25 + per-drive; opponent `base` = the old
recipe on the same gen03 window from the same gen02 warm start.

The effect is **+0.105** — three times plan 036's +0.03 bar and twice the +0.05 the 600 games were
sized for. The stopping rule was pre-committed at n=200 in `036-e4-preregistration.md` before the
data was looked at again; it landed at 220 because the enforcing watcher polls every 120 s and four
parallel games finish inside a poll gap. That is poll granularity on a fixed schedule, not
look-and-decide, and the overshoot carries no bias.

Per the pre-registered decision table the CI excludes 0.50 from above: **recipe confirmed, E4
closed.** The adopted knobs stay.

### Still open

Nothing in this plan. The value-target recipe is adopted (`train_loop.sh`, from gen04) and measured.
`EPOCHS` follow-ups are tracked in the loop's own comment, not here.

## Non-goals

- Carrying Adam moments across warm starts. Not the mechanism; the 2e-4 warm LR already
  handles the fresh-moment step.
- Larger window. Helps mechanism 1's ratio marginally, does nothing for mechanism 2, and
  stale generations are weaker data.
- Dropout. AZ-family nets do not use it; BN + weight decay is the standard recipe.
- Position-level train/val split. The current shard split is correct; a position split would
  hide the very over-fit this plan is about (`train.py:152-155`).

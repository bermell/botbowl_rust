# Plan 056 — value labels: train the value head on something closer to the truth

**Status:** Written 2026-10-05 from plan 055's override audit. Arms done 2026-10-06: **MC-averaged labels (arm F) win**: benchmark RMS −15%, its search beats the control's 0.533 head to head and its policy 0.582 at 1000 descents. **Adopted: the loop restarted 2026-10-06 with MC labels (§7).** Drives only (plan 051);
training uses Gumbel-generated data only. Judged first on a Monte Carlo value benchmark (minutes,
no games), then by plan 055's budget criterion ("more search never hurts").

## 1. Why

Plan 055's audit (exp065) measured the value head against Monte Carlo truth for the first time,
with the dice noise averaged out over 64 paired playouts per position:

| | g_gen05 | d1k gen04 |
|---|---|---|
| per-state error, RMS (MC sampling noise removed) | **0.28 TD** | **0.26 TD** |
| bias, V(s) − MC(s) | **+0.11** | +0.05 |

The search acts on far smaller differences: two-thirds of its overrides of the prior are on Q gaps
under 0.1 TD, and those realise nothing; overrides on gaps of 0.1 TD and up realise +0.06 to
+0.2. The value head's error is what limits the search, so it is what limits self-play.

The label is the suspect. Every stored position is trained toward
**0.5 × drive outcome + 0.5 × the search's root value at that position** (`--value-blend 0.5`,
plan 036):

- **The outcome** (+1 the mover scores, −1 the opponent scores, 0 nobody) is the truth but one dice
  roll of it. A good position that failed one dodge is labelled −1.
- **The root value** has low noise, but it is self-referential (the net trains on its own search's
  view of the same state). In every corpus so far it came from **minimax** search, which plan 031
  D1 measured +0.10 optimistic. Half of every label inherited that.

g_gen05, five generations further down that chain than d1k gen04, is twice as optimistic with the
same RMS. That pattern points at the minimax root values in the labels.

**Caveat on the bias.** MC is the value under the *policy's* continuation, while the value head is
trained on outcomes of *search* play, which is slightly stronger (exp064: the mean-backup search
beat its policy by ~0.03 points of a drive, ≈ +0.05 TD). So up to ~+0.05 of bias is expected from
the mismatch alone. RMS is the cleaner metric; bias above ~+0.05 is the part to remove.

## 2. The benchmark (build first)

Freeze the exp065 states as a value benchmark:
- g_gen05 on gen06: ~4100 distinct decision states.
- d1k gen04 on gen01: ~4200 distinct decision states.

Each state's MC(s) is the policy-only drive outcome from s (`policy.mc_mean` with its SE, MC under
that net's policy), and it has a reconstructable state (decision id → `override-audit`'s replay).

A small tool scores any net's V(s) on these states and reports:
- RMS error after subtracting MC sampling variance;
- bias;
- both split by board and by phase (turn start / mid-turn / mid-activation, where g05's bias is
  worst at +0.12).

Implementation: an `override-audit --value-only --decisions-from rows.jsonl` mode (replay, one
forward per state, no search, no playouts), plus a summariser line. Minutes per net.

**Leakage rule.** No arm trains on gen06 (g05's benchmark corpus) or gen01 (d1k's). The g05
benchmark is the primary one.

**Built (2026-10-05, `5a28052`):** `scripts/value_bench_freeze.py` (exp065 rows → one line per
state), `botbowl-ui value-bench` (replay, one forward), `scripts/value_bench_summary.py` (RMS with
MC noise removed, bias, scatter = RMS left after the bias, by board and phase, every net paired
state by state against the first), and the `scripts/value_bench.sh` wrapper. It is a separate
subcommand, not an `override-audit` mode. **Seconds per net.** The self-check reproduces each
benchmark net's own V(s) on every state (|V − v_ref| = 0).

The figures are on the audit's sample, unweighted (overrides over-represented), so they sit a
little above the audit's population-weighted ones (g05: 0.30 / +0.13 here vs 0.27-0.28 / +0.11).

**Baseline, every net of both lineages on the g05 benchmark (MC under g05's policy; 4097
states):**

| net | RMS | bias | scatter | paired vs g05 (dRMS) |
|---|---|---|---|---|
| g_gen05 | 0.299 | +0.132 | 0.268 | — |
| g_gen01 | 0.291 | +0.108 | 0.270 | −2.7% |
| g_gen02 | 0.283 | +0.095 | 0.267 | −5.2% |
| g_gen03 | 0.272 | +0.064 | 0.264 | −9.1% |
| g_gen04 | 0.273 | +0.074 | 0.262 | −8.8% |
| g_gen06 † | 0.283 | +0.118 | 0.257 | −5.2% |
| g_gen07 † | 0.278 | +0.098 | 0.261 | −6.9% |
| d1k gen01 | 0.273 | +0.058 | 0.267 | −8.8% |
| d1k gen02 | 0.276 | +0.059 | 0.269 | −7.8% |
| d1k gen03 | 0.275 | +0.070 | 0.266 | −8.0% |
| d1k gen04 | 0.285 | +0.082 | 0.273 | −4.6% |

† trained on gen06, the benchmark's corpus (one outcome per state seen); shown for the trend only.
Every paired difference is ≥ 4 SE. On d1k gen04's benchmark (4239 states, its own policy):
d1k 0.272 / +0.074, g05 0.279 / +0.105 (paired +2.9% RMS for g05).

What this says before any arm has trained:
- **Scatter is the same everywhere (0.26-0.27).** The nets differ almost only in bias. A label
  change that only removes bias can take RMS from 0.30 to ~0.27, but not below; reaching the
  ~0.1 TD gaps the search acts on means cutting the scatter, which no net in either lineage has
  done.
- **g_gen05 is the worst value head of both lineages, on its own states.** The bias does not
  climb generation by generation (g03 +0.06, g04 +0.07, g05 +0.13, g06 +0.12): it moves by
  ±0.03-0.06 between neighbouring fine-tunes. That looks more like fine-tune noise (plan 054's lr
  2e-4 restarts) than a steady drift from the labels, though both could be at work.
- **The blend label is optimistic on the data itself:** on gen07 (one shard) the blend-0.5
  label's mean is 0.293 and the outcome's 0.254, so the stored (minimax) root values overstate
  the drive outcome by about +0.08 on average.

## 3. Arms

Each arm fine-tunes **g_gen05** with plan 054's corrected train step:
- lr 5e-5, `--freeze-bn --init-candidate --eval-at 250,500 --select-on policy`, 3 epochs, cq τ=100.
- The policy target is untouched, so only the value label differs between arms.
- Data: gen07, which was generated by g_gen06 and is unseen by g_gen05 and not in the benchmark.
  The "big" variant adds gen02-05 (g05 trained on gen03-05 with the old label).

| arm | value label | needs |
|---|---|---|
| A control | blend 0.5 (today's) | — |
| B | blend 1.0: the drive outcome alone | — (existing flag) |
| C | TD(λ), λ = 0.8 | `prepare --value-td-lambda` |
| D | TD(λ), λ = 0.95 | same |
| E | the winner of B-D at `--value-weight 1.0` instead of 0.25 | — |
| F (conditional) | MC-averaged: mean of K = 8 policy-only playouts from each stored position | a labelling pass (the audit's `play_out`); GPU-hours per generation |

**TD(λ)**, for a sample at decision t of a drive with later decisions t+1 … t+K and outcome z, all
in the frame of the mover at t:

label(t) = (1 − λ) · Σ_{k=1}^{K} λ^{k−1} · v(t+k) + λ^{K} · z

v(t+k) is the stored `root_value` of the later decision, sign-flipped when the mover changes.
λ = 1 is the pure outcome (arm B); λ = 0 is the next decision's search value. Intermediate λ
averages many later estimates, each of which already knows the dice that fell after t, and always
ends in the real outcome. It is less self-referential than today's blend (no state's label uses
its own search value) and less noisy than the outcome alone. On existing corpora the later
`root_value`s are minimax-era and still optimistic, so TD(λ) dilutes that bias where B removes it.
On data generated under the mean backup they are honest, which favours TD(λ) from then on.

### Results (exp067, `runs/exp067/`; g05 benchmark, 4097 states)

| arm | label | RMS | bias | scatter | paired vs A | absorb vs g05: dP(played), dKL | dvalMSE vs raw outcome (gen07 held-out) |
|---|---|---|---|---|---|---|---|
| g_gen05 | — | 0.299 | +0.132 | 0.268 | — | — | — |
| A | blend 0.5 | 0.280 | +0.099 | 0.262 | — | +0.0007, −0.009 | −0.0043 |
| B | blend 1.0 | 0.278 | +0.060 | 0.271 | dRMS −0.6% (dMSE −0.0010 ± 0.0007), dbias −0.039 ± 0.001 | +0.0006, −0.009 | −0.0072 |
| C | TD(λ) 0.8 | 0.274 | +0.074 | 0.264 | dRMS −1.9% (dMSE −0.0030 ± 0.0006), dbias −0.025 ± 0.001 | +0.0007, −0.009 | −0.0016 |
| **D** | **TD(λ) 0.95** | **0.269** | **+0.051** | **0.264** | **dRMS −3.8% (dMSE −0.0059 ± 0.0006), dbias −0.047 ± 0.001** | +0.0006, −0.009 | −0.0050 |
| E | TD(λ) 0.95, value weight 1.0 | 0.271 | +0.063 | 0.264 | dRMS −3.0% (dMSE −0.0046 ± 0.0007), dbias −0.036 ± 0.001 | +0.0003, −0.008 (dtop1 +0.0006 vs D's +0.0022) | −0.0054 |
| **F** | **MC-averaged (8 policy-only playouts per gen07 train sample under g_gen05), blend 1.0** | **0.236** | **−0.001** | **0.236** | **dRMS −15.5% (dMSE −0.0223 ± 0.0015), dbias −0.099 ± 0.002; vs D −12.1%** | +0.0010, −0.009 (dtop1 +0.0028) | +0.0138 |

- **A alone takes 6.5% off g_gen05** (paired, ≥ 15 SE): a fine-tune on fresh data with plan
  054's train step, label unchanged. g05's extra bias was mostly that one fine-tune.
- **B removes the excess bias, but not the error.** The bias above +0.05 goes from 0.049 to 0.010,
  which passes §4's bias rule. RMS barely moves because the scatter rises by about as much: the
  outcome label is the noisier label. Policy unchanged in both.
- **D (TD(λ) 0.95) is the best label.** It removes the excess bias as B does (+0.051, by phase
  +0.04 to +0.08) but keeps A's scatter, so RMS falls 3.8% against the control (~10 SE). That is
  short of the 5% RMS bar, but it passes the bias rule, and the policy did not move. C (λ 0.8)
  leans more on the minimax-era root values and lands between A and D.
- **E (value weight 1.0) is worse than D** on RMS and bias, and the policy absorbs a little less.
  Keep the weight at 0.25. **D is the winner** on the benchmark.
- **F (MC-averaged labels) is the first label to move the scatter: 0.264 → 0.236**, and the first
  net in either lineage below 0.26. RMS falls 15.5% against the control and 12% against D, bias is
  ~0 in every phase and board, and the policy did not move (slightly better). The labelling cost
  ~2 h for six shards (63k samples × 8 playouts), on a GPU shared with two other jobs (~22 min per
  shard; ~7 min per shard alone).
- **The caveat.** F's labels and the benchmark's truth have the same definition (policy-only
  playouts under g_gen05; different states and dice). The zero bias is partly by construction: F
  learns the policy's value, not the search's. That is also why its val MSE against the raw
  (search-play) outcome got worse (+0.014): the target moved. The scatter drop is the real noise
  reduction, but only play can say whether it makes the search stronger. That is exp068: net check
  F, then drives F's search vs A's search (same policy, different value head), then F's search vs
  F's policy.

**Net checks, A vs D** (gen06, seed 55100, the same decision sample, 250 overrides × 32 playouts
per rung):

| descents | gain/decision A | gain/decision D | realised/predicted slope A | slope D | corr A | corr D |
|---|---|---|---|---|---|---|
| 64 | +0.001 ± 0.001 | +0.002 ± 0.001 | 0.32 | 0.47 | 0.17 | 0.25 |
| 250 | +0.002 ± 0.001 | +0.002 ± 0.001 | 0.34 | 0.47 | 0.19 | 0.27 |
| 1000 | +0.005 ± 0.002 | +0.004 ± 0.002 | 0.28 | 0.49 | 0.16 | 0.27 |
| 4000 | +0.006 ± 0.003 | +0.008 ± 0.003 | 0.29 | 0.51 | 0.12 | 0.19 |

- **Both curves are MONOTONE.**
- **D's predictions come true more often:** the slope is 0.47-0.51 at every rung against A's
  0.28-0.34, and the correlation is higher too. That is the effect §4 asked for.
- **The gain per decision has not risen yet** (within 1 SE everywhere). Overrides are rarer than
  the slope change, and 250 overrides per rung resolve ±0.001-0.003.

**Net check F** (exp068, same sample): gain per decision **+0.002 / +0.002 / +0.007 / +0.008** at
64 / 250 / 1000 / 4000, MONOTONE; value RMS 0.219 and bias −0.008 on this sample; slope 0.60 /
0.50 / 0.36 / 0.70 (noisy, but at or above D's at three rungs out of four).

| at 1000 descents | A | D | F |
|---|---|---|---|
| gain per decision | +0.005 ± 0.002 | +0.004 ± 0.002 | **+0.007 ± 0.002** |
| realised gain of overrides on Q gaps 0.03-0.1 | +0.039 ± 0.019 | −0.010 ± 0.015 | +0.048 ± 0.017 |
| on 0.1-0.3 | +0.049 ± 0.025 | +0.114 ± 0.028 | +0.084 ± 0.029 |

F has the highest gain per decision at 1000 descents, though only ~1 SE above A. Like D, its
predictions come true more often than A's, and its overrides pay from gaps of 0.03 TD. The
cheap check can't separate the three at 250 overrides per rung; the F-vs-A drive match decides
it.
- Running: `net_check.sh` on A and then D (gen06, the same seed, so the same decision sample).
  Results go in `runs/exp067/status.md`.

**Play (exp068).** F's search against F's own bare policy, Gumbel f1000 at 1000 descents, 300
pairs per board, seed 68000:

| | 14x7 | 16x9 | mean |
|---|---|---|---|
| F search@1000 vs F policy | 0.578 ± 0.014 | 0.586 ± 0.016 | **0.582 ± 0.011** |
| g_gen05 search@1000 vs its policy (exp066) | 0.522 ± 0.016 | 0.563 ± 0.016 | 0.542 ± 0.012 |
| g_gen05 search@4000 vs its policy (exp066) | 0.577 ± 0.017 | 0.603 ± 0.016 | 0.590 ± 0.012 |

**With F's value head, 1000 descents get about what g_gen05 needed 4000 for.** The search's edge
over its policy rises from 0.542 to 0.582 (+0.040, ~2.4 SE). The comparison is unpaired: other
dice, and the policies are near-identical rather than identical. The rise shows on both boards,
most on 14x7, where g_gen05's search at 1000 had no edge left at all.

**Head to head, F's search vs A's search** (both Gumbel f1000 at 1000 descents; the nets share
g_gen05's policy, so only the value head differs; paired, 300 pairs per board):
**14x7 0.538 ± 0.013, 16x9 0.528 ± 0.013, mean 0.533 ± 0.009 (~3.7 SE).**

**Verdict: MC-averaged labels (arm F) win.** They pass every check plan 056 set:
- benchmark RMS −15.5% against the control;
- policy unchanged;
- the net check stays monotone, with the highest gain per decision;
- head to head, F's search beats the control's search.

The benchmark's caveat (F's labels share its definition) is answered by play.

**Proposed next (for the user):**
- **Make MC-averaged labels the loop's value label.** Each generation gets an `mc-label` pass
  over its train shards under the generator's net: ~7 min per shard on a free GPU, so ~45 min
  per generation for 6 shards at K = 8.
- **Settings to test cheaply on the benchmark first:**
  - K (4 vs 8 vs 16);
  - mixing in the real outcome as one more sample;
  - labelling only every second sample of a drive, to halve the cost.
- **Plan 055's gate is met under the mean backup:** the search beats its policy at every budget,
  and the curve is monotone up to 4000. With F's head, 1000 descents beat the policy by 0.58.
  The user, 2026-10-06: restart training. See §7.

## 4. Metrics and decision

Per arm, in this order:

1. **Benchmark** (§2): RMS and bias on g05's states, overall and by phase.
2. **Calibration:** val MSE against the raw outcome on gen07's held-out shards, `prepare
   --value-blend 1.0` + `scripts/wdl_summary.py` (plan 050's metric), for continuity with earlier
   work.
3. **The policy did not move:** `scripts/absorb_probe.py` KL and P(played) against g_gen05 on gen07
   held-out (the arms share the policy target, so a value-label change must not cost the policy).

**Decide.** The arm with the lowest benchmark RMS wins if:
- it beats the control by ≥ 5% RMS, or cuts the bias above +0.05 by at least half;
- neither the other metric nor the policy check gets worse.

Ties go to the simpler label (B over C/D).

**Then the search check for the winner:** `scripts/net_check.sh` on it against the control net,
on a corpus generated by g_gen05 (both are its fine-tunes, so the generator's data is fair to
both). The value fix is real if:
- the search gain per decision at 1000 descents rises;
- the realised-on-predicted slope rises toward 1, because a less noisy value lets more of what
  the search predicts come true;
- the budget curve stays monotone.

A drive match of the winner's search against its own policy confirms it once.

## 5. Order and cost

| step | cost | code |
|---|---|---|
| benchmark tool + baseline (g05, d1k, control A) | ~1 h dev, minutes per net | `override-audit --value-only` |
| arms A, B (existing flags) | ~10 min GPU each | — |
| TD(λ) in `prepare` + arms C, D | ~2 h dev (with tests), ~10 min each | `botbowl-nn/src/bin/prepare.rs`, `targets.rs` |
| arm E | ~10 min | — |
| net_check on the winner vs control | < 1 h each | — |
| arm F, only if B-E leave RMS ≥ 0.25 | GPU-hours | labelling pass |

Everything up to the net_check fits in a day alongside plan 055's exp066. The winning label goes
into the loop when it restarts (plan 055's budget gate, plan 054's train step).

## 6. Open questions

- **Value-head capacity** (bigger head, deeper trunk features for the value) is a plan of its own,
  deferred (the user, 2026-10-05: bigger network changes can wait). The case for it: every net in
  both lineages has the same 0.26-0.27 scatter, and B-E did not move it.

- **Is the error mostly at mid-activation states?** g05's bias is +0.12 there and +0.05 at turn
  starts. If the benchmark confirms the split, consider a sample weighting or more mid-activation
  states per drive.
- **The continuation mismatch** (§1 caveat) could be measured directly by redoing a few hundred
  benchmark states' MC with search play instead of policy play. Expensive; only if the bias
  question stays open after the arms.

## 7. The loop restart (2026-10-06, the user's go)

`scripts/launch_plan056.sh` → `scripts/train_loop.sh` into **`runs/loopmix16x9g056`** (tier
`mix16x9g056`, nets `models/az_v7/bbnet_mix16x9g056_genNN`). Everything plans 054-056 found, in
one recipe:

| piece | setting | from |
|---|---|---|
| backup at player nodes | visit-weighted mean, proven-win short-circuit (hardcoded) | plan 055, ce4eda1 |
| generation | Gumbel m=16 at 1000 descents (`cfgs/gumbel16_f1000_gen.toml`), 8 × 300 drives, sizes centred on 16x9 | plan 053 |
| value label | **MC-averaged**: `MC_LABEL_PLAYOUTS=8` policy-only playouts per train and val sample under the generator, `VALUE_BLEND=1.0` (val labelled too from gen03) | §3 arm F |
| policy label | cq tau 100 | plan 049 |
| train step | warm from the latest net, lr 5e-5, `--freeze-bn --init-candidate --eval-at 250,500`, value weight 0.25, per-drive value weight, 3 epochs, window 3 gens; restore on **val_policy + val_value against MC-labelled val shards** (`SELECT_ON=combined`, from gen03; gen01-02 used val_policy alone, see the results) | plan 054, amended |
| init | **arm F** (`models/az_v7/plan056_armF.{onnx,pt}` = `runs/exp067/arms/F`) | §3 |
| benchmark | gateless; drives vs the fixed d1k gen04 anchor, gen04g contested sets, SPRT 0.5:0.55, cap 800 | plan 051/054 |
| per-generation diagnostics on status.md | absorption probe (plan 054 E1); `value bench` (this plan's §2, the new net paired with its generator, seconds); `net check` (plan 055 §6, the generator on its own fresh corpus, in the background) | |

**New loop knobs** (`train_loop.sh`, all off by default):
- `MC_LABEL_PLAYOUTS` / `MC_LABEL_PARALLEL`: the `mc-label` pass after generation. It writes
  `genNN/mc/shard*.jsonl` and the `.mc_labelled` marker; `window_shards` then trains on the
  labelled copies.
- `VALUE_BENCH`: the benchmark path.
- `NET_CHECK` / `NET_CHECK_CONFIG`: writes `genNN/net_check/` and a status line when done.

**Costs per generation:** generation ~2 h; MC labelling ~45-60 min (alongside the background net
check); training ~10 min; the benchmark overlaps the next generation.

**What to watch:**
- the drive score vs the d1k gen04 anchor, generation by generation (the loop's curve);
- the net check staying MONOTONE, with gain per decision at 1000 not falling;
- the value bench's RMS on the g05 benchmark. Its MC truth is g_gen05's policy, so later nets'
  figures drift as their policies move. Read it as a trend, and re-freeze a benchmark from a new
  net's audit when that drift matters;
- absorption: dlogP(played) > 0, dtop1 ≥ 0.

**Laptop:** it joins the loop's hub on :13337. The allowlist admits exp066's commits (8cc6ce7 and
earlier), since nothing a worker runs has changed since.

### Loop results (`runs/loopmix16x9g056`)

Drives vs the d1k gen04 anchor (SPRT 0.5:0.55, paired). For reference, the old Gumbel loop
(`runs/loopmix16x9g`, blend-0.5 labels, minimax, lr 2e-4) never got above 0.506 on either board
over gen01-06 (14x7 0.42-0.50, 16x9 0.42-0.51).

| gen | generated by | gen time | MC label | absorption (dP played, dtop1, dvalMSE) | value bench RMS / bias (generator → new) | net check of the generator (gain/decision @64/250/1000/4000; slope) | 14x7 vs anchor | 16x9 vs anchor |
|---|---|---|---|---|---|---|---|---|
| 01 | arm F | 72 min (laptop on) | 45 min, 0 unlabelled | +0.003, +0.002, −0.007 | 0.236 / −0.00 → 0.242 / +0.05 | +0.002/+0.002/+0.005/+0.005 MONOTONE; 0.50 | 0.506 ± 0.018 (H0, 200 pairs) | **0.583 ± 0.029 (H1, 68 pairs)** |
| 02 | gen01 | 110 min | 47 min | 0, 0, 0 (**restored its init**) | unchanged (= gen01) | +0.001/+0.002/+0.005/+0.009 MONOTONE; 0.47 | 0.508 ± 0.016 (H0, 262 pairs) | **0.551 ± 0.018 (H1, 179 pairs)** |
| 03 | gen02 (= gen01) | 233 min (laptop gone) | 59 min, train+val, 0 unlabelled; restored step 9000 of 15621 (combined) | +0.002, +0.0014, +0.003 (vs raw outcome; the head now tracks MC values) | 0.242 / +0.046 → **0.236 / +0.035 (paired −2.4%, ~9 SE)** | (running) | | |

gen01 is the first net in any Gumbel loop to beat the anchor on 16x9. The value bench's drift
(+0.05 bias) is expected (its truth is g_gen05's policy, see above); watch the trend.

**gen02 restored its init, so gen02 = gen01.**
- **What happened:** with `SELECT_ON=policy` the restore looks at val_policy alone. That stayed
  flat within ±0.002 for all 10k steps (0.5535 at step 0, 0.5541-0.5563 after). So the init "won"
  by 0.0006, which is noise. val_value fell 0.3762 → 0.3742, but that progress was discarded with
  the rest. A fine-tune whose improvement is in the value head, which is what MC labels are for,
  can't be seen by a policy-only restore.
- **Fix (`e091107`):**
  - MC-label the val shards too (`train_loop.sh` labels `$TRAIN_SHARDS $VAL_SHARDS`, and
    `window_shards` uses the labelled val copies), so val_value measures the label being trained
    instead of the single raw outcome;
  - restore on `combined` (val_policy + val_value). The absorption probe keeps the raw val
    shards, so its dvalMSE is still against the raw outcome.
- **Applied:**
  - gen01's and gen02's val shards were MC-labelled by hand (seeds 56001/56002, their generators);
  - the loop was stopped (STOP) before gen03's mc-label and relaunched on the fixed script
    (11:54, `fbedceb`).
  - gen03 is the first generation trained under the fix.
- gen02's drives re-measure gen01's net (same weights) and agree: H1 on 16x9 again (0.551), even
  on 14x7.

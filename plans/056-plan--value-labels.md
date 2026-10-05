# Plan 056 — value labels: train the value head on something closer to the truth

**Status:** Written 2026-10-05 from plan 055's override audit. In progress: the benchmark is built (§2 results); exp067 runs arms A and B, and TD(λ) is in `prepare` for C and D. Drives only (plan 051);
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
- Running: `net_check.sh` on A and then D (gen06, the same seed, so the same decision sample).
  Results go in `runs/exp067/status.md`.

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

- **Is the error mostly at mid-activation states?** g05's bias is +0.12 there and +0.05 at turn
  starts. If the benchmark confirms the split, consider a sample weighting or more mid-activation
  states per drive.
- **The continuation mismatch** (§1 caveat) could be measured directly by redoing a few hundred
  benchmark states' MC with search play instead of policy play. Expensive; only if the bias
  question stays open after the arms.

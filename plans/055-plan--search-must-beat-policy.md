# Plan 055 — the search must beat the policy before the loop can learn anything

**Status:** Written 2026-10-04 night. Phase 1 done 2026-10-05 05:13 (results below §3 phase 1):
the search loses to its policy unless it is made to override less (q floor 4000) or to back up
the mean; PUCT loses by 0.13. Phase 1 was launched as
`scripts/exp064_search_vs_policy.sh` (out `runs/exp064/`), taking over from exp063 at its gate.
**The user's decision (2026-10-04): no training loop until the search beats the policy.**
`runs/loopmix16x9g054/HOLD` blocks plan 054's automatic relaunch; delete it to allow one. Drives
only (plan 051). The laptop stays on 7a86e4d, so tonight's work changes no game code
(`botbowl-engine`, `-mcts`, `-nn`, `-play`, `-worker`, `-hub-proto`, `recon_mcts`).

## 1. The finding

plan 054's E5 (exp063) played one net, g_gen05, two ways against itself on the loop's
contested drives (paired, the gen04g sets):

- **policy-only:** `cfgs/policy_only.toml` (`gumbel_m = 1`, no noise): the prior's argmax over the
  legal moves, no lookahead.
- **search:** `gumbel16_f1000` at 1000 descents, what the loop generates and benchmarks with.

| | 14x7 | 16x9 |
|---|---|---|
| policy-only's points vs the search | 0.512 ± 0.015 (300 pairs) | 0.522 ± 0.016 (300 pairs) |

The search does not beat its own policy, on either board. On 14x7 it adds at most about +0.02.

**It is not that the search plays the policy's moves.** On the Gumbel corpus the deterministic
Gumbel pick (the gumbel σ floor-1000 rule) leaves the prior's argmax in about 22% of multi-move
decisions, and the best-Q child leaves it in 48% (plan 054 §2). The search overrules the policy
often, and **the overrides are net zero.**

**Why it decides everything.** Expert iteration trains the policy toward the search. If the
search is no stronger than the policy, the target has nothing in it, and each generation re-learns
itself. That is consistent with all four plateaus (mix16x9, vl0, d1k, Gumbel) and with plan
054's absorption probe (the fine-tune absorbs nothing). It also re-reads the earlier search
results: Gumbel beat PUCT, and more descents beat fewer, but every one of those compared search
with search. **Search vs no search was never measured.** PUCT at 1000 may have been worse than the
bare policy all along, with Gumbel only climbing back to its level.

Caveats: one net, contested drives only; "level" bounds the search's edge at about +0.02 per
board (it is, if anything, slightly negative), not exactly zero.

## 2. Hypotheses (most supported first)

**H1. The value head is too noisy to rank what the search compares.** The search acts on tiny
gaps: the top-2 Q gap is 22 points at the median (2% of a TD). The value head's per-position
error is unknown in the units that matter: val MSE (~0.09) is against one dice-driven outcome
per position, so it mixes net error with irreducible dice variance, and nobody has split the two.
If the net's error is 0.1-0.2 TD per leaf, most overrides are coin flips decided by noise.

**H2. The search exploits the value head's errors (winner's curse).** Taking the max over noisy
estimates selects the moves the net overrates. Measured: the search's root value is +0.10 more
optimistic than the bare value head (plan 031 D1, z = 52), and +0.04 to +0.09 against outcomes on
the Gumbel shards (plan 054 §7.6). A Blood Bowl turn is a long chain of decisions by the *same*
side, so the tree is mostly max nodes stacked on max nodes with no opponent reply to cancel the
optimism. Mean backup lost once (0.454, plan 032 #2), under 2026-09's leaky PUCT, not under
Gumbel.

**H3. The search model diverges from the game.** Collapsed chance outcomes (scripted bounces go
`Direction::up`), a one-turn horizon with leaves mid-turn or mid-activation, which the value head
may judge worse than the turn-start states it mostly sees.

**H4. 1000 descents is too shallow.** exp057: 16x9 roots converge only past about 4000. exp058:
4000 beats 500 visits. More budget may beat the policy where 1000 cannot.

**H5. The overridden decisions are near-ties.** If most overrides swap moves of equal real value,
no search can gain on them. It cannot be the whole story: plan 054's D2 fine-tune gained about
+0.04 on drives, so decision quality does move outcomes.

## 3. Investigation

### Phase 1 — config-only matches (exp064, tonight; no game code)

Every match is a search configuration against policy-only **on the same net**, paired drives,
the gen04g contested sets, fixed 300 pairs per board, `--seed 64000`. Points are the search's.
The policy-only side costs almost nothing, so a match costs about half a normal one.

| match | configuration | tests |
|---|---|---|
| N0 | policy-only vs policy-only | the null: must read 0.500 |
| G250 / G1000 / **G4000** | `gumbel16_f1000` at 250 / 1000 / 4000 | H4: the budget curve against the policy |
| P1000 | PUCT (`exact_iters`) at 1000 | was the old loops' search worse than no search? |
| MEAN | `gumbel16_f1000_mean` (mean backup) | H2 |
| F4000 / F300 | q floor 4000 / 300 | defer to the prior more / less: does overriding less help? |
| H2 | `gumbel16_f1000_h2` (two-turn horizon) | H3/H4: leaves past the opponent's reply |
| G1000_d1k | `gumbel16_f1000` at 1000 on d1k gen04 | a second net: is this one net's quirk? |

**Readings.**
- G4000 clearly above 0.5, with G250 < G1000 < G4000: H4. The search works but needs depth;
  the fix is budget allocation (§4).
- Flat in budget, and MEAN or F4000 above G1000: H2 (or H1). The search's errors, not its
  depth, are what cancels its gains.
- P1000 below 0.5: the PUCT loops trained on targets worse than their own policy.
- Everything about 0.5: the overrides are noise or near-ties; phase 2 decides which.

### Phase 1 results (exp064, 2026-10-05 05:13; 300 pairs per board, points are the search's)

| configuration (g_gen05 unless noted) | 14x7 | 16x9 | two-board mean |
|---|---|---|---|
| N0 policy-only vs policy-only | 0.500 | 0.500 | 0.500 (null holds) |
| **P1000** PUCT @1000 | 0.390 ± 0.016 | 0.347 ± 0.017 | **0.368 ± 0.011** |
| F300 Gumbel, q floor 300 | 0.456 ± 0.016 | 0.447 ± 0.017 | 0.451 ± 0.012 |
| G1000 Gumbel @1000 (the loop's) | 0.463 ± 0.016 | 0.500 ± 0.016 | 0.482 ± 0.012 |
| H2 two-turn horizon | 0.477 ± 0.015 | 0.497 ± 0.017 | 0.487 ± 0.011 |
| G250 Gumbel @250 | 0.500 ± 0.014 | 0.524 ± 0.016 | 0.512 ± 0.011 |
| MEAN Gumbel @1000, mean backup | 0.527 ± 0.015 | 0.525 ± 0.016 | **0.526 ± 0.011** |
| F4000 Gumbel @1000, q floor 4000 | 0.523 ± 0.014 | 0.538 ± 0.015 | **0.531 ± 0.011** |
| **G1000_d1k** Gumbel @1000 on d1k gen04 | 0.580 ± 0.015 | 0.550 ± 0.016 | **0.565 ± 0.011** |

**Readings against §3:**
- **PUCT at 1000 is far worse than no search** (−0.13). Every loop before the Gumbel run trained
  its policy toward targets from a search that loses to that policy.
- **On g_gen05, the more the search overrules the prior, the worse it plays:** q floor 300 <
  1000 < 4000 (0.451 < 0.482 < 0.531), and 250 descents beats 1000 (0.512 vs 0.482). More search
  is more opportunity to act on value error. The two-turn horizon does not help (H3/H4 as depth:
  rejected).
- **Mean backup turns a loss into a win** (0.482 → 0.526). Together with the q-floor trend this is
  H2: the max over noisy children picks the moves the value head overrates. H1 (value noise) is
  its precondition; phase 2 measures it.
- **On d1k gen04 the same search beats its policy by +0.065.** d1k gen04's policy came from PUCT
  data; g_gen05's from five Gumbel generations. Either those generations distilled the search into
  the policy (so a better policy leaves the noisy search less room), or d1k gen04's policy is
  simply weaker. The override audit on both nets separates the two.
- **Gate (§4):** F4000 and MEAN pass the "mean ≥ 0.5 + 2 SE" half but not "+0.03 on both
  boards". Not met yet. exp064b tests the two together (`gumbel16_f4000_mean`) on both nets;
  G4000 is the budget arm.

### exp064 phase 2 and exp064b (stopped 2026-10-05 09:21 at the user's call; 14x7 final, 16x9 partial)

| configuration | net | 14x7 (300 pairs, final) | 16x9 (stopped at) | two-board mean |
|---|---|---|---|---|
| G4000 Gumbel @4000 (minimax) | g_gen05 | 0.531 ± 0.015 | 0.469 ± 0.027 (119 pairs) | ~0.500 |
| MEANF4000 mean backup + q floor 4000 @1000 | g_gen05 | 0.535 ± 0.014 | **0.568 ± 0.020** (204 pairs) | **~0.552 ± 0.012** |
| MEANF4000_d1k mean backup + q floor 4000 @1000 | d1k gen04 | 0.568 ± 0.014 | 0.572 ± 0.020 (203 pairs) | ~0.570 ± 0.012 |

- **Mean backup + q floor 4000 is the best search on g_gen05**, and the two fixes stack on 16x9
  (0.568 against 0.525 and 0.538 alone) though not on 14x7 (0.535 against 0.527 / 0.523). Wide
  fans are where max-over-noise has most to exploit. On the partial 16x9 it meets §4's gate
  (+0.035 and +0.068, mean ≥ 0.5 + 2 SE); 14x7's margin is thin, and it is the best of about ten
  configurations tried on this net, so a fresh-seed confirmation is due before a loop runs on it.
- **On d1k gen04** the combination adds a little on 16x9 (0.572 vs 0.550) and nothing on 14x7.
- **G4000 (minimax)**: no better than the policy on 16x9 (0.469 at 119 pairs) and slightly better
  on 14x7. Stopped early because minimax is being retired; the budget curve is to be re-measured
  under the mean backup (the user's hypothesis: a bump down then a recovery, later and wider on
  16x9's larger fans).
- **Decision (the user, 2026-10-05): the mean backup becomes the only backup**, hardcoded (no
  knob). Next: the phase 2 override audit.

### Phase 2 — the override audit (needs a tool; tomorrow)

The direct measurement. From the Gumbel corpus take about 1000 decisions where the eval-mode
search and the policy disagree. From each, play out the drive many times (say 64, paired dice)
after the policy's move and after the search's move, with the cheap policy-only bot on both
sides. That gives a Monte Carlo value for both moves, and from it:

- **The override ledger:** the share of overrides that help, hurt or tie, and by how much; split
  by decision kind (move, block, blitz, pass, foul, end turn, positional vs simple), fan width,
  turn start vs mid-turn vs mid-activation, and the Q gap the search acted on.
- **H2 directly:** whether `V(search move) − MC(search move)` exceeds
  `V(policy move) − MC(policy move)`. The search exploiting the net shows as overrides that the
  net rates high and Monte Carlo rates low.
- **H1 directly (value calibration against Monte Carlo truth):** per-state net error, separated
  from dice variance, compared with the Q gaps the search acts on.
- **H5 directly:** the distribution of |MC(search move) − MC(policy move)|. Mostly near zero means
  near-ties.

**Engineering.** The drive machinery plays from position files; it has to start from an
arbitrary mid-turn corpus state, and the audit needs "apply move, then play out N times". It runs
locally through `botbowl-ui` (not on the hub), so it needs no worker change; if it needs
`botbowl-play`, it waits for a commit the laptop can follow, or runs on this box alone.

**Built (2026-10-05): `botbowl-ui override-audit` + `scripts/override_audit_summary.py`.** Local
only, on this box. `botbowl-play` gained only `drives::DriveStart` (the drive-end rule, now shared
by the corpus generator, the drive benchmark and the audit; behaviour-neutral), so no worker change.
Corpus states are rebuilt by replaying each trajectory from its seed (a deserialised state has no
path offerings), checked state by state. Policy move = the `policy_only` preset's (pinned by a
test); search move = a fresh search of the state under the named preset; overrides plus a 20%
control (`keep_prob` per row); 64 paired playouts per move, policy-only on both sides. Rows hold
MC mean/SE per move, the paired difference, Q/visits/priors, V after each move, V(s) (against
`policy.mc_mean` = MC(s) for H1), the root value, decision kind, fan, phase and board; see
`botbowl-ui/CLAUDE.md`. Smoke cost at 64 playouts, mixed boards: about 3.5 s of thread time per
kept row (1 s of it the search), and one override per ~5-6 searched decisions on g_gen05.

```sh
BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
cargo run --release -p botbowl-ui -- override-audit \
    --corpus runs/loopmix16x9g/gen06/shard4.jsonl runs/loopmix16x9g/gen06/shard7.jsonl \
    --model models/az_v7/bbnet_mix16x9g_gen05.onnx --nn-server /tmp/bbnn.sock \
    --search-config cfgs/gumbel16_f1000.toml --search-iters 1000 \
    --decisions 1000 --playouts 64 --parallel 8 --out runs/exp065/audit_g_gen05.jsonl
# second net: --corpus runs/loopmix16x9g/gen01/shard{4,7}.jsonl --model models/az_v7/bbnet_mix16x9d1k_gen04.onnx
scripts/override_audit_summary.py runs/exp065/audit_*.jsonl
```

### Phase 2 results (exp065, 2026-10-05 14:10; merged code, mean backup; 1000 overrides × 64 paired policy-only playouts per run; eval boards 14x7 + 16x9; each net on a corpus it generated and was not trained on)

| | g05 f1000 | g05 f4000 | d1k f1000 | d1k f4000 |
|---|---|---|---|---|
| override rate | 0.157 | 0.070 | 0.153 | 0.067 |
| realised gain per override | +0.026 ± 0.006 | +0.047 ± 0.006 | +0.035 ± 0.006 | +0.052 ± 0.006 |
| predicted gain (Q gap) | +0.082 | +0.115 | +0.094 | +0.117 |
| realised-on-predicted slope | 0.34 | 0.48 | 0.41 | 0.38 |
| **search gain per decision** | **+0.004 ± 0.001** | **+0.003 ± 0.000** | **+0.005 ± 0.001** | **+0.003 ± 0.000** |
| real effect sd of an override | 0.174 | 0.172 | 0.160 | 0.159 |
| value RMS error vs MC | 0.281 | 0.277 | 0.263 | 0.256 |
| value bias V(s) − MC(s) | +0.110 | +0.106 | +0.052 | +0.055 |

Realised gain by the Q gap the search acted on (g05 f1000; the other runs have the same shape):
< 0.01: −0.017 ± 0.016 · 0.01-0.03: −0.000 ± 0.009 · 0.03-0.1: +0.009 ± 0.010 ·
**0.1-0.3: +0.065 ± 0.014 · ≥ 0.3: +0.140 ± 0.044**.

**Readings.**
- **With the mean backup the search's overrides help on every run**, but by a third to a half of
  what the search predicts. All MC is under the policy's own continuation.
- **H1, value noise, is the dominant problem.** The net's per-state error is ~0.26-0.28 TD.
  Overrides on Q gaps under ~0.03-0.1 TD realise nothing, and those are two-thirds of them.
  Overrides on gaps of 0.1 TD and up realise +0.06 to +0.2.
- **H5, near-ties, is rejected.** An override's real effect has sd ~0.16-0.17 TD.
- **H2 cannot be separated from H1 with this design.** The search's Q overrates its pick by
  +0.06-0.07 beyond MC, the size of winner's curse this noise predicts. The net's own V shows no
  such excess.
- **g_gen05's value head is twice as optimistic as d1k gen04's** (+0.11 vs +0.05), with the same
  RMS. The suspected source is five generations of value labels half made of minimax root Q
  (plan 031 D1: +0.10 optimism).
- **q floor 4000 vs 1000.** Fewer, better overrides: 0.07 vs 0.15 of decisions, +0.05 vs +0.03
  each, and a better slope on g05. Per decision the audit puts 1000 marginally ahead (+0.004 vs
  +0.003, ~1 SE). The drive matches (exp064, partial) put 4000 ahead (≈0.552 vs 0.526 against the
  policy, ~1.6 SE). This is the disagreement exp066 exists to check: the audit's policy
  continuation may miss deeper value, or the difference is noise.
- **Next (proposed):** the value labels, now their own plan (`plans/056-plan--value-labels.md`,
  judged on a frozen MC benchmark built from these rows), and an override margin of ~0.1 TD (play
  the prior unless Q(a_s) − Q(a_p) clears it). Both are judged by the budget criterion below.

### Phase 3 — instrumentation

Log every decision where the eval bot overrules its policy (prior rank of the pick, Q gap,
visits, decision kind) into the per-game output, so the override rate and its shape come from
real play rather than corpus approximations. Report "search vs policy" next to the anchor in
every loop generation once the loop runs again: it becomes the loop's health metric.

## 4. What to do once we know

| finding | remedy |
|---|---|
| **H1** value noise | better value labels: Monte Carlo-averaged labels (several cheap policy-only playouts per stored position), TD(λ) (plan 054 E6), more value capacity; meanwhile a confidence-aware override rule: overrule the prior only when the Q gap clears the measured noise (Q minus its standard error from visits, or a floor set from the calibration) |
| **H2** exploitation | mean or soft backup at player nodes, lower-confidence-bound selection at the root, debiasing visit-inflated Q in the targets |
| **H3** model mismatch | fix the specific chance-model or horizon defects the ledger points at; leaves at turn starts only (extend each leaf to the end of the activation) |
| **H4** too shallow | spend the budget where it matters: no search on single-child roots (21% of searches), more on turn starts and high-stakes kinds; Gumbel at 4000 on 16x9 at equal cost |
| **H5** near-ties | per-decision search is not where the strength is: macro actions (a whole activation as one move), and treat the loop as policy distillation with plan 054's train-step fixes |

**Gate for the loop (revised 2026-10-05, the user): more search must never hurt.** A fixed
target such as "+0.03 over policy" says nothing about whether the system is sound. The property
self-play rests on is that **the search's improvement over the bare policy rises with budget**: if
a bigger budget plays worse, targets get worse the more is spent and the loop cannot climb. The
criterion, per net:

- gain(b), the search's per-decision improvement over the bare policy at budget b, is ≥ 0 at
  every rung of a ladder (64 → 250 → 1000 → 4000 descents);
- gain(b) is non-decreasing up the ladder within its SE; diminishing returns and a plateau at the
  top are expected and fine, a drop is not;
- when a new net breaks it, retune the search (first the override margin, §4 H1) before training
  further. Search configs are expected to need retuning as nets improve.

exp064 under minimax broke it (250 descents beat 1000 against the policy). Under the mean backup
the curve is unmeasured; §6 measures it.

## 6. The standing net check (< 1 h per net)

One script, run on every new net, one line per generation in the run's `status.md`:

1. **Search-improvement curve.** `override-audit` on a fixed decision sample (~1500 decisions,
   the net's own fresh corpus, eval boards) at each ladder budget; `override_audit_summary.py`'s
   "search gain per decision" = override rate × mean realised paired gain (non-overrides count 0),
   with its SE. ~200-250 overrides per rung, ~10-15 min per rung on the sidecar.
2. **Override calibration.** Per rung, realised-on-predicted slope and realised gain by Q-gap
   bucket: where the realised gain turns positive is the override margin to set.
3. **Value quality.** V(s) vs MC(s) (RMS net error after removing MC sampling noise, and bias) on
   ~500 fixed states, with MC replayed under the new net's own policy (~15 min).
4. **Policy absorption.** plan 054's `absorb_probe.py` against the generator (minutes).

**Validation before it is trusted.** The audit's MC is the value under the policy's own
continuation, so gain(b) can miss value that only shows deeper in a drive. Once, under the mean
backup: drive matches of g_gen05's search at 250 / 1000 / 4000 descents against policy-only
(fixed 300 pairs per board), next to the cheap curve on the same net. If they rank the budgets
alike, the cheap curve is the standing check.

**First readings (exp065, interim, ~5400-6700 searched decisions each):** search gain per
decision at 1000 descents is +0.004 ± 0.001 (g_gen05, q floor 1000), +0.003 ± 0.001 (g_gen05,
q floor 4000), +0.005 ± 0.001 (d1k gen04, q floor 1000), +0.004 ± 0.001 (d1k gen04, q floor 4000).
Resolution ±0.001 per point is enough to see a ladder's shape.

**exp066 net check, g_gen05, mean backup, q floor 1000 (2026-10-05 14:44, 33 min, gen06, eval
boards, 250 overrides × 32 playouts per rung):**

| descents | override rate | gain per decision | realised / predicted slope |
|---|---|---|---|
| 64 | 0.05 | +0.001 ± 0.001 | 0.42 |
| 250 | 0.08 | +0.001 ± 0.001 | 0.30 |
| 1000 | 0.16 | +0.004 ± 0.002 | 0.24 |
| 4000 | 0.29 | +0.006 ± 0.003 | 0.16 |

- **MONOTONE:** the gain never drops and is never below the policy. The curve is flat to 250 and
  rises from 1000; the 4000 point is within 1 SE of 1000, so it could already be flattening.
- **Every extra override is worth less.** The override rate doubles per rung while the slope
  halves. The deeper search moves off the prior more often on Q gaps the value head cannot
  resolve, which fits the value-noise reading (H1, plan 056).
- **Value head:** RMS 0.27, bias +0.11 at every rung, as exp065 found.
- **Override margin: not resolved at 250 overrides.** The only bucket clear of 2 SE is 0.1-0.3 TD
  at 1000 descents (+0.06 ± 0.03). The script's first reading ("from <0.01") used 1 SE and picked
  noise; it now needs 2 SE.
- **Drive matches** (g_gen05's Gumbel search with the mean backup and q floor 1000 against its own
  policy-only, gen04g contested sets, 300 pairs per board, seed 66000; points are the search's):

  | descents | 14x7 | 16x9 | mean |
  |---|---|---|---|
  | 250 | 0.527 ± 0.015 | 0.523 ± 0.015 | 0.525 ± 0.011 |
  | 1000 | 0.522 ± 0.016 | **0.563 ± 0.016** | **0.542 ± 0.012** |
  | 4000 | **0.577 ± 0.017** | **0.603 ± 0.016** | **0.590 ± 0.012** |

  **The criterion holds: more search never hurts, and it keeps paying up to 4000 descents.** Under
  the mean backup the drive curve is 0.525 → 0.542 → 0.590 against the bare policy. Each step is
  non-negative, and 4000 is +0.048 over 1000 (~2.8 SE). On 16x9 the curve rises at every step
  (0.523 → 0.563 → 0.603). On 14x7 it is flat to 1000 and rises at 4000 (0.527 / 0.522 / 0.577),
  close to the dip-then-recover shape the user expected, though here the dip is within noise.
  Under minimax, exp064 had 250 beating 1000.

  **Cheap curve vs drives: they rank the budgets alike.** Net check +0.001 / +0.004 / +0.006 per
  decision; drives 0.525 / 0.542 / 0.590. Both are monotone, and both rank 4000 > 1000 > 250. The
  net check is adopted as the standing check, with a drive match only when it is ambiguous. The
  cheap curve compresses the top: per decision, 4000 is only +0.002 over 1000 while the drives
  gain +0.048. It measures the gain under the policy's continuation, which misses value that
  shows deeper in a drive. Read it for shape and sign, not size.

## 5. Order

1. exp064 phase 1 (tonight, ~5-6 h with the laptop).
2. Read it against §3's readings; pick the phase-2 strata accordingly.
3. Build and run the override audit (phase 2).
4. The remedy the audit points at, judged by the same search-vs-policy match.
5. The loop, under plan 054's train step, once the gate in §4 is met.

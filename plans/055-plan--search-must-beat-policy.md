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

### Running: exp064 phase 2 and exp064b (interim, 2026-10-05 06:06 — replaced as each finishes)

| configuration | net | 14x7 so far | 16x9 |
|---|---|---|---|
| G4000 Gumbel @4000 | g_gen05 | 0.525 ± 0.020 (166 pairs) | not started |
| MEANF4000 mean backup + q floor 4000 @1000 | g_gen05 | 0.529 ± 0.015 (254 pairs) | not started |
| MEANF4000_d1k mean backup + q floor 4000 @1000 | d1k gen04 | 0.571 ± 0.015 (255 pairs) | not started |

So far: the two fixes do not stack (0.529 vs 0.526 / 0.531 alone), and on d1k gen04 they add
nothing to plain Gumbel (0.571 vs 0.580). G4000 at 0.525 against G1000's 0.463 breaks the
"more search is worse" trend from 250 → 1000: the search may be noisy at moderate budgets and
recover with much more. Unresolved until 16x9 is in.

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

**Gate for the loop.** The loop restarts only when one search configuration beats policy-only on
the same net by at least +0.03 on both boards (fixed 300 pairs, two-board mean ≥ 0.5 + 2 SE).
That configuration generates. "Search vs policy" is then measured every generation.

## 5. Order

1. exp064 phase 1 (tonight, ~5-6 h with the laptop).
2. Read it against §3's readings; pick the phase-2 strata accordingly.
3. Build and run the override audit (phase 2).
4. The remedy the audit points at, judged by the same search-vs-policy match.
5. The loop, under plan 054's train step, once the gate in §4 is met.

# Plan 055 — the search must beat the policy before the loop can learn anything

**Status:** Written 2026-10-04 night. Phase 1 (config-only matches) launched as
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

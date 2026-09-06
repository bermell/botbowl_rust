# Should one big tree beat k small ones? (And why it currently doesn't)

**Status:** Stage 0 ran 2026-09-06 and **refutes the plan's own premise** — see
Results. The search converges; plan 025's non-convergence was the pre-`e107f06`
bug. Normalised Q is clearly *worse*. The live finding is different and better:
the budget curve has a **flat spot at 1000-2000, not a ceiling**.

## The hygiene test, and why it matters

One tree with `k·N` iterations has strictly more information available than
`k` independent trees with `N` each: it can recombine through the DAG and spend
its extra budget deepening the line that matters. If splitting the compute
*wins*, the searcher is failing to use information it already has.

Plan 025 already ran that test — **on labels** — and the split won:

| label recipe | cost | TV vs held-out | top-1 vs held-out |
|---|---|---|---|
| 1 x 500 | 500 | 0.2701 | 0.65 |
| **avg of 2 x 500** | **1000** | **0.2374** | **0.68** |
| 1 x 1000 | 1000 | 0.2818 | 0.67 |

~16% less label noise at identical compute, and the independence prediction
(sqrt(0.75) = 0.866, so 0.2339 expected against 0.2374 measured) lands within
1.5% — the two trees really are drawing independent noise.

**What plan 025 explicitly left untested is play strength.** That is the gap
this plan closes, and it is the same gap that made plan 025's budget conclusion
misleading until plan 027 re-measured it as strength (label convergence said
"1000 is already too much"; strength said 250 -> 1000 is worth 0.700 at
p~0.002).

Two things make it more urgent than when plan 025 was written:

1. **The policy label is finally consumed.** Until gen06 the loop played
   `--evaluator nn-value` with scripted priors, so a 16%-better policy target
   trained a head nothing used. It now plays `--evaluator nn`.
2. **Strength saturates at 500-1000** (plan 027: 250->1000 = 0.700, 1000->2000
   = 0.508). If that ceiling exists *because* extra iterations get absorbed
   amplifying an arbitrary early lead, then fixing the amplification should
   raise the ceiling — which is the prize here. Ensembling would be a way to
   buy strength with compute again; fixing the tree would be a better one.

## The mechanism, as far as it is understood

PUCT is self-reinforcing among near-tied children: whichever takes an early
lead attracts more visits and widens it. Plan 025 frames it as a Polya urn —
running one urn longer converges to a *random* limit, not to the mean — so
tie-break noise is amplified by depth rather than averaged away. Plan 020
measured **84.8% of decisions with all-tied children Q**, which is the
population this applies to.

Why the exploration term does not correct it: `puct_value` is textbook
AlphaZero PUCT, but `Q` is in raw `leaf_score` units (`score_delta*1000 +
ball_tier*10 + carrier_tier`) while `PUCT_C = 10` is the sort of constant that
assumes `Q ∈ [-1,1]`. Plan 026 measured the best-vs-second Q gap varying **85x
across states** (p10 6, median 77, p90 510). Where the gap is large the
exploration term is noise; the tree commits.

`botbowl-mcts/CLAUDE.md` states the coupling explicitly: "`PUCT_C = 10.0` is
tuned against the leaf-score magnitudes in `score.rs` and they are coupled —
changing one without the other silently degrades search."

## Candidate fixes, best-supported first

**F1 — `PuctMode::NormalisedQ`. Already implemented (plan 026), never shipped,
and already measured to do the right thing.** It maps sibling Q into `[0,1]`
against the node's min/max and priors onto the simplex, which makes `c` mean
something state-independently. Plan 026's sweep (52 states x 7 repeats):

| arm | agree | TV | peak share | H |
|---|---|---|---|---|
| raw c=10 (shipped) | 0.60 | 0.2212 | 0.515 | 0.617 |
| raw c=30 | 0.65 | 0.2017 | 0.495 | 0.638 |
| norm c=1 | 0.69 | 0.1682 | 0.444 | 0.710 |
| norm c=4 | **0.70** | 0.1722 | 0.442 | 0.717 |

Normalised family vs shipped: **+0.082, p=0.023**. Lower peak visit share is
precisely "stops over-committing to one arbitrary action". Plan 026 filed this
as "not the fix" because it was hunting the Home/Away side bias — a different
question, for which it genuinely was not the fix. Against *this* question it is
the leading candidate and it costs nothing to test: the code exists and
`--puct-mode`/`--puct-c` are already CLI flags on `eval`.

**F2 — raw `c=30`.** Plan 026's best raw arm, marginal but free.

**F3 — FPU.** Unexplored children currently get **the parent's Q** as their
first-play estimate (`botbowl-mcts/CLAUDE.md`). If one child banks a lucky
high leaf value, its siblings still sit at parent Q and may never be tried.
AlphaZero uses FPU *reduction* (parent Q minus a constant) precisely to stop
that. Not implemented; a real but contained change.

**F4 — deterministic tie-break.** Plan 023 found forcing "prefer lowest-x"
moved the mirror match 0.703 -> 0.503, but the *mirror-covariant* version left
it at 0.633, so it is a compensation and is marked **do not ship**. Worth
re-reading before touching ordering: the covariant form is the only admissible
one, and plan 023 already showed it does not do what the naive form appears to.

## Experiments

### Stage 0 — the convergence probe: does TV shrink with budget? (run this first)

Before any games: take states from the generator the corpus uses, run the
search repeatedly at increasing budgets, and measure the total-variation
distance between independent runs at each stage. If two runs of the same state
do not agree more as the budget grows, the search is not converging, and no
amount of tuning downstream of that will help.

`botbowl-ui convergence` is exactly this instrument and already exists: N
random-start states (same `random_start.rs` the corpus uses), R independent
repeats per (state, budget) giving the run-to-run floor, budgets
`100,200,500,1000,2000,4000,8000,16000`, seeds disjoint from corpus seeds, and
`--puct-mode`/`--puct-c` from plan 026. Analysis in
`scripts/convergence_summary.py`.

Plan 025's answer was that TV between runs **grows** monotonically, 0.193 at
100 iterations to 0.383 at 16000, with peak visit share rising 0.449 -> 0.742
and top-1 agreement peaking at ~500 then falling. At 16k only 21 of 52 states
had all three repeats agreeing on the top action. The heuristic control showed
the same shape, so it is the search, not the net.

**That result has never been confirmed on current code.** Plan 025's own status
header marks it provisional: it ran on the retired `gen01` champion under the
**pre-`e107f06` buggy search**, and the raw data was deleted 2026-09-01. Since
then the side-bias root cause was fixed (plan 023), the NN evaluator changed
(gen06 plays learned priors), and plan 027 showed strength *does* improve
250 -> 1000 where 025's label metrics said it should not. Re-checking is
overdue on its own merits.

Two arms, and they are cheap — plan 025 measured 48 min for the `nn-value` arm
and 7 min for the heuristic control at 4-way parallelism, against ~3h for a
single 120-game head-to-head:

| arm | config | asks |
|---|---|---|
| S0a | `raw c=10` (production), `nn` evaluator, gen03 | does the divergence reproduce post-fix? |
| S0b | `norm c=1`, same states and seeds | does normalised Q make TV *fall* with budget? |

**This is the right screen for F1.** Plan 026 measured normalised Q at a single
budget (1000) and found better agreement and lower TV; what nobody has looked
at is the *curve*. A fix that works should turn the TV line from rising to
falling. If S0b still rises, F1 is not the fix either and B1's three hours of
games are better spent elsewhere. If it falls, B1 becomes a confirmation rather
than a fishing trip.

Order matters here: Stage 0 costs about an hour and can redirect ~15 hours of
game arms, so it runs before anything in A or B.

Every arm is a head-to-head at equal **issued iterations**, same net both
sides, paired Home/Away, scored with `scripts/paired_summary.py` as well as the
pooled rule. **120 games minimum** — plan 027 established that 60 games (SE
0.065) cannot see a single doubling, and several of these contrasts are that
size.

### A — the hygiene test

| arm | candidate | opponent | asks |
|---|---|---|---|
| A1 | ensemble 2 x 500 | single 1 x 1000 | does splitting win on *strength*, as it did on labels? |
| A2 | ensemble 4 x 250 | single 1 x 1000 | does more splitting win more? |

A win here confirms the defect on the metric that matters. A *loss* is also
informative and quite possible: the label result is about the shape of the
visit distribution, while strength is about argmax, and plan 025's own numbers
show the ensemble gains far more on TV (0.282 -> 0.237) than on top-1
(0.67 -> 0.68). Strength may simply not inherit the gain.

### B — fix the single tree

| arm | candidate | opponent | asks |
|---|---|---|---|
| B1 | single 1000, norm c=1 | single 1000, raw c=10 | does F1 beat production? |
| B2 | single 1000, raw c=30 | single 1000, raw c=10 | does F2? |
| B3 | single 1000, best of B1/B2 | ensemble 2 x 500 | **the point of the plan**: does the fixed single tree now beat the split? |

### C — does the ceiling move?

Only if B produces a winner. Re-run plan 027's saturation probe under the fixed
configuration:

| arm | candidate | opponent | asks |
|---|---|---|---|
| C1 | fixed, 2000 iters | fixed, 1000 iters | plan 027 measured 0.508 here under raw c=10. Does it become positive once the tree stops burning budget on an arbitrary lead? |
| C2 | fixed, 4000 | fixed, 2000 | where the new ceiling is |

C1 is the commercially interesting one. If the answer is yes, the budget
becomes a lever again and the 1000-iteration knee found in plan 027 was a
property of the *tuning*, not of the game.

## Implementation notes

- **B and C need no code.** `--puct-mode` / `--puct-c` and the per-side
  `--vs-puct-mode` / `--vs-puct-c` already exist on `botbowl-ui eval` (plan
  026), and `--mcts-iters` / `--opponent-iters` cover C.
- **A needs a bot-level change.** Plan 025 says implementation is contained to
  the dataset generator — true for *labels*, but a strength test needs the bot
  to ensemble at action-selection time: k fresh searches, merged root stats,
  then argmax. That is `MctsBot::with_ensemble(k)` in `dynamics.rs`.
- **Merge rule matters and should be stated.** `get_action` currently picks
  best aggregated Q, not most-visited (visits over-weight whichever path
  saturated first). The consistent ensemble merge is therefore a
  visit-weighted mean of per-tree Q, not a sum of visits. Worth measuring both
  — they can disagree — but the Q merge is the one that matches the existing
  selection rule.
- **Tree reuse is a confound.** The single-tree baseline reuses its tree across
  `get_action` calls within a turn when the horizon anchor matches
  (`dynamics.rs:1051`), so it effectively carries more than N iterations into
  later decisions. Ensemble members must each reuse their own tree, or reuse
  must be disabled on both sides. Whichever is chosen, report it — the "equal
  compute" claim depends on it.
- **Cost.** Stage 0 is ~1h for both arms. ~120 games at 1000 iters is roughly
  3h per game arm at `--parallel-games 6`; A and B together are ~15h. Run
  Stage 0 first — it is the cheapest thing here and can redirect the rest —
  then A and B in order, stopping early if A loses decisively, since B3 and C
  are only interesting once there is something to fix.
- **Scheduling.** Every arm needs the box; plan 025 records that the
  convergence probe "competes directly with generation for CPU". The training
  loop currently runs continuously (gen06 eval, then straight into gen07), so
  these need a deliberate pause — `touch runs/loop14x7/STOP` lands the loop at
  the next phase boundary without discarding work, the same seam used all week.

## What would falsify the premise

If A1 and A2 both come back near 0.500 on strength, then the searcher is *not*
leaving strength on the table and plan 025's finding is confined to label
shape. That would be a clean, useful negative: it would mean the over-committing
hurts the training target rather than the play, and the right response is to fix
the label (soften the policy target — plan 025's follow-up 2, a one-line
temperature change) rather than the search.


---

# Results

## Stage 0 (2026-09-06, 50 states x 3 repeats, gen03, `nn` evaluator)

`signal` = TV to the 16000-iteration reference; `floor` = TV between independent
runs at 16000; `top1` = agreement with the reference's chosen action.

### S0a — production `raw(c=10)`: it converges

| budget | signal | ratio to floor | top1 | \|dv\| |
|---|---|---|---|---|
| 100 | 0.6145 | 6.41 | 0.45 | 0.2440 |
| 200 | 0.5032 | 5.25 | 0.53 | 0.1734 |
| 500 | 0.3952 | 4.12 | 0.61 | 0.1224 |
| 1000 | 0.3328 | 3.47 | 0.69 | 0.1027 |
| 2000 | 0.2874 | 3.00 | 0.67 | 0.0830 |
| 4000 | 0.2192 | 2.29 | 0.73 | 0.0647 |
| 8000 | 0.1591 | 1.66 | 0.82 | 0.0498 |
| 16000 | 0.0958 | 1.00 | **0.91** | 0.0190 |

Noise floor 0.0958. Signal falls monotonically to it; top-1 agreement climbs
0.45 -> 0.91; value precision improves 13x.

**Plan 025's headline is refuted on current code.** It reported top-1 agreement
peaking at ~500 and *falling* to 0.55 by 16000, with run-to-run TV rising
0.193 -> 0.383. Neither reproduces. Its own status header called it provisional
(retired `gen01`, pre-`e107f06` search, raw data deleted) and it was right to:
the search no longer amplifies an arbitrary early lead. Both of plan 025's
headline findings have now been overturned by re-measurement — the budget
conclusion by plan 027, the non-convergence by this.

That also removes the motivation for most of this plan. There is no Polya-urn
pathology left to fix, so F3 (FPU) and F4 (tie-break) are not worth pursuing on
these grounds, and arms A (ensembles) are much less interesting: splitting
compute cannot beat a tree that is converging, except through the label-shape
effect plan 025 measured, which is a *labels* argument and belongs with the
policy-target softening idea instead.

### S0b — `normalised(c=1)`: worse, decisively

| budget | signal | top1 | vs raw |
|---|---|---|---|
| 1000 | 0.6570 | **0.25** | raw 0.3328 / 0.69 |
| 4000 | 0.4033 | 0.52 | raw 0.2192 / 0.73 |
| 16000 | 0.1727 | 0.77 | raw 0.0958 / 0.91 |

Noise floor 0.1727 against raw's 0.0958. At the production budget normalised Q
agrees with deep search **0.25 of the time against raw's 0.69**.

Worth understanding why plan 026 read it the other way. Its metric was
**run-to-run** agreement (0.69 norm vs 0.60 raw at budget 1000) — whether two
runs agree with *each other*. This measures agreement with a 16x deeper
**reference** — whether a run is *right*. Normalised Q is more self-consistent
and less correct: reproducibly mediocre. Plan 026's "not the fix" verdict
stands, for a stronger reason than it had.

## The live finding: a flat spot at 1000-2000, not a ceiling

Lay the convergence curve against plan 027's strength arms and they agree
exactly, which is the first time a label metric and a strength metric have
matched in this project:

| span | top1 gain (S0a) | strength (plan 027) |
|---|---|---|
| 250 -> 1000 | 0.53 -> 0.69 | **0.700**, p~0.002 |
| 1000 -> 2000 | 0.69 -> 0.67 (none) | 0.508 |
| 2000 -> 16000 | 0.67 -> **0.91** | **never measured** |

Plan 027 concluded "the search saturates around 1000" from two adjacent spans.
The curve says otherwise: 1000-2000 is a **local flat region**, and the largest
remaining gains sit above 4000, exactly where nothing has been measured.

**Prediction, and the next experiment: 8000 or 16000 iterations should beat
1000 on strength**, despite 2000 not doing so. If it holds, the iteration
budget is a lever again and plan 027's "keep 1000" needs qualifying to "keep
1000 *or go far past it* — 2000 is the worst of both". This supersedes arms A
and B as the priority.

The cost is the catch: 16000 iterations is 16x generation cost, so this is a
question about the *evaluation and analysis* budget, or about a much smaller
number of much better games, rather than a drop-in change to the loop.

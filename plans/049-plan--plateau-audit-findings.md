# Plan 049 — the loopmix16x9 plateau: findings that need work to confirm or refute

**Status:** Written 2026-09-29 from a three-part code audit of generation, search and training.
The engine rules were out of scope (validated by human play). Nothing here is confirmed as the
cause of the plateau yet. Each finding lists the evidence gathered so far, the mechanism it would
act through, and the cheapest test that would confirm or refute it. Results go under each item.
Scratch scripts behind the numbers are in the audit session's scratchpad
(`cqkl.py`, `byslice.py`, `trainslice.py`, `an1.py`…`an8.py`, `analyze*.py`); re-derive before
relying on them.

## Headline, 2026-09-30: virtual loss leaks and buries the best root children

Found from a web-app screenshot, after the audit: a root whose most-visited child had a 0.1% prior
and a negative Q. `bump_chosen` adds `virtual_loss` (30) to every child it selects, and the penalty
is cleared only when a backprop *replaces* that child's score. A descent cut off below the root
child, for example at a chance node that withholds its value until every outcome is scored, never
clears it, so the penalty accumulates and buries whichever children keep hitting that case. This
happens at **one worker too**, and every loop generation and benchmark ran single-worker with
virtual loss 30. An opt-in trace of real descents per root child
(`MctsConfig::trace_root_descents`, `tests/root_visit_anomaly.rs`) showed the best-Q child of one
position getting 12 of 1999 descents; at virtual loss 0 the search concentrates on it, and in
another position finds a +0.213 line the default search held at +0.078.

**exp054: gen21 with virtual loss 0 vs gen21 with virtual loss 30,** exact roll model, 500 visits,
200 per board: **0.686 ± 0.021** (W239 D71 L90, TD 751:371; 14x7 0.695, 16x9 0.677). That is the
largest effect this programme has measured, from a search change alone.

Consequences for the findings below. Every corpus behind them was searched with the leak, so its
Q values and cq targets came from searches that buried their best lines. Findings 1 (the target
copies the prior), 5 (winner's curse) and 6 (fan-width exploration) should be re-measured on a
corpus generated at virtual loss 0 before any fix is built on them. The A/B results (τ = 20,
exploration) are valid as comparisons, since both arms shared the leak, but their absolute
levels are understated.

Also found: a node's reported **visits are not descents**. Backprop sets visits to the sum of the
children's visits, so a subtree recombined under several root children is credited to each of
them, and a cut-off descent is not credited until the next full backprop. The web drawer's visit
column, the `visits` policy target and `BudgetMode::Visits` all read this count.

**Next:** 1-worker runs use `virtual_loss = 0` now (`cfgs/exact_visits_vl0.toml`). The
multi-worker fix is to revert each descent's virtual loss when the descent ends, which needs a
`recon_mcts` hook; the web app runs 8 workers. Then relaunch the loop at virtual loss 0.

## Headline, 2026-10-01: the post-fix loop regressed, and on 14x7 tau 20 explains it

`runs/loopmix16x9vl0` (launched at `a645386` from gen21 with every fix above: virtual-loss fix,
exact roll model, cq tau 20, exploration, 500 visits) read **below gen21 at every generation**.
Points vs gen13, 200 per board:

| | 14x7 | 16x9 |
|---|---|---|
| gen21, same commit, seed and search (`baseline_gen21`) | 0.535 | 0.435 |
| gen01 / gen02 / gen03 / gen04 | 0.438 / 0.468 / 0.458 / 0.445 | 0.307 / 0.333 / 0.310 / 0.385 |
| gen01-04 pooled, 800 per board | 0.452 | 0.334 |

The pooled gap is 0.485 vs 0.393, about 3.7 sigma. The drop came at gen01, and gens 2-4 are
flat. Stopped 2026-10-01 at gen06.

**Hypothesis (the user's):** the fixes made the search branch more (passes, fouls and armour
now have real outcome distributions). The budget was cut to 500 *visits* (plan 045, measured
under the leak), and tau 20 trusts that shallow search's Q more than tau 100 did. Together that
gives noisy targets, worst on the wider 16x9 fan. Measured support:
- **Real descents per decision fell from ~330-345 to ~255** at "500 visits" (exp049/053 vs
  `baseline_gen21` telemetry). Visits are DAG sums and include the reused subtree (about 60% of
  searches reuse one). The virtual-loss fix concentrates descents on shared lines, so the visit
  count reaches 500 sooner.
- The 16x9 root fan reaches 130 legal actions, and 29% of 16x9 roots have 20 or more. With ~255
  descents, most children get one or two.
- Each fine-tune barely moves validation loss (gen01: value 0.1151 → 0.1142, policy ≈ flat), so
  only games can separate the arms.

**exp055** (`scripts/archive/exp055_tau_and_budget.sh`, `a8c7f9a`). Aborted at 10:45 for plan 051, so
the numbers are partial:

| 14x7 vs gen13 | points | n |
|---|---|---|
| gen01's corpus re-prepared at **tau 100**, fine-tuned from gen21 exactly as gen01 was | **0.565 ± 0.032** | 200 |
| gen21 control, same batch | 0.543 ± 0.034 | 174 |
| gen01 (tau 20, the same corpus) | 0.438 ± 0.031 | 200 |

On 14x7, tau 100 recovers all of gen01's loss (+0.13 over tau 20 on identical data). This
**reverses** the leaky-search result (exp052: tau 20 +0.064 over tau 100). That fits the
hypothesis: once the search concentrates honestly but shallowly, sharpening onto its Q hurts.
16x9 had only 27 games (0.407), so it is unresolved.

**Still open (cheaper once plan 051 lands):**
1. tau 100's 16x9 rung (the remaining 173 games).
2. gen21 at 500 / 1000 / 2000 *real descents* (`cfgs/exact_iters.toml`) against gen13 at 500
   visits, on both boards: does strength still climb past ~255 descents, and more steeply on 16x9?
3. If it climbs, fine-tune gen21 on ~1600 games generated at the bigger budget (equal compute to
   gen01's 4800 at 500 visits) and judge it against gen21 in the same batch. If it beats its
   parent, relaunch the loop at that budget, on a descent budget rather than visits.

Whatever (2)-(3) say, the next loop should not run cq tau 20 at the current budget.

## 2026-10-02/03: the search budget, re-measured with plan 051's tools (exp056-058)

- **exp056** (drives, gen21 at N real descents vs gen21 at 500 visits, about 255 descents):
  - **14x7:** 500 descents gives 0.586 (H1), 2000 gives 0.572 (H1), 1000 gives 0.549.
  - **16x9:** 500 descents gives 0.505 ± 0.029, flat.
  - **Tau arms vs gen21:** tau100 0.516 and tau20 0.479. Not trusted yet: both nets were trained on
    corpora from the old budget.
- **exp057** (`runs/exp057/summary.txt`, convergence vs a 16000-descent reference, regret in Q
  points):
  - **14x7:** regret falls 27 -> 20 -> 14 -> 11 -> 10 over 250 / 500 / 1000 / 2000 / 4000, so it
    flattens after about 1000.
  - **16x9 mid-turn roots (about 98 legal moves):** flat from 125 to 2000 (top-1 about 0.45,
    regret about 27), improving only at 4000 (0.59, 17).
- **exp058** (16x9 drives, gen21 at 4000 descents vs 500 visits): **0.576 ± 0.028, H1 after 54
  pairs.** Strength follows the convergence curve: the big boards do gain from search, but only
  past about 2000 descents.

**Decision for the loop** (`runs/loopmix16x9d1k`, launched 2026-10-02 23:25 by
`scripts/launch_d1k.sh`): 1000 real descents on every board. 4000 on the big boards would cost
about 15 h per generation on the training box: both sides search at the generation budget, and
the trees cap the worker at about 4 streams. The candidate fix for wide fans is Gumbel root
selection with sequential halving. It spends a small budget on the best few root candidates
instead of thinly over about 100, it is used both when playing and when generating, and our cq
target is already its policy-target half. It is not written up yet. `GEN_SPLIT` in train_loop.sh
allows per-board-group budgets.

## 2026-10-03: the policy target at 1000 descents (exp059), and the d1k loop on drives

**Loop on drives vs gen21** (contested sets screened with gen21, SPRT 0.5:0.55):
- gen03: 14x7 0.464 ± 0.029 (H0), 16x9 0.396 ± 0.047 (H0).
- gen04: 14x7 0.507 ± 0.015 (H0, 308 pairs), 16x9 0.493 ± 0.024 (H0, 145 pairs).

Four generations at 1000 descents and the loop is level with gen21 at best. The loop is paused
after gen04.

**Target arms** (`scripts/archive/exp059_targets_and_wdl.sh`):
- **Recipe:** gen04's exactly, with one change each: the gen02-04 window, warm from gen03. gen04
  (cq tau 100) is the control.
- **Scoring:** drives vs gen04 on positions re-screened with gen04 (156 kept on 14x7, 209 on
  16x9, from the same 500 candidates).

| arm | 14x7 | 16x9 |
|-----|------|------|
| cq tau 50 | **0.500 ± 0.017 (fixed 300 pairs)**; the SPRT run's H0 latched at 16 pairs, see below | 0.492 ± 0.024 H0 (124) |
| cq tau 20 | 0.497 ± 0.021 H0 (150) | 0.462 ± 0.030 H0 (80) |
| Gumbel σ(q) | 0.433 ± 0.040 H0 (47) | 0.455 ± 0.033 H0 (63) |
| Gumbel σ(q), min range 50 | 0.476 ± 0.025 H0 (109) | 0.450 ± 0.034 H0 (78) |

- **Sharper is not better.** At the new budget, tau 20 and both Gumbel σ(q) arms are at or
  below tau 100. The Gumbel arms are the worst on both boards, and both trained to the end of
  the schedule (best at step 12500-15000, against 2500-5000 for the cq arms), so they move
  further from the prior.
- **tau 50 is level with tau 100.** Its 14x7 SPRT reached H0 after only 16 pairs, while the 46
  pairs it finished on scored 0.549. A fixed 300-pair re-check
  (`runs/exp059/drives/tau50_14x7_fixed/`, a fresh seed) gave exactly 0.500 ± 0.017.
- **Decision: keep cq tau 100.** Neither a sharper fixed tau nor the adaptive Gumbel σ(q) target
  helps at 1000 descents. With WDL also flat (plan 050 step 1), neither the targets nor the value
  head is what holds the loop at gen21's level. The next candidate is the search itself: Gumbel
  root selection with sequential halving (plan 053, to write).
- **Method note:** an SPRT may decide from `SPRT_MIN_PAIRS = 8` pairs, where the pentanomial
  variance is still guesswork. The early tau 50 stop shows the risk. Raise the floor to about 30
  before trusting early stops.

## The plateau being explained

`runs/loopmix16x9` gen14-21 read 0.50 ± 0.03 against the frozen gen13 anchor on 14x7 and 16x9,
while validation loss kept improving slowly (0.755 → 0.718). The fine-tune's best-val checkpoint
is usually its first (step 2500 of ~40k). Plan 047 ruled out data volume: 3.7× the data fine-tuned
from gen21 scored 0.405, from scratch 0.288, against gen21's 0.513.

## Ranking

| # | finding | confidence | cost to test |
|---|---|---|---|
| 1 | the cq policy target is nearly the generator's own prior | measured three ways | prepare + train + 400 games, ~4 h |
| 2 | checkpoint selection ships nets worse than their init | measured on gen20/21 | one log line + offline scoring, < 1 h |
| 3 | the optimiser sits at its noise floor | train loss flat, measured | one train + offline scoring, ~1 h |
| 4 | the value label is self-referential and optimistic | bias measured, effect not | prepare + train + 400 games, ~4 h |
| 5 | root pick by max-Q has a winner's curse in wide fans | drift measured | search-only A/B, ~2 h |
| 6 | effective exploration strength scales with fan width | derived from code | search-only A/B, ~2 h |
| 7 | reroll prompts may encode like their pre-roll state | speculative | one scratch test, < 1 h |
| 8 | training covers single drives, eval plays full games | structural | a full-game eval split, ~2 h |
| 9 | tree reuse misses in 28% of searches | measured | re-measure only, minutes |

1-3 interact: if the target carries almost no new information (1), a noisy optimiser (3) and a
noisy restore rule (2) are enough to leave each generation a random step away from its init,
which is what the curve shows. They should be tested together, in that order.

---

## 1. The cq policy target is nearly the generating net's own prior

**Mechanism.** `botbowl-nn/src/targets.rs` `completed_q_target` builds
`softmax(ln prior + q_mover / τ)` with τ = 100 (`train_loop.sh` `CQ_TAU`). `prior` is the
generating champion's own softmax at full weight. Q gaps between the top root children are small
against the log-prior gaps, so the target is the prior plus a nudge the fine-tune never absorbs.
Each generation then distils a blend of the last three generators' priors instead of learning
from the search. This makes plan 032's qualitative "the cq target is self-referential" note
quantitative.

**Evidence** (gen21 shards, unsolved roots with more than one child; gen15 agrees within ±0.01):

| target | KL(target ‖ prior) mean / median | argmax changes vs prior | argmax = move played |
|---|---|---|---|
| cq τ = 100 (production) | 0.053 / 0.006 nats | 7.7% | 64% |
| cq τ = 20 | 0.31 / 0.045 | 19% | 73% |
| Gumbel σ(q) (normalised q̂, (50 + maxN)·0.1 scale) | 0.50 / 0.09 | 24% | 79% |
| visit counts | 2.32 / 1.28 | 43% | 52% |

- The search's best-Q move differs from the prior's top move in **41%** of decisions. There the
  median Q gap in its favour is **27 points** (0.27 nats at τ = 100), against a median log-prior
  gap of **1.84 nats**. The target moves that move's mass from 0.075 to 0.115 (median).
- **The generating net fits each val slice best.** gen21's `prepared_val` split by source
  generation, policy KL = CE − H(target), 2800 samples per slice:

  | slice (made by) | gen18 | gen19 | gen20 | gen21 |
  |---|---|---|---|---|
  | gen19 (by gen18) | **0.048** | 0.059 | 0.059 | 0.060 |
  | gen20 (by gen19) | 0.062 | **0.048** | 0.060 | 0.062 |
  | gen21 (by gen20) | 0.073 | 0.069 | **0.058** | 0.069 |

  The generator wins every slice by 0.010-0.015 nats (paired SE ≈ 0.0014). No later net, including
  those trained on the slice, gets closer to the target than the generator's prior already was.
- The drift gets worse as the prior sharpens: TV(target, prior) 0.091 (gen02) → 0.067 (gen21),
  H(prior) 0.876 → 0.746. τ = 100 was chosen at gen03 and loses its grip as the prior sharpens.
- About 91% of `val_policy` (0.582 = H(target) 0.532 + KL 0.050) is target entropy that cannot be
  reduced. That explains "best-val at step 2500" on its own.

**Caveat.** A sharper target is not automatically better. Q rises with visits (median +41 points
per e-fold of visits, the max-over-noise effect of finding 5), so sharpening on Q also rewards the
children the prior already favoured. Visits are the most move-changing target but agree with the
move played least often.

**Test.** A one-generation, prepare-only A/B off gen21, mirroring exp048: same window
(gen19-21), same init, same val set; arms τ = 100 (control), Gumbel σ(q), and one of τ = 20 or
visits. 200 games per board vs gen13 for each arm. Adopt the best arm if it beats control by
≥ 2 SE. The generator-fits-own-slice table should also be re-run on the winner's next generation:
if the fixed point is broken, a later net should beat the generator on its own slice.
**Log per generation from now on:** KL(target ‖ prior) and the argmax-change rate, from `prepare`.

**Result, first A/B (exp050, 2026-09-30):** exp048's greedy arm with only the target changed
(warm from gen21, window gen20-22, 3 epochs, val = shards 4,7 of gen22 and gen22x under the arm's
own target). Scored against gen13 under the legacy roll model on both seats, so the control's seeds
and model match; 200 games per board.

| target | vs gen13 (400) | minus τ=100 control (paired) | minus gen21 (200) | restore |
|---|---|---|---|---|
| cq τ = 100 (control, exp048 greedy) | 0.398 | — | −0.133 ± 0.044 | step 5000 |
| **cq τ = 20** | **0.464 ± 0.025** | **+0.066 ± 0.032** | −0.072 ± 0.047 | step 7500 |
| visit counts | 0.246 ± 0.022 | −0.151 ± 0.029 | −0.282 ± 0.039 | step 35000 |

- Visit-count targets are clearly harmful: drop them.
- τ = 20 beats the control by about 2 SE, the same size as exploration in exp048 (τ20 minus the
  exploring arm: +0.001 ± 0.032). But like every fine-tune of gen21 on this window, it lands
  below gen21 itself.
- **Seed variance measured (exp052, 2026-09-30): small.** The τ = 100 control retrained with
  `--seed 1` and `--seed 2` read 0.381 and 0.422 against the unseeded run's 0.398 on the same 400
  games, an SD of 0.021 across seeds, below the ±0.025 game noise. Against the 3-seed control
  average, **τ = 20 is +0.063 ± 0.025** and **exploration (exp048) +0.065 ± 0.025**. Both effects are
  real, and they act on different things (the target vs the corpus), so they may stack; that is
  untested.
- **gen21's own level, re-measured (exp053, 2026-09-30): 0.472 ± 0.025** over the same 400 games
  (14x7 0.515, 16x9 0.430). Its loop benchmark's 0.512 was a lucky draw: replaying the same first
  100 per board gave 0.470, with the same result in only 74 of 200 games, since search is not
  reproducible run to run. Against gen21 on the same 400 games:

  | fine-tune of gen21 | minus gen21 |
  |---|---|
  | τ = 100 control, 3 seeds | −0.075, −0.091, −0.050 (each ± 0.031) |
  | **τ = 20** | **−0.009 ± 0.030** |
  | **exploration corpus (exp048)** | **−0.007 ± 0.032** |

  So the "0.11 drop" was mostly gen21's lucky benchmark. The rest is real: **a τ = 100 fine-tune
  costs about 0.07 per step**, which a flat loop curve could hide at 100 games per board. **τ = 20 and
  exploration each make the step neutral**: no loss, but no gain from a single step yet either.
- **TODO, next target A/B: Gumbel σ(q) against τ = 20.** Implemented 2026-09-30 as
  `prepare --policy-target gumbel` (`--gumbel-c-visit 50 --gumbel-c-scale 0.1
  --gumbel-min-range 0` by default). It is `softmax(ln prior + (c_visit + maxN)·c_scale·q̂)`, with
  q̂ the root's min-max-normalised completed Q, so the sharpness adapts to each root's own Q
  spread instead of a fixed τ. On gen22 shard 4 its argmax differs from cq τ=100's on 17.6% of
  multi-child roots, and it is sharper (mean entropy 0.42 vs 0.67). Run it as an exp050-style arm:
  the same window and init, prepare-only, no new games needed, and scored with the same
  legacy-model eval so exp048/exp050 stay the controls. Worth also trying `--gumbel-min-range 50`,
  so near-tied roots do not sharpen on max-over-noise (finding 5). Not tried yet either: τ between
  20 and 100.


## 2. Checkpoint selection ships nets worse than their init

**Mechanism.** `train.py:287-289` logs only `val_value` for the warm start and excludes it from
restore, so candidates start at step 2500 and the init can never be kept. Checkpoint-to-checkpoint
differences in `val_policy + val_value` are 0.001-0.003, at the noise level, so "best" is mostly
noise: a random step away from an init that may have been better.

**Evidence.** gen21's warm-start `val_value` 0.1119 beats every checkpoint (the restored step
22500 has 0.1124). On a 6000-sample val subset, CE is gen20 0.5976 vs gen21 0.6012, with equal
weighted MSE. gen21 is also worse than gen20 on the slices gen20 did not generate.

**Test.** Log the full combined criterion for the warm start (one line). Then score each shipped
net against its own init on the next generation's held-out data, gen14-21, and count how many
shipped a regression. If most did, make "keep the init" a restore candidate. A gateless loop that
can only ship noise-steps random-walks around its init.

**Result, offline part (2026-09-29): value side confirmed flat; policy side not testable offline.**
Each shipped net gen14-21 against the net it was warm-started from, both scored on the *next*
generation's held-out shards (4, 7). Those shards were generated after both nets existed and are in
no training window. Paired per sample, about 15.5k samples each:

| shipped vs init | Δ policy CE | Δ value MSE, blended label | Δ value MSE, **outcome-only label** |
|---|---|---|---|
| gen14 vs gen13 | −0.0167 ± 0.0007 | −0.0009 ± 0.0005 | +0.0006 ± 0.0009 |
| gen15 vs gen14 | −0.0139 ± 0.0007 | +0.0002 ± 0.0005 | +0.0018 ± 0.0008 |
| gen16 vs gen15 | −0.0111 ± 0.0006 | +0.0006 ± 0.0006 | +0.0014 ± 0.0010 |
| gen17 vs gen16 | −0.0113 ± 0.0006 | −0.0030 ± 0.0007 | −0.0043 ± 0.0011 |
| gen18 vs gen17 | −0.0178 ± 0.0007 | +0.0001 ± 0.0005 | −0.0004 ± 0.0009 |
| gen19 vs gen18 | −0.0124 ± 0.0006 | +0.0002 ± 0.0006 | +0.0005 ± 0.0009 |
| gen20 vs gen19 | −0.0125 ± 0.0006 | −0.0008 ± 0.0005 | +0.0013 ± 0.0009 |
| gen21 vs gen20 | −0.0148 ± 0.0006 | +0.0006 ± 0.0004 | +0.0008 ± 0.0007 |

- **The policy column is confounded and says nothing.** Generation g+1's data was generated *by*
  the shipped net g, so its cq targets are built on net g's own prior, and finding 1 already
  showed the generator always fits its own slice best. Every offline policy comparison has this
  problem: each slice's target carries its generator's prior. The audit's "gen21 is worse than
  gen20" came from slices of gen18-20 data and is confounded the same way.
- **The value column against pure outcomes is neutral, and flat.** The shipped net was worse than
  its init in 6 of 8 generations and better in 2, clearly only at gen17. The mean is about +0.0002,
  so the value head does not improve generation to generation, and which checkpoint the restore
  rule keeps is noise on the value side. That is consistent with finding 4's flat val MSE since
  gen13.
- **Remaining test (needs games):** shipped net vs its init head to head, e.g. gen21 vs gen20 and
  gen18 vs gen17, 200 games per board. If the shipped net does not win either, adopt "keep the init
  unless beaten by more than the noise" in the restore rule.


## 3. The optimiser sits at its noise floor

**Mechanism.** Adam at a constant lr of 2e-4, batch 32, no schedule and no weight averaging
(`train.py:266`, `:356-365`). If the learnable signal per generation is ~0.05 nats (finding 1),
SGD noise at this lr is the same size, and the fine-tune drifts rather than learns.

**Evidence.** gen21's train policy_loss is 0.5977 at step 2500 and 0.5977 at step 32500; train
top-1 goes 0.673 → 0.677, and the train–val gap is 0.007. gen14-20 show the same. Capacity was
ruled out earlier (plan 032 #9: 96×8 ≈ 64×6).

**Test.** Fine-tune one window (gen19-21 from gen20) at lr 2e-5, at batch 256, with cosine decay
and with an EMA of the weights. Score each on the gen21 slice's KL against the generator's own
0.058 (finding 1's table). An arm that gets below the generator on a slice it did not generate is
learning; none of the production nets do. Offline only, no games until something beats 0.058.

**Result.** (pending)

## 4. The value label is self-referential and optimistic

**Mechanism.** `targets.rs` `value_target_blended` at `--value-blend 0.5` gives half the label's
weight to the generator's own root value, which comes from its own value head through a
minimax search. That is the fixed-point pull of finding 1, plus the search's known max-over-noise
optimism (plan 031 D1). A head biased towards the side to move undervalues every state where the
opponent moves, which over-penalises turnover risk. Warm starts compound it.

**Evidence.**
- E[root − outcome] in the mover frame on gen21 is +0.07 overall, +0.16 at the mover's turn 7 and
  +0.19 to +0.24 at turn 8. The label therefore carries +0.035 overall and up to +0.12 late in a
  half.
- EndTurn Q averages −0.2 in the mover frame while the root averages +0.3.
- Plain val MSE is flat since gen13: 0.1254 (gen13), 0.1269 (gen17), 0.1257 (gen20), 0.1257
  (gen21), R² ≈ 0.73 throughout.

**Test.** First offline: score val MSE against `outcome_value` alone, net by net, and rerun
`scripts/audit_value_head_bias.py` with gen20 on gen21 rows. If the bare head's mover bias is now
well above gen03's +0.031, the blend has built it in. Then a one-generation A/B of value blend
0.5 vs 0.75 (or vs 1.0, pure outcome), prepare + train + 400 games.

**Result.** (pending)

## 5. The root pick by max-Q has a winner's curse in wide fans

**Mechanism.** `pick_best_action` (`dynamics.rs`) takes the argmax of minimax Q with no visit or
confidence requirement. With a noisy NN leaf, the child whose Q is highest is disproportionately
the one whose noise was most favourable, most of all in wide fans (move destinations, activation
choice). The trajectory then follows noisy picks, and the value label (via the blend, finding 4)
learns the returns of a policy other than the one the policy head learns: the played move
disagrees with the target's argmax 35% of the time.

**Evidence.** E[next root value − Q(played child)] in the previous mover's frame over 15,040
transitions: −15.3 ± 1.3 overall; `Move` −50.8 ± 3.2 (n = 3802), `StartBlock` −48.8 ± 6.1,
`StartHandoff` −87 ± 12, `DontUseReroll` −141 ± 38, `FollowUp` −21.7 ± 3.8. Voluntary `EndTurn`
is consistent at about ±3. Mean backup (plan 032 #2) and c/FPU tuning (#3) are settled and not
re-proposed.

**Test.** A search-only A/B on the same net: root pick = max Q among children with ≥ 5% (or 10%)
of root visits, or a lower-confidence-bound pick, against the shipped rule; 200 games per board.
Re-measure the drift table on each arm's games.

**Result.** (pending)

## 6. Effective exploration strength scales with fan width

**Mechanism.** NN priors are softmax × n (mean 1) and Q is on a ±1000 scale, so the
AlphaZero-equivalent c_puct is about 0.01·n: 0.09 at the median fan (9), 0.9 at p90 (89), against
~1.25 in AlphaZero. Narrow roots are searched almost greedily on Q, with first-play urgency at the
parent's Q. The visit distribution is therefore Q-driven, while the target (finding 1) is
prior-driven, so the two disagree most on exactly the narrow decisions.

**Evidence.** Derived from `select_node` and `botbowl-nn/src/eval.rs`. Plan 032 #3's c sweep
(3 / 10 / 30) used the heuristic evaluator on gen03 and found nothing, but it never varied the
fan-width scaling.

**Test.** A search-only A/B: priors normalised to sum 1 with c rescaled to match at the median
fan, against shipped, 200 games per board. Cheaper first look: the fraction of visits going to
non-best-Q children, binned by fan width, on existing corpora.

**Result.** (pending)

## 7. Reroll prompts may encode like their pre-roll state

**Mechanism.** `encode.rs` `encode_raw` has no input for the procedure or decision context. A
failed-roll reroll prompt may encode as nearly the same tensor as the neighbouring non-failure
state, while its value is very different. That would put irreducible noise into the value head
exactly where the search leans on it (these are common single-visit leaves).

**Evidence.** Mean mover outcome by reroll prompt on gen21: Dodge −0.36 (n = 199), GFI −0.03
(75), Pickup +0.20 (59), Catch +0.34 (32). About 4% of samples.

**Test.** A scratch test: build a pre-roll state and the matching failed-roll prompt and diff
`encode_raw`. If they differ only in the path plane, split `val_value` by top-of-stack procedure to
see whether these samples carry outsized error.

**Result (2026-09-29, `botbowl-nn/tests/reroll_prompt_encoding.rs`): partly refuted.** A failed
dodge's reroll prompt against the state a successful dodge leaves: the two differ in exactly one
plane, `path_prob` (118 cells populated after a success, none at the prompt). Every other plane and
all 18 globals are identical. So the value head *can* separate them, but only through an absent
path plane, a cue shared with every other no-path state, and nothing encodes which roll is being
rerolled (dodge vs GFI vs pickup vs catch). Not worth more investigation. The cheap fix, for the next
schema bump, is a small one-hot global for the decision kind (top-of-stack procedure). The test now
pins that the prompt and the success state stay distinguishable.

## 8. Training covers single drives; eval plays full games

**Mechanism.** The corpus is drive-bounded random starts: no setup, kickoff or ball-in-air states,
and nothing after the drive ends (the clock, casualties carried over). The anchor benchmark plays
full games from kickoff. Anything the net would need to get better at in those phases is never
trained, and the drive objective itself may be close to saturated: 83% of corpus drives end in a
touchdown (207/250 sampled).

**Test.** Split the anchor benchmark's per-game records by phase: drives that start from a kickoff
the candidate received vs kicked, and TD rate by drive start. If the candidate and anchor are even
on received-kickoff drives but differ elsewhere, the gap is in untrained phases. A heavier
follow-up is a corpus with a share of full-game self-play.

**Result.** (pending)

## 9. Tree reuse misses in 28% of searches

**Evidence.** gen21 meta: 3638 reused, 1668 `LookupMiss`, 311 `AnchorMiss` out of 5940 searches.
By procedure: `Block` 1589/1703 misses (known, plan 032 #13), `BlockAction` 455/883, `MoveAction`
2190/6670, and every Kickoff/Touchback search. A lookup miss means the real game reached a state
the tree never modelled: it measures how far the search's model is from the game.

**Test.** Re-measure after the current search changes land, then log the reuse outcome next to
"which roll resolved since the last decision" to attribute what is left.

**Result (2026-09-30): the misses are structural.** Re-measured on exp049's games (gen21, the
current search) against gen21's loop benchmark: lookup misses were 30.7% vs 31.9% overall, with
MoveAction at 32% vs 33%, BlockAction at 48% vs 52% and Block at 96% vs 93%. What is left comes
from states the search never reached, such as roll outcomes it never visited and the opponent's
turn, plus the known Block case (plan 032 #13), not from a wrong model of the game. Closed.

## Order of work

1. Finding 9 re-measure and 7's scratch test: minutes, and they sharpen the rest.
2. Findings 2 and 3 offline: no games, and they say whether 1's A/B can even show up.
3. Finding 1's prepare-only A/B, with 3's winning optimiser setting if one wins.
4. Finding 4 offline, then its A/B if the bias has grown.
5. Findings 5 and 6: search-only A/Bs on one net, which can share a night.
6. Finding 8's eval split: whenever a benchmark has just finished.

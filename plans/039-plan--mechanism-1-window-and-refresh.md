# Plan 039 — the fine-tune has nothing left to learn (mechanism 1)

**Status:** ANSWERED 2026-09-18 — both hypotheses rejected, no games bought. Results at the bottom.

## The observation

Since the plan-036 recipe went in at gen04, every fine-tune's validation curve is flat:

| gen | recipe | val_policy spread | val_value spread | restore |
|---|---|---|---|---|
| gen02 | old | 0.0117 | 0.0785 | epoch 0 |
| gen03 | old | 0.0078 | 0.0986 | epoch 0 |
| gen04 | new | 0.0289 | 0.0093 | epoch 9 |
| gen05 | new | 0.0063 | 0.0088 | epoch 7 |
| gen06 | new | 0.0051 | 0.0090 | epoch 9 |
| gen07 | new | 0.0050 | 0.0077 | epoch 2 |

Spread is max−min across every checkpoint of the run. The value head no longer degrades after its
optimum — that is plan 036 working — but neither head *moves*. The restore step is being selected
from noise: epochs 9, 7, 9, 2 are draws, and reading them as a trend is what led to the wrong
`EPOCHS` 10→15 change on 2026-09-17 (corrected to 3 on 2026-09-18).

Meanwhile the net **is** improving, just not inside a fine-tune: val_policy falls 1.63 → 1.15 from
gen02 to gen07, and the anchor curve went 0.662 → 0.925. The learning comes from the data
distribution changing under a nearly-converged net.

This is plan 036's **mechanism 1** — "two thirds of every window was in the previous fine-tune's
training set; the warm-started net is already at its optimum on that part" — which W1-W5
deliberately did not address. It is now the binding constraint.

## What plan 029 already settled, and why this is not a re-run

Plan 029 (ANSWERED 2026-09-07) tested exactly the obvious fix, at 120 games per arm:

| regime | 3× wider window | points | z | 95% CI |
|---|---|---|---|---|
| from scratch | D3 vs D1 | **0.600** | +2.51 | [0.522, 0.678] |
| warm-started | W3 vs W1 | **0.487** | −0.30 | [0.405, 0.570] |

Its conclusions 3 and 4: `WINDOW_GENS` should not change on that evidence, and warm-started
training barely moves the net at all. The second is precisely what the table above re-measures
independently, three months and one engine later.

So **widening the warm-started window is the hypothesis prior work rejected**, and from-scratch
retraining on the accumulated corpus is the one it measured as positive (+0.05-0.06 points per
doubling, three quarters volume / one quarter diversity). "Periodic from-scratch retrain" is
already a ranked follow-up in plan 032.

Two things justify measuring again rather than just citing it:

1. **Everything underneath changed.** Plan 029 ran pre-engine-fix, pre-v5/v6 encoder,
   pre-varied-players, and pre-plan-036 recipe. Every corpus it used is unreadable now.
2. **Its warm CI is wide.** [0.405, 0.570] excludes the +0.10 the from-scratch arms showed, but
   not a real +0.05.

## Arms

One held-out val set for every arm — the newest generation's val shards — so the arms differ only
in their *training* pool. Fixed step budget (`--max-steps`), not epochs: plan 029 trap 2, comparing
pools of different size at a fixed epoch count silently compares different amounts of gradient
descent. Same seed, same `--eval-every`.

| arm | init | training pool | tests |
|---|---|---|---|
| `w3_warm` | gen(N−1).pt @ 2e-4 | last 3 gens | production baseline |
| `w7_warm` | gen(N−1).pt @ 2e-4 | all gens | the window hypothesis (029 says no) |
| `w7_scratch` | random @ 1e-3 | all gens | the refresh hypothesis (029 says yes) |

Primary offline read: **does the curve move at all** — val_policy/val_value spread and restore step,
the quantity the table above shows collapsing. An arm that also restores flat has not fixed
mechanism 1 whatever its val numbers say.

Then games, pre-committed sizing: `w7_warm` vs `w3_warm` and `w7_scratch` vs `w3_warm`. Plan 032's
rule — an effect of ±0.03 needs ~600 games, ±0.05 needs ~200. Size on the effect the offline read
suggests, and write the stopping rule down before looking, as in `036-e4-preregistration.md`.

## Cost and ordering

The offline arms are free of generation games and can run alongside the loop, niced. `w7_scratch` is
the expensive one (from random init it genuinely needs the steps — plan 029 saw turnover at
17,500-92,500). Run the three offline arms first; only buy games for an arm whose curve actually
moves.

## If both fail

Then the lever is not the training set but the data: more *new* drives per generation (plan 036 W7),
which is the one thing that raises the fraction of each window the net has not already fitted. It
costs generation time, which is the loop's dominant phase, so it is last.


## Results (2026-09-18, on gen08)

Three arms, 110,000 steps each, one held-out val set (gen08 shards 4/7, in no training pool),
identical label (`cq tau 100`, blend 0.5) so `val_policy` is comparable across arms.

| arm | pool | samples | vp_spread | val_policy@restore | restore |
|---|---|---|---|---|---|
| `w3_warm` | gen06-08 | 347,501 | 0.0074 | **1.0382** | 55000 (0.50) |
| `wide_warm` | gen02-08 | 787,896 | 0.0140 | 1.0794 (+0.041) | 27500 (0.25) |
| `wide_scratch` | gen02-08 | 787,896 | 0.1380 | 1.1053 (+0.067) | 90000 (0.82) |

**Both rejected. Neither arm bought games** — the pre-stated rule was that a flat arm has not
addressed mechanism 1, and that a val_policy better than `w3_warm` is necessary before paying for a
match. Neither condition was met by either arm.

### The window hypothesis — no, again

`wide_warm` is still flat (0.0140 against production's 0.0074 — twice a very small number) and is
**worse** at the restore, +0.041 val_policy. 2.3× the training pool changed nothing about whether
the fine-tune learns. This independently reproduces plan 029 stage 3 (0.487, z=−0.30) on a different
engine, encoder, generator and value target, and settles the wide CI that justified re-measuring.

### The refresh hypothesis — it learns, and still loses

`wide_scratch` is the only arm that genuinely moves: vp_spread 0.138, 18× production's. It is doing
real learning, 1.2412 → 1.1053. And it is not budget-starved — the curve flattens by ~40k steps
(1.1199) and the last five checkpoints oscillate around 1.105 with no trend, so the restore at 90000
is convergence, not a truncation.

It converges **worse than the warm net starts**. `w3_warm`'s very first checkpoint is 1.0435, better
than `wide_scratch` ever reaches. Eight generations of accumulated weights are worth more than a
full retrain on the entire accumulated corpus.

Note this does not contradict plan 029, which compared from-scratch against from-scratch at
different data volumes (+0.05-0.06 per doubling) and never claimed from-scratch beats warm. What it
does do is close the "periodic from-scratch retrain" follow-up ranked in plan 032: at this corpus
size, a refresh costs an hour and gives back a weaker net.

### What this means

Mechanism 1 is not reachable by rearranging the training set. The flatness is not "the window is too
small" or "the net is stuck in a warm-start basin" — it is that the net has already extracted what
this corpus contains, and the corpus is what has to change.

Per the plan's own "If both fail": the remaining lever is more *new* drives per generation
(plan 036 W7) — the only thing that raises the fraction of a window the net has not already fitted.
It costs generation time, which is the loop's dominant phase, so it is a real trade rather than a
free win.

Read positively: the loop **is** compounding, just through the weights rather than through any
single fine-tune. val_policy fell 1.63 → 1.04 over gen02-08 while no individual fine-tune moved more
than 0.01.

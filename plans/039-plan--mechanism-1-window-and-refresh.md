# Plan 039 — the fine-tune has nothing left to learn (mechanism 1)

**Status:** Proposed 2026-09-18, on `runs/az14x7v6` at gen07.

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

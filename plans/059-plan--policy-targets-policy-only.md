# Plan 059 — policy targets again, judged by policy-only drives

**Status:** Started 2026-10-08 morning (exp069, `scripts/exp069_targets_policy_only.sh`).

## 1. Why

The v9 loop learns its policy slowly: absorption dP(played) is +0.001 to +0.004 per generation and
dtop1 is about 0, while the value side moves every generation (dvalMSE −0.005, the value bench).
The loop trains on cq τ=100. That verdict is from before the mean backup (plan 055, `ce4eda1`):
- exp059 (PUCT, 2026-10-03): no target beat τ=100; Gumbel σ was below it;
- plan 054 / exp063 (Gumbel, 2026-10-05): τ=50 at lr 5e-5 led the absorption probe but drew 0.510
  in drives vs τ=100's 0.531; Gumbel σ had the biggest log P(played) gain (+0.025) but cost 0.03 of
  mean P(played).
Both ran while the search did not beat its own policy (E5: policy-only 0.517 vs Gumbel@1000) and
Q was biased upward by the max backup. A soft target was then the safe copy of a search that had
little to teach. Now the search beats its policy (0.582 at 1000 descents) and the per-generation net
check finds overrides that pay, so a sharper target may hand more of it to the net.

**The user (2026-10-08): no anchor benchmark for this; the quick numbers, and policy-only matches as
the fast play signal.**

## 2. The targets

- **cq τ** (the loop's): `softmax(ln prior + Q/τ)`, Q in points (1 TD = 1000). At τ=100 a move
  100 points (0.1 TD) better gets e times the probability. One temperature for every root.
- **Gumbel σ** (Gumbel MuZero, Danihelka et al. 2022; the target that belongs to our root search):
  `softmax(ln prior + σ(q̂))`, q̂ the completed Q min-max normalised over this root's children,
  σ = (c_visit + max visits) · c_scale · q̂ (c_visit 50, c_scale 0.1). Its sharpness follows each
  root's own spread and the visits spent; `min_range` (Q points) floors the spread so a near-tie
  is not read as decisive.

## 3. exp069

One generation step replayed: parent gen06, the window the loop's gen07 trained on (gen05-07,
MC-labelled), the loop's recipe (lr 5e-5, `--freeze-bn`, 3 epochs, restore on combined val).

| arm | target |
|---|---|
| loop07 | cq τ=100 (the loop's own gen07) |
| cq100r | cq τ=100 re-run: the training noise of every column |
| cq50 | cq τ=50 |
| cq30 | cq τ=30 |
| gumbel | Gumbel σ, c_visit 50, c_scale 0.1, min_range 50 |

Per arm, against the parent:
- **absorption probe** on gen07's held-out shards, prepared once with cq τ=100 so the columns
  compare; P(played), log P(played) and top-1 = played do not depend on the target;
- **value bench** (paired);
- **policy-only paired drives** (`cfgs/policy_only.toml` on both sides, the prior's argmax; tract on
  the CPU, one forward per decision): 600 pairs per board on the contested 14x7 and 16x9 sets.
  Timing: 200 drives in 93 s on 4 threads. The first reading, gen06 vs gen05 policy-only on 16x9:
  0.497 ± 0.021 (100 pairs).

Policy-only drives see only the policy. The value bench covers the value head and the net check
(not run here) whether a sharper policy still serves the search.

## 4. Results

(pending)

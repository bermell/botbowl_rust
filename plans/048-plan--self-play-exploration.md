# Plan 048 — self-play exploration: root Dirichlet noise and move sampling (plan 032 #4)

**Status:** Code landed 2026-09-29 (`1a1b7d3`); A/B running (`scripts/exp048_explore_ab.sh`, out
`runs/exp048/`).

## Why now

`runs/loopmix16x9` plateaued at gen13 strength for gen14-21, and plan 047 ruled out the two
training-side levers left: neither 3.7× the data nor a from-scratch retrain on it helped
(0.405 / 0.288 vs gen21's 0.513). The fine-tune extracts everything this corpus has in its first
2500 steps. What is left is changing what the corpus contains, and the generator is pure argmax-Q:
given a state it plays the same move every time, and all diversity comes from the random starts
and the dice. The value head never sees what a non-greedy alternative was worth, and the policy
target never sees a root pushed off its own prior.

## What landed

See `botbowl-mcts/CLAUDE.md` § Self-play exploration. In short:

- **Root noise:** `p' = (1 − ε)·p + ε·S·Dir(α/n)` at the root of every **fresh** search tree,
  keyed to the root state so recombination stays pure. A reused tree keeps its clean root.
- **Clean targets:** the sample records the pre-noise priors (the cq target reads `ln prior`).
- **Move sampling:** each side's first K moves of a trajectory ∝ `visits^(1/T)`.
- Flags on `botbowl-ui dataset` and `botbowl-hub job generate`; `GenerateConfig.exploration`
  (hub protocol v7); per-trajectory counts in `meta.extra` (`explore_noised`, `_sampled`,
  `_deviated`) and the provenance label.

**Recipe:** ε = 0.25, α = 10 (per root α/n, KataGo style — plan 031 D4's root fan is bimodal,
median 6, p90 73), K = 2 per side, T = 1.

**Smoke test (gen21, 500 visits, 32 games per arm, tract):** noise reached 594 of 1528 decisions
(39%, the fresh roots; reuse 61%, the same as greedy's 62%); 106 moves sampled, 63 of them (59%)
differ from the best-Q move; descents per decision unchanged (351 vs 353).

## The A/B

One generation off the same champion, everything else fixed:

| arm | corpus | train window |
|---|---|---|
| greedy | `loopmix16x9/gen22` — gen21's own generation | gen20 + gen21 + gen22 |
| noisy | `exp048/gen22x` — gen21, same seeds and board draws, explore recipe | gen20 + gen21 + gen22x |

Both: the loop's gen22 train step exactly (warm from gen21, lr 2e-4, 3 epochs, cq τ 100, value
blend 0.5), and one **common** val set (shards 4,7 of both gen22 and gen22x), so both restores are
picked on the same positions. Eval: 200 games vs `anchor_mix16x9_gen13.onnx` on each of 14x7 and
16x9, 500 visits — 400 games per arm, SE ≈ 0.025 per arm, ≈ 0.035 on the difference.

The noisy corpus is a third of its arm's training window, as it would be in the loop's first
exploring generation. This measures one generation of a mechanism that should compound; a null
here is weaker evidence than a null over several generations.

## Decision (pre-committed, on noisy − greedy over the pooled 400 games per arm)

- **≥ +0.07 (2 SE):** adopt. Relaunch the loop from the noisy arm with exploration on.
- **≤ 0:** abandon #4 (plan 032's rule: noisy-corpus net ≤ greedy-corpus net).
- **In between:** relaunch the loop with exploration for three generations and judge on the
  gen13-anchored rolling-3: < 0.55 on both 14x7 and 16x9 → abandon.

Also read (not a decision input): the greedy arm against gen21's own 0.513 (same seeds, first 100
per board) — it is the loop's gen22, so it says whether the plateau continued.

## Results

(pending)

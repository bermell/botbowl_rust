# Plan 048 — self-play exploration: root Dirichlet noise and move sampling (plan 032 #4)

**Status:** A/B done 2026-09-29 22:28 — exploration helps relative to greedy (+0.068 ± 0.031), but
**both arms regressed against their parent gen21**, so it is not adopted yet (see Results). Code
landed at `1a1b7d3`; run `scripts/exp048_explore_ab.sh`, out `runs/exp048/`.

**Provenance note.** The exp048 binaries in `target/16x9` were built at 16:57:54 from the
uncommitted plan-048 tree for the smoke test and not rebuilt before launch. So the hub, `gen22x`'s
corpus and both arms' eval reports are stamped `634c605-dirty`, not `ab2b5b0`. The code is exactly
`1a1b7d3`'s: the last source edit was at 16:57:13, and `1a1b7d3..ab2b5b0` touches only this plan
and the script. `hub-allowed-commits.toml` was pointed at that hub commit to admit `ab2b5b0`
workers.

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

The corpus (4800 games each, same seeds and boards):

| | gen22 (greedy) | gen22x (explore) |
|---|---|---|
| decisions | 170,679 | 177,186 |
| median drive length | 25 | 26 |
| TD/drive | 0.833 | 0.816 |
| descents per decision | 360 | 360 |
| root noise took effect | — | 69,513 (39%) |
| sampled moves / not best-Q | — | 16,431 / 10,008 (61%) |

Training: greedy restored step 5000 (val 0.7189, val_policy 0.5956), noisy step 2500 (0.7181,
0.5944), on the common val set.

Games vs `anchor_mix16x9_gen13.onnx`, 200 per board, 500 visits, this box only:

| arm | 14x7 | 16x9 | pooled (400) |
|---|---|---|---|
| greedy | 0.407 (W66 D31 L103) | 0.388 (W53 D49 L98) | **0.398 ± 0.024** |
| noisy | 0.505 (W79 D44 L77) | 0.425 (W60 D50 L90) | **0.465 ± 0.025** |

- **noisy − greedy, paired on the same 400 games: +0.068 ± 0.031.** Just under the +0.07 adopt
  line, so this is the pre-registered "in between" branch.
- Against gen21 itself on the 200 seeds all three played (gen21 0.512): **greedy −0.133 ± 0.044,
  noisy −0.052 ± 0.046.** The greedy arm, which is the loop's own gen22 step, fell three SE below
  its parent. No loop generation since the re-anchor dropped like that (gen14-21 were
  0.445-0.545 per board), so the spread of a single fine-tune is larger than the curve suggested.

**Decision: deviate from the pre-registered branch.** The "in between" branch said to relaunch
the loop with exploration for three generations. That assumed the greedy arm would stand in for
a normal generation, and it did not: neither arm improved on gen21. Exploration made the
fine-tune hurt less, which is a real signal about the corpus, but it is not a better net.
Meanwhile the plan 049 audit found the training step itself is the likelier problem, and that
every corpus so far was searched under a wrong chance model (armour breaks as casualties, every
pass a fumble, the search running through half time; fixed in the commit after this one).
Exploration stays in the code and goes into the next loop, relaunched after those fixes and
plan 049's training-side tests, where it will be measured again.

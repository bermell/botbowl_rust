# Is the exploration constant why more search makes labels worse?

**Status:** Investigated and answered (2026-08-30). Verdict: **exploration at the shipped `c=10` is too low, but Q-normalisation is not the fix.** Raising `c` to 30 is the only change with any strength signal, and even that is marginal. A separate, larger finding fell out: the plan-023 kickoff fixes did **not** reduce the Home/Away bias.

## Question

Plan 025 measured MCTS getting *more confident and less reproducible* with more budget (run-to-run top-1 agreement 0.67 -> 0.55 from 500 to 16000 iterations; root-value precision flat across a 160x budget increase; averaging 2x500 beating 1x1000). The repo owner's read was that this indicates a defect rather than a curiosity, since one tree with 2N iterations has strictly more information than two N-trees.

Leading hypothesis: **exploration starvation from unnormalised Q.** `puct_value` is textbook AlphaZero PUCT, but `Q` is in raw leaf-score units (`score_delta*1000 + ball_tier*10 + carrier_tier`) while `c_puct ~ 1-4` assumes `Q` in `[-1,1]`, and the best-vs-2nd Q gap varies **85x** across states (p10 6, median 77, p90 510).

## What was built

`PuctMode::{Raw{c}, NormalisedQ{c, range_floor}}`, per-bot (following the `virtual_loss` precedent — env vars read at *search* time are process-global and could not express two configs in one process, which the head-to-head required). `NormalisedQ` maps sibling Q to `[0,1]` against the node's min/max and priors onto the simplex. Raw stays the default and is bit-identical, guarded by a `to_bits()` test and by the `seed_42_step_500` snapshot not moving.

Harness: `--puct-mode`/`--puct-c` on `botbowl-ui convergence` (stamped into every row) and per-side `--puct-c`/`--vs-puct-c` on `botbowl-ui eval`; `scripts/puct_sweep_summary.py`.

## Result 1 — the sweep (52 states x 7 repeats, budget 1000, heuristic)

| arm | agree | TV | peak | H |
|---|---|---|---|---|
| **raw c=10 (shipped)** | **0.60** | 0.2212 | 0.515 | 0.617 |
| raw c=30 | 0.65 | 0.2017 | 0.495 | 0.638 |
| raw c=100 | 0.62 | 0.1824 | 0.431 | 0.726 |
| raw c=1000 | 0.60 | **0.0731** | 0.213 | 0.929 |
| norm c=0.5 | 0.69 | 0.1769 | 0.450 | 0.709 |
| norm c=1 | 0.69 | 0.1682 | 0.444 | 0.710 |
| norm c=4 | **0.70** | 0.1722 | 0.442 | 0.717 |

Paired per-state tests (n=52): normalised family vs shipped `c=10` **+0.082, p=0.023**; normalised family vs the *best* raw arm (`c=30`) **+0.027, p=0.49**.

**The guard metrics earned their place immediately.** Raw `c=1000` has by far the lowest TV and would have looked like the winner on a reproducibility-only reading — but `peak` 0.213 and entropy 0.929 show it is near-uniform and deciding nothing. Never rank selection rules on TV alone.

Three advance predictions **failed**, and are recorded so they are not re-made:
1. Normalisation would beat a *tuned* constant — not established (p=0.49).
2. Normalisation would be more robust to `c` — it is not (spread 0.051 vs raw's 0.056).
3. The best raw `c` would vary with a state's Q gap — it does not (`c=30` wins in both strata; the large-gap stratum is n=6 and underpowered).

## Result 2 — strength, which is what actually decides it (100 paired games each)

| candidate vs raw c=10 | decided | share | z | p | TDs |
|---|---|---|---|---|---|
| raw c=30 | 51/86 | 0.593 | +1.73 | 0.084 | 222:182 |
| norm c=1 | 45/87 | 0.517 | +0.32 | 0.748 | **229:229** |

**Normalisation improved label reproducibility by 9 points and produced exactly zero strength gain** — a dead heat, with the TD count tied to the game. This is the reproducibility-is-not-strength trap the plan was designed around, and it is why the strength stage was non-negotiable: on the cheap metrics alone, `norm c=1` looked like the clear winner.

`raw c=30` is the only arm with any strength signal, and at p=0.084 it is suggestive, not established. The +40 TD margin points the same way.

## Verdict

- **Do not adopt `NormalisedQ`.** It is committed, opt-in, default-off, and costs nothing where it sits — but it does not buy strength, so it should not become the default on this evidence.
- **`c = 30` is worth a confirming run** (another 100-200 games) before changing the default. If it holds, it is a one-constant change. Adopting it requires re-running the `#[ignore]`d strength benchmarks, whose thresholds were tuned at `c=10`: `score_td_easy` (>=0.85), `get_the_ball_easy` (>=0.90), `get_the_ball_medium` (>=0.80), `score_td_medium` (>=0.70).
- **The non-convergence of plan 025 is still unexplained.** Exploration was a real contributor but not the mechanism; more search still makes labels sharper and less reproducible. The 2x500-averaging trick from plan 025 therefore remains worth shipping on its own merits — it was never contingent on this diagnosis.

## The successor experiment (stronger candidate than `c`)

While testing, a unit test asserted something **false**: that an explored child outranks an unexplored one. With the self-consistent `fpu == hi` that `backprop_scores` produces, an unexplored child ties on Q and wins on the bonus **for all `c > 0`, in both modes** — so the first-visit sweep is breadth-first by construction and `c` cannot change it. That is **FPU with no reduction**. Leela/KataGo subtract `c_fpu * sqrt(sum of visited priors)`, which is only expressible once Q is normalised — so `NormalisedQ`, though not a win by itself, is the enabler for the experiment that might actually matter. This is the recommended next step, ahead of further `c` tuning.

## Cross-references

- plan 025 — the non-convergence finding this set out to explain, and the averaging trick that stands regardless.
- plan 023 — the side-bias investigation, whose kickoff fixes these runs incidentally falsified as the cause (see its Result section).
- `runs/puct/` — sweep JSONL, per-arm logs, and the two head-to-head reports.

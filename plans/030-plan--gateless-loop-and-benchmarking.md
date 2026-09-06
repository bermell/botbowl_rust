# Drop the promotion gate; benchmark on a frozen anchor instead

**Status:** Agreed 2026-09-06, not yet implemented.

## Decision

Stop gating promotions. The net trained each generation becomes the generator
for the next, unconditionally — AlphaZero's arrangement rather than AlphaGo
Zero's. Replace the per-generation gate with a periodic benchmark against
**frozen** opponents, whose job is detecting plateaus and regressions over
several generations, not adjudicating single ones.

## Why

**The gate has never changed a decision.** Seven generations gated, five
rejected. The best rejected score is gen07's 0.490 — *below parity, not merely
below threshold*. A gate set at even money would have rejected all five too
(plan 028's operating-characteristic check). It is not holding back
improvements; there have been none to hold back.

**It costs 29% of the loop.** From B1's measured pace — 120 nn-vs-nn games at
1000 iterations in 297 min, so 2.48 min/game — the 100-game vs-champion rung is
~248 min of gen07's 407-minute eval phase:

| configuration | cycle | generations/day |
|---|---|---|
| today | 14.3 h | 1.0x |
| drop the vs rung | 10.2 h | **1.41x** |
| drop it, benchmark every 3rd generation | 8.4 h | **1.70x** |
| also drop the saturated `random` rung | 8.1 h | 1.76x |

**And it cannot see what we would want it to see.** At 100 games (SE 0.045) a
genuinely better net at 0.54 promotes 41% of the time, and — counter-intuitively
— *more* games makes that worse, 33% at n=400, because tightening the
distribution around a threshold that sits above the true effect pushes mass
away from it. Only the threshold moves that number, and lowering it changes no
historical decision.

## The risk, which is not hypothetical

Under a gateless loop a bad generation becomes the generator, and its data
becomes the next generation's training set. **We have already watched this
trajectory take that step.** gen05, warm-started from the rejected gen04,
scored 0.380; pooled with gen04 that era ran 0.410 over 200 games at z = −2.91 —
genuinely worse, not noise. With no gate, gen04 would have generated gen05's
corpus.

AlphaZero tolerated this because a single bad step was diluted by a corpus
orders of magnitude larger. At 4800 games/generation a bad step *is* the
corpus. So the benchmark below is load-bearing, not decorative, and it needs a
defined trigger rather than an eyeball.

## The benchmark

**Anchor on something frozen.** The vs-champion rung was never comparable
across generations, because the opponent moved every time it promoted. A frozen
reference gives a strength curve that actually means something over time.

- **`vs-anchor`** — a permanently frozen net, `bbnet_14x7_gen03.onnx` (the
  current champion, and the last net promoted on merit). Never changes, so
  successive generations are directly comparable. This is the plateau detector.
- **`scripted`** and **`mcts-heuristic`** — keep. Both still discriminate
  (gen01→gen07 moved 0.867→0.867 and 0.883→0.967) and neither can drift,
  being deterministic opponents.
- **`random`** — drop. It has read 1.000 in every generation since gen01. Zero
  information for 30 games of compute.

**Cadence:** every 3rd generation. Single-generation noise is what we are
getting out of the business of interpreting; a 3-generation spacing costs a
third of the compute and is the resolution at which the question ("are we still
improving?") is actually asked.

**Sample size:** 100 games on `vs-anchor` (SE 0.045), 30 on each fixed rung.
Not for a pass/fail call — for a trend line.

**The trigger, stated in advance so it is not rationalised later.** Alert and
pause if either:
- `vs-anchor` falls below **0.40** on any benchmark — that is ~2 SE below
  parity, the level gen05 actually reached, and a clear regression rather than
  noise; or
- `vs-anchor` fails to exceed **0.55** across **four consecutive benchmarks**
  (i.e. ~12 generations) — a plateau worth interrupting for.

Note the asymmetry: 0.40 is a fast stop on damage, 0.55-over-12-generations is
a slow stop on stagnation. Neither adjudicates a single generation.

## What else changes

- `champion.txt` becomes simply the latest trained net; `WARM_FROM` becomes
  moot, since the trajectory and the generator are the same thing again.
- Keep every net on disk. Without a gate, the ability to roll back by hand to a
  known-good checkpoint *is* the safety mechanism, and `bbnet_14x7_gen03.onnx`
  must be preserved as the anchor regardless.
- The gate machinery (`eval_summary.py --gate`, the verdict files) should stay
  in the tree, unused by the loop. It is the right tool for a deliberate A/B and
  plan 029 needs it.

## What this does not settle

Dropping the gate buys throughput; it does not explain the plateau. The
open experimental programme stands and is now cheaper to run:

- **plan 029** — does more data make better nets? Unblocked: streaming
  `prepare` (`f146409`) took peak RSS from 7.58 GiB to 14.4 MiB at 350k
  samples, and `--seed` (`73fc2df`) makes arms differ only in what is under test.
- **plan 028 C1/C2** — 8000 and 16000 iterations vs 1000. The convergence curve
  says top-1 agreement climbs 0.69 → 0.91 between them and no strength arm has
  ever run up there.
- **Window width** — the same lever as games-per-generation from the trainer's
  point of view, but very different in generation cost. Plan 029 measures it.
- **Convergence probes (TV)** — cheap (~2h), and now the natural way to sanity
  check any search change before spending games on it. Re-run whenever PUCT,
  the horizon, or the evaluator changes.

The throughput gained here is what pays for that programme.

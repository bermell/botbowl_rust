# Idea 052: weight training drives by how contested their start position is

**Status:** Idea, written 2026-10-02 from the plan 051 discussion, revised the same day after
review. Not started. **Prerequisite: plan 050 (WDL value head) lands first.** The WDL head's
start-state distribution is the natural contest measure (step 0 below), and a tanh scalar cannot
tell "certain no score" from "coin flip". It also comes **after** the tau and search-budget work
(plan 049's 2026-10-01 headline), because both change what a good training target is, and the
user's setup-training phase may run in between. When it starts, rename it to `052-plan--…` and
add results under each step.

## Why

Plan 051's screen showed how lopsided random-start positions are. Of 500 candidates per board,
with gen21 playing itself four times from each:

| board | attacker TD rate overall | attacker always scores | attacker never scores | contested (25-75%) |
|---|---|---|---|---|
| 14x7/4 | 0.657 | 50% | 18% | 164 (33%) |
| 16x9/6 | 0.607 | 40% | 19% | 206 (41%) |

So over half the corpus starts from positions whose outcome barely depends on how either side
plays. Those drives still teach something: the value head has to know what a sure win and a sure
loss look like. But they carry little information about *which moves matter*. Positions with an
uncertain outcome are where the value target varies most and where a choice decides the drive.
That makes them the most informative samples per unit of generation cost, which is hard-example
mining, or a curriculum the net sets for itself.

It also lines up training with plan 051's main metric. Drive evaluation scores bots on positions
contested for the parent, so training that leans on contested positions optimises what we
measure. That alignment is the point, and it is also a circularity: training and evaluation
would both see only what the net itself finds contested. Plan 051's guard 2 (an occasional
full-game P1 check) is therefore a **precondition** for adopting anything from this plan, not an
option.

## The design in one line

Give contested positions a **higher training weight**, with a floor that keeps every position in.
Weight, do not filter. Measure contestedness **from the net at generation time** (K = 1, no
extra cost, every position distinct) if step 0 shows the net can; fall back to **K playouts per
position** (empirical outcome spread) only if it cannot.

## Why not K playouts first (the review, 2026-10-02)

The original draft generated every position K = 4 times and used the outcome variance across the
K drives as the weight. Two problems:

- **4x fewer distinct positions at equal cost, and maximally correlated samples.** Plan 036's
  finding was that the value head overfits correlated positions. K drives from one position are
  the most correlated samples possible: identical first state, slow divergence. The expected
  result is that the K = 4 unweighted arm (W0) loses to today's corpus, and the weighted arm then
  has to overcome that headwind before it shows a net gain.
- **The signal may already be free.** The net's own start-state value is a contest estimate. If
  it tracks the empirical spread, there is no need to pay for K playouts in the training corpus;
  they stay as the ground truth that *validates* the net's estimate.

The K-playout variant is kept below as the fallback and the validation tool.

## Step 0 — does the net know what it finds contested? (zero cost, run first)

Plan 051's screen already holds, per candidate position and board, gen21's empirical attacker TD
rate over 4 self-play drives (`cfgs/positions/*.json`, `screen.kept` plus the unkept candidates
from the eval lines). Evaluate the screening net on each position's start state and compare:

- **Scalar head (today):** `1 − |v|` vs `4p(1 − p)` of the empirical rate. Report rank
  correlation and the AUC of `1 − |v|` for "in the 25–75% band".
- **WDL head (plan 050):** entropy of `(P self, P nobody, P opp)` at the start state, same two
  numbers. Also `P(nobody)` alone, since a drive-bounded "nobody scores" is a distinct contested
  outcome the scalar conflates with 0.

Also compute the same from the search root after the first move's full-budget search (root Q,
and the root WDL once 050 lands), since the generator has that for free too and it is less
miscalibrated than the raw net.

| outcome | consequence |
|---|---|
| rank corr ≥ 0.5 or AUC ≥ 0.75 for either source | build the K = 1 design (step 1a); K playouts are validation only |
| below that for both | build the K-playout fallback (step 1b) |

A `scripts/exp_contest_prior.py`-style one-off against the existing screen files. No generation.

## Step 1a — K = 1 contest weight from the net (primary design)

**Generation.** `dataset` / `job generate` stamp into each trajectory's `meta.extra` the
start-state value (scalar or WDL triple) and the first-decision root Q / root WDL. Nothing else
changes; the corpus is otherwise today's.

**Optionally, rejection sampling at the position draw.** Draw a position, score its start state
with the net, accept with probability `floor + (1 − floor) · contest`. Generation effort then goes
to contested positions instead of only the loss weight. This is the real curriculum version and
costs one net evaluation per rejected draw. Keep it a separate arm (R below), since it changes
the corpus distribution where loss weighting does not.

**`prepare`.** Per position (one drive at K = 1):

    contest(position) = entropy(WDL at start) / log 3      # or 1 − |v| with the scalar head
    w(position)       = floor + (1 − floor) · contest(position)

The floor (start at 0.25) keeps one-sided positions in the corpus, for two separate reasons:
- **The value head stays calibrated.** Train only on coin flips and values drift toward 0, which
  corrupts the search's leaf scores.
- **The net does not choose all of its own diet.** "Contested for the generator" is shaped by the
  generator, and an unweighted share keeps the corpus anchored to the plain random-start
  distribution.

**Which loss.** Start with the **value loss only**, which is where plan 036's overfit evidence
sits. A lopsided position still yields an informative policy target (how to convert, how to
stall), so the policy multiplier is its own arm (WP below), not part of the default.

**Per-row shape.** The multiplier is a property of the position, but the rows of a drive diverge
from it. Apply the full weight to the first rows and fade it toward 1 over the drive (start:
linear fade over the first 8 plies, same for every row after). Cheap, and it targets what is
actually redundant. `prepare` already writes a per-sample `weight.npy` (plan 036 W4,
`1/len(drive)`, value loss only). This is a second array so W4 stays separable.

## Step 1b — K playouts per position (fallback and validation)

Only if step 0 fails, or as the validation set for step 1a's estimate.

**Generation.** Today one corpus seed fixes both the position and the dice
(`random_start_trajectory`). Split them: game `g` plays position `base + g / K` with dice seeded
from `g`, via `botbowl_play::drives::position_state`, so the position draw matches plan 051's
sets exactly. `--playouts-per-position K` on `dataset` and `job generate`, default 1 (today's
corpus unchanged), K stamped into the trajectory meta.

**`prepare`.** Group drives by position seed, take the mover-signed drive outcome in {−1, 0, +1}
(the existing value label's sign), compute its sample variance across the K drives, normalise to
[0, 1], same floor formula as 1a. With the WDL head use the empirical outcome distribution's
entropy instead; with K = 4 the two mostly agree. K = 2 gives only "agreed or split"; K = 4
resolves five levels. Start at 4.

**Duplicate first states, corrected.** The K drives from one position share their first state
exactly. Plan 036 **rejected** exact dedup (W5) and the loop runs with it off, so all K first rows
survive, each with its own outcome label. In expectation that is the averaged target, so it is
acceptable, but say so rather than rely on a filter that is not on. Measure the row count anyway.

## Step 2 — keep training and eval positions apart

Plan 051's eval sets use seeds from 70 000 000 (14x7) and 71 000 000 (16x9). The loop's corpora
use `10_000_000 + gen·10⁶ + shard·10⁵`, which is disjoint until generation 60. Make it explicit
anyway: `prepare` takes the eval position sets and refuses any corpus position seed that appears
in one. Applies to both 1a and 1b, and to the rejection-sampled arm.

## Experiment (when it runs)

Same shape as plan 049's tau test: fine-tune from one parent, one recipe, change one thing.
Generation cost is equal across arms (4800 drives each). Arms for the primary design:

| arm | corpus | weighting |
|---|---|---|
| U | K = 1 (today) | none |
| W | K = 1 | contest weight from the net, value loss only, floor 0.25 |
| WP | K = 1 | as W, also on the policy loss |
| R | K = 1, rejection-sampled draw at floor 0.25 | none (the draw is the weighting) |

If step 0 sends it to the fallback, the arms are instead U, W0 (K = 4, no weight, isolates the
loss of distinct positions) and W (K = 4, empirical spread, floor 0.25).

Judge each arm against the parent on **drives screened by the parent** (plan 051's main metric,
SPRT at 0.5:0.55), and confirm a winner with **P1 on full games** before adopting it. Fine-tune
seed variance on this loop is about sd 0.021 (exp052), so one seed per arm resolves effects of
about 0.05 or more. Run a second seed on any arm within that of another.

**Adopt** W (or R) if it beats U on drives **and** is confirmed on games. In the fallback, also
require that W0 does not explain the gain. **Tune** the floor (0.1 / 0.25 / 0.5) only after that.

## Things to check before trusting it

- **Miscalibration.** A net-estimated contest weight is only as good as the value head. Step 0
  measures this once with gen21; re-check it on the parent of whatever run adopts the weighting,
  since a value head that drifts (plan 032's draw feedback loop) would silently reweight the
  corpus. The K-playout spread on a small sample is the cheap recurring check.
- **Circularity.** Train on what the generator finds contested, measure on what the parent finds
  contested. The P1 full-game check is the only thing outside the loop. Required, not optional.
- **Fewer distinct positions** (fallback only). The W0 arm isolates it.

## Related

- **Plan 050 (prerequisite):** the WDL head gives the start-state distribution that step 0 and
  step 1a measure contestedness from.
- Plan 051: the screen (step 0's data), the drive metric, and the guards (re-screen with the
  parent, the occasional full-game check).
- Plan 049 #1: the cq target is close to the generator's prior. Contested positions are where the
  search's Q actually differs from the prior, so this may also make the policy target more
  informative. Worth measuring the KL of target vs prior per weight bucket; it is also the
  argument for the WP arm.
- Plan 036 W3/W4/W5: the existing value-label blend and the per-drive value weight, which this
  extends rather than replaces; W5's rejection is why the duplicate-first-state note above reads
  as it does.
- Kickoff-start positions (plan 051's blind-spot note): another change to the training
  distribution. Keep it a separate A/B so the effects are not confounded.

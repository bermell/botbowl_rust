# Plan 050 — WDL value head: three-way drive outcome instead of a tanh scalar

**Status:** Written 2026-09-30. Not started. Step 1 is trainer-only and needs no Rust, no schema
bump and no generation; steps 2 and 3 are gated on its result. Results go under each step.

## Idea

Replace the single tanh value output with a three-way softmax over the drive outcome as the
mover sees it: **self scores / nobody scores / opponent scores**. This is the WDL head Leela
Chess Zero and KataGo moved to. The search consumes the expected value
`P(self) − P(opp)`, which lands in the same `[-1, 1]` slot the tanh output fills today, so the
search is unchanged until step 3 chooses to use the decomposition.

The motivating intuition was "two heads, P(I score) and P(they score), so a move that only
raises my scoring chance is learnt as such". That nuance is real, but it lives in the *utility*
the search applies to the two numbers, not in the labels: the current target already records
who scored next in the drive (`−1/0/+1`), so a second head gets **no extra supervision**. What
changes is the loss and what the head can express.

## Why it might help

- **Loss shape.** MSE on a tanh against a ternary label learns the conditional mean and
  saturates at the ends. Cross-entropy over three classes learns the distribution and has
  healthy gradients everywhere. This is the whole reason Lc0/KataGo adopted WDL, and it gave
  measurable strength.
- **Ambiguity of zero.** Scalar 0 conflates "certain no score" with "coin flip who scores".
  The categorical keeps them apart. Under a pure difference-maximising search the ordering is
  identical in expectation, so on its own this changes nothing at the root; it matters for the
  loss and for step 3.
- **A real search upside, later.** With `P(self)` and `P(opp)` separate the search can apply a
  game-context utility the scalar cannot express: up 1–0 with two turns left, denying the
  opponent is worth more than scoring again. Plan 017 explicitly punted on this ("2–1 grind").
  The net cannot learn it from single-drive trajectories, but a hand-set weighting in
  `score_leaf` can use the decomposition.
- **Diagnostics.** An explicit `P(nobody)` directly measures the draw feedback loop plan 032
  (#…, "value head learns draws are what happens") worries about, and plan 049's flat value
  MSE since gen13 gets a second, better-conditioned metric.

Why softmax-3 and not two independent sigmoids: in a drive-bounded label the events are
mutually exclusive, so two sigmoids let `P(self) + P(opp) > 1` and waste capacity learning the
constraint. Softmax gives the same two numbers plus `P(none)` for free. Flipping the mover just
swaps the two outer classes.

## Where things are today (for whoever implements this)

- **Head:** `train/src/bbnn/model.py` — conv1x1 → pool → fc(32→64) → ReLU → fc(64→1) → tanh,
  documented as "value: (N, 1) in [-1, 1] (mover-centric)". `forward_masked` mirrors it.
- **Loss:** `train/src/bbnn/train.py::compute_losses` — policy CE + value MSE, optionally
  per-drive weighted `(w·se).sum()/w.sum()` with `w = 1/len(drive)` (plan 036 W4). Total is
  `policy + value_weight · value`; production `VALUE_WEIGHT=0.25`, `PER_DRIVE_VALUE_WEIGHT=on`.
  `evaluate()` reports policy loss, value MSE and top-1 on the val loader.
- **Label:** `botbowl-data/src/lib.rs::Trajectory::backfill_outcome_value` writes a Home-centric
  `−1/0/+1` per sample ("who scored next in this drive; 0 if the drive/half ended scoreless").
  `botbowl-nn/src/targets.rs::value_target_blended` re-signs it into the mover's frame and blends
  it `λ·outcome + (1−λ)·root_Q`; `prepare --value-blend λ` (default 1.0 = pure outcome; the loop
  runs `VALUE_BLEND=0.5`, plan 036 W3). `value.npy` is `(N,)` f32.
- **Inference:** `botbowl-nn/src/eval.rs::to_home_i64` clamps the scalar to `[-1, 1]`, flips the
  sign when the mover is Away, multiplies by 1000. `botbowl-mcts/src/dynamics.rs::score_leaf`
  returns the exact `score_delta·1000` when the drive is over and otherwise the net's number;
  nothing mixes the NN value with the heuristic leaf score.
- **Split:** `scripts/train_loop.sh` holds out whole generation shards (train = 0–3,5,6;
  val = 4,7). That matters here: plan 036 showed the head memorises drive identity, so a
  position-level split would flatter both heads equally and show nothing.

## Step 1 — trainer-only A/B on an existing corpus (the couple-of-hours experiment)

**The common scalar.** The search never sees three probabilities; it sees `P(self) − P(opp)`.
So for every held-out position both heads produce one scalar prediction of the same quantity and
are scored on the same metric.

**The label.** Score against the **raw** drive outcome `−1/0/+1`, not the blended target. Half
of the blended target is the generator's own value head fed back through minimax, so "matches
the blend better" partly means "agrees with the old head". Prepare the val shards with
`--value-blend 1.0`. This is a *new* number, not the val value_mse the trainer prints today
(that one is against whatever blend `prepare` wrote).

**Arms.** All trained on the same prepared train shards, same epochs, same warm start (or all
from scratch — pick one and say which), 3 seeds each; use the latest loop generation's corpus
(gen21 or later, post virtual-loss fix):

| arm | head | train label | note |
|-----|------|-------------|------|
| A | tanh scalar, MSE | blend 0.5 | production as is |
| B | tanh scalar, MSE | blend 1.0 | isolates the blend from the head |
| C | WDL softmax-3, CE | blend 1.0, class = `round(outcome) + 1` | the candidate |

Per-drive weighting stays on in every arm (it applies to CE the same way: `(w·ce).sum()/w.sum()`).

**Pass.** C's implied-scalar MSE on the val set is **at or below B's, with the gap larger than
the seed spread**. The test favours the scalar head: B is trained directly on the metric being
scored while C is trained on CE and only implies the scalar. If C still wins, the reformulation
generalises better and step 2 is justified. If C ties: no value-quality win; proceed only for
`P(none)` and the step-3 utility. If C loses: stop, record it, close the plan.
A vs B separately tells whether the blend helps or hurts, which plan 049 §4 already doubts.

**Sanity checks, not gating.**
- Reliability diagram: bin held-out positions by predicted `P(none)`; compare to the empirical
  fraction whose drive ended scoreless. A well-shaped head tracks the diagonal.
- Argmax confusion matrix over the three classes. If C predicts "nobody" almost everywhere, the
  plan 032 draw bias is now visible directly.
- Per-board numbers on a mixed corpus (the trainer already prints per-`dims_*` val), since a
  size that lags is what plan 042 needs to see.

**Work.** In `model.py` a `value_head: "scalar" | "wdl"` switch (fc 64→3, softmax, mover-centric
class order `[self, none, opp]`); in `train.py` a CE branch in `compute_losses` honouring the
per-drive weight, and one extra metric in `evaluate()` computing the implied-scalar MSE for either
head against `value.npy`; a seed flag if there isn't one. Nine training runs. No Rust, no schema
bump, no generation.

**Results:** _(pending)_

## Step 2 — wire inference (only if step 1 passes)

- `model.py` ONNX export emits the three probabilities (or the implied scalar plus the three; the
  scalar keeps the Rust loader trivially compatible while `P(none)` reaches the debug drawer).
- `botbowl-nn/src/eval.rs`: read `P(self) − P(opp)` into `to_home_i64`; the mover flip becomes a
  class swap. Detect the head shape from the ONNX outputs rather than a schema bump —
  `NN_SCHEMA_VERSION` describes the *input* encoding and this doesn't change it. Record the head
  kind in the prepare/train manifest next to `value_target` / `value_blend`.
- `bbnn.migrate`: the champion's 64→1 layer cannot be mapped onto 64→3; re-initialise it. It is
  small and retrains fast, but the first generation's `P(none)` will be uncalibrated — note that
  in the run's `status.md`.
- One loop generation, eval rung against the champion under the usual paired ladder. Pass is the
  usual promotion gate; the value-head val metrics from step 1 should reproduce on the fresh corpus.

**Results:** _(pending)_

## Step 3 — context-dependent utility in `score_leaf` (its own experiment)

`U = w_self(score, clock) · P(self) − w_opp(score, clock) · P(opp)`, with `w` hand-set from the
game score and turns remaining (e.g. leading late → weight denial; trailing late → weight scoring).
Must stay a pure function of the state (recombination purity, see root CLAUDE.md), and the
`drive_over` exact path keeps returning `score_delta·1000` unchanged. A/B against the plain
difference on the same net via `cfgs/*.toml` presets (plan 043) so the utility is the only
variable. Belongs in plan 032's ranked queue once steps 1–2 are in.

**Results:** _(pending)_

## Out of scope

- Changing what the label *means* (game result instead of drive outcome, discounting, TD(λ)).
  Plan 017's per-drive label stays; this plan only changes how it is predicted.
- Auxiliary heads with genuinely new supervision (next-scorer *turn*, ball-carrier fate, …).
  Would need `Sample` fields that don't exist yet.

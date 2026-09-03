# What actually makes the search strong? Priors, budget, horizon

**Status:** Running overnight 2026-09-03/04. Results filled in below as arms complete.

Three search parameters have never been justified by a *strength* measurement:
which prior source the bot plays with, how many iterations it searches, and how
far ahead the horizon lets it look. Each is currently set by a decision taken
before the plan-023 bug fixes, and each is cheap to be wrong about in a way that
compounds — every generation of the training loop inherits all three.

## What is already known, and why it does not answer these questions

**Plan 025 (2026-08-30)** measured how the search *output* converges with budget
and found it does not: run-to-run top-1 agreement peaks at ~500 iterations and
then *falls* (0.67 → 0.55 by 16k) while peak visit share rises monotonically
(0.45 → 0.74). More search makes the policy label sharper and less reproducible.
Two caveats matter here:

- Its own status header marks the finding **provisional**: it ran on `gen01`
  under the pre-`e107f06` buggy search, and `gen01` has since been retired for
  learned side-miscalibration. The raw data was deleted 2026-09-01.
- It measured **label convergence, not playing strength**. "The visit
  distribution keeps moving" and "the bot plays better" are different claims,
  and the second is the one that decides `MCTS_ITERS`.

**Plan 026 (2026-08-30)** found exploration at the shipped `c=10` too low, `c=30`
marginally better, Q-normalisation not the fix. Unchanged here; these arms all
run at the shipped `raw(c=10)` so the comparison is against production.

**Plan 020** found the *value* head to be the bottleneck and learned priors not
— `nn-value` 0.83 vs full-`nn` 0.75 TDs/game, with the gen-0 value head "actively
steering the search away from scoring lines". That measurement is why the loop
plays `nn-value`. It was taken on **gen-0**, three promoted generations ago.

## The gap this plan fills

All three questions get the same instrument: **a head-to-head between two bots
that differ in exactly one parameter**, same weights on both sides, paired
Home/Away on a shared seed set so both arms face identical situations. Score is
points `(W + D/2)/N`, the same rule the promotion gate uses.

Self-play with one variable changed is the cleanest available strength signal.
It cannot tell us a bot is good in absolute terms — only which of two is better,
which is exactly what a parameter choice needs.

### E1 — priors: does the trained policy head beat the scripted one?

The loop plays `--evaluator nn-value`: NN leaf values, **scripted** priors
(`dynamics.rs:258`, and `:626` "NN priors replace scripted priors"). So the
policy head is trained every generation and never used to play. This is why
`val_policy` improves monotonically to epoch 9 with no strength consequence
while `val_value` overfits by epoch 0-2, and why value-only best-val restore is
the correct selection rule *today*.

If learned priors win, the loop has been discarding half of what it trains.

### E2 — budget: is 1000 iterations right?

`MCTS_ITERS=1000` has been the setting since plan 020 and is the single largest
lever on generation cost — wall time is close to linear in it. The repo owner's
framing is the right test: **if the same net with more iterations beats itself
with fewer, the budget should go up.** Plan 025 suggests the opposite (labels
degrade past ~500) but measured a different quantity on a retired net under a
buggy search.

Both directions are actionable: 500 winning would halve generation cost.

### E3 — horizon: does deeper lookahead unlock better play?

Since plan 014 the search stops once the agent's turn counter advances once —
one own-turn plus the opponent's reply — or on any score change or game over.
One turn-pair has never been compared against two.

Implemented this session as `HorizonAnchor::turn_depth` /
`MctsBot::with_horizon_turns` / `--horizon-turns` (commit 4b38838). Depth 1 is
bit-identical to the historical condition. A score stays terminal at every
depth, because `score_delta`'s `{-1,0,+1}` range is what `PureTd` and the NN
value bridge read.

**Expect this arm to be the most confounded.** At equal iterations a deeper
search spends more compute per iteration, so a win partly measures "more
compute" rather than "better shape". Equal-iteration is still the right first
question — *does depth help at all* — but a win needs a follow-up at equal wall
clock before it changes production.

## Method

- Same champion net on both sides of every arm; only the named parameter differs.
- Paired Home/Away, shared `--seed` base per arm, disjoint from corpus seeds.
- Points `(W + D/2)/N`. At 60 games SE ≈ 0.065, so a 60-game arm resolves a
  ~0.13 effect at 1 SE and nothing smaller. These are **directional reads**, not
  significance tests, and are reported as such.
- Arms run in priority order and each writes its own report, so a run that is
  cut short still yields every completed arm.
- Sidecar (`nn_server.py`) on for throughput; the box is otherwise idle.

## Results

*(filled in as arms complete)*

| arm | question | games | points | record | read |
|---|---|---|---|---|---|
| E1-200 | learned vs scripted priors, 200 iters | 24 | **0.604** | W12 D5 L7, TD 60:52 | learned priors ahead, ~1 SE — not significant, direction favourable |

## Open questions this does not settle

- Whether any of these interact. A deeper horizon may need a different budget;
  learned priors may only pay off at higher iteration counts. All arms here vary
  one parameter against production defaults.
- Whether strength at 200 iters predicts strength at 1000. E1's cheap screen
  assumes priors matter *more* with less search to correct them, so a null at
  200 is strong evidence against, but a win at 200 needs confirming at 1000.
- Absolute strength. Every arm is self-play; none says the bot is good.

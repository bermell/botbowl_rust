# Idea 052: weight training drives by how contested their start position is

**Status:** Idea, written 2026-10-02 from the plan 051 discussion. Not started. It comes **after**
the tau and search-budget work (plan 049's 2026-10-01 headline), because both change what a good
training target is, and the user's setup-training phase may run in between. When it starts,
rename it to `052-plan--…` and add results under each step.

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
measure. That alignment is the point, and it is also why plan 051's occasional full-game check
matters here.

## The design in one line

Generate every position **K times** instead of once, measure each position's outcome spread
across its K drives, and give contested positions a **higher training weight**, with a floor that
keeps every position in. Weight, do not filter, so the screen costs nothing extra: the K drives
*are* the training data.

## Pieces

**1. K playouts per position in generation.** Today one corpus seed fixes both the position and
the dice (`random_start_trajectory`). Split them:
- game `g` plays position `base + g / K`;
- with dice seeded from `g`;
- via `botbowl_play::drives::position_state`, so the position draw matches plan 051's sets
  exactly.

Add a `--playouts-per-position K` knob to `dataset` and `job generate`, default 1, which means
today's corpus unchanged. Stamp K into the trajectory meta.

**2. A contest weight per position, in `prepare`.** Group drives by position seed. Then take the
mover-signed drive outcome in {−1, 0, +1} (the existing value label's sign), compute its sample
variance across the K drives, and normalise it to [0, 1]. Then:

    w(position) = floor + (1 − floor) · spread(position)

The floor (start at 0.25) keeps one-sided positions in the corpus, for two separate reasons:
- **The value head stays calibrated.** Train only on coin flips and values drift toward 0, which
  corrupts the search's leaf scores.
- **The net does not choose all of its own diet.** "Contested for the generator" is shaped by the
  generator, and an unweighted share keeps the corpus anchored to the plain random-start
  distribution. An alternative to the floor is a fixed fraction of K = 1 shards beside the K > 1
  ones. Pick whichever the A/B favours.

`prepare` already writes a per-sample `weight.npy` (plan 036 W4, `1/len(drive)`, applied to the
value loss only). This adds a position multiplier that applies to **both** the policy and value
losses, written as its own array so W4 stays separable.

**3. Keep training and eval positions apart.** Plan 051's eval sets use seeds from 70 000 000
(14x7) and 71 000 000 (16x9). The loop's corpora use `10_000_000 + gen·10⁶ + shard·10⁵`, which is
disjoint until generation 60. Make it explicit anyway: `prepare` takes the eval position sets and
refuses any corpus position seed that appears in one.

## Things to check before trusting it

- **Duplicate first states.** The K drives from one position share their first state exactly, and
  `prepare`'s exact-duplicate filter keeps one. That is probably right: one sample with a target
  averaged over K searches would be better still. But measure how many rows it drops.
- **Fewer distinct positions.** At equal generation cost, K = 4 means a quarter of the positions.
  The A/B must separate "fewer positions" from "weighted positions", hence the W0 arm below.
- **Which spread.** Outcome variance counts defender scores as informative too. An alternative is
  the attacker scoring rate's `4p(1 − p)`, as in plan 051's screen. With K = 4 the two mostly
  agree. Start with variance.
- **Choosing K.** K = 2 gives only "agreed or split"; K = 4 resolves five levels. Start at 4.

## Experiment (when it runs)

Same shape as plan 049's tau test: fine-tune from one parent, one recipe, change one thing. The
generation cost is equal across arms (4800 drives each).

| arm | corpus | weighting |
|---|---|---|
| U | K = 1 (today) | none |
| W0 | K = 4 | none, which isolates the loss of distinct positions |
| W | K = 4 | contest weight, floor 0.25 |

Judge each arm against the parent on **drives screened by the parent** (plan 051's main metric,
SPRT at 0.5:0.55), and confirm a winner with **P1 on full games** before adopting it. Fine-tune
seed variance on this loop is about sd 0.021 (exp052), so one seed per arm resolves effects of
about 0.05 or more. Run a second seed on any arm within that of another.

**Adopt** W if it beats U on drives and is confirmed on games, and W0 does not explain the gain.
**Tune** the floor (0.1 / 0.25 / 0.5) only after that.

## Related

- Plan 051: the screen, the drive metric, and the guards (re-screen with the parent, an occasional
  full-game check).
- Plan 049 #1: the cq target is close to the generator's prior. Contested positions are where the
  search's Q actually differs from the prior, so this may also make the policy target more
  informative. Worth measuring the KL of target vs prior per weight bucket.
- Plan 036 W3/W4: the existing value-label blend and the per-drive value weight, which this
  extends rather than replaces.
- Kickoff-start positions (plan 051's blind-spot note): another change to the training
  distribution. Keep it a separate A/B so the effects are not confounded.

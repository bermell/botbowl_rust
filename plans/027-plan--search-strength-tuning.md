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
| E2b | 2000 vs 1000 iters | 60 | **0.508** | W24 D13 L23, TD 185:180 | dead even. Doubling the budget buys nothing. |
| E2d | 500 vs 250 iters | 60 | **0.517** | W26 D10 L24, TD 183:173 | dead even |
| E2a | 1000 vs 500 iters | 60 | **0.575** | W30 D9 L21, TD 209:182 | the only budget arm with a signal — z=+1.15, p≈0.25 |

### E2b — doubling the budget does nothing (2026-09-03, 120 min)

0.508 against a 0.500 null with SE 0.065, and the TD totals are as level as the
result (185:180). **Do not raise `MCTS_ITERS`.** Wall time is close to linear in
the budget, so 2000 would have doubled a 5-hour generate phase to buy an effect
we cannot distinguish from zero.

This corroborates plan 025 from an independent direction and largely retires its
"provisional" caveat. 025 found the *policy label* stops improving past ~500
iterations, measured as run-to-run convergence on a retired net under the
pre-`e107f06` search. This finds the *play* does not improve past 1000 either,
measured as strength, post-fix, on gen03. Two different instruments, two
different quantities, same conclusion: the search saturates well below the
shipped budget.

The open question flips direction. If 1000 is already past saturation, the
interesting number is not how much higher we can go but **how much lower we can
drop before strength degrades** — and every halving is a halving of generation
cost. E2a (1000 v 500) now carries the most practical value in the matrix, and a
500 v 250 arm is worth more than the planned 4000 v 2000, which after this
result is very unlikely to show anything.

### E2d — halving again also does nothing (2026-09-03, 58 min)

500 v 250: **0.517** (W26 D10 L24, TD 183:173). So 2000 ≈ 1000 and 500 ≈ 250.
Note these are adjacent halvings, and adjacent ties do not compose — small
effects can accumulate across a 4x span — which is why `e2f-1000v250` runs the
span directly.

### E2a — 1000 beats 500, and the three budget arms together suggest a plateau

1000 v 500: **0.575** (W30 D9 L21, TD 209:182), the only budget arm with any
signal — though at SE 0.065 that is z=+1.15, p≈0.25, so it is suggestive, not
established. The three arms read together:

| arm | result | reading |
|---|---|---|
| 2000 v 1000 | 0.508 | nothing above 1000 |
| **1000 v 500** | **0.575** | the one place search seems to pay |
| 500 v 250 | 0.517 | nothing below 500 |

Taken at face value that is a plateau with an edge near 1000: enough search to
matter, past the point where more helps. **`MCTS_ITERS=1000` looks close to
right** — arrived at without measurement, but not wrong.

This retracts the "halve it for free" reading I drew from E2b and E2d. Two
adjacent ties do not license dropping the budget when the halving *between*
them is the one that costs something. `e2f-1000v250` tests the 4x span directly
and is the arm that settles it: if the 1000-v-500 edge is real and cumulative it
should show up larger there; if e2f also lands near 0.50, then all four arms are
noise around a flat response and the honest conclusion is that budget does not
matter anywhere in 250-2000.

Caveat on all four: 60 games resolves ~0.13 at one SE. A true 0.55 effect is
below this instrument's resolution, and 0.55 per generation is not nothing.

---

## The bigger finding: a Home/Away side bias that dwarfs every parameter

E2d's side split is the giveaway: candidate **10-17 as Home, 16-7 as Away**.
The side is worth more than the parameter under test. Pooling every eval report
on disk (`scripts/side_bias_summary.py`):

| subset | n decided | Away share | 95% CI | z |
|---|---|---|---|---|
| all rows | 562 | 55.7% | [51.6, 59.7] | +2.70 |
| informative rows only¹ | 445 | 56.9% | [52.2, 61.4] | +2.89 |
| mirror-like arms | 116 | **62.1%** | [53.0, 70.4] | +2.60 |

¹ A shutout carries no side signal — beating `random` 30-0 gives a 15/15 split
by construction, not by balance — so those rows are excluded.

**Every parameter arm so far has come back ~0.50. The side you play is worth
~0.07-0.12.** For "what makes a strong bot", that ordering matters more than
any single arm: we have been tuning the small knobs.

### Why it stayed hidden, and what it is *not*

A candidate's win rate cannot see this. Seats alternate Home/Away (`eval.rs:329`,
`g % 2`), so a side advantage cancels out of the headline number — which is
exactly why every arm reads ~0.50 while the sides underneath are lopsided.

It is **not** plan 021's "0.40 mirror anomaly". That is a *seat* statistic, and
re-reading its abandoned run at 52 games (W19 D4 L29) it is 19/48 decided:
p ≈ 0.15, not significant. It also fails to reproduce here — the candidate seat
scored 0.508, 0.517 and 0.604 across three near-mirrors. Two different effects
have been sharing one name.

### Mechanism candidates

- **Last turn.** `game_procs.rs:90`: the *receiving* team takes the first turn
  of each round, so the *kicking* team takes the **last turn of each half**.
  In Blood Bowl the last word is worth a lot. This is symmetric across
  Home/Away only if the coin is fair.
- **The toss.** `scripted.rs:40` pins the call to Heads and `:52` always
  Receives, so the toss *loser* kicks. In eval the coin is genuinely rolled
  (`play_game` sets `DiceMode::RollDice`), so this should be 50/50 — but it has
  never been checked. Note `GameStateBuilder` *fixes* the coin to Heads for
  non-CoinToss states (`gamestate.rs:238`, comment `//Away`), which is the
  random-start path **the training corpus is generated from**.
- **A genuine board/setup asymmetry** on 14x7 that survives conditioning on the
  toss.

A discriminating detail is already visible: **Away scores only ~53% of TDs but
wins ~57-62% of games.** The edge is not in scoring more, it is in converting
scores into wins — which points at *when* scores land, i.e. turn order, rather
than at raw play strength.

### How it gets resolved

Stream 3 (`scripts/exp_side_bias.sh`) runs true mirrors — identical evaluator,
identical iterations, identical horizon on both sides — with `--per-game-out`,
which logs `kicking_first_half` per game. `scripts/kickoff_effect.py` then
separates the three: is the coin fair, does the kicking team win more, and is
there a residual side effect inside both toss branches. Heuristic first, since
it needs no NN and no GPU: if the bias shows there, it is the engine or the
board, not the net.

### The corpus is NOT biased — tested and refuted

The obvious worry is that self-play with a side bias teaches the value head a
side prior rather than a position evaluation, the failure plan 023's postscript
describes when it retires `gen01` for "learned side-miscalibration". Extra
reason to suspect it: `--mode random-start` does **not** roll the coin. The
builder fixes it (`gamestate.rs:238-242`), so Away kicks in *every* corpus game
— and the kicking team owns the last turn.

Checked against the `z_home` line every shard log writes per game:

| gen | n | home win | away win | draw | Away share of decided |
|---|---|---|---|---|---|
| gen00 | 4800 | 1905 | 1965 | 930 | 50.8% |
| gen01 | 4800 | 1893 | 1892 | 1015 | 50.0% |
| gen02 | 4800 | 1946 | 1895 | 959 | 49.3% |
| gen03 | 4800 | 1936 | 1878 | 986 | 49.2% |
| gen04 | 4800 | 1901 | 1937 | 962 | 50.5% |
| **pooled** | **24000** | 9581 | 9567 | 4852 | **49.96%** (z = −0.1) |

Flat. 19,148 decided games and no detectable side effect, despite Away kicking
in all of them. The reason is structural: random-start games are drive-bounded
and truncated, so they never play the half — and it is the half, with its fixed
8+8 turns and a last turn that belongs to the kicker, where the advantage
accrues. **The bias exists only in full games from kickoff.**

That is a real narrowing. It rules out board or setup geometry (which would show
in both regimes) and points hard at turn order. It also means the corpus is
clean, and the "biased data poisons the value head" story is refuted rather than
merely unproven. Draw rate in the corpus is 20.2%, for reference.

### What the bias actually costs us: variance, not a rigged gate

Because seats alternate, the side effect cancels out of a candidate's expected
score — the promotion gate is **unbiased**. What it does instead is inflate
variance: a large chunk of each game's outcome is decided by which side you drew
rather than by the parameter under test.

And the harness already has the fix half-built. `eval.rs:329-330` pairs games —
`g % 2` swaps seats while `g / 2` shares the seed, so games `2i` and `2i+1` are
the same situation played from both sides. That is textbook common random
numbers. But the report then scores every game **independently**: `LadderRow`
holds pooled W/D/L, and nothing differences the pair. The variance reduction the
pairing was designed to deliver is being discarded at the last step.

Scoring the per-pair difference instead cancels the side term exactly, at zero
extra compute (`scripts/paired_summary.py`).

**How much that is worth is an open question, and less than I first claimed.**
The saving depends on what fraction of a game's variance is *shared within a
pair* (the situation and the coin, which the shared seed fixes) versus
independent per-game noise (dice, per-seat bot RNG). Blood Bowl is dice-heavy,
so the independent term is probably large. A synthetic check — 62% side effect,
zero true skill difference, side realised independently per game — gives a
paired SE only **1.07x** tighter, worth about 15% more games. Real data should
do better, because a shared seed genuinely fixes the coin and the opening
situation for both games of the pair, but "better than 1.07x" is not "material"
until measured.

So the honest claim is narrower: pairing is free, strictly correct, and removes
a term that is currently pure noise; whether it rescues an 0.05 effect at 60
games is unknown, and `paired_summary.py` prints both estimators side by side so
the real ratio can be read off stream 3's data rather than assumed. Retracting
the stronger version: this is probably *not* the reason all four budget arms read
flat.

## Open questions this does not settle

- Whether any of these interact. A deeper horizon may need a different budget;
  learned priors may only pay off at higher iteration counts. All arms here vary
  one parameter against production defaults.
- Whether strength at 200 iters predicts strength at 1000. E1's cheap screen
  assumes priors matter *more* with less search to correct them, so a null at
  200 is strong evidence against, but a win at 200 needs confirming at 1000.
- Absolute strength. Every arm is self-play; none says the bot is good.

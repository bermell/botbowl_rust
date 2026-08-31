# Home/Away side bias: what the 100-game mirror match found

**Status:** Open, but no longer blind. Closes plan 021 open issue 5: the mirror
anomaly is **real** (~650 games across five independent runs, 0.667-0.727 Home
share at 1000 iterations). Five candidate *causes* have been fixed and each
measured not to move it: B1 kickoff aim, B1b receiving-team argument, B2
inverted post-touchdown kickoff (`4ccbce2`), H-c non-mirror-invariant
throw-in/bounce models (`316dbca`), and the pathfinder's route tie-break
(`21c2e09`). Two side biases *were* localised and fixed: `ScriptedBot`'s
touchback tie-break (`dcc578c`, an Away bias of 0.113) and the instrument's
inability to see side-relative results (`ae43fa2`, `80175cf`).

**Two things changed on 2026-08-31** (see the two "Result (2026-08-31)"
sections):

1. **Action ordering is a large lever but not the cause.** Forcing the
   search's exact-tie tie-break to "prefer the lowest-x action" takes the
   mirror-match Home share from **0.703 to 0.503** (z = −3.7); forcing the
   *mirror-covariant* version of the same rule leaves it at 0.633 (z = −0.45
   against a matched control). So the hypothesis is refuted as a cause and the
   low-x arm is a compensation, not a fix — do not ship it. What it leaves is a
   sharp contradiction: with a covariant tie-break and a pipeline that is now
   property-tested mirror-exact, the mirror match should read 0.500.
2. **The bias is measurable without games.** A paired mirror probe on the
   search's root value reads −65 ± 5 (t = −12) at 1000 iterations, 0 at small
   budgets, and **exactly 0 under the y-reflection control**. That turns a
   2-hour 200-game experiment into a 10-minute one, and it has already
   exonerated the entire state-transition and evaluation pipeline by property
   test: `leaf_score`, priors, pruning, the legal-action set, the engine's
   transitions, movement routes, the chance model, and `apply_action` at depth.

The live lead is now the **selection / aggregation layer** (PUCT descent,
virtual loss, minimax backprop, the `recon_mcts` DAG). The next step is to
property-test it the way the transition layer now is.

Everything before the 2026-08-31 sections was investigated read-only while the
plan-022 weekend loop was running, from code reading and data already on disk.

## The measurement

Pre-flight mirror match, `runs/loop14x7/mirror.json` — 100 games, identical heuristic bot on both sides, 1000 iters, seed 424242, 14x7 tier:

- Report card line: `0.44 (W44 D15 L41 TD 209:211, home 29-13 away 15-28)`
- **Re-expressed side-centrically** (recovered per-game from `mirror.log`, validated against the JSON): the physical Home team went **W57 D15 L28 = 0.645 of points**, i.e. **67.1% of decided games**.

Two traps in reading that report line, both of which cost us a wrong first reading:

1. **The aggregate `win_rate 0.44` cannot see this bias.** Sides alternate, so the aggregate is mechanically forced toward 0.50. Same for `TD 209:211`: `score_for` (`botbowl-ui/src/eval.rs:162-167`) is *candidate-relative and pooled over both sides*, so in a mirror it is near-balanced **by construction**. The Home-vs-Away TD split is **not recoverable** from `mirror.json` — an instrument gap worth closing.
2. **`home 29-13` / `away 15-28` are the same 100 games counted twice**, from the candidate's perspective. They are not two independent measurements. Home-side wins = 29 + 28 = 57.

**Significance.** The 100 games are 50 seed-pairs (`eval.rs:184-186`: side alternates, `seed = args.seed + g/2`). Clustering on the 50 independent pairs: mean +0.58 games/pair, sd 1.357, **t = 3.02, p ≈ 0.003**. Naive per-game z = 3.15 slightly overstates it. Stable across the run (first 50 games 30-6-14, last 50 27-9-14), so not drift or contamination.

## Verified bugs (found by code reading, both confirmed by hand)

### B1 — Kickoff aim is off by one, and is the only x-asymmetric expression in the engine

`botbowl-engine/src/core/gamestate.rs:843-850`:

```rust
TeamType::Home => Position::new((w / 4, mid_y)),
TeamType::Away => Position::new((w * 3 / 4, mid_y)),
```

At engine width 16 (14x7 tier) the receiving halves are Away `[1,7]` (centre 4) and Home `[8,14]` (centre 11). Home's kick aims at `w/4` = 4 — correctly centred. Away's kick aims at `3w/4` = **12**, but the mirror of 4 under `x → 15-x` is **11**. One column too deep. The error is exactly +1 at every width ≡ 0 (mod 4), **including the full 28-wide pitch** (21 vs the correct 20).

A zero-CPU Monte Carlo (deviate D6×D8 from the aim, then one bounce, against the real setup formations) showed that patching the aim to 11 reproduces the Away-receiving row *exactly* — clean proof that 12 is the bug:

| | P(touchback) | P(catch on landing) | mean dist to receiver | mean dist to nearest kicker |
|---|---|---|---|---|
| Home receives (aim 12, buggy) | 0.602 | 0.039 | 2.39 | 4.51 |
| Home receives (aim 11, fixed) | 0.596 | 0.060 | 2.30 | 4.27 |
| Away receives (aim 4) | 0.596 | 0.060 | 2.30 | 4.27 |

**Magnitude is the weak link, not existence.** ~60% of kickoffs are touchbacks (perfectly symmetric — the receiver picks from its mirrored formation), so the bug only touches ~40% of kickoffs. And **the sign is genuinely ambiguous**: Home's reception lands 1 square further from its own catcher (worse) but also ~1 further from the nearest opponent (better — less contested pickup, which on a 4-player board is often what decides a drive).

**Gotcha for whoever fixes it:** `gamestate.rs:1722-1734` (`kickoff_position`) pins the buggy value by *restating the same formula*. It is a tautological test, not an independent check, and it will "fail" on a correct fix. Replace it with a mirror-invariance assertion (`aim(Home)` and `aim(Away)` reflect onto each other under `x → width-1-x`), which is the property actually wanted and which would have caught this.

**Related, same function:** `ball_procs.rs:301` calls `get_best_kickoff_aim_for(team)` with the *receiving* team, but the function's parameter means the *kicking* team — so the no-receivers-on-pitch fallback drops the ball in the wrong half. Rare path, carries the same off-by-one.

### B2 — Post-touchdown kickoff is inverted: the scorer receives again

`botbowl-engine/src/core/procedures/ball_procs.rs:325`:

```rust
game_state.info.kickoff_by_team = Some(other_team(game_state.get_player_unsafe(self.id).stats.team));
```

`kickoff_by_team` is consumed at `game_procs.rs:78` and handed to `do_kickoff(team)`, which sets `info.kicking_this_drive = kicking_team` (`game_procs.rs:45`) — so the value is the **kicking** team, and it is set to `other_team(scorer)`. The conceding team kicks, i.e. **the team that just scored receives the next kickoff**. Real Blood Bowl is the opposite.

This is **side-symmetric on its own**, so it cannot *cause* a Home bias. It matters twice over anyway:

- It makes scoring self-reinforcing, which is the only mechanism that plausibly turns B1's small per-kickoff edge into a 14.5-point win-rate swing.
- It corrupts **evaluation**, not the drive-bounded training corpus. **Correction (verified 2026-08-30):** an earlier draft of this plan claimed B2 was a training-data quality bug affecting "every corpus to date". That is **false for the drive-bounded corpora** (plan 021 onward, including every generation of the 022 loop). The stop predicate at `botbowl-ui/src/dataset.rs:218` ends the episode *the instant the score changes*:
  ```rust
  s.home.score != start_home_score || s.away.score != start_away_score || s.info.half != start_half
  ```
  so the post-touchdown kickoff is never played inside a training episode. Combined with random-start building at `Turn{turn:1}` and never running a coin toss (zero `"Kickoff"` events in any shard), **neither B1 nor B2 touches the drive-bounded corpus.** The claim does hold for plan 020's full-game corpora.
- What is compromised instead is the **control signal**: every ladder rung, report card and promotion gate is a full game from `CoinToss`, so all of them are measured under both bugs. Both nets face the same distorted rules, so a gate comparison is *fair* — but B2 amplifies small edges into large win-rate gaps, which systematically over-rewards whoever scores first and biases selection toward that style. Treat pre-fix strength numbers as measured in a slightly different game, and re-run any gate you intend to rely on.
- The pair correlation in the mirror data implies the receiving side is worth **δ ≈ 0.37 in points** (receiver takes ~68%), a very large possession advantage consistent with the snowball.

## Ruled out, with evidence (negative results worth keeping)

- **Harness side assignment and pairing are correct.** `eval.rs:184-185` alternates Home on even `g`, Away on odd, with `seed + g/2` shared by the pair; `eval.rs:193-204` attributes the split correctly. `unfinished: 0`, no truncation.
- **Both bots are identically configured.** `--opponent-iters` does default to `--mcts-iters` (`eval.rs:290`), both sides come from the same `make_bot` (`eval.rs:89-97, 305`) with the same workers, evaluator, priors, pruning, dice mode. `scripts/train_loop.sh` passes neither override.
- **The heuristic leaf evaluator is Home-centric but provably antisymmetric** (`botbowl-mcts/src/score.rs:17-68`). Every term flips exactly under `x → 15-x` + team swap, including `carrier_distance_value`'s hard-coded `26` offset (which biases toward "someone holds the ball", not toward a side).
- **MCTS perspective/sign handling is symmetric everywhere**: `dynamics.rs:530-540` (`home_perspective`, FPU flip), `:770-775` (`q_perspective`, virtual loss applied after the flip), `:639` (`want_max`), `:1299-1303` (`q_sign`).
- **Priors and pruning are mover-relative only** (`priors.rs:111` uses `get_endzone_x(agent_team)`; `pruning.rs` has no side logic outside tests).
- **Tie-break direction bias — investigated and killed.** `gamestate.rs:1376` sorts actions ascending by `(PosAT, x, y)` and selection uses `max_by`/`max_by_key` (which return the *last* maximum) — that would have been a systematic "+x drift", devastating in a mirror. But children live in a `HashMap` (`recon_mcts/src/tree.rs:471`) with std's per-process `RandomState`, so tie order is pseudorandom, not x-ordered. (This also explains why paired games diverge at all: only 24/50 pairs produced the same physical winner.)
- **Dice, direction tables, geometry, rosters, formations are exact mirrors.** `Coin` = `gen_range(1..=2)`; `ALL_DIRECTIONS` has dx=+1 and dx=−1 three times each; `setup_line` (`kickoff_procs.rs:233-292`) places Home at `los+dx` and Away at `los-dx` with the same dy — an exact reflection about x=7.5.
- **In-drive play is symmetric at n=4800 — the strongest negative result.** `runs/loop14x7/gen00/shard*.jsonl`, identical bot both sides: home TDs **6386** vs away **6403** (share 0.4993 ± 0.0044). Those episodes contain **zero kickoffs** (`grep -c '"Kickoff"'` = 0) because random-start builds at `BuilderState::Turn{turn:1}`. Everything inside a drive is therefore exonerated at ±0.9% (2σ), which is what forces the cause into the kickoff / coin-toss / half machinery.

## Refuted hypothesis (recorded so it isn't re-raised)

**"`kickoff_by_team` survives the half boundary, giving a duplicate kickoff to open half 2."** Checked and **false**. In `Half::step` (`game_procs.rs`), `self.kickoff = info.kickoff_by_team.take()` runs in the `else` branch *before* the `home_turn == 8 && away_turn == 8` early return, so the flag is cleared and the pending kickoff is discarded with the proc. `Half::new(2)`'s `!started` branch then sets its own single kickoff. No duplicate.

## Result (2026-08-30): the fixes landed, and the bias did NOT move

B1, B1b and B2 are fixed (commit `4ccbce2`, with property tests replacing the two
tests that had encoded the same mistakes). Two 100-game heuristic-vs-heuristic
matches run *after* the fixes measure the side split as follows — Home-side wins
are `wins_as_home + losses_as_away`:

| run | Home | Away | Home share | z | p |
|---|---|---|---|---|---|
| pre-fix mirror (this plan) | 57 | 28 | 0.671 | +3.15 | 0.0017 |
| post-fix, norm c=1 vs raw c=10 | 58 | 29 | **0.667** | +3.11 | 0.0019 |
| post-fix, raw c=30 vs raw c=10 | 61 | 25 | **0.709** | +3.88 | 0.0001 |

**The kickoff fixes did not reduce the Home advantage at all** (0.671 -> 0.667).
The `norm c=1` row is the cleaner of the two comparisons: that configuration is a
dead heat in strength against the baseline (0.517, p=0.75), so it functions as a
mirror match for side purposes.

Consequences:

- **H1 (the kickoff-aim off-by-one) is refuted as the cause.** It was a real bug —
  verified by reading and by the Monte-Carlo table above — but it is not what
  produces the Home advantage. The measured effect it could produce was always
  small and of ambiguous sign; that caution was warranted.
- **H-b (a fluke, or an effect much smaller than measured) is now dead.** Three
  independent runs totalling ~270 games all land at 0.67-0.71 with p <= 0.002.
  The effect is real and roughly 2:1.
- B2 remains a genuine rules bug worth having fixed (the scorer now kicks off, so
  scoring is no longer self-reinforcing) — it just was not the side-bias
  mechanism, being side-symmetric, exactly as this plan predicted.

**The cheapest next test is now deferred item 3, and it is cheap: a RandomBot
vs RandomBot full-game mirror.** With no search involved it isolates rules and
setup asymmetry from bot behaviour entirely, and random bots are fast. If the
bias survives there, it is purely engine/setup and every search-side hypothesis
below can be dropped.

## Result (2026-08-30, part 2): the ladder without search — rung 1 and rung 2

Instrument: `botbowl-ui eval` gained `--candidate-bot mcts|scripted|random`,
`--rungs a,b,c` and `--per-game-out PATH` (commit `ae43fa2`), so
`run_ladder_rung` — still the only bot-vs-bot win-rate instrument — can seat a
non-search bot and log **side-relative** results per game. That closes deferred
item 5. All runs below are the 14x7 tier (`BOARD_SIZE_W=14 BOARD_SIZE_H=7
BOARD_PLAYERS=4`), full games from `CoinToss`.

**Read the counting note first.** `ScriptedBot` is deterministic, so the two
games of a seed-pair are the *same physical game* played twice with the
candidate label swapped (verified: 800/800 pairs identical). Scripted numbers
below are therefore de-duplicated to one game per seed. `RandomBot` is not
deterministic, so its games all count.

| rung | n (decided) | Home | Away | Home share | z | p |
|---|---|---|---|---|---|---|
| **1.** scripted mirror, as shipped | 344 | 39 | 305 | **0.113** | −14.3 | <1e-40 |
| 1b. scripted mirror, tie-break flipped | 341 | 315 | 26 | **0.924** | +15.6 | <1e-40 |
| 1c. scripted mirror, **fixed** | 613 | 326 | 287 | 0.532 | +1.58 | 0.11 |
| **2.** random mirror | 3605 | 1750 | 1855 | **0.485** | −1.75 | 0.08 |
| 3. MCTS heuristic mirror, **200** iters | 251 | 133 | 118 | 0.530 | +0.95 | 0.34 |
| 3b. MCTS heuristic mirror, **1000** iters, H-c fixed | 150 | 100 | 50 | **0.667** | +4.08 | 4e-5 |
| (for reference) MCTS heuristic, 1000 iters, pre-fix ×3 | 258 | 176 | 82 | 0.682 | +5.9 | <1e-8 |

### Rung 1 — the bias survives without search, but it is a *different* bias

`ScriptedBot` vs `ScriptedBot` gives the **Away** team 305 of 344 decided games
(0.113 Home share, z = −14.3), and side TDs 787:1439. Opposite sign to the MCTS
Home bias and five times the magnitude — so this is not the same phenomenon,
and rung 1 does **not** transfer the MCTS finding to the engine.

It localises to one line. `AvailableActions` is sorted ascending by absolute
`(PosAT, x, y)` (`gamestate.rs`'s `get_all_actions`), `ScriptedBot` had no
branch for `PosAT::SelectPosition`, so the **touchback receiver** fell through
to `first_legal_simple_or_any` → the lowest-x receiver, for *both* teams. Home
attacks toward x=1, so that is Home's most exposed player, on the line of
scrimmage; for Away it is the safest deep one. Flipping every tie-break in the
bot to "take the last" flips the result to 0.924 (row 1b) — a clean bracket
around the cause. Handling `SelectPosition` relative to the team's own
attacking direction (commit `dcc578c`, with a mirror-invariance regression
test) lands it at 0.532, n.s., and raises the mirror's TD rate from 5.6 to 9.0
per game — deep-carrier is simply the better play here.

**Consequence for the ladder:** every report card's `scripted` rung was, until
this fix, an opponent that played one side far better than the other. The
aggregate `win_rate` still cancelled it (sides alternate), but the rung was
noisier and weaker than it looked.

### Rung 2 — random mirror is null, and that is the load-bearing result

20 000 `RandomBot` vs `RandomBot` games at 14x7: 1750 Home wins, 1855 Away,
16 395 draws. Home share **0.485**, z = −1.75 — 95% CI [0.469, 0.501]. Random
bots *do* score here (0.21 TDs/game, 3605 decided games), so the instrument is
weak per game but well powered in aggregate, and its interval excludes 0.67 by
more than 5σ and excludes 0.113 completely.

Conditioning on the coin toss shows the machinery behaving exactly as it
should: the **receiving** team wins more, whichever side it is (Away receives:
1008−753; Home receives: 997−847), and the coin itself is fair (9933 / 10067).

**There is no engine-, rules- or setup-level side bias of the observed
magnitude.** Combined with rung 1's localisation to a bot-side tie-break, and
with the earlier n=4800 in-drive null, the engine is now exonerated from three
independent directions.

### Rung 3 is therefore live — and the bias is search-*budget* dependent

The same instrument, same board, heuristic MCTS mirror at **200** iterations:
Home 133, Away 118, share 0.530 (z = +0.95, n=300 games / 150 seed-pairs;
pair-clustered t = +0.94). At **1000** iterations the same match-up gives
0.667–0.709 across three runs. A bias that appears only as the search deepens
cannot be a property of the rules — it is a property of the search.

That points squarely at **H-c**: the search's chance models were written in
board coordinates. `throw_in_outcome` took the first in-bounds `D3`, and
`D3::One` is `(1, ±1)` on a y-sideline, so both bots believed sideline throw-ins
always travel toward +x (Away's attacking direction); `bounce_outcomes`
collapsed the out-of-bounds directions to `oob.first()` in `ALL_DIRECTIONS`
order, which starts with `dx = +1`, and that representative fixes the square
the throw-in is taken from. Both now prefer the axis-aligned exit (`dx == 0`,
else `dy == 0`), which maps onto itself under `x -> width-1-x` (commit
`316dbca`). A first attempt that fanned the throw-in uniformly over D3 was
reverted: it grew the search tree enough to break `solved_early_stop`'s budget.

**Measured, and H-c is refuted.** 180 games, heuristic mirror, 1000 iterations,
14x7, *with* the H-c fix (`316dbca`): Home 100, Away 50, draw 30 —
**0.667** Home share, z = +4.08; pair-clustered over the 90 seed-pairs,
t = +4.14. Side TDs 490:314 (0.609, z = +6.2). Against the pooled pre-fix
1000-iteration baseline (176-82, 0.682) that is a change of −0.016, z = −0.32:
**no effect at all**. Both coin-toss conditions carry it (kicking-first-half
Home: 53-24; Away: 47-26), so it is not a receive effect.

So the mirror-invariance fix was worth making but is *not* the mechanism, and
the score is now: B1, B1b, B2 and H-c all refuted as causes.

What the 180 games *do* establish, on the same instrument as the 200-iteration
run above:

- **The effect reproduces cleanly at 1000 iterations on fresh seeds** (0.667,
  z = +4.1) — a fourth independent replication, now ~450 games total, all
  landing 0.667–0.709.
- **It is budget-dependent.** 0.530 at 200 iterations vs 0.667 at 1000, on the
  same board, bots and instrument: z = +2.65 between the two, p = 0.008. This
  is the sharpest handle anyone has had on the phenomenon, and it is a cheap
  one — 200-iteration games run at ~11 s against ~77 s.
- **Home out-scores Away, it does not merely out-convert.** 490 TDs to 314. A
  win-conversion story (e.g. clock or turn order at the half boundary) does not
  fit.


### What this rules in and out

- **Out:** the engine's rules, geometry, setup, kickoff and coin-toss
  machinery, at any magnitude near the measured effect (rung 2, plus rung 1c
  and the n=4800 in-drive null).
- **Out:** "some bot bug that happens to favour Home" as a general explanation
  — the *scripted* bot's side bug favours Away.
- **In:** a search-side mechanism, active only at higher iteration counts.
- **General lesson, now with two instances:** any deterministic choice made in
  **absolute board coordinates** — action ordering, an outcome representative,
  a "first legal" fallback — is a side bias waiting to happen, because the two
  teams attack in opposite x directions. Plan 023 originally cleared tie-breaks
  because MCTS's children live in a `HashMap`; that is true of MCTS's *tree*,
  but not of the bots and models that read `get_all_actions()` or
  `ALL_DIRECTIONS` directly. Prefer mirror-invariance property tests
  (`x -> width-1-x` + team swap) over pinning the current value.

## Open hypotheses and the experiments that would settle them

**H-a — The instrument's pairing shares the coin toss.** MCTS models the coin as always-Heads (`roll_outcomes.rs:33-53` routes `RequestedRoll::Coin` to `scripted_result` → `Coin::Heads` with probability 1.0 at `:214`), so the Away bot's call is *deterministic*, not a tie-break. Both games of a seed-pair therefore share the coin result, the toss winner, and **who receives** — so the Home/Away split is confounded with the receiving side, on 50 coin draws rather than 100. Quantitatively **insufficient** to explain the effect on its own (it would need Home to have received in ~35-45 of 50 seeds, a 2.8-5.6σ deviation for a fair coin), but it widens the error bars.

**H-b — Residual: the effect is genuine but smaller than measured.** z ≈ 3.0 on a single 100-game instrument, in an analysis that went looking for exactly this pattern. Given B1's small first-order magnitude and arguable sign, "real but ~2× smaller, with B1 + the B2 snowball supplying part of it" is the most defensible reading until reproduced.

**H-c — MCTS's kickoff/throw-in models are not mirror-invariant. REFUTED as
the cause (2026-08-30), fixed anyway in `316dbca`; see the rung-3 result
above.** Original statement: `roll_outcomes.rs:65-81` collapses a throw-in to one deterministic outcome, trying `D3::One` first; for a y-sideline throw-in `get_throw_in_direction` (`ball_procs.rs:154-157`) maps `D3::One → (1, ±1)`, so **both bots believe sideline throw-ins always travel toward +x** (Away's attacking direction). Likewise `bounce_outcomes` collapses OOB directions to `oob.first()` in `ALL_DIRECTIONS` order, preferring +x (`:150-152`), and `Deviate` is modelled as `(D6::One, up)`. These are in-drive phenomena and so are largely excluded by the n=4800 null — but they are genuine non-mirror-invariant modelling assumptions and worth fixing on their own merits (a D3-uniform or side-relative representative).

### Deferred tests, cheapest first (all need the machine idle)

1. **Re-run the same 100-game mirror with a different `--seed` base** (~2h). The single most informative experiment: it tests H-a and H-b simultaneously. If the split reproduces near 57-28, it is mechanistic; if it moves wildly, it was seed-set luck.
2. **Run the mirror at `BOARD_SIZE_W=16` — no patch required** (~2h). At engine width 18 the buggy formula is *accidentally* mirror-symmetric (`w/4` = 4, `3w/4` = 13, and `17-4` = 13 ✓). If the bias collapses at 16x7 but persists at 14x7, B1 is confirmed as the source. Confounded by the board-size change, but zero patch risk.
3. ~~**`RandomBot` vs `RandomBot` full-game mirror.**~~ **Done** (20 000 games): null, 0.485. See "Result part 2", rung 2.
4. **After fixing B1 and B2, re-run the mirror.** Expect both the TD rate (currently 4.2/game) and the side split to move. Fix and measure them *separately* — B2 changes the scoring dynamics of every game, so bundling them makes the result uninterpretable.
5. ~~**Close the instrument gap**~~ **Done** (`ae43fa2`, `80175cf`): `LadderRow` carries `tds_by_home`/`tds_by_away`, `--per-game-out` logs side-relative scores and `kicking_first_half` per game, and `eval_summary.py` prints the side-centric split. The coin-toss *winner* and the kick/receive *choice* are still not recorded.

## What to try next (2026-08-30, in priority order)

1. **Bisect the budget.** 200 iterations is clean and 1000 is 2:1; ~100 games at
   400 and at 700 (roughly 30 and 55 minutes on 3 cores) would say whether the
   effect switches on sharply or scales with depth. A threshold implicates
   something that only deep search reaches (the half boundary, the second
   drive); smooth scaling implicates a bias the search *amplifies*.
2. **Instrument the drive, not the game.** Home out-scores Away 490:314 in full
   games, but heuristic MCTS from random-start mid-drive states is symmetric at
   n=4800. The difference is everything between the kickoff and the first
   decision of the drive. Log per *drive*: who kicked, where the ball landed,
   who ended up carrying, and who scored. That interval is now small enough to
   inspect directly.
3. **Property-test the pieces rather than reading them.** `leaf_score` is
   antisymmetric by inspection, MCTS's perspective handling is symmetric by
   inspection, and `ScriptedBot` looked side-agnostic by inspection too — right
   up until a 400-game mirror said 0.113. A `mirror(state)` helper (reflect x,
   swap teams) would turn all of these into one-line assertions:
   `leaf_score(mirror(s)) == -leaf_score(s)`, `search(mirror(s)) ==
   mirror(search(s))` at a fixed seed. That helper is the single highest-value
   piece of infrastructure left here.
4. **Suspects not yet excluded**: the search's always-Heads coin model
   (`scripted.rs`, only Away ever calls the toss); tree reuse across moves; the
   pruning rules under deeper search; the second-half kickoff swap.

## Result (2026-08-31): action ordering **is** a lever — and a big one

Hypothesis under test: `AvailableActions` / `get_all_actions()` is sorted
ascending by `(PosAT, x, y)` (`gamestate.rs`'s `get_all_actions`), so
"earlier in the list" means "lower x", which is *directionally meaningful and
opposite for the two sides*. The earlier dismissal (see "Ruled out") argued
that MCTS's children live in a `HashMap` whose `RandomState` randomises
iteration order, so `select_node`'s `max_by` cannot drift in +x. That
argument is about *whether the shipped code has a directional preference*.
It says nothing about whether a directional preference **would matter**, and
that is the testable question.

### The instrument

`BLOOD_MCTS_TIE_BREAK` (commit `cc37aac`, `TieBreak` in `dynamics.rs`) adds a
deterministic tie-break to `select_node` (both PUCT arms) and to
`pick_best_action`, applied *after* the value comparison so it only decides
exact ties:

- `hash` — **shipped behaviour**: `cmp_actions` returns `Equal`, so `max_by`
  keeps whichever child the children `HashMap` yielded last. Arbitrary,
  randomised per process.
- `asc` — prefer the action that sorts **first** in `(PosAT, x, y)`: "take
  the lowest-x option", for both teams.
- `desc` — prefer the action that sorts **last**: "take the highest-x option",
  for both teams.
- `mover` (added later, `4539280`) — prefer the action nearest the **mover's
  own** attacking endzone; the only one of the four that is
  mirror-*covariant*, and therefore the only one that could ship.

`asc` and `desc` bracket the largest side bias any in-search ordering
preference can produce, which is what makes a mirror match between them
decisive. That the lever has bite was checked first on the cheap
`convergence` probe: repeat-to-repeat agreement on the root pick at 1000
iterations rises from **26/40** under `hash` to **39/40** under `asc`, so
ties really are what decides the shipped search's run-to-run
nondeterminism; and `asc` and `desc` disagree with each other on the root
pick in 39 of 80 cells.

### The measurement

Heuristic MCTS mirror match, 1000 iterations, 14x7 tier
(`BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4`), full games from
`CoinToss`, three arms on the **same two seed bases** (500000 / 600000, 100
games each) so the arms face identical situations. Home-side wins counted
per game from `--per-game-out` (`home_score` vs `away_score`), so the
`wins_as_home + losses_as_away` trap cannot bite. `t` is on the 100
independent seed-pairs.

| arm | decided | Home | Away | draw | Home share | z | pair t | side TD H:A |
|---|---|---|---|---|---|---|---|---|
| `hash` (shipped) | 172 | 121 | 51 | 28 | **0.703** | +5.34 | +5.46 | 527:344 (0.605) |
| `asc` (prefer low x) | 159 | 80 | 79 | 41 | **0.503** | +0.08 | +0.07 | 339:353 (0.490) |
| `desc` (prefer high x) | 165 | 120 | 45 | 35 | **0.727** | +5.84 | +5.55 | 450:272 (0.623) |

- `asc` − `desc` = **−0.224**, z = −4.15, p ≈ 3e-5 (95% CI −0.330 .. −0.118).
- `asc` − `hash` = −0.200, z = −3.72, p ≈ 2e-4.
- `desc` − `hash` = +0.024, z = +0.5, n.s.

**Ordering moves the bias, by a lot.** Changing nothing but the direction in
which the search breaks exact ties moves the mirror-match Home share from
0.70 to 0.50, and takes the side TD split with it (0.605 → 0.490). This is
the same shape as `ScriptedBot`'s touchback tie-break, one level up: a
preference expressed in absolute board coordinates, inside the search this
time. **It is not, however, the cause** — see "The `mover` arm" below, where
the mirror-covariant version of the same rule leaves the bias untouched.

The `hash` arm is also a **fifth independent replication** of the effect
itself (0.703, z = +5.3), on fresh seeds and after the B1/B1b/B2/H-c fixes.

Conditioning on the coin toss kills **H-a** outright: both the effect and
its removal appear independently in each condition, so this is not a
receive effect and the shared coin within a seed-pair is not doing the work.

| arm | Away kicks first | Home kicks first |
|---|---|---|
| `hash` | H62-A23, 0.729 (z = +4.23) | H59-A28, 0.678 (z = +3.32) |
| `asc` | H40-A44, 0.476 (z = −0.44) | H40-A35, 0.533 (z = +0.58) |
| `desc` | H61-A23, 0.726 (z = +4.15) | H59-A22, 0.728 (z = +4.11) |

### The `mover` arm: ordering is a knob, **not** the cause

The bracket's shape is not what a naive ordering story predicts. If the
shipped `hash` order were an unbiased coin between "early" and "late",
`hash` would sit near 0.5 and `asc`/`desc` would bracket it. Instead
`hash` ≈ `desc` ≈ 0.71 and only the low-x preference moves anything. That
leaves two readings — "the tie-break is the cause and `asc` is the fix"
versus "the tie-break is a knob and `asc` is a compensation" — and
`TieBreak::Mover` separates them, because a *mirror-covariant* rule cannot
by construction create or cancel a side bias.

Fourth arm, same instrument, fresh seed bases (700000 / 710000 / 720000,
50 games each), on a build that also carries the pathfinder fix, with a
matched `hash` control on the identical seeds:

| arm | decided | Home | Away | draw | Home share | z | pair t | side TD |
|---|---|---|---|---|---|---|---|---|
| `mover` (mirror-covariant) | 109 | 69 | 40 | 41 | **0.633** | +2.78 | +2.49 | 286:211 (0.575) |
| `hash` control | 124 | 82 | 42 | 26 | 0.661 | +3.59 | +3.74 | 380:305 (0.555) |

`mover` − `hash` = **−0.028**, z = −0.45, 95% CI [−0.151, +0.095]. **The
mirror-covariant tie-break leaves the bias intact.**

So the verdict on the hypothesis this experiment was built for:

- **Action-enumeration ordering is refuted as the *cause*.** Making the
  search's tie-break mirror-covariant — the only version of the rule that
  could be responsible for a side bias — changes nothing.
- **It is confirmed as a large *lever*.** `asc` really does take the mirror
  match from 0.70 to 0.50 (z = −3.7), and `desc` really is +0.22 above it. A
  directional tie-break inside the search is worth ±0.11 of Home share, so
  this family of bug stays dangerous even though it is not the culprit here.
- `asc` is therefore a **compensation, not a fix**, and must not ship: it is
  an absolute-coordinate rule, tuned to this board's geometry, and would
  itself fail a mirror-invariance test.
- Note the direction, worth remembering: "prefer the option toward *your own*
  attacking endzone" (which `asc` is, for Home) makes that side play
  **worse**, not better. The same sign shows up in the value probe below —
  `asc` is the arm where the search is *least* Away-optimistic and Home wins
  *least*. Optimism and strength run opposite here.
- **The most useful thing the `mover` arm establishes is a contradiction to
  chase.** Under `mover` the tie-break is mirror-covariant and the whole
  transition + evaluation pipeline is property-tested mirror-exact (next
  section), so the search *ought* to be equivariant and the mirror match
  *ought* to read 0.500. It reads 0.633 at z = +2.8. Something in the
  selection/aggregation layer is still not covariant — residual candidates:
  chance-outcome ties `mover` does not order (three directions share each
  `dx`), the `HashMap` order that survives wherever `cmp_actions` returns
  `Equal`, and order-dependent f64 rounding in the chance-node expectation
  (`weighted_sum` accumulates in map order).
- Caveat on cross-arm comparison: `asc` (0.503) and `mover` (0.633) have
  almost the same value-probe reading (−52 and −54), so **the value probe does
  not predict the win rate across arms**. They are two related but distinct
  measurements; do not treat either as a proxy for the other.


## Result (2026-08-31): a `mirror(state)` helper, and what it turned into assertions

Plan 023's own next-step list called a `mirror(state)` helper "the single
highest-value piece of infrastructure left here". It now exists, and it has
already paid for itself twice.

`GameState::mirrored()` (`gamestate.rs`) reflects a state about the board's
vertical midline (`x -> width-1-x`) and swaps the benches: players, board
index, ball, bounce squares, both `TeamState`s, the team-typed `GameInfo`
fields, `available_actions` (via `AvailableActions::mirrored`), and the live
`Half` procedure's kicking team. Out of scope and documented as such: the
rest of the procedure stack (so the result must not be *stepped*) and path
offerings (their `Node` chains carry their own positions).

For things that must be stepped — a search — `botbowl-mcts/tests/common/`
carries `mirror_playable`, which rebuilds a random-start state reflected and
team-swapped through `GameStateBuilder`, plus `fingerprint` /
`mirror_fingerprint`: an ID-free rendering of everything a bot can read off
a state, and the same rendering of its mirror computed field-by-field from
the original. Every test below asserts the rebuild equals the mirror on that
fingerprint before it measures anything, so the instrument validates itself.

What that established (all in `botbowl-mcts/tests/`, all green):

| property | file | coverage |
|---|---|---|
| `leaf_score(mirror s) == -leaf_score(s)` | `mirror_symmetry.rs` | 300 states |
| `prior_for` and `should_prune` are mover-relative | `mirror_symmetry.rs` | ~thousands of (state, action) pairs |
| the legal-action set mirrors | `mirror_symmetry.rs` | 150 states |
| `mirror(step(s,a,r)) == step(mirror s, mirror a, mirror r)` | `mirror_transitions.rs` | 2123 engine transitions, dice pinned |
| movement **routes** mirror | `mirror_transitions.rs` | 25121 risky routes |
| `roll_outcomes::enumerate` mirrors as a distribution | `mirror_chance_model.rs` | 3919 chance nodes (134 D8 bounces, 43 deviates) |
| the *search's* `available_actions` + `apply_action` mirror at depth | `mirror_apply_action.rs` | ~60k steps over 4000 lockstep walks |

The last row is the strong one: it covers priors (compared via
`BbAction`'s `prior_bits`), pruning (via the enumerated sets), the scripted
quiescent picks and `sole_legal_action`, and the composition of all of them
to the full depth of a search horizon. So **the entire state-transition and
evaluation pipeline the search is built from is now property-tested
mirror-exact**, which is a much stronger statement than the three
"symmetric by inspection" claims it replaces.

### Two more absolute-coordinate bugs, found by these tests

**The pathfinder's route tie-break** (`21c2e09`) — the fourth instance of
this plan's general lesson, and the one with real gameplay consequences.
`Move(dest)` names a destination, not a route. When two routes tie on
everything `Node::is_better_than` compares (probability, block dice, foul
target, remaining movement, cumulative distance) it returns `false`, so the
winner is whichever node reached the square *first*, i.e. whichever
direction came first in `expand_node`'s iteration over `ALL_DIRECTIONS`.
That list is `(1,1), (0,1), (-1,1), (1,0), (-1,0), (1,-1), (0,-1), (-1,-1)`:
every `dx = +1` entry precedes its `dx = -1` partner, and reflecting the
list in x gives a permutation of itself rather than itself. **On a tie the
pathfinder stepped toward +x — Away's attacking direction — for both
teams.**

Destinations and probabilities were unaffected, which is why it survived: a
risk-free reroute is invisible. It becomes visible the moment the route
crosses a tackle zone, because the two mirrored players then dodge from
*different squares* — exposed to different opponents, and on a failure
landing in different places. Measured: **1080 of 25121 risky routes (4.3%)**
did not mirror, and 31 of 2353 steps of a mirrored MCTS `apply_action` walk
diverged. `Direction::all_directions_toward(attacking_dx)` now picks the
expansion order from the mover's own attacking direction; both counts are 0
after the fix. Two engine tests had pinned the old arbitrary route
(`one_long_path`, `handoff_failed_catch_bounces_from_receiver_square`) — the
same "tautological test" pattern this plan flagged for `kickoff_position`;
both now assert the meaningful thing instead.

**Mirroring a state must mirror the turn order** (`58773f3`) — a bug in the
*instrument*, recorded because it cost a wrong reading. `Half::step` picks
the next team turn from its **own** `kicking_this_half` field, not from
`GameInfo`: `other(kicking)` when the two turn counters are level,
`kicking` otherwise. A mirror that swaps every team-typed field of
`GameInfo` therefore still leaves the alternation pointing at the original
kicking team, and the mirrored state quietly hands its mover two
consecutive turns. The first version of the search-symmetry probe below
measured a large asymmetry that was entirely this artefact.

## Result (2026-08-31): the search's *value* is side-asymmetric, and the estimator is clean

`mirror_symmetry.rs::search_side_bias_by_budget` (`#[ignore]`d; run with
`--ignored --nocapture`) searches `s` and `mirror_playable(s)` at a fixed
budget and reports

```text
mean( root_value(s) + root_value(mirror s) )
```

`root_value` is Home-centric, so for an equivariant search the two are exact
negatives and the sum is 0. Algebraically the sum equals
`R(the Home-to-move member) - R(the Away-to-move member)` where `R` is the
mover-relative root value, so it measures *how much more optimistic the
search is for an Away mover than for a Home mover in the mirrored
position*. It needs no games, so it is cheap enough to ablate.

**It is exactly 0 at small budgets and grows with search depth** (n=300,
default settings):

| budget | 2 | 5 | 20 | 50 | 200 | 1000 |
|---|---|---|---|---|---|---|
| mean sum | +0.00 | +0.5 | −1.9 | −9.7 | −34.5 | −65.5 |

At n=800 and 1000 iterations: **−65.5 ± 5.4, t = −12.1**, present in every
turn stratum (−45 to −70 for turns 1–7, −125 at turn 8) and the same in both
halves of the `s`-is-Home-to-move / `s`-is-Away-to-move split (−62 and −75,
each t ≈ −9), so it is not an artefact of the two construction routes.

**The control that makes it worth believing.** Blood Bowl is also symmetric
under the **y** reflection (top-for-bottom, teams unchanged), a symmetry that
involves no Home/Away labels at all. The same estimator on the same states
under that map reads **+3.0 ± 3.5 (t = +0.9)** at 200 iterations and
**−0.3 ± 3.0 (t = −0.1)** at 1000. So the estimator returns zero on a board
symmetry and −65 on the side symmetry: this is a Home/Away effect, not an
estimator artefact.

**Ablations at 1000 iterations** (n=400–800):

| arm | mean sum |
|---|---|
| default | −65 |
| `virtual_loss=0` | −136 |
| `puct=normalised c=1` | −33 |
| tie-break `asc` | −52 |
| tie-break `desc` | −75 |
| tie-break `mover` (player nodes) | −55 |
| tie-break `mover` + mirror-covariant chance order | −54 |
| `virtual_loss=0` **and** tie-break `mover` | −129 |
| `BLOOD_MCTS_HORIZON=off` | −72 |
| `BLOOD_MCTS_MEMORY=get` (replay-based DAG equality) | −64 |
| after the pathfinder fix | −65 (unchanged) |

Three things follow, and they are the sharpest constraints anyone has had
here:

1. **The exploration rule modulates it strongly** — virtual loss halves it,
   normalised-Q halves it the other way. That is the signature of an
   asymmetry the *search dynamics* amplify, not a constant offset.
2. **Tie-break direction modulates it, in the same order as the game
   results** (`asc` −52 → 0.503 Home, `hash` −65 → 0.703, `desc` −75 →
   0.727): more Away-favouring value asymmetry goes with *more Home wins*.
   The natural reading is that an over-optimistic search plays worse, so
   over-valuing Away's position makes Away play worse.
3. **A mirror-covariant tie-break does not remove it** (−54 vs −65, and the
   search's mirrored-root-pick agreement jumps from 48% to 78% at 200
   iterations), and neither does the pathfinder fix, the horizon, or the
   DAG-equality strategy. So ordering is a knob, not the source.

### What this rules in and out

- **Out, now by property test rather than by reading:** `leaf_score`,
  `prior_for`, `should_prune`, the legal-action set, the engine's transition
  function, movement routes, the search's chance-outcome model, and the whole
  `available_actions` + `apply_action` pipeline at depth. None of them is the
  source.
- **Out:** the pathfinder route tie-break (a real bug, fixed, no effect on
  either measurement).
- **Out:** chance-node sweep order over direction-valued outcomes; the
  search horizon (`HORIZON=off` gives −72); and `recon_mcts`'s node-equality
  strategy (`MEMORY=get`, which replays action sequences instead of comparing
  stored states, gives −64).
- **In, and now the live lead:** something in the *selection / aggregation*
  layer — PUCT descent, virtual loss, the minimax/expectation backprop, the
  `recon_mcts` DAG and its `HashMap`-ordered children — that treats a
  Home-to-move root differently from the mirrored Away-to-move root. Every
  Home/Away branch in that layer (`home_perspective`, `want_max`, `q_sign`,
  the FPU flip) reads symmetric; four such readings have now been wrong, so
  the next step is to make them assertions.

### Next steps, in priority order

1. **Property-test the selection and backprop layer**, the way the
   transition layer now is — and note the `mover` arm makes this a
   *contradiction*, not merely a gap: an equivariant pipeline plus a
   covariant tie-break should give 0.500 and gives 0.633. The natural
   assertion:
   `search(mirror s) == mirror(search s)` **exactly**, at a fixed budget,
   under `TieBreak::Mover` and with a deterministic hasher — which means
   giving `recon_mcts`'s children map a fixed `BuildHasher` behind a test
   feature. Without that, the residual `HashMap` nondeterminism makes exact
   equality untestable and we are stuck doing statistics on a value that
   should be provable. **This is the highest-value item left.**
2. ~~**Measure the `mover` arm in games.**~~ **Done** — 0.633, indistinguishable
   from a matched `hash` control (z = -0.45). Ordering is a knob, not the cause.
3. **Do not ship `asc`.** It removes the measured bias but is an
   absolute-coordinate rule: it is tuned to this board's geometry and would
   itself fail a mirror-invariance test. `Mover` is the shape a fix has to
   have.
4. **Re-run the ladder rungs after any of this lands.** Every report card,
   promotion gate and champion decision to date was measured under a search
   whose two sides play at measurably different strength.

## A separate finding: the gen-0 net is side-miscalibrated

While quantifying the above, the gen01 `nn-value` shards (generated by `bbnet_14x7_gen00`) showed an **opposite and larger** side asymmetry than the heuristic corpus. Provisional (n=225 drives, sampled mid-run) but consistent across all five shards:

| corpus | n decided | P(Home scores the drive TD) | z |
|---|---|---|---|
| heuristic pooled (gen00 + gen01 s5-7) | 5252 | 0.516 | +2.3 |
| **nn-value (gen00 net)** | 173 | **0.318** | **−4.8** |

Controlling for the active team: P(Home scores \| Home active) = 0.413 vs P(Away scores \| Away active) = 0.783 — **both** conditions shifted ~19 pp toward Away, so this is not a turn-order effect. The net plays Away materially better than Home, or equivalently its value head is side-miscalibrated.

**Likely contributor, worth testing:** `kicking_first_half` and `kicking_this_drive` are the **constant default `Away`** in all 6825 random-start trajectories, because random-start builds at `Turn{turn:1}` and never runs `CoinToss` (`gamestate.rs:361-366`). The net therefore trains on a corpus where those side-identifying features never vary, then is evaluated in full games from `CoinToss` where they do — a train/eval distribution mismatch on precisely a side-identifying feature. Candidate fixes: randomise those fields in `apply_game_context`, or drop them from the encoding if they carry no in-drive signal.

Note this does **not** invalidate the promotion gate: every ladder rung plays paired Home/Away, so the aggregate `win_rate` still cancels side effects.

## A benign artefact, recorded so it isn't mistaken for a bug later

The heuristic corpus shows a small drive-TD bias, P(Home) = 0.516 (+1.6 pp, z = +2.3), which is **harness bookkeeping, not an engine asymmetry**. `botbowl-curriculum/src/random_start.rs:361-363`:

```rust
state.info.home_turn = turn;
state.info.away_turn = if active == TeamType::Home { turn - 1 } else { turn };
```

**Home is always the turn leader**, so a Home-active drive has one more team-turn before half-end than an Away-active drive at the same `turn`. Visible directly in the no-score rates (18.0% Home-active vs 21.8% Away-active), and the effect flips *Away*-positive at the clock edge (turn 8: P(Home) = 0.360, 392/592 no-score). Per-shard differentials flip sign (+26, +22, −40, +17, +44, +11, +41, +4), ruling out a seeding artefact.

## Result (2026-08-31, part 3): exact search-mirror equivariance — one real bug found and fixed, the aggregate bias unmoved

Executed the plan's own top-priority next step: turn `search(mirror s) ==
mirror(search s)` into a provable assertion instead of a statistical one.

**Infrastructure.** `recon_mcts` gained an opt-in `deterministic_hash`
feature (`Cargo.toml`, `src/tree.rs`): behind it, the children `HashMap`
inside `Node`/`Children::Branch` uses `BuildHasherDefault<DefaultHasher>`
(a fixed-seed hasher) instead of the per-process-random `RandomState`, so
iteration order is reproducible across runs given the same insertion
sequence. `botbowl-mcts/Cargo.toml` unions it in for test targets only via
a `[dev-dependencies]` override on the same path dependency (resolver
`"2"` keeps this out of the production lib build). New exact-equality
tests live in `botbowl-mcts/tests/mirror_search_exact.rs`: build a state
and `mirror_playable(s)`, run both with `.with_workers(1)` (no thread
scheduling nondeterminism) and `TieBreak::Mover`, and assert the root pick,
root value (exact negation), root visits, and every root child's
(visits, q, solved) match exactly after mirroring.

**Two real bugs found and fixed**, both instances of the plan's standing
lesson ("any deterministic choice made in absolute board coordinates is a
side bias waiting to happen") one level deeper than rung 3's fix:

1. **`chance_key` (the `Mover` tie-break's ordering for chance-node
   selection) omitted `dy`.** Three of `ALL_DIRECTIONS`' eight entries
   share any given `dx` (e.g. `dx=+1`: `(1,1)`, `(1,0)`, `(1,-1)`), so
   those three tied under `Mover` and fell through to `recon_mcts`'s
   arbitrary children-map order during PUCT descent at chance nodes —
   which subset of tied outcomes gets explored first, and therefore which
   ones accrue visits at a small budget, is not guaranteed to correspond
   between a state and its mirror. `dy` passes through x-mirroring
   unchanged (only `dx` negates), so ordering by it directly is itself
   mirror-covariant. Fixed by extending the key to `(dx, dy, tag)`.
2. **`backprop_scores`'s chance-node expectation summed in `HashMap`
   order.** `weighted_sum: f64 += prob * q.score` accumulated over
   `child_scores_and_actions` in whatever order `recon_mcts` handed them
   back — a function of each action's hash, which is unrelated between a
   state and its mirror (mirroring negates every direction-valued
   outcome's x-component, scrambling the hash). f64 addition is not
   associative, so the final `avg as i64` truncation could land on
   opposite sides of an integer boundary for mirrored searches. Caught
   directly: budget-20 exact-equality run on state 13 of a 40-state batch
   failed with `StartBlock (9,6)` q = **-523** vs its mirror
   `StartBlock (6,6)` q = **-522** — every other root child (`StartMove`
   ×3, `StartBlitz` ×3, `EndTurn`) matched exactly. Fixed by sorting into
   `canonical_chance_key` (every `Direction` field folded through `|dx|`,
   everything else raw) before summing — collapsing each `(Q, A)` pair to
   plain data first, since holding two `lockref::Ref`s across a sort
   deadlocks under the `lockref-guard` contention check (plan 013).

**Result: `search_mirrors_exactly_at_budget_2` and `..._5` are now exactly
green** (40 states, `TieBreak::Mover`, `deterministic_hash`) — the search
is provably equivariant at these budgets, not just measured symmetric.
**`..._20` still fails**, same state, same child (`StartBlock`, now off by
one *elsewhere* in that subtree — `RequestedRoll::BlockDice` collapses to
one scripted `Pow` outcome, so the divergence is not the block-dice roll
itself but something below it: a pushback-square choice, an armor/injury
roll, or a further chance node down that path, not yet localised). Marked
`#[ignore]` with the repro pinned in a doc comment rather than left red.

**The aggregate value-probe bias is unmoved by either fix.**
`search_side_bias_by_budget` at 1000 iterations, `TieBreak::Mover`, n=300
(fresh seed base 23_010, post-fix build): **-61.3 ± 7.1, t = -8.6** —
statistically indistinguishable from the pre-fix `Mover` reading of -54 to
-65 in the 2026-08-31 part-1 ablation table. Both bugs found here are
real, worth having fixed, and demonstrated by the exact-equality harness
to matter at small budgets — but they are not the mechanism behind the
aggregate search-value asymmetry, the same shape of result as H1 (the
kickoff-aim bug), H-c (the throw-in/bounce models) and the pathfinder tie-
break before them: real, fixed, measured to not move the headline number.

**Where this leaves the search:** the exact-equality harness is now
real infrastructure — `mirror_search_exact.rs` plus the
`deterministic_hash` feature — and it works: it found two genuine bugs by
proof rather than by statistics, in under an hour, that five prior
statistical ablations (asc/desc/mover tie-breaks, virtual-loss=0,
normalised-Q, horizon=off, `MEMORY=get`) had not surfaced. The residual
`..._20` failure is the next concrete, reproducible lead — smaller in
scope than "the selection/backprop layer" (the prior framing), now
"whatever is below a `StartBlock` resolution in this one traced case."
Given the aggregate bias's magnitude (-61) and this bug's demonstrated
size (±1 on one child at budget 20), plausible reads are (a) there are
several more bugs of this exact shape stacking up across deeper subtrees,
or (b) the dominant mechanism is structurally different from a
tie-break/summation-order bug — e.g. genuine, non-arbitrary PUCT float
divergence from FPU/prior computation once two mirrored subtrees' visit
counts diverge even slightly (which the −65 → −136 virtual-loss=0 ablation
and the −33 normalised-Q ablation both suggest: the *exploration dynamics*
modulate the effect strongly, which a single-cause tie-break bug would
not do). Continuing to bisect the `..._20` failure with the harness now in
hand — rather than further global ablations — is the highest-value next
step; not attempted further here given the scope of one session.

## Cross-references

- plan 021 open issue 5 — the 0.40 mirror anomaly this closes; and its "Home/Away asymmetry audit" next step.
- plan 022 — the Home/Away split instrument and the mirror-match pre-flight that produced the data.
- `runs/loop14x7/mirror.json`, `mirror.log` — the 100-game report.
- `runs/loop14x7/gen00/shard*.jsonl` — the n=4800 in-drive null.

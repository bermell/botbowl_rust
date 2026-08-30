# Home/Away side bias: what the 100-game mirror match found

**Status:** Investigation done (2026-08-28/29), fixes and follow-up tests not yet complete. Closes plan 021 open issue 5 — the mirror anomaly is **real**, not n=30 noise. Two engine bugs were verified by code reading; the causal chain from them to the measured effect is *not* established, and the deferred experiments below are what would settle it.

Investigated read-only while the plan-022 weekend loop was running (all 4 physical cores busy), so nothing here involved a build, a test run, or a simulated game. Everything below came from reading code and from data already on disk.

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

## Open hypotheses and the experiments that would settle them

**H-a — The instrument's pairing shares the coin toss.** MCTS models the coin as always-Heads (`roll_outcomes.rs:33-53` routes `RequestedRoll::Coin` to `scripted_result` → `Coin::Heads` with probability 1.0 at `:214`), so the Away bot's call is *deterministic*, not a tie-break. Both games of a seed-pair therefore share the coin result, the toss winner, and **who receives** — so the Home/Away split is confounded with the receiving side, on 50 coin draws rather than 100. Quantitatively **insufficient** to explain the effect on its own (it would need Home to have received in ~35-45 of 50 seeds, a 2.8-5.6σ deviation for a fair coin), but it widens the error bars.

**H-b — Residual: the effect is genuine but smaller than measured.** z ≈ 3.0 on a single 100-game instrument, in an analysis that went looking for exactly this pattern. Given B1's small first-order magnitude and arguable sign, "real but ~2× smaller, with B1 + the B2 snowball supplying part of it" is the most defensible reading until reproduced.

**H-c — MCTS's kickoff/throw-in models are not mirror-invariant.** `roll_outcomes.rs:65-81` collapses a throw-in to one deterministic outcome, trying `D3::One` first; for a y-sideline throw-in `get_throw_in_direction` (`ball_procs.rs:154-157`) maps `D3::One → (1, ±1)`, so **both bots believe sideline throw-ins always travel toward +x** (Away's attacking direction). Likewise `bounce_outcomes` collapses OOB directions to `oob.first()` in `ALL_DIRECTIONS` order, preferring +x (`:150-152`), and `Deviate` is modelled as `(D6::One, up)`. These are in-drive phenomena and so are largely excluded by the n=4800 null — but they are genuine non-mirror-invariant modelling assumptions and worth fixing on their own merits (a D3-uniform or side-relative representative).

### Deferred tests, cheapest first (all need the machine idle)

1. **Re-run the same 100-game mirror with a different `--seed` base** (~2h). The single most informative experiment: it tests H-a and H-b simultaneously. If the split reproduces near 57-28, it is mechanistic; if it moves wildly, it was seed-set luck.
2. **Run the mirror at `BOARD_SIZE_W=16` — no patch required** (~2h). At engine width 18 the buggy formula is *accidentally* mirror-symmetric (`w/4` = 4, `3w/4` = 13, and `17-4` = 13 ✓). If the bias collapses at 16x7 but persists at 14x7, B1 is confirmed as the source. Confounded by the board-size change, but zero patch risk.
3. **`RandomBot` vs `RandomBot` full-game mirror.** Isolates rules/setup asymmetry from bot behaviour entirely — if the bias survives with no search at all, it is pure engine.
4. **After fixing B1 and B2, re-run the mirror.** Expect both the TD rate (currently 4.2/game) and the side split to move. Fix and measure them *separately* — B2 changes the scoring dynamics of every game, so bundling them makes the result uninterpretable.
5. **Close the instrument gap**: record side-relative TDs, `kicking_first_half`, the coin-toss winner and the kick/receive choice per game in the eval report, so the next mirror can distinguish a scoring-rate bias from a win-conversion bias. Currently impossible — see trap 1 above.

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

## Cross-references

- plan 021 open issue 5 — the 0.40 mirror anomaly this closes; and its "Home/Away asymmetry audit" next step.
- plan 022 — the Home/Away split instrument and the mirror-match pre-flight that produced the data.
- `runs/loop14x7/mirror.json`, `mirror.log` — the 100-game report.
- `runs/loop14x7/gen00/shard*.jsonl` — the n=4800 in-drive null.

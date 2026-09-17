# Plan 036 E4 — pre-committed stopping rule

**Written 2026-09-17 16:40 CEST, before any look at the match beyond the n=73 interim.**
Committed to git so the timestamp is checkable.

## Why this exists

The match was sized at 600 games for an effect of +0.05 (SE 0.018, 80% power). The n=73 interim
read 0.589 ± 0.049 — a larger effect than the design assumed, which needs only ~200 games for 80%
power. Stopping *because a result has crossed significance* is optional stopping and inflates the
false-positive rate, so the stop has to be fixed in advance instead. This is that commitment.

## The rule

- **Stop at n = 200 games** (100 Home/Away pairs). Not "200 or when it looks good" — 200.
- **Primary statistic:** paired points score over completed pairs, from
  `scripts/paired_summary.py` on `runs/exp036/e4/e4_match.games.jsonl`.
  Candidate = `new` (the adopted recipe), opponent = `base`. Null 0.50.
- **Significance:** |z| > 1.96 on the paired SE. At n=200 the SE is ~0.032, so the rule resolves
  an effect of ±0.063 or larger.

## The decisions, fixed now

| paired result | reading | action |
|---|---|---|
| CI excludes 0.50, **above** | the recipe makes a better player | keep it; record E4 as confirmed |
| CI straddles 0.50 | no *large* effect either way | **keep the recipe** — it is already adopted on the val evidence, and E4 was always a guard, not the basis for adoption |
| CI excludes 0.50, **below** | the recipe made the player worse | revert `train_loop.sh` to `VALUE_BLEND=1.0 VALUE_WEIGHT=1.0 PER_DRIVE_VALUE_WEIGHT=off`, keep `EPOCHS` under review separately |

The middle row is the one worth stating out loud: a straddle is **not** evidence the recipe is
useless. n=200 cannot see +0.03, and plan 036's own bar was +0.03. A null here means "not large",
and the honest report is the CI, not a verdict.

## Not covered by this rule

`EPOCHS=15` is not under test here — both E4 arms trained at 10 epochs on purpose, so the budget
could not confound the value-target comparison.


---

## Outcome (recorded 2026-09-17 22:00)

Stopped at 220 games (poll granularity: the watcher checks every 120 s and four parallel games
finish inside a gap; the condition was `>= 200` on a fixed schedule, not look-and-decide).

**Paired 0.605 ± 0.029 over 110 pairs. z = 3.62, p = 0.0003, 95% CI [0.548, 0.662].**

CI excludes 0.50 from above → **row 1: keep the recipe, E4 confirmed.** Effect +0.105, versus the
+0.063 this n could resolve and the +0.03 plan 036 asked for.

# Plan 011 — Baseline numbers

This is the captured pre-optimisation baseline for plan 011. Re-run
`cargo test --release -p botbowl-mcts --test expand_bench -- --ignored
--nocapture` and re-fill these tables after every optimisation commit
so the deltas are quotable in commit messages.

## Run metadata

- Date: 2026-05-24
- Commit: `0aab0f7` (`docs: align stale docstrings + grand plan with the new DiceMode API`)
- Host: Darwin arm64 (macOS, Apple silicon)
- Seed: `0xCAFE_1234`
- Builder: `cargo --release`

## `expand_bench` (two consecutive runs, stable to ≤5%)

### Scenario A — `score_td_easy` (1 home player, 6 legal actions, team=Home)

|                            | run 1     | run 2     |
|----------------------------|-----------|-----------|
| `get_action` 1000 iters    | 2.37 ms   | 2.07 ms   |
| → per `tree.step` (µs)     | 2.37      | 2.07      |
| `GameState::clone`         | 3198 ns   | 3259 ns   |
| `apply_action(StartMove)`  | 4300 ns   | 4346 ns   |
| → micro_step share (apply − clone) | 1102 ns | 1087 ns |

### Scenario B — `full_teams` (11+11 players, 63 legal actions, team=Home)

|                            | run 1     | run 2     |
|----------------------------|-----------|-----------|
| `get_action` 1000 iters    | 4.80 ms   | 4.58 ms   |
| → per `tree.step` (µs)     | 4.80      | 4.58      |
| `GameState::clone`         | 3548 ns   | 3572 ns   |
| `apply_action(StartMove)`  | 4638 ns   | 4643 ns   |
| → micro_step share (apply − clone) | 1090 ns | 1071 ns |

## samply percentages

_Not captured this pass. Re-record before deciding L1/L3 ordering._
The recipe is in `PROFILING.md`; expected fields to fill:

| Frame (cumulative) | score_td_easy | full_teams |
|--------------------|---------------|------------|
| `PathFinder::player_paths` | …% | …% |
| `<GameState as Clone>::clone` | …% | …% |
| `BloodBowlDynamics::apply_action` | …% | …% |
| `select_node` + `leaf_score` | …% | …% |

## Hypotheses (from plan 011, Step 1)

- **H1 — paths > 50% of `tree.step` wall-clock.** _Provisionally
  refuted by the wall-clock numbers above._ The `apply_action - clone`
  delta (the time spent inside `micro_step` itself, of which
  `PathFinder::player_paths` is a chunk) is ~1.1 µs vs total
  `apply_action` ~4.6 µs. Even attributing the entire `apply - clone`
  delta to pathing, that's at most ~24% (full_teams) to ~46%
  (score_td_easy) of `apply_action`. As a share of `tree.step`
  (per_iter ~2-5 µs), pathing is ~20-50%. **samply confirmation
  required before committing to a fix.**
- **H2 — clone < 30% of `tree.step` cumulative.** _Provisionally
  refuted in the opposite direction._ `GameState::clone` alone is
  3.2-3.6 µs, comparable to or larger than a whole `tree.step`
  (2-5 µs). One step doesn't equal one clone — `step_into` doesn't
  clone during descent — but at the leaf, expansion clones once per
  child. With 63 children on a full-teams turn-start, that's
  63 × 3.5 µs ≈ 220 µs of pure cloning per expansion. Strong signal
  that **clone cost is the leading lever**, not paths.
- **H3 — select_node + leaf_score < 10%.** Unknown — needs samply.

## Next-pass recommendation

The wall-clock baseline argues for **reordering plan 011's sequencing**:

1. **First**: Cheap-wins B (bitmask `used_skills` / `PlayerStats.skills`)
   and an audit of what `GameState::clone` is actually copying. Plan
   011's "Cheap-win A" is partially done (`paths` already uses `Arc`,
   not `Rc`) — the remaining win is Arc-ing the outer `FullPitch` so
   the 476 inner `Option<Arc<Node>>` cells aren't deep-copied. Cheap-win
   C (Arc the `AvailableActions` box) is also a clean small PR.
2. **Second**: re-measure. If clone drops to ≤1 µs, `apply_action`
   should fall to ~2 µs and `tree.step` proportionally, without
   needing the lazy-paths invasion.
3. **Only after that**: L1 (lazy paths). Only worth the engine-side
   complexity if paths still dominate the residual `apply - clone`
   delta after clone is cheap. The baseline suggests it might not.
4. **L3** (progressive widening) becomes more attractive if clone
   trims work — bounding fan-out to top-k by prior multiplies the
   savings.
5. **L2** (placeholder children) remains the last resort.

Action for next session: run samply, fill the percentages table, and
either confirm or revise the recommendation above. If samply agrees
that clone dominates, start a `cheap-wins-B+C` branch.

## How to update this file

```sh
cd botbowl_rust
cargo test --release -p botbowl-mcts --test expand_bench \
    -- --ignored --nocapture | grep EXPAND_BENCH
```

Copy the numbers into a new dated section above this one. Keep the
old section so deltas are reviewable in `git log`.

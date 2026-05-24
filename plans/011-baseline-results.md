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

## Call-counts (`expand_bench_call_counts`, single-thread, 1000 iters)

The counter wrapper (`CountingDynamics` in `tests/expand_bench.rs`)
bumps an atomic on each `GameDynamics` method.

| Counter / `tree.step()` | score_td_easy | full_teams |
|-------------------------|---------------|------------|
| `apply_action`          | 1.00          | 1.06       |
| `available_actions`     | 0.00 (totals: 3) | 0.02 (totals: 20) |
| `select_node`           | 1.00          | 1.00       |
| `score_leaf`            | 0.00 (totals: 2) | 0.02 (totals: 19) |
| `backprop_scores`       | 0.00 (totals: 1) | 0.00 (totals: 1) |
| per `tree.step()` (single-thread) | ~6 µs | ~9 µs |

**Big finding**: at 200 000 iters the totals stay at 3 / 20 unique
expansions (same as at 1000 iters). The DAG saturates within the
first handful of `tree.step()`s; the remaining ~99.98% of iterations
just descend one level, hit a recombined existing node, and return.
This is *not* a flaw — it's recombination working well — but it means
the bottleneck is "descent + select + one apply", not "expand
N children".

The single-thread per-step (~6-9 µs) is roughly 2× the parallel
`get_action` per-step (~2-5 µs from `expand_bench_main`). The gap is
parallel speedup against RwLock / registry contention, not extra work.

## samply (`expand_bench_for_samply`, 200 000 iters, single thread)

Recipe (now reproducible — see `PROFILING.md`):
```
RUSTFLAGS="-C debuginfo=2" cargo test --release -p botbowl-mcts \
    --test expand_bench --no-run
samply record --save-only -o /tmp/prof.json --rate 4000 -- \
    target/release/deps/expand_bench-XXXX expand_bench_for_samply \
    --ignored --nocapture
python3 tools/samply_flatten.py /tmp/prof.json \
    target/release/deps/expand_bench-XXXX
```
`debuginfo=2` is required — `line-tables-only` leaves the binary
without function symbols and samply emits hex addresses. `--save-only`
runs without the parallel `expand_bench_main` (which deadlocks under
SIGPROF + `std::thread::scope`).

### Inclusive %, single-threaded, 200k iters across both scenarios

(One frame = "this function appears anywhere in stack at sample time".
So apply_action's clone is also counted under apply_action.)

| pct | function |
|---:|---|
| 55.8% | `<GameState as Clone>::clone` |
| 55.8% | `recon_mcts::Node::get_state` (the caller — root state recompute per `tree.step`) |
| 43.7% | `recon_mcts::Tree::make_branch` |
| 29.2% | `<Cloned<I> as UncheckedIterator>::next_unchecked` (clone path's iterator machinery) |
| 20.8% | `BloodBowlDynamics::apply_action` |
| 18.7% | `SmallVec::extend` (positional-action enumeration during expansion) |
| 16.1% | `BloodBowlDynamics::select_node` + `puct_value` + `prior_for` (grouped) |
| 12.4% | `GameState::micro_step` |
| 10.5% | `hashbrown::RawIterRange::fold_impl` (registry hash bucket walks) |
|  4.5% | `priors::prior_for` self |
|  3.5% | `AvailableActions` clone/drop (positional + paths FullPitches) |
|  0.08% | **`PathFinder::player_paths`** |
|  0.05% | `score_leaf` / `optimistic_leaf_score` / `leaf_score` |

### Self-time top hits (where the CPU actually is)

| pct | function |
|---:|---|
| 43.7% | `Tree::make_branch` body (BranchWip loop, locking, smallvec bookkeeping) |
| 17.4% | `SmallVec::extend` |
|  7.9% | `Cloned::next_unchecked` (likely in `[[Option<…>; H]; W]::clone`) |
|  5.0% | `<GameState as Clone>::clone` body |
|  4.9% | `hashbrown::RawIterRange::fold_impl` |
|  4.1% | `priors::prior_for` |
|  1.6% | `drop_in_place [Option<FullPitch<SmallVec<[PosAT; 4]>>>]` |
|  1.4% | `Node::on_drop` |
|  1.0% | `puct_value` |
|  0.5% | `GameState::get_active_player` |
|  0.2% | `GameState::micro_step` body itself |

## Hypotheses (from plan 011, Step 1) — verdicts

- **H1 — `PathFinder::player_paths` >50% of `tree.step`** — **REFUTED.**
  samply puts pathing at **0.08%**. Three orders of magnitude off the
  plan's hypothesis. Reason: pathing only runs during
  `apply_action(StartMove)`, and with a saturated recombining DAG
  expansion fires ~once per 50 000 iters.
- **H2 — `GameState::clone` <30% of `tree.step`** — **REFUTED in the
  opposite direction.** samply puts inclusive clone at **55.8%**.
  Reason: `Node::get_state()` runs at every `tree.step` to recover the
  root state (HashOnly drops child states), and that path clones the
  root once per step.
- **H3 — `select_node + leaf_score` <10%** — **mixed.** `select_node +
  prior_for + puct_value` is 16.1% (over the 10% bar). `leaf_score / FF`
  is 0.05% (well under).

## Next-pass recommendation (substantially revised)

The samply data argues for **reordering plan 011's sequencing
significantly** and **dropping L1**:

1. **Highest-value, smallest change: tree reuse across `get_action`
   calls.** Not in plan 011 ("Out of scope") but the data points
   straight at it: 55% of CPU is `Node::get_state` cloning the root,
   and a tree built for one move is mostly discarded between moves.
   If the root keeps its state and is just `move_root`'d down the
   PV between calls, the per-step clone cost drops dramatically. This
   is "plans/010-or-new"-shaped work, not 011.
2. **Cheap-win B (bitmask skills)** still has a clear self-time
   target: `<GameState as Clone>::clone` self is 5%, drop_in_place of
   FullPitches is 1.6%, the per-clone smallvec/option churn shows up
   in `Cloned::next_unchecked` (7.9% self). Bitmasking the two
   HashSet<Skill>s removes 23 HashMap allocs per clone — likely 1-2%
   wall-clock for low risk. Worth doing.
3. **Cheap-win C (Arc the `AvailableActions` box)** — `Box` is on a
   hot drop path (`drop_in_place Box<AvailableActions>` is in the
   inclusive top 30). Arc::make_mut on write trades clones for
   refcount bumps. Worth doing.
4. **Drop plan 011's L1 (lazy paths).** 0.08% is not worth invasive
   engine changes. Document this and move on.
5. **L3 (progressive widening) only helps in scenarios with much
   higher expansion-to-descent ratios.** The current recombining
   regime barely expands; widening won't matter until the
   tree-reuse + StoreState changes increase expansion's share. Defer.
6. **L2 (placeholder children) — defer.** Same reason as L3.

Open questions for next session:
- Is `make_branch`'s 43% self-time real, or is samply
  mis-attributing? `make_branch` runs ~20 times in 200 000 iters; even
  generous per-call cost shouldn't add to 43%. Plausible: samply is
  attributing inlined `apply_action / Vec::clone` time to its caller.
  Worth one orthogonal check (e.g. `cargo build --release -C
  debuginfo=2 -C inline-threshold=0` and re-profile, or look at
  inline frames in the profile UI).
- `Cloned::next_unchecked` at 7.9% self is suspicious — is it the
  array-of-clones inside `[[Option<...>; 17]; 28]::clone` (the
  FullPitch board / paths clone)? If so, **Cheap-win A's "Arc the
  outer FullPitch"** is on the table too.

## How to update this file

```sh
cd botbowl_rust
cargo test --release -p botbowl-mcts --test expand_bench \
    -- --ignored --nocapture | grep EXPAND_BENCH
```

Copy the numbers into a new dated section above this one. Keep the
old section so deltas are reviewable in `git log`.

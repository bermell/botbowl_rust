# Plan 008 — Run `recon_mcts` concurrently from `MctsBot` (completed)

**Status:** Completed. `MctsBot::get_action` runs `n_workers` (`std::thread::available_parallelism`
by default) workers via `std::thread::scope`; the total `iterations_per_move` budget is split across
them. Tests pin to `with_workers(1)` for determinism.

**Priority:** #3 in v4. Tackle _after_ 006 and 007 so we don't multiply correctness bugs across threads.

## Why this matters

`recon_mcts`'s whole design point is concurrent DAG growth with work-stealing across worker threads
(`recon_mcts/src/lib.rs` crate doc, and the tree.rs worker-coordination logic). Today, `MctsBot::get_action` does:

```rust
for _ in 0..self.iterations_per_move {
    tree.step();
}
```

— a serial loop on the calling thread. That's leaving the library's primary feature on the floor. Estimated 4–8×
iters/sec headroom for free on a modern laptop, plus better tree-shape (work-stealing helps idle threads from getting
stuck behind slow expansions).

## Files to read first

- `botbowl_rust/botbowl-mcts/src/dynamics.rs`
  - `MctsBot::get_action`, lines 377-428. The whole loop.
- `recon_mcts/src/lib.rs` — the crate-level doc-comment explains the intended threading model. Especially the
  work-stealing description.
- `recon_mcts/src/tree.rs`
  - `Tree::new`, line 1432.
  - `Tree::step`, line 1450 — already designed to be called from multiple threads concurrently? Check the signature
    (`&self`) and the lock layout.
  - `Tree` and its surrounding `Arc`/`Mutex`/lockref usage.
- `recon_mcts/src/lockref.rs` — the synchronisation primitive the tree uses.
- `recon_mcts/tests/nim/benchmark_2048.rs` — see how the reference drives the tree from multiple threads
  (`cargo run --bin benchmark_2048 -p recon_mcts-test_nim --release`).
- `recon_mcts/tests/nim/test_mcts_2048.rs` — production-shape concurrent driver. Cribbable.
- `botbowl_rust/botbowl-engine/src/core/gamestate.rs` — `GameState` must be `Send + Sync` (or at least `Send`) for any
  of this to work. Check it.

## Questions to investigate

1. **What's `Tree::step`'s threading contract?** `&self` (cheap to share via `Arc`) or `&mut self` (need external sync)?
   The work-stealing claim implies the former.
2. **Is `BloodBowlDynamics` `Send + Sync`?** It's a unit-like struct (`#[derive(Default, Clone, Copy)]`, no state) so
   yes. Confirm.
3. **Is `GameState` `Send`?** Sub-question: any `Rc<…>`? The CLAUDE.md says `AvailableActions` holds
   `FullPitch<Option<Rc<Node>>>` — `Rc` is NOT `Send`. We may need to swap to `Arc` either in the engine _or_
   clone-on-search by converting at `MctsBot::get_action` time.
4. **How does the reference benchmark spawn workers?** Plain `std::thread`? `rayon`? Crossbeam scope? Match their
   pattern.
5. **What's the right worker count?** `num_cpus::get()` is the obvious default, but with state-clone-heavy iterations
   there may be a sweet spot lower than physical cores. Benchmark.
6. **Does our `fetch_add` in `select_node` (`dynamics.rs:201, 222`) play correctly with concurrent descents?** It's
   `Relaxed`; multiple threads may race past `select_node` and `fetch_add` the same child. Coordinate with plan 007 — if
   that visit counter goes away, this question goes away too.
7. **Iterations-per-move budget meaning under concurrency.** Today `iterations_per_move` is the loop count. With N
   workers, do we want "N workers × M iters each" (total = N·M) or "split M across workers" (total = M)? Pick one and
   document.

## Proposed approach

1. Confirm `GameState: Send`. If `Rc` blocks it:
   - Easiest first step: convert the `Rc<Node>` in `AvailableActions` to `Arc<Node>`. Check if this is touched elsewhere
     in the engine.
   - This is an engine change, not an MCTS change — coordinate via a small separate commit in
     `botbowl_rust/botbowl-engine/`.
2. Once `GameState: Send`, refactor `MctsBot::get_action`:
   - Wrap `tree` in `Arc<Tree<...>>` (or however the reference does it).
   - Spawn `n_workers` threads (default `num_cpus::get()`), each running `for _ in 0..iters_per_worker { tree.step() }`.
   - Join all, then read `get_next_move_info` from the calling thread.
3. Add a config field `n_workers: usize` to `MctsBot` (default = cpus).
4. Run `cargo run --bin benchmark_2048 -p recon_mcts-test_nim --release` first to make sure the reference works on this
   machine. Then mirror the pattern.

## Tests / success criteria

- All existing MCTS tests pass.
- `ScoreTdEasy` / `ScoreTdMedium` success rates are within ±2pp of pre-change measurements. Faster wall-clock is the
  goal; quality should not regress.
- Bench: time `MctsBot::new(1000).get_action(state)` single-threaded vs parallel on a fixed seed. Record the iters/sec
  ratio in the commit message.
- Run the suite with `RUST_TEST_THREADS=1` and without to make sure parallel test execution doesn't break anything.

## Pitfalls

- **`expose_rolls`/`fixes`/`rng_enabled` setup in `get_action`** must happen _before_ the tree is built and shared — no
  thread should be mutating root state.
- **Deterministic seed.** `root_state.rng_enabled = true` plus multiple threads = nondeterministic search even with a
  fixed seed. Either reseed each worker (need a `seed` field on `MctsBot`) or accept nondeterminism for parallel runs.
  Document the choice.
- **Tests may rely on determinism.** `score_td_easy.rs` uses a fixed seed `0xCAFE_1234` and a tight threshold. Parallel
  search may shift the measured rate. Either widen the threshold or pin `n_workers = 1` for tests. Prefer the latter to
  keep correctness signals sharp.
- **The `recon_mcts::Node::on_drop` panic** documented in `plans/005-learnings--mcts-chance-nodes.md` may be more likely
  to fire under concurrency. Have a reproducer ready.
- **Don't combine with plan 010 (FF) in one commit.** Concurrency + FF was exactly the v3 combo that produced the drop
  panic.

## Out of scope

- Tree reuse across calls.
- Removing per-step `GameState` cloning (separate perf work).
- GPU / SIMD / fancy scheduling.

## Results (2026-05-24)

Shipped in two commits, sequenced per the plan:

1. **Engine `Rc<Node>` → `Arc<Node>`**: surgical rename across `core/pathing.rs`, `core/model.rs`, `scripted_bot.rs`. No
   behavior change; engine tests green; downstream crates rebuild clean.
2. **`MctsBot` parallel workers**: `Arc<Tree>` + `std::thread::scope`. New `n_workers` field defaults to
   `std::thread::available_parallelism()`. Total budget (`iterations_per_move`) is split across workers; the field's
   semantic is preserved so test thresholds remain calibrated. New `with_workers(n)` builder; integration tests pin to
   `with_workers(1)`.

**Measured speedup (10-core laptop, release build):**

| Iters/move | Workers | Wall-clock (2 trials, ScoreTdEasy) | Speedup |
| ---------- | ------- | ---------------------------------- | ------- |
| 1000       | 1       | 97ms                               | —       |
| 1000       | 10      | 101ms                              | 0.96×   |
| 20000      | 1       | 651ms                              | —       |
| 20000      | 10      | 368ms                              | 1.77×   |

Speedup grows with budget — at 1k iters/move, thread spawn dominates. The 1.77× ceiling at 20k is well below the plan's
loose 4–8× estimate; bottlenecks are (a) the `fetch_add(Relaxed)` race on `BbScore.visits` (multiple threads claim the
same child before the leaf returns — plan 007 territory) and (b) per-step `GameState.clone()` (explicit out-of-scope).
Bench code lives in `botbowl-mcts/tests/parallel_bench.rs` as `#[ignore]`.

**Test pinning:** all four curriculum integration tests (`score_td_{easy,medium}`, `get_the_ball_{easy,medium}`) use
`.with_workers(1)`. `ScoreTdEasy` (previously flagged flaky at the 0.80 threshold) is now deterministic on the seed —
see updated `mcts-test-flake-score-td-easy` memory.

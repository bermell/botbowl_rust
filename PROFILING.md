# Profiling the MCTS hot path

How to measure where `MctsBot` spends its time. Set up alongside plan 011
(profile and cut state-derivation cost). See
`plans/011-baseline-results.md` for the latest captured numbers.

## Quick wall-clock numbers (no extra tooling)

`botbowl-mcts/tests/expand_bench.rs` is a manual microbench printing three
per-op timings on two start states (`score_td_easy` — single-player
lecture, and `full_teams` — 11+11 players, 60+ legal actions):

```
cargo test --release -p botbowl-mcts --test expand_bench \
    -- --ignored --nocapture
```

Each run emits stable-format `EXPAND_BENCH …=…` lines so you can diff
runs mechanically.

## CPU profile with samply (recommended)

Install once: `cargo install --locked samply`.

Release builds strip frame pointers and debug info; samply can't resolve
hot frames without them. Enable debug line tables for the release
profile **temporarily** when profiling — don't commit this; production
release builds shouldn't carry the size cost. Either:

- Set the env var (no file edit needed):
  ```
  RUSTFLAGS="-C debuginfo=line-tables-only" \
      cargo build --release -p botbowl-mcts --test expand_bench
  ```
- Or paste into `botbowl_rust/Cargo.toml` for the duration of the
  profiling session, then revert:
  ```toml
  [profile.release]
  debug = "line-tables-only"
  ```

Record a profile of the bench:

```
samply record -- cargo test --release -p botbowl-mcts \
    --test expand_bench expand_bench_main -- --ignored --nocapture
```

samply opens the resulting profile in a local browser viewer.

### What to look at in the profile

Plan 011 hypothesises these percentages of `tree.step()` wall-clock; the
profile lets us confirm or refute them.

- `PathFinder::player_paths` (cumulative, callees included) — plan
  hypothesis H1: >50%.
- `<botbowl_engine::core::gamestate::GameState as Clone>::clone`
  (cumulative) — plan hypothesis H2: <30%.
- `BloodBowlDynamics::apply_action` (cumulative) — clones happen inside
  it via the by-value parameter, so this should subsume both above.
- `BloodBowlDynamics::select_node` + `botbowl_mcts::score::leaf_score` —
  plan hypothesis H3: <10% combined.

Capture the breakdown in `plans/011-baseline-results.md` so future
optimisation commits can quote a delta against it.

## Allocation profile (follow-up, not wired yet)

`dhat-rs` would give us allocs/iter, but wiring it requires installing
its custom global allocator, which is intrusive enough that we want a
dedicated commit. Deferred — record allocation counts only if the CPU
profile leaves the bottleneck ambiguous. When we do wire it, add
`dhat = "0.3"` as a `dev-dependency` of `botbowl-mcts` behind a Cargo
feature so the default test profile is unaffected.

## Existing related benches

- `botbowl-mcts/tests/score_td_easy.rs::mcts_lifts_random_baseline` —
  50 × 1000-iter trials, asserts success rate. Same seed
  (`0xCAFE_1234`), so its wall-clock is comparable to `expand_bench`'s
  `score_td_easy` workload.
- `botbowl-mcts/tests/parallel_bench.rs::bench_parallel_vs_serial` —
  serial vs `available_parallelism` workers at 20k iters. Use it to
  re-measure parallel scaling after expansion-cost changes.

All three are `#[ignore]`d, so default `cargo test` doesn't run them.

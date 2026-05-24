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

**Build with full debug info.** `line-tables-only` is *not* enough on
macOS — samply emits hex addresses instead of function names if the
binary has no symbols. Build the test binary with `RUSTFLAGS`:

```sh
cd botbowl_rust
RUSTFLAGS="-C debuginfo=2" cargo test --release \
    -p botbowl-mcts --test expand_bench --no-run
```

The `--no-run` step prints the test binary path
(`target/release/deps/expand_bench-XXXXXX`). Copy that path.

**Profile the single-threaded test, not the parallel one.** The main
`expand_bench_main` uses `std::thread::scope` for MCTS workers and
deadlocks under samply's `SIGPROF`. Use `expand_bench_for_samply`
instead — it drives `Tree` single-threaded via the `CountingDynamics`
wrapper, runs ~2 s, and produces clean attribution:

```sh
samply record --save-only -o /tmp/expand_bench_profile.json --rate 4000 \
    -- target/release/deps/expand_bench-XXXXXX \
       expand_bench_for_samply --ignored --nocapture
```

### Flatten to a self / inclusive % table

`samply load` opens a browser UI. For a CLI breakdown that can be
diffed in commits, use `tools/samply_flatten.py` (committed alongside
this doc — it symbolicates via `atos` on macOS):

```sh
python3 tools/samply_flatten.py /tmp/expand_bench_profile.json \
    target/release/deps/expand_bench-XXXXXX
```

Output: top-30 self-time + top-30 inclusive-time tables, plus a
"grouped" section that bins frames into plan-011-relevant buckets
(pathing, clone, apply, select, recon_mcts internals, hashbrown, etc.).
Copy the numbers into `plans/011-baseline-results.md`.

### What to look at in the profile

The baseline numbers and hypothesis verdicts are in
`plans/011-baseline-results.md`. Key findings to compare against
when re-profiling after a change:

- `PathFinder::player_paths` (inclusive): baseline ~0.08% — H1 refuted.
- `<GameState as Clone>::clone` (inclusive): baseline ~55.8% — H2
  refuted in the opposite direction.
- `Tree::make_branch` (self-time): baseline ~43.7% — top hot function
  by self-time despite running only ~20 times per 200 000 iters
  (worth re-checking with inline frames).
- `BloodBowlDynamics::apply_action` (inclusive): baseline ~20.8%.
- `BloodBowlDynamics::select_node + prior_for + puct_value`:
  baseline ~16.1%.

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

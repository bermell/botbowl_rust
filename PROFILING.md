# Profiling the MCTS hot path

How to measure where `MctsBot` spends its time. **Training-loop throughput is the current
focus** (root `CLAUDE.md`, plan 058): measure before and after every speedup. Standing tools:

| Tool | Measures |
|------|----------|
| `scripts/perf_search_bench.sh OUT [TAG]` | Search CPU as instructions retired for a fixed, reproducible workload, plus a trajectory hash (equal hash = search unchanged). Linux `perf`; builds into `target/perf-16x9`. |
| `scripts/perf_gen_bench.sh OUT [NET] [PARALLEL_LIST] [GAMES]` | Generation games/min, process CPU and sidecar batch/GPU stats at each `--parallel-games`. |
| `scripts/nn_throughput_probe.sh <arm>` | Which stream source (shards, games per process, MCTS workers) feeds the NN sidecar best. |
| `BLOOD_NN_PROFILE=1` | Prints the process-wide NN forward counters (`NN_PROFILE` line: forwards, time per forward). |
| `BLOOD_MCTS_STATS=1` | Prints the `MCTS_TELEMETRY` line (tree reuse, recombination). |

The samply recipe below is for CPU flame profiles of a single search. Historical context:
`plans/completed/011-plan--profile-and-cut-expansion-cost--completed.md`, plan 058 §7.

## Quick wall-clock numbers (no extra tooling)

`botbowl-mcts/tests/expand_bench.rs` is a manual microbench printing three per-op timings on two
start states (`score_td_easy` — single-player lecture, and `full_teams` — 11+11 players, 60+
legal actions):

```
cargo test --release -p botbowl-mcts --features expand_bench --test expand_bench \
    -- --ignored --nocapture
```

Each run emits stable-format `EXPAND_BENCH …=…` lines so you can diff runs mechanically.

## CPU profile with samply (recommended)

Install once: `cargo install --locked samply`.

**Build with full debug info.** `line-tables-only` is _not_ enough on macOS — samply emits hex
addresses instead of function names if the binary has no symbols. Build the test binary with the
workspace's `profiling` profile (release + `debug = 2`, in its own `target/profiling/`, so it never
invalidates the release build the way `RUSTFLAGS` did):

```sh
cargo test --profile profiling --features expand_bench \
    -p botbowl-mcts --test expand_bench --no-run
```

Any binary works the same way, e.g. `cargo build --profile profiling -p botbowl-ui`
→ `target/profiling/botbowl-ui`.

The `--no-run` step prints the test binary path (`target/profiling/deps/expand_bench-XXXXXX`).
Copy that path.

**Profile the single-threaded test, not the parallel one.** The main `expand_bench_main` uses
`std::thread::scope` for MCTS workers and deadlocks under samply's `SIGPROF`. Use
`expand_bench_for_samply` instead — it drives `Tree` single-threaded via the `CountingDynamics`
wrapper and produces clean attribution:

```sh
samply record --save-only -o /tmp/expand_bench_profile.json --rate 4000 \
    -- target/profiling/deps/expand_bench-XXXXXX \
       expand_bench_for_samply --ignored --nocapture
```

### Flatten to a self / inclusive % table

`samply load` opens a browser UI. For a CLI breakdown that can be diffed in commits, use
`tools/samply_flatten.py` (committed alongside this doc — it symbolicates via `atos` on macOS):

```sh
python3 tools/samply_flatten.py /tmp/expand_bench_profile.json \
    target/profiling/deps/expand_bench-XXXXXX
```

Output: top-30 self-time + top-30 inclusive-time tables, plus a "grouped" section that bins
frames into perf-relevant buckets (pathing, clone, apply, select, recon_mcts internals,
hashbrown, etc.).

## Existing related benches

- `botbowl-mcts/tests/score_td_easy.rs::mcts_lifts_random_baseline` — 50 × 1000-iter trials,
  asserts success rate. Same seed (`0xCAFE_1234`), so its wall-clock is comparable to
  `expand_bench`'s `score_td_easy` workload.
- `botbowl-mcts/tests/parallel_bench.rs::bench_parallel_vs_serial` — serial vs
  `available_parallelism` workers. Use it to re-measure parallel scaling.

All three are `#[ignore]`d, so default `cargo test` doesn't run them.

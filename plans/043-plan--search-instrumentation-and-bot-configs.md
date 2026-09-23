# Plan 043 — Search instrumentation and named bot configs

**Status: implemented.** Landed in one pass; the numbers below are from the first real runs.

## Why

Three things the bot does constantly were invisible in production runs, and one thing we wanted to
compare could not be named.

1. **Tree reuse.** `run_search` has always re-rooted its cached tree into the node matching the new
   state (plan 015 Step 1), and has always computed `was_reused` — but that variable was used
   *only in a panic message*. When reuse works the search starts from a DAG that already holds a
   plan; when it fails the plan is thrown away. We had no idea how often, and no way to tell the
   causes apart.
2. **Recombination.** `recon_mcts` counted registry hits and misses but only dumped them to stderr
   behind `BLOOD_MCTS_STATS=1`, so nothing on disk carried them. And the number needed to decide
   whether recombination *pays* was missing entirely: how often a hash-bucket candidate is rejected
   by the state comparison. Under `StoreState` each comparison clones and compares two whole
   `GameState`s, so a rejected one is pure waste.
3. **Valuation in the web debug view.** `evaluator_value` existed but rendered as a bare
   `net value +0.412`, in the searching agent's frame, and only after a **bot** move.
4. **No named configuration.** `MctsConfig` held every knob, but the only ways to set it were
   `BLOOD_MCTS_*` and a growing set of hand-written `--backup/--vs-backup` twin flags. There was no
   way to commit a configuration, stamp it into a run, or play net-X-under-A against net-X-under-B.

## What landed

### recon_mcts — the probe counters

`RegistryInfo` gains `probes`, `eq_checks`, `eq_hash_equal`, `eq_rejects`, `lookup_probes`,
`lookup_hits`, plus `snapshot() -> RecombinationStats` (a plain `Copy` value with `Sub` for deltas
and `AddAssign` for folding).

The interesting part is `eq_rejects`. `HashSet::get` hides the bucket stage — a `None` return is
indistinguishable from "a candidate was compared and rejected" — so the counter lives inside
`impl PartialEq for Node`. That impl also backs each node's `parents` set, so a raw counter would
be contaminated; a `thread_local` tally harvested by a `ProbeGuard` scoped to each registry probe
keeps it exact. The guard is held across `HashSet::get`/`insert` and nothing else. Always on, no
feature gate — a `Cell` bump is nothing against the state comparison it measures, and
`botbowl-mcts`'s `LEAF_STATS` is the in-repo precedent.

`eq_hash_equal` splits genuine 64-bit hash agreement from hashbrown's ~1/128 7-bit tag brushes,
which turned out to matter (below).

`recon_mcts/tests/recombination_stats.rs` pins it, including a deliberately collision-prone `Hash`
that drives rejections without changing which states recombine.

### botbowl-mcts — reuse outcome and the telemetry accumulator

`telemetry.rs`: `ReuseOutcome` (`Reused | Disabled | NoCache | AnchorMiss | MarkerMiss |
LookupMiss | NoPath`), `ReuseDecision` (outcome + top proc + pruned action fan + path length),
`TreeReuseStats` (totals and `by_proc`), `ActionFanHistogram` (sparse and exact, so percentiles are
fans that actually occurred), `RecombinationCounts` (a serde-able mirror — `recon_mcts` is
dependency-free by design), and `SearchTelemetry` tying them together.

Plain `u64`s on `MctsBot`, not atomics: `get_action` takes `&mut self` and every write happens on
the owning thread before the workers spawn. One tally per bot, because the web server deliberately
runs several differently-tuned bots in one process and a global could not be attributed.

**The subtle bit, found by running it:** the per-search recombination delta must take its baseline
from the tree the search *actually runs on*, read after the reuse attempt resolves. Reading it from
the cached tree first looks right and is wrong — a `lookup_miss` probes the cached tree and then
searches a brand-new one, so the difference saturates to zero and silently drops that decision's
entire cost. It showed up as `probes != hits + misses` in the first end-to-end report.
`tests/tree_reuse_stats.rs` now drives long enough to force a rebuild and asserts the accounting
closes.

### Named presets

`MctsConfig` is `Copy` + serde, with its serde default pointed at `MctsConfig::new` — **not**
`Default`, which is `from_env()`. A named configuration that quietly absorbed a stray
`BLOOD_MCTS_*` would not be reproducible, which is the whole point of naming one.
`deny_unknown_fields` turns a typo into an error rather than a knob that silently stays put.

`cfgs/*.toml` + `botbowl_play::bots::load_mcts_config` → `SearchConfig.config`. Reaching it:
`--bot-config` / `--vs-config` on `botbowl-ui eval`, `--bot-config` on `dataset`, and the
flag-for-flag twins on `botbowl-hub job eval` / `job generate`. Unset is *exactly* the old
behaviour. Set, it replaces the configuration wholesale, and clap marks it `conflicts_with` the
per-knob flags so a run can never be half-preset, half-flag.

The name travels separately from the knobs — the rung label, `Report.candidate_config`,
`TrajectoryMeta.extra["mcts_config"]` — because `SearchConfig` must stay `Copy` for the hub wire.
`PROTOCOL_VERSION` 3 → 4.

### Where the numbers land

- **eval:** `EvalGameLine.telemetry` is `SearchTelemetry` itself, not a flattened copy, so
  `LadderRow::record` folds it with `SearchTelemetry::merge` — one implementation, used identically
  by `botbowl-ui` and by the hub rebuilding a report from workers' lines. `play_ladder_game`
  **drains** the bot rather than reading it, which is what makes a per-game line mean "this game".
  The hand-written `Serialize` gained a second trailing optional under the same postcard rule as
  `board`; three tests pin it.
- **dataset:** flattened into `TrajectoryMeta.extra` — the documented extension point, so no
  `FORMAT_VERSION` bump and no reader change.
- **web:** `SearchReport.health` drives an inspector "Search health" block. The valuation is now
  labelled by team (`favours Home 0.42`) and, via a new `ServerMsg::Valuation` emitted from
  `GameSession::view`, **live during the human's turn** — one forward pass per board change.
- **opt-in:** `--trace-reuse <path>` writes a per-decision JSONL row with the action list. Off by
  default, because formatting action lists for every decision of every game is real work for a
  question nobody is asking most of the time.

## What it measured, first time out

Heuristic evaluator, 16x9/3 board, 2 games, 200 iterations — a small run, but the shape is clear.

```
searches 181 · reuse 86 (47.5%) · anchor_miss 37 · lookup_miss 57 · no_path 0
  MoveAction   reused=42  lookup_miss=28
  Turn         reused=29  anchor_miss=31
  Block        reused=0   lookup_miss=9
  FollowUp     reused=6
recombination: hit rate 10.2% · eq_reject_rate 82% · 85% of comparisons are full 64-bit hash agreements
```

Three things worth following up, in `plans/032`:

1. **`Block` never reuses.** Nine block decisions, nine lookup misses. `MoveAction` reuses two
   thirds of the time, so this is not a general failure. *(Chased down — `botbowl-mcts/tests/
   block_reuse.rs` and plans/032 item 13. Two mechanisms, 14 + 12 of 26 sampled misses: the
   quiescent loop resolves the die with `scripted_pick` so no node is ever built, and where a node
   **is** built it carries `block_outcomes`' representative dice rather than the faces rolled. The
   real finding was downstream of that: the bot picks a different die from the script **51%** of
   the time, so the tree values every future block under a policy it then overrules.)*
2. **`Turn` splits evenly between reuse and `anchor_miss`.** The anchor misses are turn boundaries
   and unavoidable. Fine, and it is the baseline the other rows should be read against.
3. **The state hash is not discriminating, and now we know what that costs.** *(Fixed — see
   `plans/044-plan--state-hash-discrimination.md`: 36.8% -> 0% colliding states, 4.07 -> 0.12
   comparisons per probe, 6-9% faster search.)* 82% of state
   comparisons are rejected, and 85% of them had *equal 64-bit hashes* — far beyond chance.
   `GameState::hash` (`botbowl-engine/src/core/gamestate.rs:566`) deliberately hashes only
   `proc_stack.len()` and `proc_stack_top()`, with the comment "collisions are corrected by
   PartialEq". They are corrected — the answers are right — but each correction clones and compares
   two whole `GameState`s. A longer single-bot probe measured **7.7 comparisons per registry
   probe**. Hashing a little more of the proc stack is the obvious experiment, and it is now
   measurable: recombination's hit rate against its wasted-compare rate, in every report.

Whether recombination is worth keeping at all is the question the user asked this for, and the
instrument now answers it per run rather than per guess.

## Verification

- `cd recon_mcts && cargo test` (4 new tests) and `cargo fmt` — required by `.cursor/rules`.
- `cargo test --workspace`, including `lazy_mover_identity` and `mirror_search_exact`: the counters
  are observational and search output is byte-identical.
- `cargo check -p botbowl-web-client --target wasm32-unknown-unknown` for the proto changes.
- End to end: an `eval` run with `--bot-config`, `--trace-reuse` and `--per-game-out` where the
  per-game telemetry sums to `report.json`'s and to the trace's row count; a `dataset` run whose
  `meta.extra` carries the same keys.
- `cargo test --workspace --release -- --ignored` (the bot benchmark suite) — every MCTS capability
  benchmark passes.

**One pre-existing failure, not caused by this plan.** `lazy_mover_identity`'s `#[ignore]`d
`search_output_unchanged_full_matrix` fails, and fails **byte-identically at pristine `HEAD`**
(verified in a throwaway worktree). Its in-file doc comment blames the encoder change and says the
heuristic arm is unaffected; that is out of date — the first divergence is a *heuristic* cell
(`budget=200 state=1`, `root_visits` 122 vs 128). The golden was blessed on 2026-09-16 (`d286f22`)
and six engine behaviour commits have landed since, three of them pathing/rules fixes. Re-blessing
it is its own reviewable job. The **default-on** heuristic golden is up to date and passes with
this change, which is the evidence that the counters here are observational.

## Not done

- **No distributed `--trace-reuse`.** The per-decision trace is a local diagnostic; the aggregate
  crosses the wire and that is what a distributed report needs.
- **The hub still never runs the lecture battery** (`lectures: Vec::new()`), so any telemetry
  attached to lectures is absent from every distributed run. Pre-existing, noted here because it is
  the kind of asymmetry this plan otherwise removed.

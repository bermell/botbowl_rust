# Plan 013 — `recon_mcts` memory-mode comparison

## TL;DR

Plan 012's "cycle from hash collision" hypothesis is **confirmed**, but
switching `recon_mcts` markers (`HashOnly` → `GetState` / `StoreState`)
does **not** fix the hang. It substitutes corruption for stack
overflow. The underlying DAG-depth pathology is the real problem; the
marker choice only changes how it manifests.

- `HashOnly` — corruption panics in 0.3–0.8 s (illegal action /
  None unwrap inside `recon_mcts`). Real-world bug, not just a hang.
- `GetState` — no corruption, but recursive `Node::get_state()`
  blows the stack in 16 s (expand bench) – 133 s (score_td_easy).
- `StoreState` — no corruption, but stack-overflows even faster
  (0.6 s on expand bench, 15 s on score_td_easy). Probably a
  drop-time `Arc<Node>` recursion.
- All three modes also panic immediately under multi-thread at
  `recon_mcts/src/tree.rs:886` (the CAS unwrap, plan 012 Step D) —
  this is orthogonal to marker choice.

**Recommendation**: keep `HashOnly` as the default (none of the
alternatives are runnable as-is), and pursue plan 012 Step A
(`set_min_depth` pre-check) and/or Step F (MCTS horizon bound) as the
actual fix. The new env-var hook stays committed because it's a
cheap diagnostic.

## Update — DAG-shape probe (2026-05-31, post-commit)

A follow-up instrumentation pass added `BLOOD_MCTS_STATS=1` to `MctsBot::get_action`. When set, the macro dumps the
`recon_mcts::RegistryInfo` (hits / misses / len) and walks
`tree.find_children_sorted_with_depth()` for a depth distribution,
right before the tree drops. A new `tree_shape` test target runs a
single `get_action` at controllable iter budgets so we can sample the
tree before any overflow.

Two findings tear up the assumptions in plan 011:

### Finding 1 — `HashOnly`'s "recombination" is mostly fake

| iters | `HashOnly` reuse | `GetState` reuse | `StoreState` reuse |
| ----: | ---------------: | ---------------: | -----------------: |
|    50 |           0.7563 |           0.0218 |             0.0218 |
|   200 |           0.9010 |           0.0393 |             0.0328 |
|   500 |           0.7668 |           0.5508 |             0.5335 |
|  1000 |              n/a |           0.6696 |             0.6692 |

`HashOnly`'s 75–90 % "reuse" at small budgets dissolves to 2–5 % under
structural equality. Real recombination doesn't kick in until ~500
iters; even then it's 55–67 %, not the 99.98 % plan 011 recorded. Plan
011's number was measuring hash collisions, not recombinations.

`HashOnly` at iters ≥ 50 also makes the registry's depth walk
_non-terminating_ (the depth helper recursively visits a node's
children and the cyclic graph from collisions causes infinite
recursion → stack overflow). That's direct evidence that `HashOnly`
constructs a true cyclic graph, not a DAG.

### Finding 2 — DAG depth is modest under correct equality

Single-call probe on `ScoreTdEasy::new().setup(seed=0xCAFE_1234)`,
single-thread `StoreState`:

|  iters | reg_len | max_depth | nodes at depth ≥ 21 |
| -----: | ------: | --------: | ------------------: |
|     50 |     494 |         6 |                   0 |
|    200 |   1 356 |         6 |                   0 |
|    500 |   3 130 |        29 |                 100 |
|  1 000 |   4 123 |        54 |                 411 |
|  2 000 |   7 131 |        57 |               1 137 |
|  5 000 |  14 811 |        56 |               2 407 |
| 10 000 |  25 750 |        57 |               3 685 |

Max depth caps around 54–57 even at 10 000 iters. That ceiling looks
load-bearing — it's the depth at which the search hits states the
scoring heuristic treats as terminal-enough that PUCT stops descending
further. Comfortable headroom over the 2 MB default thread stack:
54 × ~1 KB-per-`get_state`-frame ≈ 54 KB.

### So why does the actual benchmark still overflow?

The first `get_action` in `score_td_easy.rs` runs to depth 54 and
prints stats cleanly. The **second** `get_action` overflows before
emitting either stats line. The bot's intermediate move advanced the
state to one whose search tree reaches a deeper terminal — but we
can't see how deep because the dump happens after the workers join,
and the worker dies first.

This matches the plan-012 description: the DAG-depth pathology is
state-dependent, not a constant cost. Bounding the horizon (Step F)
caps the depth deterministically and removes the dependency on which
state the bot stumbles into.

## Original run metadata

- Date: 2026-05-31
- Host: Darwin arm64 (macOS, Apple silicon)
- Seed: `0xCAFE_1234` (curriculum lectures use their own seed)
- Builder: `cargo test --release -p botbowl-mcts --no-run`
- Plumbing committed in this branch:
  - `MemoryMode` enum + `with_memory_mode` builder on `MctsBot`
    (`botbowl-mcts/src/dynamics.rs`).
  - Runtime selection via `BLOOD_MCTS_MEMORY={hash|get|store}`.
  - Worker-count override via `BLOOD_MCTS_WORKERS=N` (added so
    `expand_bench_main`, which otherwise uses `available_parallelism`,
    can be pinned to 1 worker for clean per-marker comparison).

## Results

### Single-thread (`BLOOD_MCTS_WORKERS=1`)

| Test                | Marker     | Status                                  | Wall clock | Notes                                                                                 |
| ------------------- | ---------- | --------------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| `score_td_easy`     | HashOnly   | PANIC `is_legal_action` (engine 1002)   | 0.39 s     | MCTS picked an action illegal in the actual state — clear hash-collision corruption.  |
| `score_td_easy`     | GetState   | STACK OVERFLOW                          | 133.59 s   | Recursive `Node::get_state()` walking from leaf to root.                              |
| `score_td_easy`     | StoreState | STACK OVERFLOW                          | 14.95 s    | Stored state per node, so not `get_state()` recursion — likely drop-time Arc cascade. |
| `get_the_ball_easy` | HashOnly   | PANIC `Option::unwrap()` `tree.rs:1516` | 0.32 s     | recon_mcts expected a child node that wasn't there — same corruption family.          |
| `get_the_ball_easy` | GetState   | TIMEOUT @ 300 s (no panic, no overflow) | 300.01 s   | Made progress but glacially; would have hit OOM or stack later.                       |
| `get_the_ball_easy` | StoreState | TIMEOUT @ 300 s (no panic, no overflow) | 300.02 s   | RSS climbed to ~570 MB; still searching.                                              |
| `expand_bench_main` | HashOnly   | PANIC `is_legal_action`                 | 0.34 s     | Same corruption mode as `score_td_easy`.                                              |
| `expand_bench_main` | GetState   | STACK OVERFLOW                          | 16.18 s    |                                                                                       |
| `expand_bench_main` | StoreState | STACK OVERFLOW                          | 0.55 s     | Fastest-failing scenario — useful future regression target.                           |

### Multi-thread (`BLOOD_MCTS_WORKERS` unset → `available_parallelism`)

| Test                | Marker     | Status                                | Wall clock | Notes                                                                  |
| ------------------- | ---------- | ------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `expand_bench_main` | HashOnly   | PANIC CAS `tree.rs:886` + lock poison | 0.50 s     | Plan 012 Step D — multi-thread `score_gen` CAS unwraps.                |
| `expand_bench_main` | GetState   | PANIC CAS `tree.rs:886` + lock poison | 1.01 s     | Same CAS bug; reaches it slightly later because tree growth is slower. |
| `expand_bench_main` | StoreState | PANIC CAS `tree.rs:886` + lock poison | 0.08 s     | Same CAS bug.                                                          |

**Update (2026-06-01)**: the multi-thread CAS panic at `tree.rs:886`
is fixed in `recon_mcts` commit `8368bee`
("tree: retry the score_gen CAS instead of unwrapping it"). The
migration from `compare_and_swap` to `compare_exchange` had left
`.unwrap()` in place; replacing it with a retry-on-Err `match`
unblocks short-budget multi-thread runs end-to-end. With the fix
applied, `expand_bench_main` at 10 workers completes
`ScoreTdEasy/1000-iter` at 745 µs/iter wall, and `parallel_bench`
runs cleanly at ≤5 000 iters.

A **second** concurrency issue surfaces at higher budgets: `parallel_bench`
at `iters ≥ ~8 000`, 10 workers, deadlocks (all 10 workers in a
stopped state, RSS plateaus at ~420 MB, CPU drops to 0%). Likely an
AB/BA lock-order issue in `Node::connect_child`'s acquire of
`parent.children.write` + `child.parents.write` — under enough
contention two workers can take them in opposing orders. Out of
scope for this plan; lands as a separate follow-up (plan 015 or a
recon_mcts upstream fix).

Per-step (µs) numbers are not extractable from any of the runs above —
every configuration crashes before `bench_get_action` finishes its
20-trial loop. Plan 011's `2.37 µs / step` figure is therefore the
last clean datapoint we have; this experiment didn't move it.

## Interpretation

### Why `HashOnly` panics rather than hanging

Plan 012 documented a _hang_ in `set_min_depth`. This run shows
**panics in <1 s** instead. Most likely the curriculum scenarios on
this branch are slightly different (or `apply_action` has tightened)
so the hash collision now produces an immediate illegal-action /
missing-child outcome before the DAG even gets dense enough to enter
the `set_min_depth` quadratic. Same root cause — `HashOnly` merges
distinct states — different downstream symptom.

### Why `GetState` and `StoreState` stack-overflow

- `Node::get_state()` at `recon_mcts/src/tree.rs:768-783` is
  recursive: when `state` is `None` it upgrades a parent `Weak<Node>`
  and calls `get_state()` on the parent, then applies the action.
  `GetState` clears state on every node (line 431), so equality checks
  on deep nodes recurse the full depth of the DAG.
- `OnDrop` for `Node` at `tree.rs:1124-1143` calls `get_state()` when
  the dropped node has children but no stored state. So during Tree
  teardown the same recursion fires.
- `StoreState` keeps state per node, so it shouldn't trigger the
  `get_state()` recursion. Its overflow is almost certainly the drop
  cascade of `Arc<Node>` itself (each parent's drop runs its children's
  drops, depth-first). At the DAG depths this branch reaches, that
  blows the default 2 MB thread stack.

Both modes therefore expose the same underlying fact: the post-refactor
DAG is genuinely deep, not just hash-collision-densified. `HashOnly`
hid this by squashing distinct states into one node.

### Why multi-thread CAS fires for every marker

`tree.rs:886`'s `compare_exchange(..., Release, Relaxed).unwrap()` is
not retried on failure. Any contention on `score_gen` panics. With
multiple workers all racing through `set_min_depth` / `make_branch`
this trips almost immediately, regardless of which marker is in use.
This is plan 012's Step D; the experiment doesn't change its status.

## What changed in the tree

- `botbowl-mcts/src/dynamics.rs`
  - Imports now bring in `GetState` and `StoreState`.
  - New `pub enum MemoryMode { HashOnly, GetState, StoreState }` +
    `pub fn with_memory_mode(self, MemoryMode) -> Self` on `MctsBot`.
  - `get_action` resolves `MemoryMode::resolve(self.memory_mode)`
    (`BLOOD_MCTS_MEMORY` env var wins when set), and a local macro
    `run_with_marker!` is matched over the three arms — each arm
    monomorphises a different `Tree<..., MARKER, ...>`.
  - `n_workers` is now `BLOOD_MCTS_WORKERS`-overridable for the same
    reason: lets `expand_bench_main` be pinned to single-thread for
    apples-to-apples marker sweeps without editing the bench.

Default `MctsBot::new(N)` still uses `HashOnly` and
`available_parallelism()` workers — existing callers see no behaviour
change.

## What next

Order is the same as the plan-012 ordering, with the marker question
now closed out:

1. **Plan 012 Step A** (`set_min_depth` pre-check) — still the
   highest-leverage, lowest-risk change. The pathology this experiment
   surfaced (deep DAG) is exactly what Step A targets.
2. **Plan 012 Step F** (bound MCTS horizon by team-turn / score
   change) — gives a structural cap on DAG depth and removes the
   stack-overflow risk from `GetState`/`StoreState` too, in case we
   want to revisit them.
3. **Plan 012 Step D** (CAS retry at `tree.rs:886`) — needed before
   any multi-thread bench can run, but only worth tackling after A+F
   make the single-thread case healthy.
4. _(Deferred)_ Restoring `GetState` viability would also need
   `recon_mcts::Node::get_state` (and probably the drop chain)
   converted from recursive to iterative. Not worth doing until the
   DAG-depth issue is bounded — at which point the simpler `HashOnly`
   may stay fine.

## Reproducing

```sh
cd botbowl_rust
cargo test --release -p botbowl-mcts --no-run
EB=$(ls -t target/release/deps/expand_bench-* | grep -v '\.d$' | head -1)
STE=$(ls -t target/release/deps/score_td_easy-* | grep -v '\.d$' | head -1)
GTBE=$(ls -t target/release/deps/get_the_ball_easy-* | grep -v '\.d$' | head -1)

for MODE in hash get store; do
  echo "==== single-thread, $MODE ===="
  BLOOD_MCTS_WORKERS=1 BLOOD_MCTS_MEMORY=$MODE \
      /usr/bin/time -p timeout 300 $STE --ignored --nocapture --test-threads=1
  BLOOD_MCTS_WORKERS=1 BLOOD_MCTS_MEMORY=$MODE \
      /usr/bin/time -p timeout 300 $GTBE --ignored --nocapture --test-threads=1
  BLOOD_MCTS_WORKERS=1 BLOOD_MCTS_MEMORY=$MODE \
      /usr/bin/time -p timeout 120 $EB expand_bench_main --ignored --nocapture
done
```

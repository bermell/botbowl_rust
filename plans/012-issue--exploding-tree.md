# Issue 012 — MCTS tree explodes after the step-function refactor

## Symptom

After the `step_with_roll_or_action` refactor (commit `c0f216a`, tests
fixed in `1b824b7`), `MctsBot::get_action` no longer terminates on the
curriculum scenarios.

- Single-threaded, `MctsBot::new(N).with_workers(1)` on `ScoreTdEasy`:
  - `N ≤ 13`: completes in ~20 ms.
  - `N ≥ 14` (varies — sometimes 15, sometimes 25): hangs indefinitely,
    killed at any timeout we set.
  - Hang depth is **non-deterministic** between runs because
    `recon_mcts` walks `HashMap`s with the default randomised hasher.
- Multi-threaded: same hang, plus the worker threads also produce a
  `Result::unwrap()` panic at `recon_mcts/src/tree.rs:886`
  (`score_gen` CAS that's not retried).
- All ignored bot benchmarks that use `MctsBot`
  (`botbowl-mcts/tests/{get_the_ball_*,score_td_*,parallel_bench,expand_bench}`)
  hang or panic. The scripted-bot benchmarks
  (`botbowl-curriculum/tests/{get_the_ball,score_td}.rs`) all pass.
- Default `cargo test --workspace` (no `--ignored`) is fully green
  (149 tests) and the engine behaves correctly — the regression is
  isolated to the MCTS layer.

## Root cause (confirmed by samply)

15-second single-thread profile of the hang on `ScoreTdEasy`, 200 iters:

```
## Inclusive top
100.00%  recon_mcts::tree::Tree::step
 99.99%  recon_mcts::tree::Tree::make_branch
 99.94%  recon_mcts::tree::Node::set_min_depth      ← bottleneck
 43.79%  core::hash::BuildHasher::hash_one
 34.30%  recon_mcts::unique_heap::UniqueHeap::push
 30.67%  hashbrown::HashMap::insert
 27.93%  recon_mcts::unique_heap::UniqueHeap::pop
  0.05%  botbowl_engine::core::gamestate::step_with_roll_or_action

## Self-time top
24.97%  core::hash::BuildHasher::hash_one
23.95%  recon_mcts::tree::Node::set_min_depth
18.81%  core::hash::sip::Hasher::write
13.75%  hashbrown::raw::RawIterRange::fold_impl
 5.42%  hashbrown::HashMap::insert
 5.00%  hashbrown::RawTable::remove_entry
 4.40%  UniqueHeap::pop
 3.63%  UniqueHeap::push
```

The botbowl engine accounts for **0.05%** of total time. Effectively
100% of the hang is inside `recon_mcts::Node::set_min_depth`.

### What `set_min_depth` does

`recon_mcts/src/tree.rs:978`. When `Tree::create_scored_child` connects
a parent → child edge (line 1552 / 1576), `set_min_depth(&child)` runs
a BFS down the DAG, recomputing each descendant's max-path-from-root
depth via `update_depth` (line 961). The BFS uses a
`UniqueHeap<ArcWrap<Node>>` whose dedup hashes the full node tuple,
which is where the `hash_one` / `Hasher::write` / `RawIterRange` time
goes.

`update_depth` (`tree.rs:961-976`) is `1 + max(parent.depth)`, and the
new depth is `max(current, new_min_depth)` — i.e. _depth only grows_,
and only the longest path matters. The BFS in `set_min_depth` skips
descendants when the current node's depth didn't grow (`tree.rs:990`)
so in a "steady state" DAG it should bottom out quickly.

### Why the refactor exposed it

Pre-refactor `apply_action` did a single `micro_step` and left the
engine **mid-procedure** — between square-steps of `Move`, between
sub-procedures of a `Block`, etc. Those mid-procedure states carry
enough path information in their `proc_stack` that two distinct action
sequences rarely produced the same hash. The DAG was sparse and
`set_min_depth` walks were tiny.

Post-refactor `apply_action` drains to **canonical decision points**.
Many action sequences now land on hash-equivalent states (a player at
some square + ball state + turn marker is the same state regardless
of which dodge or GFI roll path got us there). The DAG becomes
dramatically denser; every new edge can promote some descendant's
depth, and the BFS that propagates the promotion now walks a far
larger subgraph.

Empirically:

- Per `tree.step`: ~100 µs for the first ~12-14 iterations
  (still-sparse DAG), then one step takes ≥120 s (DAG hit the
  density tipping point).
- The number of `botbowl_engine` calls is unchanged. It's the
  internal bookkeeping that's quadratic-ish.

### Adjacent smells worth keeping in mind

- `recon_mcts/src/tree.rs:886` — `compare_exchange(...).unwrap()` on
  the `score_gen` CAS. With a single worker this never fires; with
  multiple workers and the denser DAG, the score-gen CAS contends
  and the unwrap panics. The dynamics-side comment at
  `botbowl-mcts/src/dynamics.rs:183` ("while bisecting the v3
  reconstruction panic") is the same family of bug — `recon_mcts`
  has known concurrency edges that are smelling worse under the
  bigger DAG, but the underlying cause is upstream of this issue,
  not new.
- `botbowl-mcts/src/dynamics.rs:361` `optimistic_leaf_score` lost
  its `max_steps` cap when the refactor rewrote it. Not the hang
  source (verified by adding a cap and re-running — still hangs)
  but worth restoring as a defensive guard before any roll-chain
  pathology bites.

## What's been ruled out

These were checked while bisecting:

- `step_with_roll_or_action` infinite loop — trip-wired at 500
  inner `micro_step`s; never fires.
- `optimistic_leaf_score` runaway roll chain — capped at 32 iters,
  hang persists.
- Slow `apply_action` calls — instrumented with a 50 ms threshold;
  no calls exceed it during the hang.
- Large `available_actions` fan-out — aggressive `Move(target)`
  pruning (203 → ~15 actions) does not change the hang at all.
- Engine-layer regression — only the workspace's two `MctsBot` test
  files hang. Engine unit tests (94), MCTS unit tests (38), UI
  snapshots, and the scripted-bot benchmarks all pass cleanly.

## Possible next steps

Loosely ordered cheapest-first. Each is a separable workstream.

### A. Cheap pre-check in `set_min_depth` (recon_mcts) — recommended first

`set_min_depth` is called from `create_scored_child` once per new edge.
For the connect-to-existing-node case (`tree.rs:1550-1559`), the new
parent can only promote the child's depth if
`parent.depth + 1 > child.depth`. Add a fast pre-check that returns
without touching the `UniqueHeap` when that's not true.

- Risk: low — `update_depth` already takes the max of parents, so
  skipping when the new edge can't promote is provably safe.
- Effort: ~20 lines of code in `recon_mcts/src/tree.rs`. New
  unit test covers the "no-op connect" path.
- Expected payoff: collapses the 99.94% bottleneck to near-zero
  for steady-state recombinations. The remaining `set_min_depth`
  calls (genuine depth promotions) are a tiny fraction of edges.

### B. Cheaper hashing inside the `UniqueHeap` (recon_mcts)

The heap dedupes by `*const Node` pointer but `BuildHasher::hash_one`
inclusive is 43.79% — the `Hash` impl is doing more than hashing the
pointer. Audit `ArcWrap<Node>`'s `Hash` impl and reduce it to a
single `usize`-as-pointer hash.

- Risk: low if the dedup key truly is the pointer identity.
- Effort: small, mostly auditing.
- Expected payoff: ~5× cheaper-per-heap-op even after step A — useful
  for the future where the DAG is genuinely large.

### C. Restore the FF safety cap on `optimistic_leaf_score`

`botbowl-mcts/src/dynamics.rs:361`. Re-add the `budget: u32 = 32`
guard from the pre-refactor code. Not load-bearing for this issue but
removes a latent deadlock vector (a long bounce/catch chain at a
chance leaf would otherwise spin until OOM).

- Risk: zero.
- Effort: ~5 lines.

### D. Fix the `score_gen` CAS retry in `recon_mcts/src/tree.rs:886`

The `compare_exchange(..., Ordering::Release, Ordering::Relaxed).unwrap()`
needs to be a retry loop, not an unwrap. Today it panics on contention
and the worker thread dies silently. With steps A+B the DAG stays
small enough that this rarely fires, but the panic itself is a real
bug. Likely also relevant to the "v3 reconstruction panic" referenced
in `dynamics.rs:183`.

- Risk: medium — needs to keep the `score_gen` semantics correct
  under contention.
- Effort: small code change, but requires understanding the
  ordering contract around `score_gen`.
- Expected payoff: multi-threaded MCTS stops crashing in
  `expand_bench` once A unblocks termination.

### E. Auto-resolve "scripted" engine decisions inside `apply_action`

The TODO at `botbowl-mcts/src/dynamics.rs:158-161` already flags this.
After `step_with_roll_or_action` returns, if the resulting state
offers exactly one legal action (or only scripted setup/kickoff
phase actions), keep applying defaults until a real decision surfaces.
Independently valuable — cleans up the tree shape so MCTS only sees
strategic decisions — and gives some breathing room even before A
lands.

- Risk: low if we limit to genuine no-choice states.
- Effort: medium — need to enumerate which engine action sets count
  as "scripted" (setup positional placements, kickoff aim, single
  legal continuation).
- Expected payoff: smaller, cleaner DAG; faster MCTS independent
  of recon_mcts performance.

### F. Bound the MCTS horizon by team-turn / score change

If the search keeps wandering into setup-after-TD or the opponent's
turn (the post-drain canonical states include all of that), bound
the horizon: track `(home_turn, away_turn, home.score, away.score)`
at the root and have `available_actions` return `None` (terminal)
once the current state diverges. The leaf gets scored as-is — the TD
bonus shows up in `leaf_score`.

- Risk: medium — restricts MCTS scope to "this team's turn", which
  is fine for the curriculum lectures but a real change for a
  full-game bot.
- Effort: medium — needs a root marker plumbed into
  `BloodBowlDynamics`.
- Expected payoff: bounded tree size regardless of recon_mcts
  scaling; useful insurance independent of A.

### G. Move-target pruning in `pruning.rs`

The `StartMove` activation exposes ~200 `Move(dest)` siblings. Even
once A makes the tree affordable, that fan-out is bigger than
strictly necessary. Prune to strategic destinations (on the ball,
adjacent-to-ball, adjacent-to-carrier, scoring squares,
1-step-from-active). Already prototyped during the investigation and
verified pure-function-of-`(state, action)`; just didn't help the
specific hang.

- Risk: low if the priors heuristics cover the strategic
  destinations the bot needs.
- Effort: small.
- Expected payoff: faster convergence per MCTS iter once A unblocks
  the search.

### H. Wider `recon_mcts` rethink: lazy depth, sloppy depth, batched updates

If A doesn't fully solve it (or if the DAG can grow large enough that
even the steady-state set_min_depth load matters), revisit whether
`depth` needs to be a tightly-tracked `max(path)` or can be
"eventually consistent" — updated on demand by traversal code that
actually consumes it. Big design change in `recon_mcts`; only
worth doing if A+B don't put MCTS back in the 2-5 µs/step ballpark
that the baseline (`plans/011-baseline-results.md`) records.

## Suggested ordering

1. **A** (`set_min_depth` pre-check) — highest leverage, lowest risk,
   directly targets the 99.94% bottleneck. Land this first and re-run
   `expand_bench` to verify the per-step cost returns to the
   ~2-5 µs ballpark from plan 011's baseline.
2. **C** (FF cap restore) — drop it in alongside A; the diff is
   trivial and removes a latent foot-gun.
3. **D** (CAS retry in `recon_mcts`) — needed for multi-threaded
   `expand_bench` to stop panicking. After A the contention should
   drop, but the unwrap is still a bug.
4. **E** (scripted-action drain in `apply_action`) — orthogonal
   quality improvement; addresses the TODO that's already in the
   code. Land independently of A-D.
5. **B** (hash specialisation) — only worth doing once the DAG is
   genuinely large again; A's pre-check will hide B's payoff until
   then.
6. **F** / **G** — defer until we know whether A-E are enough. Both
   are useful if/when we want MCTS to handle full Blood Bowl games
   rather than curriculum-scale scenarios.
7. **H** — last resort if the algorithmic complexity of
   `set_min_depth` turns out to be the real ceiling.

## Reproducing the profile

```sh
cd botbowl_rust
RUSTFLAGS="-C debuginfo=2" cargo test --release -p botbowl-mcts \
    --test mcts_hang_profile --no-run
samply record --save-only -o /tmp/mcts_hang_prof.json --rate 4000 \
    --duration 15 -- \
    target/release/deps/mcts_hang_profile-XXXX hang \
    --ignored --nocapture --test-threads=1
python3 tools/samply_flatten.py /tmp/mcts_hang_prof.json \
    target/release/deps/mcts_hang_profile-XXXX
```

The `mcts_hang_profile` test target was scratch and isn't checked in;
recreate it from `MctsBot::new(200).with_workers(1)` against
`ScoreTdEasy::new().setup(rng)` with seed `0xCAFE_1234`. Kill the test
process after `samply` finishes recording so the profile flushes.

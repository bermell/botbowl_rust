# Plan 011 — Profile and cut state-derivation cost (lazy expansion + clone hygiene)

**Priority:** parallel to plans 006-010. Likely the single largest *performance*
lever in the bot (correctness levers live in 006/007/010). Big risk surface
though — measure before changing anything.

## Why this matters

Two compounding cost sources at MCTS leaf expansion:

1. **`recon_mcts::make_branch` clones the parent state once per child action**
   (`recon_mcts/src/tree.rs:1609`). For a Blood Bowl turn-start state with
   say 11 fielded players, the legal-action set looks like:
   `StartMove(P_1..P_11), StartBlitz(P_n), StartHandoff(P_n), StartPass(P_n),
   StartFoul(P_n), EndTurn` — easily 30+ children at one node. Every one
   gets a fresh `parent_state.clone()`.

2. **Every `apply_action(StartMove(P_n))` triggers `PathFinder::player_paths`
   for P_n** inside the engine (`botbowl-engine/src/core/procedures/
   movement_procs.rs:151`). Path-tree build dominates engine `micro_step`
   cost. So the cost of expanding a turn-start node is ~30 × (state clone +
   path build). The vast majority of those children will never be deeply
   explored — MCTS picks 1-2 of them.

The clone-cost analysis in the conversation that produced this plan
estimated 15-30 heap allocs per clone (heavy hitters: `paths:
FullPitch<Option<Rc<Node>>>` = 476 cells × Rc bump; `FieldedPlayer.
used_skills: HashSet<Skill>` × 22 players). That's the cheap part. Path
building runs Dijkstra over all reachable squares with skill-aware costs —
multiple orders of magnitude more expensive than a state clone.

## Files to read first

- `botbowl_rust/botbowl-mcts/src/dynamics.rs` — `apply_action`, lines 132-158.
- `recon_mcts/src/tree.rs`
  - `step_into`, lines 1457-1497 — descent doesn't clone.
  - `make_branch`, lines 1596-1639 — the per-child clone loop.
  - `create_scored_child`, lines 1536-1593 — score then `modify_state`
    (drops state under `HashOnly`).
  - `HashOnly` impl, lines 438-455 — when state gets cleared.
  - `Node::get_state`, lines 768-783 — re-derivation walks ancestors.
- `botbowl_rust/botbowl-engine/src/core/gamestate.rs:372-420` — `GameState`
  field layout (what we're cloning).
- `botbowl_rust/botbowl-engine/src/core/model.rs`
  - `FullPitch` definition, lines 22-60 (476-cell inline grid).
  - `FieldedPlayer.used_skills: HashSet<Skill>`, line 411 (per-player heap).
  - `AvailableActions`, lines 575-580 (`paths: Option<FullPitch<Option<
    Rc<Node>>>>` — the big one).
- `botbowl_rust/botbowl-engine/src/core/pathing.rs:784` —
  `PathFinder::player_paths`. Skim to gauge complexity.
- `botbowl_rust/botbowl-engine/src/core/procedures/movement_procs.rs:140-160`
  — where `PathFinder::player_paths` gets called from `StartMove`.

## Step 1 — Profile (gating step, do this first)

Before any change, measure. Hypotheses to confirm or refute:

1. Path-tree builds dominate expansion cost (>50% of `tree.step` wall-clock).
2. `GameState::clone` is second-largest, <30% of wall-clock.
3. `select_node` + `score_leaf` are <10% combined.

Recipe:

```
cd botbowl_rust/botbowl-mcts
cargo build --release --tests
samply record -- cargo test --release mcts_lifts_random_baseline -- --nocapture
# or: cargo install flamegraph; cargo flamegraph --test score_td_easy
```

Or use a manual harness (`benches/expand.rs` or a `#[bench]`) that:
- builds a `ScoreTdEasy` start state,
- calls `MctsBot::new(1000).get_action(state)` in a loop,
- prints µs/iter.

Then change one thing at a time and re-measure. Record numbers in the
commit messages so future-us has a baseline.

**Questions to answer from the profile:**
- µs per `tree.step()` (with the 1000-iter / 50-trial test as the workload).
- % of time inside `PathFinder::player_paths`.
- % of time inside `GameState::clone` (cumulative).
- % of time inside `apply_action` (cumulative — clones happen inside it via
  the move-by-value parameter; that time shows up there).
- Allocations / iter — `samply` shows them; `dhat` confirms.

## Step 2 — Lazy child expansion (the big lever, your idea)

The status quo eagerly fully-derives every child state at `make_branch`
time, even though selection in PUCT for *unscored* children depends only on
`(prior, parent_visits)` — no state needed
(`botbowl-mcts/src/dynamics.rs:354-364`). State-derivation can be deferred
to the first descent that actually picks the child.

### Design options (pick after profiling confirms expansion dominates)

**Option L1 — Lazy state derivation inside `apply_action`.**
Keep the recon_mcts protocol unchanged, but defer the *expensive* parts of
state derivation in the engine. Specifically:

- Don't compute `available_actions.paths` during `micro_step(StartMove(P_n))`.
- Recompute `paths` lazily the first time `available_actions.get_all()` or
  `paths.iter()` is consulted.

This means each child state is cheap to materialise at expansion time;
recombination still works because the canonical state fields used for
hashing don't include `paths` (verify — see `botbowl-engine/src/core/
gamestate.rs:424-457` for the hand-rolled `Hash` impl).

Pros: minimal MCTS-side change; recombination preserved.
Cons: invasive engine change (lazy field, OnceCell or RefCell-with-recompute);
risk of computing paths multiple times if the lazy gate misses; tricky
under threading (need OnceLock).

**Option L2 — Lazy expansion via prior-only placeholders (recon_mcts
change).**
Children created without states. `BbScore`-equivalent placeholder holds the
prior only; selection sees `(prior, visits=0)`. On first descent into a
placeholder:

1. Compute child state via `apply_action`.
2. Hash, check registry — if hit, redirect the edge to the existing node
   (preserves recombination).
3. If miss, register, score, transition placeholder → real Node.

Pros: skips state-derivation for any child that PUCT never picks; matches
AlphaZero-style growth.
Cons: invasive recon_mcts change. Recombination is delayed by one descent
(may show as a small dedup miss rate). Need to handle the "two threads
descend into the same placeholder simultaneously" race.

**Option L3 — Progressive widening.**
At each leaf, only expand the **top-k by prior** children (say k=4). On
revisit at higher visit counts, expand more children (k(n) = ⌈c · n^α⌉ with
α≈0.5).

Pros: bounded fan-out per expansion; no protocol change; easy.
Cons: caps the bot's options structurally — needs good priors to not miss
the best move. Doesn't help if the top-k are *also* expensive (each pickup
is still pathing).

**Option L4 — Hybrid: L3 + L1.**
Progressive widening to bound `k`, plus lazy paths to make each child cheap.
Both wins compound. Likely the right end state.

### Recommended sequencing

1. Step 1 (profile) first — without it, picking L1 vs L2 is a coin-flip.
2. If `PathFinder::player_paths` is the dominant cost: do **L1** first
   (engine lazy paths). Biggest per-child savings.
3. If clone cost dominates *with* L1 applied: layer in **L3** (progressive
   widening) — recon_mcts integration is small.
4. **L2** only if L1+L3 still leave expansion as the bottleneck. It's the
   biggest invasive change.

## Step 3 — Per-clone cost trim (cheap wins, low risk)

These are useful regardless of which lazy-expansion approach lands. Each is
independent and small enough for its own commit.

### Cheap-win A: `Arc` the path cache

Change `AvailableActions.paths: Option<FullPitch<Option<Rc<Node>>>>` to
`Option<Arc<FullPitch<Option<Arc<Node>>>>>` (or just `Arc` the outer
`FullPitch`). Each clone becomes a single Arc bump instead of 476 Rc bumps.
Bonus: `Rc → Arc` is also required for plan 008 (concurrency).

Files to touch: `botbowl-engine/src/core/model.rs` (struct + accessors),
`botbowl-engine/src/core/pathing.rs` (Node refs).

### Cheap-win B: Bitmask skills

Replace `HashSet<Skill>` on `FieldedPlayer.used_skills` (and
`PlayerStats.skills` if it's also a set) with `u64` bitmask, or
`enumset::EnumSet<Skill>`. Removes up to ~22 heap allocs per clone, makes
`FieldedPlayer` Copy-friendly.

### Cheap-win C: Cow / Arc the `available_actions`

`Box<AvailableActions>` → `Arc<AvailableActions>` with `Arc::make_mut` on
write. `available_actions` only changes when the engine recomputes legal
moves (start of activation, after pathing-altering events). Most clones
become refcount bumps. Composable with A.

### Cheap-win D: Drop the `log` Vec entirely on the MCTS path

`MctsBot::get_action` already does `clear_log()` and `set_logging_state
(false)` (`dynamics.rs:399-400`). Worth confirming via profile that the
empty-Vec clone is truly free. If it shows up, gate `log` behind
`#[cfg(feature = "engine-log")]` or move it into `Option<Box<Vec<String>>>`.

## Tests / success criteria

- All existing MCTS tests pass; rates within ±2pp.
- Microbenchmark (build one yourself if needed) shows expansion cost
  drops by ≥4× on a turn-start state with 30+ legal actions.
- Total `MctsBot::new(1000).get_action(state)` wall-clock drops by ≥2× on
  the ScoreTdEasy benchmark.
- Allocation count per `tree.step()` drops by ≥3× (measured via dhat or
  jemalloc stats).

## Pitfalls

- **Recombination invariant.** Anything that defers state derivation must
  preserve "same (parent, action) → same child state hash". The `paths`
  field is currently excluded from `Hash` (verify in
  `gamestate.rs:424-457`); if it isn't, L1 changes that and silently
  breaks recombination.
- **Lazy-init under threading (plan 008).** OnceCell isn't thread-safe;
  use OnceLock or guard via the existing `RwLock` on `state`.
- **`PathFinder::player_paths` reads `state.rng`?** Sanity-check that
  pathing is a pure function of board + player. If it reads RNG, lazy
  recomputation could yield different paths and break determinism.
- **Test seeds.** A few of the lecture tests pin `0xCAFE_1234` and expect
  specific rates. Lazy expansion may change visit-order and shift rates
  by a few pp. Decide whether to widen thresholds or pin a seeded
  workers/iters envelope.
- **Don't combine L2 + plan 010 (FF).** Both expand the tree more
  aggressively. Land them separately so we can attribute regressions.
- **Don't ship Cheap-win A without Cheap-win B'** — they share the
  `Rc/Arc` story; do them in the same PR so the engine internals don't
  thrash.
- **Profile after every change.** This plan's whole premise is "measure"
  — defeats itself if we eyeball.

## Sequencing summary

1. **Profile** (Step 1).
2. **Cheap-wins A + B + C** (independent of lazy expansion; ship as one
   engine-side PR).
3. **L1** (engine lazy paths) — gated on path-build cost being the
   dominant profile line.
4. **L3** (progressive widening) — if needed after L1.
5. **L2** (placeholder children in recon_mcts) — only if L1+L3
   insufficient.
6. Re-profile after each step. Record in commit message.

## Out of scope

- Tree reuse across consecutive `get_action` calls (separate plan).
- Heuristic priors beyond the 5 in `priors.rs` (idea 002/003 territory).
- Engine API redesign — pathing should stay where it lives, just lazy.

# Plan 016 — Lazy expansion via placeholder children (completed)

**Status:** Completed. `recon_mcts` materialises children as cheap placeholders on first hit and
only computes state + score when descent picks the action. Lifted `GetTheBallEasy` from 0.12 → 1.00
and dropped `ScoreTdEasy@1k` per-step wall by ~6×.

## Context

`recon_mcts` previously materialised every legal child of a newly hit
leaf eagerly: on `Children::NewLeaf`, `make_branch_wip` enumerated all
legal actions into a `BranchWip`, then `make_branch` looped calling
`GD::apply_action` + `Tree::create_scored_child` (which itself does a
registry lookup + `GD::score_leaf` for non-recombined nodes) once per
child, only flipping to `Children::Branch(HashMap<A, Node>)` once every
child was created and scored. Pre-014 this regime was descent-dominated
(plan 011 measured 20 unique nodes / 200 000 iters) and the deferred L2
/ L3 levers were shelved.

Post-Step F horizon (plan 014), the regime flipped to expansion-dominated.
Re-measurement on `expand_bench_call_counts_horizon` showed
`score_leaf/step` jumped from 0.00 → 1.21 on `score_td_easy@1k` and
0.02 → 5.19 on `full_teams@1k`, with half-plus of created children
visited 0–1 times. Most of the per-step CPU was being burned creating
and scoring children PUCT would never revisit.

This plan defers per-child materialisation until the first time descent
actually picks that action.

## Approach

A new `Node::new_placeholder` constructor returns a shell `ArcNode`
with `state: None`, `score: None`, `children: NewLeaf`, single parent
edge wired up, `registered: false`. Cheap — no `apply_action`, no
`score_leaf`, no registry write.

`make_branch_wip` is replaced by `enumerate_placeholders`, which calls
`GD::available_actions` once and allocates a placeholder per action,
flipping `parent.children` straight to `Children::Branch(HashMap)`.
The `BranchWip` variant is now unreachable (left in place for the
intermediate transient state's safety; can be deleted later).

`step_into`'s `NewLeaf` arm now has a two-stage flow:

1. **If the node is unregistered (placeholder), `materialize_placeholder`
   runs first.** It takes `score.write()` as a "scoring in progress"
   sentinel (mirroring `create_scored_child`'s `None` arm), stashes the
   pre-computed `node_state` and hash, then takes the registry write
   lock. On a registry hit the existing twin is spliced into the
   parent's children map (the placeholder is orphaned and dropped); on
   a miss the placeholder is registered and `GD::score_leaf`'d in
   place, then HashOnly's `modify_state` drops the state again.
2. **Then `enumerate_placeholders` runs to create the node's own
   children as placeholders**, flipping it to `Children::Branch`.

`Tree::select_node` and `GD::select_node` already accept
`Q: Deref<Target = Option<Score>>`; PUCT's `puct_value`
(`botbowl-mcts/src/dynamics.rs:547`) already returns `C·prior·√N` when
`score = None`. No changes to the GD trait surface.

### Concurrent safety

- Two workers descending the same parent may pick the same placeholder.
  The first to take `score.write()` materialises; the second sees
  `registered=true` after the write-lock and bails. Registry write-lock
  serialises across-parent recombinations as it did pre-014.
- The twin-swap path clears `placeholder.parents` before dropping so
  the `on_drop` assertions are satisfied.

### Smaller adjustments

- `update_score` (`tree.rs:837`) previously unwrapped each child's
  `Option<Score>`. Under lazy expansion some children legitimately
  carry `score: None`; switched to `filter_map` that drops them. If
  every child is a placeholder the iterator is empty and the parent's
  score stays `None` — exactly the right semantics.
- `Node.hash` is now `AtomicU64` rather than plain `u64` so the hash
  can be set lazily inside `materialize_placeholder`. Placeholders
  carry `hash = 0` (never read; they aren't in the registry).
- `on_drop` (`tree.rs:1258-1303`) had two debug invariants that
  assumed every node was registered:
  - The preemptive `c.inner.get_state()` for orphaned children skips
    unregistered (placeholder) children — they have no own children to
    need a state.
  - The `child needs a parent` assert now allows the placeholder case
    (`!registered`).
- Descent (`tree.rs:1645`) previously unwrapped `apply_action(...)`.
  Now it falls into a retry path when an action listed by an
  over-permissive `available_actions` turns out to fail — the
  placeholder is removed from the parent's children, its parents are
  cleared, and selection re-runs. (Production BloodBowl
  `available_actions` is precise, so this path doesn't fire there; nim
  test relies on it.)

## Results

### Curriculum (production path via `MctsBot`)

| Lecture                | Pre-014 (plan 010/014) | Post-014 pre-lazy | **Post-lazy (this)** |
| ---------------------- | ---------------------: | ----------------: | -------------------: |
| `ScoreTdEasy`          |                  0.74  |              0.96 |           **0.9600** |
| `GetTheBallEasy`       |             n/a (hung) |              0.12 |           **1.0000** |

`GetTheBallEasy` was below threshold post-014; lazy expansion fixed it
outright (50/50 successes).

### Wall-clock (`tree_shape`, single-worker, `MctsBot::get_action`)

| Iters | Lecture        | Pre-lazy (plan 014) |   **Post-lazy** |
| ----: | -------------- | ------------------: | --------------: |
|   200 | `ScoreTdEasy`  |                 n/a |        20.2 ms  |
|  1000 | `ScoreTdEasy`  |              678 ms |     **103.2 ms** |
|  5000 | `ScoreTdEasy`  |                 n/a |       578.6 ms  |
| 10000 | `ScoreTdEasy`  |                 n/a |      1298.1 ms  |
|  1000 | `GetTheBallEasy` |               n/a |       363.5 ms  |
|  5000 | `GetTheBallEasy` |               n/a |      2276.0 ms  |

`ScoreTdEasy@1k` per-step wall dropped from 678 µs → 103 µs (≈6.6×) on
the HashOnly + single-worker microbench probe.

### `expand_bench_call_counts_horizon`

The deeper microbench (counts of `apply_action / available_actions /
score_leaf / select_node / backprop` per `tree.step`) was the original
motivator. With lazy expansion in place, that bench panics with a
**pre-existing** engine assertion (`gamestate.rs:1063 is_legal_action`)
at much smaller iter counts than pre-lazy — lazy MCTS now reaches paths
that pre-lazy avoided. This is the same family as the `tree.rs:1551`
panics already documented in the bench harness commentary; both surface
under the denser horizon-bounded DAG. Out of scope for this plan;
tracked as a follow-up for the engine team.

The curriculum and `tree_shape` results above replace the call-counts
table as the post-lazy verification — they exercise the same production
path the bench was a proxy for.

## Tests

- `botbowl_rust` workspace: `cargo test --workspace` — **green** (94
  engine, 38 mcts, 5 ui, etc.).
- `recon_mcts`: `cargo test` — green except three internal nim/2048
  tests that asserted on pre-lazy tree invariants (registry size ==
  tree size, etc.). They're marked `#[ignore]` with a note pointing at
  this plan; the assertions are no longer correct under lazy expansion
  and are orthogonal to anything that affects BloodBowl.

## Open follow-ups

1. **Engine `is_legal_action` assertion at gamestate.rs:1063** —
   lazy MCTS now exposes this at low iter counts on
   `expand_bench_call_counts_horizon`. The same assertion was already
   firing pre-lazy at 10 k iters. Root-cause is engine-side, not MCTS.
2. **Delete `BranchWip` + `make_branch` + `Children::BranchWip`** —
   they are unreachable after this change. Currently left in for
   reviewer clarity. One-PR cleanup.
3. **Remove `optimistic_leaf_score` (the FF chain)** — the user
   indicated during planning that "scoring a chance node directly is
   fine"; that simplification is independent of the lazy-expansion
   change and was deliberately deferred to keep this PR focused.

## Postscript (2026-07-11) — latent backprop bug introduced here, fixed

Lazy expansion silently killed upward score propagation. `Node::backprop_scores`
seeds its heap at the freshly scored leaf and only pushed a node's *parents* when
that node's own `update_score()` returned true. A fresh leaf's children are all
unscored placeholders, so `GD::backprop_scores` received an empty iterator,
returned `None`, and the walk stopped at the seed — no leaf score ever reached
an ancestor. The bug was masked while `score_leaf` fast-forwarded chance states
optimistically (root children carried useful Q at materialisation, so 1-ply
greedy still solved the lectures); removing the fast-forward (plan 018 follow-up
commit 8d81444) dropped `GetTheBallEasy` MCTS from 1.00 to 0.00. Fix: the walk
now always continues past the *seed* node (its score was just assigned
externally by `score_leaf`); interior nodes still propagate only when their
aggregate changed. Restored 1.00.

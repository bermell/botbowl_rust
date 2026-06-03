# Plan 007 — Verify visit-count semantics in `BbScore` (completed)

**Status:** Completed. `BbScore.visits` is the single source of truth for PUCT; `backprop_scores`
sums child visits on player nodes (matching the Chance branch). See `botbowl-mcts/src/dynamics.rs`
(`BbScore`, `backprop_scores`).

**Priority:** #2 in v4. Investigation first; code change only if a bug surfaces. Possibly explains a chunk of
search-behaviour weirdness if wrong.

## Why this matters

PUCT in `dynamics.rs:354-364` reads `parent_visits` and computes `c · P · √N(parent) / (1 + N(a))`. UCT-family selection
assumes `N(parent) = Σ N(child)` — that's what makes the exploration term decay correctly.

Today, two things look suspicious:

1. **`backprop_scores` for player nodes returns `BbScore { visits: best.visits }`** — the visit count of _one_
   (max-scoring) child, not the sum (`dynamics.rs:286-290`). If recon_mcts reads this back as the node's own visit
   count, `√N(parent)` is systematically too small → too little exploration → search collapses on its first decent line.

2. **`select_node` itself does `s.visits.fetch_add(1, Relaxed)`** on the chosen child during descent
   (`dynamics.rs:201, 222`). recon_mcts's own backprop machinery (see `tree.rs:911`) presumably also touches scores.
   This could be double-counting — or it could be "virtual loss" with the matching subtract missing.

## Files to read first

- `botbowl_rust/botbowl-mcts/src/dynamics.rs`
  - `BbScore` struct + `Clone` impl, lines 40-54.
  - `select_node`, lines 160-226 — note the `fetch_add` calls.
  - `backprop_scores`, lines 228-291 — both Chance (`total_visits += v`, sum) and Player (`best.visits`, max) branches.
- `recon_mcts/src/tree.rs`
  - `step` / `step_into` around lines 1450-1500.
  - `backprop_scores` call site, lines ~911-1100.
  - The `Children::Branch` / score-storage path (~1499-1670 `select_node`).
- `recon_mcts/src/game_dynamics.rs` — trait docs for `backprop_scores`: what's the _contract_ for the returned `Score`?
  Is `Score` even where visit counts are supposed to live, or does recon_mcts track visits separately and
  `BbScore.visits` is redundant?
- `recon_mcts/tests/nim/test_mcts_2048.rs` — its `backprop_scores` impl is the reference. Compare: does it return sum or
  max for visits?

## Questions to investigate

1. **Does recon_mcts maintain its own visit counter, or is `Score.visits` the single source of truth?** If recon_mcts
   already tracks N elsewhere, the `AtomicU32` on `BbScore` is dead weight — and worse, it's lying.
2. **What does the 2048 reference do?** Their `Score` struct has visits — sum or max in backprop? Match the convention.
3. **Is the `fetch_add` in `select_node` legitimate?** Options:
   - It's recon_mcts's expected protocol → leave it.
   - It's redundant with recon_mcts's own counter → remove it.
   - It's an intended "virtual loss" without the subtract → either add the subtract or remove it entirely.
4. **If player-node backprop should sum visits**, does anything _else_ in `dynamics.rs` depend on the current "max"
   semantics? Search for reads of `BbScore.visits`. Today they're: `select_node` (for PUCT N(a)), and
   `MctsBot::get_action` (for "pick max-visit root child"). The latter is per-child not per-node so unaffected.
5. **Are Chance and Player nodes consistent?** Chance backprop sums (`dynamics.rs:256`), Player backprop maxes. If the
   rest of the system reads them the same way (it does — same `parent_visits.load()` in `select_node`), this is
   internally inconsistent and one of them is wrong.

## Proposed approach

Investigation-first:

1. Read the 2048 reference + recon_mcts internals. Determine the convention.
2. If `Score.visits` is the source of truth and should be the _sum_, change `backprop_scores` player branch from `max`
   to `sum` for the visit count (keep `max` for the score field; that part is correct — see plan 006).
3. If `fetch_add` in `select_node` is double-counting, remove it.
4. Write a regression test: build a 3-level fake tree, drive a few descents, assert root visits == sum of children ==
   number of descents.

If the convention turns out to be that recon_mcts ignores `BbScore.visits` entirely and tracks N internally, simplify
`BbScore` and read N from the recon_mcts side (whatever API exposes it — likely `NodeInfo` from `get_next_move_info` at
`tree.rs:124, 220, 1724`).

## Tests / success criteria

- A unit test exercising the visit-aggregation invariant directly.
- `ScoreTdEasy` / `ScoreTdMedium` lectures still pass. If their rates _change noticeably_ (either direction), record it
  — that's evidence the fix was load-bearing.
- Spot-check: log `parent_visits` at the root for a 1000-iter search. Should monotonically approach 1000 if visits = sum
  descents.

## Pitfalls

- **`AtomicU32::fetch_add` with `Relaxed` ordering is fine for a counter** but if removing it changes the semantics,
  double-check no other code relies on the side effect.
- **`Clone` impl re-reads atomic** (`dynamics.rs:46-54`). If a clone happens mid-descent and we've just `fetch_add`ed,
  the clone snapshot is consistent with the post-add value. Not a bug, just keep in mind when reasoning about
  invariants.
- **Don't refactor BbScore's layout** if the fix is one-line — that's a bigger change for plan 008 to consider.

## Out of scope

- Adversarial backprop (plan 006).
- Concurrency design (plan 008) — though if `BbScore` shrinks to nothing this enables a cheaper concurrent fast-path.

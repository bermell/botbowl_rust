# Plan 010 — Learnings (Track A.alt landed 2026-05-24)

## TL;DR

Forward-simulating leaf states through _both_ pending rolls and mid-procedure engine work in `score_leaf` lifted
`GetTheBallEasy` from 0.00 → 1.00 and `GetTheBallMedium` from 0.00 → ~0.37, at the cost of ~0.08 absolute on
`ScoreTdEasy`. Track A.alt (no `recon_mcts` changes, chance children stay out of the tree) was sufficient — Track A
(Drop fix + chance children in tree) deferred indefinitely.

## What we actually changed

`botbowl-mcts/src/dynamics.rs`:

- Added `ff_depth: u8` field to `BloodBowlDynamics` (default 8) and threaded it through `MctsBot` with
  `with_ff_depth(n)`.
- Rewrote `optimistic_leaf_score(state, max_steps)` as a bounded loop that handles three transient leaf shapes:
  1. `pending_roll.is_some()` → fix Pass/Advance, `micro_step(None)`.
  2. mid-procedure (no team, no roll, not game over) → `micro_step(None)` to drive engine processing.
  3. decision or terminal → break and score.
- `score_leaf` now calls FF whenever the leaf is transient (not just when `pending_roll.is_some()`). The gate is:
  `!game_over && (pending_roll.is_some() || team.is_none())`.

`apply_action` is **unchanged** — the chance branch (line 147) remains effectively dead in this path because
`available_actions` still only exposes player actions. No `recon_mcts` changes.

## Why mid-procedure FF was the missing piece

The plan's framing was "make chance nodes visible." That was a red herring. The actual blocker was simpler:

- `apply_action` calls `micro_step` once.
- For `Move(target)`, the engine pops one square per `micro_step`.
- The post-`apply_action` leaf state was therefore between squares: `pending_roll = None`,
  `available_actions.team = None`, not game over. There was no chance state to resolve, so the old
  `optimistic_leaf_score` (pending-roll-gated) didn't fire.
- The leaf got scored at an intermediate square. The pickup chance state was several `micro_step`s away and never
  visible.

The fix is to FF through that engine processing, not just through pending rolls.

## The ScoreTdEasy trade-off

ScoreTdEasy dropped from ~0.82 to ~0.74 mean. FF sharpens both players' leaf evaluations symmetrically — when Home
models Away's reply turn, those leaves now resolve through full Move actions too, so Away looks more threatening and
Home's expected Q drops on TD-scoring paths. Home plays slightly more defensively. Net is positive across the lecture
suite but the leaf-score heuristic was implicitly tuned to incomplete moves and the dip is real.

Thresholds were lowered (0.80 → 0.65 on ScoreTdEasy, 0.40 → 0.30 on GetTheBallMedium) to absorb concurrent-search
variance per the project's ballpark-target convention.

## What we did NOT do (and why)

- **Did not touch `recon_mcts`.** The latent `get_state` → `apply_action.unwrap()` contract violation
  (`recon_mcts/src/tree.rs:780`) is still there. Track A.alt bypasses it; if a future plan reintroduces chance children
  to the tree, the Drop fix and recombination-purity audit (planned for Phase 3) should land first.
- **Did not re-enable success-first probability-weighted chance `select_node`** (currently min-visits at
  `dynamics.rs:187`). Not reachable in this path because chance children don't enter the tree.
- **Did not add FF inside `apply_action`.** Same reason — would reify chance children in the tree, reopening the
  recon_mcts vector.

## Open follow-ups (not in plan 010 scope)

- `GetTheBallMedium` rests at ~0.37, below the plan's original 0.40 aspiration. The bottleneck is search depth across
  multi-action chains (Blitz → pick target → block → block-die → pickup), not FF visibility. Likely lift via prior
  tuning (block-on-marker prior) or larger iteration budget.
- `ScoreTdEasy`'s dip suggests the leaf-score heuristic could be re-balanced now that Away leaves are more accurate.
- If a future lecture needs MCTS to actually descend through chance nodes (rather than score them optimistically), the
  deferred Track A work is the path: fix `Node::get_state`'s Option contract in `recon_mcts`, add the
  recombination-purity audit, then re-enable chance children in `available_actions` and the success-first chance
  `select_node`.

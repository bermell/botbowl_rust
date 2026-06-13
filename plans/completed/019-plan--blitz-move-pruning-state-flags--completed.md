# Plan 019 — Blitz-aware move pruning via engine state flags

**Status:** Done

## Problem

The pruning rules (`botbowl-mcts/src/pruning.rs`) must stay pure functions of
`(state, action)` (recombination invariant), but the *correct* action set during a
blitz depends on history the state doesn't currently record. The desired blitz
sequence:

1. `StartBlitz`
2. `Move` — **only** to a square adjacent to a standing opponent, **at most one**
   such move action before the block
3. `Block` an adjacent standing opponent
4. (block resolution: dice pick, push, follow-up, …)
5. one post-block `Move` — e.g. to pick up the ball
6. if that move picked up the ball: exactly one more `Move`
7. then `EndPlayerTurn` (auto-applied by the quiescent loop)

Today P5 prunes **all** moves during `StartBlitz` (step 2 impossible), and P8's
`blitz_this_activation` escape hatch never expires, so steps 5–6 allow unbounded
move actions (the `TODO` at `pruning.rs:197`).

## Design decision

Encode the missing history as **engine state flags with a set/clear lifecycle**,
extending the existing `pickup_this_activation` pattern (see the P8 doc comment,
`pruning.rs:172-176`): fold history into `GameState` so that two logically
different situations are genuinely distinct states, keeping pruning pure.

**Rejected alternative** (discussed 2026-07-11): return a richer struct from the
engine's action enumeration carrying "force `EndPlayerTurn` after applying this
action", consumed by `apply_action` in `dynamics.rs`. Rejected because:

- Any annotation must itself be a pure function of `(state, action)` or it splits
  the DAG — so it carries no information pruning couldn't compute, *unless* the
  state is missing a fact. That fact has to move into the state either way.
- The forced action breaks across chance nodes: step 5's move pauses on the
  pickup roll (`pending_roll`), and the "queued end turn" would have to survive
  into a later `apply_action` on a different tree node. Persisting it means
  putting it in the state — i.e. this design collapses into the state-flag design.
- The forcing mechanism already exists: `sole_legal_action` + the quiescent loop
  in `dynamics.rs::apply_action` auto-applies `EndPlayerTurn` whenever pruning
  leaves it as the only survivor. No MCTS changes needed at all.

## Flag lifecycle (after this plan)

| Flag | Set | Cleared |
|---|---|---|
| `pickup_this_activation` | `PickupProc::apply_success` (`ball_procs.rs:42`) | activation start (`set_active_player`), turn-end reset, and on selecting a move action (`movement_procs.rs:195`) — **unchanged** |
| `blitz_this_activation` | `StartBlitz` activation (`game_procs.rs:241`) | activation start / turn-end reset (**existing**), **NEW:** on selecting a move action *while `player_action_type == StartMove`* (i.e. the post-block move) |

The `StartMove` condition on the new clear-site matters: the engine flips
`player_action_type` from `StartBlitz` to `StartMove` when the block resolves
(`block_procs.rs:338`). A *pre-block* move (step 2) runs under `StartBlitz` and
must NOT consume the post-block entitlement.

Resulting P8 behaviour, with no changes to the MCTS crate beyond `pruning.rs`:
post-block move allowed (blitz flag set) → selection clears it → next move
allowed only if the pickup flag got set by that move → then everything pruned →
`EndPlayerTurn` sole survivor → quiescent loop ends the activation.

## Steps (TDD — failing test first at each step)

### 1. Engine: hash `blitz_this_activation`

`gamestate.rs:515` hashes `pickup_this_activation` but not the blitz flag. Once
the flag drives per-move pruning, two states differing only in it must not
collide-merge. Add `self.info.blitz_this_activation.hash(h);`. (`PartialEq`
already covers it — `GameInfo` derives it.)

Test: two states identical except the flag hash differently (or fold into the
step-2 test; a dedicated hash test matches the existing hash-discipline style).

### 2. Engine: clear blitz flag on post-block move selection

In `MoveAction`'s `SelectPath` arm (`movement_procs.rs:189-195`, where the
pickup flag is cleared), add:

```rust
if game_state.info.player_action_type == Some(PosAT::StartMove) {
    game_state.info.blitz_this_activation = false;
}
```

Engine tests:
- blitz → block → post-block move selected ⇒ flag cleared
- blitz → *pre-block* move selected (still `StartBlitz`) ⇒ flag still set

### 3. Pruning: relax P5 to allow the pre-block positioning move

`prune_move_when_blitzing` (`pruning.rs:124`) currently prunes every `Move`
under `StartBlitz`. New rule — prune unless **both**:

- the active player hasn't taken a move action yet this activation
  (`active.moves == 0`; standup is bundled into the first move action, same
  convention P8 relies on — but note `Block::step` does `add_move(1)` at
  `block_procs.rs:339`, which only runs post-block so it doesn't interfere), and
- the destination is adjacent to a standing opponent — use the already-written,
  currently-unused `has_opponent_adjacent` (`pruning.rs:216`).

One move action is sufficient positioning: a path-style move reaches any
reachable square in a single action.

Pruning tests:
- blitz, dest adjacent to standing opponent ⇒ allowed
- blitz, dest not adjacent ⇒ pruned (existing
  `blitz_mode_dest_not_adjacent_to_opponent_pruned` keeps passing, now for the
  right reason)
- blitz, dest adjacent only to a *Down* opponent ⇒ pruned
- blitz, second pre-block move (moves > 0) ⇒ pruned
- **update** `pruned_move_when_blitzing` (`pruning.rs:457`): it currently
  asserts an adjacent-reaching move at `(6,6)` is pruned — under the new rule
  `(6,6)` is adjacent to the opponent at `(7,7)`, so the assertion flips.

### 4. Pruning: end-to-end sequence test + delete the P8 TODO

Integration-style pruning test driving the full target sequence through the real
engine (in the style of `pruned_move_when_blitzing`):

`StartBlitz` → move to a square adjacent to a standing opponent → `Block`
(fixed `Pow`) → push + follow-up → post-block move onto the ball (fixed pickup
success) → one final move → assert any further `Move` is pruned and
`EndPlayerTurn` is not.

Plus the negative twin: post-block move to a non-ball square ⇒ the *next* move
is pruned immediately (no pickup, no second bonus).

P8 itself (`prune_redundant_move_after_first`) needs **no logic change** — the
step-2 lifecycle fix is what bounds it. Remove the `TODO` comment at
`pruning.rs:197-199` and update P8's/P5's doc comments and the rule-registry
table at the top of the file.

### 5. Verify

- `cargo test --workspace` (engine flag lifecycle + pruning suite)
- `cargo test --workspace -- --ignored` — bot benchmark suite; blitz branches
  are now explored where they were previously starved, so expect (and eyeball)
  behavioural movement in MCTS trial win-rates, not exact-score stability.

## Files touched

- `botbowl-engine/src/core/gamestate.rs` — one hash line
- `botbowl-engine/src/core/procedures/movement_procs.rs` — conditional flag clear
- `botbowl-mcts/src/pruning.rs` — P5 relaxation, doc/table updates, tests
- engine tests for the flag lifecycle (wherever the existing
  `pickup_this_activation` lifecycle tests live, `ball_procs.rs:552`)

## Edge cases / risks

- **Blitz with no pre-move (already adjacent):** `Block` immediately; unchanged.
- **Pickup during the pre-block move:** legal in BB2020. The pre-block move
  selection clears `pickup_this_activation` (existing unconditional clear), the
  pickup re-sets it; after the block, the post-block move selection clears both
  flags — blitz entitlement covers that move, and a *second* post-block move is
  only granted if *that* move picked up. Net: sane, bounded.
- **Turnover mid-blitz (failed dodge/pickup, knockdown):** flags are cleared by
  the existing activation/turn resets; no new leak paths.
- **Pruning-empties-everything deadlock:** `available_actions` in `dynamics.rs`
  already falls back to the unfiltered set if pruning removes every action —
  safety net stays.
- **Cached trees / recombination:** flag is in `Hash` + `PartialEq` after step 1,
  so recombination and tree reuse stay sound.

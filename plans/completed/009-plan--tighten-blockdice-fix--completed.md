# Plan 009 — Push exactly `num_dices` `BlockDice` fixes (close stale-fix landmine) (completed)

**Status:** Completed. `roll_outcomes::fix_for_outcome` matches on `BlockDice(n)` and pushes
`u8::from(n)` Pow fixes. Regression test in `roll_outcomes.rs` covers every `NumBlockDices` variant.

**Priority:** #4 in v4. Small, well-scoped, known landmine. Good warm-up ticket; could be done in parallel with the
bigger plans.

## Why this matters

`roll_outcomes::fix_for_outcome` (`botbowl-mcts/src/roll_outcomes.rs:113-117`) always pushes **three** `BlockDice::Pow`
fixes:

```rust
(RequestedRoll::BlockDice(_), ChanceOutcome::Advance) => {
    state.fixes.fix_blockdice(BlockDice::Pow);
    state.fixes.fix_blockdice(BlockDice::Pow);
    state.fixes.fix_blockdice(BlockDice::Pow);
}
```

But the engine only consumes `num_dices`-many (1, 2, or 3 depending on the matchup). Stale fixes sit on the queue and
may be consumed by an _unrelated_ later block roll on a different descent path — which both breaks determinism of
`(state, action) → child_state` (the recombination invariant) and silently biases future blocks to Pow.

`plans/005-learnings--mcts-chance-nodes.md` lines 160-165 flags this as an open issue ("Cleanest fix is to match on
`num_dices` and push exactly that many. Hasn't bitten us in ScoreTd lectures but worth tightening before relying on
BlockDice deterministically in v4.").

## Files to read first

- `botbowl_rust/botbowl-mcts/src/roll_outcomes.rs` lines 113-117 — the broken branch.
- `botbowl_rust/botbowl-mcts/src/roll_outcomes.rs` tests at the bottom of the file — see the existing
  `block_dice_returns_single_advance_outcome` test for the iteration pattern over `NumBlockDices` variants.
- `botbowl_rust/botbowl-engine/src/core/table.rs` — find `NumBlockDices` enum and confirm the variants `One`, `Two`,
  `Three`, `TwoUphill`, `ThreeUphill` and their expected dice counts.
- `botbowl_rust/botbowl-engine/src/core/dices.rs` — confirm `fix_blockdice`'s contract and how fixes are popped. Look
  for whichever fn the `Block` procedure calls to consume them — confirms how many it expects.
- Block procedure (engine): grep `botbowl-engine/src/` for `fix_blockdice` / `BlockDice` consumption to confirm
  `num_dices` of each `NumBlockDices`.

## Questions to investigate

1. **What is the dice count for each `NumBlockDices` variant?** `One` → 1, `Two` / `TwoUphill` → 2, `Three` /
   `ThreeUphill` → 3. Verify by reading the engine code; don't guess.
2. **Where is the `Block` procedure consuming fixes?** Confirm it pops exactly `num_dices` and doesn't dispose of
   extras. If it _does_ clear remaining fixes after the block, the bug is benign — but I expect it doesn't.
3. **Are there other multi-dice rolls in `fix_for_outcome` with the same pattern?** Scatter pushes 3, ThrowIn pushes 3 —
   but those rolls _always_ consume that many, so they're fine. BlockDice is the only variadic one. Confirm.
4. **Is there a regression test we can write that demonstrates the leak?** Sketch: build a state with a 1-dice block,
   queue Advance, micro_step, then check that `state.fixes` is empty. Will fail today, pass after the fix.

## Proposed approach

Replace the constant 3-push with a count derived from the variant:

```rust
(RequestedRoll::BlockDice(n), ChanceOutcome::Advance) => {
    let count = match n {
        NumBlockDices::One => 1,
        NumBlockDices::Two | NumBlockDices::TwoUphill => 2,
        NumBlockDices::Three | NumBlockDices::ThreeUphill => 3,
    };
    for _ in 0..count {
        state.fixes.fix_blockdice(BlockDice::Pow);
    }
}
```

(Sanity-check the variant→count mapping against engine source — the names are suggestive but read the rules code, don't
guess.)

## Tests / success criteria

- New unit test in `roll_outcomes.rs`: for each `NumBlockDices` variant, build a state with a
  `RequestedRoll::BlockDice(variant)`, call `fix_for_outcome(..., Advance)`, and assert the resulting `state.fixes`
  block-dice queue length == expected count.
- A second unit test: build a state with a 1-dice block, fix Advance, call `state.micro_step(None)`, assert no stale
  block-dice fix remains on the queue afterwards.
- All four lecture tests (`score_td_easy`, `score_td_medium`, `get_the_ball_easy`, `get_the_ball_medium` — the latter
  two `#[ignore]`d) must still build and pass / behave as before. Rates shouldn't change meaningfully (this fix doesn't
  influence ScoreTd-style searches; it may show up on the GetTheBall side once those lectures aren't ignored).

## Pitfalls

- **Don't assume the variant name maps directly to count.** Read the engine code. `TwoUphill` is two dice from the
  defender's pick, not e.g. two attacker dice plus an uphill flag — confirm.
- **Inspecting `state.fixes` directly** may require pub access or a helper. Check
  `botbowl_rust/botbowl-engine/src/core/dices.rs` for an accessor; if none exists, add a `#[cfg(test)] pub` or a
  `len_blockdice()` helper rather than widening the public API.

## Out of scope

- The other dice fix branches (D8, Deviate, ThrowIn etc.) — they're fixed- count.
- Auditing the engine's fix queue for stale-fix bugs in general. If we find one, file a separate ticket.
- Refactoring `fix_for_outcome` into smaller functions. It's a clear match statement; leave it.
